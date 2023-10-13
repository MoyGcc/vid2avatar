import hydra
import kaolin
import numpy as np
import torch
import torch.nn as nn
from kaolin.ops.mesh import index_vertices_by_faces
from torch.autograd import grad

from ..utils import utils
from .deformer import SMPLDeformer
from .density import AbsDensity, LaplaceDensity
from .networks import ImplicitNet, RenderingNet
from .ray_sampler import ErrorBoundSampler
from .sampler import PointInSpace
from .smpl import SMPLServer


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import hydra
import kaolin
from kaolin.ops.mesh import index_vertices_by_faces
import einops


class V2A(nn.Module):
    def __init__(self, opt, betas_path, gender, num_training_frames):
        super().__init__()

        # Foreground networks
        self.implicit_network = ImplicitNet(opt.implicit_network)
        self.rendering_network = RenderingNet(opt.rendering_network)

        # Background networks
        self.bg_implicit_network = ImplicitNet(opt.bg_implicit_network)
        self.bg_rendering_network = RenderingNet(opt.bg_rendering_network)

        # Frame latent encoder
        self.frame_latent_encoder = nn.Embedding(
            num_training_frames, opt.bg_rendering_network.dim_frame_encoding
        )
        self.sampler = PointInSpace()

        betas = np.load(betas_path)
        self.use_smpl_deformer = opt.use_smpl_deformer
        self.gender = gender

        # pre-defined bounding sphere
        self.sdf_bounding_sphere = 3.0

        # threshold for the out-surface points
        self.threshold = 0.05

        self.density = LaplaceDensity(**opt.density)
        self.bg_density = AbsDensity()

        self.ray_sampler = ErrorBoundSampler(
            self.sdf_bounding_sphere, inverse_sphere_bg=True, **opt.ray_sampler
        )
        self.smpl_server = SMPLServer(gender=self.gender, betas=betas)

        if self.use_smpl_deformer:
            self.deformer = SMPLDeformer(smpl=self.smpl_server, betas=betas)

        if opt.smpl_init:
            smpl_model_state = torch.load(
                hydra.utils.to_absolute_path("./assets/smpl_init.pth")
            )
            self.implicit_network.load_state_dict(
                smpl_model_state["model_state_dict"])

        self.mesh_v_cano = self.smpl_server.verts_c
        self.mesh_f_cano = torch.tensor(
            self.smpl_server.smpl.faces.astype(np.int64), device=self.mesh_v_cano.device
        )
        self.mesh_face_vertices = index_vertices_by_faces(
            self.mesh_v_cano, self.mesh_f_cano
        )

    def sdf_func_with_smpl_deformer(self, x, cond, smpl_tfs, smpl_verts, smpl_weights):
        x_flat = einops.rearrange(x, "b n s p -> b (n s) p")
        if hasattr(self, "deformer"):
            x_c, outlier_mask = self.deformer.forward(
                x_flat,
                smpl_tfs,
                return_weights=False,
                inverse=True,
                smpl_verts=smpl_verts,
                smpl_weights=smpl_weights,
            )

            output = self.implicit_network(x_c, cond)
            sdf = output[:, :, 0:1]
            feature = output[:, :, 1:]
            if not self.training:
                # set a large SDF value for outlier points
                sdf[outlier_mask] = 4.0

        return sdf, x_c, feature

    def check_off_in_surface_points_cano_mesh(self, x_cano, N_samples, threshold=0.05):
        batch_size = x_cano.size(0)
        distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            x_cano.contiguous(), self.mesh_face_vertices.repeat(batch_size, 1, 1, 1)
        )

        distance = torch.sqrt(distance)  # kaolin outputs squared distance
        sign = kaolin.ops.mesh.check_sign(
            self.mesh_v_cano.repeat(batch_size, 1, 1), self.mesh_f_cano, x_cano
        ).float()
        sign = 1 - 2 * sign
        signed_distance = sign * distance
        num_pixels = x_cano.shape[1] // N_samples
        signed_distance = signed_distance.view(
            batch_size, num_pixels, N_samples, 1)

        minimum = torch.min(signed_distance, 2)[0]
        index_off_surface = (minimum > threshold).squeeze(-1)
        index_in_surface = (minimum <= 0.0).squeeze(-1)
        return index_off_surface, index_in_surface

    def forward(self, input):
        # Parse model input
        torch.set_grad_enabled(True)
        # intrinsics = input["intrinsics"]
        # pose = input["pose"]
        uv = input["uv"]
        camera_pos = input["camera_poses"]
        camera_rotate = input["camera_rotates"]

        scale = input["scale"][:, None].float()

        smpl_pose = input["smpl_pose"]
        smpl_shape = input["smpl_shape"]
        smpl_trans = input["smpl_trans"]
        smpl_output = self.smpl_server(
            scale, smpl_trans, smpl_pose, smpl_shape)

        smpl_tfs = smpl_output["smpl_tfs"]

        cond = {"smpl": smpl_pose[:, 3:] / np.pi}
        # if self.training:
        #     if input["current_epoch"] < 20 or input["current_epoch"] % 20 == 0:
        #         cond = {"smpl": smpl_pose[:, 3:] * 0.0}

        ray_dirs, cam_loc = utils.get_camera_params_equirect(
            uv, camera_pos=camera_pos, camera_rotate=camera_rotate
        )
        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1)

        z_vals, _ = self.ray_sampler.get_z_vals(
            ray_dirs, cam_loc, self, cond, smpl_output
        )

        z_vals, z_vals_bg = z_vals
        z_max = z_vals[:, :, -1]
        z_vals = z_vals[:, :, :-1]
        N_samples = z_vals.shape[2]

        points = cam_loc.unsqueeze(-2) + \
            z_vals.unsqueeze(-1) * ray_dirs.unsqueeze(-2)

        dirs = ray_dirs.unsqueeze(2).repeat(1, 1, N_samples, 1)
        (
            sdf_output,
            canonical_points,
            feature_vectors,
        ) = self.sdf_func_with_smpl_deformer(
            points,
            cond,
            smpl_tfs,
            smpl_output["smpl_verts"],
            smpl_output["smpl_weights"],
        )

        sdf_output = sdf_output.squeeze(-1).view(batch_size,
                                                 num_pixels, N_samples)

        if self.training:
            (
                index_off_surface,
                index_in_surface,
            ) = self.check_off_in_surface_points_cano_mesh(
                canonical_points, N_samples, threshold=self.threshold
            )
            canonical_points = canonical_points.view(
                batch_size, num_pixels, N_samples, 3
            )

            # sample canonical SMPL surface pnts for the eikonal loss
            smpl_verts_c = self.smpl_server.verts_c.repeat(batch_size, 1, 1)

            indices = torch.randperm(smpl_verts_c.shape[1])[:num_pixels].cuda()
            verts_c = torch.index_select(smpl_verts_c, 1, indices)
            sample = self.sampler.get_points(verts_c, global_ratio=0.0)

            sample.requires_grad_()
            # sample = utils.frequency_encoding(sample)
            local_pred = self.implicit_network(sample, cond)[..., 0:1]
            grad_theta = gradient(sample, local_pred)

            differentiable_points = canonical_points

        else:
            differentiable_points = canonical_points.view(
                batch_size, num_pixels, N_samples, 3
            )
            grad_theta = None

        # differentiable_points = differentiable_points.view(-1, 3)
        # sdf_output = sdf_output.view(-1, 1)
        # view = -dirs.view(-1, 3)
        # points_flat = points.view(-1, 3)

        view = -dirs

        if differentiable_points.shape[0] > 0:
            fg_rgb_flat, others = self.get_rbg_value(
                points,
                differentiable_points,
                view,
                cond,
                smpl_tfs,
                feature_vectors=feature_vectors,
                is_training=self.training,
            )
            normal_values = others["normals"]

        if "image_id" in input.keys():
            frame_latent_code = self.frame_latent_encoder(input["image_id"])
        else:
            frame_latent_code = self.frame_latent_encoder(input["idx"])

        fg_rgb = fg_rgb_flat.view(batch_size, num_pixels, N_samples, 3)
        normal_values = normal_values.view(
            batch_size, num_pixels, N_samples, 3)
        weights, bg_transmittance = self.volume_rendering(
            z_vals, z_max, sdf_output)

        fg_rgb_values = torch.sum(weights.unsqueeze(-1) * fg_rgb, 2)

        # Background rendering
        if input["idx"] is not None:
            N_bg_samples = z_vals_bg.shape[2]
            z_vals_bg = torch.flip(
                z_vals_bg,
                dims=[
                    -1,
                ],
            )  # 1--->0

            bg_dirs = ray_dirs.unsqueeze(2).repeat(1, 1, N_bg_samples, 1)
            bg_locs = cam_loc.unsqueeze(2).repeat(1, 1, N_bg_samples, 1)

            bg_points = self.depth2pts_outside(
                bg_locs, bg_dirs, z_vals_bg
            )  # [..., N_samples, 4]

            bg_points_flat = einops.rearrange(
                bg_points, "b n s p -> b (n s) p")
            bg_dirs_flat = einops.rearrange(bg_dirs, "b n s p -> b (n s) p")
            bg_output = self.bg_implicit_network(
                bg_points_flat, {"frame": frame_latent_code}
            )
            bg_sdf = bg_output[..., :1]
            bg_sdf = bg_sdf.view(batch_size, num_pixels, N_bg_samples)

            bg_feature_vectors = bg_output[..., 1:]

            bg_rendering_output = self.bg_rendering_network(
                None, None, bg_dirs_flat, None, bg_feature_vectors, frame_latent_code
            )
            if bg_rendering_output.shape[-1] == 4:
                bg_rgb_flat = bg_rendering_output[..., :-1]
                shadow_r = bg_rendering_output[..., -1]
                bg_rgb = bg_rgb_flat.view(
                    batch_size, num_pixels, N_bg_samples, 3)
                shadow_r = shadow_r.view(
                    batch_size, num_pixels, N_bg_samples, 1)
                bg_rgb = (1 - shadow_r) * bg_rgb
            else:
                bg_rgb_flat = bg_rendering_output
                bg_rgb = bg_rgb_flat.view(
                    batch_size, num_pixels, N_bg_samples, 3)
            bg_weights = self.bg_volume_rendering(z_vals_bg, bg_sdf)
            bg_rgb_values = torch.sum(bg_weights.unsqueeze(-1) * bg_rgb, 2)
        else:
            bg_rgb_values = torch.ones_like(
                fg_rgb_values, device=fg_rgb_values.device)

        # Composite foreground and background
        bg_rgb_values = bg_transmittance.unsqueeze(-1) * bg_rgb_values
        rgb_values = fg_rgb_values + bg_rgb_values

        normal_values = torch.sum(weights.unsqueeze(-1) * normal_values, 2)

        # Depth map
        depth = torch.norm(points - cam_loc.unsqueeze(2), dim=-1)
        depth = torch.sum(weights * depth, dim=-1)

        if self.training:
            output = {
                "points": points,
                "rgb_values": rgb_values,
                "normal_values": normal_values,
                # "index_outside": input["index_outside"],
                "index_off_surface": index_off_surface,
                "index_in_surface": index_in_surface,
                "acc_map": torch.sum(weights, -1),
                "sdf_output": sdf_output,
                "grad_theta": grad_theta,
                "epoch": input["current_epoch"],
            }
        else:
            fg_output_rgb = fg_rgb_values + bg_transmittance.unsqueeze(
                -1
            ) * torch.ones_like(fg_rgb_values, device=fg_rgb_values.device)
            output = {
                "acc_map": torch.sum(weights, -1),
                "rgb_values": rgb_values,
                "fg_rgb_values": fg_output_rgb,
                "normal_values": normal_values,
                "sdf_output": sdf_output,
                "depth": depth,
            }
        return output

    def get_rbg_value(
        self, x, points, view_dirs, cond, tfs, feature_vectors, is_training=True
    ):
        points = einops.rearrange(points, "b n s p -> b (n s) p")

        pnts_c = points
        others = {}

        gradients, feature_vectors = self.forward_gradient(
            pnts_c, cond, tfs, create_graph=is_training, retain_graph=is_training
        )
        # ensure the gradient is normalized
        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6)

        view_dirs = einops.rearrange(view_dirs, "b n s p -> b (n s) p")

        fg_rendering_output = self.rendering_network(
            pnts_c, normals, view_dirs, cond["smpl"], feature_vectors
        )

        rgb_vals = fg_rendering_output[..., :3]
        others["normals"] = normals
        return rgb_vals, others

    def forward_gradient(self, pnts_c, cond, tfs, create_graph=True, retain_graph=True):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)

        pnts_d = self.deformer.forward_skinning(pnts_c, tfs)
        num_dim = pnts_d.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(
                pnts_d, requires_grad=False, device=pnts_d.device)
            d_out[:, :, i] = 1
            grad = torch.autograd.grad(
                outputs=pnts_d,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=create_graph,
                retain_graph=True if i < num_dim - 1 else retain_graph,
                only_inputs=True,
            )[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)

        grads_inv = grads.inverse()

        output = self.implicit_network(pnts_c, cond)
        sdf = output[..., :1]

        feature = output[..., 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=pnts_c,
            grad_outputs=d_output,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True,
        )[0]

        return torch.einsum("bni,bnij->bnj", gradients, grads_inv), feature

    def volume_rendering(self, z_vals, z_max, sdf):
        batch_size, num_pixels, N_samples = z_vals.shape
        density = self.density(sdf)

        # included also the dist from the sphere intersection
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, z_max.unsqueeze(-1) - z_vals[..., -1:]], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat(
            [
                torch.zeros(dists.shape[0], dists.shape[1], 1).to(
                    z_vals.device),
                free_energy,
            ],
            dim=-1,
        )  # add 0 for transperancy 1 at t_0
        # probability of it is not empty here
        alpha = 1 - torch.exp(-free_energy)
        transmittance = torch.exp(
            -torch.cumsum(shifted_free_energy, dim=-1)
        )  # probability of everything is empty up to now
        fg_transmittance = transmittance[..., :-1]
        weights = alpha * fg_transmittance  # probability of the ray hits something here
        bg_transmittance = transmittance[
            ..., -1
        ]  # factor to be multiplied with the bg volume rendering

        return weights, bg_transmittance

    def bg_volume_rendering(self, z_vals_bg, bg_sdf):
        bg_density = self.bg_density(bg_sdf)
        # bg_density = bg_density_flat.view(
        #     -1, z_vals_bg.shape[1]
        # )  # (batch_size * num_pixels) x N_samples

        bg_dists = z_vals_bg[..., :-1] - z_vals_bg[..., 1:]
        bg_dists = torch.cat(
            [
                bg_dists,
                torch.tensor([1e10])
                .to(z_vals_bg.device)
                .unsqueeze(0)
                .unsqueeze(1)
                .repeat(bg_dists.shape[0], bg_dists.shape[1], 1),
            ],
            -1,
        )

        # LOG SPACE
        bg_free_energy = bg_dists * bg_density
        bg_shifted_free_energy = torch.cat(
            [
                torch.zeros(bg_dists.shape[0], bg_dists.shape[1], 1).to(
                    z_vals_bg.device
                ),
                bg_free_energy[..., :-1],
            ],
            dim=-1,
        )  # shift one step
        # probability of it is not empty here
        bg_alpha = 1 - torch.exp(-bg_free_energy)
        bg_transmittance = torch.exp(
            -torch.cumsum(bg_shifted_free_energy, dim=-1)
        )  # probability of everything is empty up to now
        bg_weights = (
            bg_alpha * bg_transmittance
        )  # probability of the ray hits something here

        return bg_weights

    def depth2pts_outside(self, ray_o, ray_d, depth):
        """
        ray_o, ray_d: [..., 3]
        depth: [...]; inverse of distance to sphere origin
        """

        o_dot_d = torch.sum(ray_d * ray_o, dim=-1)
        under_sqrt = o_dot_d**2 - (
            (ray_o**2).sum(-1) - self.sdf_bounding_sphere**2
        )
        d_sphere = torch.sqrt(under_sqrt) - o_dot_d
        p_sphere = ray_o + d_sphere.unsqueeze(-1) * ray_d
        p_mid = ray_o - o_dot_d.unsqueeze(-1) * ray_d
        p_mid_norm = torch.norm(p_mid, dim=-1)

        rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
        rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
        phi = torch.asin(p_mid_norm / self.sdf_bounding_sphere)
        theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
        rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

        # now rotate p_sphere
        # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        p_sphere_new = (
            p_sphere * torch.cos(rot_angle)
            + torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle)
            + rot_axis
            * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True)
            * (1.0 - torch.cos(rot_angle))
        )
        p_sphere_new = p_sphere_new / \
            torch.norm(p_sphere_new, dim=-1, keepdim=True)
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

        return pts


def gradient(inputs, outputs):
    d_points = torch.ones_like(
        outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0][:, :, -3:]
    return points_grad
