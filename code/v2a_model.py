import einops
import pytorch_lightning as pl
import torch.optim as optim
from lib.model.v2a import V2A
from lib.model.body_model_params import BodyModelParams
import cv2
import torch
from lib.model.loss import Loss
import hydra
import os
import numpy as np
from lib.utils.meshing import generate_mesh
from kaolin.ops.mesh import index_vertices_by_faces
import trimesh
from lib.model.deformer import skinning
from lib.utils import utils
from torch.nn.functional import interpolate


class V2AModel(pl.LightningModule):
    def __init__(self, opt) -> None:
        super().__init__()

        self.opt = opt
        num_training_frames = (
            opt.dataset.metainfo.end_frame - opt.dataset.metainfo.start_frame
        )
        self.betas_path = os.path.join(
            hydra.utils.to_absolute_path("."),
            opt.dataset.metainfo.data_dir,
            "mean_shape.npy",
        )
        self.gender = opt.dataset.metainfo.gender
        self.model = V2A(opt.model, self.betas_path,
                         self.gender, num_training_frames)
        self.start_frame = opt.dataset.metainfo.start_frame
        self.end_frame = opt.dataset.metainfo.end_frame
        self.training_modules = ["model"]

        self.training_indices = list(range(self.start_frame, self.end_frame))
        self.body_model_params = BodyModelParams(
            num_training_frames, model_type="smpl")
        self.load_body_model_params()
        optim_params = self.body_model_params.param_names
        for param_name in optim_params:
            self.body_model_params.set_requires_grad(
                param_name, requires_grad=True)
        self.training_modules += ["body_model_params"]

        self.loss = Loss(opt.model.loss)
        self.automatic_optimization = False

    def load_body_model_params(self):
        body_model_params = {
            param_name: [] for param_name in self.body_model_params.param_names
        }
        data_root = os.path.join(".", self.opt.dataset.metainfo.data_dir)
        data_root = hydra.utils.to_absolute_path(data_root)

        body_model_params["betas"] = torch.tensor(
            np.load(os.path.join(data_root, "mean_shape.npy"))[None],
            dtype=torch.float32,
        )
        poses = np.load(os.path.join(data_root, "poses.npy"))
        num_samples = poses.shape[0]
        poses = poses.reshape(num_samples, -1)
        body_model_params["global_orient"] = torch.tensor(
            poses[self.training_indices][:, :3],
            dtype=torch.float32,
        )
        body_model_params["body_pose"] = torch.tensor(
            poses[self.training_indices][:, 3:],
            dtype=torch.float32,
        )
        body_model_params["transl"] = torch.tensor(
            np.load(os.path.join(data_root, "normalize_trans.npy")).squeeze()[
                self.training_indices
            ],
            dtype=torch.float32,
        )

        for param_name in body_model_params.keys():
            self.body_model_params.init_parameters(
                param_name, body_model_params[param_name], requires_grad=False
            )

    def configure_optimizers(self):
        params = [
            {"params": self.model.parameters(), "lr": self.opt.model.learning_rate}
        ]
        body_params = [
            {
                "params": self.body_model_params.parameters(),
                "lr": self.opt.model.body_param_learning_rate,
            }
        ]

        optimizer = optim.Adam(
            params, lr=self.opt.model.learning_rate, eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.opt.model.sched_milestones,
            gamma=self.opt.model.sched_factor,
        )

        body_param_optimizer = optim.RMSprop(
            body_params, lr=self.opt.model.body_param_learning_rate
        )
        return optimizer, body_param_optimizer

    def training_step(self, batch):
        opt, opt_bp = self.optimizers()
        opt.zero_grad()
        opt_bp.zero_grad()
        inputs, targets = batch

        batch_idx = inputs["idx"]

        body_model_params = self.body_model_params(batch_idx)
        inputs["smpl_pose"] = torch.cat(
            (body_model_params["global_orient"],
             body_model_params["body_pose"]), dim=1
        )
        inputs["smpl_shape"] = body_model_params["betas"]
        inputs["smpl_trans"] = body_model_params["transl"]

        inputs["current_epoch"] = self.current_epoch
        model_outputs = self.model(inputs)

        loss_output = self.loss(model_outputs, targets)
        for k, v in loss_output.items():
            if k in ["loss"]:
                self.log(k, v.item(), prog_bar=True, on_step=True)
            else:
                self.log(k, v.item(), prog_bar=True, on_step=True)

        self.manual_backward(loss_output["loss"])
        opt.step()
        opt_bp.step()
        return loss_output["loss"]

    def training_epoch_end(self, outputs) -> None:
        cond = {"smpl": torch.zeros(1, 69).float().cuda()}
        mesh_canonical = generate_mesh(
            lambda x: self.get_sdf_from_canonical(x, cond),
            self.model.smpl_server.verts_c[0],
            point_batch=10000,
            res_up=2,
        )
        self.model.mesh_v_cano = torch.tensor(
            mesh_canonical.vertices[None], device=self.model.mesh_v_cano.device
        ).float()
        self.model.mesh_f_cano = torch.tensor(
            mesh_canonical.faces.astype(np.int64),
            device=self.model.mesh_v_cano.device,
        )
        self.model.mesh_face_vertices = index_vertices_by_faces(
            self.model.mesh_v_cano, self.model.mesh_f_cano
        )
        return super().training_epoch_end(outputs)

    def get_sdf_from_canonical(self, x, cond):
        x = x.view(-1, 3)
        mnfld_pred = self.model.implicit_network(x, cond)[:, :, 0].view(-1, 1)
        return {"sdf": mnfld_pred}

    def get_deformed_mesh_fast_mode(self, verts, smpl_tfs, smpl_weights):
        weights = self.model.deformer.query_weights(verts, smpl_weights)
        verts_deformed = (
            skinning(verts, weights,
                     smpl_tfs).data.cpu().numpy()[0]
        )
        return verts_deformed

    def validation_step(self, batch, *args, **kwargs):
        output = {}
        inputs, targets = batch
        inputs["current_epoch"] = self.current_epoch
        self.model.eval()

        body_model_params = self.body_model_params(inputs["idx"])
        inputs["smpl_pose"] = torch.cat(
            (body_model_params["global_orient"],
             body_model_params["body_pose"]), dim=1
        )
        inputs["smpl_shape"] = body_model_params["betas"]
        inputs["smpl_trans"] = body_model_params["transl"]

        cond = {"smpl": inputs["smpl_pose"][:, 3:] / np.pi}
        mesh_canonical = generate_mesh(
            lambda x: self.get_sdf_from_canonical(x, cond),
            self.model.smpl_server.verts_c[0],
            point_batch=10000,
            res_up=3,
        )

        mesh_canonical = trimesh.Trimesh(
            mesh_canonical.vertices, mesh_canonical.faces)

        output.update({"canonical_mesh": mesh_canonical})

        total_pixels = targets["img_size"][0] * targets["img_size"][1]
        total_pixels = int(total_pixels.cpu().numpy())

        split = utils.split_input(
            inputs,
            total_pixels,
            n_pixels=min(targets["pixel_per_batch"], total_pixels),
        )

        res = []
        for s in split:
            out = self.model(s)

            for k, v in out.items():
                try:
                    out[k] = v.detach()
                except:
                    out[k] = v

            res.append(
                {
                    "rgb_values": out["rgb_values"].detach(),
                    "normal_values": out["normal_values"].detach(),
                    "fg_rgb_values": out["fg_rgb_values"].detach(),
                }
            )
        batch_size = targets["rgb"].shape[0]

        model_outputs = utils.merge_output(
            res, total_pixels, batch_size)

        output.update(
            {
                "rgb_values": model_outputs["rgb_values"].detach().clone(),
                "normal_values": model_outputs["normal_values"].detach().clone(),
                "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
                **targets,
            }
        )

        return output

    def validation_step_end(self, batch_parts):
        return batch_parts

    def validation_epoch_end(self, outputs) -> None:
        img_size = outputs[0]["img_size"]

        rgb_pred = torch.cat([output["rgb_values"]
                             for output in outputs], dim=0)
        rgb_pred = rgb_pred.view(*img_size, -1)

        fg_rgb_pred = torch.cat([output["fg_rgb_values"]
                                for output in outputs], dim=0)
        fg_rgb_pred = fg_rgb_pred.view(*img_size, -1)

        normal_pred = torch.cat([output["normal_values"]
                                for output in outputs], dim=0)
        normal_pred = (normal_pred.view(*img_size, -1) + 1) / 2

        rgb_gt = torch.cat([output["rgb"]
                           for output in outputs], dim=1).squeeze(0)
        rgb_gt = rgb_gt.view(*img_size, -1)
        if "normal" in outputs[0].keys():
            normal_gt = torch.cat(
                [output["normal"] for output in outputs], dim=1
            ).squeeze(0)
            normal_gt = (normal_gt.view(*img_size, -1) + 1) / 2
            normal = torch.cat([normal_gt, normal_pred], dim=0).cpu().numpy()
        else:
            normal = torch.cat([normal_pred], dim=0).cpu().numpy()

        rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        rgb = (rgb * 255).astype(np.uint8)

        fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        os.makedirs("rendering", exist_ok=True)
        os.makedirs("normal", exist_ok=True)
        os.makedirs("fg_rendering", exist_ok=True)

        canonical_mesh = outputs[0]["canonical_mesh"]
        canonical_mesh.export(f"rendering/{self.current_epoch}.ply")

        cv2.imwrite(f"rendering/{self.current_epoch}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"normal/{self.current_epoch}.png", normal[:, :, ::-1])
        cv2.imwrite(
            f"fg_rendering/{self.current_epoch}.png", fg_rgb[:, :, ::-1])

    def test_step(self, batch, *args, **kwargs):
        cfg = self.opt.dataset

        img_size = cfg.metainfo.img_size
        output_img_size = cfg.test.output_img_size if cfg.test.output_img_size is not None else img_size
        pixel_per_batch = cfg.test.pixel_per_batch

        total_pixels = batch["uv"].size(0) * batch["uv"].size(1)
        num_splits = (total_pixels + pixel_per_batch - 1) // pixel_per_batch

        scale = batch["scale"]

        body_model_params = self.body_model_params(batch["idx"])
        smpl_shape = (
            body_model_params["betas"]
            if body_model_params["betas"].dim() == 2
            else body_model_params["betas"].unsqueeze(0)
        )
        smpl_trans = body_model_params["transl"]
        smpl_pose = torch.cat(
            (body_model_params["global_orient"],
             body_model_params["body_pose"]), dim=1
        )

        smpl_outputs = self.model.smpl_server(
            scale, smpl_trans, smpl_pose, smpl_shape)
        smpl_tfs = smpl_outputs["smpl_tfs"]
        cond = {"smpl": smpl_pose[:, 3:] / np.pi}

        results = []

        for i in range(num_splits):
            indices = list(range(i * pixel_per_batch,
                           min((i+1) * pixel_per_batch, total_pixels)))
            batch_inputs = {
                "uv": batch["uv"][:, indices],
                "camera_poses": batch["camera_poses"],
                "camera_rotates": batch["camera_rotates"],
                "smpl_pose": smpl_pose,
                "smpl_shape": smpl_shape,
                "smpl_trans": smpl_trans,
                "idx": batch["idx"] if "idx" in batch.keys() else None,
                "scale": scale,
            }

            batch_inputs.update(
                {
                    "smpl_pose": torch.cat(
                        (
                            body_model_params["global_orient"],
                            body_model_params["body_pose"],
                        ),
                        dim=1,
                    )
                }
            )
            batch_inputs.update({"smpl_shape": body_model_params["betas"]})
            batch_inputs.update({"smpl_trans": body_model_params["transl"]})

            with torch.no_grad():
                model_outputs = self.model(batch_inputs)
            results.append(
                {
                    "rgb_values": model_outputs["rgb_values"].detach().clone(),
                    "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
                    "normal_values": model_outputs["normal_values"].detach().clone(),
                    "acc_map": model_outputs["acc_map"].detach().clone(),
                    "depth": model_outputs["depth"].detach().clone(),
                }
            )

        idx = int(batch['idx'].cpu().numpy())

        if cfg.test.rendering_gt.is_use:
            # rgb gt
            os.makedirs("test_rendering_gt", exist_ok=True)

            rgb_gt = batch['rgb']
            rgb_gt = rgb_gt.reshape(*img_size, -1)
            rgb_gt = rgb_gt.cpu().numpy()
            rgb_gt = (rgb_gt * 255).astype(np.uint8)
            rgb_gt = rgb_gt[:, :, ::-1]

            cv2.imwrite(
                f"test_rendering_gt/{idx:04d}.png", rgb_gt)

        if cfg.test.normal_map.is_use:
            os.makedirs("test_normal", exist_ok=True)

            normal_pred = torch.cat([result["normal_values"]
                                    for result in results], dim=0)
            normal_pred = (normal_pred.reshape(*output_img_size, -1) + 1) / 2
            normal_pred = torch.cat([normal_pred], dim=0).cpu().numpy()
            normal_pred = (normal_pred * 255).astype(np.uint8)
            normal_pred = normal_pred[:, :, ::-1]

            cv2.imwrite(
                f"test_normal/{idx:04d}.png", normal_pred)

        if cfg.test.depth_map.is_use:
            os.makedirs("test_depth", exist_ok=True)

            depth_pred = torch.cat([result["depth"]
                                    for result in results], dim=0)
            depth_pred = depth_pred.reshape(*output_img_size, -1)
            depth_pred = torch.cat([depth_pred], dim=0).cpu().numpy()

            depth_pred = depth_pred / depth_pred.max()  # 0 ~ 1
            depth_pred = depth_pred * 255
            depth_pred = depth_pred.astype(np.uint8)
            depth_pred = depth_pred[:, :, ::-1]
            # depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

            cv2.imwrite(
                f"test_depth/{idx:04d}.png", depth_pred)

        if cfg.test.mesh.is_use:
            os.makedirs("test_mesh", exist_ok=True)

            mesh_canonical = generate_mesh(
                lambda x: self.get_sdf_from_canonical(x, cond),
                self.model.smpl_server.verts_c[0],
                point_batch=cfg.test.mesh.point_batch,
                res_up=4,
            )
            verts = torch.tensor(mesh_canonical.vertices, device=batch["uv"].device,
                                 dtype=torch.float).unsqueeze(0)
            verts_deformed = self.get_deformed_mesh_fast_mode(
                verts, smpl_tfs, smpl_outputs["smpl_weights"]
            )
            mesh_deformed = trimesh.Trimesh(
                vertices=verts_deformed, faces=mesh_canonical.faces, process=False
            )

            mesh_canonical.export(
                f"test_mesh/{idx:04d}_canonical.ply")
            mesh_deformed.export(
                f"test_mesh/{idx:04d}_deformed.ply")

        if cfg.test.rendering.is_use:
            os.makedirs("test_rendering", exist_ok=True)

            rgb_pred = torch.cat([result["rgb_values"]
                                  for result in results], dim=0)
            rgb_pred = rgb_pred.reshape(*output_img_size, -1)
            rgb_pred = torch.cat([rgb_pred], dim=0).cpu().numpy()
            rgb_pred = (rgb_pred * 255).astype(np.uint8)
            rgb_pred = rgb_pred[:, :, ::-1]

            cv2.imwrite(
                f"test_rendering/{idx:04d}.png", rgb_pred)

        if cfg.test.fg_rendering.is_use:
            os.makedirs("test_fg_rendering", exist_ok=True)

            fg_rgb_pred = torch.cat([result["fg_rgb_values"]
                                    for result in results], dim=0)
            fg_rgb_pred = fg_rgb_pred.reshape(*output_img_size, -1)
            fg_rgb_pred = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
            fg_rgb_pred = (fg_rgb_pred * 255).astype(np.uint8)
            fg_rgb_pred = fg_rgb_pred[:, :, ::-1]

            cv2.imwrite(
                f"test_fg_rendering/{idx:04d}.png", fg_rgb_pred)

        if cfg.test.mask.is_use:
            os.makedirs("test_mask", exist_ok=True)

            pred_mask = torch.cat([result["acc_map"]
                                  for result in results], dim=0)
            pred_mask = pred_mask.reshape(*output_img_size, -1)
            pred_mask = pred_mask.cpu().numpy() * 255

            cv2.imwrite(
                f"test_mask/{idx:04d}.png", pred_mask)

        if cfg.test.relight.is_use:
            os.makedirs("test_relight", exist_ok=True)

            rgb_gt = batch['rgb']
            uv = batch["uv"]
            view_dir = utils.equirect_to_spherical(uv)

            normal_pred = torch.cat([result["normal_values"]
                                    for result in results], dim=0)
            normal_pred = normal_pred.reshape(1, -1, 3)

            assert (rgb_gt.shape == normal_pred.shape)

            lighting_sum = 0
            for light_color, light_dir in zip(cfg.test.relight.light_colors, cfg.test.relight.light_directions):
                light_dir = torch.tensor(light_dir, device=rgb_gt.device)
                light_dir = light_dir / torch.norm(light_dir)
                light_dir = light_dir.unsqueeze(0).unsqueeze(
                    0).repeat(*rgb_gt.shape[:2], 1)  # (b, n, 1)
                light_color = torch.tensor(light_color, device=rgb_gt.device)
                light_color = light_color.unsqueeze(0).unsqueeze(
                    0).repeat(*rgb_gt.shape[:2], 1)  # (b, n, 1)

                lighting = phong_lighting(normal_pred,
                                          light_dir,
                                          light_color,
                                          view_dir,
                                          shininess=cfg.test.relight.shininess,
                                          ambient=cfg.test.relight.ambient,
                                          diffuse=cfg.test.relight.diffuse,
                                          specular=cfg.test.relight.specular)

                lighting_sum += lighting

            relighted_rgb = rgb_gt * lighting_sum
            relighted_rgb = relighted_rgb.clamp(min=0.0, max=1.0)

            relighted_rgb = relighted_rgb.reshape(*output_img_size, -1)

            relighted_rgb = torch.cat([relighted_rgb], dim=0).cpu().numpy()
            relighted_rgb = (relighted_rgb * 255).astype(np.uint8)
            relighted_rgb = relighted_rgb[:, :, ::-1]

            cv2.imwrite(
                f"test_relight/{idx:04d}.png", relighted_rgb)


def phong_lighting(normal_map,
                   light_dir,
                   light_color,
                   view_dir,
                   specular=1.0,
                   diffuse=0.8,
                   ambient=0.2,
                   shininess=32):
    """
    normal_map (torch.Tensor): (b, n, 3)
    light_dir (torch.Tensor): (b, n, 3)
    light_color (torch.Tensor): (b, n, 3)
    view_dir (torch.Tensor): (b, n, 3)
    """

    # Specular
    reflect_dir = 2 * torch.einsum("bnd,bnd->bn", normal_map,
                                   light_dir).unsqueeze(-1) * normal_map - light_dir  # Law of Reflection
    cos_theta = torch.einsum(
        "bnd,bnd->bn", reflect_dir, -view_dir).unsqueeze(-1)
    cos_theta = torch.clamp(cos_theta, min=0.0, max=1.0)
    specular_term = (cos_theta ** shininess)
    specular_component = specular * specular_term * light_color

    # Diffuse
    diffuse_term = torch.einsum(
        "bnd,bnd->bn", light_dir, normal_map).unsqueeze(-1)
    diffuse_term = torch.clamp(diffuse_term, min=0.0, max=1.0)
    diffuse_component = diffuse * diffuse_term * light_color

    # Ambient
    ambient_component = ambient * light_color

    lighting = ambient_component + diffuse_component + specular_component

    return lighting
