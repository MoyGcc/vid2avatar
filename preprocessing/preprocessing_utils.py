"""This module contains simple helper functions and classes for preprocessing """
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.transforms import euler_angles_to_matrix
import math
from pytorch3d.renderer.cameras import CamerasBase

DEFAULT_DTYPE = torch.float32
INVALID_TRANS = np.ones(3) * -1


def smpl_to_pose(
    model_type="smplx",
    use_hands=True,
    use_face=True,
    use_face_contour=False,
    openpose_format="coco25",
):
    """Returns the indices of the permutation that maps OpenPose to SMPL
    Parameters
    ----------
    model_type: str, optional
        The type of SMPL-like model that is used. The default mapping
        returned is for the SMPLX model
    use_hands: bool, optional
        Flag for adding to the returned permutation the mapping for the
        hand keypoints. Defaults to True
    use_face: bool, optional
        Flag for adding to the returned permutation the mapping for the
        face keypoints. Defaults to True
    use_face_contour: bool, optional
        Flag for appending the facial contour keypoints. Defaults to False
    openpose_format: bool, optional
        The output format of OpenPose. For now only COCO-25 and COCO-19 is
        supported. Defaults to 'coco25'
    """
    if openpose_format.lower() == "coco25":
        if model_type == "smpl":
            return np.array(
                [
                    24,
                    12,
                    17,
                    19,
                    21,
                    16,
                    18,
                    20,
                    0,
                    2,
                    5,
                    8,
                    1,
                    4,
                    7,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                ],
                dtype=np.int32,
            )
        elif model_type == "smplh":
            body_mapping = np.array(
                [
                    52,
                    12,
                    17,
                    19,
                    21,
                    16,
                    18,
                    20,
                    0,
                    2,
                    5,
                    8,
                    1,
                    4,
                    7,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                ],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        34,
                        35,
                        36,
                        63,
                        22,
                        23,
                        24,
                        64,
                        25,
                        26,
                        27,
                        65,
                        31,
                        32,
                        33,
                        66,
                        28,
                        29,
                        30,
                        67,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        49,
                        50,
                        51,
                        68,
                        37,
                        38,
                        39,
                        69,
                        40,
                        41,
                        42,
                        70,
                        46,
                        47,
                        48,
                        71,
                        43,
                        44,
                        45,
                        72,
                    ],
                    dtype=np.int32,
                )
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == "smplx":
            body_mapping = np.array(
                [
                    55,
                    12,
                    17,
                    19,
                    21,
                    16,
                    18,
                    20,
                    0,
                    2,
                    5,
                    8,
                    1,
                    4,
                    7,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                    63,
                    64,
                    65,
                ],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        37,
                        38,
                        39,
                        66,
                        25,
                        26,
                        27,
                        67,
                        28,
                        29,
                        30,
                        68,
                        34,
                        35,
                        36,
                        69,
                        31,
                        32,
                        33,
                        70,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        52,
                        53,
                        54,
                        71,
                        40,
                        41,
                        42,
                        72,
                        43,
                        44,
                        45,
                        73,
                        49,
                        50,
                        51,
                        74,
                        46,
                        47,
                        48,
                        75,
                    ],
                    dtype=np.int32,
                )

                mapping += [lhand_mapping, rhand_mapping]

            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(
                    76, 127 + 17 * use_face_contour, dtype=np.int32
                )
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError("Unknown model type: {}".format(model_type))
    elif openpose_format == "coco19":
        if model_type == "smpl":
            return np.array(
                [24, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28],
                dtype=np.int32,
            )
        elif model_type == "smpl_neutral":
            return np.array(
                [
                    14,
                    12,
                    8,
                    7,
                    6,
                    9,
                    10,
                    11,
                    2,
                    1,
                    0,
                    3,
                    4,
                    5,
                    16,
                    15,
                    18,
                    17,
                ],
                dtype=np.int32,
            )

        elif model_type == "smplh":
            body_mapping = np.array(
                [52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 53, 54, 55, 56],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        34,
                        35,
                        36,
                        57,
                        22,
                        23,
                        24,
                        58,
                        25,
                        26,
                        27,
                        59,
                        31,
                        32,
                        33,
                        60,
                        28,
                        29,
                        30,
                        61,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        49,
                        50,
                        51,
                        62,
                        37,
                        38,
                        39,
                        63,
                        40,
                        41,
                        42,
                        64,
                        46,
                        47,
                        48,
                        65,
                        43,
                        44,
                        45,
                        66,
                    ],
                    dtype=np.int32,
                )
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == "smplx":
            body_mapping = np.array(
                [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        37,
                        38,
                        39,
                        60,
                        25,
                        26,
                        27,
                        61,
                        28,
                        29,
                        30,
                        62,
                        34,
                        35,
                        36,
                        63,
                        31,
                        32,
                        33,
                        64,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        52,
                        53,
                        54,
                        65,
                        40,
                        41,
                        42,
                        66,
                        43,
                        44,
                        45,
                        67,
                        49,
                        50,
                        51,
                        68,
                        46,
                        47,
                        48,
                        69,
                    ],
                    dtype=np.int32,
                )

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(
                    70, 70 + 51 + 17 * use_face_contour, dtype=np.int32
                )
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError("Unknown model type: {}".format(model_type))
    elif openpose_format == "h36":
        if model_type == "smpl":
            return np.array(
                [2, 5, 8, 1, 4, 7, 12, 24, 16, 18, 20, 17, 19, 21], dtype=np.int32
            )
        elif model_type == "smpl_neutral":
            # return np.array([2,1,0,3,4,5,12,13,9,10,11,8,7,6], dtype=np.int32)
            return [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    else:
        raise ValueError("Unknown joint format: {}".format(openpose_format))


def render_trimesh(renderer, mesh, R, T, mode="np"):
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None, ..., :3] / 255
    renderer.set_camera(R, T)
    image = renderer.render_mesh_recon(verts, faces, colors=colors, mode=mode)[0]
    image = (255 * image).data.cpu().numpy().astype(np.uint8)

    return image


def estimate_translation_cv2(
    joints_3d,
    joints_2d,
    focal_length=600,
    img_size=np.array([512.0, 512.0]),
    proj_mat=None,
    cam_dist=None,
):
    if proj_mat is None:
        camK = np.eye(3)
        camK[0, 0], camK[1, 1] = focal_length, focal_length
        camK[:2, 2] = img_size // 2
    else:
        camK = proj_mat
    _, _, tvec, inliers = cv2.solvePnPRansac(
        joints_3d,
        joints_2d,
        camK,
        cam_dist,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=20,
        iterationsCount=100,
    )

    if inliers is None:
        return INVALID_TRANS
    else:
        tra_pred = tvec[:, 0]
        return tra_pred


class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer(
                "joint_maps", torch.tensor(joint_maps, dtype=torch.long)
            )

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)


def transform_mat(R, t):
    """Creates a batch of transformation matrices
    Args:
        - R: Bx3x3 array of a batch of rotation matrices
        - t: Bx3x1 array of a batch of translation vectors
    Returns:
        - T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


# transform SMPL such that the target camera extrinsic will be met
def transform_smpl(curr_extrinsic, target_extrinsic, smpl_pose, smpl_trans, T_hip):
    R_root = cv2.Rodrigues(smpl_pose[:3])[0]
    transf_global_ori = (
        np.linalg.inv(target_extrinsic[:3, :3]) @ curr_extrinsic[:3, :3] @ R_root
    )

    target_extrinsic[:3, -1] = (
        curr_extrinsic[:3, :3] @ (smpl_trans + T_hip)
        + curr_extrinsic[:3, -1]
        - smpl_trans
        - target_extrinsic[:3, :3] @ T_hip
    )

    smpl_pose[:3] = cv2.Rodrigues(transf_global_ori)[0].reshape(3)
    smpl_trans = np.linalg.inv(target_extrinsic[:3, :3]) @ smpl_trans  # we assume

    return target_extrinsic, smpl_pose, smpl_trans


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return "rho = {}".format(self.rho)

    def forward(self, residual):
        squared_res = residual**2
        dist = torch.div(squared_res, squared_res + self.rho**2)
        return self.rho**2 * dist


class PerspectiveCamera(nn.Module):
    FOCAL_LENGTH = 50 * 128

    def __init__(
        self,
        rotation=None,
        translation=None,
        focal_length_x=None,
        focal_length_y=None,
        batch_size=1,
        center=None,
        dtype=torch.float32,
    ):
        super(PerspectiveCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer("zero", torch.zeros([batch_size], dtype=dtype))

        if focal_length_x is None or type(focal_length_x) == float:
            focal_length_x = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_x is None else focal_length_x,
                dtype=dtype,
            )

        if focal_length_y is None or type(focal_length_y) == float:
            focal_length_y = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_y is None else focal_length_y,
                dtype=dtype,
            )

        self.register_buffer("focal_length_x", focal_length_x)
        self.register_buffer("focal_length_y", focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer("center", center)

        if rotation is None:
            rotation = (
                torch.eye(3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)
            )

        rotation = nn.Parameter(rotation, requires_grad=False)
        self.register_parameter("rotation", rotation)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)

        translation = nn.Parameter(translation, requires_grad=True)
        self.register_parameter("translation", translation)

    def forward(self, points):
        device = points.device
        with torch.no_grad():
            camera_mat = torch.zeros(
                [self.batch_size, 2, 2], dtype=self.dtype, device=points.device
            )
            camera_mat[:, 0, 0] = self.focal_length_x
            camera_mat[:, 1, 1] = self.focal_length_y

        camera_transform = transform_mat(
            self.rotation, self.translation.unsqueeze(dim=-1)
        )

        homog_coord = torch.ones(
            list(points.shape)[:-1] + [1], dtype=points.dtype, device=device
        )
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum("bki,bji->bjk", [camera_transform, points_h])

        img_points = torch.div(
            projected_points[:, :, :2], projected_points[:, :, 2].unsqueeze(dim=-1)
        )
        img_points = torch.einsum(
            "bki,bji->bjk", [camera_mat, img_points]
        ) + self.center.unsqueeze(dim=1)
        return img_points


class Renderer:
    def __init__(self, principal_point=None, img_size=None, cam_intrinsic=None):
        super().__init__()

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.cam_intrinsic = cam_intrinsic
        self.image_size = img_size
        self.render_img_size = np.max(img_size)

        principal_point = [
            -(self.cam_intrinsic[0, 2] - self.image_size[1] / 2.0)
            / (self.image_size[1] / 2.0),
            -(self.cam_intrinsic[1, 2] - self.image_size[0] / 2.0)
            / (self.image_size[0] / 2.0),
        ]
        self.principal_point = torch.tensor(
            principal_point, device=self.device
        ).unsqueeze(0)

        self.cam_R = (
            torch.from_numpy(
                np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
            )
            .cuda()
            .float()
            .unsqueeze(0)
        )

        self.cam_T = torch.zeros((1, 3)).cuda().float()

        half_max_length = max(self.cam_intrinsic[0:2, 2])
        self.focal_length = torch.tensor(
            [
                (self.cam_intrinsic[0, 0] / half_max_length).astype(np.float32),
                (self.cam_intrinsic[1, 1] / half_max_length).astype(np.float32),
            ]
        ).unsqueeze(0)

        self.cameras = SfMPerspectiveCameras(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            R=self.cam_R,
            T=self.cam_T,
            device=self.device,
        )

        self.lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, 0.0]],
            ambient_color=((1, 1, 1),),
            diffuse_color=((0, 0, 0),),
            specular_color=((0, 0, 0),),
        )

        self.raster_settings = RasterizationSettings(
            image_size=self.render_img_size,
            faces_per_pixel=10,
            blur_radius=0,
            max_faces_per_bin=30000,
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, raster_settings=self.raster_settings
        )

        self.shader = SoftPhongShader(
            device=self.device, cameras=self.cameras, lights=self.lights
        )

        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

    def set_camera(self, R, T):
        self.cam_R = R
        self.cam_T = T
        self.cam_R[:, :2, :] *= -1.0
        self.cam_T[:, :2] *= -1.0
        self.cam_R = torch.transpose(self.cam_R, 1, 2)
        self.cameras = SfMPerspectiveCameras(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            R=self.cam_R,
            T=self.cam_T,
            device=self.device,
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, raster_settings=self.raster_settings
        )
        self.shader = SoftPhongShader(
            device=self.device, cameras=self.cameras, lights=self.lights
        )
        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

    def render_mesh_recon(self, verts, faces, R=None, T=None, colors=None, mode="npat"):
        """
        mode: normal, phong, texture
        """
        with torch.no_grad():
            mesh = Meshes(verts, faces)

            normals = torch.stack(mesh.verts_normals_list())
            front_light = -torch.tensor([0, 0, -1]).float().to(verts.device)
            shades = (
                (normals * front_light.view(1, 1, 3))
                .sum(-1)
                .clamp(min=0)
                .unsqueeze(-1)
                .expand(-1, -1, 3)
            )
            results = []
            # shading
            if "p" in mode:
                mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)
            # normal
            if "n" in mode:
                normals_vis = normals * 0.5 + 0.5
                normals_vis = normals_vis[:, :, [2, 1, 0]]
                mesh_normal = Meshes(
                    verts, faces, textures=Textures(verts_rgb=normals_vis)
                )
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)
            return torch.cat(results, axis=1)


class EquirectangularCamera(CamerasBase):
    def __init__(self, R, T, image_size, world_coordinates: bool = False, device="cpu"):
        kwargs = {"image_size": image_size} if image_size is not None else {}
        super().__init__(
            device=device,
            R=R,
            T=T,
            **kwargs,
        )
        self.device = device
        self.world_coordinates = world_coordinates
        self.R = R
        self.T = T

        self.R = self.R.to(self.device)
        self.T = self.T.to(self.device)

    def transform_points(self, points, **kwargs) -> torch.Tensor:
        """
        Transform input points from camera space to image space.
        Args:
            points: tensor of (..., 3). E.g., (P, 3) or (1, P, 3), (M, P, 3)
            eps: tiny number to avoid zero divsion

        Returns:
            torch.Tensor
            when points take shape (P, 3) or (1, P, 3), output is (N, P, 3)
            when points take shape (M, P, 3), output is (M, P, 3)
            where N is the number of transforms, P number of points
        """
        # project from world space to camera space
        if self.world_coordinates:
            world_to_view_transform = self.get_world_to_view_transform(
                R=self.R, T=self.T
            )
            points = world_to_view_transform.transform_points(points.to(self.device))
        else:
            points = points.to(self.device)

        points = points.squeeze(0)
        assert points.dim() == 2, "check the dimension of points (P, 3)"
        # project from camera space to image space
        long, lat = self.point_to_long_lat(points)
        xy = self.long_lat_to_equirect(long, lat)
        return xy.unsqueeze(0)

    def forward(self, points):
        return self.transform_points(points)

    def point_to_long_lat(self, p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        long = torch.atan2(y, x)
        lat = torch.atan2(z, (x.pow(2) + y.pow(2)).sqrt())
        return long, lat

    def long_lat_to_equirect(self, long, lat):
        x = long / math.pi
        y = 2 * lat / math.pi
        return torch.stack([x, y], dim=-1)

    def long_lat_to_point(self, long, lat):
        x = torch.cos(lat) * torch.cos(long)
        y = torch.cos(lat) * torch.sin(long)
        z = torch.sin(lat)
        return torch.stack([x, y, z], dim=-1)

    def equirect_to_long_lat(self, p):
        long = p[..., 0] * math.pi
        lat = p[..., 1] * math.pi / 2
        return long, lat


# class Renderer:
#     def __init__(self, R, T, device="cpu") -> None:
#         self.device = device

#         self.cameras = EquirectangularCamera(
#             R=R, T=T, image_size=((720, 576),), device=device
#         )

#         self.lights = PointLights(
#             device=self.device,
#             location=[[0.0, 0.0, 0.0]],
#             ambient_color=((1, 1, 1),),
#             diffuse_color=((0, 0, 0),),
#             specular_color=((0, 0, 0),),
#         )

#         self.raster_settings = RasterizationSettings(
#             image_size=np.max([720, 576]),
#             faces_per_pixel=10,
#             blur_radius=0,
#             max_faces_per_bin=30000,
#         )
#         self.rasterizer = MeshRasterizer(
#             cameras=self.cameras, raster_settings=self.raster_settings
#         )

#         self.shader = SoftPhongShader(
#             device=self.device, cameras=self.cameras, lights=self.lights
#         )

#         self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

#     def forward(self, verts, faces, colors=None, mode=None):
#         with torch.no_grad():
#             mesh = Meshes(verts, faces)

#             normals = torch.stack(mesh.verts_normals_list())
#             front_light = -torch.tensor([0, 0, -1]).float().to(verts.device)
#             shades = (
#                 (normals * front_light.view(1, 1, 3))
#                 .sum(-1)
#                 .clamp(min=0)
#                 .unsqueeze(-1)
#                 .expand(-1, -1, 3)
#             )
#             results = []
#             # shading
#             mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
#             image_phong = self.renderer(mesh_shading)

#             results.append(image_phong)
#             return torch.cat(results, axis=1)


def euler_angles_to_matrix_zxy(euler_angles):
    x_angles, y_angles, z_angles = (
        euler_angles[:, 0],
        euler_angles[:, 1],
        euler_angles[:, 2],
    )
    cos_x, sin_x = np.cos(x_angles), np.sin(x_angles)
    cos_y, sin_y = np.cos(y_angles), np.sin(y_angles)
    cos_z, sin_z = np.cos(z_angles), np.sin(z_angles)

    rotation_matrices = np.zeros((euler_angles.shape[0], 3, 3))

    rotation_matrices[:, 0, 0] = cos_z * cos_y - sin_z * sin_x * sin_y
    rotation_matrices[:, 0, 1] = -cos_x * sin_z
    rotation_matrices[:, 0, 2] = cos_y * sin_x * sin_z + cos_z * sin_y

    rotation_matrices[:, 1, 0] = cos_y * sin_z + cos_z * sin_x * sin_y
    rotation_matrices[:, 1, 1] = cos_z * cos_x
    rotation_matrices[:, 1, 2] = sin_z * sin_y - cos_z * cos_y * sin_x

    rotation_matrices[:, 2, 0] = -cos_x * sin_y
    rotation_matrices[:, 2, 1] = sin_x
    rotation_matrices[:, 2, 2] = cos_x * cos_y

    return rotation_matrices


def load_pos_init(init_pos_path, indices=[[]]):
    if init_pos_path is not None:
        with open(init_pos_path, "r") as f:
            init = f.readlines()
        init = list(map(lambda x: list(map(lambda y: float(y), x.split(" "))), init))

    init = np.array(init)
    if indices:
        init = init[indices]
    init = coord_y_up_to_minus_y_up_translate(init)
    # init = cm_to_mm(init)

    return init


def load_rotate_init(init_rotate_path, indices=[[]]):
    if init_rotate_path is not None:
        with open(init_rotate_path, "r") as f:
            init = f.readlines()
        init = list(map(lambda x: list(map(lambda y: float(y), x.split(" "))), init))

    init = np.array(init)
    if indices:
        init = init[indices]
    init = coord_y_up_to_minus_y_up_rotate(init)
    init = degree_to_radian(init)

    R = euler_angles_to_matrix_zxy(init)
    return R


def coord_y_up_to_minus_y_up_translate(T):
    x = T[:, 0]
    y = -T[:, 1]
    z = -T[:, 2]
    return np.stack([x, y, z], axis=-1)


def coord_y_up_to_minus_y_up_rotate(R):
    roll = R[:, 0]
    pitch = -R[:, 1]
    yaw = -R[:, 2]
    return np.stack([roll, pitch, yaw], axis=-1)


def degree_to_radian(degree):
    return degree * math.pi / 180


def radian_to_degree(radian):
    return radian * 180 / math.pi


def mm_to_cm(x):
    return x / 10


def cm_to_mm(x):
    return x * 10
