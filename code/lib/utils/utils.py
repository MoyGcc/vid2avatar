import numpy as np
import cv2
import torch
from torch.nn import functional as F
import math
import pytorch3d.transforms as transforms
from pytorch3d.transforms import euler_angles_to_matrix
from scipy.interpolate import interp2d
import einops


def split_input(model_input, total_pixels, n_pixels=10000):
    """
    Split the input to fit Cuda memory for large resolution.
    Can decrease the value of n_pixels in case of cuda out of memory error.
    """

    split = []

    for i, indx in enumerate(
        torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)
    ):
        data = model_input.copy()
        data["uv"] = torch.index_select(model_input["uv"], 1, indx)
        split.append(data)
    return split


def merge_output(res, total_pixels, batch_size):
    """Merge the split output."""

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat(
                [r[entry].reshape(batch_size, -1, 1) for r in res], 1
            ).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat(
                [r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res], 1
            ).reshape(batch_size * total_pixels, -1)
    return model_outputs


def get_psnr(img1, img2, normalize_rgb=False):
    if normalize_rgb:  # [-1,1] --> [0,1]
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0

    mse = torch.mean((img1 - img2) ** 2)
    psnr = -10.0 * torch.log(mse) / torch.log(torch.Tensor([10.0]).cuda())

    return psnr


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def rotation_pitch(data, angle):
    batch_size = data.size(0)
    if angle.dim() == 1:  # (1,)
        angle = angle.tile((batch_size, 1))  # (b, 1)
    zero = torch.zeros((batch_size, 1), dtype=torch.float32).to(data.device)  # (b, 1)
    one = torch.ones((batch_size, 1), dtype=torch.float32).to(data.device)  # (b, 1)
    pitch = torch.cat(
        [
            torch.cat([one, zero, zero], dim=1).unsqueeze(-1),
            torch.cat([zero, torch.cos(angle), -torch.sin(angle)], dim=1).unsqueeze(-1),
            torch.cat([zero, torch.sin(angle), torch.cos(angle)], dim=1).unsqueeze(-1),
        ],
        dim=2,
    )  # (b, 1) => (b, 3, 3)
    # (b, 3, 3), (b, 3) => (b, 3)
    data = pitch.bmm(data.unsqueeze(-1)).squeeze(-1)
    return data


def rotation_yaw(data, angle):
    batch_size = data.size(0)
    if angle.dim() == 1:  # (1,)
        angle = angle.tile((batch_size, 1))  # (b, 1)
    zero = torch.zeros((batch_size, 1), dtype=torch.float32).to(data.device)  # (b, 1)
    one = torch.ones((batch_size, 1), dtype=torch.float32).to(data.device)  # (b, 1)
    yaw = torch.cat(
        [
            torch.cat([torch.cos(angle), zero, torch.sin(angle)], dim=1).unsqueeze(-1),
            torch.cat([zero, one, zero], dim=1).unsqueeze(-1),
            torch.cat([-torch.sin(angle), zero, torch.cos(angle)], dim=1).unsqueeze(-1),
        ],
        dim=2,
    )  # (b, 1) => (b, 3, 3)
    # (b, 3, 3), (b, 3) => (b, 3)
    data = yaw.bmm(data.unsqueeze(-1)).squeeze(-1)
    return data


def rotation_roll(data, angle):
    batch_size = data.size(0)
    if angle.dim() == 1:  # (1,)
        angle = angle.tile((batch_size, 1))  # (b, 1)
    zero = torch.zeros((batch_size, 1), dtype=torch.float32).to(data.device)  # (b, 1)
    one = torch.ones((batch_size, 1), dtype=torch.float32).to(data.device)  # (b, 1)
    roll = torch.cat(
        [
            torch.cat([torch.cos(angle), -torch.sin(angle), zero], dim=1).unsqueeze(-1),
            torch.cat([torch.sin(angle), torch.cos(angle), zero], dim=1).unsqueeze(-1),
            torch.cat([zero, zero, one], dim=1).unsqueeze(-1),
        ],
        dim=2,
    )  # (b, 1) => (b, 3, 3)
    # (b, 3, 3), (b, 3) => (b, 3)
    data = roll.bmm(data.unsqueeze(-1)).squeeze(-1)
    return data


def zup_to_yup(p):
    """transform the coordinate system from z-up to y-up. It is identical to rotate 90 degree counter clock-wise along x-axis."""

    if isinstance(p, torch.Tensor):
        x = p[..., 0]
        y = p[..., 2]
        z = -p[..., 1]
        p_new = torch.stack([x, y, z], dim=-1)
    elif isinstance(p, np.ndarray):
        x = p[..., 0]
        y = p[..., 2]
        z = -p[..., 1]
        p_new = np.stack([x, y, z], axis=-1)
    else:
        raise TypeError(
            "Unsupported data type. Only numpy ndarray and torch Tensor are supported."
        )
    return p_new


def equirect_to_longlat(p):
    if isinstance(p, torch.Tensor):
        long = p[..., 0] * math.pi
        lat = p[..., 1] * math.pi / 2
    elif isinstance(p, np.ndarray):
        long = p[..., 0] * math.pi
        lat = p[..., 1] * math.pi / 2
    else:
        raise TypeError(
            "Unsupported data type. Only numpy ndarray and torch Tensor are supported."
        )
    return long, lat


def longlat_to_point(long, lat):
    if isinstance(long, torch.Tensor):
        x = torch.cos(lat) * torch.cos(long)
        y = torch.cos(lat) * torch.sin(long)
        z = torch.sin(lat)
        p = torch.stack([x, y, z], dim=-1)
    elif isinstance(long, np.ndarray):
        x = np.cos(lat) * np.cos(long)
        y = np.cos(lat) * np.sin(long)
        z = np.sin(lat)
        p = np.stack([x, y, z], axis=-1)
    else:
        raise TypeError(
            "Unsupported data type. Only numpy ndarray and torch Tensor are supported."
        )
    return p


def equirect_to_spherical(equi):
    """Equirectangular 2d points to 3D points on a unit sphere. Fov 180 degree of equirectangular images is assumed. 3D points are based on y-up coordinate system.

    Refer to http://paulbourke.net/dome/dualfish2sphere/diagram.pdf

    Args:
        equi (torch.Tensor or np.ndarray): equirectangular 2d points, ranging from -1 to 1.

    Returns:
        points: 3d points on a unit sphere, ranging from -1 to 1.
    """

    x = equi[..., 0]
    y = equi[..., 1]

    # normalize x from (-1, 1) to (-1, 0). Range of x should be (-1, 0) where the equirectangular image's fov=180, (-1, 1) where fov=360. Fov 180 is assumed in our case.
    x = (x - 1) / 2

    # x should be inverted because longitude starts from right-most to left-most.
    x = -x

    if isinstance(equi, torch.Tensor):
        equi = torch.stack([x, y], dim=-1)
    elif isinstance(equi, np.ndarray):
        equi = np.stack([x, y], axis=-1)
    else:
        raise TypeError(
            "Unsupported data type. Only numpy ndarray and torch Tensor are supported."
        )

    long, lat = equirect_to_longlat(equi)
    p = longlat_to_point(long, lat)

    # convert to y-up because 3d points are based on y-up coordinate system.
    p = zup_to_yup(p)

    return p


def get_camera_params_equirect(uv, camera_pos, camera_rotate):
    _, num_samples, _ = uv.shape
    direc_cam = equirect_to_spherical(uv)

    camera_pitch = camera_rotate[..., 0]
    camera_yaw = camera_rotate[..., 1]
    camera_roll = camera_rotate[..., 2]

    pitch = camera_pitch.unsqueeze(-1)
    yaw = camera_yaw.unsqueeze(-1)
    roll = camera_roll.unsqueeze(-1)

    direc_cam = direc_cam.unsqueeze(-1)

    R_world_to_camera = euler_angles_to_matrix(
        torch.cat([roll, pitch, yaw], dim=-1), convention="ZXY"
    )
    R_world_to_camera = R_world_to_camera.unsqueeze(1).repeat(1, num_samples, 1, 1)
    direc_world = torch.einsum("bnij,bnjk->bnik", R_world_to_camera, direc_cam).squeeze(
        -1
    )

    return direc_world, camera_pos


def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:, :4])
        p = torch.eye(4).repeat(pose.shape[0], 1, 1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else:  # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc


def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (
        (
            x
            - cx.unsqueeze(-1)
            + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
            - sk.unsqueeze(-1) * y / fy.unsqueeze(-1)
        )
        / fx.unsqueeze(-1)
        * z
    )
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)


def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3)).cuda()
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi**2 + qj**2)
    return R


def rot_to_quat(R):
    batch_size, _, _ = R.shape
    q = torch.ones((batch_size, 4)).cuda()

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:, 0] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
    q[:, 1] = (R21 - R12) / (4 * q[:, 0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


# def get_sphere_intersections(cam_loc, ray_directions, r=1.0):
#     # Input: n_rays x 3 ; n_rays x 3
#     # Output: n_rays x 1, n_rays x 1 (close and far)

#     ray_cam_dot = torch.bmm(
#         ray_directions.view(-1, 1, 3), cam_loc.view(-1, 3, 1)
#     ).squeeze(-1)
#     under_sqrt = ray_cam_dot**2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r**2)

#     # sanity check
#     if (under_sqrt <= 0).sum() > 0:
#         print("BOUNDING SPHERE PROBLEM!")
#         exit()

#     sphere_intersections = (
#         torch.sqrt(under_sqrt) * torch.Tensor([-1, 1]).cuda().float() - ray_cam_dot
#     )
#     sphere_intersections = sphere_intersections.clamp_min(0.0)

#     return sphere_intersections


def get_sphere_intersections(cam_loc, ray_directions, r=1.0):
    # Input: n_rays x 3 ; n_rays x 3
    # Output: n_rays x 1, n_rays x 1 (close and far)

    cam_loc = einops.rearrange(cam_loc, "b n p -> b n p 1")
    ray_directions = einops.rearrange(ray_directions, "b n p -> b n 1 p")

    ray_cam_dot = torch.einsum("bnxp, bnpx -> bnx", ray_directions, cam_loc)
    under_sqrt = ray_cam_dot**2 - (cam_loc.norm(2, -2) ** 2 - r**2)

    # sanity check
    if (under_sqrt <= 0).sum() > 0:
        print("BOUNDING SPHERE PROBLEM!")
        exit()

    sphere_intersections = (
        torch.sqrt(under_sqrt) * torch.Tensor([-1, 1]).to(under_sqrt.device)
        - ray_cam_dot
    )
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections


def bilinear_interpolation(xs, ys, dist_map):
    x1 = np.floor(xs).astype(np.int32)
    y1 = np.floor(ys).astype(np.int32)
    x2 = x1 + 1
    y2 = y1 + 1

    dx = np.expand_dims(np.stack([x2 - xs, xs - x1], axis=1), axis=1)
    dy = np.expand_dims(np.stack([y2 - ys, ys - y1], axis=1), axis=2)
    Q = np.stack(
        [dist_map[x1, y1], dist_map[x1, y2], dist_map[x2, y1], dist_map[x2, y2]], axis=1
    ).reshape(-1, 2, 2)
    return np.squeeze(dx @ Q @ dy)  # ((x2 - x1) * (y2 - y1)) = 1


def get_index_outside_of_bbox(samples_uniform, bbox_min, bbox_max):
    samples_uniform_row = samples_uniform[:, 0]
    samples_uniform_col = samples_uniform[:, 1]
    index_outside = np.where(
        (samples_uniform_row < bbox_min[0])
        | (samples_uniform_row > bbox_max[0])
        | (samples_uniform_col < bbox_min[1])
        | (samples_uniform_col > bbox_max[1])
    )[0]
    return index_outside


def weighted_sampling(data, img_size, num_sample, bbox_ratio=0.9):
    """
    More sampling within the bounding box
    """

    # calculate bounding box
    mask = data["object_mask"]
    where = np.asarray(np.where(mask))
    bbox_min = where.min(axis=1)
    bbox_max = where.max(axis=1)

    num_sample_bbox = int(num_sample * bbox_ratio)
    # samples_bbox_indices = np.random.choice(
    #     list(range(where.shape[1])), size=num_sample_bbox, replace=False
    # )
    # samples_bbox = where[:, samples_bbox_indices].transpose()
    samples_bbox = np.random.rand(num_sample_bbox, 2)
    samples_bbox = samples_bbox * (bbox_max - bbox_min) + bbox_min

    num_sample_uniform = num_sample - num_sample_bbox
    samples_uniform = np.random.rand(num_sample_uniform, 2)
    samples_uniform *= (img_size[0] - 1, img_size[1] - 1)

    # get indices for uniform samples outside of bbox
    index_outside = (
        get_index_outside_of_bbox(samples_uniform, bbox_min, bbox_max) + num_sample_bbox
    )

    indices = np.concatenate([samples_bbox, samples_uniform], axis=0)
    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            # new_val = np.stack(
            #     [
            #         bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
            #         for i in range(val.shape[2])
            #     ],
            #     axis=-1,
            # )

            new_val = cv2.remap(
                src=val.astype(np.float32),
                map1=(indices[:, 1]).astype(np.float32),
                map2=(indices[:, 0]).astype(np.float32),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            # new_val = np.where(new_val > 0.0, 1.0, 0.0)
            new_val = val[
                indices.astype(np.int32)[:, 0], indices.astype(np.int32)[:, 1]
            ].astype(np.float32)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val

    return output, index_outside


def frequency_encoding(x, L=10):
    encoding = []
    for i in range(L):
        encoding.append(torch.sin(math.pi * x * 2**i))
    for i in range(L):
        encoding.append(torch.cos(math.pi * x * 2**i))
    encoding = torch.cat(encoding, dim=-1)
    return encoding
