import os
import glob
import hydra
import cv2
import numpy as np
import torch
from lib.utils import utils
from scipy.spatial.transform import Rotation as R

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def flip_y_2d_points(points, y_range=[-1.0, 1.0]):
    """Flip along y values of 2d points array.

    Args:
        points (torch.Tensor or np.ndarray): N-array 2d points, size of last dimension is 2.
        y_range (list, optional): Min/max of y values. Defaults to [-1.0, 1.0].

    Raises:
        TypeError: Only numpy ndarray and torch Tensor are supported.

    Returns:
        new_points: flipped points.
    """
    y_min = y_range[0]
    y_max = y_range[1]

    if isinstance(points, torch.Tensor):
        x = points[..., 0]
        y = points[..., 1]
        new_y = -y + y_min + y_max
        new_points = torch.stack([x, new_y], dim=-1)
    elif isinstance(points, np.ndarray):
        x = points[..., 0]
        y = points[..., 1]
        new_y = -y + y_min + y_max
        new_points = np.stack([x, new_y], axis=-1)
    else:
        raise TypeError(
            "Unsupported data type. Only numpy ndarray and torch Tensor are supported."
        )
    return new_points


def get_right_camera_pos(camera_poses, camera_rotates):
    right_shift = np.array([0.06, 0, 0])

    right_shifts = []
    for camera_rotate in camera_rotates:
        camera_pitch = camera_rotate[..., 0]  # (1,1)
        camera_yaw = camera_rotate[..., 1]  # (1,1)
        camera_roll = camera_rotate[..., 2]  # (1,1)

        R_world_to_camera = R.from_euler(
            "zxy", (camera_roll, camera_pitch, camera_yaw), degrees=False
        ).as_matrix()

        right_shift_world = np.matmul(R_world_to_camera, right_shift)
        right_shifts.append(right_shift_world)
    right_shifts = np.stack(right_shifts)
    return camera_poses + right_shifts


class Dataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        root = os.path.join(".", metainfo.data_dir)
        root = hydra.utils.to_absolute_path(root)

        self.start_frame = metainfo.start_frame
        self.end_frame = metainfo.end_frame
        self.skip_step = 1

        # images
        img_left_dir = os.path.join(root, "image", "left")
        img_right_dir = os.path.join(root, "image", "right")

        img_left_paths = sorted(glob.glob(f"{img_left_dir}/*.png"))
        img_right_paths = sorted(glob.glob(f"{img_right_dir}/*.png"))

        assert len(img_left_paths) == len(img_right_paths)

        self.img_paths = sorted(glob.glob(f"{img_left_dir}/*.png")) + sorted(
            glob.glob(f"{img_right_dir}/*.png")
        )

        # only store the image paths to avoid OOM
        self.img_size = tuple(metainfo.img_size)
        self.n_images = len(self.img_paths)

        # coarse projected SMPL masks, only for sampling
        mask_left_dir = os.path.join(root, "mask", "left")
        mask_right_dir = os.path.join(root, "mask", "right")
        self.mask_paths = sorted(glob.glob(f"{mask_left_dir}/*.png")) + sorted(
            glob.glob(f"{mask_right_dir}/*.png")
        )

        # self.shape = np.load(os.path.join(root, "mean_shape.npy"))
        # self.poses = np.load(os.path.join(root, "poses.npy"))
        trans = np.load(os.path.join(root, "normalize_trans.npy"))

        # cameras
        camera_poses = np.load(os.path.join(root, "camera_pos.npy"))
        camera_rotates = np.load(os.path.join(root, "camera_rotate.npy"))

        assert camera_poses.shape[0] == camera_rotates.shape[0] == len(
            img_left_paths)
        self.indices = np.concatenate(
            [np.arange(camera_poses.shape[0]), np.arange(camera_poses.shape[0])], axis=0
        )
        camera_poses_right = get_right_camera_pos(camera_poses, camera_rotates)

        camera_poses = np.concatenate(
            [camera_poses, camera_poses_right], axis=0)
        camera_rotates = np.concatenate(
            [camera_rotates, camera_rotates], axis=0)

        max_distance_from_camera_to_artist = np.linalg.norm(
            np.concatenate([trans, trans], axis=0).squeeze(1) - camera_poses, axis=-1
        ).max()

        scene_bounding_sphere = 3.0
        # self.scale = 1 / scale_mats[0][0, 0]
        self.scale = 1 / (
            max_distance_from_camera_to_artist * 1.1 / scene_bounding_sphere
        )

        self.camera_poses = torch.tensor(camera_poses, dtype=torch.float32)
        self.camera_rotates = torch.tensor(camera_rotates, dtype=torch.float32)

        # normalize
        # self.trans = self.trans * self.scale
        self.camera_poses = self.camera_poses * self.scale

        # other properties
        self.num_sample = split.num_sample
        self.sampling_strategy = "weighted"

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # normalize RGB
        img = cv2.imread(self.img_paths[idx])

        # preprocess: BGR -> RGB -> Normalize

        img = img[:, :, ::-1] / 255

        # img = utils.read_image(self.img_paths[idx])
        # img = utils.clip_and_convert_rgb_to_srgb(img)

        mask = cv2.imread(self.mask_paths[idx])
        # preprocess: BGR -> Gray -> Mask
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # dilate_kernel = np.ones((20, 20), np.uint8)
        # mask_for_sampling = cv2.dilate(mask, dilate_kernel)

        mask = mask > 0
        mask_for_sampling = mask > 0
        # mask = mask / 255.0

        img_size = self.img_size

        x = np.linspace(-0.5, 0.5, img_size[0], endpoint=False)
        y = np.linspace(-0.5, 0.5, img_size[1], endpoint=False)
        uv = np.stack(np.meshgrid(x, y, indexing="xy"), axis=-1)  # (h, w, 2)
        uv = flip_y_2d_points(uv, y_range=[-0.5, 0.5])

        # smpl_params = torch.zeros([86]).float()
        # smpl_params[0] = torch.from_numpy(np.asarray(self.scale)).float()

        # smpl_params[1:4] = torch.from_numpy(self.trans[idx]).float()
        # smpl_params[4:76] = torch.from_numpy(self.poses[idx]).reshape(-1).float()
        # smpl_params[76:] = torch.from_numpy(self.shape).float()

        if self.num_sample > 0:
            data = {
                "rgb": img,
                "uv": uv,
                "object_mask": mask_for_sampling,
                "mask": mask,
            }

            samples, index_outside = utils.weighted_sampling(
                data, img_size, self.num_sample
            )
            inputs = {
                "uv": samples["uv"].astype(np.float32),
                # "intrinsics": self.intrinsics_all[idx],
                # "pose": self.pose_all[idx],
                "camera_poses": self.camera_poses[idx],
                "camera_rotates": self.camera_rotates[idx],
                # "smpl_params": smpl_params,
                # "index_outside": index_outside,
                "idx": self.indices[idx],
                "scale": self.scale,
            }
            images = {
                "rgb": samples["rgb"].astype(np.float32),
                "mask": samples["mask"].astype(np.float32),
            }
            return inputs, images
        else:
            inputs = {
                "uv": uv.reshape(-1, 2).astype(np.float32),
                # "intrinsics": self.intrinsics_all[idx],
                # "pose": self.pose_all[idx],
                "camera_poses": self.camera_poses[idx],
                "camera_rotates": self.camera_rotates[idx],
                # "smpl_params": smpl_params,
                "idx": self.indices[idx],
                "scale": self.scale,
            }
            images = {
                "rgb": img.reshape(-1, 3).astype(np.float32),
                "img_size": self.img_size,
                "mask": mask.reshape(-1).astype(np.float32),
            }
            return inputs, images


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = Dataset(metainfo, split)
        self.pixel_per_batch = split.pixel_per_batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        self.data = self.dataset[idx]
        inputs, images = self.data

        inputs = {
            "uv": inputs["uv"],
            "camera_poses": inputs["camera_poses"],
            "camera_rotates": inputs["camera_rotates"],
            # "intrinsics": inputs["intrinsics"],
            # "pose": inputs["pose"],
            # "smpl_params": inputs["smpl_params"],
            "idx": idx,
            "scale": inputs["scale"],
        }
        images = {
            "rgb": images["rgb"],
            "img_size": images["img_size"],
            "pixel_per_batch": self.pixel_per_batch,
            "mask": images["mask"],
        }
        return inputs, images


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = Dataset(metainfo, split)
        if split.output_img_size:
            self.output_img_size = tuple(split.output_img_size)
        else:
            self.output_img_size = self.dataset.img_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        x = np.linspace(-0.5, 0.5, self.output_img_size[0], endpoint=False)
        y = np.linspace(-0.5, 0.5, self.output_img_size[1], endpoint=False)
        uv = np.meshgrid(x, y, indexing="xy")  # (2, h, w)
        u = uv[0] * (self.dataset.img_size[0] / self.output_img_size[0])
        v = uv[1] * (self.dataset.img_size[1] / self.output_img_size[1])
        uv = np.stack([u, v], axis=-1)  # (h, w, 2)
        uv = flip_y_2d_points(uv, y_range=[-0.5, 0.5])
        uv = uv.reshape(-1, 2).astype(np.float32)

        inputs, images = data
        data = {
            "rgb": images["rgb"],
            "mask": images["mask"],
            "uv": uv,
            "camera_poses": inputs["camera_poses"],
            "camera_rotates": inputs["camera_rotates"],
            # "smpl_params": inputs["smpl_params"],
            "idx": idx,
            "scale": inputs["scale"],
        }
        return data
