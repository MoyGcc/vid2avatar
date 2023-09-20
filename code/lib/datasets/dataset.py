import os
import glob
import hydra
import cv2
import numpy as np
import torch
from lib.utils import utils

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        root = os.path.join("../data", metainfo.data_dir)
        root = hydra.utils.to_absolute_path(root)

        self.start_frame = metainfo.start_frame
        self.end_frame = metainfo.end_frame
        self.skip_step = 1
        self.training_indices = list(
            range(metainfo.start_frame, metainfo.end_frame, self.skip_step)
        )

        # images
        img_dir = os.path.join(root, "image")
        self.img_paths = sorted(glob.glob(f"{img_dir}/*.png"))

        # only store the image paths to avoid OOM
        self.img_paths = [self.img_paths[i] for i in self.training_indices]
        
        self.img_size = tuple(metainfo.img_size)

        self.n_images = len(self.img_paths)

        # coarse projected SMPL masks, only for sampling
        mask_dir = os.path.join(root, "mask")
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.mask_paths = [self.mask_paths[i] for i in self.training_indices]

        self.shape = np.load(os.path.join(root, "mean_shape.npy"))
        self.poses = np.zeros_like(
            np.load(os.path.join(root, "poses.npy"))[self.training_indices]
        )
        self.trans = np.zeros_like(
            np.load(os.path.join(root, "normalize_trans.npy"))[self.training_indices]
        )
        # cameras
        start_index = metainfo.start_index
        image_indices = [start_index + i * 3 for i in range(0, self.n_images)]

        self.camera_pos = utils.load_pos_init(metainfo.camera_pos_path, image_indices)
        self.camera_rot = utils.load_rotate_init(
            metainfo.camera_rotate_path, image_indices
        )

        camera_dict = np.load(os.path.join(root, "cameras_normalize.npz"))
        scale_mats = [
            camera_dict["scale_mat_%d" % idx].astype(np.float32)
            for idx in self.training_indices
        ]
        world_mats = [
            camera_dict["world_mat_%d" % idx].astype(np.float32)
            for idx in self.training_indices
        ]

        self.scale = 1 / scale_mats[0][0, 0]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = utils.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        assert len(self.intrinsics_all) == len(self.pose_all)

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
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 0

        uv = np.mgrid[: self.img_size[0], : self.img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = torch.from_numpy(np.asarray(self.scale)).float()

        smpl_params[1:4] = torch.from_numpy(self.trans[idx]).float()
        smpl_params[4:76] = torch.from_numpy(self.poses[idx]).float()
        smpl_params[76:] = torch.from_numpy(self.shape).float()

        if self.num_sample > 0:
            data = {
                "rgb": img,
                "uv": uv,
                "object_mask": mask,
            }

            samples, index_outside = utils.weighted_sampling(
                data, self.img_size, self.num_sample
            )
            inputs = {
                "uv": samples["uv"].astype(np.float32),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "camera_pos": self.camera_pos[idx],
                "camera_rot": self.camera_rot[idx],
                "smpl_params": smpl_params,
                "index_outside": index_outside,
                "idx": idx,
                "img_size": self.img_size,
            }
            images = {"rgb": samples["rgb"].astype(np.float32)}
            return inputs, images
        else:
            inputs = {
                "uv": uv.reshape(-1, 2).astype(np.float32),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "camera_pos": self.camera_pos[idx],
                "camera_rot": self.camera_rot[idx],
                "smpl_params": smpl_params,
                "idx": idx,
                "img_size": self.img_size,
            }
            images = {
                "rgb": img.reshape(-1, 3).astype(np.float32),
                "img_size": self.img_size,
            }
            return inputs, images


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = Dataset(metainfo, split)
        self.pixel_per_batch = split.pixel_per_batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image_id = int(np.random.choice(len(self.dataset), 1))
        self.data = self.dataset[image_id]
        inputs, images = self.data

        inputs = {
            "uv": inputs["uv"],
            "intrinsics": inputs["intrinsics"],
            "pose": inputs["pose"],
            "smpl_params": inputs["smpl_params"],
            "image_id": image_id,
            "idx": inputs["idx"],
        }
        images = {
            "rgb": images["rgb"],
            "img_size": images["img_size"],
            "pixel_per_batch": self.pixel_per_batch,
        }
        return inputs, images


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = Dataset(metainfo, split)
        self.pixel_per_batch = split.pixel_per_batch
        if split.output_img_size:
            self.output_img_size = tuple(split.output_img_size)
        else:
            self.output_img_size = self.dataset.img_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        uv = np.mgrid[: self.output_img_size[0], : self.output_img_size[1]].astype(np.int32)
        u = uv[0] * (self.dataset.img_size[0] / self.output_img_size[0])
        v = uv[1] * (self.dataset.img_size[1] / self.output_img_size[1])
        uv = np.stack([u, v], axis=0)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)
        uv = uv.reshape(-1, 2).astype(np.float32)

        inputs, images = data
        inputs = {
            "uv": uv,
            "intrinsics": inputs["intrinsics"],
            "pose": inputs["pose"],
            "smpl_params": inputs["smpl_params"],
            "idx": inputs["idx"],
        }
        images = {"rgb": images["rgb"], 
                "img_size": self.dataset.img_size,
                "output_img_size": self.output_img_size
                }
        return inputs, images, self.pixel_per_batch, idx
