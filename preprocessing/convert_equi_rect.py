import fire
import numpy as np
import cv2 as cv
import math
import glob


def convert_equi_rectlinear(input_path, output_path, width=3840, height=3840, fov=90):
    imgs = glob.glob("{}/*.png".format(input_path))

    for i, img in enumerate(imgs):
        img = cv.imread(img)
        img_L = img[:, :, :3]

        x = np.linspace(-1, 1, width, endpoint=False)
        y = np.linspace(-1, 1, height, endpoint=False)
        idx = np.stack(np.meshgrid(x, y), axis=-1)

        # equi to rect
        idx_equi = get_rect2equi_mapping(idx, height, width, fov)
        img_L = convert_image(img_L, idx_equi)

        # rect to equi
        # idx_rect = get_equi2rect_mapping(idx, height, width, fov)
        # img_L = convert_image(img_L, idx_rect)

        cv.imwrite("{}/output_{}.png".format(output_path, i), img_L)


def get_rect2equi_mapping(idx_rect, height, width, fov):
    """
    idx_rect: (-1 ~ 1, -1 ~ 1)
    idx_equi: (-1 ~ 1, -1 ~ 1)
    """

    # denormalize (-1, 1) => (0, height or width)
    idx_rect = denormalize_points(idx_rect, height, width)

    # Calculate focal length based on the desired rectilinear image dimensions and FOV
    focal_length = (width / 2) / math.tan(degree_to_radian(fov / 2))

    # Convert rectilinear coordinates to normalized device coordinates
    x = (idx_rect[..., 0] - width / 2) / focal_length
    y = (idx_rect[..., 1] - height / 2) / focal_length

    # Convert normalized device coordinates to spherical coordinates
    long = np.arctan2(y, x)
    lat = np.arctan2(1.0, np.sqrt(x**2 + y**2))

    # # spherical coordinates to equirectangular coordinates.
    x = 1 + (2 * long / math.pi)
    y = 2 * lat / math.pi

    idx_equi = np.stack([x, y], axis=-1)

    return idx_equi


def convert_image(input, idx):
    """
    idx: (-1 ~ 1, -1 ~ 1)
    """
    height, width, _ = input.shape

    xmap = idx[..., 0]
    ymap = idx[..., 1]

    # denormalize (-1, 1) => (0, height or width)
    xmap = (xmap + 1) / 2 * height
    ymap = (ymap + 1) / 2 * width

    xmap = xmap.astype(np.float32)
    ymap = ymap.astype(np.float32)

    output = cv.remap(src=input, map1=xmap, map2=ymap, interpolation=cv.INTER_CUBIC)
    return output


def denormalize_points(points, height, width):
    # (-1~1,-1~1) => (0~width, 0~height)
    x = (points[..., 0] + 1) / 2 * width
    y = (points[..., 1] + 1) / 2 * height
    points = np.stack([x, y], axis=-1)
    return points


def get_equi2rect_mapping(idx_equi, height, width, fov):
    """
    idx_equi: (-1 ~ 1, -1 ~ 1)
    idx_rect: (-1 ~ 1, -1 ~ 1)
    """

    # denormalize (-1, 1) => (0, height or width)
    idx_equi = denormalize_points(idx_equi, height, width)

    # Calculate focal length based on the desired rectilinear image dimensions and FOV
    focal_length = (width / 2) / math.tan(degree_to_radian(fov / 2))

    # equirectangular coordinates to spherical coordinates.
    x = (idx_equi[..., 0] - width / 2) / (width / 2)
    y = (idx_equi[..., 1] - height / 2) / (height / 2)
    long = (x - 1) * math.pi / 2
    lat = y * math.pi / 2

    # shperical coordinates to normalized device coordinates.
    x = np.cos(lat) * np.cos(long)
    y = np.cos(lat) * np.sin(long)
    z = np.sin(lat)
    x = x / (z + 1e-8)
    y = y / (z + 1e-8)

    x = focal_length * x / (width / 2)
    y = focal_length * y / (height / 2)

    idx_rect = np.stack([x, y], axis=-1)

    return idx_rect


def degree_to_radian(degree):
    return degree * math.pi / 180


if __name__ == "__main__":
    fire.Fire(convert_equi_rectlinear)

# import cv2
# import numpy as np


# def equirectangular_to_rectilinear(
#     equirect_img, rectilinear_width, rectilinear_height, fov
# ):
#     height, width = equirect_img.shape[:2]

#     rectilinear_img = np.zeros(
#         (rectilinear_height, rectilinear_width, 3), dtype=np.uint8
#     )

#     fov_rad = np.radians(fov)  # FOV를 라디안으로 변환
#     focal_length = rectilinear_width / (2 * np.tan(fov_rad / 2))
#     center_x = rectilinear_width / 2
#     center_y = rectilinear_height / 2

#     for y in range(rectilinear_height):
#         for x in range(rectilinear_width):
#             theta = np.pi * (y / rectilinear_height - 0.5)
#             phi = 2 * np.pi * (x / rectilinear_width - 0.5)
#             x_equirect = int(width * (phi / (2 * np.pi)))
#             y_equirect = int(height * (theta / np.pi + 0.5))
#             rectilinear_img[y, x] = equirect_img[y_equirect, x_equirect]

#     return rectilinear_img


# # Equirectangular 이미지 로드
# equirect_img = cv2.imread(
#     "/media/AI/Common/Recon3D/v2a/data/test_preprocess/image/EOT_s01_t2_denoise_v1.13607.png"
# )

# # 변환할 Rectilinear 이미지의 크기 설정
# rectilinear_width = 3840
# rectilinear_height = 3840

# # 시야각(FOV) 설정
# fov = 90  # 90도 FOV

# # Equirectangular 이미지를 Rectilinear 이미지로 변환
# rectilinear_img = equirectangular_to_rectilinear(
#     equirect_img, rectilinear_width, rectilinear_height, fov
# )

# # Rectilinear 이미지 저장
# cv2.imwrite(
#     "/media/AI/Common/Recon3D/v2a/data/test_preprocess/image_rect/EOT_s01_t2_denoise_v1.13607.png",
#     rectilinear_img,
# )
