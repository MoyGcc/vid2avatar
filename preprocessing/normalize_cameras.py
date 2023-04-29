# Adapted from https://github.com/lioryariv/volsdf/blob/main/DATA_CONVENTION.md
import cv2
import numpy as np
import argparse

def get_center_point(num_cams,cameras):
    A = np.zeros((3 * num_cams, 3 + num_cams))
    b = np.zeros((3 * num_cams, 1))
    camera_centers=np.zeros((3,num_cams))
    for i in range(num_cams):
        if 'cam_%d' % i in cameras:
            P0 = cameras['cam_%d' % i][:3, :] 
        else:
            P0 = cameras[i]
        K = cv2.decomposeProjectionMatrix(P0)[0]
        R = cv2.decomposeProjectionMatrix(P0)[1]
        c = cv2.decomposeProjectionMatrix(P0)[2]
        c = c / c[3]
        camera_centers[:,i]=c[:3].flatten()

        v=R[2,:]
        A[3 * i:(3 * i + 3), :3] = np.eye(3)
        A[3 * i:(3 * i + 3), 3 + i] = -v
        b[3 * i:(3 * i + 3)] = c[:3]

    return camera_centers

def normalize_cameras(original_cameras_filename,output_cameras_filename,num_of_cameras, scene_bounding_sphere=3.0):
    cameras = np.load(original_cameras_filename)
    if num_of_cameras==-1:
        all_files=cameras.files
        maximal_ind=0
        for field in all_files:
            maximal_ind=np.maximum(maximal_ind,int(field.split('_')[-1]))
        num_of_cameras=maximal_ind+1

    camera_centers = get_center_point(num_of_cameras, cameras)

    center = np.array([0, 0, 0])

    max_radius = np.linalg.norm((center[:, np.newaxis] - camera_centers), axis=0).max() * 1.1

    normalization = np.eye(4).astype(np.float32)

    normalization[0, 0] = max_radius / scene_bounding_sphere
    normalization[1, 1] = max_radius / scene_bounding_sphere
    normalization[2, 2] = max_radius / scene_bounding_sphere

    cameras_new = {}
    for i in range(num_of_cameras):
        cameras_new['scale_mat_%d' % i] = normalization
        cameras_new['world_mat_%d' % i] = cameras['cam_%d' % i].copy()

    np.savez(output_cameras_filename, **cameras_new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Normalizing cameras')
    parser.add_argument('--input_cameras_file', type=str, default="cameras.npz",
                        help='the input cameras file')
    parser.add_argument('--output_cameras_file', type=str, default="cameras_normalize.npz",
                        help='the output cameras file')
    parser.add_argument('--number_of_cams',type=int, default=-1,
                        help='Number of cameras, if -1 use all')

    args = parser.parse_args()
    normalize_cameras(args.input_cameras_file, args.output_cameras_file, args.number_of_cams)