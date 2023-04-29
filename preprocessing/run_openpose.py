import sys
import cv2
import os
import numpy as np
import argparse
import time
import glob
from sklearn.neighbors import NearestNeighbors
def get_bbox_center(img_path, mask_path):
    _img = cv2.imread(img_path)
    W, H = _img.shape[1], _img.shape[0]

    mask = cv2.imread(mask_path)[:, :, 0]
    where = np.asarray(np.where(mask))
    bbox_min = where.min(axis=1)
    bbox_max = where.max(axis=1)
    left, top, right, bottom = bbox_min[1], bbox_min[0], bbox_max[1], bbox_max[
        0]
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, W)
    bottom = min(bottom, H)
    bbox_center = np.array([left + (right - left) / 2, top + (bottom - top) / 2])
    return bbox_center

def main(args):
    try:
        sys.path.append(args.openpose_dir + '/build/python')
        # we use the python binding of openpose
        from openpose import pyopenpose as op
        DIR = './raw_data'
        # Flags
        params = dict()
        params['model_folder'] = args.openpose_dir + '/models/'
        params['scale_number'] = 1
        params['scale_gap'] = 0.25
        params['net_resolution'] = '720x480'

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Read frames on directory
        img_dir = f'{DIR}/{args.seq}/frames'
        imagePaths = op.get_images_on_directory(img_dir)
        maskPaths = sorted(glob.glob(f'{img_dir}/../init_mask/*.png'))
        start = time.time()

        if not os.path.exists(f'{img_dir}/../openpose'):
            os.makedirs(f'{img_dir}/../openpose')

        # Process and display images
        nbrs = NearestNeighbors(n_neighbors=1)
        for idx, imagePath in enumerate(imagePaths):
            datum = op.Datum()
            imageToProcess = cv2.imread(imagePath)
            maskPath = maskPaths[idx]
            bbox_center = get_bbox_center(imagePath, maskPath)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            poseKeypoints = datum.poseKeypoints

            nbrs.fit(poseKeypoints[:, 8, :2])

            actor = nbrs.kneighbors(bbox_center.reshape(1, -1), return_distance=False).ravel()[0]
            poseKeypoints = poseKeypoints[actor]
            np.save(f'{img_dir}/../openpose/%04d.npy' % idx, poseKeypoints)
            cv2.imwrite(f'{img_dir}/../openpose/%04d.png' % idx, datum.cvOutputData)
        end = time.time()
        print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
    except Exception as e:
        print(e)
        sys.exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run OpenPose on a sequence")
    # directory of openpose
    parser.add_argument('--openpose_dir', type=str, help="Directory of openpose")
    # sequence name
    parser.add_argument('--seq', type=str, help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_args()
    main(args)