import numpy as np
import pickle as pkl
import torch
import trimesh
import cv2
import os
from tqdm import tqdm
import glob
import argparse
from preprocessing_utils import (smpl_to_pose, PerspectiveCamera, Renderer, render_trimesh, \
                                estimate_translation_cv2, transform_smpl)
from loss import joints_2d_loss, pose_temporal_loss, get_loss_weights

def main(args):
    device = torch.device("cuda:0")
    seq = args.seq
    gender = args.gender
    DIR = './raw_data'
    img_dir = f'{DIR}/{seq}/frames'   
    romp_file_dir = f'{DIR}/{seq}/ROMP'
    img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
    romp_file_paths = sorted(glob.glob(f"{romp_file_dir}/*.npz"))

    from smplx import SMPL
    smpl_model = SMPL('../code/lib/smpl/smpl_model', gender=gender).to(device)
    
    input_img = cv2.imread(img_paths[0])
    if args.source == 'custom':
        focal_length = max(input_img.shape[0], input_img.shape[1])
        cam_intrinsics = np.array([[focal_length, 0., input_img.shape[1]//2],
                                   [0., focal_length, input_img.shape[0]//2],
                                   [0., 0., 1.]])
    elif args.source == 'neuman':
        NeuMan_DIR = '' # path to NeuMan dataset
        with open(f'{NeuMan_DIR}/{seq}/sparse/cameras.txt') as f:
            lines = f.readlines()
        cam_params = lines[3].split()
        cam_intrinsics = np.array([[float(cam_params[4]), 0., float(cam_params[6])], 
                                   [0., float(cam_params[5]), float(cam_params[7])], 
                                   [0., 0., 1.]])
    elif args.source == 'deepcap':
        DeepCap_DIR = '' # path to DeepCap dataset
        with open(f'{DeepCap_DIR}/monocularCalibrationBM.calibration') as f:
            lines = f.readlines()

        cam_params = lines[5].split()
        cam_intrinsics = np.array([[float(cam_params[1]), 0., float(cam_params[3])], 
                                   [0., float(cam_params[6]), float(cam_params[7])], 
                                   [0., 0., 1.]])
    else:
        print('Please specify the source of the dataset (custom, neuman, deepcap). We will continue to update the sources in the future.')
        raise NotImplementedError
    renderer = Renderer(img_size = [input_img.shape[0], input_img.shape[1]], cam_intrinsic=cam_intrinsics)

    if args.mode == 'mask':
        if not os.path.exists(f'{DIR}/{seq}/init_mask'):
            os.makedirs(f'{DIR}/{seq}/init_mask')
    elif args.mode == 'refine':
        if not os.path.exists(f'{DIR}/{seq}/init_refined_smpl'):
            os.makedirs(f'{DIR}/{seq}/init_refined_smpl')
        if not os.path.exists(f'{DIR}/{seq}/init_refined_mask'):
            os.makedirs(f'{DIR}/{seq}/init_refined_mask')
        if not os.path.exists(f'{DIR}/{seq}/init_refined_smpl_files'):
            os.makedirs(f'{DIR}/{seq}/init_refined_smpl_files')
        openpose_dir = f'{DIR}/{seq}/openpose'
        openpose_paths = sorted(glob.glob(f"{openpose_dir}/*.npy"))
        opt_num_iters=150
        weight_dict = get_loss_weights()
        cam = PerspectiveCamera(focal_length_x=torch.tensor(cam_intrinsics[0, 0], dtype=torch.float32),
                                focal_length_y=torch.tensor(cam_intrinsics[1, 1], dtype=torch.float32),
                                center=torch.tensor(cam_intrinsics[0:2, 2]).unsqueeze(0)).to(device)
        mean_shape = []
        smpl2op_mapping = torch.tensor(smpl_to_pose(model_type='smpl', use_hands=False, use_face=False,
                                            use_face_contour=False, openpose_format='coco25'), dtype=torch.long).cuda()
    elif args.mode == 'final':
        refined_smpl_dir = f'{DIR}/{seq}/init_refined_smpl_files'
        refined_smpl_mask_dir = f'{DIR}/{seq}/init_refined_mask'
        refined_smpl_paths = sorted(glob.glob(f"{refined_smpl_dir}/*.pkl"))
        refined_smpl_mask_paths = sorted(glob.glob(f"{refined_smpl_mask_dir}/*.png"))

        save_dir = f'../data/{seq}'
        if not os.path.exists(os.path.join(save_dir, 'image')):
            os.makedirs(os.path.join(save_dir, 'image'))
        if not os.path.exists(os.path.join(save_dir, 'mask')):
            os.makedirs(os.path.join(save_dir, 'mask'))

        scale_factor = args.scale_factor
        smpl_shape = np.load(f'{DIR}/{seq}/mean_shape.npy')
        T_hip = smpl_model.get_T_hip(betas=torch.tensor(smpl_shape)[None].float().to(device)).squeeze().cpu().numpy()

        K = np.eye(4)
        K[:3, :3] = cam_intrinsics
        K[0, 0] = K[0, 0] / scale_factor
        K[1, 1] = K[1, 1] / scale_factor
        K[0, 2] = K[0, 2] / scale_factor
        K[1, 2] = K[1, 2] / scale_factor

        dial_kernel = np.ones((20, 20),np.uint8)

        output_trans = []
        output_pose = []
        output_P = {}

    last_j3d = None
    actor_id = 0
    cam_extrinsics = np.eye(4)
    R = torch.tensor(cam_extrinsics[:3,:3])[None].float()
    T = torch.tensor(cam_extrinsics[:3, 3])[None].float() 
    for idx, img_path in enumerate(tqdm(img_paths)):
        input_img = cv2.imread(img_path)
        if args.mode == 'mask' or args.mode == 'refine':
            seq_file = np.load(romp_file_paths[idx], allow_pickle=True)['results'][()]
            
            # tracking in case of two persons or wrong ROMP detection
            if len(seq_file['smpl_thetas']) >= 2:
                dist = []
                if idx == 0:
                    last_j3d = seq_file['joints'][actor_id]
                for i in range(len(seq_file['smpl_thetas'])):
                    dist.append(np.linalg.norm(seq_file['joints'][i].mean(0) - last_j3d.mean(0, keepdims=True)))
                actor_id = np.argmin(dist)
            smpl_verts = seq_file['verts'][actor_id]
            pj2d_org = seq_file['pj2d_org'][actor_id]
            joints3d = seq_file['joints'][actor_id]
            last_j3d = joints3d.copy()
            tra_pred = estimate_translation_cv2(joints3d, pj2d_org, proj_mat=cam_intrinsics)
            
            smpl_verts += tra_pred

            if args.mode == 'refine':
                openpose = np.load(openpose_paths[idx])
                openpose_j2d = torch.tensor(openpose[:, :2][None], dtype=torch.float32, requires_grad=False, device=device)
                openpose_conf = torch.tensor(openpose[:, -1][None], dtype=torch.float32, requires_grad=False, device=device)

                smpl_shape = seq_file['smpl_betas'][actor_id][:10]
                smpl_pose = seq_file['smpl_thetas'][actor_id]
                smpl_trans = tra_pred

                opt_betas = torch.tensor(smpl_shape[None], dtype=torch.float32, requires_grad=True, device=device)
                opt_pose = torch.tensor(smpl_pose[None], dtype=torch.float32, requires_grad=True, device=device)
                opt_trans = torch.tensor(smpl_trans[None], dtype=torch.float32, requires_grad=True, device=device)

                opt_params = [{'params': opt_betas, 'lr': 1e-3},
                            {'params': opt_pose, 'lr': 1e-3},
                            {'params': opt_trans, 'lr': 1e-3}]
                optimizer = torch.optim.Adam(opt_params, lr=2e-3, betas=(0.9, 0.999))
                if idx == 0:
                    last_pose = [opt_pose.detach().clone()]
                loop = tqdm(range(opt_num_iters))
                for it in loop:
                    optimizer.zero_grad()

                    smpl_output = smpl_model(betas=opt_betas,
                                             body_pose=opt_pose[:,3:],
                                             global_orient=opt_pose[:,:3],
                                             transl=opt_trans)
                    smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

                    smpl_joints_2d = cam(torch.index_select(smpl_output.joints, 1, smpl2op_mapping))
                    
                    loss = dict()
                    loss['J2D_Loss'] = joints_2d_loss(openpose_j2d, smpl_joints_2d, openpose_conf)
                    loss['Temporal_Loss'] = pose_temporal_loss(last_pose[0], opt_pose)
                    w_loss = dict()
                    for k in loss:
                        w_loss[k] = weight_dict[k](loss[k], it)
                    
                    tot_loss = list(w_loss.values())
                    tot_loss = torch.stack(tot_loss).sum()
                    tot_loss.backward()
                    optimizer.step()

                    l_str = 'Iter: %d' % it
                    for k in loss:
                        l_str += ', %s: %0.4f' % (k, weight_dict[k](loss[k], it).mean().item())
                        loop.set_description(l_str)

            smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
            R = torch.tensor(cam_extrinsics[:3,:3])[None].float()
            T = torch.tensor(cam_extrinsics[:3, 3])[None].float() 
            rendered_image = render_trimesh(renderer, smpl_mesh, R, T, 'n')
            if input_img.shape[0] < input_img.shape[1]:
                rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...] 
            else:
                rendered_image = rendered_image[:,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2]   
            valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]
            
            if args.mode == 'mask':
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_mask', '%04d.png' % idx), valid_mask*255)
            elif args.mode == 'refine':
                output_img = (rendered_image[:,:,:-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_refined_smpl', '%04d.png' % idx), output_img)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_refined_mask', '%04d.png' % idx), valid_mask*255)
                last_pose.pop(0)
                last_pose.append(opt_pose.detach().clone())
                smpl_dict = {}
                smpl_dict['pose'] = opt_pose.data.squeeze().cpu().numpy()
                smpl_dict['trans'] = opt_trans.data.squeeze().cpu().numpy()
                smpl_dict['shape'] = opt_betas.data.squeeze().cpu().numpy()

                mean_shape.append(smpl_dict['shape'])
                pkl.dump(smpl_dict, open(os.path.join(f'{DIR}/{seq}/init_refined_smpl_files', '%04d.pkl' % idx), 'wb'))
        elif args.mode == 'final':
            input_img = cv2.resize(input_img, (input_img.shape[1] // scale_factor, input_img.shape[0] // scale_factor))
            seq_file = pkl.load(open(refined_smpl_paths[idx], 'rb'))

            mask = cv2.imread(refined_smpl_mask_paths[idx])
            mask = cv2.resize(mask, (mask.shape[1] // scale_factor, mask.shape[0] // scale_factor))

            # dilate mask to obtain a coarse bbox
            mask = cv2.dilate(mask, dial_kernel)

            cv2.imwrite(os.path.join(save_dir, 'image/%04d.png' % idx), input_img)
            cv2.imwrite(os.path.join(save_dir, 'mask/%04d.png' % idx), mask)

            smpl_pose = seq_file['pose']
            smpl_trans = seq_file['trans']

            # transform the spaces such that our camera has the same orientation as the OpenGL camera
            target_extrinsic = np.eye(4)
            target_extrinsic[1:3] *= -1
            target_extrinsic, smpl_pose, smpl_trans = transform_smpl(cam_extrinsics, target_extrinsic, smpl_pose, smpl_trans, T_hip)
            smpl_output = smpl_model(betas=torch.tensor(smpl_shape)[None].float().to(device),
                                     body_pose=torch.tensor(smpl_pose[3:])[None].float().to(device),
                                     global_orient=torch.tensor(smpl_pose[:3])[None].float().to(device),
                                     transl=torch.tensor(smpl_trans)[None].float().to(device))
            smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

            # we need to center the human for every frame due to the potentially large global movement
            v_max = smpl_verts.max(axis=0)
            v_min = smpl_verts.min(axis=0)
            normalize_shift = -(v_max + v_min) / 2.

            trans = smpl_trans + normalize_shift
            
            target_extrinsic[:3, -1] = target_extrinsic[:3, -1] - (target_extrinsic[:3, :3] @ normalize_shift)

            P = K @ target_extrinsic
            output_trans.append(trans)
            output_pose.append(smpl_pose)
            output_P[f"cam_{idx}"] = P

    if args.mode == 'refine':
        mean_shape = np.array(mean_shape)
        np.save(f'{DIR}/{seq}/mean_shape.npy', mean_shape.mean(0))
    if args.mode == 'final':
        np.save(os.path.join(save_dir, 'poses.npy'), np.array(output_pose))
        np.save(os.path.join(save_dir, 'mean_shape.npy'), smpl_shape)
        np.save(os.path.join(save_dir, 'normalize_trans.npy'), np.array(output_trans))
        np.savez(os.path.join(save_dir, "cameras.npz"), **output_P)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing data")
    # video source
    parser.add_argument('--source', type=str, default='custom', help="custom video or dataset video")
    # sequence name
    parser.add_argument('--seq', type=str)
    # gender
    parser.add_argument('--gender', type=str, help="gender of the actor: MALE or FEMALE")
    # mode
    parser.add_argument('--mode', type=str, help="mask mode or refine mode: mask or refine or final")
    # scale factor for the input image
    parser.add_argument('--scale_factor', type=int, default=2, help="scale factor for the input image")
    args = parser.parse_args()
    main(args)