from preprocessing_utils import GMoF
import torch
num_joints = 25
joints_to_ign = [1,9,12]
joint_weights = torch.ones(num_joints)   
joint_weights[joints_to_ign] = 0
joint_weights = joint_weights.reshape((-1,1)).cuda()

robustifier = GMoF(rho=100)

def get_loss_weights():
    loss_weight = {'J2D_Loss': lambda cst, it: 1e-2 * cst,
                   'Temporal_Loss': lambda cst, it: 6e0 * cst,
                  }
    return loss_weight

def joints_2d_loss(gt_joints_2d=None, joints_2d=None, joint_confidence=None):

    joint_diff = robustifier(gt_joints_2d - joints_2d)
    joints_2dloss = torch.mean((joint_confidence*joint_weights[:, 0]).unsqueeze(-1) ** 2 * joint_diff)
    return joints_2dloss

def pose_temporal_loss(last_pose, param_pose):
    temporal_loss = torch.mean(torch.square(last_pose - param_pose))
    return temporal_loss