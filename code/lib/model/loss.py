import torch
from torch import nn
from torch.nn import functional as F

class Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.eikonal_weight = opt.eikonal_weight
        self.bce_weight = opt.bce_weight
        self.opacity_sparse_weight = opt.opacity_sparse_weight
        self.in_shape_weight = opt.in_shape_weight
        self.eps = 1e-6
        self.milestone = 200
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
    
    # L1 reconstruction loss for RGB values
    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss
    
    # Eikonal loss introduced in IGR
    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1)**2).mean()
        return eikonal_loss

    # BCE loss for clear boundary
    def get_bce_loss(self, acc_map):
        binary_loss = -1 * (acc_map * (acc_map + self.eps).log() + (1-acc_map) * (1 - acc_map + self.eps).log()).mean() * 2
        return binary_loss

    # Global opacity sparseness regularization 
    def get_opacity_sparse_loss(self, acc_map, index_off_surface):
        opacity_sparse_loss = self.l1_loss(acc_map[index_off_surface], torch.zeros_like(acc_map[index_off_surface]))
        return opacity_sparse_loss

    # Optional: This loss helps to stablize the training in the very beginning
    def get_in_shape_loss(self, acc_map, index_in_surface):
        in_shape_loss = self.l1_loss(acc_map[index_in_surface], torch.ones_like(acc_map[index_in_surface]))
        return in_shape_loss

    def forward(self, model_outputs, ground_truth):
        nan_filter = ~torch.any(model_outputs['rgb_values'].isnan(), dim=1)
        rgb_gt = ground_truth['rgb'][0].cuda()
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'][nan_filter], rgb_gt[nan_filter])
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        bce_loss = self.get_bce_loss(model_outputs['acc_map'])
        opacity_sparse_loss = self.get_opacity_sparse_loss(model_outputs['acc_map'], model_outputs['index_off_surface'])
        in_shape_loss = self.get_in_shape_loss(model_outputs['acc_map'], model_outputs['index_in_surface'])
        curr_epoch_for_loss = min(self.milestone, model_outputs['epoch']) # will not increase after the milestone

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.bce_weight * bce_loss + \
               self.opacity_sparse_weight * (1 + curr_epoch_for_loss ** 2 / 40) * opacity_sparse_loss + \
               self.in_shape_weight * (1 - curr_epoch_for_loss / self.milestone) * in_shape_loss
        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'bce_loss': bce_loss,
            'opacity_sparse_loss': opacity_sparse_loss,
            'in_shape_loss': in_shape_loss,
        }