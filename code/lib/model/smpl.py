import torch
import hydra
import numpy as np
from ..smpl.body_models import SMPL

class SMPLServer(torch.nn.Module):

    def __init__(self, gender='neutral', betas=None, v_template=None):
        super().__init__()


        self.smpl = SMPL(model_path=hydra.utils.to_absolute_path('lib/smpl/smpl_model'),
                         gender=gender,
                         batch_size=1,
                         use_hands=False,
                         use_feet_keypoints=False,
                         dtype=torch.float32).cuda()

        self.bone_parents = self.smpl.bone_parents.astype(int)
        self.bone_parents[0] = -1
        self.bone_ids = []
        self.faces = self.smpl.faces
        for i in range(24): self.bone_ids.append([self.bone_parents[i], i])

        if v_template is not None:
            self.v_template = torch.tensor(v_template).float().cuda()
        else:
            self.v_template = None

        if betas is not None:
            self.betas = torch.tensor(betas).float().cuda()
        else:
            self.betas = None

        # define the canonical pose
        param_canonical = torch.zeros((1, 86),dtype=torch.float32).cuda()
        param_canonical[0, 0] = 1
        param_canonical[0, 9] = np.pi / 6
        param_canonical[0, 12] = -np.pi / 6
        if self.betas is not None and self.v_template is None:
            param_canonical[0,-10:] = self.betas
        self.param_canonical = param_canonical

        output = self.forward(*torch.split(self.param_canonical, [1, 3, 72, 10], dim=1), absolute=True)
        self.verts_c = output['smpl_verts']
        self.joints_c = output['smpl_jnts']
        self.tfs_c_inv = output['smpl_tfs'].squeeze(0).inverse()


    def forward(self, scale, transl, thetas, betas, absolute=False):
        """return SMPL output from params
        Args:
            scale : scale factor. shape: [B, 1]
            transl: translation. shape: [B, 3]
            thetas: pose. shape: [B, 72]
            betas: shape. shape: [B, 10]
            absolute (bool): if true return smpl_tfs wrt thetas=0. else wrt thetas=thetas_canonical. 
        Returns:
            smpl_verts: vertices. shape: [B, 6893. 3]
            smpl_tfs: bone transformations. shape: [B, 24, 4, 4]
            smpl_jnts: joint positions. shape: [B, 25, 3]
        """

        output = {}

        # ignore betas if v_template is provided
        if self.v_template is not None:
            betas = torch.zeros_like(betas)

        
        smpl_output = self.smpl.forward(betas=betas,
                                        transl=torch.zeros_like(transl),
                                        body_pose=thetas[:, 3:],
                                        global_orient=thetas[:, :3],
                                        return_verts=True,
                                        return_full_pose=True,
                                        v_template=self.v_template)

        verts = smpl_output.vertices.clone()
        output['smpl_verts'] = verts * scale.unsqueeze(1) + transl.unsqueeze(1) * scale.unsqueeze(1)

        joints = smpl_output.joints.clone()
        output['smpl_jnts'] = joints * scale.unsqueeze(1) + transl.unsqueeze(1) * scale.unsqueeze(1)

        tf_mats = smpl_output.T.clone()
        tf_mats[:, :, :3, :] = tf_mats[:, :, :3, :] * scale.unsqueeze(1).unsqueeze(1)
        tf_mats[:, :, :3, 3] = tf_mats[:, :, :3, 3] + transl.unsqueeze(1) * scale.unsqueeze(1)

        if not absolute:
            tf_mats = torch.einsum('bnij,njk->bnik', tf_mats, self.tfs_c_inv)
        
        output['smpl_tfs'] = tf_mats
        output['smpl_weights'] = smpl_output.weights
        return output