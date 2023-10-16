import torch
import torch.nn.functional as F
from .smpl import SMPLServer
from pytorch3d import ops
import einops


class SMPLDeformer:
    def __init__(self, smpl: SMPLServer, max_dist=0.1, K=1, betas=None):
        super().__init__()

        self.max_dist = max_dist
        self.K = K
        self.smpl = smpl
        smpl_params_canoical = self.smpl.param_canonical.clone()
        smpl_params_canoical[:, 76:] = (
            torch.tensor(betas).float().to(self.smpl.param_canonical.device)
        )
        cano_scale, cano_transl, cano_thetas, cano_betas = torch.split(
            smpl_params_canoical, [1, 3, 72, 10], dim=1
        )
        smpl_output = self.smpl(cano_scale, cano_transl,
                                cano_thetas, cano_betas)
        self.smpl_verts = smpl_output["smpl_verts"]
        self.smpl_weights = smpl_output["smpl_weights"]

    def forward(
        self,
        x,
        smpl_tfs,
        return_weights=True,
        inverse=False,
        smpl_verts=None,
        smpl_weights=None,
    ):
        if x.shape[0] == 0:
            return x
        batch_size = x.size(0)
        if smpl_verts is None or smpl_weights is None:
            weights, outlier_mask = self.query_skinning_weights_smpl_multi(
                x,
                smpl_verts=self.smpl_verts.repeat(batch_size, 1, 1),
                smpl_weights=self.smpl_weights.repeat(batch_size, 1, 1),
            )
        else:
            # TODO: check which smpl weights to use
            weights, outlier_mask = self.query_skinning_weights_smpl_multi(
                x, smpl_verts=smpl_verts, smpl_weights=smpl_weights
            )
        if return_weights:
            return weights

        x_transformed = skinning(x, weights, smpl_tfs, inverse=inverse)

        return x_transformed, outlier_mask

    def forward_skinning(self, xc, smpl_tfs):
        batch_size = xc.size(0)
        weights, _ = self.query_skinning_weights_smpl_multi(
            xc,
            smpl_verts=self.smpl_verts.repeat(batch_size, 1, 1),
            smpl_weights=self.smpl_weights.repeat(batch_size, 1, 1),
        )
        x_transformed = skinning(xc, weights, smpl_tfs, inverse=False)

        return x_transformed

    def query_skinning_weights_smpl_multi(self, pts, smpl_verts, smpl_weights):
        batch_size = pts.size(0)
        distance_batch, index_batch, neighbor_points = ops.knn_points(
            pts,
            smpl_verts,
            K=self.K,
            return_nn=True,
        )
        distance_batch = torch.clamp(distance_batch, max=4)
        weights_conf = torch.exp(-distance_batch)
        distance_batch = torch.sqrt(distance_batch)
        weights_conf = weights_conf / weights_conf.sum(-1, keepdim=True)

        weights = []
        for i in range(batch_size):
            weights.append(smpl_weights[i, index_batch[i]])
        weights = torch.stack(weights, dim=0)
        weights = torch.sum(
            weights * weights_conf.unsqueeze(-1), dim=-2).detach()

        outlier_mask = (distance_batch[..., 0] > self.max_dist).unsqueeze(-1)
        return weights, outlier_mask

    def query_weights(self, xc, smpl_weights):
        weights = self.forward(
            xc, None, return_weights=True, inverse=False, smpl_weights=smpl_weights
        )
        return weights


def skinning(x, w, tfs, inverse=False):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
        w_tf_inverse = w_tf.inverse()

        x_h = torch.einsum("bpij,bpj->bpi", w_tf_inverse, x_h)
    else:
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)
    return x_h[:, :, :3]
