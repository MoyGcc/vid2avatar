import torch


class PointInSpace:
    def __init__(self, global_sigma=0.5, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    def get_points(self, pc_input=None, local_sigma=None, global_ratio=0.125):
        """Sample one point near each of the given point + 1/8 uniformly. 
        Args:
            pc_input (tensor): sampling centers. shape: [B, N, D]
        Returns:
            samples (tensor): sampled points. shape: [B, N + N / 8, D]
        """

        batch_size, sample_size, dim = pc_input.shape
        if local_sigma is None:
            sample_local = pc_input + (torch.randn_like(pc_input) * self.local_sigma)
        else:
            sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma)
        sample_global = (
            torch.rand(batch_size, int(sample_size * global_ratio), dim, device=pc_input.device)
            * (self.global_sigma * 2)
        ) - self.global_sigma

        sample = torch.cat([sample_local, sample_global], dim=1)

        return sample