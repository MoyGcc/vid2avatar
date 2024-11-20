import abc
import torch
from lib.utils import utils


class RaySampler(metaclass=abc.ABCMeta):
    def __init__(self, near, far):
        self.near = near
        self.far = far

    @abc.abstractmethod
    def get_z_vals(self, ray_dirs, cam_loc, model):
        pass


class UniformSampler(RaySampler):
    def __init__(
        self,
        scene_bounding_sphere,
        near,
        N_samples,
        take_sphere_intersection=False,
        far=-1,
    ):
        super().__init__(
            near, 2.0 * scene_bounding_sphere if far == -1 else far
        )  # default far is 2*R
        self.N_samples = N_samples
        self.scene_bounding_sphere = scene_bounding_sphere
        self.take_sphere_intersection = take_sphere_intersection

    def get_z_vals(self, ray_dirs, cam_loc, model):
        if not self.take_sphere_intersection:
            near, far = (
                self.near * torch.ones(ray_dirs.shape[0], ray_dirs.shape[1], 1).cuda(),
                self.far * torch.ones(ray_dirs.shape[0], ray_dirs.shape[1], 1).cuda(),
            )
        else:
            sphere_intersections = utils.get_sphere_intersections(
                cam_loc, ray_dirs, r=self.scene_bounding_sphere
            )
            near = (
                self.near * torch.ones(ray_dirs.shape[0], ray_dirs.shape[1], 1).cuda()
            )
            far = sphere_intersections[:, :, 1:]

        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples).cuda()
        z_vals = near * (1.0 - t_vals) + far * (t_vals)

        if model.training:
            # get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).cuda()

            z_vals = lower + (upper - lower) * t_rand

        return z_vals


class ErrorBoundSampler(RaySampler):
    def __init__(
        self,
        scene_bounding_sphere,
        near,
        N_samples,
        N_samples_eval,
        N_samples_extra,
        eps,
        beta_iters,
        max_total_iters,
        inverse_sphere_bg=False,
        N_samples_inverse_sphere=0,
        add_tiny=0.0,
    ):
        super().__init__(near, 2.0 * scene_bounding_sphere)
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.uniform_sampler = UniformSampler(
            scene_bounding_sphere,
            near,
            N_samples_eval,
            take_sphere_intersection=inverse_sphere_bg,
        )

        self.N_samples_extra = N_samples_extra

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny

        self.inverse_sphere_bg = inverse_sphere_bg
        if inverse_sphere_bg:
            N_samples_inverse_sphere = 32
            self.inverse_sphere_sampler = UniformSampler(
                1.0, 0.0, N_samples_inverse_sphere, False, far=1.0
            )

    def get_z_vals(self, ray_dirs, cam_loc, model, cond, smpl_output):
        smpl_tfs = smpl_output["smpl_tfs"]
        smpl_verts = smpl_output["smpl_verts"]
        smpl_weights = smpl_output["smpl_weights"]

        beta0 = model.density.get_beta().detach()

        # Start with uniform sampling
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        samples, samples_idx = z_vals, None

        # Get maximum beta from the upper bound (Lemma 2)
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (
            dists**2.0
        ).sum(-1)
        beta = torch.sqrt(bound)

        total_iters, not_converge = 0, True

        # VolSDF Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(-2) + samples.unsqueeze(-1) * ray_dirs.unsqueeze(
                -2
            )
            # points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            # points_flat = points.view(-1, 3)
            # Calculating the SDF only for the new sampled points
            model.implicit_network.eval()
            with torch.no_grad():
                samples_sdf, _, _ = model.sdf_func_with_smpl_deformer(
                    points,
                    cond,
                    smpl_tfs,
                    smpl_verts=smpl_verts,
                    smpl_weights=smpl_weights,
                )
            model.implicit_network.train()

            if samples_idx is not None:
                sdf_merge = torch.cat(
                    [
                        sdf.squeeze(-1).view(
                            z_vals.shape[0],
                            z_vals.shape[1],
                            z_vals.shape[2] - samples.shape[2],
                        ),
                        samples_sdf.squeeze(-1).view(
                            samples.shape[0], samples.shape[1], samples.shape[2]
                        ),
                    ],
                    -1,
                )
                sdf = (
                    torch.gather(sdf_merge, 2, samples_idx)
                    .view(sdf_merge.shape[0], -1)
                    .unsqueeze(-1)
                )
            else:
                sdf = samples_sdf

            # Calculating the bound d* (Theorem 1)
            d = sdf.squeeze(-1).view(z_vals.shape)
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            a, b, c = dists, d[..., :-1].abs(), d[..., 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros(
                z_vals.shape[0], z_vals.shape[1], z_vals.shape[2] - 1
            ).cuda()
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
            d_star = (
                d[..., 1:].sign() * d[..., :-1].sign() == 1
            ) * d_star  # Fixing the sign

            # Updating beta using line search
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.repeat(z_vals.shape[0], z_vals.shape[1]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.0
                curr_error = self.get_error_bound(
                    beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star
                )
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max

            # Upsample more points
            density = model.density(sdf.view(z_vals.shape), beta=beta.unsqueeze(-1))

            # TODO: need to be more clean
            # TODO: cuda -> to device
            dists = torch.cat(
                [
                    dists,
                    torch.tensor([1e10])
                    .to(dists.device)
                    .unsqueeze(0)
                    .repeat(dists.shape[0], dists.shape[1], 1),
                ],
                -1,
            )
            free_energy = dists * density
            shifted_free_energy = torch.cat(
                [
                    torch.zeros(dists.shape[0], dists.shape[1], 1).to(dists.device),
                    free_energy[..., :-1],
                ],
                dim=-1,
            )
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = (
                alpha * transmittance
            )  # probability of the ray hits something here

            # Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                """Sample more points proportional to the current error bound"""

                N = self.N_samples_eval

                bins = z_vals
                error_per_section = (
                    torch.exp(-d_star / beta.unsqueeze(-1))
                    * (dists[..., :-1] ** 2.0)
                    / (4 * beta.unsqueeze(-1) ** 2)
                )
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (
                    torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
                ) * transmittance[..., :-1]

                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

            else:
                """Sample the final sample set to be used in the volume rendering integral"""

                N = self.N_samples

                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat(
                    [torch.zeros_like(cdf[..., :1]), cdf], -1
                )  # (batch, len(bins))

            # Invert CDF
            if (not_converge and total_iters < self.max_total_iters) or (
                not model.training
            ):
                u = (
                    torch.linspace(0.0, 1.0, steps=N)
                    .to(cdf.device)
                    .unsqueeze(0)
                    .repeat(cdf.shape[0], cdf.shape[1], 1)
                )
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N]).cuda()
            # u = u.contiguous()

            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

            matched_shape = [
                inds_g.shape[0],
                inds_g.shape[1],
                inds_g.shape[2],
                cdf.shape[-1],
            ]
            cdf_g = torch.gather(cdf.unsqueeze(2).expand(matched_shape), -1, inds_g)
            bins_g = torch.gather(bins.unsqueeze(2).expand(matched_shape), -1, inds_g)

            denom = cdf_g[..., 1] - cdf_g[..., 0]
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

            # Adding samples if we not converged
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

        z_samples = samples

        near, far = (
            self.near * torch.ones(ray_dirs.shape[0], ray_dirs.shape[1], 1).cuda(),
            self.far * torch.ones(ray_dirs.shape[0], ray_dirs.shape[1], 1).cuda(),
        )
        if (
            self.inverse_sphere_bg
        ):  # if inverse sphere then need to add the far sphere intersection
            far = utils.get_sphere_intersections(
                cam_loc, ray_dirs, r=self.scene_bounding_sphere
            )[:, :, 1:]

        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[2])[: self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(
                    0, z_vals.shape[2] - 1, self.N_samples_extra
                ).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, :, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        # add some of the near surface points
        idx = torch.randint(
            z_vals.shape[-1],
            (
                z_vals.shape[0],
                z_vals.shape[1],
            ),
        ).cuda()
        z_samples_eik = torch.gather(z_vals, 2, idx.unsqueeze(-1))

        if self.inverse_sphere_bg:
            z_vals_inverse_sphere = self.inverse_sphere_sampler.get_z_vals(
                ray_dirs, cam_loc, model
            )
            z_vals_inverse_sphere = z_vals_inverse_sphere * (
                1.0 / self.scene_bounding_sphere
            )
            z_vals = (z_vals, z_vals_inverse_sphere)

        return z_vals, z_samples_eik

    def get_error_bound(self, beta, model, sdf, z_vals, dists, d_star):
        density = model.density(sdf.squeeze(-1).view(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat(
            [
                torch.zeros(dists.shape[0], dists.shape[1], 1).to(dists.device),
                dists * density[..., :-1],
            ],
            dim=-1,
        )
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists**2.0) / (4 * beta**2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (
            torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
        ) * torch.exp(-integral_estimation[..., :-1])

        return bound_opacity.max(-1)[0]
