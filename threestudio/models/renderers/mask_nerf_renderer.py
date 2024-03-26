from dataclasses import dataclass, field
from functools import partial

import nerfacc
import threestudio
import torch
import torch.nn.functional as F
from threestudio.models.background.base import BaseBackground
from threestudio.models.estimators import ImportanceEstimator
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import create_network_with_input_encoding
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.systems.utils import parse_optimizer, parse_scheduler_to_instance
from threestudio.utils.ops import chunk_batch, get_activation, validate_empty_rays
from threestudio.utils.typing import *


@threestudio.register("mask-nerf-volume-renderer")
class StableNeRFVolumeRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        eval_chunk_size: int = 160000
        randomized: bool = True

        near_plane: float = 0.0
        far_plane: float = 1e10

        return_comp_normal: bool = False
        return_normal_perturb: bool = False

        # in ["occgrid", "proposal", "importance"]
        estimator: str = "occgrid"

        # for occgrid
        grid_prune: bool = True
        prune_alpha_threshold: bool = True

        # for proposal
        proposal_network_config: Optional[dict] = None
        prop_optimizer_config: Optional[dict] = None
        prop_scheduler_config: Optional[dict] = None
        num_samples_per_ray_proposal: int = 64

        # for importance
        num_samples_per_ray_importance: int = 64

        # for memory
        train_max_nums: int = 20000000
        train_max_nums_static: int = 6000000

        grid_prune_update: bool = True
        occ_grid_res: Optional[int] = 32
        eval_normal: Optional[bool] = False
        occ_n: Optional[int] = 16 #1
        occ_ema_decay: Optional[float] = 0.95
        occ_thre: Optional[float] = 0.01
        occ_frame_updates: Optional[bool] = False
        occ_ema_decay_init_zero: Optional[bool] = False
        occ_thre_post_init: Optional[float] = None

        scene_single_obj: bool = False
        dil_kernel_size: Optional[int] = 0
        er_kernel_size: Optional[int] = 0
        dil_iters: Optional[int] = 1
        dil_occ_frame_updates: Optional[bool] = False

        use_bg_image: bool = False
        bg_image_path: str = "/path/to/inference/bg/image"
        bg_image_height: int = 256
        bg_image_width: int = 256

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        if self.cfg.estimator == "occgrid":
            self.estimator = nerfacc.OccGridEstimator(
                roi_aabb=self.bbox.view(-1), resolution=self.cfg.occ_grid_res, levels=1
            )
            if not self.cfg.grid_prune:
                self.estimator.occs.fill_(True)
                self.estimator.binaries.fill_(True)
            self.render_step_size = (
                1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
            )
            self.randomized = self.cfg.randomized
        elif self.cfg.estimator == "importance":
            self.estimator = ImportanceEstimator()
        elif self.cfg.estimator == "proposal":
            self.prop_net = create_network_with_input_encoding(
                **self.cfg.proposal_network_config
            )
            self.prop_optim = parse_optimizer(
                self.cfg.prop_optimizer_config, self.prop_net
            )
            self.prop_scheduler = (
                parse_scheduler_to_instance(
                    self.cfg.prop_scheduler_config, self.prop_optim
                )
                if self.cfg.prop_scheduler_config is not None
                else None
            )
            self.estimator = nerfacc.PropNetEstimator(
                self.prop_optim, self.prop_scheduler
            )

            def get_proposal_requires_grad_fn(
                target: float = 5.0, num_steps: int = 1000
            ):
                schedule = lambda s: min(s / num_steps, 1.0) * target

                steps_since_last_grad = 0

                def proposal_requires_grad_fn(step: int) -> bool:
                    nonlocal steps_since_last_grad
                    target_steps_since_last_grad = schedule(step)
                    requires_grad = steps_since_last_grad > target_steps_since_last_grad
                    if requires_grad:
                        steps_since_last_grad = 0
                    steps_since_last_grad += 1
                    return requires_grad

                return proposal_requires_grad_fn

            self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()
            self.randomized = self.cfg.randomized
        else:
            raise NotImplementedError(
                "Unknown estimator, should be one of ['occgrid', 'proposal', 'importance']."
            )

        # for proposal
        self.vars_in_forward = {}

        self.dil_kernel = self.cfg.dil_kernel_size != 0
        self.er_kernel = self.cfg.er_kernel_size != 0
        self.dil_kernel_bool = True
        if self.dil_kernel:
            self.register_buffer('kernel_dil', torch.ones((1, 1, *(self.cfg.dil_kernel_size,)*3)))
        if self.er_kernel:
            self.register_buffer('kernel_er', torch.ones((1, 1, *(self.cfg.er_kernel_size,)*3)))


    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        proxy_center: Optional[Tensor] = None,
        proxy_rotation_mat: Optional[Tensor] = None,
        out_raw: Optional[bool] = False,
        sampling_dict: Dict[str, Tensor] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        if sampling_dict is None:
            batch_size, height, width = rays_o.shape[:3]
            rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
            rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
            light_positions_flatten: Float[Tensor, "Nr 3"] = (
                light_positions.reshape(-1, 1, 1, 3)
                .expand(-1, height, width, -1)
                .reshape(-1, 3)
            )
            n_rays = rays_o_flatten.shape[0]

            if self.cfg.estimator == "occgrid":
                if not self.cfg.grid_prune:
                    with torch.no_grad():
                        ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                            rays_o_flatten,
                            rays_d_flatten,
                            sigma_fn=None,
                            near_plane=self.cfg.near_plane,
                            far_plane=self.cfg.far_plane,
                            render_step_size=self.render_step_size,
                            alpha_thre=0.0,
                            stratified=self.randomized,
                            cone_angle=0.0,
                            early_stop_eps=0,
                        )
                else:

                    def sigma_fn(t_starts, t_ends, ray_indices):
                        t_starts, t_ends = t_starts[..., None], t_ends[..., None]
                        t_origins = rays_o_flatten[ray_indices]
                        t_positions = (t_starts + t_ends) / 2.0
                        t_dirs = rays_d_flatten[ray_indices]
                        positions = t_origins + t_dirs * t_positions
                        sigma = chunk_batch(
                            self.geometry.forward_density,
                            self.cfg.eval_chunk_size,
                            positions,
                        )[..., 0]
                        return sigma

                    with torch.no_grad():
                        ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                            rays_o_flatten,
                            rays_d_flatten,
                            sigma_fn=sigma_fn if self.cfg.prune_alpha_threshold else None,
                            near_plane=self.cfg.near_plane,
                            far_plane=self.cfg.far_plane,
                            render_step_size=self.render_step_size,
                            alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                            stratified=self.randomized,
                            cone_angle=0.0,
                        )
            elif self.cfg.estimator == "proposal":

                def prop_sigma_fn(
                    t_starts: Float[Tensor, "Nr Ns"],
                    t_ends: Float[Tensor, "Nr Ns"],
                    proposal_network,
                ):
                    t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                    t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                    positions: Float[Tensor, "Nr Ns 3"] = (
                        t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                    )
                    aabb_min, aabb_max = self.bbox[0], self.bbox[1]
                    positions = (positions - aabb_min) / (aabb_max - aabb_min)
                    selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
                    density_before_activation = (
                        proposal_network(positions.view(-1, 3))
                        .view(*positions.shape[:-1], 1)
                        .to(positions)
                    )
                    density: Float[Tensor, "Nr Ns 1"] = (
                        get_activation("shifted_trunc_exp")(density_before_activation)
                        * selector[..., None]
                    )
                    return density.squeeze(-1)

                t_starts_, t_ends_ = self.estimator.sampling(
                    prop_sigma_fns=[partial(prop_sigma_fn, proposal_network=self.prop_net)],
                    prop_samples=[self.cfg.num_samples_per_ray_proposal],
                    num_samples=self.cfg.num_samples_per_ray,
                    n_rays=n_rays,
                    near_plane=self.cfg.near_plane,
                    far_plane=self.cfg.far_plane,
                    sampling_type="uniform",
                    stratified=self.randomized,
                    requires_grad=self.vars_in_forward["requires_grad"],
                )
                ray_indices = (
                    torch.arange(n_rays, device=rays_o_flatten.device)
                    .unsqueeze(-1)
                    .expand(-1, t_starts_.shape[1])
                )
                ray_indices = ray_indices.flatten()
                t_starts_ = t_starts_.flatten()
                t_ends_ = t_ends_.flatten()
            elif self.cfg.estimator == "importance":

                def prop_sigma_fn(
                    t_starts: Float[Tensor, "Nr Ns"],
                    t_ends: Float[Tensor, "Nr Ns"],
                    proposal_network,
                ):
                    t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                    t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                    positions: Float[Tensor, "Nr Ns 3"] = (
                        t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                    )
                    with torch.no_grad():
                        geo_out = chunk_batch(
                            proposal_network,
                            self.cfg.eval_chunk_size,
                            positions.reshape(-1, 3),
                            output_normal=False,
                        )
                        density = geo_out["density"]
                    return density.reshape(positions.shape[:2])

                t_starts_, t_ends_ = self.estimator.sampling(
                    prop_sigma_fns=[partial(prop_sigma_fn, proposal_network=self.geometry)],
                    prop_samples=[self.cfg.num_samples_per_ray_importance],
                    num_samples=self.cfg.num_samples_per_ray,
                    n_rays=n_rays,
                    near_plane=self.cfg.near_plane,
                    far_plane=self.cfg.far_plane,
                    sampling_type="uniform",
                    stratified=self.randomized,
                )
                ray_indices = (
                    torch.arange(n_rays, device=rays_o_flatten.device)
                    .unsqueeze(-1)
                    .expand(-1, t_starts_.shape[1])
                )
                ray_indices = ray_indices.flatten()
                t_starts_ = t_starts_.flatten()
                t_ends_ = t_ends_.flatten()
            else:
                raise NotImplementedError

            ray_indices, t_starts_, t_ends_ = validate_empty_rays(
                ray_indices, t_starts_, t_ends_
            )
            ray_indices = ray_indices.long()
            t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
            t_origins = rays_o_flatten[ray_indices]
            t_dirs = rays_d_flatten[ray_indices]
            t_light_positions = light_positions_flatten[ray_indices]
            t_positions = (t_starts + t_ends) / 2.0
            positions = t_origins + t_dirs * t_positions
            t_intervals = t_ends - t_starts
        
        else:
            rays_d_flatten = sampling_dict["rays_d_flatten"]
            ray_indices = sampling_dict["ray_indices"]
            t_starts = sampling_dict["t_starts"]
            t_ends = sampling_dict["t_ends"]
            t_dirs = sampling_dict["t_dirs"]
            t_light_positions = sampling_dict["t_light_positions"]
            t_positions = sampling_dict["t_positions"]
            positions = sampling_dict["positions"]
            t_intervals = sampling_dict["t_intervals"]

        # Translate and rotate
        if proxy_center is not None:
            positions = positions - proxy_center
        if proxy_rotation_mat is not None:
            positions = torch.matmul(positions, proxy_rotation_mat)
        positions_mask = torch.logical_or(positions < -1.0, positions > 1.0)
        positions = positions.clamp(-1.0, 1.0)
        MAX_N = self.cfg.train_max_nums
        QUERY_N = positions.shape[0]
        mask_positions = torch.randperm(QUERY_N, device=positions.device)
        mask_positions = mask_positions < MAX_N
        if self.training:
            if QUERY_N > MAX_N:
                with torch.no_grad():
                    geo_out_mask = chunk_batch(
                        self.geometry,
                        self.cfg.eval_chunk_size,
                        positions[~mask_positions],
                        output_normal=self.material.requires_normal or self.cfg.eval_normal,
                    )
                    rgb_fg_all_mask = chunk_batch(
                        self.material,
                        self.cfg.eval_chunk_size,
                        viewdirs=t_dirs[~mask_positions],
                        positions=positions[~mask_positions],
                        light_positions=t_light_positions[~mask_positions],
                        **geo_out_mask
                    )

                geo_out = self.geometry(
                    positions[mask_positions],
                    output_normal=self.material.requires_normal or self.cfg.eval_normal,
                )
                rgb_fg_all = self.material(
                    viewdirs=t_dirs[mask_positions],
                    positions=positions[mask_positions],
                    light_positions=t_light_positions[mask_positions],
                    **geo_out,
                    **kwargs
                )
                for key in geo_out:
                    if torch.is_tensor(geo_out[key]):
                        if geo_out[key].shape[0] == MAX_N:
                            new_out = torch.zeros(
                                QUERY_N,
                                *geo_out[key].shape[1:],
                                device=geo_out[key].device
                            )
                            new_out[mask_positions] = geo_out[key]
                            new_out[~mask_positions] = geo_out_mask[key]
                            geo_out[key] = new_out

                new_rgb_fg_all = torch.zeros(
                    QUERY_N, *rgb_fg_all.shape[1:], device=rgb_fg_all.device
                )
                new_rgb_fg_all[mask_positions] = rgb_fg_all
                new_rgb_fg_all[~mask_positions] = rgb_fg_all_mask
                rgb_fg_all = new_rgb_fg_all

                comp_rgb_bg = self.background(dirs=rays_d_flatten)
            else:
                geo_out = self.geometry(
                    positions, output_normal=self.material.requires_normal or self.cfg.eval_normal
                )
                rgb_fg_all = self.material(
                    viewdirs=t_dirs,
                    positions=positions,
                    light_positions=t_light_positions,
                    **geo_out,
                    **kwargs
                )
                comp_rgb_bg = self.background(dirs=rays_d_flatten)
        else:
            geo_out = chunk_batch(
                self.geometry,
                self.cfg.eval_chunk_size,
                positions,
                output_normal=self.material.requires_normal or self.cfg.eval_normal,
            )
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out
            )
            comp_rgb_bg = chunk_batch(
                self.background, self.cfg.eval_chunk_size, dirs=rays_d_flatten
            )
        if out_raw:
            out = {
                "rgb_fg_all": rgb_fg_all,
                "t_positions": t_positions,
                "density": geo_out["density"],
                "t_starts": t_starts,
                "t_ends": t_ends,
                "ray_indices": ray_indices,
                "comp_rgb_bg": comp_rgb_bg,
                "points": positions,
                "deformation": geo_out["dx"],
                "dx": geo_out["dx"],
                "elastic_loss": geo_out["elastic_loss"],
                "divergence_loss": geo_out["divergence_loss"],
                "dx_shifted": geo_out["dx_shifted"],
                "positions_mask": positions_mask,
                }
            if "normal" in geo_out:
                n_rays = rays_d_flatten.shape[0]
                batch_size = 1
                weights: Float[Tensor, "Nr 1"]
                weights_, trans_, _ = nerfacc.render_weight_from_density(
                    t_starts[..., 0],
                    t_ends[..., 0],
                    geo_out["density"][..., 0],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                if self.training and self.cfg.estimator == "proposal":
                    self.vars_in_forward["trans"] = trans_.reshape(n_rays, -1)

                weights = weights_[..., None]
                opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
                    weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
                )
                comp_normal = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    values=geo_out["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal = (comp_normal + 1.0) / 2.0 * opacity  # for visualization
                comp_normal = comp_normal.view(sampling_dict["batch_size"], sampling_dict["height"], sampling_dict["width"], 3)
            else:
                comp_normal = None
            out.update(
                {
                    "comp_normal": comp_normal,
                }
            )

            return out

        weights: Float[Tensor, "Nr 1"]
        weights_, trans_, _ = nerfacc.render_weight_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            geo_out["density"][..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        if self.training and self.cfg.estimator == "proposal":
            self.vars_in_forward["trans"] = trans_.reshape(n_rays, -1)

        weights = weights_[..., None]
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )
        if geo_out["dx"] is not None:
            deformation: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
                weights[..., 0], values=geo_out["dx"], ray_indices=ray_indices, n_rays=n_rays
            )
            deformation = deformation.view(batch_size, height, width, -1)
        else:
            deformation = None
        # populate depth and opacity to each point
        t_depth = depth[ray_indices]
        z_variance = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=(t_positions - t_depth) ** 2,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

        if bg_color is None:
            bg_color = comp_rgb_bg
        else:
            if bg_color.shape[:-1] == (batch_size,):
                # e.g. constant random color used for Zero123
                # [bs,3] -> [bs, 1, 1, 3]):
                bg_color = bg_color.unsqueeze(1).unsqueeze(1)
                #        -> [bs, height, width, 3]):
                bg_color = bg_color.expand(-1, height, width, -1)

        if bg_color.shape[:-1] == (batch_size, height, width):
            bg_color = bg_color.reshape(batch_size * height * width, -1)

        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)
        out = {
            "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
            "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
            "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
            "opacity": opacity.view(batch_size, height, width, 1),
            "depth": depth.view(batch_size, height, width, 1),
            "z_variance": z_variance.view(batch_size, height, width, 1),
            "deformation": deformation,
            "elastic_loss": geo_out["elastic_loss"],
            "divergence_loss": geo_out["divergence_loss"],
            "dx": geo_out["dx"],
            "dx_shifted": geo_out["dx_shifted"],
            "comp_normal": geo_out["normal"] if "normal" in geo_out and self.cfg.eval_normal else None
        }
        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "points": positions,
                    **geo_out,
                }
            )
            if "normal" in geo_out:
                if self.cfg.return_comp_normal:
                    comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights[..., 0],
                        values=geo_out["normal"],
                        ray_indices=ray_indices,
                        n_rays=n_rays,
                    )
                    comp_normal = F.normalize(comp_normal, dim=-1)
                    comp_normal = (
                        (comp_normal + 1.0) / 2.0 * opacity
                    )  # for visualization
                    out.update(
                        {
                            "comp_normal": comp_normal.view(
                                batch_size, height, width, 3
                            ),
                        }
                    )
                if self.cfg.return_normal_perturb:
                    normal_perturb = self.geometry(
                        positions + torch.randn_like(positions) * 1e-2,
                        output_normal=self.material.requires_normal,
                    )["normal"]
                    out.update({"normal_perturb": normal_perturb})
        else:
            if "normal" in geo_out:
                comp_normal = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    values=geo_out["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal = (comp_normal + 1.0) / 2.0 * opacity  # for visualization
                out.update(
                    {
                        "comp_normal": comp_normal.view(batch_size, height, width, 3),
                    }
                )
        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        if self.cfg.estimator == "occgrid":
            if self.dil_kernel and self.dil_kernel_bool and not self.cfg.dil_occ_frame_updates:
                estimator_occ_grid = self.estimator.binaries.float()
                if self.er_kernel:
                    estimator_occ_grid = 1 - torch.clamp(torch.nn.functional.conv3d(1-estimator_occ_grid.float(), self.kernel_er, padding='same'), 0, 1)
                    estimator_occ_grid = estimator_occ_grid > 0.5
                for _ in range(self.cfg.dil_iters):
                    estimator_occ_grid = torch.clamp(torch.nn.functional.conv3d(estimator_occ_grid.float(), self.kernel_dil, padding='same'), 0, 1)
                    estimator_occ_grid = estimator_occ_grid > 0.5
                self.estimator.binaries = estimator_occ_grid
                self.estimator.occs = estimator_occ_grid.reshape(self.estimator.occs.shape).float()
                self.dil_kernel_bool = False
            if self.cfg.grid_prune and self.cfg.grid_prune_update:

                def occ_eval_fn(x):
                    density = self.geometry.forward_density(x)
                    # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
                    return density * self.render_step_size
                if self.training and not on_load_weights:
                    encoding = self.geometry.encoding.encoding
                    if encoding.static or (not encoding.static and not self.cfg.occ_frame_updates):
                        self.estimator.update_every_n_steps(
                        step=global_step, occ_eval_fn=occ_eval_fn,
                        )
                    else:
                        if global_step % self.cfg.occ_n == 0:
                            num_frames = encoding.cfg["num_frames"]
                            t0 = torch.FloatTensor(1).uniform_(0, 1/num_frames).item()
                            frame_times = torch.linspace(t0, t0+(num_frames-1)/num_frames, num_frames)
                            frame_times = frame_times[torch.randperm(num_frames)]
                            occs_orig = self.estimator.occs.clone()
                            binaries_orig = self.estimator.binaries.clone()
                            occs = []
                            for frame_time in frame_times:
                                encoding.frame_time = frame_time.item()
                                self.estimator.update_every_n_steps(
                                    step=global_step, occ_eval_fn=occ_eval_fn, n=self.cfg.occ_n,
                                    ema_decay=0.0, occ_thre=self.cfg.occ_thre,
                                )
                                occs.append(self.estimator.occs.clone())
                                self.estimator.occs = occs_orig.clone()
                                self.estimator.binaries = binaries_orig.clone()
                            occs = torch.stack(occs).amax(0)
                            if self.cfg.occ_ema_decay_init_zero:
                                self.estimator.occs = occs
                                if self.cfg.occ_thre_post_init is not None:
                                    self.cfg.occ_thre = self.cfg.occ_thre_post_init
                                self.cfg.occ_ema_decay_init_zero = False
                            else:
                                self.estimator.occs = torch.maximum(occs_orig * self.cfg.occ_ema_decay, occs)
                            self.estimator.binaries = (
                                # self.estimator.occs > torch.clamp(self.estimator.occs.mean(), max=self.cfg.occ_thre)
                                self.estimator.occs > self.cfg.occ_thre
                            ).view(self.estimator.binaries.shape)
                            if self.cfg.dil_occ_frame_updates:
                                if self.dil_kernel:
                                    estimator_occ_grid = self.estimator.binaries
                                    if self.er_kernel:
                                        estimator_occ_grid = 1 - torch.clamp(torch.nn.functional.conv3d(1-estimator_occ_grid.float(), self.kernel_er, padding='same'), 0, 1)
                                        estimator_occ_grid = estimator_occ_grid > 0.5
                                    for _ in range(self.cfg.dil_iters):
                                        estimator_occ_grid = torch.clamp(torch.nn.functional.conv3d(estimator_occ_grid.float(), self.kernel_dil, padding='same'), 0, 1)
                                        estimator_occ_grid = estimator_occ_grid > 0.5
                                    self.estimator.binaries = estimator_occ_grid
        elif self.cfg.estimator == "proposal":
            if self.training:
                requires_grad = self.proposal_requires_grad_fn(global_step)
                self.vars_in_forward["requires_grad"] = requires_grad
            else:
                self.vars_in_forward["requires_grad"] = False

    def update_step_end(self, epoch: int, global_step: int) -> None:
        if self.cfg.estimator == "proposal" and self.training:
            self.estimator.update_every_n_steps(
                self.vars_in_forward["trans"],
                self.vars_in_forward["requires_grad"],
                loss_scaler=1.0,
            )

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        if self.cfg.estimator == "proposal":
            self.prop_net.train()
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        if self.cfg.estimator == "proposal":
            self.prop_net.eval()
        return super().eval()