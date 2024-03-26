from dataclasses import dataclass

import nerfacc
import threestudio
import torch
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.utils.ops import chunk_batch, load_resize_image
from threestudio.utils.typing import *
from threestudio.utils.object_trajectory import SceneTrajectory


@threestudio.register("mask-nerf-volume-renderer-multi")
class StableNeRFVolumeRendererMulti(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        eval_chunk_size: int = 160000
        randomized: bool = True

        near_plane: float = 0.0
        far_plane: float = 1e10

        return_comp_normal: bool = False
        return_normal_perturb: bool = False

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
        train_max_nums: int = 6000000
        train_max_nums_static: int = 6000000
        
        eval_normal: Optional[bool] = False
        grid_prune_update: bool = True
        scene_single_obj: bool = False
        occ_n: Optional[int] = 16 #1
        occ_ema_decay: Optional[float] = 0.95
        occ_ema_decay_init_zero: Optional[bool] = False
        occ_thre: Optional[float] = 0.01
        occ_frame_updates: Optional[bool] = False
        dil_kernel_size: Optional[int] = 0
        er_kernel_size: Optional[int] = 0
        dil_iters: Optional[int] = 0
        occ_thre_post_init: Optional[float] = None
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
        background_scene: BaseBackground,
        config_scene: Dict,
        renderer_type: str = "mask-nerf-volume-renderer",
        estimator_res: int = 32,
        adjust_offset: bool = False,
        adjust_offset_density: bool = False,
        adjust_offset_density_thres: float = 0.1,
    ) -> None:
        super().configure(None, None, background_scene)
        self.num_objs = len(config_scene.traj_kwargs)
        self.estimator_res = estimator_res
        self.estimators = torch.nn.ModuleList([nerfacc.OccGridEstimator(
            roi_aabb=self.bbox.view(-1), resolution=self.estimator_res, levels=1
        ) for _ in range(self.num_objs)]
        )
        self.render_step_size = (
            1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
        )
        self.randomized = self.cfg.randomized
        self.renderers = torch.nn.ModuleList([threestudio.find(renderer_type)(
            self.cfg,
            geometry=geometry[i],
            material=material[i],
            background=background[i],
        ) for i in range(self.num_objs)]
        )
        self.obj_trajs = SceneTrajectory(
                config_scene,
                estimators=self.estimators,
                renderers=self.renderers,
            )

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        render_scene: bool = True,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        self.update_object_nerfs()
        batch_size, height, width = rays_o.shape[:3]
        n_rays = rays_o.shape[:3].numel()
        batch_size, height, width = rays_o.shape[:3]
        rgb_fg_all_list = []
        t_positions_list = []
        density_list = []
        t_starts_list = []
        t_ends_list = []
        ray_indices_list = []
        points_list = []
        deformation_list = []
        dx_list = []
        dx_shifted_list = []
        elastic_loss_list = []
        divergence_loss_list = []
        proxy_center_list = []
        positions_mask_list = []
        comp_normal_list = []
        if render_scene:
            comp_rgb_bg = None
            rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
            rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
            light_positions_flatten: Float[Tensor, "Nr 3"] = (
                light_positions.reshape(-1, 1, 1, 3)
                .expand(-1, height, width, -1)
                .reshape(-1, 3)
            )
        else:
            comp_rgb_bg_list = []
        for obj_idx, (renderer, estimator) in enumerate(zip(self.renderers, self.estimators)):
            renderer.cfg.train_max_nums = self.cfg.train_max_nums
            if render_scene:
                if self.cfg.scene_single_obj and kwargs["nerf_to_render_idx"] is not None and kwargs["nerf_to_render_idx"] != obj_idx:
                    continue
                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        sigma_fn=None,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                        early_stop_eps=0,
                    )
                ray_indices = ray_indices.long()
                t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
                t_origins = rays_o_flatten[ray_indices]
                t_dirs = rays_d_flatten[ray_indices]
                t_light_positions = light_positions_flatten[ray_indices]
                t_positions = (t_starts + t_ends) / 2.0
                positions = t_origins + t_dirs * t_positions
                t_intervals = t_ends - t_starts
                sampling_dict = {}
                sampling_dict["rays_d_flatten"] = rays_d_flatten
                sampling_dict["ray_indices"] = ray_indices
                sampling_dict["t_starts"] = t_starts
                sampling_dict["t_ends"] = t_ends
                sampling_dict["t_dirs"] = t_dirs
                sampling_dict["t_light_positions"] = t_light_positions
                sampling_dict["t_positions"] = t_positions
                sampling_dict["positions"] = positions
                sampling_dict["t_intervals"] = t_intervals
                sampling_dict["batch_size"] = batch_size
                sampling_dict["height"] = height
                sampling_dict["width"] = width

                if self.training:
                    comp_rgb_bg = self.background(dirs=rays_d_flatten)
                else:
                    if self.cfg.use_bg_image:
                        bg_image = load_resize_image(self.cfg.bg_image_path, self.cfg.bg_image_height, self.cfg.bg_image_width)
                        bg_image = bg_image.permute(1, 2, 0).unsqueeze(0).expand(batch_size, -1, -1, -1)
                        comp_rgb_bg = bg_image.reshape(-1, 3).to(rays_d_flatten.device)
                    else:
                        comp_rgb_bg = chunk_batch(
                            self.background, self.cfg.eval_chunk_size, dirs=rays_d_flatten
                        )
            else:
                sampling_dict = None
            proxy_rotation_mat = self.obj_trajs.objs[obj_idx].proxy_rotation_mat
            proxy_size = self.obj_trajs.objs[obj_idx].proxy_size
            proxy_center = self.obj_trajs.objs[obj_idx].proxy_center
            proxy_z_offset = self.obj_trajs.objs[obj_idx].proxy_z_offset
            proxy_center = torch.tensor(proxy_center, device=rays_o.device)
            proxy_center[-1] = proxy_center[-1] + proxy_z_offset
            proxy_size = torch.tensor(proxy_size, device=proxy_center.device)
            proxy_center = proxy_center.clip(proxy_size/2, 1-proxy_size/2)
            proxy_center = proxy_center*2 - 1
            out_renderer = renderer(
                rays_o, rays_d, light_positions, bg_color, proxy_center=proxy_center,
                proxy_rotation_mat=proxy_rotation_mat,
                out_raw=True, sampling_dict=sampling_dict, **kwargs
                )
            rgb_fg_all_list.append(out_renderer["rgb_fg_all"])
            t_positions_list.append(out_renderer["t_positions"])
            density_list.append(out_renderer["density"])
            t_starts_list.append(out_renderer["t_starts"])
            t_ends_list.append(out_renderer["t_ends"])
            ray_indices_list.append(out_renderer["ray_indices"])
            if not render_scene:
                comp_rgb_bg_list.append(out_renderer["comp_rgb_bg"])
            points_list.append(out_renderer["points"])
            deformation_list.append(out_renderer["deformation"])
            elastic_loss_list.append(out_renderer["elastic_loss"])
            divergence_loss_list.append(out_renderer["divergence_loss"])
            dx_shifted_list.append(out_renderer["dx_shifted"])
            dx_list.append(out_renderer["dx"])
            proxy_center_list.append(proxy_center)
            positions_mask_list.append(out_renderer["positions_mask"])
            comp_normal_list.append(out_renderer["comp_normal"])

        rgb_fg_all = torch.cat(rgb_fg_all_list, 0)
        density = torch.cat(density_list, 0)
        t_starts = torch.cat(t_starts_list, 0)
        t_ends = torch.cat(t_ends_list, 0)
        ray_indices = torch.cat(ray_indices_list, 0)
        if not render_scene:
            comp_rgb_bg = torch.cat(comp_rgb_bg_list, 0)
        points = torch.cat(points_list, 0)
        t_positions = torch.cat(t_positions_list, 0)
        deformation = torch.cat(deformation_list, 0) if deformation_list[0] is not None else None
        elastic_loss = torch.cat(elastic_loss_list, 0) if elastic_loss_list[0] is not None else None
        divergence_loss = torch.cat(divergence_loss_list, 0) if divergence_loss_list[0] is not None else None
        dx_shifted = torch.cat(dx_shifted_list, 0) if dx_shifted_list[0] is not None else None
        dx = torch.cat(dx_list, 0) if dx_list[0] is not None else None
        positions_mask_list = torch.cat(positions_mask_list, 0) if positions_mask_list[0] is not None else None
        comp_normal_list = torch.cat(comp_normal_list, 0) if comp_normal_list[0] is not None else None
        proxy_center = torch.stack(proxy_center_list, 0).unsqueeze(0)

        if positions_mask_list is not None and positions_mask_list.any():
            indices = positions_mask_list.any(-1) == False
            t_starts = t_starts[indices]
            rgb_fg_all = rgb_fg_all[indices]
            density = density[indices]
            t_ends = t_ends[indices]
            points = points[indices]
            t_positions = t_positions[indices]
            ray_indices = ray_indices[indices]
            if deformation is not None:
                deformation = deformation[indices]
            if elastic_loss is not None:
                elastic_loss = elastic_loss[indices]
            if divergence_loss is not None:
                divergence_loss = divergence_loss[indices]
            if dx_shifted is not None:
                dx_shifted = dx_shifted[indices]
            if dx is not None:
                dx = dx[indices]

        # Sort rays across all objects (first in depth, then across rays) TODO: Only once sorting
        if len(self.renderers) > 1:
            _, indices = t_positions.sort(0)
            indices = indices.squeeze(-1)
            t_starts = t_starts[indices]
            rgb_fg_all = rgb_fg_all[indices]
            density = density[indices]
            t_ends = t_ends[indices]
            points = points[indices]
            t_positions = t_positions[indices]
            ray_indices = ray_indices[indices]
            if deformation is not None:
                deformation = deformation[indices]
            if elastic_loss is not None:
                elastic_loss = elastic_loss[indices]
            if divergence_loss is not None:
                divergence_loss = divergence_loss[indices]
            if dx_shifted is not None:
                dx_shifted = dx_shifted[indices]
            if dx is not None:
                dx = dx[indices]

            ray_indices, indices = ray_indices.sort()
            rgb_fg_all = rgb_fg_all[indices]
            density = density[indices]
            t_starts = t_starts[indices]
            t_ends = t_ends[indices]
            points = points[indices]
            t_positions = t_positions[indices]
            if deformation is not None:
                deformation = deformation[indices]
            if elastic_loss is not None:
                elastic_loss = elastic_loss[indices]
            if divergence_loss is not None:
                divergence_loss = divergence_loss[indices]
            if dx_shifted is not None:
                dx_shifted = dx_shifted[indices]
            if dx is not None:
                dx = dx[indices]
        weights: Float[Tensor, "Nr 1"]
        weights_, _, _ = nerfacc.render_weight_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            density[..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
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
        if deformation is not None:
            deformation: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
                weights[..., 0], values=deformation, ray_indices=ray_indices, n_rays=n_rays
            )
            deformation = deformation.view(batch_size, height, width, -1)

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
            if bg_color.shape == (batch_size, height, width, 3):
                bg_color = bg_color.reshape(-1, 3)
        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)
        out = {
            "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
            "opacity": opacity.view(batch_size, height, width, 1),
            "depth": depth.view(batch_size, height, width, 1),
            "z_variance": z_variance.view(batch_size, height, width, -1),
            "deformation": deformation,
            "elastic_loss": elastic_loss,
            "divergence_loss": divergence_loss,
            "dx_shifted": dx_shifted,
            "dx": dx,
            "proxy_center": proxy_center,
            "comp_normal": comp_normal_list,
        }
        if self.training:
            out.update(
                {
                    "weights": weights,
                }
            )
        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        if global_step == 0:
            self.update_object_nerfs(set_init=True)
        for renderer in self.renderers:
            renderer.update_step(epoch, global_step, on_load_weights)
    
    def update_object_nerfs(self, set_init: bool = False):
        if set_init:
            self.obj_trajs.update_objs([0.0 for _ in range(self.num_objs)])
        else:
            frame_times = [renderer.geometry.encoding.encoding.frame_time for renderer in self.renderers]
            static = all([renderer.geometry.encoding.encoding.static for renderer in self.renderers])
            if not static:
                self.obj_trajs.update_objs(frame_times)

    def train(self, mode=True):
        for renderer in self.renderers:
            renderer.randomized = mode and self.cfg.randomized
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        for renderer in self.renderers:
            renderer.randomized = False
        self.randomized = False
        return super().eval()
