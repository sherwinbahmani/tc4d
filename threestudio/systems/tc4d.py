from dataclasses import dataclass

import torch
import numpy as np
import copy

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot, TVLoss
from threestudio.utils.typing import *
from threestudio.utils.config_scene import load_config


@threestudio.register("tc4d-system")
class TC4D(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'geometry', 'texture']
        stage: str = "coarse"
        visualize_samples: bool = False
        multi_rate: Optional[int] = None
        multi_rate_perc: Optional[float] = None
        simultan_vid_mod: Optional[int] = None
        simultan_vid_mod_perc: Optional[float] = None
        num_objs: int = 1
        scene_single_obj: Optional[bool] = False
        scene_setup_path: Optional[str] = None
        multi_obj: Optional[bool] = True
        use_traj_length_frame_range: Optional[bool] = False
        traj_length_frame_range: Optional[float] = None
        traj_visu: Optional[bool] = True

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        self.geometry = torch.nn.ModuleList()
        self.material = torch.nn.ModuleList()
        self.background = torch.nn.ModuleList()
        if self.cfg.scene_setup_path is not None:
            scene_setup = load_config(self.cfg.scene_setup_path)
            self.cfg_scene = scene_setup.config_scene
        else:
            self.cfg_scene = None
        num_objs = len(self.cfg_scene['traj_kwargs'])
        for _ in range(num_objs):
            self.geometry.append(threestudio.find(self.cfg.geometry_type)(self.cfg.geometry))
            self.material.append(threestudio.find(self.cfg.material_type)(self.cfg.material))
            self.background.append(threestudio.find(self.cfg.background_type)(
                self.cfg.background
            ))
        self.background_scene = threestudio.find(self.cfg.background_type)(
                self.cfg.background
            )
        self.renderer= threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
            background_scene=self.background_scene,
            config_scene=self.cfg_scene,
        )
        self.simultan = self.cfg.get("simultan", False)
        self.static = self.cfg.geometry.pos_encoding_config.get("static", True)
        self.multi_rate = self.cfg.get("multi_rate", None)
        self.multi_rate_perc = self.cfg.get("multi_rate_perc", None)
        self.single_view_img = self.cfg.simultan_vid_mod not in [1, None] or self.cfg.simultan_vid_mod_perc not in [1.0, None] or (self.cfg.simultan_vid_mod is None and self.static)
        self.guidance = None
        self.prompt_processor = None
        self.prompt_utils = None
        self.is_multi_view_sds = self.multi_rate is not None or self.multi_rate_perc not in [0.0, None]
        self.is_video_sds = not self.static
        self.tv_loss = TVLoss()
        self.traj_coords = None
        self.proxy_center = None
        if self.is_multi_view_sds:
            self.cfg.prompt_processor_multi_view["prompt"] = self.cfg.prompt_processor["prompt"]

            self.guidance_multi_view = threestudio.find(self.cfg.guidance_type_multi_view)(self.cfg.guidance_multi_view)
            self.prompt_processor_multi_view = threestudio.find(self.cfg.prompt_processor_type_multi_view)(
                self.cfg.prompt_processor_multi_view
            )
            
            self.prompt_utils_multi_view_objs, self.prompt_utils_multi_view_scene = self.get_prompt_processors(
                self.cfg.prompt_processor_multi_view, self.cfg.prompt_processor_type_multi_view
                )
            self.guidance = self.guidance_multi_view
            self.prompt_utils = self.prompt_utils_multi_view_scene
        if self.is_video_sds:
            self.cfg.prompt_processor_video["prompt"] = self.cfg.prompt_processor["prompt"]
            self.guidance_video = threestudio.find(self.cfg.guidance_type_video)(self.cfg.guidance_video)
            self.prompt_processor_video = threestudio.find(self.cfg.prompt_processor_type_video)(
                self.cfg.prompt_processor_video
            )
            self.prompt_utils_video = self.prompt_processor_video()
            if self.guidance is None:
                self.guidance = self.guidance_video
                self.prompt_utils = self.prompt_utils_video
        if self.single_view_img:
            self.guidance_single_view = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.prompt_utils_single_view_objs, self.prompt_utils_single_view_scene = self.get_prompt_processors(
                self.cfg.prompt_processor, self.cfg.prompt_processor_type
                )
            if self.guidance is None:
                self.guidance = self.guidance_single_view
                self.prompt_utils = self.prompt_utils_single_view
    
    def get_prompt_processors(self, prompt_processor, prompt_processor_type):
        prompt_processor_objs = []
        for i, prompt_obj in enumerate(self.cfg_scene.prompt_processor.prompt):
            prompt_processor_obj_cfg = copy.copy(prompt_processor)
            prompt_processor_obj_cfg["prompt"] = prompt_obj
            if "pretrained_model_name_or_path" in self.cfg_scene.prompt_processor:
                pretrained_model_name_or_path = self.cfg_scene.prompt_processor.pretrained_model_name_or_path[i]
            else:
                pretrained_model_name_or_path = prompt_processor.pretrained_model_name_or_path
            prompt_processor_obj_cfg["pretrained_model_name_or_path"] = pretrained_model_name_or_path
            prompt_processor_obj = threestudio.find(prompt_processor_type)(prompt_processor_obj_cfg)()
            prompt_processor_objs.append(prompt_processor_obj)
        prompt_processor_scene = threestudio.find(prompt_processor_type)(
            prompt_processor
        )()
        return prompt_processor_objs, prompt_processor_scene

    def on_validation_start(self) -> None:
        super().on_validation_start()
        self.load_multi_ckpt_cfg()
        self.do_update_step(self.true_current_epoch, self.true_global_step)
    
    def on_test_start(self) -> None:
        super().on_test_start()
        self.load_multi_ckpt_cfg()
        self.traj_coords = None
        self.proxy_center = None
    
    def load_multi_ckpt_cfg(self):
        # Load checkpoints from separately trained models into one scene
        if 'checkpoints' in self.cfg_scene:
            ckpt_paths = self.cfg_scene['checkpoints']
            for obj_idx, ckpt_path in enumerate(ckpt_paths):
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt['state_dict']
                for _ in range(len(state_dict)):
                    k, v = state_dict.popitem(False)
                    if 'trajs' in k:
                        continue
                    elif 'geometry.0' in k:
                        k = k.replace('geometry.0', f'geometry.{obj_idx}')
                    elif 'background.0' in k:
                        k = k.replace('background.0', f'background.{obj_idx}')
                    elif 'estimators.0' in k:
                        k = k.replace('estimators.0', f'estimators.{obj_idx}')
                    elif 'estimators.0' in k:
                        k = k.replace('estimators.0', f'estimators.{obj_idx}')
                    elif 'renderers.0' in k:
                        k = k.replace('renderers.0', f'renderers.{obj_idx}')
                    state_dict[k] = v
                self.load_state_dict(state_dict, strict=False)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "nerf_to_render_idx" not in batch or not self.cfg.multi_obj:
            batch["nerf_to_render_idx"] = 0
        if self.cfg.stage == "geometry":
            render_outs_all = self.renderer(**batch, render_normal=True, render_rgb=False)
        else:
            if not self.static:
                render_outs = []
                # TODO: Handle batch size higher than 1
                batch["frame_times"] = batch["frame_times"].flatten()
                for frame_idx, frame_time in enumerate(batch["frame_times"].tolist()):
                    for geometry in self.geometry:
                        geometry.encoding.encoding.frame_time = frame_time
                    if batch['train_dynamic_camera']:
                        batch_frame = {}
                        for k_frame, v_frame in batch.items():
                            if isinstance(v_frame, torch.Tensor):
                                if v_frame.shape[0] == batch["frame_times"].shape[0]:
                                    v_frame_up = v_frame[[frame_idx]].clone()
                                else:
                                    v_frame_up = v_frame.clone()
                            else:
                                v_frame_up = v_frame
                            batch_frame[k_frame] = v_frame_up
                        render_out = self.renderer(**batch_frame)
                    else:
                        render_out = self.renderer(**batch)
                    if not self.training:
                        for k, v in render_out.items():
                            if isinstance(v, torch.Tensor):
                                render_out[k] = v.cpu()
                    render_outs.append(render_out)
                render_outs_all = {}
                for k in render_out:
                    render_out_k = [render_out_i[k] for render_out_i in render_outs]
                    render_outs_all[k] = torch.cat(render_out_k) if render_out_k[0] is not None else None
            else:
                render_outs_all = self.renderer(**batch)
        return {
            **render_outs_all,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        for geometry in self.geometry:
            geometry.set_density_grid()

    def training_step(self, batch, batch_idx):
        if self.cfg.use_traj_length_frame_range:
            traj_length = self.renderer.base_renderer.obj_trajs.lengths[batch['nerf_to_render_idx']]
            num_frames = batch['frame_times'].shape[0]
            frame_range = min(self.cfg.traj_length_frame_range/traj_length, 1.)
            t0 = torch.FloatTensor(1).uniform_(0, 1-frame_range+frame_range/num_frames).item()
            frame_times = torch.linspace(t0, t0+(num_frames-1)*frame_range/num_frames, num_frames, device=batch['frame_times'].device)
            for k in ['frame_times', 'frame_times_video']:
                batch[k] = frame_times
        
        is_video = batch["is_video"]
        batch_size = batch['c2w'].shape[0]
        if batch['train_dynamic_camera']:
            batch_size = batch_size // batch['frame_times'].shape[0]
        if not self.static:
            # Freeze static components
            for p in self.parameters():
                p.requires_grad_(False)
        if is_video: 
            guidance = self.guidance_video
            prompt_utils = self.prompt_utils_video
            static = self.static
            for geometry in self.geometry:
                geometry.encoding.encoding.is_video = True
                geometry.encoding.encoding.set_temp_param_grad(True)
        else:
            # TODO: Also video prompt per object
            if batch['single_view']:
                guidance = self.guidance_single_view
                if batch["render_scene"] and not self.cfg.scene_single_obj:
                    prompt_utils = self.prompt_utils_single_view_scene
                else:
                    prompt_utils = self.prompt_utils_single_view_objs[batch['nerf_to_render_idx']]
            else:
                guidance = self.guidance_multi_view
                if batch["render_scene"] and not self.cfg.scene_single_obj:
                    prompt_utils = self.prompt_utils_multi_view_scene
                else:
                    prompt_utils = self.prompt_utils_multi_view_objs[batch['nerf_to_render_idx']]
            static = True
            num_static_frames = 1
            batch["frame_times"] = batch["frame_times"][torch.randperm(batch["frame_times"].shape[0])][:num_static_frames]
            video_grad = not self.static
            for geometry in self.geometry:
                geometry.encoding.encoding.is_video = video_grad
                geometry.encoding.encoding.set_temp_param_grad(video_grad)
        out = self(batch)
        if not self.static:
            if static:
                batch['num_frames'] = num_static_frames
            else:
                batch['num_frames'] = self.cfg.geometry.pos_encoding_config.num_frames 

        if self.cfg.stage == "geometry":
            guidance_inp = out["comp_normal"]
            guidance_out = guidance(
                guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False
            )
        else:
            guidance_inp = out["comp_rgb"]
            if static:
                # TODO: Parallelize
                guidance_out_list = [guidance(guidance_inp_i, prompt_utils, **batch, rgb_as_latents=False) for guidance_inp_i in guidance_inp.split(batch_size)]
                guidance_out = {k: torch.zeros_like(v) for k, v in guidance_out_list[0].items()}
                for guidance_out_i in guidance_out_list:
                    for k, v in guidance_out.items():
                        guidance_out[k] = v + guidance_out_i[k]
                for k, v in guidance_out.items():
                    guidance_out[k] = v / len(guidance_out_list)
            else:
                guidance_out = guidance(
                    guidance_inp, prompt_utils, **batch, rgb_as_latents=False
                )
        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.cfg.stage == "coarse":
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            if self.C(self.cfg.loss.lambda_sparsity) > 0:
                loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
                self.log("train/loss_sparsity", loss_sparsity)
                loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
                self.log("train/loss_opaque", loss_opaque)
                loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if self.C(self.cfg.loss.lambda_z_variance) > 0:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)
            
            if self.C(self.cfg.loss.lambda_deformation) > 0 and "deformation" in out and out["deformation"] is not None:
                loss_deformation = self.tv_loss(out["deformation"].permute(3, 0, 1, 2))
                self.log("train/loss_deformation", loss_deformation)
                loss += loss_deformation * self.C(self.cfg.loss.lambda_deformation)
            
            if self.C(self.cfg.loss.lambda_elastic) > 0 and "elastic_loss" in out and out["elastic_loss"] is not None:
                weights = out['weights'].detach()
                loss_elastic = weights * out["elastic_loss"]
                loss_elastic = loss_elastic.mean()
                loss += loss_elastic * self.C(self.cfg.loss.lambda_elastic)
            if self.C(self.cfg.loss.lambda_divergence) > 0 and "divergence_loss" in out and out["divergence_loss"] is not None:
                weights = out['weights'].detach()
                loss_divergence = weights * out["divergence_loss"]
                loss_divergence = loss_divergence.mean()
                loss += loss_divergence * self.C(self.cfg.loss.lambda_divergence)
            if self.C(self.cfg.loss.lambda_rigidity) > 0 and "dx" in out and out["dx"] is not None:
                loss_rigidity = (out["dx"] - out["dx_shifted"]).abs().mean()
                self.log("train/loss_rigidity", loss_rigidity)
                loss += loss_rigidity * self.C(self.cfg.loss.lambda_rigidity)
            
            # For hexplanes
            if self.C(self.cfg.loss.lambda_tv) > 0:
                loss_tv = self.renderer.geometry.encoding.encoding.tv_loss()
                self.log("train/loss_tv", loss_tv)
                loss += loss_tv * self.C(self.cfg.loss.lambda_tv)

        elif self.cfg.stage == "geometry":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
        elif self.cfg.stage == "texture":
            pass
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        with torch.no_grad():
            out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out and out["comp_normal"] is not None
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

        if not self.static:
            batch_video = {k: v for k, v in batch.items() if k != "frame_times"}
            batch_video["frame_times"] = batch["frame_times_video"]
            with torch.no_grad():
                out_video = self(batch_video)
            if self.traj_coords is None:
                batch_video = {k: v for k, v in batch.items() if k != "frame_times"}
                batch_video["frame_times"] = batch["frame_times_video"]
                with torch.no_grad():
                    out_video = self(batch_video)
                proxy_center = out_video["proxy_center"]
                frame_num, obj_num = proxy_center.shape[0], proxy_center.shape[1]
                proxy_center = proxy_center.reshape(-1, proxy_center.shape[-1])
                proxy_center_homo = torch.cat([proxy_center, torch.ones(proxy_center.shape[0], 1).to(proxy_center.device)], dim=1).unsqueeze(-1)
                w2c = torch.inverse(batch_video["c2w"]).repeat(frame_num * obj_num, 1, 1)
                # construct the camera intrinsic matrix
                intrinsic = torch.zeros((frame_num * obj_num, 3, 4)).to(proxy_center.device)
                intrinsic[:, 0, 0] = intrinsic[:, 1, 1] = batch_video["focal_length"]
                intrinsic[:, 2, 2] = 1.
                intrinsic[:, 0, 2] = batch_video["height"] * 0.5
                intrinsic[:, 1, 2] = batch_video["width"] * 0.5
                proxy_center_camera = torch.bmm(w2c.to(proxy_center_homo.device), proxy_center_homo)
                proxy_center_camera = proxy_center_camera / proxy_center_camera[:, -1].unsqueeze(-1)
                proxy_center_2d = torch.bmm(intrinsic, proxy_center_camera)
                proxy_center_2d = proxy_center_2d / proxy_center_2d[:, -1].unsqueeze(-1)
                proxy_center_2d = proxy_center_2d.squeeze(-1).reshape(frame_num, obj_num, -1)
                proxy_center_2d = proxy_center_2d.int()
                coordinate = torch.zeros_like(proxy_center_2d)
                coordinate[..., 0] = proxy_center_2d[..., 1]
                coordinate[..., 1] = batch_video["width"] - proxy_center_2d[..., 0]

                self.traj_coords = coordinate
            else:
                frame_num, obj_num = self.traj_coords.shape[0], self.traj_coords.shape[1]
            coordinate = self.traj_coords
            proxy_rgb = out_video["comp_rgb"].clone()
            palette = [
                torch.tensor([1.0, 0.0, 0.0]),
                torch.tensor([0.0, 1.0, 0.0]),
                torch.tensor([0.0, 0.0, 1.0]),
                torch.tensor([1.0, 1.0, 0.0]),
                torch.tensor([1.0, 0.0, 1.0]),
                torch.tensor([0.0, 1.0, 1.0]),
                torch.tensor([1.0, 1.0, 1.0]),
                torch.tensor([0.0, 0.0, 0.0]),
            ]
            for i in range(frame_num):
                for j in range(obj_num):
                    x, y = coordinate[i, j, 0], coordinate[i, j, 1]
                    # avoid extreme camera view cause the trajectory point out of index
                    if x < 0 or x >= batch_video["height"] - 1 or y < 0 or y >= batch_video["width"] - 1:
                        continue

                    # always visualize the whole trajectory
                    color_strength = batch_video["height"] // 256
                    for k in range(x - color_strength, x + color_strength + 1):
                        for l in range(y - color_strength, y + color_strength + 1):
                            if k < 0 or k >= batch_video["height"] - 1 or l < 0 or l >= batch_video["width"] - 1:
                                continue
                            proxy_rgb[:, k, l, :] = palette[j]

                    for k in range(x - 2 * color_strength, x + 2 * color_strength + 1):
                        for l in range(y - 2 * color_strength, y + 2 * color_strength + 1):
                            if k < 0 or k >= batch_video["height"] - 1 or l < 0 or l >= batch_video["width"] - 1:
                                continue
                            proxy_rgb[i, k, l, :] = torch.tensor([1.0, 1.0, 1.0])

            self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}_video_traj.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": proxy_rgb,
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out_video
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
            video=True
            )
            self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}_video_traj.png",
            (
                [
                    {
                        "type": "rgba",
                        "img": torch.cat((proxy_rgb, out_video['opacity']), -1),
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out_video
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
            video=True
            )
            self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}_video.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out_video["comp_rgb"],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out_video
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out_video["comp_normal"],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out_video and out_video["comp_normal"] is not None
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out_video["opacity"],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
            video=True
        )

        if self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance_single_view.sample(
                            self.prompt_utils, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance_single_view.sample_lora(self.prompt_utils, **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        if batch_idx == 0:
            self.out_depths = []
        with torch.no_grad():
            out = self(batch)
        depth = out["depth"][0, :, :, 0].detach().cpu().numpy()
        self.out_depths.append(depth)
        if "comp_rgb" in out:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name="test_step",
                step=self.true_global_step,
            )
            self.save_image_grid(
                f"it{self.true_global_step}-test-rgba/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgba",
                            "img": torch.cat((out["comp_rgb"][0], out['opacity'][0]), -1),
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name="test_step",
                step=self.true_global_step,
            )
        if "comp_normal" in out and out["comp_normal"] is not None:
            self.save_image_grid(
                f"it{self.true_global_step}-test-normal/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                ),
                name="test_step",
                step=self.true_global_step,
            )
        if not self.static:
            batch_static = {k: v for k, v in batch.items() if k != "frame_times"}
            batch_static["frame_times"] = torch.zeros_like(batch["frame_times"])
            if self.cfg.traj_visu:
                time_idx = torch.where(batch["frame_times"].item() == batch["frame_times_video"])[1][0]
                batch_video = {k: v for k, v in batch.items() if k != "frame_times"}
                batch_video["frame_times"] = batch["frame_times_video"][[0], time_idx]
                if self.proxy_center is None:
                    batch_video = {k: v for k, v in batch.items() if k != "frame_times"}
                    batch_video["frame_times"] = batch["frame_times_video"]
                    with torch.no_grad():
                        out_video = self(batch_video)
                    proxy_center = out_video["proxy_center"]
                    self.proxy_center = proxy_center
                else:
                    proxy_center = self.proxy_center
                frame_num, obj_num = proxy_center.shape[0], proxy_center.shape[1]
                proxy_center = proxy_center.reshape(-1, proxy_center.shape[-1])
                proxy_center_homo = torch.cat([proxy_center, torch.ones(proxy_center.shape[0], 1).to(proxy_center.device)], dim=1).unsqueeze(-1)
                w2c = torch.inverse(batch_video["c2w"]).repeat(frame_num * obj_num, 1, 1)
                # construct the camera intrinsic matrix
                intrinsic = torch.zeros((frame_num * obj_num, 3, 4)).to(proxy_center.device)
                intrinsic[:, 0, 0] = intrinsic[:, 1, 1] = batch_video["focal_length"]
                intrinsic[:, 2, 2] = 1.
                intrinsic[:, 0, 2] = batch_video["height"] * 0.5
                intrinsic[:, 1, 2] = batch_video["width"] * 0.5
                proxy_center_camera = torch.bmm(w2c.to(proxy_center_homo.device), proxy_center_homo)
                proxy_center_camera = proxy_center_camera / proxy_center_camera[:, -1].unsqueeze(-1)
                proxy_center_2d = torch.bmm(intrinsic, proxy_center_camera)
                proxy_center_2d = proxy_center_2d / proxy_center_2d[:, -1].unsqueeze(-1)
                proxy_center_2d = proxy_center_2d.squeeze(-1).reshape(frame_num, obj_num, -1)
                proxy_center_2d = proxy_center_2d.int()
                coordinate = torch.zeros_like(proxy_center_2d)
                coordinate[..., 0] = proxy_center_2d[..., 1]
                coordinate[..., 1] = batch_video["width"] - proxy_center_2d[..., 0]
                self.traj_coords = coordinate
                frame_num, obj_num = self.traj_coords.shape[0], self.traj_coords.shape[1]
                with torch.no_grad():
                    out_video = self(batch_video)
                out = out_video
                coordinate = self.traj_coords
                proxy_rgb = out["comp_rgb"].clone()
                proxy_rgb = proxy_rgb[[0]]
                palette = [
                    torch.tensor([208/255, 0.0, 0.0]),
                    torch.tensor([80/255, 200/255, 120/255]),
                    torch.tensor([0,1.0,1.0]),
                    torch.tensor([138/255,43/255,226/255]),
                    torch.tensor([1.0, 0.0, 1.0]),
                    torch.tensor([0.0, 1.0, 1.0]),
                    torch.tensor([1.0, 1.0, 1.0]),
                    torch.tensor([0.0, 0.0, 0.0]),
                ]

                for i in range(frame_num):
                    for j in range(obj_num):
                        x, y = coordinate[i, j, 0], coordinate[i, j, 1]
                        # avoid extreme camera view cause the trajectory point out of index
                        if x < 0 or x >= batch["height"] - 1 or y < 0 or y >= batch["width"] - 1:
                            continue

                        # always visualize the whole trajectory
                        color_strength = batch["height"] // 256
                        for k in range(x - color_strength, x + color_strength + 1):
                            for l in range(y - color_strength, y + color_strength + 1):
                                if k < 0 or k >= batch["height"] - 1 or l < 0 or l >= batch["width"] - 1:
                                    continue
                                proxy_rgb[:, k, l, :] = palette[j]

                    for j in range(obj_num):
                        x, y = coordinate[time_idx, j, 0], coordinate[time_idx, j, 1]
                        for k in range(x - 2 * color_strength, x + 2 * color_strength + 1):
                            for l in range(y - 2 * color_strength, y + 2 * color_strength + 1):
                                if k < 0 or k >= batch["height"] - 1 or l < 0 or l >= batch["width"] - 1:
                                    continue
                                proxy_rgb[0, k, l, :] = torch.tensor([1.0, 1.0, 1.0])

                # self.save_image_grid(
                # f"it{self.true_global_step}-{batch['index'][0]}_video_traj.png",
                # (
                #     [
                #         {
                #             "type": "rgb",
                #             "img": proxy_rgb,
                #             "kwargs": {"data_format": "HWC"},
                #         },
                #     ]
                #     if "comp_rgb" in out_video
                #     else []
                # ),
                # name="test_step",
                # step=self.true_global_step,
                # video=True
                # )
                self.save_image_grid(
                    f"it{self.true_global_step}-test_video_traj/{batch['index'][0]}.png",
                    (
                        [
                            {
                                "type": "rgb",
                                "img": proxy_rgb[0],
                                "kwargs": {"data_format": "HWC"},
                            },
                        ]
                    ),
                    name="test_step",
                    step=self.true_global_step,
                )
                self.save_image_grid(
                    f"it{self.true_global_step}-test-rgba_video_traj/{batch['index'][0]}.png",
                    (
                        [
                            {
                                "type": "rgba",
                                "img": torch.cat((proxy_rgb[0], out['opacity'][0]), -1),
                                "kwargs": {"data_format": "HWC"},
                            },
                        ]
                    ),
                    name="test_step",
                    step=self.true_global_step,
                )

            if "comp_normal" in out and out["comp_normal"] is not None:
                self.save_image_grid(
                    f"it{self.true_global_step}-test-normal_static/{batch_static['index'][0]}.png",
                    (
                        [
                            {
                                "type": "rgb",
                                "img": out["comp_normal"][0],
                                "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                            }
                        ]
                    ),
                    name="test_step",
                    step=self.true_global_step
                )

    def on_test_epoch_end(self):
        fps_setups = [10, 20, 30, 40, 50, 60]
        for fps in fps_setups:
            self.save_img_sequence(
                f"it{self.true_global_step}-test{fps}",
                f"it{self.true_global_step}-test",
                "(\d+)\.png",
                save_format="mp4",
                fps=fps,
                name=f"test{fps}",
                step=self.true_global_step,
            )
            if self.cfg.traj_visu and not self.static:
                self.save_img_sequence(
                    f"it{self.true_global_step}-test_video_traj{fps}",
                    f"it{self.true_global_step}-test_video_traj",
                    "(\d+)\.png",
                    save_format="mp4",
                    fps=fps,
                    name=f"test_video_traj{fps}",
                    step=self.true_global_step,
                    rgba=True,
                )
        # if not self.static:
        #     self.save_img_sequence(
        #             f"it{self.true_global_step}-test_static",
        #             f"it{self.true_global_step}-test_static",
        #             "(\d+)\.png",
        #             save_format="mp4",
        #             fps=fps,
        #             name=f"test_static",
        #             step=self.true_global_step,
        #         )
        #     if self.renderer.cfg.eval_normal:
        #         self.save_img_sequence(
        #             f"it{self.true_global_step}-test-normal_static",
        #             f"it{self.true_global_step}-test-normal_static",
        #             "(\d+)\.png",
        #             save_format="mp4",
        #             fps=fps,
        #             name=f"test-normal_static",
        #             step=self.true_global_step,
        #         )
        out_depths = np.stack(self.out_depths)
        perc_min_val, perc_max_val = 6, 99
        non_zeros_depth = out_depths[out_depths != 0]
        self.visu_perc_min_depth = np.percentile(non_zeros_depth, perc_min_val)
        self.visu_perc_max_depth = np.percentile(non_zeros_depth, perc_max_val)
        depth_color_maps = ['inferno', 'jet']
        for depth_color_map in depth_color_maps:
            for i, depth in enumerate(out_depths):
                self.save_image_grid(
                    f"it{self.true_global_step}-test-depth-{depth_color_map}/{i}.png",
                    [
                        {
                            "type": "grayscale",
                            "img": depth,
                            "kwargs": {"cmap": depth_color_map, "data_range": 'nonzero'},
                        },
                    ],
                    name="depth_test_step",
                    step=self.true_global_step,
                )
        extra_renderings = [f'depth-{depth_color_map}' for depth_color_map in depth_color_maps]
        for extra_rendering in extra_renderings:
            for fps in fps_setups:
                self.save_img_sequence(
                    f"it{self.true_global_step}-test-{extra_rendering}{fps}",
                    f"it{self.true_global_step}-test-{extra_rendering}",
                    "(\d+)\.png",
                    save_format="mp4",
                    fps=fps,
                    name=f"test{fps}",
                    step=self.true_global_step,
                )
