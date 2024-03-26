import threestudio
from threestudio.utils.typing import *
from threestudio.utils.bounding_boxes import scale_and_shift_box

import torch
import math
from scipy.spatial.transform import Rotation as R
import scipy
import numpy as np

@threestudio.register("trajectory")
class SceneTrajectory(torch.nn.Module):

    def __init__(
            self, 
            config_scene,
            estimators,
            renderers,
            ) -> None:
        super().__init__()
        self.num_objs = len(config_scene.traj_kwargs)
        self.objs = torch.nn.ModuleList([
            ObjectTrajectory(
                traj_kwargs=config_scene.traj_kwargs[obj_idx],
                estimator=estimators[obj_idx],
                estimator_sub=renderers[obj_idx].estimator,
            )
            for obj_idx in range(self.num_objs)
        ])
        self.lengths = [sum(obj.lengths) for obj in self.objs]
    
    def update_objs(self, frame_times):
        for obj_traj, frame_time in zip(self.objs, frame_times):
            obj_traj.update_obj(frame_time)

@threestudio.register("object-trajectory")
class ObjectTrajectory(torch.nn.Module):

    def __init__(
            self,
            traj_kwargs,
            **kwargs
            ) -> None:
        super().__init__()
        self.proxy_size = traj_kwargs.proxy_size
        self.proxy_rotation_mat = None
        self.trajs = torch.nn.ModuleList([
            threestudio.find(traj_sub_kwargs.traj_type)(
                traj_kwargs=traj_sub_kwargs,
                proxy_size=traj_kwargs.proxy_size,
                **kwargs
            )
            for traj_sub_kwargs in traj_kwargs.trajs
        ])
        self.lengths = [traj.length for traj in self.trajs]
        self.get_lengths()
        self.set_translation_offsets()
    
    def set_translation_offsets(self):
        # Set offsets based on start position or previous end position
        for traj_idx, traj in enumerate(self.trajs):
            if traj_idx == 0:
                traj.set_translation_offset(traj.proxy_center_init)
            else:
                traj.set_translation_offset(self.trajs[traj_idx-1].translation_end)
    
    def update_obj(self, frame_time: float):
        if len(self.trajs) > 1:
            frame_time = torch.tensor(frame_time, device=self.frame_time_cumsum.device).clip(0, self.frame_time_cumsum[-1])
            traj_idx = torch.where((self.frame_time_cumsum - frame_time) >= 0)[0][0]
            frame_time = (frame_time - self.frame_times_start[traj_idx])/self.frame_time_range[traj_idx]
        else:
            traj_idx = 0
        self.proxy_center, self.proxy_rotation_mat = self.trajs[traj_idx].update_obj(frame_time)
        self.proxy_z_offset = self.trajs[traj_idx].proxy_z_offset
        if self.trajs[traj_idx].proxy_size_scaling is not None:
            self.proxy_size = self.trajs[traj_idx].proxy_size

    def get_lengths(self):
        lengths = torch.tensor(self.lengths)
        self.frame_time_range = lengths/lengths.sum()
        self.frame_time_cumsum = self.frame_time_range.cumsum(0)
        frame_time_start = torch.zeros(1, device=self.frame_time_cumsum.device)
        self.frame_times_start = torch.cat((frame_time_start, self.frame_time_cumsum[:-1]))

    def get_pos(self):
        pass

    def get_rot(self):
        pass

@threestudio.register("general-trajectory")
class GeneralTrajectory(torch.nn.Module):

    def __init__(
            self,
            traj_kwargs,
            proxy_size,
            estimator,
            estimator_sub,
            adjust_offset_density: bool = False,
            adjust_offset_density_thres: float = 0.001,
            # TODO: read from config
            proxy_is_floor: bool = False,
            proxy_z_offset: float = 0.0,
            ) -> None:
        super().__init__()
        self.register_buffer("proxy_size", torch.tensor(proxy_size))
        self.estimator = estimator
        self.estimator_sub = estimator_sub
        self.traj_kwargs = traj_kwargs
        self.adjust_offset_density = adjust_offset_density
        self.adjust_offset_density_thres = adjust_offset_density_thres
        self.proxy_is_floor = proxy_is_floor
        self.proxy_z_offset = proxy_z_offset
        self.dil_kernel_size = traj_kwargs.get('dil_kernel_size', 5)
        self.er_kernel_size = traj_kwargs.get('er_kernel_size', 0)
        self.proxy_size_scaling = traj_kwargs.get('proxy_size_scaling', None)
        self.length = traj_kwargs.get('length', 0.3)
        if self.proxy_size_scaling is not None:
            self.proxy_size_scaling = torch.tensor(self.proxy_size_scaling)
            self.register_buffer("proxy_size_orig", torch.tensor(proxy_size))
        self.dil_kernel = self.dil_kernel_size != 0
        if self.dil_kernel:
            self.register_buffer('kernel_dil', torch.ones((1, 1, *(self.dil_kernel_size,)*3)))
        self.er_kernel = self.er_kernel_size != 0
        if self.er_kernel:
            self.register_buffer('kernel_er', torch.ones((1, 1, *(self.er_kernel_size,)*3)))
    
    def set_pos_init(self):
        self.proxy_center_init = None

    @property
    def estimator_res(self):
        return self.estimator.resolution[0].item()

    @property
    def est_size(self):
        return self.estimator_sub.binaries.shape[1]

    @property
    def device(self):
        return self.estimator.device
    
    def update_obj(self, frame_time: float):
        self.reset_estimator()
        # Rotate and translate object
        position, rotation = self.get_translation_rotation(frame_time)
        rotation = rotation + self.angle_offset
        rotation_mat = self.get_rotation_mat(rotation, frame_time)
        # Rotate and translate estimator
        self.set_estimator(position, rotation)
        return position, rotation_mat

    def get_translation_rotation(self, frame_time: float):
        rotation = self.get_rotation(frame_time)
        if np.isnan(rotation):
            rotation = 0.
        translation = self.get_translation(frame_time=frame_time, rotation=rotation) 
        translation = translation + self.translation_start.to(translation.device)
        return translation, rotation

    def set_translation_offset(self, translation_end_previous: torch.Tensor = None):
        self.translation_start = self.get_translation(self.start_end_vals[0])
        if translation_end_previous is not None:
            self.translation_start = translation_end_previous - self.translation_start
        self.translation_end = self.get_translation(self.start_end_vals[1]) + self.translation_start

    def reset_estimator(self):
        self.estimator.occs.fill_(False)
        self.estimator.binaries.fill_(False)

    def get_rotation_mat(self, rotation: float, frame_time: float):
        rotation_mat = R.from_euler('z', rotation).as_matrix()
        rotation_mat = torch.from_numpy(rotation_mat).to(self.proxy_size.dtype).to(self.device)
        proxy_size = self.proxy_size
        if self.proxy_size_scaling is not None:
            proxy_size = self.proxy_size_orig*(1 + self.proxy_size_scaling.to(self.device) * frame_time)
            self.proxy_size = proxy_size
        rotation_mat = rotation_mat/proxy_size
        return rotation_mat
    
    def set_estimator(self, position: List[float], rotation: float):
        proxy_size_sub_min = proxy_size_sub_max = self.proxy_size/2
        if self.adjust_offset_density:
            binaries = (self.estimator_sub.occs.view(self.estimator_sub.binaries.shape) > self.adjust_offset_density_thres)
            if binaries.sum() == 0:
                proxy_z_offset = 0
            else:
                if self.proxy_is_floor:
                    proxy_z_offset = (1-torch.where(binaries)[-1].max()/(self.est_size-1))*self.proxy_size[-1]
                else:
                    proxy_z_offset = -(torch.where(binaries)[-1].min()/(self.est_size-1))*self.proxy_size[-1]
            position = position.clone()
            position[-1] = position[-1] + proxy_z_offset
            self.proxy_z_offset = proxy_z_offset
        with torch.no_grad():
            estimator_occ_grid = scale_and_shift_box(
                proxy_size_sub_min, proxy_size_sub_max, rotation, position, self.device, self.estimator_res, self.estimator_res, self.estimator_sub
                )
        if self.dil_kernel:
            estimator_occ_grid = torch.clamp(torch.nn.functional.conv3d(estimator_occ_grid, self.kernel_dil, padding='same'), 0, 1)
        if self.er_kernel:
            estimator_occ_grid = 1 - torch.clamp(torch.nn.functional.conv3d(1-estimator_occ_grid, self.kernel_er, padding='same'), 0, 1)
        # estimator_occ_grid = estimator_occ_grid > 0.5
        # estimator_occ_grid = estimator_occ_grid > 0.0
        # estimator_occ_grid = estimator_occ_grid == 1.0
        # estimator_occ_grid = estimator_occ_grid > 0.01
        estimator_occ_grid = estimator_occ_grid > 0.0
        self.estimator.binaries = estimator_occ_grid
        self.estimator.occs = estimator_occ_grid.reshape(self.estimator.occs.shape).float()

@threestudio.register("spline-trajectory")
class SplineTrajectory(GeneralTrajectory):

    def __init__(
            self,
            traj_kwargs,
            linearize: bool = False,
            **kwargs
            ) -> None:
        super().__init__(traj_kwargs, **kwargs)
        self.linearize = linearize
        self.coords = np.array(traj_kwargs.coords)
        self.inter_tck, _ = scipy.interpolate.splprep(
            [self.coords[:,0], self.coords[:,1], self.coords[:,2]],
            k=min(len(self.coords)-1, 3),
            s=0
            )
        self.angle_offset = math.radians(traj_kwargs.angle_offset)
        self.time_offset = traj_kwargs.get('time_offset', 0.)
        self.start_end_vals = [0.0, 1.0]
        if self.linearize:
            self.segments = self.segment_spline()
        else:
            self.segments = None
        self.set_pos_init()
        self.calc_length()

    def set_pos_init(self):
        self.proxy_center_init = torch.nn.Parameter(self.eval_spline(0), requires_grad=False)
    
    def eval_spline(
            self, frame_time: float, der: int = 0, device: torch.device = "cpu",
            dtype: np.dtype = np.float32, out_torch: bool = True, transform_time: bool = True
            ):
        if self.time_offset != 0.:
            frame_time = (frame_time + self.time_offset) % 1
        if transform_time and self.linearize:
            frame_time = self.transform_time(frame_time)
        inter = np.array(scipy.interpolate.splev(frame_time, self.inter_tck, der), dtype=dtype)
        if out_torch:
            inter = torch.tensor(inter, device=device)
        return inter

    def get_translation(self, frame_time: float, **kwargs):
        translation = self.eval_spline(frame_time, device=self.device)
        return translation

    def get_rotation(self, frame_time: float):
        derivative = self.eval_spline(frame_time, der=1)
        rotation = torch.arctan(derivative[1]/derivative[0]).item()
        if derivative[0] < 0:
            rotation = math.pi + rotation
        return rotation
    
    def calc_length(self):
        self.length = self.segment_spline()[-1]
    
    def segment_spline(self, segment_count: int = 1000):
        t_values = np.linspace(0, 1, segment_count + 1)
        points = self.eval_spline(t_values, out_torch=False, transform_time=False)
        points = np.array(points)
        segments = np.zeros(segment_count + 1)
        segments[1:] = np.linalg.norm(points[..., 1:] - points[..., :-1], axis=0).cumsum()
        return segments

    def transform_time(self, u: np.ndarray):
        if u == 1:
            return 1
        num_segments = len(self.segments) - 1
        d = u * self.segments[num_segments]
        k = np.argmax(self.segments > d) - 1
        lo = self.segments[k]
        hi = self.segments[k + 1]
        k_sub = (d - lo) / (hi - lo)
        u_constant = (k + k_sub)/num_segments
        return u_constant

@threestudio.register("curvature-trajectory")
class CurvatureTrajectory(GeneralTrajectory):

    def __init__(
            self,
            traj_kwargs,
            **kwargs
            ) -> None:
        super().__init__(traj_kwargs, **kwargs)
        self.radius = traj_kwargs.radius
        self.angle_start = math.radians(traj_kwargs.angle_start)
        self.angle_range = math.radians(traj_kwargs.angle_range)
        self.angle_offset = math.radians(traj_kwargs.angle_offset)
        self.time_offset = traj_kwargs.get('time_offset', 0.)
        self.start_end_vals = [self.angle_start, self.angle_start + self.angle_range]
        self.calc_length()
        self.set_pos_init()
    
    def set_pos_init(self):
        if "pos_start" in self.traj_kwargs:
            self.proxy_center_init = torch.nn.Parameter(torch.tensor(self.traj_kwargs.pos_start), requires_grad=False)

    def get_translation(self, rotation: float, **kwargs):
        translation = [self.radius*math.cos(rotation), self.radius*math.sin(rotation), 0.0]
        translation = torch.tensor(translation)
        return translation

    def get_rotation(self, frame_time: float):
        if self.time_offset != 0.:
            frame_time = (frame_time + self.time_offset) % 1
        rotation = self.angle_start + self.angle_range * frame_time
        return rotation
    
    def calc_length(self):
        self.length = self.angle_range * self.radius

@threestudio.register("static-trajectory")
class StaticTrajectory(GeneralTrajectory):

    def __init__(
            self,
            traj_kwargs,
            **kwargs
            ) -> None:
        super().__init__(traj_kwargs, **kwargs)
        self.pos_start = traj_kwargs.pos_start
        self.start_end_vals = [self.pos_start, self.pos_start]
        self.angle_offset = math.radians(traj_kwargs.angle_offset)
        self.calc_length()
        self.set_pos_init()
    
    def set_pos_init(self):
        self.proxy_center_init = torch.nn.Parameter(torch.tensor(self.traj_kwargs.pos_start), requires_grad=False)

    def get_translation(self, rotation: float, **kwargs):
        translation = [0.0, 0.0, 0.0]
        translation = torch.tensor(translation)
        return translation

    def get_rotation(self, frame_time: float):
        rotation = 0
        return rotation
    
    def calc_length(self):
        pass