import math

import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import threestudio
from threestudio.utils.base import Updateable
from threestudio.utils.config import config_to_primitive
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *

class ProgressiveBandFrequency(nn.Module, Updateable):
    def __init__(self, in_channels: int, config: dict):
        super().__init__()
        self.N_freqs = config["n_frequencies"]
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2 ** torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = config.get("n_masking_step", 0)
        self.update_step(
            None, None
        )  # mask should be updated at the beginning each step

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq * x) * mask]
        return torch.cat(out, -1)

    def update_step(self, epoch, global_step, on_load_weights=False):
        if self.n_masking_step <= 0 or global_step is None:
            self.mask = torch.ones(self.N_freqs, dtype=torch.float32)
        else:
            self.mask = (
                1.0
                - torch.cos(
                    math.pi
                    * (
                        global_step / self.n_masking_step * self.N_freqs
                        - torch.arange(0, self.N_freqs)
                    ).clamp(0, 1)
                )
            ) / 2.0
            threestudio.debug(
                f"Update mask: {global_step}/{self.n_masking_step} {self.mask}"
            )


class TCNNEncoding(nn.Module):
    def __init__(self, in_channels, config, dtype=torch.float32) -> None:
        super().__init__()
        self.n_input_dims = in_channels
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims

    def forward(self, x):
        return self.encoding(x)

class PEEncoder(nn.Module):

    def __init__(self, x_dim=4, min_deg=0, max_deg=4, use_identity: bool = False):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )
        self.n_output_dims = x_dim*2*(max_deg-min_deg)

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.cat((torch.sin(xb), torch.cos(xb)), dim=-1)
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class TCNNEncodingSpatialTimeDeform(nn.Module):
    def __init__(self, in_channels, config, dtype=torch.float32, init_time_zero=True) -> None:
        super().__init__()
        self.n_input_dims = in_channels
        config["otype"] = "HashGrid"
        self.num_frames = 1 #config["num_frames"]
        self.static = config["static"]
        self.compute_elastic_loss = config['compute_elastic_loss']
        self.elastic_loss_alpha = config['elastic_loss_alpha']
        self.compute_rigidity_loss = config['compute_rigidity_loss']
        self.rigidity_loss_std = config['rigidity_loss_std']
        self.rigidity_loss_time = config['rigidity_loss_time']
        self.compute_divergence_loss = config['compute_divergence_loss']
        self.div_type = config['div_type']
        if self.compute_divergence_loss:
            config["time_network_config"]["activation"] = "Sine"
        self.cfg = config_to_primitive(config)
        self.pe_encoding = config["time_encoding_config"]["otype"] == "PE"
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(self.n_input_dims, self.cfg, dtype=dtype)
            if self.pe_encoding:
                self.encoding_time = PEEncoder(self.n_input_dims + 1)
            else:
                self.encoding_time = get_encoding(self.n_input_dims + 1, config["time_encoding_config"])
        self.time_network = get_mlp(
            self.encoding_time.n_output_dims, self.n_input_dims, config["time_network_config"]
            )
        self.n_output_dims = self.encoding.n_output_dims
        self.frame_time = None
        if self.static:
            self.set_temp_param_grad(requires_grad=False)
        if init_time_zero and self.pe_encoding:
            self.init_params_zero(self.time_network.layers[-1])
        self.use_key_frame = config.get("use_key_frame", False)
        self.is_video = True
        self.update_occ_grid = False
    
    def init_params_zero(self, param_list):
        if isinstance(param_list, nn.Parameter):
            nn.init.zeros_(param_list.data)
        else:
            for param in param_list.parameters():
                nn.init.zeros_(param.data)
    
    def set_temp_param_grad(self, requires_grad=False):
        self.set_param_grad(self.encoding_time, requires_grad=requires_grad)
        self.set_param_grad(self.time_network, requires_grad=requires_grad)
        self.set_param_grad(self.encoding, requires_grad=self.static)

    def set_param_grad(self, param_list, requires_grad=False):
        if isinstance(param_list, nn.Parameter):
            param_list.requires_grad = requires_grad
        else:
            for param in param_list.parameters():
                param.requires_grad = requires_grad
    
    def warp(self, x):
        return self.time_network(self.encoding_time(x))

    def log1p_safe(self, x):
        return torch.log1p(torch.minimum(x, torch.tensor(3e37, device=x.device)))


    def exp_safe(self, x):
        return torch.exp(torch.minimum(x, torch.tensor(87.5, device=x.device)))

    def expm1_safe(self, x):
        return torch.expm1(torch.minimum(x, torch.tensor(87.5, device=x.device)))

    def safe_sqrt(x, eps=1e-7):
        safe_x = torch.where(x == 0, torch.ones_like(x) * eps, x)
        return torch.sqrt(safe_x)

    def general_loss_with_squared_residual(self, squared_x, alpha, scale):
        """
        Source: https://github.com/google/nerfies
        """
        eps = torch.tensor(torch.finfo(torch.float32).eps, device=squared_x.device)
        alpha = torch.tensor(alpha, device=squared_x.device)

        # This will be used repeatedly.
        squared_scaled_x = squared_x / (scale ** 2)

        # The loss when alpha == 2.
        loss_two = 0.5 * squared_scaled_x
        # The loss when alpha == 0.
        loss_zero = self.log1p_safe(0.5 * squared_scaled_x)
        # The loss when alpha == -infinity.
        loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
        # The loss when alpha == +infinity.
        loss_posinf = self.expm1_safe(0.5 * squared_scaled_x)

        # The loss when not in one of the above special cases.
        # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
        beta_safe = torch.maximum(eps, torch.abs(alpha - 2.))
        # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
        alpha_safe = torch.where(
            torch.greater_equal(alpha, 0.), torch.ones_like(alpha),
            -torch.ones_like(alpha)) * torch.maximum(eps, torch.abs(alpha))
        loss_otherwise = (beta_safe / alpha_safe) * (
            torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

        # Select which of the cases of the loss to return.
        loss = torch.where(
            alpha == -torch.inf, loss_neginf,
            torch.where(
                alpha == 0, loss_zero,
                torch.where(
                    alpha == 2, loss_two,
                    torch.where(alpha == torch.inf, loss_posinf, loss_otherwise))))
        return scale * loss
    
    def elastic_loss(self, x_frame_time, dx, alpha=-2.0, scale=0.03, delta = 1e-6, eps = 1e-6):
        b = x_frame_time.shape[0]
        m = dx.shape[1]
        n = x_frame_time.shape[1]
        device = x_frame_time.device
        J = torch.zeros((b, m, n), device=device)
        I = torch.eye(n, device=device)
        in_warp = torch.cat([x_frame_time+delta*I[:, j] for j in range(n)])
        with torch.no_grad():
            out_warp = self.warp(in_warp)
        J = torch.stack(out_warp.split(b), -1)
        J = (J-dx.unsqueeze(-1))/ delta
        svals = torch.linalg.svdvals(J)
        log_svals = torch.log(torch.max(svals, torch.tensor(eps, device=device)))
        sq_residual = torch.sum(log_svals**2, dim=-1)
        loss = self.general_loss_with_squared_residual(sq_residual, alpha=alpha, scale=scale)
        return loss

    def divergence_loss(self, x_frame_time, dx, delta = 1e-6, div_clamp = 50, perc_points=0.25):
        sample_num = int(perc_points * x_frame_time.shape[0])
        random_idx = torch.randperm(x_frame_time.shape[0])[:sample_num]
        x_frame_time, dx = x_frame_time[random_idx], dx[random_idx]
        b = x_frame_time.shape[0]
        m = dx.shape[1]
        n = x_frame_time.shape[1]
        device = x_frame_time.device
        J = torch.zeros((b, m, n), device=device)
        I = torch.eye(n, device=device)
        in_warp = torch.cat([x_frame_time+delta*I[:, j] for j in range(n)])
        with torch.no_grad():
            out_warp = self.warp(in_warp)
        out_warp = torch.stack(out_warp.split(b), -1)
        J1 = (out_warp-dx.unsqueeze(-1))/ delta

        in_warp2 = torch.cat([x_frame_time+2*delta*I[:, j] for j in range(n)])
        with torch.no_grad():
            out_warp2 = self.warp(in_warp2)
        out_warp2 = torch.stack(out_warp2.split(b), -1)
        J2 = (out_warp2-out_warp)/ delta

        grad_second_order = torch.sum(((J2 - J1) / delta), dim=-1)
        
        if self.div_type == 'l2':
            divergence = torch.clamp(torch.square(grad_second_order), 0.1, div_clamp)
        elif self.div_type == 'l1':
            divergence = torch.clamp(torch.abs(grad_second_order), 0.1, div_clamp)
        return divergence

    def forward(self, x, grid, out_all=False):
        if self.update_occ_grid and not isinstance(self.frame_time, float):
            frame_time = self.frame_time
        else:
            if (self.static or not self.training) and self.frame_time is None:
                frame_time = torch.zeros((self.num_frames, 1), device=x.device, dtype=x.dtype).expand(x.shape[0], 1)
            else:
                if self.frame_time is None:
                    frame_time = 0.0
                else:
                    frame_time = self.frame_time
                frame_time = (torch.ones((self.num_frames, 1), device=x.device, dtype=x.dtype)*frame_time).expand(x.shape[0], 1)
            frame_time = frame_time.view(-1, 1)
        if not self.static:
            x_frame_time = torch.cat((x, frame_time), 1)
            dx = self.warp(x_frame_time)
            if self.compute_divergence_loss:
                divergence_loss = self.divergence_loss(x_frame_time, dx)
            else:
                divergence_loss = None
            if self.compute_elastic_loss:
                elastic_loss = self.elastic_loss(x_frame_time, dx, alpha=self.elastic_loss_alpha).unsqueeze(-1)
            else:
                elastic_loss = None
            if self.compute_rigidity_loss:
                if self.rigidity_loss_time:
                    x_frame_time_shifted = torch.randn_like(x_frame_time) * self.rigidity_loss_std + x_frame_time
                    x_frame_time_shifted = x_frame_time_shifted.clamp(0, 1)
                else:
                    x_shifted = torch.randn_like(x) * self.rigidity_loss_std + x
                    x_shifted = x_shifted.clamp(0, 1)
                    x_frame_time_shifted = torch.cat((x_shifted, frame_time), 1)
                dx_shifted = self.warp(x_frame_time_shifted)
            else:
                dx_shifted = None
            if self.pe_encoding:
                dx = torch.tanh(dx/0.5)*0.5
                frame_time_non_lin = frame_time[0]**0.35
                dx = dx*frame_time_non_lin
            x = x + dx
        else:
            dx = None
        enc = self.encoding(x)
        if out_all:
            if not self.static:
                return enc, dx, elastic_loss, divergence_loss, dx_shifted
            else:
                return enc, None, None, None, None
        else:
            return enc


class ProgressiveBandHashGrid(nn.Module, Updateable):
    def __init__(self, in_channels, config, dtype=torch.float32):
        super().__init__()
        self.n_input_dims = in_channels
        encoding_config = config.copy()
        encoding_config["otype"] = "Grid"
        encoding_config["type"] = "Hash"
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = config["n_levels"]
        self.n_features_per_level = config["n_features_per_level"]
        self.start_level, self.start_step, self.update_steps = (
            config["start_level"],
            config["start_step"],
            config["update_steps"],
        )
        self.current_level = self.start_level
        self.mask = torch.zeros(
            self.n_level * self.n_features_per_level,
            dtype=torch.float32,
            device=get_rank(),
        )

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask
        return enc

    def update_step(self, epoch, global_step, on_load_weights=False):
        current_level = min(
            self.start_level
            + max(global_step - self.start_step, 0) // self.update_steps,
            self.n_level,
        )
        if current_level > self.current_level:
            threestudio.debug(f"Update current level to {current_level}")
        self.current_level = current_level
        self.mask[: self.current_level * self.n_features_per_level] = 1.0


class CompositeEncoding(nn.Module, Updateable):
    def __init__(self, encoding, include_xyz=False, xyz_scale=1.0, xyz_offset=0.0):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = (
            include_xyz,
            xyz_scale,
            xyz_offset,
        )
        self.n_output_dims = (
            int(self.include_xyz) * self.encoding.n_input_dims
            + self.encoding.n_output_dims
        )

    def forward(self, x, *args, **kwargs):
        return (
            self.encoding(x, *args, **kwargs)
            if not self.include_xyz
            else torch.cat(
                [x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args, **kwargs)], dim=-1
            )
        )


def get_encoding(n_input_dims: int, config) -> nn.Module:
    # input suppose to be range [0, 1]
    encoding: nn.Module
    if config.otype == "ProgressiveBandFrequency":
        encoding = ProgressiveBandFrequency(n_input_dims, config_to_primitive(config))
    elif config.otype == "ProgressiveBandHashGrid":
        encoding = ProgressiveBandHashGrid(n_input_dims, config_to_primitive(config))
    elif config.otype == "HashGridSpatialTimeDeform":
        encoding = TCNNEncodingSpatialTimeDeform(n_input_dims, config)
    else:
        encoding = TCNNEncoding(n_input_dims, config_to_primitive(config))
    encoding = CompositeEncoding(
        encoding,
        include_xyz=config.get("include_xyz", False),
        xyz_scale=2.0,
        xyz_offset=-1.0,
    )  # FIXME: hard coded
    return encoding


class VanillaMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False, bias=config.get("bias", False)),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False, bias=config.get("bias", False)
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True, bias=config.get("bias", False))
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last, bias=False):
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)


class TCNNNetwork(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict) -> None:
        super().__init__()
        with torch.cuda.device(get_rank()):
            self.network = tcnn.Network(dim_in, dim_out, config)

    def forward(self, x):
        return self.network(x).float()  # transform to float32


def get_mlp(n_input_dims, n_output_dims, config) -> nn.Module:
    network: nn.Module
    if config.otype == "VanillaMLP":
        network = VanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    else:
        assert (
            config.get("sphere_init", False) is False
        ), "sphere_init=True only supported by VanillaMLP"
        network = TCNNNetwork(n_input_dims, n_output_dims, config_to_primitive(config))
    return network


class NetworkWithInputEncoding(nn.Module, Updateable):
    def __init__(self, encoding, network):
        super().__init__()
        self.encoding, self.network = encoding, network

    def forward(self, x):
        return self.network(self.encoding(x))


class TCNNNetworkWithInputEncoding(nn.Module):
    def __init__(
        self,
        n_input_dims: int,
        n_output_dims: int,
        encoding_config: dict,
        network_config: dict,
    ) -> None:
        super().__init__()
        with torch.cuda.device(get_rank()):
            self.network_with_input_encoding = tcnn.NetworkWithInputEncoding(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                encoding_config=encoding_config,
                network_config=network_config,
            )

    def forward(self, x):
        return self.network_with_input_encoding(x).float()  # transform to float32


def create_network_with_input_encoding(
    n_input_dims: int, n_output_dims: int, encoding_config, network_config
) -> nn.Module:
    # input suppose to be range [0, 1]
    network_with_input_encoding: nn.Module
    if encoding_config.otype in [
        "VanillaFrequency",
        "ProgressiveBandHashGrid",
    ] or network_config.otype in ["VanillaMLP"]:
        encoding = get_encoding(n_input_dims, encoding_config)
        network = get_mlp(encoding.n_output_dims, n_output_dims, network_config)
        network_with_input_encoding = NetworkWithInputEncoding(encoding, network)
    else:
        network_with_input_encoding = TCNNNetworkWithInputEncoding(
            n_input_dims=n_input_dims,
            n_output_dims=n_output_dims,
            encoding_config=config_to_primitive(encoding_config),
            network_config=config_to_primitive(network_config),
        )
    return network_with_input_encoding