import torch

from diffusers import StableVideoDiffusionPipeline, DDIMScheduler
# from diffusers.src.diffusers.pipleines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing
from diffusers.utils import load_image, export_to_video, export_to_gif

from PIL import Image
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field

import torch
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *
from threestudio.models.prompt_processors.base import PromptProcessorOutput

import numpy as np


@threestudio.register("svd-guidance")
class StableVideoDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = None
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        grad_clip: Optional[
            Any
        ] = None 
        half_precision_weights: bool = True

        min_guidance_scale: float = 1.0
        max_guidance_scale: float = 3.0

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.98
        anneal_start_step: Optional[int] = None

        width: Optional[int] = 512
        height: Optional[int] = 512

        weighting_strategy: str = "sds"

        step_ratio: Optional[float] = None
        low_ram_vae: int = -1

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Video Diffusion ...")

        self.dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        # Create model
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, torch_dtype=torch.float16, variant="fp16"
        )
        pipe.to(self.device)

        self.pipe = pipe

        self.num_train_timesteps = self.pipe.scheduler.config.num_train_timesteps if self.cfg.weighting_strategy == 'sds' else 25
        self.pipe.scheduler.set_timesteps(self.num_train_timesteps, device=self.device)  # set sigma for euler discrete scheduling

        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = None
        self.image = None
        self.target_cache = None

    def encode_image(self, image):
        image = image * 2 -1
        if self.cfg.low_ram_vae > 0:
            vnum = self.cfg.low_ram_vae
            mask_vae = torch.randperm(image.shape[0]) < vnum
            with torch.no_grad():
                latents_mask = torch.cat(
                    [
                        self.pipe._encode_vae_image(
                            image[~mask_vae][i : i + 1],
                            self.device, num_videos_per_prompt=1, do_classifier_free_guidance=False,
                        )
                        for i in range(image.shape[0] - vnum)
                    ],
                    dim=0,
                )
            latents = torch.cat(
                [
                    self.pipe._encode_vae_image(
                            image[mask_vae][i : i + 1],
                            self.device, num_videos_per_prompt=1, do_classifier_free_guidance=False,
                        )
                    for i in range(vnum)
                ],
                dim=0,
            )
            latents_full = torch.zeros(
                image.shape[0],
                *latents.shape[1:],
                device=latents.device,
                dtype=latents.dtype,
            )
            latents_full[~mask_vae] = latents_mask
            latents_full[mask_vae] = latents
            latents = self.pipe.vae.config.scaling_factor * latents_full
        else:
            latents = self.pipe._encode_vae_image(image, self.device, num_videos_per_prompt=1, do_classifier_free_guidance=False)
            latents = self.pipe.vae.config.scaling_factor * latents
        return latents
    
    def embed_image(self, image, num_videos_per_prompt=1, do_classifier_free_guidance=True):
        dtype = next(self.pipe.image_encoder.parameters()).dtype

        # We normalize the image before resizing to match with the original implementation.
        # Then we unnormalize it after resizing.
        image = image * 2.0 - 1.0
        image = self._resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0

        # Normalize the image with for CLIP input
        image = self.pipe.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=self.device, dtype=dtype)
        image_embeddings = self.pipe.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])
        return image_embeddings
 
    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        rgb_as_latents: bool = False,
        num_frames: int = 14,
        **kwargs,
    ):
        rgb = rgb.permute(0, 3, 1, 2)
        batch_size = rgb.shape[0]
        image_embeddings = self.embed_image(rgb[[0]])
        # rgb = rgb.to(self.dtype)
        # interp to 512x512 to be fed into vae.
        rgb_512 = F.interpolate(rgb, (self.cfg.height, self.cfg.width), mode="bilinear", align_corners=False)
        # encode image into latents with vae, requires grad!
        # latents = self.pipe._encode_image(rgb_512, self.device, num_videos_per_prompt=1, do_classifier_free_guidance=True)
        latents = self.encode_image(rgb_512.to(self.dtype))
        latents = latents.unsqueeze(0)
        image_cond = rgb_512[[0]]

        if self.cfg.step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(self.cfg.step_ratio)
            t = np.round((1 - self.cfg.step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((1,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (1,), dtype=torch.long, device=self.device)

        w = (1 - self.alphas[t]).view(1, 1, 1, 1)


        if self.cfg.weighting_strategy == 'sds':
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                t = self.num_train_timesteps - t.item()
                # add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.pipe.scheduler.add_noise(latents, noise, self.pipe.scheduler.timesteps[t:t+1]) # t=0 noise;t=999 clean
                noise_pred = self.pipe(
                    image=image_cond,
                    image_embeddings=image_embeddings, 
                    height=self.cfg.height,
                    width=self.cfg.width,
                    latents=latents_noisy,
                    output_type='noise', 
                    denoise_beg=t,
                    denoise_end=t + 1,
                    min_guidance_scale=self.cfg.min_guidance_scale,
                    max_guidance_scale=self.cfg.max_guidance_scale,
                    num_frames=batch_size,
                    num_inference_steps=self.num_train_timesteps
                ).frames[0]
            
            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[1]
            return {
                "loss_sds_video": loss,
                "grad_norm": grad.norm(),
            }

    def _resize_with_antialiasing(self, input, size, interpolation="bicubic", align_corners=True):
        h, w = input.shape[-2:]
        factors = (h / size[0], w / size[1])

        # First, we have to determine sigma
        # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
        sigmas = (
            max((factors[0] - 1.0) / 2.0, 0.001),
            max((factors[1] - 1.0) / 2.0, 0.001),
        )

        # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
        # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
        # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
        ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

        # Make sure it is odd
        if (ks[0] % 2) == 0:
            ks = ks[0] + 1, ks[1]

        if (ks[1] % 2) == 0:
            ks = ks[0], ks[1] + 1

        input = self._gaussian_blur2d(input, ks, sigmas)

        output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
        return output

    def _compute_padding(self, kernel_size):
        """Compute padding tuple."""
        # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
        if len(kernel_size) < 2:
            raise AssertionError(kernel_size)
        computed = [k - 1 for k in kernel_size]

        # for even kernels we need to do asymmetric padding :(
        out_padding = 2 * len(kernel_size) * [0]

        for i in range(len(kernel_size)):
            computed_tmp = computed[-(i + 1)]

            pad_front = computed_tmp // 2
            pad_rear = computed_tmp - pad_front

            out_padding[2 * i + 0] = pad_front
            out_padding[2 * i + 1] = pad_rear

        return out_padding


    def _filter2d(self, input, kernel):
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

        tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

        height, width = tmp_kernel.shape[-2:]

        padding_shape: list[int] = self._compute_padding([height, width])
        input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

        # kernel and input tensor reshape to align element-wise or batch-wise params
        tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
        input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

        # convolve the tensor with the kernel.
        output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

        out = output.view(b, c, h, w)
        return out


    def _gaussian(self, window_size: int, sigma):
        if isinstance(sigma, float):
            sigma = torch.tensor([[sigma]])

        batch_size = sigma.shape[0]

        x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

        if window_size % 2 == 0:
            x = x + 0.5

        gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

        return gauss / gauss.sum(-1, keepdim=True)


    def _gaussian_blur2d(self, input, kernel_size, sigma):
        if isinstance(sigma, tuple):
            sigma = torch.tensor([sigma], dtype=input.dtype)
        else:
            sigma = sigma.to(dtype=input.dtype)

        ky, kx = int(kernel_size[0]), int(kernel_size[1])
        bs = sigma.shape[0]
        kernel_x = self._gaussian(kx, sigma[:, 1].view(bs, 1))
        kernel_y = self._gaussian(ky, sigma[:, 0].view(bs, 1))
        out_x = self._filter2d(input, kernel_x[..., None, :])
        out = self._filter2d(out_x, kernel_y[..., None])

        return out
