"""Functions and classes for loading and handling models conveniently."""
import contextlib
from typing import Union, Dict, Optional

import torch
from torch import Tensor

from xreal.ldm.models.autoencoder import AutoencoderKL
from xreal.ldm.models.diffusion.ddpm import LatentDiffusion
from xreal.ldm.models.diffusion.ddim import DDIMSampler
import albumentations as A
import cv2
import albumentations.pytorch
import numpy as np

class xrealAEModel:
    def __init__(
            self,
            model_path: str,
            device: Union[str, int, torch.device] = 'cuda'
    ) -> None:
        self.device = device

        with contextlib.redirect_stdout(None):
            self.model = AutoencoderKL(
                embed_dim=3,
                ckpt_path=model_path,
                ddconfig={
                    'double_z': True,
                    'z_channels': 3,
                    'resolution': 256,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': (1, 2, 4),
                    'num_res_blocks': 2,
                    'attn_resolutions': [],
                    'dropout': 0.0
                },
                lossconfig={'target': 'torch.nn.Identity'}
            )
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        return self.model.encode(x).mode()

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        return self.model.decode(z)


class xrealLDM:
    def __init__(
            self,
            model_path: str,
            ae_path: Optional[str] = None,
            device: Union[str, int, torch.device] = 'cuda'
    ) -> None:
        self.device = device
        with contextlib.redirect_stdout(None):
            self.model = self._init_checkpoint(model_path, ae_path)

        self.model = self.model.to(self.device)
        self.model.model = self.model.model.to(self.device)
        self.model.eval()

        self.sample_shape = [
            self.model.model.diffusion_model.in_channels,
            self.model.model.diffusion_model.image_size,
            self.model.model.diffusion_model.image_size
        ]
        self.anatomy_mask_resize = A.Compose([
            A.augmentations.geometric.resize.Resize(256, 256, interpolation=cv2.INTER_CUBIC),
        ])

        self.pathology_mask_resize = A.Compose([
            A.augmentations.geometric.resize.Resize(64, 64, interpolation=cv2.INTER_CUBIC),
        ])
        self.soft_scale = 0.2


    @torch.no_grad()
    def sample(
            self,
            batch_size: int = 1,
            sampling_steps: int = 100,
            eta: float = 1.0,
            decode: bool = True,
            *args,
            **kwargs
    ) -> Tensor:
        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps, batch_size=batch_size, shape=self.sample_shape, eta=eta, verbose=False
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    @torch.no_grad()
    def sample_spatial_input(
            self,
            target_img: Tensor,
            mask: Tensor,
            sampling_steps: int = 100,
            eta: float = 1.0,
            decode: bool = True,
            *args,
            **kwargs
    ) -> Tensor:
        # Encode original image via AE
        target_enc = self.model.encode_first_stage(target_img).mode()

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            batch_size=target_img.shape[0],
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            mask=mask,
            x0=target_enc
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    def _init_checkpoint(
            self, model_path: str, ae_path: Optional[str] = None
    ) -> LatentDiffusion:
        config_dict = self._get_config_dict(ae_path)
        model = LatentDiffusion(**config_dict)

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict['state_dict'], strict=False)
        return model

    @staticmethod
    def _get_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            'linear_start': 0.0015,
            'linear_end': 0.0295,
            'num_timesteps_cond': 1,
            'log_every_t': 200,
            'timesteps': 1000,
            'first_stage_key': 'image',
            'image_size': 64,
            'channels': 3,
            'monitor': 'val/loss_simple_ema',
            'unet_config': xrealLDM._get_unet_config_dict(),
            'first_stage_config': xrealLDM._get_first_stage_config_dict(ae_path),
            'cond_stage_config': '__is_unconditional__'
        }

    @staticmethod
    def _get_unet_config_dict() -> Dict:
        return {
            'target': 'xreal.ldm.modules.diffusionmodules.openaimodel.UNetModel',
            'params': {
                'image_size': 64,
                'in_channels': 3,
                'out_channels': 3,
                'model_channels': 224,
                'attention_resolutions': [8, 4, 2],
                'num_res_blocks': 2,
                'channel_mult': [1, 2, 3, 4],
                'num_head_channels': 32,
            }
        }

    @staticmethod
    def _get_first_stage_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            'target': 'xreal.ldm.models.autoencoder.AutoencoderKL',
            'params': {
                'embed_dim': 3,
                'ckpt_path': ae_path,
                'ddconfig': {
                    'double_z': True,
                    'z_channels': 3,
                    'resolution': 256,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': (1, 2, 4),
                    'num_res_blocks': 2,
                    'attn_resolutions': [],
                    'dropout': 0.0
                },
                'lossconfig': {'target': 'torch.nn.Identity'}
            }
        }


class xRealModel(xrealLDM):
    def __init__(
            self, 
            model_path: str,
            ae_path: str = None,
            anatomy_controller_path: str = None,
            device: Union[str, int, torch.device] = 'cuda'

            ):
        super().__init__(model_path, ae_path, device)
        # init mask to image ae
        self.anatomy_controller = xrealAEModel(model_path=anatomy_controller_path, device=device)
        
        self.resize_256 = A.Compose([
            A.augmentations.geometric.resize.Resize(256, 256, interpolation=cv2.INTER_CUBIC),
        ])

        self.resize_64 = A.Compose([
            A.augmentations.geometric.resize.Resize(64, 64, interpolation=cv2.INTER_CUBIC),
        ])
        self.soft_scale = 0.2
        self.device = device

    @torch.no_grad()
    def sample(
            self,
            sampling_steps: int = 100,
            eta: float = 1.0,
            decode: bool = True,
            conditioning: str = '',
            *args,
            **kwargs,
    ) -> Tensor:
        conditioning = self.model.get_learned_conditioning(conditioning)

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            conditioning=conditioning,
            batch_size=1,
            shape=self.sample_shape,
            eta=eta,
            verbose=False
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    @torch.no_grad()
    def sample_spatial_input(
            self,
            target_img: Tensor,
            mask: Tensor,
            sampling_steps: int = 100,
            eta: float = 1.0,
            decode: bool = True,
            conditioning: str = '',
            return_intermediates: bool = False,
            unconditional_guidance_scale: float = 1.0,
            unconditional_conditioning: str = None,
            prior: bool = False,
            mask_timesteps = None ,
            use_mask_linear_scale = False,
            mask_polynomial = None,
            *args,
            **kwargs
    ) -> Tensor:
        assert target_img.shape[0] == 1, 'Method implemented only for batch size = 1.'

        if not prior:
            # Encode original image via AE
            target_enc = self.model.encode_first_stage(target_img).sample() * self.model.scale_factor
        else:
            target_enc = target_img
        conditioning = self.model.get_learned_conditioning(conditioning)
        if unconditional_conditioning is not None:
            unconditional_conditioning = self.model.get_learned_conditioning(unconditional_conditioning)

        ddim = DDIMSampler(self.model)
        samples, intermediates = ddim.sample(
            sampling_steps,
            conditioning=conditioning,
            batch_size=1,
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            mask=mask,
            x0=target_enc,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            mask_timesteps=mask_timesteps,
            use_mask_linear_scale=use_mask_linear_scale,
            mask_polynomial=mask_polynomial
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        if return_intermediates:
            return samples, intermediates
        return samples


    @torch.no_grad()
    def sample_controlled(
            self,
            anatomy_mask: Tensor,
            pathology_mask: Tensor,
            pathology: str,
            sampling_steps: int = 100,
            unconditional_guidance_scale = 7.0,
    ):
        
        """
            Args:
                anatomy_mask: Tensor, shape (256, 256) --> resized to (1, 1, 256, 256)
                pathology_mask: Tensor, shape (64, 64)
                pathology: str, pathology to be added to the image
            
            Returns:
                x_a_gen: Tensor, shape (1, 3, 256, 256)
                x_p_gen: Tensor, shape (1, 3, 256, 256)
        """

        print('Generating Image...')

        # self.pathology_mask_resize
        anatomy_mask = torch.tensor(anatomy_mask).unsqueeze(0)  # expand dimensions to (1, H, W)
        anatomy_mask = anatomy_mask.repeat(3, 1, 1)
        if anatomy_mask.max() >= 1:
            anatomy_mask = anatomy_mask/4
        
        # for LDM 
        anatomy_mask_smaller = self.resize_64(image = anatomy_mask.permute(1,2,0).float().detach().cpu().numpy())['image'] # 64, 64
        anatomy_mask_smaller = np.where(anatomy_mask_smaller > 0, 1, self.soft_scale)
        anatomy_mask_smaller = torch.from_numpy(anatomy_mask_smaller).permute(2,0,1).to(torch.float32)

        # for anatomy controller
        anatomy_mask = self.resize_256(image = anatomy_mask.permute(1,2,0).float().detach().cpu().numpy())['image']     
        anatomy_mask = torch.from_numpy(anatomy_mask).permute(2,0,1).to(torch.float32)

        z_x_hat = self.anatomy_controller.encode(anatomy_mask.float().unsqueeze(0).to(self.device))
        x_hat = self.anatomy_controller.decode(z_x_hat)
        x_hat.clamp_(0, 1)

        pathology_mask = torch.tensor(pathology_mask).unsqueeze(0)  # expand dimensions to (1, H, W)
        # pathology_mask = pathology_mask.repeat(3, 1, 1)
        if pathology_mask.shape != (1,64,64):
            pathology_mask = self.resize_64(image = pathology_mask.unsqueeze(0).permute(1,2,0).float().detach().cpu().numpy())['image']
            pathology_mask = torch.from_numpy(pathology_mask).permute(2,0,1).unsqueeze(0).to(self.device)
        
        # anatomy control with no disease
        x_a_gen = self.sample_spatial_input(
            target_img=x_hat,
            mask=anatomy_mask_smaller.to(self.device),
            sampling_steps=sampling_steps,
            conditioning = "",
            mask_timesteps=50,
        )
        x_a_gen.clamp_(0, 1)

        # pathology control
        x_p_gen = self.sample_spatial_input(
            target_img=x_a_gen,
            mask=(1-pathology_mask).unsqueeze(0).to(self.device),
            sampling_steps=sampling_steps,
            conditioning=pathology,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning="",
        )
        x_p_gen.clamp_(0, 1)

        del x_a_gen
        return x_p_gen[0].detach().cpu().numpy().transpose(1, 2, 0)


    @staticmethod
    def _get_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            'linear_start': 0.0015,
            'linear_end': 0.0295,
            'num_timesteps_cond': 1,
            'log_every_t': 200,
            'timesteps': 1000,
            'first_stage_key': 'image',
            'cond_stage_key': 'caption',
            'image_size': 64,
            'channels': 3,
            'cond_stage_trainable': True,
            'conditioning_key': 'crossattn',
            'monitor': 'val/loss_simple_ema',
            'scale_factor': 0.18215,
            'unet_config': xRealModel._get_unet_config_dict(),
            'first_stage_config': xRealModel._get_first_stage_config_dict(ae_path),
            'cond_stage_config': xRealModel._get_cond_config_dict()
        }

    @staticmethod
    def _get_cond_config_dict() -> Dict:
        return {
            'target': 'xreal.ldm.modules.encoders.modules.BERTEmbedder',
            'params': {
                'n_embed': 1280,
                'n_layer': 32,
            }
        }

    @staticmethod
    def _get_unet_config_dict() -> Dict:
        return {
            'target': 'xreal.ldm.modules.diffusionmodules.openaimodel.UNetModel',
            'params': {
                'image_size': 64,
                'in_channels': 3,
                'out_channels': 3,
                'model_channels': 224,
                'attention_resolutions': [8, 4, 2],
                'num_res_blocks': 2,
                'channel_mult': [1, 2, 4, 4],
                'num_heads': 8,
                'use_spatial_transformer': True,
                'transformer_depth': 1,
                'context_dim': 1280,
                'use_checkpoint': True,
                'legacy': False,
            }
        }
