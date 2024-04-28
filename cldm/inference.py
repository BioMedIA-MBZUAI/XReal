"""Functions and classes for loading and handling models conveniently."""
import contextlib
from typing import Union, Dict, Optional

import torch
from torch import Tensor

from cldm.cldm import ControlLDM
from cldm.ddim_hacked import DDIMSampler

from omegaconf import OmegaConf


class ControlNetModel:
    def __init__(
            self,
            model_path: str,
            # ae_path: Optional[str] = None,
            device: Union[str, int, torch.device] = 'cuda',
            config_path: str = '../configs/xreal-diff-t2i_cnet.yml'
    ) -> None:
        self.device = device
        self.model_config_dict = OmegaConf.load(config_path)['model']['params']
        with contextlib.redirect_stdout(None):
            self.model = self._init_checkpoint(model_path)

        self.model = self.model.to(self.device)
        self.model.model = self.model.model.to(self.device)
        self.model.eval()

        # TODO: remove paths

        self.sample_shape = [
            self.model.model.diffusion_model.in_channels,
            self.model.model.diffusion_model.image_size,
            self.model.model.diffusion_model.image_size
        ]

    def _init_checkpoint(
            self, model_path: str
    ) -> ControlLDM:
        # config_dict = self._get_config_dict(ae_path)
        config_dict = self.model_config_dict
        print(f'Initializing model')
        model = ControlLDM(**config_dict)

        print(f'Loading checkpoint from {model_path}')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict['state_dict'], strict=False)
        return model

    @torch.no_grad()
    def sample(
            self,
            sampling_steps: int = 100,
            eta: float = 1.0,
            decode: bool = True,
            conditioning: str = '',
            input_mask = None,
            *args,
            **kwargs,
    ) -> Tensor:
        if kwargs['batch_size'] is None:
            kwargs['batch_size'] = 1
        conditioning_emb = self.model.get_learned_conditioning(conditioning)

        conditioning_input = {
            'c_concat':[input_mask.to(self.device)],
            'c_crossattn': [conditioning_emb.to(self.device)],
        }

        ddim = DDIMSampler(self.model)
        
        samples, _ = ddim.sample(
            sampling_steps,
            conditioning=conditioning_input,
            batch_size=kwargs['batch_size'],
            shape=self.sample_shape,
            eta=eta,
            verbose=False
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples
