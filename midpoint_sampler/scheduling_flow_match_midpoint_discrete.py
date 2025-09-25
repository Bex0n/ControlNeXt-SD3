# This should go to diffusers library. Remember to change related __init__.py files.
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMatchMidpointDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchMidpointDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Heun scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 2

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        self.num_inference_steps = num_inference_steps

        # 1. Create the base sigmas from which everything is derived
        timesteps_base = np.linspace(
            self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps + 1
        )
        sigmas_base = timesteps_base / self.config.num_train_timesteps
        sigmas_base = self.config.shift * sigmas_base / (1 + (self.config.shift - 1) * sigmas_base)
        sigmas_base = torch.from_numpy(sigmas_base).to(dtype=torch.float32, device=device)
        
        # 2. Create the interleaved sigma schedule for the Midpoint method
        sigma_mid = (sigmas_base[:-1] + sigmas_base[1:]) / 2.0
        interleaved_sigmas = torch.zeros(2 * num_inference_steps, device=device, dtype=torch.float32)
        interleaved_sigmas[0::2] = sigmas_base[:-1]
        interleaved_sigmas[1::2] = sigma_mid
        
        # 3. The final sigmas list includes the final 0.0 for the last step
        self.sigmas = torch.cat([interleaved_sigmas, torch.zeros(1, device=device)])
        
        # 4. The timesteps for the pipeline loop are derived from sigmas, excluding the final 0.0 value
        self.timesteps = self.sigmas[:-1] * self.config.num_train_timesteps

        # 5. Reset state
        self.dt = None
        self.sample = None
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    @property
    def state_in_first_order(self):
        return self.dt is None

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchMidpointDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_flow_match_heun_discrete.FlowMatchHeunDiscreteSchedulerOutput`] tuple.

        Returns:
            [`~schedulers.scheduling_flow_match_heun_discrete.FlowMatchHeunDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_flow_match_heun_discrete.FlowMatchHeunDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchHeunDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        if self.state_in_first_order:
            sigma = self.sigmas[self.step_index]
            sigma_next_full = self.sigmas[self.step_index + 2]
        else: # Second stage
            sigma = self.sigmas[self.step_index - 1]
            sigma_mid = self.sigmas[self.step_index]
        # sigma = self.sigmas[self.step_index]
        # sigma_next = self.sigmas[self.step_index + 1]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            noise = randn_tensor(
                model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
            )
            eps = noise * s_noise
            if self.state_in_first_order:
                sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        if self.state_in_first_order:
            # 1. Calculate derivative (k1) at the start point.
            denoised = sample - model_output * sigma
            derivative = (sample - denoised) / sigma_hat
            
            # 2. Calculate full and half step sizes.
            dt = sigma_next_full - sigma_hat
            
            # 3. Store original sample and full dt for the second stage.
            self.sample = sample
            self.dt = dt
            
            # 4. Take a half-step forward and return the midpoint sample.
            prev_sample = sample + derivative * (dt * 0.5)

        else:
            # 1. Calculate the derivative at the midpoint (k_mid).
            denoised = sample - model_output * sigma_mid
            midpoint_derivative = (sample - denoised) / sigma_mid
            
            # 2. Retrieve the original sample and full step size.
            dt = self.dt
            original_sample = self.sample
            
            # 3. Take a FULL step from the ORIGINAL sample using the MIDPOINT derivative.
            prev_sample = original_sample + midpoint_derivative * dt
            
            # 4. Reset state for the next pair of steps.
            self.dt = None
            self.sample = None
        # denoised = sample - model_output * sigma
        # derivative = (sample - denoised) / sigma_hat
        # dt = sigma_next - sigma_hat
        # prev_sample = sample + derivative * dt


        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchMidpointDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
