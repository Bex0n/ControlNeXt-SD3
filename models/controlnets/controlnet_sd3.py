# Copyright 2025 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.controlnets.controlnet import zero_module
from diffusers.models.embeddings import (CombinedTimestepTextProjEmbeddings,
                                         PatchEmbed)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SD3ControlNeXtEncoder(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    A lightweight ControlNeXt-style encoder for Stable Diffusion 3.

    This model takes a control condition image and processes it through a shallow
    transformer to produce a single feature tensor suitable for injection into
    the main SD3Transformer2DModel.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 1, # Using 1 transformer block for a lightweight model
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        extra_conditioning_channels: int = 3,
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim

        # --- Key Change 2: Simplified input embedding ---
        self.control_image_embed = zero_module(
            PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels + extra_conditioning_channels,
                embed_dim=self.inner_dim,
            )
        )

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)

        # The core processing blocks
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        controlnet_cond: torch.Tensor,
        timestep: torch.LongTensor,
        pooled_projections: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Embed the control image condition into a sequence of tokens
        control_tokens = self.control_image_embed(controlnet_cond)

        # 2. Prepare timestep and text embeddings
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 3. Process the control tokens through the transformer block(s)
        for block in self.transformer_blocks:
            _, control_tokens = block(
                hidden_states=control_tokens,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

        return control_tokens