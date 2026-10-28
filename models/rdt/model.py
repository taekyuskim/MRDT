# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from collections import OrderedDict

import torch
import torch.nn as nn

from models.rdt.blocks import (FinalLayer, RDTBlock, TimestepEmbedder,
                               get_1d_sincos_pos_embed_from_grid,
                               get_multimodal_cond_pos_embed,
                               get_nd_sincos_pos_embed_from_grid)


class RDT(nn.Module):
    """
    Class for Robotics Diffusion Transformers.
    """
    def __init__(
        self,
        output_dim=128,
        horizon=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        max_lang_cond_len=1024,
        img_cond_len=4096,
        lang_pos_embed_config=None,
        img_pos_embed_config=None,
        dtype=torch.bfloat16,
        mlp_ratio=1.0,
        cond_injection='token',
    ):
        """cond_injection: 'token' (original RDT: timestep, control frequency and state are
        prepended tokens) or 'adaln_single' (PixArt-alpha: timestep + an external condition
        vector modulate every block; the sequence holds only the target tokens [+ extra tokens])."""
        super().__init__()
        if cond_injection not in ('token', 'adaln_single'):
            raise ValueError(f'Unknown cond_injection {cond_injection!r}')
        self.cond_injection = cond_injection
        adaln = cond_injection == 'adaln_single'
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.dtype = dtype
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config

        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = None if adaln else TimestepEmbedder(hidden_size, dtype=dtype)
        if adaln:  # one global modulation MLP shared by all blocks (adaLN-single)
            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        
        # We will use trainable sin-cos embeddings
        # [timestep; state; action] (token) / [action] (adaln_single)
        self.x_pos_embed = nn.Parameter(
            torch.zeros(1, horizon if adaln else horizon+3, hidden_size))
        # Language conditions
        self.lang_cond_pos_embed = nn.Parameter(
            torch.zeros(1, max_lang_cond_len, hidden_size))
        # Image conditions. A config of one image group whose first axis is negative
        # (e.g. ('image', (-5, 8, 8))) has no slot axis: the views form an unordered set,
        # so a single per-view embedding is stored and shared by every slot (tied, also
        # after training).
        self.img_pos_shared_slots = self._shared_slots(img_pos_embed_config, img_cond_len)
        self.img_cond_pos_embed = nn.Parameter(
            torch.zeros(1, img_cond_len // (self.img_pos_shared_slots or 1), hidden_size))

        self.blocks = nn.ModuleList([
            RDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, adaln=adaln) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, output_dim, adaln=adaln)
        self.initialize_weights()

    @staticmethod
    def _shared_slots(config, img_cond_len):
        if config is None or len(config) != 1:
            return None
        sizes = config[0][1]
        if not isinstance(sizes, (tuple, list)) or len(sizes) < 2 or sizes[0] >= 0:
            return None
        slots = -int(sizes[0])
        if img_cond_len % slots:
            raise ValueError('img_cond_len must be a multiple of the number of slots')
        return slots

    def _img_pos(self):
        if self.img_pos_shared_slots:
            return self.img_cond_pos_embed.repeat(1, self.img_pos_shared_slots, 1)
        return self.img_cond_pos_embed

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding
        if self.cond_injection == 'adaln_single':
            # Only target tokens: a full-width 2-D sin/cos over the square patch grid.
            side = int(round(self.horizon ** 0.5))
            if side * side != self.horizon:
                raise ValueError('adaln_single expects a square target token grid')
            x_pos_embed = get_nd_sincos_pos_embed_from_grid(self.hidden_size, (side, side)).reshape(self.horizon, -1)
        else:
            x_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict([
                    ('timestep', 1),
                    ('ctrl_freq', 1),
                    ('state', 1),
                    ('action', self.horizon),
                ])
            )
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        if self.lang_pos_embed_config is None:
            lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.max_lang_cond_len))
        else:
            lang_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.lang_pos_embed_config),
                embed_modality=False
            )
        self.lang_cond_pos_embed.data.copy_(
            torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))
        
        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.img_cond_len))
        else:
            img_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.img_pos_embed_config),
                embed_modality=False
            )
        if self.img_pos_shared_slots:  # every slot has the same rows; keep one copy
            img_cond_pos_embed = img_cond_pos_embed.reshape(self.img_pos_shared_slots, -1, self.hidden_size)[0]
        self.img_cond_pos_embed.data.copy_(
            torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))

        # Initialize timestep and control freq embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.freq_embedder is not None:
            nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)
        if self.cond_injection == 'adaln_single':
            # adaLN-Zero: zero modulation -> zero gates -> every block starts as the identity
            nn.init.constant_(self.t_block[1].weight, 0)
            nn.init.constant_(self.t_block[1].bias, 0)
            
        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # Move all the params to given data type:
        self.to(self.dtype)

    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None,
                cond=None, extra=None, extra_mask=None):
        """
        Forward pass of RDT.
        
        x: (B, T, D), state + action token sequence, T = horizon + 1,
            dimension D is assumed to be the same as the hidden size.
        freq: (B,), a scalar indicating control frequency.
        t: (B,) or (1,), diffusion timesteps.
        lang_c: (B, L_lang, D) or None, language condition tokens (variable length),
            dimension D is assumed to be the same as the hidden size.
        img_c: (B, L_img, D) or None, image condition tokens (fixed length),
            dimension D is assumed to be the same as the hidden size.
        lang_mask: (B, L_lang) or None, language condition mask (True for valid).
        img_mask: (B, L_img) or None, image condition mask (True for valid).
        cond: (B, D), adaln_single only: condition vector added to the timestep embedding.
        extra: (B, L_extra, D) or None, extra self-attention tokens appended after x (already
            embedded, e.g. clean context latents); extra_mask: (B, L_extra) True = attend.
        """
        if self.cond_injection == 'adaln_single':
            return self._forward_adaln(x, t, lang_c, img_c, lang_mask, img_mask, cond, extra, extra_mask)
        t = self.t_embedder(t).unsqueeze(1)             # (B, 1, D) or (1, 1, D)
        freq = self.freq_embedder(freq).unsqueeze(1)    # (B, 1, D)
        # Append timestep to the input tokens
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = torch.cat([t, freq, x], dim=1)               # (B, T+1, D)
        
        # Add multimodal position embeddings
        x = x + self.x_pos_embed
        n_main = x.shape[1]
        self_mask = None
        if extra is not None:
            self_mask = torch.cat([torch.ones(x.shape[:2], dtype=torch.bool, device=x.device), extra_mask.bool()], dim=1)
            x = torch.cat([x, extra], dim=1)
        # Note the lang is of variable length
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self._img_pos()

        # Forward pass. 'alternate' (original RDT): even blocks cross-attend language,
        # odd blocks image. 'image': every block cross-attends the image condition.
        if getattr(self, 'cross_attn_mode', 'alternate') == 'image':
            conds, masks = [img_c, img_c], [img_mask, img_mask]
        else:
            conds = [lang_c, img_c]
            masks = [lang_mask, img_mask]
        for i, block in enumerate(self.blocks):
            c, mask = conds[i%2], masks[i%2]
            x = block(x, c, mask) if self_mask is None else block(x, c, mask, self_mask=self_mask)
        # Inject the language condition at the final layer
        x = self.final_layer(x)                         # (B, T+1, out_channels)

        # Only preserve the action tokens
        x = x[:, -self.horizon:] if extra is None else x[:, n_main - self.horizon:n_main]
        return x

    def _forward_adaln(self, x, t, lang_c, img_c, lang_mask, img_mask, cond, extra, extra_mask):
        if cond is None:
            raise ValueError('adaln_single needs the condition vector `cond`')
        B = x.shape[0]
        t = self.t_embedder(t)                          # (B, D) or (1, D)
        if t.shape[0] == 1:
            t = t.expand(B, -1)
        c = t + cond                                    # timestep + condition (DiT: t + y)
        t_mod = self.t_block(c)                         # (B, 6D), shared by all blocks
        x = x + self.x_pos_embed
        n_main = x.shape[1]
        self_mask = None
        if extra is not None:
            self_mask = torch.cat([torch.ones(x.shape[:2], dtype=torch.bool, device=x.device), extra_mask.bool()], dim=1)
            x = torch.cat([x, extra], dim=1)
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self._img_pos()
        if getattr(self, 'cross_attn_mode', 'alternate') == 'image':
            conds, masks = [img_c, img_c], [img_mask, img_mask]
        else:
            conds, masks = [lang_c, img_c], [lang_mask, img_mask]
        for i, block in enumerate(self.blocks):
            x = block(x, conds[i % 2], masks[i % 2], t_mod=t_mod, self_mask=self_mask)
        x = self.final_layer(x, c)
        return x[:, :n_main][:, :self.horizon]
