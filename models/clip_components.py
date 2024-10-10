from collections import OrderedDict

import torch
import torch.nn as nn

from models.mlp import MLP
from models.vadapter import VAdapter


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            attn_mask: torch.Tensor = None,
            l_adapter=None,
            config=None,
            if_fusion=None,
            visual_attr_adapter=None,
            visual_obj_adapter=None
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_head
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.l_adapter = l_adapter
        self.config = config
        self.if_fusion = if_fusion
        if visual_attr_adapter is None:
            self.visual_attr_adapter = VAdapter(d_model, d_model // 64)
            self.visual_obj_adapter = VAdapter(d_model, d_model // 64)
        else:
            self.visual_attr_adapter = visual_attr_adapter
            self.visual_obj_adapter = visual_obj_adapter
        if not l_adapter:
            self.main_attr_weight = nn.Parameter(torch.ones([]))
            self.counterpart_attr_weight = nn.Parameter(torch.ones([]))
            self.main_obj_weight = nn.Parameter(torch.ones([]))
            self.counterpart_obj_weight = nn.Parameter(torch.ones([]))
        if if_fusion:
            self.main_attr_fusion_weight = nn.Parameter(torch.ones([]))
            self.main_obj_fusion_weight = nn.Parameter(torch.ones([]))
            self.visual_fusion_mlp = MLP(d_model * 3, d_model, num_layers=2)

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, input):
        x, text = input
        if text is not None:
            x = x + self.attention(self.ln_1(x))
            if self.l_adapter and self.config.l_adapter_location == 'in':
                x, _ = self.l_adapter((x, text))
            x = x + self.mlp(self.ln_2(x))
            if self.l_adapter:
                x, _ = self.l_adapter((x, text))
            return x, text
        else:
            x_self, x_attr, x_obj, x_same_attr, x_same_obj = x
            x_self = x_self + self.attention(self.ln_1(x_self))
            x_attr = x_attr + self.attention(self.ln_1(x_attr))
            x_obj = x_obj + self.attention(self.ln_1(x_obj))
            x_same_attr = x_same_attr + self.attention(self.ln_1(x_same_attr))
            x_same_obj = x_same_obj + self.attention(self.ln_1(x_same_obj))
            if self.if_fusion and self.config.v_adapter_location == 'in':
                x_attr = x_attr + self.main_attr_weight * self.visual_attr_adapter(
                    x_attr, x_same_attr
                )
                x_same_attr = (
                        x_same_attr
                        + self.counterpart_attr_weight
                        * self.visual_attr_adapter(x_same_attr, x_attr)
                )
                x_obj = x_obj + self.main_obj_weight * self.visual_obj_adapter(
                    x_obj, x_same_obj
                )
                x_same_obj = (
                        x_same_obj
                        + self.counterpart_obj_weight
                        * self.visual_obj_adapter(x_same_obj, x_obj)
                )
                x_self = (
                        x_self
                        + self.main_attr_fusion_weight * x_attr
                        + self.main_obj_fusion_weight * x_obj
                )
            x_self = x_self + self.mlp(self.ln_2(x_self))
            x_attr = x_attr + self.mlp(self.ln_2(x_attr))
            x_obj = x_obj + self.mlp(self.ln_2(x_obj))
            x_same_attr = x_same_attr + self.mlp(self.ln_2(x_same_attr))
            x_same_obj = x_same_obj + self.mlp(self.ln_2(x_same_obj))
            if self.if_fusion:
                x_attr = x_attr + self.main_attr_weight * self.visual_attr_adapter(
                    x_attr, x_same_attr
                )
                x_same_attr = (
                        x_same_attr
                        + self.counterpart_attr_weight
                        * self.visual_attr_adapter(x_same_attr, x_attr)
                )
                x_obj = x_obj + self.main_obj_weight * self.visual_obj_adapter(
                    x_obj, x_same_obj
                )
                x_same_obj = (
                        x_same_obj
                        + self.counterpart_obj_weight
                        * self.visual_obj_adapter(x_same_obj, x_obj)
                )
                x_self = (
                        x_self
                        + self.main_attr_fusion_weight * x_attr
                        + self.main_obj_fusion_weight * x_obj
                )
            return [x_self, x_attr, x_obj, x_same_attr, x_same_obj], text


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            attn_mask: torch.Tensor = None,
            l_adapter=None,
            config=None,
            visual_attr_adapter=None,
            visual_obj_adapter=None
    ):
        super().__init__()
        self.config = config
        self.width = width
        self.layers = layers
        if not l_adapter:
            res_layers = []
            i = 0
            for _ in range(layers):
                if i >= layers - config.v_adapter_layers and config.has_v_adapter:
                    if_fusion = True
                else:
                    if_fusion = False
                res_layers.append(
                    ResidualAttentionBlock(
                        width,
                        heads,
                        attn_mask=attn_mask,
                        if_fusion=if_fusion,
                        visual_attr_adapter=visual_attr_adapter,
                        visual_obj_adapter=visual_obj_adapter
                    )
                )
                i += 1
            self.resblocks = nn.Sequential(*res_layers)
        else:
            if config:
                indice = (
                    layers - config.l_adater_layers
                    if layers - config.l_adater_layers > 0
                    else 0
                )
            else:
                indice = layers - 3
            res_layers = []
            i = 0
            for _ in range(layers):
                if i < indice or not config.l_adapter_context:
                    res_layers.append(ResidualAttentionBlock(width, heads, attn_mask))
                else:
                    res_layers.append(
                        ResidualAttentionBlock(
                            width,
                            heads,
                            attn_mask,
                            l_adapter=l_adapter,
                            config=config,
                        )
                    )
                i += 1
            self.resblocks = nn.Sequential(*res_layers)

    def forward(self, x):
        return self.resblocks(x)[0]


class VisionTransformer(nn.Module):
    def __init__(
            self,
            input_resolution: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            output_dim: int,
            config,
            visual_attr_adapter=None,
            visual_obj_adapter=None
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width,
            layers,
            heads,
            config=config,
            visual_attr_adapter=visual_attr_adapter,
            visual_obj_adapter=visual_obj_adapter
        )
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.attr_proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.obj_proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def before_forward(self, x):
        x = self.conv1(
            x
        )
        x = x.reshape(
            x.shape[0], x.shape[1], -1
        )
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding.to(
            x.dtype
        )
        x = self.ln_pre(x)

        x = x.permute(
            1, 0, 2
        )
        return x

    def forward(self, x: torch.Tensor):
        x_self, x_same_attr, x_same_obj = x
        x_self = self.before_forward(x_self)
        x_same_attr = self.before_forward(x_same_attr)
        x_same_obj = self.before_forward(x_same_obj)
        x_attr = x_self.clone()
        x_obj = x_self.clone()
        x_self, x_attr, x_obj, x_same_attr, x_same_obj = self.transformer(
            ([x_self, x_attr, x_obj, x_same_attr, x_same_obj], None)
        )
        x_self = x_self.permute(1, 0, 2)
        x_attr = x_attr.permute(1, 0, 2)
        x_obj = x_obj.permute(1, 0, 2)
        x_same_attr = x_same_attr.permute(1, 0, 2)
        x_same_obj = x_same_obj.permute(1, 0, 2)
        x_self = self.ln_post(x_self[:, 0, :])
        x_attr = self.ln_post(x_attr[:, 0, :])
        x_obj = self.ln_post(x_obj[:, 0, :])
        x_same_attr = self.ln_post(x_same_attr[:, 0, :])
        x_same_obj = self.ln_post(x_same_obj[:, 0, :])
        if self.proj is not None:
            x_self = x_self @ self.proj
            x_attr = x_attr @ self.attr_proj
            x_obj = x_obj @ self.obj_proj
            x_same_attr = x_same_attr @ self.attr_proj
            x_same_obj = x_same_obj @ self.obj_proj
        return [x_self, x_attr, x_obj, x_same_attr, x_same_obj]
