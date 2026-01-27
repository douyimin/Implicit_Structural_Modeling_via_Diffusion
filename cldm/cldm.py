import einops
import numpy as np
import torch
import torch as th
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
import itertools
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, \
    AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.ema import LitEma  # 复用已有的LitEma
import matplotlib.pyplot as plt
# from lora_diffusion import inject_trainable_lora, extract_lora_ups_down
from contextlib import contextmanager, nullcontext


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range + 1e-6)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=planes)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=planes)
        if inplanes != planes and stride == 2:
            self.res_conv = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
                                          nn.MaxPool2d(2, 2))
        elif inplanes != planes and stride != 2:
            self.res_conv = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes == planes and stride == 2:
            self.res_conv = nn.Sequential(nn.MaxPool2d(2, 2))
        else:
            self.res_conv = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.res_conv is not None:
            residual = self.res_conv(residual)
        out += residual
        out = self.silu(out)
        return out


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            nn.Conv2d(hint_channels, 16, 3, stride=1, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=16),
            nn.SiLU(),
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 96, stride=2),
            BasicBlock(96, 256, stride=2),
            zero_module(nn.Conv2d(256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control,
                 sd_locked=True,  # 主干是否锁定
                 use_control_ema=True,  # 是否对control网络使用EMA
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_horiz = instantiate_from_config(control_stage_config)
        self.control_fault = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.sd_locked = sd_locked
        self.use_control_ema = use_control_ema
        self.control_fault_scales = [1.0] * 13
        self.control_horiz_scales = [1.0] * 13
        self.clip_txt = torch.from_numpy(np.load('clip_txt.npy'))

        # =============== EMA 配置（自动判断） ===============
        #
        # 逻辑：
        # 1. 如果主干锁定(sd_locked=True)，主干参数不更新，对其做EMA没有意义
        #    -> 关闭主干EMA，节省显存
        # 2. 只对实际训练的control网络做EMA
        #
        # ====================================================
        # self.use_ema = True
        # 处理主干EMA：如果主干锁定，关闭其EMA以节省显存
        if self.sd_locked and self.use_ema:
            print("[ControlLDM] sd_locked=True, disabling EMA for frozen backbone to save memory")
            self.use_ema = False
            # 删除父类创建的model_ema以释放显存
            if hasattr(self, 'model_ema'):
                del self.model_ema

        # 处理control网络的EMA

        if self.use_control_ema:
            self.control_fault_ema = LitEma(self.control_fault)
            self.control_horiz_ema = LitEma(self.control_horiz)
            print(f"[ControlLDM] Keeping EMAs for control_fault: {len(list(self.control_fault_ema.buffers()))} params")
            print(f"[ControlLDM] Keeping EMAs for control_horiz: {len(list(self.control_horiz_ema.buffers()))} params")

        # 打印EMA状态总结
        print(f"[ControlLDM] EMA Summary:")
        print(f"  - Backbone (self.model): use_ema={self.use_ema}")
        print(f"  - Control networks: use_control_ema={self.use_control_ema}")
        # ====================================================

    @contextmanager
    def ema_scope(self, context=None):
        """
        上下文管理器：临时切换到EMA参数进行推理

        根据当前配置，自动处理：
        1. 主干EMA（如果 use_ema=True）
        2. control网络EMA（如果 use_control_ema=True）
        """
        # 处理主干EMA（仅当 use_ema=True 且 model_ema 存在时）
        if self.use_ema and hasattr(self, 'model_ema'):
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights for backbone")

        # 处理control网络的EMA
        if self.use_control_ema:
            self.control_fault_ema.store(self.control_fault.parameters())
            self.control_fault_ema.copy_to(self.control_fault)
            self.control_horiz_ema.store(self.control_horiz.parameters())
            self.control_horiz_ema.copy_to(self.control_horiz)
            if context is not None:
                print(f"{context}: Switched to EMA weights for control networks")

        try:
            yield None
        finally:
            # 恢复主干参数
            if self.use_ema and hasattr(self, 'model_ema'):
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights for backbone")

            # 恢复control网络参数
            if self.use_control_ema:
                self.control_fault_ema.restore(self.control_fault.parameters())
                self.control_horiz_ema.restore(self.control_horiz.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights for control networks")

    def on_train_batch_end(self, *args, **kwargs):
        """
        每个训练batch结束后更新EMA

        仅更新实际在训练的模块的EMA
        """
        # 更新主干EMA（仅当主干在训练时）
        if self.use_ema and hasattr(self, 'model_ema'):
            self.model_ema(self.model)

        # 更新control网络的EMA
        if self.use_control_ema:
            self.control_fault_ema(self.control_fault)
            self.control_horiz_ema(self.control_horiz)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        fault_hint = batch['fault']
        horiz_hint = batch['horiz']

        if bs is not None:
            fault_hint = fault_hint[:bs]
            horiz_hint = horiz_hint[:bs]
        c = torch.repeat_interleave(
            self.clip_txt, repeats=fault_hint.shape[0], dim=0).to(fault_hint.dtype).to(fault_hint.device)
        fault_hint = fault_hint.to(self.device)
        horiz_hint = horiz_hint.to(self.device)
        # control = einops.rearrange(control, 'b h w c -> b c h w')
        horiz_hint = horiz_hint.to(memory_format=torch.contiguous_format).float()
        fault_hint = fault_hint.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], fault=fault_hint, horiz=horiz_hint)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        control_horiz = self.control_horiz(x=x_noisy, hint=cond['horiz'], timesteps=t, context=cond_txt)
        control_fault = self.control_fault(x=x_noisy, hint=cond['fault'], timesteps=t, context=cond_txt)

        control = [c1 * s1 + c2 * s2 for c1, c2, s1, s2 in
                   zip(control_horiz, control_fault, self.control_horiz_scales, self.control_fault_scales)]

        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control,
                              only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=1.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        fault_hint, horiz_hint = c['fault'][:N], c['horiz'][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["fault"], log["horiz"] = fault_hint, horiz_hint

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # 使用ema_scope进行采样
            ema_scope = self.ema_scope if use_ema_scope else nullcontext
            with ema_scope("Sampling"):
                # get denoise row
                samples, z_denoise_row = self.sample_log(
                    cond=c,
                    batch_size=N, ddim=use_ddim,
                    ddim_steps=ddim_steps, eta=ddim_eta)
                x_samples = self.decode_first_stage(samples)
                log["samples"] = x_samples
                if plot_denoise_rows:
                    denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                    log["denoise_row"] = denoise_grid

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["horiz"].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_fault.parameters()) + list(self.control_horiz.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    @classmethod
    def load_from_checkpoint_with_ema_to_model(cls, checkpoint_path, config=None, map_location='cpu', strict=True):
        """
        从checkpoint加载模型，将EMA参数加载到主训练网络中

        Args:
            checkpoint_path: checkpoint文件路径
            config: 模型配置（如果为None，则从checkpoint中读取）
            map_location: 加载设备
            strict: 是否严格匹配参数

        Returns:
            加载好的模型实例
        """
        print(f"[ControlLDM] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        state_dict = checkpoint.get('state_dict', checkpoint)

        # 用于存储处理后的state_dict
        new_state_dict = {}

        # 定义EMA到主网络的映射关系
        ema_mappings = {
            'control_fault_ema': 'control_fault',
            'control_horiz_ema': 'control_horiz',
            'model_ema': 'model',
        }

        # 收集EMA参数（LitEma使用shadow_前缀存储）
        ema_params = {}
        for key, value in state_dict.items():
            for ema_name in ema_mappings.keys():
                # LitEma的参数存储格式通常是 "{ema_name}.shadow_{param_name}"
                if key.startswith(f"{ema_name}."):
                    ema_params[key] = value
                    break

        print(f"[ControlLDM] Found {len(ema_params)} EMA parameters in checkpoint")

        # 处理所有参数
        for key, value in state_dict.items():
            is_ema_param = False

            # 检查是否是EMA参数
            for ema_name, model_name in ema_mappings.items():
                if key.startswith(f"{ema_name}."):
                    is_ema_param = True
                    # 保留原始EMA参数（用于恢复EMA状态）
                    new_state_dict[key] = value

                    # 将EMA参数映射到主网络
                    # LitEma的参数格式: "{ema_name}.shadow_{flat_index}" 或直接是buffer名
                    param_suffix = key[len(f"{ema_name}."):]

                    # 如果是shadow_格式的参数，需要特殊处理
                    # LitEma使用扁平化的索引来存储shadow参数
                    # 这里我们直接将EMA参数保留，稍后在实例方法中处理
                    break

            if not is_ema_param:
                new_state_dict[key] = value

        return new_state_dict, ema_params

    def load_ema_to_model(self, checkpoint_path, map_location='cpu'):
        """
        实例方法：从checkpoint加载EMA参数到主训练网络

        将checkpoint中的EMA参数加载到：
        1. 主训练网络（使用EMA参数初始化训练权重）
        2. EMA模块（正常恢复EMA状态）

        Args:
            checkpoint_path: checkpoint文件路径
            map_location: 加载设备
        """
        print(f"[ControlLDM] Loading EMA weights to model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        state_dict = checkpoint.get('state_dict', checkpoint)

        # 分离出不同类型的参数
        control_fault_ema_params = {}
        control_horiz_ema_params = {}
        model_ema_params = {}
        other_params = {}

        for key, value in state_dict.items():
            if key.startswith('control_fault_ema.'):
                control_fault_ema_params[key] = value
            elif key.startswith('control_horiz_ema.'):
                control_horiz_ema_params[key] = value
            elif key.startswith('model_ema.'):
                model_ema_params[key] = value
            else:
                other_params[key] = value

        print(f"[ControlLDM] Found EMA params - control_fault: {len(control_fault_ema_params)}, "
              f"control_horiz: {len(control_horiz_ema_params)}, model: {len(model_ema_params)}")

        # 1. 首先加载非EMA参数到对应模块
        missing_keys, unexpected_keys = self.load_state_dict(other_params, strict=False)

        # 2. 恢复EMA模块的状态
        if self.use_control_ema:
            if control_fault_ema_params:
                self._load_ema_state(self.control_fault_ema, control_fault_ema_params, 'control_fault_ema')
            else:
                # 没有EMA参数，用加载后的模型参数重新初始化EMA
                self.control_fault_ema = LitEma(self.control_fault)
                print("[ControlLDM] No control_fault_ema params found, initialized from model weights")
            
            if control_horiz_ema_params:
                self._load_ema_state(self.control_horiz_ema, control_horiz_ema_params, 'control_horiz_ema')
            else:
                # 没有EMA参数，用加载后的模型参数重新初始化EMA
                self.control_horiz_ema = LitEma(self.control_horiz)
                print("[ControlLDM] No control_horiz_ema params found, initialized from model weights")

        if self.use_ema and hasattr(self, 'model_ema'):
            if model_ema_params:
                self._load_ema_state(self.model_ema, model_ema_params, 'model_ema')
            else:
                # 没有EMA参数，用加载后的模型参数重新初始化EMA
                self.model_ema = LitEma(self.model)
                print("[ControlLDM] No model_ema params found, initialized from model weights")

        # 3. 将EMA参数复制到主训练网络
        self._copy_ema_to_model()

        print(f"[ControlLDM] Successfully loaded EMA weights to model")

    def _load_ema_state(self, ema_module, ema_params, prefix):
        """
        加载EMA模块的状态

        Args:
            ema_module: LitEma实例
            ema_params: EMA参数字典
            prefix: 参数前缀
        """
        # 构建去掉前缀的state_dict
        ema_state = {}
        for key, value in ema_params.items():
            new_key = key[len(f"{prefix}."):]
            ema_state[new_key] = value

        # 加载到EMA模块
        ema_module.load_state_dict(ema_state)
        print(f"[ControlLDM] Loaded {len(ema_state)} params to {prefix}")

    def _copy_ema_to_model(self):
        """
        将EMA参数复制到主训练网络

        这样训练会从EMA参数开始，而不是从原始训练参数开始
        """
        # 复制control网络的EMA到主网络
        if self.use_control_ema:
            if hasattr(self, 'control_fault_ema'):
                self.control_fault_ema.copy_to(self.control_fault)
                print(f"[ControlLDM] Copied control_fault_ema -> control_fault")

            if hasattr(self, 'control_horiz_ema'):
                self.control_horiz_ema.copy_to(self.control_horiz)
                print(f"[ControlLDM] Copied control_horiz_ema -> control_horiz")

        # 复制主干的EMA到主网络（如果存在）
        if self.use_ema and hasattr(self, 'model_ema'):
            self.model_ema.copy_to(self.model)
            print(f"[ControlLDM] Copied model_ema -> model")

    def load_checkpoint_resume_from_ema(self, checkpoint_path, map_location='cpu'):
        """
        便捷方法：从checkpoint恢复训练，使用EMA参数初始化

        这个方法会：
        1. 加载checkpoint中的所有参数
        2. 将EMA参数加载到EMA模块
        3. 将EMA参数复制到主训练网络（从EMA参数开始继续训练）

        Args:
            checkpoint_path: checkpoint文件路径
            map_location: 加载设备
        """
        self.load_ema_to_model(checkpoint_path, map_location)

    @staticmethod
    def load_checkpoint_only_ema_to_new_model(checkpoint_path, map_location='cpu'):
        """
        静态工具方法：从checkpoint中提取EMA参数，转换为可以直接加载到新模型的格式

        返回一个state_dict，其中EMA参数已经被转换为对应的主网络参数名

        Args:
            checkpoint_path: checkpoint文件路径
            map_location: 加载设备

        Returns:
            dict: 转换后的state_dict，EMA参数已映射到主网络参数名
        """
        print(f"[ControlLDM] Extracting EMA weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}

        # 映射关系：EMA buffer名 -> 主网络参数前缀
        ema_to_model_prefix = {
            'control_fault_ema': 'control_fault',
            'control_horiz_ema': 'control_horiz',
            'model_ema': 'model',
        }

        # 用于追踪已处理的EMA参数
        processed_ema_keys = set()

        for key, value in state_dict.items():
            used = False

            # 检查是否是EMA参数
            for ema_prefix, model_prefix in ema_to_model_prefix.items():
                if key.startswith(f"{ema_prefix}."):
                    processed_ema_keys.add(key)
                    used = True
                    # 保留EMA参数（用于EMA模块）
                    new_state_dict[key] = value
                    break

            # 非EMA参数直接保留
            if not used:
                new_state_dict[key] = value

        print(f"[ControlLDM] Processed {len(processed_ema_keys)} EMA keys")
        return new_state_dict

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_fault = self.control_fault.cuda()
            self.control_horiz = self.control_horiz.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_fault = self.control_fault.cpu()
            self.control_horiz = self.control_horiz.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    @torch.no_grad()
    @rank_zero_only
    @torch.autocast(device_type='cuda')
    def validation_step(self, batch, batch_idx):
        ddim_sampler = DDIMSampler(self)
        horiz, fault = batch
        horiz, fault = horiz.cuda(), fault.cuda()
        horiz = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(horiz)
        cond = {'c_crossattn': [self.clip_txt.to(horiz.device)], 'fault': fault, 'horiz': horiz}
        B, C, H, W = horiz.shape
        shape = (4, H // 8, W // 8)

        # 在验证时使用EMA参数
        with self.ema_scope("Validation"):
            samples, intermediates = ddim_sampler.sample(50, 1,
                                                         shape, cond, verbose=False, eta=0.,
                                                         unconditional_guidance_scale=1., )

        x_samples = self.decode_first_stage(samples)
        x_samples = (np.clip(x_samples[0].mean(dim=0).float().cpu().numpy(), a_min=-1, a_max=1) + 1) / 2
        x_samples = normalization(x_samples)
        horiz = (horiz.float().cpu().numpy()[0, 0] + 1) / 2
        mask = (horiz > 0.).astype(np.float32)[..., None]
        horiz = plt.get_cmap('jet')(horiz)

        jet_img = plt.get_cmap('jet')(x_samples)
        horiz_img = plt.get_cmap('tab20')(x_samples)
        horiz_img = np.where(mask, horiz, horiz_img)
        img = np.concatenate([jet_img, horiz_img], axis=0)
        plt.imsave(f'val_fig/{str(self.global_step).zfill(6)}_{batch_idx}.png', img)

# =============== 使用说明 ===============
#
# 自动判断逻辑：
# ┌─────────────┬──────────────┬──────────────────────────┐
# │ sd_locked   │ 主干EMA      │ Control网络EMA           │
# ├─────────────┼──────────────┼──────────────────────────┤
# │ True (默认) │ ❌ 自动关闭  │ ✅ 启用 (use_control_ema)│
# │ False       │ ✅ 保持父类  │ ✅ 启用 (use_control_ema)│
# └─────────────┴──────────────┴──────────────────────────┘
#
# Checkpoint 自动保存：
# - control_fault_ema (如果 use_control_ema=True)
# - control_horiz_ema (如果 use_control_ema=True)
# - model_ema (仅当 sd_locked=False 且 use_ema=True)
#
# 配置示例（YAML）：
# model:
#   target: cldm.cldm.ControlLDM
#   params:
#     sd_locked: true        # 锁定主干 -> 自动关闭主干EMA
#     use_control_ema: true  # 对control网络使用EMA
#     use_ema: true          # 父类参数，会被sd_locked覆盖
#
# =======================================