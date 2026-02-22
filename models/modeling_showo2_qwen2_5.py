# coding=utf-8
# Copyright 2025 NUS Show Lab.
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
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoConfig
from torch.nn.attention.flex_attention import BlockMask
from .misc import seg_prediction,velocity_prediction, next_token_prediction, interpolate_pos_encoding,init_special_token_embeddings
from .modeling_siglip import SiglipModel
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .modules import DiffusionHeadConfig
from .modules import ModulatedAttentionBlock, RMSNorm, PatchEmbed, TimestepEmbedder, FinalLayer
from .qwen2_dual import Qwen2ForCausalLM
from .condLoRA_add import ConcatFusiona_M,ConditionProjector
from .condLoRA import ConcatFusion_L,ConcatFusion_M,ConcatFusion_S
from .segmentor import TokensToVolume

# ============（H, W, D；comp=(cD,cH,cW)） ============
class SpatialAPE3D_HWD(nn.Module):
    """
   """
    def __init__(
        self,
        dim: int,
        spatial_size,
        comp,
        num_freqs: int = 8,
        base: float = 2.0,
        learnable_proj: bool = True,
        pos_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_freqs = num_freqs
        self.base = base
        self.pos_scale = pos_scale

        H, W, D = spatial_size
        cD, cH, cW = comp
        assert H % cH == 0 and W % cW == 0 and D % cD == 0, \
            f"spatial_size={spatial_size} 必须能被 comp={comp} 整除"
        self.Hg, self.Wg, self.Dg = H // cH, W // cW, D // cD
        self.N = self.Hg * self.Wg * self.Dg

        # 网格坐标（归一化到 [0,1]）
        y = torch.linspace(0, 1, self.Hg)  # H
        x = torch.linspace(0, 1, self.Wg)  # W
        z = torch.linspace(0, 1, self.Dg)  # D
        yy, xx, zz = torch.meshgrid(y, x, z, indexing='ij')  # (Hg,Wg,Dg)

        # 多频率 Fourier 特征：shape (N, 6F)
        freqs = (self.base ** torch.arange(self.num_freqs))  # (F,)
        def fe(t):
            t = 2 * torch.pi * (t[..., None] * freqs)        # (...,F)
            return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)  # (...,2F)
        fy, fx, fz = fe(yy), fe(xx), fe(zz)                  # (Hg,Wg,Dg,2F)
        feats = torch.cat([fy, fx, fz], dim=-1).reshape(self.N, -1)  # (N, 6F)
        self.register_buffer("feats", feats, persistent=False)

        # 可学习投影（零初始化）
        in_dim = feats.shape[-1]  # 6 * num_freqs
        self.proj = nn.Linear(in_dim, dim, bias=False)
        if not learnable_proj:
            for p in self.proj.parameters():
                p.requires_grad_(False)
        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)  # ★ 关键：零初始化，初期不扰动

    def forward(self, x_like: torch.Tensor) -> torch.Tensor:
        # 每次前向使用当前权重投影 feats，保证可学习
        feats = self.feats.to(device=x_like.device, dtype=x_like.dtype)    # (N, 6F)
        pos = self.proj(feats) * float(self.pos_scale)                     # (N, dim)
        return pos.unsqueeze(0)                                            # (1, N, dim)



class Showo2Qwen2_5(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            llm_vocab_size=None,
            llm_model_path='',
            load_from_showo=False,
            image_latent_dim=16,
            image_latent_height=16,
            image_latent_width=16,
            video_latent_height=16,
            video_latent_width=16,
            patch_size=2,
            hidden_size=2048,
            clip_latent_dim=1152,
            num_diffusion_layers=10,
            add_time_embeds=True,
            add_qk_norm=False,
            spatial_size_my=[192,192,40],
            clip_pretrained_model_path="google/siglip-so400m-patch14-384",
            **kwargs,
    ):
        super().__init__()

        self.llm_config = AutoConfig.from_pretrained(llm_model_path)
        self.llm_config.share_layer_num = kwargs['share_layer_num']
        print('llm num_hidden_layers:',self.llm_config.num_hidden_layers)
        if load_from_showo:
            self.showo = Qwen2ForCausalLM(self.llm_config)
        else:
            self.showo = Qwen2ForCausalLM.from_pretrained(llm_model_path, 
                                                          attn_implementation='sdpa')
        self.showo.resize_token_embeddings(llm_vocab_size)
        
        # patch embedding layer for semantic layers
        self.image_embedder_und = PatchEmbed(
            patch_size=patch_size,
            in_chans=image_latent_dim,
            embed_dim=clip_latent_dim,
        )

        # projector
        self.image_embedder_gen = PatchEmbed(
            patch_size=patch_size,
            in_chans=image_latent_dim,
            embed_dim=hidden_size,
        )

        # projector2
        self.image_embedder_gen2 = PatchEmbed(
            patch_size=patch_size,
            in_chans=image_latent_dim,
            embed_dim=hidden_size,
        )

        print('showo2qwen2_5 spatial size: ',spatial_size_my)

        # initialize semantic layers from siglip
        siglip_model = SiglipModel.from_pretrained(clip_pretrained_model_path)
        self.position_embedding = siglip_model.vision_model.embeddings.position_embedding
        self.und_trans = siglip_model.vision_model.encoder
        del self.und_trans.layers[-1]
        self.register_buffer("image_position_ids",
                             torch.arange(image_latent_height * image_latent_width).expand((1, -1)),
                             persistent=False)

        self.fusion_proj = nn.Sequential(
            RMSNorm(clip_latent_dim + hidden_size),
            nn.Linear(clip_latent_dim + hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        
        #[ CNN-version] cond input condition
        # self.cond_proj=ConditionProjector(dim=hidden_size)
        # self.cond_proj= ConcatFusiona_M(dim=hidden_size,init_on_build=True)
        # [Linear attention version] cond input condition
        self.cond_proj=ConcatFusion_M(
            config=self.llm_config,
            dim=hidden_size,
            patch_size=patch_size,
            # spatial_size=spatial_size_my,
            # comp=(4,16,16),
            init_on_build=True
        )
        self.spe = SpatialAPE3D_HWD(
            dim=hidden_size,
            spatial_size=spatial_size_my,
            comp=(4,16,16),
        )
        self.segmentor=TokensToVolume(
            spatial_size=spatial_size_my
        )

        # adjust for diffusion head
        self.diffusion_head_config = DiffusionHeadConfig()
        self.time_embed = TimestepEmbedder(self.diffusion_head_config.hidden_size)
        if hidden_size != self.diffusion_head_config.hidden_size:
            self.diff_proj = nn.Sequential(
                nn.Linear(hidden_size, self.diffusion_head_config.hidden_size),
                nn.GELU(),
                nn.Linear(self.diffusion_head_config.hidden_size, self.diffusion_head_config.hidden_size)
            )
            self.time_embed_proj = nn.Linear(self.diffusion_head_config.hidden_size, hidden_size)
        self.diffusion_head_a = nn.ModuleList(
            [ModulatedAttentionBlock(self.diffusion_head_config, layer_idx) for layer_idx in
             range(num_diffusion_layers)]
        )
        self.diffusion_head_b = FinalLayer(self.diffusion_head_config.hidden_size, patch_size, image_latent_dim)

        self.reset_parameters()
    
    def reset_vocbulary(self,tokenizer=None):
        # resize for special tokenizers
        # init special token embeddings
        try:
            self.showo.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass
        try:
            init_special_token_embeddings(self.showo,tokenizer)
        except Exception as e:
            print("[warn] init_special_token_embeddings failed:", e)
        # self.showo.resize_token_embeddings(len(tokenizer))
        # init_special_token_embeddings(self.showo,tokenizer)
        # self.showo.resize_token_embeddings(llm_vocab_size)


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def reset_parameters(self):

        # Initialize image embedders
        w1 = self.image_embedder_und.proj.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        nn.init.constant_(self.image_embedder_und.proj.bias, 0)

        w2 = self.image_embedder_gen.proj.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.image_embedder_gen.proj.bias, 0)

        # Initialize transformer layers for understanding encoding and diffusion head
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        _basic_init(self.und_trans)
        _basic_init(self.fusion_proj)
        _basic_init(self.diffusion_head_a)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out output layers
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.diffusion_head_b.linear.weight, 0)
        nn.init.constant_(self.diffusion_head_b.linear.bias, 0)

    def unpatchify(self, x, h, w, T=0):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.config.image_latent_dim
        p = self.image_embedder_gen.patch_size[0]
        if T == 0:
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], h * p * w * p, c))
        else:
            x = x.reshape(shape=(x.shape[0], T, h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], T, h * p * w * p, c))
        return imgs


    @classmethod
    def from_pretrained(cls, ckpt_path, config=None, map_location="cpu", **kwargs):
        model = cls(config).to("cpu")
        sd = torch.load(ckpt_path, map_location=map_location)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

        model_sd = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)  # <- 关键：strict=False

        # 可在这里调用针对新增模块的初始化
        for name, module in model.named_modules():
            if name.startswith("cond_proj"):
                init_module(module)

        return model

    def forward_und_only(
            self,
            text_tokens=None,
            image_latents=None,
            t=None,
            attention_mask=None,
            text_masks=None,
            image_masks=None,
            text_labels=None,
            image_labels=None,
            modality_positions=None,
            output_hidden_states=True,
            max_seq_len=None,
            device='cuda:0',
            **kwargs,
    ):
        # print('image latents',image_latents.shape)
        T = 0
        input_embeds = self.showo.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        if len(image_latents.shape) != 4:
            b, c, T, h, w = image_latents.shape
        else:
            b, c, h, w = image_latents.shape

        if T == 0:
            # print('image latents',image_latents.shape)
            image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
            image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
        else:
            # (B, C, T, H, W) --> (BT, C, H, W)
            image_latents = rearrange(image_latents, 'b c t h w -> (b t) c h w')
            # (BT, C, H, W) --> (BT, L=H/p*W/p, D)
            # print('image latents',image_latents.shape)
            image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
            image_embeds_und = image_embeds_und.reshape(b, T, -1, self.config.clip_latent_dim)
            image_embeds_und = rearrange(image_embeds_und, 'b t l d -> (b t) l d')

            image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
            image_embeds_gen = image_embeds_gen.reshape(b, T, -1, self.config.hidden_size)
            image_embeds_gen = rearrange(image_embeds_gen, 'b t l d -> b (t l) d')

        # go through semantic layers
        p = self.config.patch_size
        h_, w_ = h // p, w // p
        # specific for fixed resolution of 432x432
        if self.position_embedding.weight.shape[0] == self.image_position_ids.shape[-1]:
            image_embeds_und = image_embeds_und + self.position_embedding(self.image_position_ids)
            image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
        # interpolate position embeddings for dynamic resolution
        else:
            image_embeds_und = image_embeds_und + interpolate_pos_encoding(
                self.config.clip_latent_dim,
                self.position_embedding,
                h_,
                w_,
                1,
            )
            image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
        if T != 0:
            image_embeds_und = image_embeds_und.reshape(b, T, image_embeds_und.shape[1], -1)
            image_embeds_und = rearrange(image_embeds_und, 'b t l d -> b (t l) d')

        # spatial (-temporal) fusion
        image_embeds = self.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

        time_embeds = self.time_embed(t, dtype)
        if hasattr(self, 'time_embed_proj'):
            time_embeds_proj = self.time_embed_proj(time_embeds)
        else:
            time_embeds_proj = time_embeds

        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                # print(length)
                if self.config.add_time_embeds:
                    # print('enter add time embeds')
                    input_embeds[i, offset] = time_embeds_proj[i * modality_positions.size(1) + j]
                    # length - 1 because we add 1 to the num_image_tokens when add_time_embeds=True
                    # it's necessary to include :length-1, as sometimes we may skip some idle images when length=0
                    input_embeds[i, offset + 1:offset + 1 + length - 1] = \
                        image_embeds[i * modality_positions.size(1) + j, :max(length - 1, 0)]
                else:
                    # print('do not enter add time embeds!!!!!!')
                    input_embeds[i, offset:offset + length] = image_embeds[i * modality_positions.size(1) + j, :length]

        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            # latent_cond_embeddings=latent_cond_embeddings,
        )

        logits, last_hidden_states = outputs['logits'], outputs['hidden_states'][-1]

        if text_labels[0] is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            return logits, loss_ntp
        else:
            return logits

    def forward(
            self,
            text_tokens=None,
            input_embeds=None,
            image_latents=None,
            t=None,
            attention_mask=None,
            text_masks=None,
            image_masks=None,
            image_latents_cond=None,
            text_labels=None,
            image_labels=None,
            modality_positions=None,
            first_frame_as_cond=False,
            only_denoise_last_image=False,
            guidance_scale=0.0,
            output_hidden_states=True,
            max_seq_len=None,
            device='cuda:0',
            # spatial_size_my=None,
            alpha_t=None,
            seg_masks=None,
            mode='train',
            max_new_tokens=100,
            temperature=1.0,
            top_k=None,
            eos_token=None,
            **kwargs,
    ):
        
        if image_latents_cond is not None:
            image_latents_cond2=image_latents_cond
            # print('[!] image_latents_cond',image_latents_cond.shape)
        T = 0
        if image_latents is None:
            # text-only
            logits = self.showo(input_ids=text_tokens, attention_mask=attention_mask)
            return logits
        else:
            # multimoidal understanding and generatiopn
            if input_embeds==None:
                input_embeds = self.showo.model.embed_tokens(text_tokens)
            dtype = input_embeds.dtype
            # print(image_latents.shape)
            if len(image_latents.shape) != 4:
                b, c, T, h, w = image_latents.shape
            else:
                b, c, h, w = image_latents.shape

            # print('[!] sample T',T)

            # go through dual-path extraction
            if T == 0:
                image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
                image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
                
            else:
                # (B, C, T, H, W) --> (BT, C, H, W)
                image_latents = rearrange(image_latents, 'b c t h w -> (b t) c h w')
                # (BT, C, H, W) --> (BT, L=H/p*W/p, D)
                # print('image_latents',image_latents.shape)
                image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
                image_embeds_und = image_embeds_und.reshape(b, T, -1, self.config.clip_latent_dim)
                image_embeds_und = rearrange(image_embeds_und, 'b t l d -> (b t) l d')

                image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
                image_embeds_gen = image_embeds_gen.reshape(b, T, -1, self.config.hidden_size)
                image_embeds_gen = rearrange(image_embeds_gen, 'b t l d -> b (t l) d')

            # print('[!] before semantic')
            # go through semantic layers
            p = self.config.patch_size
            h_, w_ = h // p, w // p
            # specific for fixed resolution of 432x432 728 (224,224,150)
            if self.position_embedding.weight.shape[0] == self.image_position_ids.shape[-1]:
                image_embeds_und = image_embeds_und + self.position_embedding(self.image_position_ids)
                image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
            # interpolate position embeddings for dynamic resolution
            else:
                image_embeds_und = image_embeds_und + interpolate_pos_encoding(
                    self.config.clip_latent_dim,
                    self.position_embedding,
                    h_,
                    w_,
                    1,
                )
                image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']

            if T != 0:
                image_embeds_und = image_embeds_und.reshape(b, T, image_embeds_und.shape[1], -1)
                image_embeds_und = rearrange(image_embeds_und, 'b t l d -> b (t l) d')
            # spatial (-temporal) fusion
            image_embeds = self.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))
            

            if image_labels is not None:
                if T == 0:
                    image_labels = rearrange(image_labels, 'b c h w -> b (h w) c')
                    image_labels = image_labels.reshape(shape=(b, h_, w_, p, p, c))
                    image_labels = image_labels.reshape(shape=(b, h_ * w_, p * p * c))
                else:
                    # (B, C, T, H/p, W/p)
                    image_labels = rearrange(image_labels, 'b c t h w -> b (t h w) c')
                    image_labels = image_labels.reshape(shape=(b, T, h_, w_, p, p, c))
                    image_labels = image_labels.reshape(shape=(b, T * h_ * w_, p * p * c))
            # print('[!] before time',t,dtype)
            time_embeds = self.time_embed(t, dtype)
            # print(time_embeds)
            if hasattr(self, 'time_embed_proj'):
                time_embeds_proj = self.time_embed_proj(time_embeds)
            else:
                time_embeds_proj = time_embeds

            # print('[!] time embeds',time_embeds.shape)

            # print('[!] before image_masks')
            # print(image_masks.shape)
            # structure text and image embeddings into sequences
            if image_labels is not None:
                new_image_labels = torch.zeros([b, max_seq_len, p * p * c], device=device, dtype=dtype)
                image_masks = image_masks[:, :, None].repeat(1, 1, p * p * c)
            
            # print('[!] before modality')

            for i, modality_batch in enumerate(modality_positions):
                for j, (offset, length) in enumerate(modality_batch):
                    if self.config.add_time_embeds:
                        if image_latents_cond is not None:
                                out1,out2=self.cond_proj(image_latents_cond2.to(dtype))
                                # print('add the image cond')
                                image_embeds += out1
                                
                        
                        if image_labels is not None:
                            # length - 1 because we add 1 to the num_image_tokens when add_time_embeds=True
                            # it's necessary to include :length-1, as sometimes we may skip some idle images when length=0
                            # mask the position of time embedding
                            # it's necessary to include :length-1, as sometimes we may skip some idle images when length=0
                            new_image_labels[i, offset + 1:offset + 1 + length - 1] = image_labels[
                                                            i * modality_positions.size(1) + j, :max(length - 1, 0)]
                            image_masks[i, offset] = 0
                        # else:
                        #     input_embeds[i, offset] = time_embeds_proj[i * modality_positions.size(1) + j]
                        # print('before spe')
                        image_embeds+=self.spe(image_embeds)
                        input_embeds[i, offset] = time_embeds_proj[i * modality_positions.size(1) + j]
                        # length - 1 because we add 1 to the num_image_tokens when add_time_embeds=True
                        # it's necessary to include :length-1, as sometimes we may skip some idle images when length=0
                        input_embeds[i, offset + 1:offset + 1 + length - 1] = image_embeds[
                                                                                i * modality_positions.size(1) + j,
                                                                                :max(length - 1, 0)] 
                    else:
                        print('donot enter time embed!')
                        break
                        if image_latents_cond!=None:
                                image_embeds=self.cond_proj(image_embeds_cond)
                        if image_labels is not None:
                            new_image_labels[i, offset:offset + length] = image_labels[
                                                                          i * modality_positions.size(1) + j, :length]
                        
                        image_embeds+=self.spe(image_embeds)  
                        input_embeds[i, offset:offset + length] = image_embeds[i * modality_positions.size(1) + j,
                                                                  :length]
                        
            # print('[!] before showo', input_embeds.shape)

            outputs = self.showo(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                # position_ids=position_ids,
                output_hidden_states=output_hidden_states,
                latent_cond_embeddings=image_latents_cond
            )


            
            # print('hidden states layers:', len(outputs['hidden_states']))

            # seg for latent #######################
            hidden_states0=outputs['hidden_states'][self.llm_config.share_layer_num[0]]
            hidden_states1=outputs['hidden_states'][self.llm_config.share_layer_num[1]]
            hidden_states2=outputs['hidden_states'][self.llm_config.share_layer_num[2]]
            # print('num_image_latent length:',length)
            input_seg0=[]
            input_seg1=[]
            input_seg2=[]
            for i, modality_batch in enumerate(modality_positions):
                    for j, (offset, length) in enumerate(modality_batch):
                        input_seg0.append(hidden_states0[i, offset+1 :offset + 1 + length - 1].unsqueeze(0))
                        input_seg1.append(hidden_states1[i, offset+1 :offset + 1 + length - 1].unsqueeze(0))
                        input_seg2.append(hidden_states2[i, offset+1 :offset + 1 + length - 1].unsqueeze(0))
            seg_predicted_mask=self.segmentor(torch.cat(input_seg0),
                                              torch.cat(input_seg1),
                                              torch.cat(input_seg2))
            if seg_masks!=None:
                seg_loss,focal_ce_loss,dice_loss=seg_prediction(seg_predicted_mask,seg_masks,item_alpha=alpha_t)

            # diffusion head to predict vector fields
            if hasattr(self, 'diff_proj'):
                last_hidden_states = self.diff_proj(last_hidden_states)
            position_ids = torch.arange(last_hidden_states.shape[1], device=last_hidden_states.device).unsqueeze(0)

                  # last_hidden_states+=image_latents_cond
            # diffusion head 
            for layer in self.diffusion_head_a:
                last_hidden_states = layer(last_hidden_states,
                                           adaln_input=time_embeds,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           modality_positions=modality_positions,
                                           )[0]
            v_pred = self.diffusion_head_b(last_hidden_states, time_embeds, modality_positions)


            if mode =='train':
                if image_latents_cond is not None:
                    # print('v_pred shape:',v_pred.shape,'image masks sum:',image_masks.sum())
                    loss_flow = velocity_prediction(v_pred, new_image_labels[:v_pred.shape[0]], image_masks)
                    if seg_masks!=None:
                        print(f'loss_flow: {loss_flow}, loss_seg: {seg_loss}')
                        return logits, loss_flow,(seg_loss,focal_ce_loss,dice_loss)
                    else:
                        print(f'loss_flow: {loss_flow}')
                        return logits, loss_flow
                else:
                    loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
                    if seg_masks!=None:
                        print(f'loss_ntp: {loss_ntp}, loss_seg: {seg_loss}')
                        return logits, loss_ntp,(seg_loss,focal_ce_loss,dice_loss)
                    else:
                        print(f'loss_ntp: {loss_ntp}')
                        return logits, loss_ntp
            else:

                # inference for t2i
                if image_latents_cond is not None:
                    v_pred_ = []
                    num_imgs = 0
                    for i, modality_batch in enumerate(modality_positions):
                        for j, (offset, length) in enumerate(modality_batch):
                                v_pred_.append(v_pred[i, offset:offset + length])
                                num_imgs += 1
                    v_pred_ = torch.stack(v_pred_)
                    # offset + 1:offset + 1 + length - 1

                #     # remove the time embedding
                    if self.config.add_time_embeds:
                        # print('remove time_embeds')
                        v_pred_ = v_pred_[:, 1:, :]

                    #     # unpatchify
                    v_pred_ = self.unpatchify(v_pred_, h_, w_, T=T)
                    v_pred_ = rearrange(v_pred_, 'b t l c -> b c t l')
                    v_pred_ = v_pred_.reshape(num_imgs, c, T, h, w)

                    return v_pred_,None#seg_predicted_mask
                else:
                    return logits,None#seg_predicted_mask

    # @torch.no_grad()
    def t2i_generate(
            self,
            image_latents=None,
            image_latents_cond =None,
            t=None,
            text_tokens=None,
            attention_mask=None,
            modality_positions=None,
            max_seq_len=None,
            guidance_scale=0.0,
            **kwargs,
    ):
        if guidance_scale > 0.0:
            if t.shape[-1] != text_tokens.shape[0]:
                t_cond, t_uncond = torch.chunk(t, 2)
                t_cond[:-1] = 1.0
                t_uncond[:-1] = 1.0
                t = torch.cat([t_cond, t_uncond])
            v,seg_predicted_mask = self(text_tokens,
                        image_latents=image_latents,
                        t=t,
                        attention_mask=attention_mask,
                        modality_positions=modality_positions,
                        mode='infer',
                        image_latents_cond=image_latents_cond,
                        guidance_scale=guidance_scale,
                        output_hidden_states=True,
                        max_seq_len=max_seq_len)
            v_cond, v_uncond = torch.chunk(v, 2)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            return torch.cat([v, v], dim=0)

        else:
            # print(t)
            if t.shape[-1] != text_tokens.shape[0]:
                t[:-1] = 1.0
            v,seg_predicted_mask = self(text_tokens,
                        image_latents=image_latents,
                        t=t,
                        attention_mask=attention_mask,
                        modality_positions=modality_positions,
                        mode='infer',
                        image_latents_cond=image_latents_cond,
                        guidance_scale=guidance_scale,
                        output_hidden_states=True,
                        max_seq_len=max_seq_len)
            return v


    def mmu_generate(
        self,
        text_tokenizer= None,
        text_tokens=None,           # [B, T] LongTensor
        image_latents=None,
        attention_mask=None,        # [B, T] 1=valid, 0=pad  或  [B,1,1,T] 加性掩码
        modality_positions=None,
        max_seq_len=None,
        spatial_size_my=None,
        device=None,
        max_new_tokens=100,
        temperature=1.0,
        t=None,
        top_k=None,
        eos_token=None,
    ):

        self.eval()
        if device is None:
            device = text_tokens.device if text_tokens is not None else next(self.parameters()).device

        # ---- init ids ----
        cur_ids = text_tokens.to(device)

        # ---- build additive attention mask (float, 0 for valid, -inf for pad) ----
        att_norm = None
        if attention_mask is not None:
            # print('enter attention mask not None')
            att = attention_mask.to(device)
            hidden_dtype = self.showo.get_input_embeddings().weight.dtype
            if att.dim() == 1:
                att = att.unsqueeze(0)
            if att.dim() == 2:
                min_val = torch.finfo(hidden_dtype).min
                pad_mask = (att == 0)                        # True = pad/blocked
                att_norm = torch.zeros_like(att, dtype=hidden_dtype, device=device)
                att_norm = att_norm.masked_fill(pad_mask, min_val)
                att_norm = att_norm[:, None, None, :]       # [B,1,1,T]
            elif att.dim() == 4:
                att_norm = att.to(dtype=hidden_dtype, device=device)
            else:
                # fallback：只做 dtype 对齐
                att_norm = att.to(dtype=hidden_dtype, device=device)

        results = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits,seg_mask_predicted = self(
                    text_tokens=cur_ids,
                    image_latents=image_latents,
                    t=t,
                    attention_mask=att_norm,                     # 传规范化后的掩码
                    modality_positions=modality_positions,
                    mode="infer",
                    output_hidden_states=True,
                    max_seq_len=cur_ids.size(1) if max_seq_len is None else max_seq_len,
                    spatial_size_my=spatial_size_my,
                    device=device,
                )


                last = logits[:, -1, :]
                # last = logits[:, -1, :]
                special_tokens = ["<CRT>", "<RT>", "<TMZ>", "<SURGERY>", "<AM>"]
                # ids = text_tokenizer.convert_tokens_to_ids(special_tokens)
                # print('special ids',ids)
                probs = torch.softmax(last, dim=-1)[0, [0,1,2,3,4]]
                print({tok: float(p) for tok, p in zip(special_tokens, probs)})

                # 只保留 5 个类别 token
                # allowed = torch.as_tensor(ids, device=last.device, dtype=torch.long)
                # mask = torch.full_like(last, float("-inf"))
                # mask[:, allowed] = 0
                # last = last + mask

                next_id = last.argmax(dim=-1, keepdim=True)   # [B, 1]
                # if top_k is not None:
                #     k = min(int(top_k), last.size(-1))
                #     topv, _ = torch.topk(last, k)
                #     last[last < topv[:, [-1]]] = float("-inf")
                # probs = F.softmax(last, dim=-1)

                # next_id = torch.multinomial(probs, num_samples=1)      # [B,1]
                # print(next_id)
                token_v = int(next_id[0, 0].item())
                results.append(token_v)
                if eos_token is not None and token_v == int(eos_token):
                    break

                # append token
                cur_ids = torch.cat([cur_ids, next_id], dim=1)

                # extend additive mask on the key-length axis（最后一维）
                if att_norm is not None:
                    zero_col = torch.zeros(
                        (att_norm.size(0), att_norm.size(1), att_norm.size(2), 1),
                        dtype=att_norm.dtype,
                        device=att_norm.device,
                    )
                    att_norm = torch.cat([att_norm, zero_col], dim=3)

        return seg_mask_predicted,probs,results

