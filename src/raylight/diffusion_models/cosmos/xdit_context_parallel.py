from typing import Optional
from enum import Enum

from einops import rearrange
import torch


class DataType(Enum):
    IMAGE = "image"
    VIDEO = "video"


def usp_general_dit_forward(
    self,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    fps: Optional[torch.Tensor] = None,
    image_size: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    scalar_feature: Optional[torch.Tensor] = None,
    data_type: Optional[DataType] = DataType.VIDEO,
    latent_condition: Optional[torch.Tensor] = None,
    latent_condition_sigma: Optional[torch.Tensor] = None,
    condition_video_augment_sigma: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    Args:
        x: (B, C, T, H, W) tensor of spatial-temp inputs
        timesteps: (B, ) tensor of timesteps
        crossattn_emb: (B, N, D) tensor of cross-attention embeddings
        crossattn_mask: (B, N) tensor of cross-attention masks
        condition_video_augment_sigma: (B,) used in lvg(long video generation), we add noise with this sigma to
            augment condition input, the lvg model will condition on the condition_video_augment_sigma value;
            we need forward_before_blocks pass to the forward_before_blocks function.
    """

    crossattn_emb = context
    crossattn_mask = attention_mask

    inputs = self.forward_before_blocks(
        x=x,
        timesteps=timesteps,
        crossattn_emb=crossattn_emb,
        crossattn_mask=crossattn_mask,
        fps=fps,
        image_size=image_size,
        padding_mask=padding_mask,
        scalar_feature=scalar_feature,
        data_type=data_type,
        latent_condition=latent_condition,
        latent_condition_sigma=latent_condition_sigma,
        condition_video_augment_sigma=condition_video_augment_sigma,
        **kwargs,
    )
    x, affline_emb_B_D, crossattn_emb, crossattn_mask, rope_emb_L_1_1_D, adaln_lora_B_3D, original_shape = (
        inputs["x"],
        inputs["affline_emb_B_D"],
        inputs["crossattn_emb"],
        inputs["crossattn_mask"],
        inputs["rope_emb_L_1_1_D"],
        inputs["adaln_lora_B_3D"],
        inputs["original_shape"],
    )
    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = inputs["extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D"].to(x.dtype)
    del inputs

    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        assert (
            x.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
        ), f"{x.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape} {original_shape}"

    transformer_options = kwargs.get("transformer_options", {})
    for _, block in self.blocks.items():
        assert (
            self.blocks["block0"].x_format == block.x_format
        ), f"First block has x_format {self.blocks[0].x_format}, got {block.x_format}"

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            x += extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D
        x = block(
            x,
            affline_emb_B_D,
            crossattn_emb,
            crossattn_mask,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_3D=adaln_lora_B_3D,
            transformer_options=transformer_options,
        )

    x_B_T_H_W_D = rearrange(x, "T H W B D -> B T H W D")

    x_B_D_T_H_W = self.decoder_head(
        x_B_T_H_W_D=x_B_T_H_W_D,
        emb_B_D=affline_emb_B_D,
        crossattn_emb=None,
        origin_shape=original_shape,
        crossattn_mask=None,
        adaln_lora_B_3D=adaln_lora_B_3D,
    )

    return x_B_D_T_H_W


def usp_mini_train_dit_forward(
    self,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    fps: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    x_B_C_T_H_W = x
    timesteps_B_T = timesteps
    crossattn_emb = context
    """
    Args:
        x: (B, C, T, H, W) tensor of spatial-temp inputs
        timesteps: (B, ) tensor of timesteps
        crossattn_emb: (B, N, D) tensor of cross-attention embeddings
    """
    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
        x_B_C_T_H_W,
        fps=fps,
        padding_mask=padding_mask,
    )

    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)
    t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder[1](self.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype))
    t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

    # for logging purpose
    affline_scale_log_info = {}
    affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
    self.affline_scale_log_info = affline_scale_log_info
    self.affline_emb = t_embedding_B_T_D
    self.crossattn_emb = crossattn_emb

    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        assert (
            x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
        ), f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"

    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0),
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        "transformer_options": kwargs.get("transformer_options", {}),
    }
    for block in self.blocks:
        x_B_T_H_W_D = block(
            x_B_T_H_W_D,
            t_embedding_B_T_D,
            crossattn_emb,
            **block_kwargs,
        )

    x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
    x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
    return x_B_C_Tt_Hp_Wp
