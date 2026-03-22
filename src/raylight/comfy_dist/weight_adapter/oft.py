import logging
from typing import Optional

import torch
from torch.distributed.tensor import DTensor
import comfy.model_management
from .base import (
    WeightAdapterBase,
    WeightAdapterTrainBase,
    weight_decompose,
    factorization,
)


class OFTDiff(WeightAdapterTrainBase):
    def __init__(self, weights):
        super().__init__()
        blocks, rescale, alpha, _ = weights

        self.oft_blocks = torch.nn.Parameter(blocks)
        if rescale is not None:
            self.rescale = torch.nn.Parameter(rescale)
            self.rescaled = True
        else:
            self.rescaled = False
        self.block_num, self.block_size, _ = blocks.shape
        self.constraint = float(alpha)
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=False)

    def __call__(self, w):
        org_dtype = w.dtype
        I = torch.eye(self.block_size, device=self.oft_blocks.device)

        q = self.oft_blocks - self.oft_blocks.transpose(1, 2)
        normed_q = q
        if self.constraint:
            q_norm = torch.norm(q) + 1e-8
            if q_norm > self.constraint:
                normed_q = q * self.constraint / q_norm
        r = (I + normed_q) @ (I - normed_q).float().inverse()

        _, *shape = w.shape
        org_weight = w.to(dtype=r.dtype)
        org_weight = org_weight.unflatten(0, (self.block_num, self.block_size))
        weight = torch.einsum(
            "k n m, k n ... -> k m ...",
            r,
            org_weight,
        ).flatten(0, 1)
        if self.rescaled:
            weight = self.rescale * weight
        return weight.to(org_dtype)

    def passive_memory_usage(self):
        return sum(param.numel() * param.element_size() for param in self.parameters())


class OFTAdapter(WeightAdapterBase):
    name = "oft"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights

    @classmethod
    def create_train(cls, weight, rank=1, alpha=1.0):
        out_dim = weight.shape[0]
        block_size, block_num = factorization(out_dim, rank)
        block = torch.zeros(
            block_num, block_size, block_size, device=weight.device, dtype=torch.float32
        )
        return OFTDiff((block, None, alpha, None))

    def to_train(self):
        return OFTDiff(self.weights)

    @classmethod
    def load(
        cls,
        x: str,
        lora: dict[str, torch.Tensor],
        alpha: float,
        dora_scale: torch.Tensor,
        loaded_keys: set[str] = None,
    ) -> Optional["OFTAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()
        blocks_name = "{}.oft_blocks".format(x)
        rescale_name = "{}.rescale".format(x)

        blocks = None
        if blocks_name in lora.keys():
            blocks = lora[blocks_name]
            if blocks.ndim == 3:
                loaded_keys.add(blocks_name)
            else:
                blocks = None
        if blocks is None:
            return None

        rescale = None
        if rescale_name in lora.keys():
            rescale = lora[rescale_name]
            loaded_keys.add(rescale_name)

        weights = (blocks, rescale, alpha, dora_scale)
        return cls(loaded_keys, weights)

    def calculate_weight(
        self,
        weight,
        key,
        strength,
        strength_model,
        offset,
        function,
        intermediate_dtype=torch.float32,
        original_weight=None,
        device_mesh=None,
    ):
        v = self.weights
        blocks = v[0]
        rescale = v[1]
        alpha = v[2]
        if alpha is None:
            alpha = 0
        dora_scale = v[3]

        blocks = comfy.model_management.cast_to_device(
            blocks, weight.device, intermediate_dtype
        )
        if rescale is not None:
            rescale = comfy.model_management.cast_to_device(
                rescale, weight.device, intermediate_dtype
            )

        block_num, block_size, *_ = blocks.shape

        try:
            I = torch.eye(block_size, device=blocks.device, dtype=blocks.dtype)
            q = blocks - blocks.transpose(1, 2)
            normed_q = q
            if alpha > 0:
                q_norm = torch.norm(q) + 1e-8
                if q_norm > alpha:
                    normed_q = q * alpha / q_norm
            r = (I + normed_q) @ (I - normed_q).float().inverse()
            r = r.to(weight)
            I_w = torch.eye(block_size, device=weight.device, dtype=weight.dtype)
            _, *shape = weight.shape
            lora_diff = torch.einsum(
                "k n m, k n ... -> k m ...",
                (r * strength) - strength * I_w,
                weight.view(block_num, block_size, *shape),
            ).view(-1, *shape)
            if dora_scale is not None:
                weight = weight_decompose(
                    dora_scale,
                    weight,
                    lora_diff,
                    alpha,
                    strength,
                    intermediate_dtype,
                    function,
                )
            else:
                if isinstance(weight, DTensor):
                    weight += DTensor.from_local(function((strength * lora_diff).type(weight.dtype)), device_mesh)
                else:
                    weight += function((strength * lora_diff).type(weight.dtype))
        except Exception as e:
            logging.error("ERROR {} {} {}".format(self.name, key, e))
        return weight
