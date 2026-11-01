# SPDX-License-Identifier: MIT
"""RGB8 conversion in Torch, with optional compilation of conversion/layout."""
from functools import lru_cache, partial
from typing import Optional, Union

import numpy as np
import torch
from torch.nn import functional as F

from ._rgb8 import PreparedDescriptorInput as PreparedDescriptorInput
from ._rgb8 import describe_prepared as describe_prepared
from ._rgb8 import fp32_codebook as fp32_codebook
from ._rgb8 import prepare_rgb8 as _prepare_rgb8

__all__ = [
    "PreparedDescriptorInput",
    "describe_prepared",
    "fp32_codebook",
    "prepare_rgb8",
]


def _convert_rgb8(source, table, source_is_bgr, target_hw=None):
    if target_hw is not None:
        height, width, channels = source.shape
        th, tw = target_hw
        if height >= th and width >= tw and height % th == 0 and width % tw == 0:
            # Integer half-pixel grids read only the central pair per cell.
            # Odd factors retain a third sample: native interpolation still
            # reads the zero-weight neighbour, which matters for NaN/Inf LUTs.
            sy, sx = height // th, width // tw
            ky, kx = min(sy, 2 + sy % 2), min(sx, 2 + sx % 2)
            if ky < sy or kx < sx:
                y, x = (sy - ky) // 2, (sx - kx) // 2
                source = source.reshape(th, sy, tw, sx, channels)[
                    :, y : y + ky, :, x : x + kx, :
                ].reshape(th * ky, tw * kx, channels)

    image = table[source.to(torch.int32)].permute(2, 0, 1).unsqueeze(0).contiguous()
    return image.flip(1) if source_is_bgr else image


@lru_cache(maxsize=1)
def _compiled_converter():
    return torch.compile(_convert_rgb8, dynamic=True)


def _resize_rgb8(source, table, target_hw, source_is_bgr, compile_conversion=False):
    convert = _compiled_converter() if compile_conversion else _convert_rgb8
    image = convert(source, table, source_is_bgr, target_hw)
    if source.shape[:2] != target_hw:
        image = F.interpolate(
            image, size=target_hw, mode="bilinear", align_corners=False
        )
    return image


def prepare_rgb8(
    rgb: np.ndarray,
    *,
    target_hw: tuple[int, int] = (512, 512),
    device: Union[torch.device, str] = "cuda",
    codebook_device: Optional[torch.Tensor] = None,
    compile_conversion: bool = False,
) -> PreparedDescriptorInput:
    return _prepare_rgb8(
        rgb,
        target_hw=target_hw,
        device=device,
        codebook_device=codebook_device,
        resize=partial(_resize_rgb8, compile_conversion=compile_conversion),
    )
