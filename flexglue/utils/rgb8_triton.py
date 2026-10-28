# SPDX-License-Identifier: MIT
"""Optional Torch/Triton RGB8 preparation for CUDA descriptors.

Imported only by the opt-in descriptor path; no external image runtime is needed.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
import triton
import triton.language as tl


def fp32_codebook() -> np.ndarray:
    """FP32 values from float64(code)/255 for every uint8 code."""
    values = np.arange(256, dtype=np.float64) / np.float64(255.0)
    return values.astype(np.float32)


def select_upload_pixels(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """Borrow exact OpenCV BGR backing, or fall back to owning RGB bytes."""
    if type(rgb) is not np.ndarray or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected HWC RGB8 NumPy image")
    width = rgb.shape[1]
    if rgb.strides == (width * 3, 3, -1):
        bgr = rgb[:, :, ::-1]
        if bgr.flags.c_contiguous and np.shares_memory(bgr, rgb):
            return bgr, 1
    if rgb.flags.c_contiguous:
        return rgb, 0
    return np.array(rgb, dtype=np.uint8, copy=True, order="C"), 0


@lru_cache(maxsize=1)
def _kernel():
    @triton.jit
    def resize(src, table, out,
               SRC_W: tl.constexpr, SRC_H: tl.constexpr,
               DST_W: tl.constexpr, DST_H: tl.constexpr,
               SCALE_X: tl.constexpr, SCALE_Y: tl.constexpr,
               SOURCE_IS_BGR: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < (DST_W * DST_H * 3)
        channel = i % 3
        source_channel = tl.where(SOURCE_IS_BGR, 2 - channel, channel)
        pixel = i // 3
        ox = pixel % DST_W
        oy = pixel // DST_W
        output_i = channel * (DST_W * DST_H) + pixel
        if SRC_W == DST_W and SRC_H == DST_H:
            code = tl.load(src + pixel * 3 + source_channel, mask, other=0).to(tl.int32)
            value = tl.load(table + code, mask, other=0.0)
        else:
            # Host-rounded FP32 scales, align_corners=False half-pixel source order.
            scale_x = tl.full((BLOCK,), SCALE_X, tl.float32)
            scale_y = tl.full((BLOCK,), SCALE_Y, tl.float32)
            sx = tl.maximum(scale_x * (ox.to(tl.float32) + 0.5) - 0.5, 0.0)
            sy = tl.maximum(scale_y * (oy.to(tl.float32) + 0.5) - 0.5, 0.0)
            x0 = tl.minimum(tl.floor(sx).to(tl.int32), SRC_W - 1)
            y0 = tl.minimum(tl.floor(sy).to(tl.int32), SRC_H - 1)
            x1 = tl.minimum(x0 + 1, SRC_W - 1)
            y1 = tl.minimum(y0 + 1, SRC_H - 1)
            wx = sx - x0.to(tl.float32)
            wy = sy - y0.to(tl.float32)
            a = ((y0 * SRC_W + x0) * 3 + source_channel)
            b = ((y0 * SRC_W + x1) * 3 + source_channel)
            c = ((y1 * SRC_W + x0) * 3 + source_channel)
            d = ((y1 * SRC_W + x1) * 3 + source_channel)
            v00 = tl.load(table + tl.load(src + a, mask, other=0).to(tl.int32), mask, other=0.0)
            v01 = tl.load(table + tl.load(src + b, mask, other=0).to(tl.int32), mask, other=0.0)
            v10 = tl.load(table + tl.load(src + c, mask, other=0).to(tl.int32), mask, other=0.0)
            v11 = tl.load(table + tl.load(src + d, mask, other=0).to(tl.int32), mask, other=0.0)
            w0, w1 = 1.0 - wx, wx
            h0, h1 = 1.0 - wy, wy
            value = h0 * (w0 * v00 + w1 * v01) + h1 * (w0 * v10 + w1 * v11)
        tl.store(out + output_i, value, mask)

    return resize


@dataclass
class PreparedDescriptorInput:
    tensor: torch.Tensor
    original_size: torch.Tensor
    owners: tuple
    producer_stream: torch.cuda.Stream

    def require_current_stream(self) -> None:
        current = torch.cuda.current_stream(self.tensor.device)
        if current.cuda_stream != self.producer_stream.cuda_stream:
            raise RuntimeError("descriptor input must be consumed on its producer stream")


def prepare_rgb8(rgb: np.ndarray, *, target_hw: tuple[int, int] = (512, 512),
                 device: Union[torch.device, str] = "cuda",
                 codebook_device: Optional[torch.Tensor] = None) -> PreparedDescriptorInput:
    if type(rgb) is not np.ndarray or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected HWC RGB8 NumPy image")
    src_h, src_w = (int(v) for v in rgb.shape[:2])
    if not (1 <= src_h <= 65535 and 1 <= src_w <= 65535 and src_h * src_w <= 64_000_000):
        raise ValueError("source grid outside bounded prototype")
    if (not isinstance(target_hw, tuple) or len(target_hw) != 2
            or any(type(v) is not int or not 1 <= v <= 4096 for v in target_hw)
            or target_hw[0] * target_hw[1] * 3 * 4 > 512 * 1024**2):
        raise ValueError("target grid outside bounded prototype")
    dst_h, dst_w = target_hw
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("explicit CUDA device required")
    current_index = torch.cuda.current_device()
    if device.index not in (None, current_index):
        raise ValueError("prototype requires the caller's current CUDA device")
    device = torch.device("cuda", current_index)
    stream = torch.cuda.current_stream(device)
    host_pixels, source_is_bgr = select_upload_pixels(rgb)
    # Blocking Torch copy owns device bytes before the borrowed host view can retire.
    source = torch.from_numpy(host_pixels).to(device=device, non_blocking=False)
    if source.dtype != torch.uint8 or source.shape != (src_h, src_w, 3) or not source.is_contiguous():
        raise RuntimeError("RGB8 upload contract differs")
    device = source.device
    if codebook_device is None:
        table = torch.from_numpy(fp32_codebook()).to(device=device, non_blocking=False)
    else:
        table = codebook_device
        if (not isinstance(table, torch.Tensor) or table.device != device
                or table.dtype != torch.float32 or tuple(table.shape) != (256,)
                or not table.is_contiguous()):
            raise ValueError("caller-owned codebook binding differs")
    output = torch.empty((1, 3, dst_h, dst_w), dtype=torch.float32, device=device)
    if output.stride() != (3 * dst_h * dst_w, dst_h * dst_w, dst_w, 1):
        raise RuntimeError("output must have contiguous NCHW strides")
    count = dst_h * dst_w * 3
    # Installed Triton 3.8 CUDA backend uses Torch's current raw stream.
    # Request FMA contraction explicitly for the qualified interpolation order.
    # No libdevice FTZ is needed for this bounded normal-value interpolation.
    _kernel()[((count + 255) // 256,)](
        source, table, output, src_w, src_h, dst_w, dst_h,
        float(np.float32(src_w / dst_w)), float(np.float32(src_h / dst_h)),
        bool(source_is_bgr), 256, num_warps=4,
        enable_fp_fusion=True, enable_reflect_ftz=False,
    )
    source.record_stream(stream)
    table.record_stream(stream)
    output.record_stream(stream)
    size = torch.tensor([[src_w, src_h]], device=device, dtype=torch.float32)
    return PreparedDescriptorInput(output, size, (rgb, host_pixels, source, table), stream)


@torch.no_grad()
def describe_prepared(descriptor, prepared: PreparedDescriptorInput, *, keypoints=None) -> dict:
    prepared.require_current_stream()
    x = prepared.tensor
    normalized = (x - descriptor.mean) / descriptor.std
    if descriptor.conf.mp == "float16":
        normalized = normalized.to(torch.float16)
    features = descriptor.head(descriptor.backbone(normalized))
    if descriptor.conf.mp == "float16":
        features = features.float()
    pred = {"features": features, "image_size": prepared.original_size.to(x)}
    if keypoints is not None:
        pred["descriptors"] = descriptor.sample_descriptors(features, keypoints, pred["image_size"])
        pred["keypoints"] = keypoints
    return pred


