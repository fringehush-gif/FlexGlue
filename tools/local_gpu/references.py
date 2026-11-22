# SPDX-License-Identifier: MIT
"""Frozen qualification controls, never selected by production code."""

import math

import torch
import torch.nn.functional as F


# Original function body from fork #19 head
# 444274bce53dc0f0b3b17934921a961cd29b543f, flexglue/utils/ops.py.
# Only the function name changed. Keeps the old corner product and line blur.
def legacy_lattice_kde(
    points: torch.Tensor, weights: torch.Tensor, std: float
) -> torch.Tensor:
    """Weighted gaussian kde (B, N) of points (B, N, D) in [-1, 1] at themselves."""
    if not math.isfinite(std) or std <= 0 or not 0 < std * std < math.inf:
        raise ValueError("std must be finite, positive and have a finite square.")
    b, n, d = points.shape
    pitch = std / 2  # splat to a lattice, blur it, read it back: cost independent of N
    size = math.ceil(2 / pitch) + 2
    if size**d > torch.iinfo(torch.int64).max:
        raise ValueError("std produces an unrepresentable lattice.")
    # Validate kernel arithmetic before allocating the lattice or indexing it.
    s = math.sqrt(std**2 - pitch**2 / 3)
    r = math.ceil(3 * s / pitch)
    precision = torch.finfo(torch.float32)
    # Pitch conversion and multiplication each round once to float32.
    max_root = math.sqrt(precision.max) / (1 + precision.eps / 2) ** 2
    if r * pitch > max_root or 2 * s * s > precision.max:
        raise ValueError("std exceeds the finite float32 kernel range.")
    u = ((points.float() + 1) / pitch).clamp(0, size - 1 - 1e-4)
    base, frac = u.floor().long(), u.frac()
    # Below this precision bound, the existing upper margin stays representable.
    if (size - 1) * precision.eps >= 1e-4:
        # Float32 can round to the final cell: keep its mass at the upper corner.
        if size >= 2**63 - 2**39:
            # A rounded 2**63 cannot be represented by the int64 conversion.
            base = torch.where(u >= float(2**63), size - 1, base)
        frac = torch.where(base >= size - 1, 1.0, frac)
        base = base.clamp(max=size - 2)
    corners = torch.cartesian_prod(
        *[torch.arange(2, device=points.device)] * d
    ).reshape(-1, d)
    strides = size ** torch.arange(d - 1, -1, -1, device=points.device)
    point_idx = (base * strides).sum(-1)
    corner_idx = (corners * strides).sum(-1)
    idx = (point_idx[..., None] + corner_idx).flatten(1)  # (B, N * 2^d)
    tent = torch.where(corners.bool(), frac[:, :, None], 1 - frac[:, :, None]).prod(-1)
    lattice = u.new_zeros(b, size**d)
    lattice.scatter_add_(1, idx, (tent * weights[..., None].float()).flatten(1))
    # Coordinate construction buffers are no longer needed during the blur.
    del u, base, frac, point_idx, corner_idx, corners, strides
    # splat and read tents add pitch^2 / 6 each
    k = torch.arange(-r, r + 1, device=points.device, dtype=lattice.dtype) * pitch
    kernel = (std / s * torch.exp(-(k**2) / (2 * s**2))).view(1, 1, -1)
    lattice = lattice.view(b, *[size] * d)
    for dim in range(1, d + 1):
        x = lattice.movedim(dim, -1)
        x = F.conv1d(x.reshape(-1, 1, size), kernel, padding=r).view(x.shape)
        lattice = x.movedim(-1, dim)
    lattice = lattice.reshape(b, -1)
    return (lattice.gather(1, idx).view_as(tent) * tent).sum(-1)


def correlation_control(f0, f1, warp, radius, *, fp64_sum=False):
    """Original product/reduction from #25 bda4b7be; optional FP64 sum oracle."""
    b, c, h, w = f0.shape
    r = torch.linspace(-2 * radius, 2 * radius, 2 * radius + 1, device=f0.device)
    dy, dx = torch.meshgrid(r / h, r / w, indexing="ij")
    window = torch.stack((dx, dy), -1).reshape(-1, 2)
    coords = (warp[..., None, :] + window).reshape(b, h, -1, 2)
    sampled = F.grid_sample(f1, coords, align_corners=False).reshape(b, c, h, w, -1)
    scaled = f0[..., None] / c**0.5
    if fp64_sum:
        scaled, sampled = scaled.double(), sampled.double()
    return (scaled * sampled).sum(1).permute(0, 3, 1, 2)


def selected_root_control(scores, promote):
    """Out-of-place selected-prefix control; no unused-root backward paths."""
    if promote and scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    rho = torch.arange(
        1, scores.shape[-1] + 1, device=scores.device, dtype=scores.dtype
    )
    mean = scores.cumsum(-1) / rho
    delta = (
        (1 - rho * (scores.square().cumsum(-1) / rho - mean.square())) / rho
    ).clamp_min(0)
    with torch.no_grad():
        index = (mean - delta.sqrt() <= scores).sum(-1, keepdim=True) - 1
    return mean.gather(-1, index) - delta.gather(-1, index).sqrt()
