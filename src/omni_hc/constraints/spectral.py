from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _validate_shapelist_2d(shapelist: Sequence[int]) -> tuple[int, int]:
    if len(shapelist) != 2:
        raise ValueError(f"Expected a 2D shapelist, got {tuple(shapelist)!r}")
    height, width = int(shapelist[0]), int(shapelist[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid 2D grid shape {(height, width)!r}")
    return height, width


def reshape_channels_last_to_grid(
    tensor: torch.Tensor, *, shapelist: Sequence[int]
) -> torch.Tensor:
    height, width = _validate_shapelist_2d(shapelist)
    if tensor.ndim != 3:
        raise ValueError(
            "Expected a flattened field with shape (batch, n_points, channels), "
            f"got {tuple(tensor.shape)!r}"
        )
    expected_points = height * width
    if tensor.shape[1] != expected_points:
        raise ValueError(
            f"Expected {expected_points} points for shape {(height, width)}, "
            f"got {tensor.shape[1]}"
        )
    return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], height, width)


def reshape_grid_to_channels_last(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(
            "Expected a grid field with shape (batch, channels, height, width), "
            f"got {tuple(tensor.shape)!r}"
        )
    return tensor.reshape(tensor.shape[0], tensor.shape[1], -1).transpose(1, 2)


def normalize_padding_2d(padding: int | Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(padding, int):
        value = int(padding)
        return (value, value, value, value)
    values = tuple(int(v) for v in padding)
    if len(values) == 2:
        pad_y, pad_x = values
        return (pad_x, pad_x, pad_y, pad_y)
    if len(values) == 4:
        return values
    raise ValueError(
        "padding must be an int, a (pad_y, pad_x) pair, or a "
        "(left, right, top, bottom) tuple"
    )


def pad_spatial_2d(
    tensor: torch.Tensor,
    padding: int | Sequence[int],
    *,
    mode: str,
) -> torch.Tensor:
    pads = normalize_padding_2d(padding)
    if pads == (0, 0, 0, 0):
        return tensor
    if mode == "zeros":
        return F.pad(tensor, pads, mode="constant", value=0.0)
    return F.pad(tensor, pads, mode=mode)


def crop_spatial_2d(
    tensor: torch.Tensor, padding: int | Sequence[int]
) -> torch.Tensor:
    left, right, top, bottom = normalize_padding_2d(padding)
    h_end = tensor.shape[-2] - bottom if bottom > 0 else tensor.shape[-2]
    w_end = tensor.shape[-1] - right if right > 0 else tensor.shape[-1]
    return tensor[..., top:h_end, left:w_end]


def spectral_wavenumbers_2d(
    height: int,
    width: int,
    *,
    dy: float,
    dx: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    ky = 2.0 * torch.pi * torch.fft.fftfreq(height, d=dy)
    kx = 2.0 * torch.pi * torch.fft.fftfreq(width, d=dx)
    ky = ky.to(device=device, dtype=dtype).view(1, 1, height, 1)
    kx = kx.to(device=device, dtype=dtype).view(1, 1, 1, width)
    return ky, kx


def fft_leray_project_2d(
    field: torch.Tensor,
    *,
    dy: float,
    dx: float,
) -> torch.Tensor:
    if field.ndim != 4 or field.shape[1] != 2:
        raise ValueError(
            "Expected a 2D vector field with shape (batch, 2, height, width), "
            f"got {tuple(field.shape)!r}"
        )
    height, width = field.shape[-2], field.shape[-1]
    ky, kx = spectral_wavenumbers_2d(
        height,
        width,
        dy=dy,
        dx=dx,
        device=field.device,
        dtype=field.real.dtype,
    )
    field_ft = torch.fft.fft2(field, dim=(-2, -1))
    k2 = kx.square() + ky.square()
    safe_k2 = torch.where(k2 == 0, torch.ones_like(k2), k2)
    k_dot_field = kx * field_ft[:, 0:1] + ky * field_ft[:, 1:2]

    proj_x = field_ft[:, 0:1] - (kx * k_dot_field) / safe_k2
    proj_y = field_ft[:, 1:2] - (ky * k_dot_field) / safe_k2
    zero_mode = k2 == 0
    proj_x = torch.where(zero_mode, field_ft[:, 0:1], proj_x)
    proj_y = torch.where(zero_mode, field_ft[:, 1:2], proj_y)
    projected_ft = torch.cat([proj_x, proj_y], dim=1)
    return torch.fft.ifft2(projected_ft, dim=(-2, -1)).real


def spectral_divergence_2d(
    field: torch.Tensor,
    *,
    dy: float,
    dx: float,
) -> torch.Tensor:
    if field.ndim != 4 or field.shape[1] != 2:
        raise ValueError(
            "Expected a 2D vector field with shape (batch, 2, height, width), "
            f"got {tuple(field.shape)!r}"
        )
    height, width = field.shape[-2], field.shape[-1]
    ky, kx = spectral_wavenumbers_2d(
        height,
        width,
        dy=dy,
        dx=dx,
        device=field.device,
        dtype=field.real.dtype,
    )
    field_ft = torch.fft.fft2(field, dim=(-2, -1))
    div_ft = 1j * (kx * field_ft[:, 0:1] + ky * field_ft[:, 1:2])
    return torch.fft.ifft2(div_ft, dim=(-2, -1)).real


def spectral_gradient_2d(
    field: torch.Tensor,
    *,
    dy: float,
    dx: float,
) -> torch.Tensor:
    if field.ndim != 4 or field.shape[1] != 1:
        raise ValueError(
            "Expected a scalar field with shape (batch, 1, height, width), "
            f"got {tuple(field.shape)!r}"
        )
    height, width = field.shape[-2], field.shape[-1]
    ky, kx = spectral_wavenumbers_2d(
        height,
        width,
        dy=dy,
        dx=dx,
        device=field.device,
        dtype=field.dtype,
    )
    field_ft = torch.fft.fft2(field, dim=(-2, -1))
    grad_x = torch.fft.ifft2(1j * kx * field_ft, dim=(-2, -1)).real
    grad_y = torch.fft.ifft2(1j * ky * field_ft, dim=(-2, -1)).real
    return torch.cat([grad_x, grad_y], dim=1)


def spectral_curl_2d(
    field: torch.Tensor,
    *,
    dy: float,
    dx: float,
) -> torch.Tensor:
    if field.ndim != 4 or field.shape[1] != 2:
        raise ValueError(
            "Expected a 2D vector field with shape (batch, 2, height, width), "
            f"got {tuple(field.shape)!r}"
        )
    height, width = field.shape[-2], field.shape[-1]
    ky, kx = spectral_wavenumbers_2d(
        height,
        width,
        dy=dy,
        dx=dx,
        device=field.device,
        dtype=field.real.dtype,
    )
    field_ft = torch.fft.fft2(field, dim=(-2, -1))
    curl_ft = 1j * (kx * field_ft[:, 1:2] - ky * field_ft[:, 0:1])
    return torch.fft.ifft2(curl_ft, dim=(-2, -1)).real


def spectral_poisson_solve_2d(
    rhs: torch.Tensor,
    *,
    dy: float,
    dx: float,
) -> torch.Tensor:
    if rhs.ndim != 4 or rhs.shape[1] != 1:
        raise ValueError(
            "Expected a scalar field with shape (batch, 1, height, width), "
            f"got {tuple(rhs.shape)!r}"
        )
    height, width = rhs.shape[-2], rhs.shape[-1]
    ky, kx = spectral_wavenumbers_2d(
        height,
        width,
        dy=dy,
        dx=dx,
        device=rhs.device,
        dtype=rhs.dtype,
    )
    rhs_ft = torch.fft.fft2(rhs, dim=(-2, -1))
    k2 = kx.square() + ky.square()
    # Avoid division by zero by setting k2=0 to 1 (DC component)
    safe_k2 = torch.where(k2 == 0, torch.ones_like(k2), k2)
    solution_ft = -rhs_ft / safe_k2
    solution_ft = torch.where(k2 == 0, torch.zeros_like(solution_ft), solution_ft)

    return torch.fft.ifft2(solution_ft, dim=(-2, -1)).real
