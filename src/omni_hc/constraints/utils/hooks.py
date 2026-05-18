from __future__ import annotations

import torch
import torch.nn as nn


def _parse_path(path: str) -> tuple[str, int | None]:
    """Split 'blocks.-1[1]' into ('blocks.-1', 1); plain paths return (path, None)."""
    if path.endswith("]") and "[" in path:
        module_path, _, idx_str = path.rpartition("[")
        return module_path, int(idx_str[:-1])
    return path, None


def _resolve_module_by_path(module: nn.Module, module_path: str) -> nn.Module:
    current = module
    for part in module_path.split("."):
        if part.lstrip("-").isdigit():
            current = current[int(part)]
            continue
        if not hasattr(current, part):
            raise ValueError(
                f"Invalid latent_module path '{module_path}', missing '{part}'"
            )
        current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise ValueError(f"Resolved object at '{module_path}' is not an nn.Module")
    return current


class ForwardHookLatentExtractor:
    """Captures one or more module outputs and exposes them as a latent tensor.

    Each path may include an optional tuple index suffix, e.g. 'blocks.-1[1]'
    to select the second element of a tuple output. Without a suffix, a tuple
    output falls back to element 0 (backward-compatible).

    When multiple paths are given, the captured tensors are concatenated along
    the last dimension. All tensors must share the same shape except on dim=-1.
    """

    def __init__(self, model: nn.Module, module_path: str | list[str]):
        paths = [module_path] if isinstance(module_path, str) else list(module_path)
        self.module_paths = paths
        self._latents: dict[str, torch.Tensor | None] = {p: None for p in paths}
        self._handles = []
        for path in paths:
            module_path_clean, idx = _parse_path(path)
            target = _resolve_module_by_path(model, module_path_clean)
            self._handles.append(target.register_forward_hook(self._make_capture(path, idx)))

    def _make_capture(self, path: str, idx: int | None):
        def _capture(_module, _inputs, output):
            if isinstance(output, tuple):
                self._latents[path] = output[idx if idx is not None else 0]
            else:
                self._latents[path] = output
        return _capture

    def reset(self):
        self._latents = {p: None for p in self.module_paths}

    def get(self):
        tensors = [self._latents[p] for p in self.module_paths]
        if any(t is None for t in tensors):
            return None
        if len(tensors) == 1:
            return tensors[0]
        shapes = [t.shape[:-1] for t in tensors]
        if len(set(shapes)) > 1:
            raise ValueError(
                f"Latent tensors have incompatible shapes for concatenation: "
                f"{[t.shape for t in tensors]}. All dims except the last must match."
            )
        return torch.cat(tensors, dim=-1)

    def remove(self):
        for h in self._handles:
            h.remove()

