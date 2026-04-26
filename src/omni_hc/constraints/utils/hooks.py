from __future__ import annotations

import torch.nn as nn


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
    """Captures a module output and exposes it as a latent tensor."""

    def __init__(self, model: nn.Module, module_path: str):
        self.module_path = module_path
        self.latent = None
        target_module = _resolve_module_by_path(model, module_path)
        self.handle = target_module.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self.latent = output[0] if isinstance(output, tuple) else output

    def reset(self):
        self.latent = None

    def get(self):
        return self.latent

    def remove(self):
        self.handle.remove()

