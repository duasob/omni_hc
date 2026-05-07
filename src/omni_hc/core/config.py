from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def parse_dotted_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse CLI overrides like ``constraint.mode=latent_head`` into a dict."""
    overrides: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key cannot be empty: {item}")

        value = yaml.safe_load(raw_value)
        cursor = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            if not part:
                raise ValueError(f"Override path contains an empty segment: {item}")
            existing = cursor.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ValueError(f"Override path conflicts with existing value: {item}")
            cursor = existing
        leaf = parts[-1]
        if not leaf:
            raise ValueError(f"Override path contains an empty segment: {item}")
        cursor[leaf] = value
    return overrides


def load_composed_config(path: str | Path) -> dict[str, Any]:
    return _load_composed_config(Path(path).resolve(), seen=set())


def _load_composed_config(path: Path, seen: set[Path]) -> dict[str, Any]:
    if path in seen:
        cycle = " -> ".join(str(item) for item in list(seen) + [path])
        raise ValueError(f"Detected config inheritance cycle: {cycle}")

    raw = load_yaml_file(path)
    extends = raw.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    if not isinstance(extends, list):
        raise ValueError(f"'extends' must be a string or list in {path}")

    seen.add(path)
    merged: dict[str, Any] = {}
    for relative in extends:
        parent_path = (path.parent / str(relative)).resolve()
        parent_cfg = _load_composed_config(parent_path, seen)
        merged = deep_merge(merged, parent_cfg)
    seen.remove(path)

    return deep_merge(merged, raw)
