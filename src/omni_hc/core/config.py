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

