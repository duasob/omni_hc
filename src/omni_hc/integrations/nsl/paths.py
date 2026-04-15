from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_nsl_root(explicit: str | Path | None = None, cfg: dict | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))

    backend_cfg = (cfg or {}).get("backend", {})
    if isinstance(backend_cfg, dict) and backend_cfg.get("nsl_root"):
        candidates.append(Path(str(backend_cfg["nsl_root"])))

    env_root = os.environ.get("OMNI_HC_NSL_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    root = repo_root()
    candidates.append(root / "external" / "Neural-Solver-Library")

    checked: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        checked.append(resolved)
        if resolved.exists():
            return resolved

    formatted = "\n".join(f"- {path}" for path in checked)
    raise FileNotFoundError(
        "Could not resolve Neural-Solver-Library.\n"
        "Provide --nsl-root, set OMNI_HC_NSL_ROOT, set backend.nsl_root in the config, "
        "or place the checkout at external/Neural-Solver-Library.\n"
        f"Checked:\n{formatted}"
    )
