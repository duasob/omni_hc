from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .types import RunRef


@dataclass
class RunData:
    metrics: dict
    config: dict


class MissingRun(Exception):
    pass


class MissingMetric(Exception):
    pass


def load_run(run: RunRef, repo_root: Path) -> RunData:
    run_dir = run.resolve(repo_root)
    metrics_path = run_dir / "test_metrics.yaml"
    config_path = run_dir / "resolved_config.yaml"
    if not metrics_path.exists():
        raise MissingRun(f"missing test_metrics.yaml: {metrics_path}")
    metrics_doc = yaml.safe_load(metrics_path.read_text()) or {}
    metrics = metrics_doc.get("metrics", {})
    if not isinstance(metrics, dict):
        raise MissingRun(f"unexpected metrics shape in {metrics_path}")
    config = (
        yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    ) or {}
    return RunData(metrics=metrics, config=config)


def get_metric(data: RunData, key: str) -> float:
    if key not in data.metrics:
        raise MissingMetric(key)
    value = data.metrics[key]
    if not isinstance(value, (int, float)):
        raise MissingMetric(f"non-numeric metric {key}={value!r}")
    return float(value)
