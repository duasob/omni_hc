from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .types import MetricFileRef, RunRef


@dataclass
class RunData:
    metrics: dict
    config: dict


class MissingRun(Exception):
    pass


class MissingMetric(Exception):
    pass


def _load_metrics_doc(metrics_path: Path, config_path: Path | None = None) -> RunData:
    if not metrics_path.exists():
        raise MissingRun(f"missing metrics file: {metrics_path}")
    metrics_doc = yaml.safe_load(metrics_path.read_text()) or {}
    metrics = metrics_doc.get("metrics", metrics_doc)
    if not isinstance(metrics, dict):
        raise MissingRun(f"unexpected metrics shape in {metrics_path}")
    config = {}
    if config_path is not None and config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}
    return RunData(metrics=metrics, config=config)


def load_run(run: RunRef, repo_root: Path) -> RunData:
    run_dir = run.resolve(repo_root)
    metrics_path = run_dir / "test_metrics.yaml"
    config_path = run_dir / "resolved_config.yaml"
    return _load_metrics_doc(metrics_path, config_path)


def load_metric_file(ref: MetricFileRef, metrics_dir: Path) -> RunData:
    return _load_metrics_doc(ref.resolve(metrics_dir))


def _derived_metric(data: RunData, key: str) -> float | None:
    if key in {
        "constraint/neg_spacing_count",
        "constraint/neg_spacing_fraction",
        "constraint/neg_spacing_worst_sample_fraction",
    }:
        min_dx = data.metrics.get("constraint/min_dx")
        min_dy = data.metrics.get("constraint/min_dy")
        if isinstance(min_dx, (int, float)) and isinstance(min_dy, (int, float)):
            if min(float(min_dx), float(min_dy)) >= 0.0:
                return 0.0
    if key in {
        "constraint/flipped_cell_count_worst",
        "constraint/flipped_cell_fraction_worst",
    }:
        min_area = data.metrics.get("constraint/min_oriented_cell_area")
        if isinstance(min_area, (int, float)) and float(min_area) >= 0.0:
            return 0.0
    if key == "constraint/below_y_bottom_violation_count":
        bottom_error = data.metrics.get("constraint/bottom_y_abs_error_max")
        if isinstance(bottom_error, (int, float)) and float(bottom_error) <= 0.0:
            return 0.0
    return None


def get_metric(data: RunData, key: str) -> float:
    if key not in data.metrics:
        derived = _derived_metric(data, key)
        if derived is None:
            raise MissingMetric(key)
        return derived
    value = data.metrics[key]
    if not isinstance(value, (int, float)):
        raise MissingMetric(f"non-numeric metric {key}={value!r}")
    return float(value)
