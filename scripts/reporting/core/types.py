from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence


@dataclass(frozen=True)
class RunRef:
    """Reference to a model run, by full output path relative to repo root."""

    path: str

    def resolve(self, repo_root: Path) -> Path:
        return repo_root / self.path


@dataclass(frozen=True)
class MetricFileRef:
    """Reference to generated report metrics, relative to --output-dir/metrics."""

    path: str

    def resolve(self, metrics_dir: Path) -> Path:
        return metrics_dir / self.path


@dataclass(frozen=True)
class Row:
    """One cell in a macro table: pull `metric_key` from `run` and bind it to `macro`."""

    run: RunRef | MetricFileRef | None
    metric_key: str | Sequence[str] | None
    macro: str
    format: str = "{:.2e}"
    # Optional override: if provided, use this literal value instead of looking up a run.
    literal: str | None = None


@dataclass(frozen=True)
class ReportArtifact:
    name: str
    chapter: int
    kind: Literal["tex_macros", "figure"]
    output_subpath: str  # relative to --output-dir
    rows: Sequence[Row] = field(default_factory=list)
    render: Callable | None = None  # for kind="figure"
    # for kind="tex_macros": optional hook to mutate the resolved cell values
    # before they are written (e.g. bold the minimum within a row group).
    postprocess: Callable | None = None
