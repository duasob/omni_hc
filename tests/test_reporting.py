from pathlib import Path

import yaml

from scripts.reporting.build import _write_missing_runs_file
from scripts.reporting.core.collect import get_metric, load_metric_file, load_run
from scripts.reporting.core.emit_tex import CellResult
from scripts.reporting.core.emit_tex import TBD, render_macro_table
from scripts.reporting.core.types import MetricFileRef, ReportArtifact, Row, RunRef
from scripts.reporting.registry.ch5 import (
    effective_training_samples,
    render_provenance_table,
)
from scripts.reporting.registry.ns_report import diagnose_command_for_ns_run


def test_reporting_collector_reads_run_and_generated_metrics(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "test_metrics.yaml").write_text(
        yaml.safe_dump({"metrics": {"constraint/a": 1.25}})
    )
    metrics_dir = tmp_path / "report" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "generated.yaml").write_text(
        yaml.safe_dump({"metrics": {"constraint/b": 2.5}})
    )

    assert get_metric(load_run(RunRef("outputs/demo"), tmp_path), "constraint/a") == 1.25
    assert get_metric(load_metric_file(MetricFileRef("generated.yaml"), metrics_dir), "constraint/b") == 2.5


def test_macro_rendering_handles_numeric_literal_and_missing(tmp_path: Path):
    metrics_dir = tmp_path / "report" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "generated.yaml").write_text(
        yaml.safe_dump({"metrics": {"constraint/a": 0.001}})
    )
    artifact = ReportArtifact(
        name="demo",
        chapter=5,
        kind="tex_macros",
        output_subpath="macros/demo.tex",
        rows=[
            Row(run=MetricFileRef("generated.yaml"), metric_key="constraint/a", macro=r"\A"),
            Row(run=None, metric_key=None, macro=r"\Slash", literal="/"),
            Row(run=None, metric_key=None, macro=r"\Missing"),
        ],
    )

    content, results = render_macro_table(artifact, tmp_path, metrics_dir=metrics_dir)

    assert r"\newcommand{\A}{\ensuremath{1.00\times 10^{-3}}}" in content
    assert r"\newcommand{\Slash}{/}" in content
    assert rf"\newcommand{{\Missing}}{{{TBD}}}" in content
    assert [result.ok for result in results] == [True, True, False]


def test_macro_rendering_uses_metric_fallbacks(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "test_metrics.yaml").write_text(
        yaml.safe_dump({"metrics": {"constraint/replacement": 4.0}})
    )
    artifact = ReportArtifact(
        name="demo",
        chapter=5,
        kind="tex_macros",
        output_subpath="macros/demo.tex",
        rows=[
            Row(
                run=RunRef("outputs/demo"),
                metric_key=("constraint/preferred", "constraint/replacement"),
                macro=r"\A",
            ),
        ],
    )

    content, results = render_macro_table(artifact, tmp_path)

    assert r"\newcommand{\A}{\ensuremath{4.00\times 10^{0}}}" in content
    assert results[0].ok


def test_macro_rendering_formats_flops(tmp_path: Path):
    metrics_dir = tmp_path / "report" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "generated.yaml").write_text(
        yaml.safe_dump({"metrics": {"cost/flops": 1.25e12}})
    )
    artifact = ReportArtifact(
        name="demo",
        chapter=5,
        kind="tex_macros",
        output_subpath="macros/demo.tex",
        rows=[
            Row(
                run=MetricFileRef("generated.yaml"),
                metric_key="cost/flops",
                macro=r"\Flops",
                format="flops",
            ),
        ],
    )

    content, results = render_macro_table(artifact, tmp_path, metrics_dir=metrics_dir)

    assert r"\newcommand{\Flops}{\ensuremath{1.25\,\mathrm{TFLOPs}}}" in content
    assert results[0].ok


def test_macro_rendering_formats_latex_percent(tmp_path: Path):
    metrics_dir = tmp_path / "report" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "generated.yaml").write_text(
        yaml.safe_dump({"metrics": {"constraint/fraction": 0.022444}})
    )
    artifact = ReportArtifact(
        name="demo",
        chapter=5,
        kind="tex_macros",
        output_subpath="macros/demo.tex",
        rows=[
            Row(
                run=MetricFileRef("generated.yaml"),
                metric_key="constraint/fraction",
                macro=r"\Fraction",
                format="percent",
            ),
        ],
    )

    content, results = render_macro_table(artifact, tmp_path, metrics_dir=metrics_dir)

    assert r"\newcommand{\Fraction}{2.24\%}" in content
    assert results[0].ok


def test_plasticity_neg_spacing_fraction_can_be_derived(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "plasticity"
    run_dir.mkdir(parents=True)
    (run_dir / "test_metrics.yaml").write_text(
        yaml.safe_dump(
            {"metrics": {"constraint/min_dx": 0.1, "constraint/min_dy": 0.2}}
        )
    )

    value = get_metric(
        load_run(RunRef("outputs/plasticity"), tmp_path),
        "constraint/neg_spacing_worst_sample_fraction",
    )

    assert value == 0.0


def test_plasticity_flipped_cell_count_can_be_derived(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "plasticity"
    run_dir.mkdir(parents=True)
    (run_dir / "test_metrics.yaml").write_text(
        yaml.safe_dump({"metrics": {"constraint/min_oriented_cell_area": 0.1}})
    )

    value = get_metric(
        load_run(RunRef("outputs/plasticity"), tmp_path),
        "constraint/flipped_cell_count_worst",
    )

    assert value == 0.0


def test_plasticity_bottom_boundary_metrics_can_be_derived_within_tolerance(
    tmp_path: Path,
):
    run_dir = tmp_path / "outputs" / "plasticity"
    run_dir.mkdir(parents=True)
    (run_dir / "test_metrics.yaml").write_text(
        yaml.safe_dump(
            {"metrics": {"constraint/bottom_y_abs_error_max": 7.0e-6}}
        )
    )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {"diagnostics": {"bottom_boundary_tolerance": 1.0e-4}}
        )
    )

    data = load_run(RunRef("outputs/plasticity"), tmp_path)

    assert get_metric(data, "constraint/bottom_boundary_violation_count") == 0.0
    assert get_metric(data, "constraint/bottom_boundary_violation_fraction") == 0.0


def test_missing_runs_file_deduplicates_runnable_run_refs(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text("{}")
    (run_dir / "best.pt").write_text("checkpoint")
    artifact = ReportArtifact(
        name="demo",
        chapter=5,
        kind="tex_macros",
        output_subpath="macros/demo.tex",
        rows=[
            Row(run=RunRef("outputs/demo"), metric_key="missing/a", macro=r"\A"),
            Row(run=RunRef("outputs/demo"), metric_key="missing/b", macro=r"\B"),
            Row(run=MetricFileRef("gt.yaml"), metric_key="missing/c", macro=r"\C"),
        ],
    )
    results = [
        CellResult(r"\A", TBD, "missing metric: missing/a", False),
        CellResult(r"\B", TBD, "missing metric: missing/b", False),
        CellResult(r"\C", TBD, "missing: missing metrics file", False),
    ]
    output_path = tmp_path / "missing_runs.txt"

    count = _write_missing_runs_file(
        output_path, [artifact], {"demo": results}, tmp_path
    )

    content = output_path.read_text()
    assert count == 1
    assert (
        "test: --config outputs/demo/resolved_config.yaml "
        "--checkpoint outputs/demo/best.pt"
    ) in content
    assert content.count("test:") == 1
    assert "generated metric source gt.yaml" in content


def test_missing_runs_file_can_include_all_run_refs(tmp_path: Path):
    run_a = tmp_path / "outputs" / "a"
    run_b = tmp_path / "outputs" / "b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    for run_dir in (run_a, run_b):
        (run_dir / "resolved_config.yaml").write_text("{}")
        (run_dir / "best.pt").write_text("checkpoint")
    artifact = ReportArtifact(
        name="demo",
        chapter=5,
        kind="tex_macros",
        output_subpath="macros/demo.tex",
        rows=[
            Row(run=RunRef("outputs/a"), metric_key="metric/a", macro=r"\A"),
            Row(run=RunRef("outputs/b"), metric_key="metric/b", macro=r"\B"),
        ],
    )
    results = [
        CellResult(r"\A", "1", "outputs/a", True),
        CellResult(r"\B", TBD, "missing metric: metric/b", False),
    ]
    output_path = tmp_path / "refresh_runs.txt"

    count = _write_missing_runs_file(
        output_path, [artifact], {"demo": results}, tmp_path, mode="all"
    )

    content = output_path.read_text()
    assert count == 2
    assert "runfile_mode: all" in content
    assert "test: --config outputs/a/resolved_config.yaml" in content
    assert "test: --config outputs/b/resolved_config.yaml" in content


def test_missing_runs_file_keeps_test_and_diagnose_commands_distinct(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "outputs" / "a"
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text("{}")
    (run_dir / "best.pt").write_text("checkpoint")
    artifact = ReportArtifact(
        name="demo",
        chapter=5,
        kind="tex_macros",
        output_subpath="macros/demo.tex",
        rows=[
            Row(run=RunRef("outputs/a"), metric_key="metric/a", macro=r"\A"),
            Row(run=MetricFileRef("ch5_cost_metrics.yaml"), metric_key="cost/a", macro=r"\C"),
        ],
    )
    results = [
        CellResult(r"\A", "1", "outputs/a", True),
        CellResult(r"\C", TBD, "missing metric: cost/a", False),
    ]
    monkeypatch.setattr(
        "scripts.reporting.build.cost_diagnostic_runs",
        lambda: (RunRef("outputs/a"),),
    )

    output_path = tmp_path / "refresh_runs.txt"
    count = _write_missing_runs_file(
        output_path, [artifact], {"demo": results}, tmp_path, mode="all"
    )

    content = output_path.read_text()
    assert count == 2
    assert "test: --config outputs/a/resolved_config.yaml" in content
    assert "diagnose: --config outputs/a/resolved_config.yaml" in content


def test_ns_diagnose_command_falls_back_to_component_config(tmp_path: Path):
    command = diagnose_command_for_ns_run(
        RunRef("outputs/navier_stokes/none/ono/final/seed_42"),
        tmp_path,
    )

    assert command == (
        "diagnose: --benchmark navier_stokes --backbone ONO --constraint none "
        "--budget final --seed 42 "
        "--override paths.output_dir=outputs/navier_stokes/none/ono/final/seed_42 "
        "--write-yaml"
    )


def test_missing_runs_file_uses_ns_diagnose_fallback(monkeypatch, tmp_path: Path):
    artifact = ReportArtifact(
        name="ns_hook",
        chapter=4,
        kind="tex_macros",
        output_subpath="macros/demo.tex",
        rows=[
            Row(
                run=MetricFileRef("ch4_ns_cost_metrics.yaml"),
                metric_key="hook/OnoBase/params_m",
                macro=r"\C",
            ),
        ],
    )
    results = [
        CellResult(
            r"\C",
            TBD,
            "missing metric: hook/OnoBase/params_m",
            False,
        ),
    ]
    monkeypatch.setattr(
        "scripts.reporting.build.ns_cost_diagnostic_runs",
        lambda: (RunRef("outputs/navier_stokes/none/ono/final/seed_42"),),
    )

    output_path = tmp_path / "missing_runs.txt"
    count = _write_missing_runs_file(
        output_path, [artifact], {"ns_hook": results}, tmp_path
    )

    content = output_path.read_text()
    assert count == 1
    assert "# ns_cost: outputs/navier_stokes/none/ono/final/seed_42" in content
    assert "diagnose: --benchmark navier_stokes --backbone ONO" in content


def test_effective_training_samples_honours_validation_split():
    assert effective_training_samples({"data": {"ntrain": 100}, "training": {"val_size": 20}}) == 80
    assert effective_training_samples({"data": {"ntrain": 100}, "training": {"val_size": 0}}) == 100


def test_ch5_provenance_table_uses_resolved_config(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark": {"name": "darcy_2d"},
                "experiment": {"backbone": "Transolver", "constraint": "none"},
                "data": {"ntrain": 1000},
                "training": {"num_epochs": 500, "val_size": 200},
            }
        )
    )
    artifact = ReportArtifact(
        name="cv_baseline",
        chapter=5,
        kind="tex_macros",
        output_subpath="macros/demo.tex",
        rows=[Row(run=RunRef("outputs/demo"), metric_key="metric/a", macro=r"\A")],
    )

    table = render_provenance_table(tmp_path, artifacts=(artifact,))

    assert "Darcy & Baseline & Transolver & none & 500 & 800" in table
    assert "Run directory" not in table
    assert "outputs/demo" not in table
