"""CLI entrypoint for the report build system.

Usage:
    python -m scripts.reporting.build                              # all artifacts
    python -m scripts.reporting.build --chapter 5                  # one chapter
    python -m scripts.reporting.build --name cv_constrained        # one artifact
    python -m scripts.reporting.build --output-dir ../fyp_report/generated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.emit_tex import write_artifact
from .registry import ARTIFACTS


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("could not locate repo root (no pyproject.toml ancestor)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reporting.build")
    parser.add_argument(
        "--chapter", type=int, default=None, help="only build artifacts for this chapter"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="only build the artifact with this name"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="root directory for generated files (default: artifacts/report/)",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    output_dir = args.output_dir or (repo_root / "artifacts/report")
    output_dir = output_dir.resolve()

    selected = [
        a
        for a in ARTIFACTS
        if (args.chapter is None or a.chapter == args.chapter)
        and (args.name is None or a.name == args.name)
    ]
    if not selected:
        print("no artifacts matched filters", file=sys.stderr)
        return 1

    print(f"output dir: {output_dir}")
    n_ok = n_missing = 0
    for artifact in selected:
        if artifact.kind != "tex_macros":
            print(f"  skip (kind={artifact.kind}): {artifact.name}")
            continue
        results = write_artifact(artifact, repo_root, output_dir)
        ok = sum(r.ok for r in results)
        missing = len(results) - ok
        n_ok += ok
        n_missing += missing
        status = "OK" if missing == 0 else f"{missing} TBD"
        print(f"  {artifact.name:<20} {ok}/{len(results)} cells  [{status}]")
        for r in results:
            if not r.ok:
                print(f"    - {r.macro}: {r.source}")

    print(f"\ntotal: {n_ok} resolved, {n_missing} TBD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
