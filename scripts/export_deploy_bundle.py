"""Export the small, serving-only artefact bundle used by the Docker image.

The API (:mod:`agrik.api`) and the dashboard (:mod:`agrik.dashboard`) read
precomputed artefacts and never re-ingest, so a deployment only needs a few
megabytes of panels and model files - not the ~19 GB satellite-granule cache
under ``data/raw/external/``.

This script copies exactly the serving inputs into a bundle directory
(default ``deploy/seed/``) laid out like the repository root, so the
container's relative paths (``data/``, ``models/``) resolve unchanged:

    deploy/seed/data/raw/*.csv          dataset panels shown in the Data tab
    deploy/seed/data/processed/**       master panel + validation reports
    deploy/seed/data/features/**        feature store consumed by API/UI
    deploy/seed/models/**               joblibs, model card, comparison, predictions

Run ``python -m agrik`` first (generates the artefacts), then this script,
then ``docker build .``. Everything copied is provenance-stamped real-feed
output - the bundle carries the same ``data_is_synthetic`` flags as the
dashboard, so a deployment can never silently mislabel synthetic data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from agrik.settings import get_settings

# Subtrees that the serving layer reads, relative to their settings root.
# ``raw`` is CSVs only: the external granule cache is never deployable.
_RAW_GLOBS = ("*.csv",)
_FULL_TREE_DIRS = ("processed", "features")


def _collect_source_files(data_root: Path, models_root: Path) -> list[tuple[Path, str]]:
    """Return (source_path, bundle_relative_target) for every serving artefact."""
    files: list[tuple[Path, str]] = []
    raw = data_root / "raw"
    for pattern in _RAW_GLOBS:
        for path in sorted(raw.glob(pattern)):
            if path.is_file() and path.name != ".gitkeep":
                files.append((path, f"data/raw/{path.name}"))
    for sub in _FULL_TREE_DIRS:
        root = data_root / sub
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.name != ".gitkeep":
                    rel = path.relative_to(data_root).as_posix()
                    files.append((path, f"data/{rel}"))
    if models_root.is_dir():
        for path in sorted(models_root.rglob("*")):
            if path.is_file() and path.name != ".gitkeep":
                rel = path.relative_to(models_root).as_posix()
                files.append((path, f"models/{rel}"))
    return files


def export_bundle(out_dir: Path) -> dict:
    """Copy serving artefacts into ``out_dir`` (repo-root layout). Returns summary."""
    s = get_settings()
    required = s.features_dir / "features_panel.csv", s.models_dir / "model_card.json"
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing artefact(s): " + ", ".join(missing)
            + "\nRun `python -m agrik` first, then re-run this script."
        )

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    files = _collect_source_files(s.data_root, s.models_dir)
    total_bytes = 0
    for src, rel_target in files:
        dest = out_dir / rel_target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        total_bytes += src.stat().st_size

    sources: dict = {}
    manifest_path = s.features_dir / "features_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources = manifest.get("dataset_sources", {}) or {}
    # Same rule the serving layer uses: a dataset falling back to synthetic is
    # flagged in its provenance value; never let a bundle hide that.
    is_synthetic = any("synthetic" in str(v).lower() for v in sources.values())

    return {
        "out": str(out_dir),
        "files": len(files),
        "bytes": total_bytes,
        "dataset_sources": sources,
        "data_is_synthetic": is_synthetic,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("deploy") / "seed",
        help="Bundle directory (default: deploy/seed).",
    )
    args = parser.parse_args(argv)

    summary = export_bundle(args.out)
    mb = summary["bytes"] / (1024 * 1024)
    print(f"Bundle written to {summary['out']} ({summary['files']} files, {mb:.2f} MB)")
    print(f"Dataset sources : {json.dumps(summary['dataset_sources'])}")
    print(f"Data is synthetic: {summary['data_is_synthetic']}")
    if summary["data_is_synthetic"]:
        print("WARNING: the bundled artefacts are SYNTHETIC - build from real feeds first.")
    print("Next: docker build . && docker compose up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
