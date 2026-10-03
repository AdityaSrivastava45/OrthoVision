"""CLI: inspect, manifest, study-summary, visualize."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from orthovision.data.catalog import KneeMRIDataset
from orthovision.data.config import load_data_config
from orthovision.data.logutil import configure_logging, get_logger
from orthovision.data.manifest import write_manifests
from orthovision.data.visualize import visualize_study


def _dataset(args: argparse.Namespace) -> KneeMRIDataset:
    cfg = load_data_config(args.config)
    configure_logging(cfg.logging_level, cfg.log_file)
    return KneeMRIDataset(cfg)


def cmd_inspect(args: argparse.Namespace) -> int:
    ds = _dataset(args)
    stats = ds.scan()
    log = get_logger("cli")
    log.info("inspect stats: %s", stats)
    print(json.dumps(stats.__dict__, indent=2))
    if args.manifest:
        paths = write_manifests(ds)
        print("manifests:", {k: str(v) for k, v in paths.items()})
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    ds = _dataset(args)
    ds.scan()
    dest = Path(args.dest) if args.dest else None
    paths = write_manifests(ds, dest)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


def cmd_study_summary(args: argparse.Namespace) -> int:
    ds = _dataset(args)
    summary = ds.study_summary(args.study)
    payload = {
        "study_id": summary.study_id,
        "n_series": summary.n_series,
        "n_valid_series": summary.n_valid_series,
        "n_warning_series": summary.n_warning_series,
        "n_invalid_series": summary.n_invalid_series,
        "available_planes": summary.available_planes,
        "total_instances": summary.total_instances,
        "validation_warnings": summary.validation_warnings,
        "series": [s.__dict__ for s in summary.series],
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    ds = _dataset(args)
    study = ds.get_study(args.study)
    dest = Path(args.dest) if args.dest else ds.config.viz_dir / args.study
    paths = visualize_study(study, dest, n_slices=args.n_slices, include_invalid=args.include_invalid)
    print(json.dumps([str(p) for p in paths], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orthovision-data", description="OrthoVision DICOM inspection pipeline")
    p.add_argument("--config", default="configs/data.yaml", help="Path to data YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    insp = sub.add_parser("inspect", help="Scan dataset and print counts")
    insp.add_argument("--manifest", action="store_true", help="Also write manifests")
    insp.set_defaults(func=cmd_inspect)

    man = sub.add_parser("manifest", help="Write CSV/JSON manifests")
    man.add_argument("--dest", default=None)
    man.set_defaults(func=cmd_manifest)

    summ = sub.add_parser("study-summary", help="Structured summary for one study")
    summ.add_argument("--study", required=True, help="StudyInstanceUID")
    summ.set_defaults(func=cmd_study_summary)

    viz = sub.add_parser("visualize", help="Save slice mosaics for a study")
    viz.add_argument("--study", required=True)
    viz.add_argument("--dest", default=None)
    viz.add_argument("--n-slices", type=int, default=5)
    viz.add_argument("--include-invalid", action="store_true")
    viz.set_defaults(func=cmd_visualize)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
