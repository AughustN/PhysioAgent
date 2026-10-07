#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

KG_DIR = Path(__file__).resolve().parents[1]
RESOURCES = KG_DIR / "resources"

MAPPINGS = [
    ("ICD9CM", "CCSCM", None),
    ("ICD10CM", "CCSCM", None),
    ("ICD9PROC", "CCSPROC", None),
    ("ICD10PROC", "CCSPROC", None),
    ("NDC", "ATC", 3),
]


def export_one(source: str, target: str, level: int | None, out_dir: Path) -> int:
    from pyhealth.medcode import CrossMap

    crossmap = CrossMap.load(source, target)
    table = getattr(crossmap, "mapping", None)
    if table is None:
        raise SystemExit(f"pyhealth CrossMap({source}->{target}) exposes no .mapping; "
                         "the API changed and this exporter needs updating")

    import pyhealth

    path = out_dir / f"pyhealth_{source.lower()}_to_{target.lower()}.csv"
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([source, target, "pyhealth_version"])
        version = getattr(pyhealth, "__version__", "unknown")
        for code, targets in sorted(table.items()):
            values = targets if isinstance(targets, (list, tuple)) else [targets]
            values = [str(v) for v in values if v]
            if level is not None:
                values = [v[:4] for v in values if len(v) >= 4]
            values = sorted(set(values))
            if not values:
                continue
            writer.writerow([str(code), ",".join(values), version])
            rows += 1
    print(f"  {source} -> {target}: {rows:,} codes -> {path.name}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(RESOURCES))
    ap.add_argument("--only", nargs="*", metavar="SRC",
                    help="export only these source vocabularies")
    args = ap.parse_args()

    try:
        import pyhealth  # noqa: F401
    except ImportError:
        raise SystemExit(
            "pyhealth is not installed. This script is a ONE-OFF export -- install it "
            "here (pip install pyhealth==1.1.6, the version TRACER pins), run this, "
            "commit the CSVs, and nothing else in the pipeline needs pyhealth again.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [m for m in MAPPINGS if not args.only or m[0] in args.only]
    print(f"exporting {len(wanted)} mappings to {out_dir}")
    total = sum(export_one(*mapping, out_dir) for mapping in wanted)
    print(f"\n{total:,} rows written. Commit these; build_visits picks them up "
          f"automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
