#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from physioagent.kg.scope.filter import DEFAULT_PROFILE, Scope, load_scope, profiles

KG_DIR = Path(__file__).resolve().parents[1]
RESOURCES = KG_DIR / "resources"
DEFAULT_OUT = KG_DIR / "data"

PRESCRIPTION_CHUNK = 2_000_000

MIN_CONCEPTS = 2


def resolve_mimic_root(explicit: Optional[str]) -> Path:
    for candidate in (explicit, os.environ.get("MIMIC_ROOT")):
        if candidate:
            root = Path(candidate).expanduser()
            if (root / "hosp").exists():
                return root
            raise SystemExit(f"no hosp/ under {root}")
    here = Path(__file__).resolve().parents[3] / "mimic-exploration" / "data"
    for root in sorted(here.glob("mimic-iv-*")):
        if (root / "hosp").exists():
            return root
    raise SystemExit(
        "MIMIC-IV not found. Pass --mimic-root or set $MIMIC_ROOT to the directory "
        "holding hosp/ (the full dump -- the slim build has no prescriptions table)."
    )


def _table_path(root: Path, module: str, name: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        path = root / module / f"{name}{suffix}"
        if path.exists():
            return path
    raise SystemExit(f"table {module}/{name} not found under {root}")


def read_table(root: Path, module: str, name: str, usecols: list[str],
               chunksize: Optional[int] = None):
    path = _table_path(root, module, name)
    return pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                       dtype=str, keep_default_na=False, na_filter=False)


def _norm_icd(code: str) -> str:
    return code.replace(".", "").replace(" ", "").upper()


MIN_BACKOFF_PREFIX = 3


def _backoff_ccs(code: str, table: dict[str, str]) -> Optional[str]:
    for end in range(len(code) - 1, MIN_BACKOFF_PREFIX - 1, -1):
        parent = code[:end]
        if parent in table:
            return table[parent]
    return None


PYHEALTH_MAPS = {
    "ICD9CM_to_CCSCM.csv": "pyhealth_icd9cm_to_ccscm.csv",
    "ICD10CM_to_CCSCM.csv": "pyhealth_icd10cm_to_ccscm.csv",
    "ICD9PROC_to_CCSPROC.csv": "pyhealth_icd9proc_to_ccsproc.csv",
    "ICD10PROC_to_CCSPROC.csv": "pyhealth_icd10proc_to_ccsproc.csv",
}
NDC_TO_ATC = "pyhealth_ndc_to_atc.csv"

MAP_SOURCE: dict[str, str] = {}


def load_icd_crosswalk(filename: str) -> dict[str, str]:
    preferred = RESOURCES / PYHEALTH_MAPS.get(filename, "")
    if preferred.name and preferred.exists():
        MAP_SOURCE[filename] = "pyhealth"
        path = preferred
    else:
        MAP_SOURCE[filename] = "shipped"
        path = RESOURCES / filename
    frame = pd.read_csv(path, dtype=str)
    icd_col, ccs_col = frame.columns[0], frame.columns[1]
    table: dict[str, str] = {}
    for icd, ccs in zip(frame[icd_col], frame[ccs_col]):
        if not isinstance(icd, str) or not isinstance(ccs, str):
            continue
        key = _norm_icd(icd)
        if key and key not in table:
            table[key] = ccs.split(",")[0].strip()
    return table


def load_ndc_to_atc3() -> dict[str, str]:
    path = RESOURCES / NDC_TO_ATC
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str)
    table: dict[str, str] = {}
    for ndc, atc in zip(frame.iloc[:, 0], frame.iloc[:, 1]):
        if isinstance(ndc, str) and isinstance(atc, str):
            key = ndc.strip()
            if key and key not in table:
                table[key] = atc.split(",")[0].strip()
    return table


def load_drug_patterns() -> list[tuple[re.Pattern[str], str]]:
    patterns: list[tuple[re.Pattern[str], str]] = []
    with (RESOURCES / "drug_to_atc3.csv").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("pattern,"):
                continue
            pattern, _, rest = line.partition(",")
            atc3 = rest.partition(",")[0].strip()
            if pattern and atc3:
                patterns.append((re.compile(pattern.strip(), re.IGNORECASE), atc3))
    if not patterns:
        raise SystemExit("drug_to_atc3.csv carries no usable patterns")
    return patterns


NON_THERAPEUTIC = re.compile(
    r"\bflush(es)?\b|\birrigation\b|\birrigant\b|\bfor irrigation\b", re.IGNORECASE)


class DrugMapper:
    def __init__(self, patterns: list[tuple[re.Pattern[str], str]], scope: Scope,
                 ndc_table: Optional[dict[str, str]] = None,
                 name_fallback: bool = True) -> None:
        self._patterns = patterns
        self._scope = scope
        self._ndc = ndc_table or {}
        self._name_fallback = name_fallback
        self._cache: dict[str, Optional[str]] = {}
        self.unmapped: Counter[str] = Counter()
        self.routes: Counter[str] = Counter()

    def _by_name(self, name: str) -> Optional[str]:
        if name in self._cache:
            return self._cache[name]
        hit = None
        for pattern, atc3 in self._patterns:
            if pattern.search(name):
                hit = atc3
                break
        self._cache[name] = hit
        return hit

    def __call__(self, drug: str, ndc: Optional[str] = None) -> Optional[str]:
        name = drug.strip().lower()
        if not name:
            return None
        if NON_THERAPEUTIC.search(name):
            self.routes["excluded_non_therapeutic"] += 1
            return None

        hit = None
        if ndc and self._ndc:
            hit = self._ndc.get(ndc.strip())
            if hit:
                self.routes["ndc"] += 1
        if hit is None and self._name_fallback:
            hit = self._by_name(name)
            if hit:
                self.routes["name"] += 1
        if hit is None:
            self.unmapped[name] += 1
            self.routes["unmapped"] += 1
            return None
        return hit if hit in self._scope.drugs else None


class IcdMisses:
    def __init__(self) -> None:
        self.codes: Counter[str] = Counter()
        self.by_version: Counter[str] = Counter()
        self.backoff_rows: Counter[str] = Counter()
        self.backoff_in_scope: Counter[str] = Counter()
        self.backoff_concepts: Counter[str] = Counter()

    def record(self, code: str, version: str, table: dict[str, str],
               in_scope: frozenset[str], namer) -> None:
        self.by_version[f"icd{version}"] += 1
        self.codes[f"{version}:{code}"] += 1
        ccs = _backoff_ccs(code, table)
        if ccs is None:
            return
        self.backoff_rows[f"icd{version}"] += 1
        if ccs in in_scope:
            self.backoff_in_scope[f"icd{version}"] += 1
            self.backoff_concepts[namer(ccs)] += 1

    def summary(self, top: int) -> dict:
        return {
            "rows_by_version": dict(self.by_version),
            "distinct_codes": len(self.codes),
            "top_codes": dict(self.codes.most_common(top)),
            "backoff_would_map": dict(self.backoff_rows),
            "backoff_would_be_in_scope": dict(self.backoff_in_scope),
            "backoff_concepts": dict(self.backoff_concepts.most_common()),
        }


def collect_diagnoses(root: Path, scope: Scope,
                      visits: dict[str, dict[str, set[str]]],
                      misses: Optional[IcdMisses] = None,
                      backoff: str = "icd10") -> dict[str, int]:
    cm9 = load_icd_crosswalk("ICD9CM_to_CCSCM.csv")
    cm10 = load_icd_crosswalk("ICD10CM_to_CCSCM.csv")
    stats = Counter()
    frame = read_table(root, "hosp", "diagnoses_icd",
                       ["subject_id", "hadm_id", "icd_code", "icd_version"])
    for subject, hadm, code, version in zip(frame["subject_id"], frame["hadm_id"],
                                            frame["icd_code"], frame["icd_version"]):
        stats["rows"] += 1
        table = cm10 if version.strip() == "10" else cm9
        norm = _norm_icd(code)
        ccs = table.get(norm)
        rescued = False
        if ccs is None:
            if backoff == "icd10" and version.strip() == "10":
                ccs = _backoff_ccs(norm, table)
                rescued = ccs is not None
            if ccs is None:
                stats["unmapped_icd"] += 1
                if misses is not None:
                    misses.record(norm, version.strip(), table, scope.conditions,
                                  scope.condition_name)
                continue
            stats["backoff_mapped"] += 1
        if ccs not in scope.conditions:
            continue
        if rescued:
            stats["backoff_in_scope"] += 1
        visit = visits.setdefault(hadm, {"subject_id": subject, "conditions": set(),
                                         "procedures": set(), "drugs": set()})
        visit["conditions"].add(ccs)
        stats["in_scope"] += 1
    return dict(stats)


def _note_time(visit: dict, kind: str, ccs: str, stamp: str) -> None:
    if not stamp:
        return
    times = visit.setdefault(f"{kind}_times", {})
    known = times.get(ccs)
    if known is None or stamp < known:
        times[ccs] = stamp


def collect_admittimes(root: Path, visits: dict[str, dict[str, set[str]]]) -> dict[str, int]:
    stats = Counter()
    frame = read_table(root, "hosp", "admissions", ["hadm_id", "admittime"])
    for hadm, admittime in zip(frame["hadm_id"], frame["admittime"]):
        stats["rows"] += 1
        visit = visits.get(hadm)
        if visit is None:
            continue
        visit["admittime"] = str(admittime).strip()
        stats["dated"] += 1
    stats["undated"] = len(visits) - stats["dated"]
    return dict(stats)


def collect_procedures(root: Path, scope: Scope,
                       visits: dict[str, dict[str, set[str]]],
                       misses: Optional[IcdMisses] = None) -> dict[str, int]:
    pr9 = load_icd_crosswalk("ICD9PROC_to_CCSPROC.csv")
    pr10 = load_icd_crosswalk("ICD10PROC_to_CCSPROC.csv")
    stats = Counter()
    frame = read_table(root, "hosp", "procedures_icd",
                       ["subject_id", "hadm_id", "icd_code", "icd_version", "chartdate"])
    for subject, hadm, code, version, chartdate in zip(
            frame["subject_id"], frame["hadm_id"], frame["icd_code"],
            frame["icd_version"], frame["chartdate"]):
        stats["rows"] += 1
        table = pr10 if version.strip() == "10" else pr9
        norm = _norm_icd(code)
        ccs = table.get(norm)
        if ccs is None:
            stats["unmapped_icd"] += 1
            if misses is not None:
                misses.record(norm, version.strip(), table, scope.procedures,
                              scope.procedure_name)
            continue
        if ccs not in scope.procedures:
            continue
        visit = visits.setdefault(hadm, {"subject_id": subject, "conditions": set(),
                                         "procedures": set(), "drugs": set()})
        visit["procedures"].add(ccs)
        _note_time(visit, "procedure", ccs, str(chartdate).strip())
        stats["in_scope"] += 1
        if not str(chartdate).strip():
            stats["no_chartdate"] += 1
    return dict(stats)


def collect_drugs(root: Path, mapper: DrugMapper,
                  visits: dict[str, dict[str, set[str]]],
                  chunksize: int = PRESCRIPTION_CHUNK) -> dict[str, int]:
    stats = Counter()
    for chunk in read_table(root, "hosp", "prescriptions",
                            ["subject_id", "hadm_id", "drug", "ndc", "starttime"],
                            chunksize=chunksize):
        for subject, hadm, drug, ndc, starttime in zip(
                chunk["subject_id"], chunk["hadm_id"], chunk["drug"], chunk["ndc"],
                chunk["starttime"]):
            stats["rows"] += 1
            if not hadm:
                continue
            atc3 = mapper(drug, ndc)
            if atc3 is None:
                continue
            visit = visits.setdefault(hadm, {"subject_id": subject, "conditions": set(),
                                             "procedures": set(), "drugs": set()})
            visit["drugs"].add(atc3)
            _note_time(visit, "drug", atc3, str(starttime).strip())
            stats["in_scope"] += 1
            if not str(starttime).strip():
                stats["no_starttime"] += 1
        print(f"  prescriptions: {stats['rows']:,} rows read, "
              f"{stats['in_scope']:,} in scope", flush=True)
    return dict(stats)


def iter_records(visits: dict[str, dict[str, set[str]]], scope: Scope) -> Iterator[dict]:
    for hadm, visit in visits.items():
        conditions = sorted(scope.condition_name(c) for c in visit["conditions"])
        procedures = sorted(scope.procedure_name(c) for c in visit["procedures"])
        drugs = sorted(scope.drug_name(c) for c in visit["drugs"])
        if len(conditions) + len(procedures) + len(drugs) < MIN_CONCEPTS:
            continue
        record = {
            "hadm_id": hadm,
            "subject_id": visit["subject_id"],
            "admittime": visit.get("admittime", ""),
            "conditions": conditions,
            "procedures": procedures,
            "drugs": drugs,
        }
        for kind, namer in (("procedure", scope.procedure_name), ("drug", scope.drug_name)):
            times = visit.get(f"{kind}_times")
            if times:
                record[f"{kind}_times"] = {namer(code): stamp
                                           for code, stamp in sorted(times.items())}
        yield record


def _print_misses(indent: str, misses: IcdMisses) -> None:
    if not misses.by_version:
        return
    by_v = ", ".join(f"{k} {v:,}" for k, v in sorted(misses.by_version.items()))
    print(f"{indent}unmapped by vocabulary: {by_v}  "
          f"({len(misses.codes):,} distinct codes)")
    if misses.backoff_rows:
        catch = ", ".join(f"{k} {v:,}" for k, v in sorted(misses.backoff_rows.items()))
        scoped = ", ".join(f"{k} {v:,}"
                           for k, v in sorted(misses.backoff_in_scope.items())) or "none"
        print(f"{indent}a parent-prefix backoff would map: {catch}; "
              f"landing in scope: {scoped}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic-root", help="directory holding hosp/ (default: $MIMIC_ROOT)")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=profiles())
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--chunksize", type=int, default=PRESCRIPTION_CHUNK)
    ap.add_argument("--report-unmapped", type=int, default=40, metavar="N",
                    help="write the N most frequent unmatched drug names AND the N most "
                         "frequent unmatched ICD codes into the report. This is how "
                         "drug_to_atc3.csv gets extended, and how the crosswalks' FY2019 "
                         "vintage becomes a number -- read it after the first real run "
                         "rather than guessing at spellings or at how much is missing.")
    ap.add_argument("--no-name-fallback", action="store_true",
                    help="map drugs by NDC only, dropping rows whose ndc is null -- "
                         "TRACER's behaviour exactly. Off by default because MIMIC's ndc "
                         "column is sparse and the name table recovers those rows.")
    ap.add_argument("--dx-backoff", choices=("icd10", "none"), default="icd10",
                    help="map an ICD-10 code the FY2019 CCS crosswalk does not hold "
                         "through its nearest parent prefix (N1832 -> N183 -> CCS 158). "
                         "On by default: MIMIC-IV 3.1 reaches 2022 and the stage-3a/3b "
                         "split of N18.3 -- this project's core concept -- is one of the "
                         "codes CMS added after the crosswalk's vintage. Held out "
                         "against the shipped table it agrees on 99.63%% of ICD-10 "
                         "codes. `none` reproduces every run before 2026-09-20. There is "
                         "no icd9 setting, deliberately -- see IcdMisses.")
    ap.add_argument("--skip-drugs", action="store_true",
                    help="diagnoses and procedures only; for a fast structural check")
    args = ap.parse_args()

    root = resolve_mimic_root(args.mimic_root)
    scope = load_scope(args.profile)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"mimic root : {root}")
    print(f"profile    : {args.profile}  ({scope.n_concepts} concepts)")
    print(f"out dir    : {out_dir}\n")

    visits: dict[str, dict[str, set[str]]] = {}
    dx_misses, px_misses = IcdMisses(), IcdMisses()

    print("diagnoses_icd ...", flush=True)
    dx_stats = collect_diagnoses(root, scope, visits, dx_misses, args.dx_backoff)
    print(f"  {dx_stats.get('rows', 0):,} rows, {dx_stats.get('in_scope', 0):,} in scope, "
          f"{dx_stats.get('unmapped_icd', 0):,} ICD codes not in the CCS crosswalk")
    if args.dx_backoff != "none":
        print(f"  parent-prefix backoff ({args.dx_backoff}) rescued "
              f"{dx_stats.get('backoff_mapped', 0):,} rows, "
              f"{dx_stats.get('backoff_in_scope', 0):,} of them in scope")
    _print_misses("  ", dx_misses)

    print("procedures_icd ...", flush=True)
    px_stats = collect_procedures(root, scope, visits, px_misses)
    print(f"  {px_stats.get('rows', 0):,} rows, {px_stats.get('in_scope', 0):,} in scope, "
          f"{px_stats.get('unmapped_icd', 0):,} ICD codes not in the CCS crosswalk")
    _print_misses("  ", px_misses)

    ndc_table = load_ndc_to_atc3()
    print(f"drug route : NDC table {'present' if ndc_table else 'ABSENT'}"
          f"{'' if ndc_table else ' (run kg.ehr.export_pyhealth_maps for TRACER parity)'}"
          f", name fallback {'off' if args.no_name_fallback else 'on'}")
    mapper = DrugMapper(load_drug_patterns(), scope, ndc_table,
                        name_fallback=not args.no_name_fallback)
    rx_stats: dict[str, int] = {}
    if args.skip_drugs:
        print("prescriptions ... SKIPPED (--skip-drugs)")
    else:
        print("prescriptions ...", flush=True)
        rx_stats = collect_drugs(root, mapper, visits, args.chunksize)

    print("admissions ...", flush=True)
    adm_stats = collect_admittimes(root, visits)
    print(f"  dated {adm_stats.get('dated', 0):,} of {len(visits):,} admissions")

    visits_path = out_dir / "visits_all.jsonl"
    kept = 0
    concept_freq: Counter[str] = Counter()
    with visits_path.open("w", encoding="utf-8") as handle:
        for record in iter_records(visits, scope):
            handle.write(json.dumps(record) + "\n")
            kept += 1
            concept_freq.update(record["conditions"] + record["procedures"]
                                + record["drugs"])

    report = {
        "profile": args.profile,
        "scope": scope.summary(),
        "mimic_root": str(root),
        "admissions_seen": len(visits),
        "admissions_written": kept,
        "admissions_dropped_too_few_concepts": len(visits) - kept,
        "diagnoses": dx_stats,
        "dx_backoff": args.dx_backoff,
        "diagnoses_unmapped": dx_misses.summary(args.report_unmapped),
        "procedures": px_stats,
        "procedures_unmapped": px_misses.summary(args.report_unmapped),
        "prescriptions": rx_stats,
        "concepts_observed": len(concept_freq),
        "concepts_never_seen": sorted(set(scope.names()) - set(concept_freq)),
        "concept_frequency": dict(concept_freq.most_common()),
        "unmapped_drugs_top": dict(mapper.unmapped.most_common(args.report_unmapped)),
        "unmapped_drug_names": len(mapper.unmapped),
        "mapping_tables": dict(MAP_SOURCE),
        "ndc_table_present": bool(ndc_table),
        "drug_routes": dict(mapper.routes),
    }
    report_path = out_dir / "scope_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nwrote {kept:,} admissions -> {visits_path}")
    print(f"      {len(visits) - kept:,} dropped for fewer than {MIN_CONCEPTS} concepts")
    print(f"      {len(concept_freq)}/{scope.n_concepts} concepts actually observed")
    if mapper.routes:
        print(f"      drug rows by route: {dict(mapper.routes)}")
    if report["concepts_never_seen"]:
        print(f"      never seen: {', '.join(report['concepts_never_seen'][:8])}"
              + (" ..." if len(report["concepts_never_seen"]) > 8 else ""))
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
