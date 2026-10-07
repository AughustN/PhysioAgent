#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from physioagent.kg.ehr.build_visits import (
    DEFAULT_OUT, MIN_CONCEPTS, NON_THERAPEUTIC, PRESCRIPTION_CHUNK, DrugMapper, _backoff_ccs,
    _norm_icd, load_drug_patterns, load_icd_crosswalk, load_ndc_to_atc3, read_table,
    resolve_mimic_root,
)
from physioagent.kg.scope.filter import DEFAULT_PROFILE, load_scope, profiles

RX_COLS = ["subject_id", "hadm_id", "starttime", "drug_type", "drug", "ndc",
           "prod_strength", "dose_val_rx", "dose_unit_rx", "route"]


def pick_subjects(root: Path, n: int, seed: int, min_admissions: int) -> list[str]:
    frame = read_table(root, "hosp", "admissions", ["subject_id"])
    counts = frame["subject_id"].value_counts()
    pool = sorted(counts[counts >= min_admissions].index)
    return random.Random(seed).sample(pool, n)


def load_admissions(root: Path, subjects: set[str]) -> dict[str, list[dict]]:
    cols = ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"]
    frame = read_table(root, "hosp", "admissions", cols)
    frame = frame[frame["subject_id"].isin(subjects)].sort_values("admittime")
    out: dict[str, list[dict]] = defaultdict(list)
    for row in frame.to_dict("records"):
        out[row["subject_id"]].append(row)
    return out


def titles(root: Path, table: str) -> dict[tuple[str, str], str]:
    frame = read_table(root, "hosp", table, ["icd_code", "icd_version", "long_title"])
    return {(v.strip(), _norm_icd(c)): t
            for c, v, t in zip(frame["icd_code"], frame["icd_version"], frame["long_title"])}


def rows_by_hadm(root: Path, table: str, cols: list[str], hadms: set[str],
                 chunksize: int | None = None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    chunks = read_table(root, "hosp", table, cols, chunksize=chunksize)
    for chunk in (chunks if chunksize else [chunks]):
        for row in chunk[chunk["hadm_id"].isin(hadms)].to_dict("records"):
            out[row["hadm_id"]].append(row)
    return out


def stored_visits(path: Path, hadms: set[str]) -> dict[str, dict]:
    found = {}
    if not path.exists():
        return found
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if str(record["hadm_id"]) in hadms:
                found[str(record["hadm_id"])] = record
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic-root", help="directory holding hosp/ (default: $MIMIC_ROOT)")
    ap.add_argument("--visits", default=str(DEFAULT_OUT / "visits_all.jsonl"))
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=profiles())
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--min-admissions", type=int, default=1,
                    help="only sample patients with at least this many admissions")
    ap.add_argument("--subject", nargs="+", help="show these patients instead of random ones")
    ap.add_argument("--all-rows", action="store_true",
                    help="print every prescription row instead of grouping them")
    args = ap.parse_args()

    root = resolve_mimic_root(args.mimic_root)
    scope = load_scope(args.profile)
    subjects = args.subject or pick_subjects(root, args.n, args.seed, args.min_admissions)
    admissions = load_admissions(root, set(subjects))
    hadms = {a["hadm_id"] for rows in admissions.values() for a in rows}
    print(f"{len(subjects)} patients, {len(hadms)} admissions; reading raw rows under {root} ...",
          flush=True)

    dx = rows_by_hadm(root, "diagnoses_icd", ["hadm_id", "seq_num", "icd_code", "icd_version"],
                      hadms)
    pr = rows_by_hadm(root, "procedures_icd",
                      ["hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"], hadms)
    rx = rows_by_hadm(root, "prescriptions", RX_COLS, hadms, chunksize=PRESCRIPTION_CHUNK)
    dx_titles, pr_titles = titles(root, "d_icd_diagnoses"), titles(root, "d_icd_procedures")
    dx_tables = {"9": load_icd_crosswalk("ICD9CM_to_CCSCM.csv"),
                 "10": load_icd_crosswalk("ICD10CM_to_CCSCM.csv")}
    pr_tables = {"9": load_icd_crosswalk("ICD9PROC_to_CCSPROC.csv"),
                 "10": load_icd_crosswalk("ICD10PROC_to_CCSPROC.csv")}
    ndc_table = load_ndc_to_atc3()
    mapper = DrugMapper(load_drug_patterns(), scope, ndc_table)
    stored = stored_visits(Path(args.visits), hadms)

    def drug_atc3(drug: str, ndc: str) -> tuple[str | None, str]:
        name = drug.strip().lower()
        if NON_THERAPEUTIC.search(name):
            return None, "excluded: flush/irrigation"
        hit = ndc_table.get(ndc.strip()) if ndc.strip() and ndc_table else None
        return hit or mapper._by_name(name), ""

    for subject in subjects:
        rows = admissions.get(subject, [])
        print("\n" + "#" * 100)
        print(f"PATIENT subject_id {subject} · {len(rows)} admissions in hosp/admissions")
        for adm in rows:
            hadm = adm["hadm_id"]
            concepts = {"conditions": set(), "procedures": set(), "drugs": set()}
            print("\n" + "=" * 100)
            print(f"hadm_id {hadm} · {adm['admission_type']} · "
                  f"{adm['admittime']} -> {adm['dischtime']}")

            print(f"\n[diagnoses_icd] {len(dx[hadm])} rows   seq  code  -> CCS")
            for row in sorted(dx[hadm], key=lambda r: int(r["seq_num"] or 0)):
                version, code = row["icd_version"].strip(), _norm_icd(row["icd_code"])
                ccs, via = dx_tables[version].get(code), ""
                if ccs is None and version == "10":
                    ccs = _backoff_ccs(code, dx_tables[version])
                    via = "backoff" if ccs else ""
                kept = ccs is not None and ccs in scope.conditions
                if kept:
                    concepts["conditions"].add(scope.condition_name(ccs))
                print(f"  {row['seq_num']:>3}  ICD{version:<3}{code:<9}"
                      f"{dx_titles.get((version, code), '')[:46]:<47}-> CCS {ccs or '—':<5}"
                      f"{via:<8}{'KEPT ' + scope.condition_name(ccs) if kept else '·'}")

            print(f"\n[procedures_icd] {len(pr[hadm])} rows   seq  chartdate  code  -> CCS")
            for row in sorted(pr[hadm], key=lambda r: int(r["seq_num"] or 0)):
                version, code = row["icd_version"].strip(), _norm_icd(row["icd_code"])
                ccs = pr_tables[version].get(code)
                kept = ccs is not None and ccs in scope.procedures
                if kept:
                    concepts["procedures"].add(scope.procedure_name(ccs))
                print(f"  {row['seq_num']:>3}  {row['chartdate']:<11}ICD{version:<3}{code:<9}"
                      f"{pr_titles.get((version, code), '')[:40]:<41}-> CCS {ccs or '—':<5}"
                      f"{'KEPT ' + scope.procedure_name(ccs) if kept else '·'}")

            drugs = sorted(rx[hadm], key=lambda r: r["starttime"])
            print(f"\n[prescriptions] {len(drugs)} rows   (NDC route "
                  f"{'ON' if ndc_table else 'OFF: drug text only'})")
            groups: dict[tuple, list[dict]] = defaultdict(list)
            for row in drugs:
                key = (row["drug"], row["ndc"], row["route"]) if not args.all_rows \
                    else (row["drug"], row["ndc"], row["route"], id(row))
                groups[key].append(row)
            for key, members in sorted(groups.items(), key=lambda kv: kv[1][0]["starttime"]):
                first = members[0]
                atc3, note = drug_atc3(first["drug"], first["ndc"])
                kept = atc3 is not None and atc3 in scope.drugs
                if kept:
                    concepts["drugs"].add(scope.drug_name(atc3))
                dose = f"{first['dose_val_rx']} {first['dose_unit_rx']}".strip()
                print(f"  {first['starttime'][:16]:<17}{first['drug_type']:<6}"
                      f"{first['drug'][:34]:<35}ndc {first['ndc'] or '—':<12}"
                      f"{first['prod_strength'][:18]:<19}{dose[:12]:<13}{first['route'][:8]:<9}"
                      f"x{len(members):<4}-> ATC {atc3 or '—':<5}"
                      f"{'KEPT ' + scope.drug_name(atc3) if kept else (note or '·')}")

            total = sum(len(v) for v in concepts.values())
            print(f"\n=> CONCEPTS extracted: {total}")
            for kind in ("conditions", "procedures", "drugs"):
                print(f"   {kind:<11}{sorted(concepts[kind])}")
            record = stored.get(hadm)
            if record is not None:
                same = all(set(record.get(k, [])) == concepts[k] for k in concepts)
                print(f"   visits_all.jsonl: present, "
                      f"{'identical' if same else 'DIFFERS -- stored: ' + json.dumps({k: record.get(k, []) for k in concepts})}")
            elif total < MIN_CONCEPTS:
                print(f"   visits_all.jsonl: absent (< {MIN_CONCEPTS} concepts, cannot form a "
                      "co-occurrence pair) -> this admission adds nothing to the graph")
            else:
                print("   visits_all.jsonl: absent although it has enough concepts -- "
                      "file missing or built with another profile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
