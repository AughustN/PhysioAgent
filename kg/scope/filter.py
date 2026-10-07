from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Optional

import yaml

KG_DIR = Path(__file__).resolve().parents[1]
RESOURCES = KG_DIR / "resources"
ONTOLOGY = Path(__file__).resolve().parent / "ontology.yaml"

DEFAULT_PROFILE = "default"

ATC_LEVEL = 3.0


class ScopeError(ValueError):
    pass


@dataclass(frozen=True)
class Scope:
    profile: str
    conditions: frozenset[str]
    procedures: frozenset[str]
    drugs: frozenset[str]
    seed_terms: Mapping[str, tuple[str, ...]]
    _condition_names: Mapping[str, str]
    _procedure_names: Mapping[str, str]
    _drug_names: Mapping[str, str]

    def condition_name(self, code: str) -> str:
        return self._condition_names[str(code)]

    def procedure_name(self, code: str) -> str:
        return self._procedure_names[str(code)]

    def drug_name(self, code: str) -> str:
        return self._drug_names[str(code)]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted({*self._condition_names.values(),
                             *self._procedure_names.values(),
                             *self._drug_names.values()}))

    def seed_concept_sets(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(v) for v in self.seed_terms.values())

    @property
    def n_concepts(self) -> int:
        return len(self.conditions) + len(self.procedures) + len(self.drugs)

    def summary(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "conditions": len(self.conditions),
            "procedures": len(self.procedures),
            "drugs": len(self.drugs),
            "concepts": self.n_concepts,
            "distinct_names": len(self.names()),
            "seed_themes": len(self.seed_terms),
            "seed_terms": sum(len(v) for v in self.seed_terms.values()),
        }


@lru_cache(maxsize=None)
def _ccs_table(filename: str) -> dict[str, str]:
    table: dict[str, str] = {}
    with (RESOURCES / filename).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[row["code"].strip()] = row["name"].strip().lower()
    return table


@lru_cache(maxsize=None)
def _atc_table() -> dict[str, str]:
    table: dict[str, str] = {}
    with (RESOURCES / "ATC.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row["level"] or 0) == ATC_LEVEL:
                table[row["code"].strip()] = row["name"].strip().lower()
    return table


def _check(kind: str, declared: Mapping, table: Mapping[str, str],
           group: str, errors: list[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for code, name in declared.items():
        code = str(code).strip()
        if code not in table:
            errors.append(f"{group}.{kind}: {code!r} is not a code in the {kind} table")
            continue
        if str(name).strip().lower() != table[code]:
            errors.append(
                f"{group}.{kind}: {code!r} is {table[code]!r}, not {str(name).strip().lower()!r}"
            )
            continue
        resolved[code] = table[code]
    return resolved


def load_ontology(path: Optional[Path] = None) -> dict:
    with (path or ONTOLOGY).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_scope(profile: str = DEFAULT_PROFILE, path: Optional[Path] = None) -> Scope:
    doc = load_ontology(path)
    profiles = doc.get("profiles") or {}
    if profile not in profiles:
        raise ScopeError(f"unknown profile {profile!r}; have {sorted(profiles)}")
    spec = profiles[profile]

    groups = doc.get("groups") or {}
    unknown = [g for g in spec["groups"] if g not in groups]
    if unknown:
        raise ScopeError(f"profile {profile!r} names groups that do not exist: {unknown}")

    conditions: dict[str, str] = {}
    procedures: dict[str, str] = {}
    drugs: dict[str, str] = {}
    errors: list[str] = []

    for name in spec["groups"]:
        group = groups[name] or {}
        conditions.update(_check("conditions", group.get("conditions") or {},
                                 _ccs_table("CCSCM.csv"), name, errors))
        procedures.update(_check("procedures", group.get("procedures") or {},
                                 _ccs_table("CCSPROC.csv"), name, errors))
        drugs.update(_check("drugs", group.get("drugs") or {},
                            _atc_table(), name, errors))

    if errors:
        raise ScopeError(
            f"ontology.yaml disagrees with the code tables ({len(errors)} problems):\n  "
            + "\n  ".join(errors)
        )

    seeds: dict[str, tuple[str, ...]] = {}
    if spec.get("seed_terms"):
        for theme, terms in (doc.get("seed_terms") or {}).items():
            seeds[theme] = tuple(str(t).strip().lower() for t in terms)

    return Scope(
        profile=profile,
        conditions=frozenset(conditions),
        procedures=frozenset(procedures),
        drugs=frozenset(drugs),
        seed_terms=seeds,
        _condition_names=conditions,
        _procedure_names=procedures,
        _drug_names=drugs,
    )


def profiles(path: Optional[Path] = None) -> tuple[str, ...]:
    return tuple(sorted((load_ontology(path).get("profiles") or {})))


def umls_gate(path: Optional[Path] = None) -> dict[str, tuple[str, ...]]:
    gate = (load_ontology(path).get("umls") or {})
    return {
        "semantic_types": tuple(gate.get("semantic_types") or ()),
        "mesh_prefixes": tuple(gate.get("mesh_prefixes") or ()),
    }


def _fmt(values: Iterable[str]) -> str:
    return ", ".join(sorted(values))


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Print a resolved ontology profile.")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=profiles())
    ap.add_argument("--names", action="store_true", help="list every concept name")
    args = ap.parse_args()

    scope = load_scope(args.profile)
    print(json.dumps(scope.summary(), indent=2))
    if args.names:
        print("\nconditions:", _fmt(scope._condition_names.values()))
        print("\nprocedures:", _fmt(scope._procedure_names.values()))
        print("\ndrugs:", _fmt(scope._drug_names.values()))
        for theme, terms in scope.seed_terms.items():
            print(f"\nseed[{theme}]:", _fmt(terms))


if __name__ == "__main__":
    main()
