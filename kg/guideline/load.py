#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import yaml

from physioagent.utils.rate_limit import RateLimiter
from physioagent.kg.sources.pubmed_source import (
    NCBI_PER_MIN_NO_KEY,
    NCBI_PER_MIN_WITH_KEY,
    Entrez,
    _get,
    env_value,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifest.yaml"
DAILYMED = "https://dailymed.nlm.nih.gov/dailymed/services/v2/"
HL7 = "{urn:hl7-org:v3}"
LOINC = "2.16.840.1.113883.6.1"
SECTION_CODES = {"DOSAGE AND ADMINISTRATION": "34068-7"}
ACCESS_TYPES = ("manual_pdf", "pmc_api", "dailymed_api")


class LoadError(RuntimeError):
    pass


@dataclass
class Result:
    id: str
    access: str
    status: str
    files: list[str] = field(default_factory=list)
    sha256: Optional[str] = None
    notes: list[str] = field(default_factory=list)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def bundle_sha256(paths: Iterable[Path]) -> str:
    return sha256_bytes("\n".join(sha256_file(p) for p in paths).encode())


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    ids = [s["id"] for s in manifest["sources"]]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise LoadError(f"duplicate source ids: {duplicates}")
    for source in manifest["sources"]:
        if source["access"] not in ACCESS_TYPES:
            raise LoadError(f"{source['id']}: unknown access {source['access']!r}")
    return manifest


def set_lock(manifest_text: str, source_id: str, retrieved_on: str, sha256: str) -> str:
    start = manifest_text.index(f"  - id: {source_id}\n")
    end = manifest_text.find("\n  - id: ", start + 1)
    end = len(manifest_text) if end < 0 else end
    block = manifest_text[start:end]
    block = re.sub(r"(?m)^    retrieved_on: .*$", f"    retrieved_on: {retrieved_on}", block)
    block = re.sub(r"(?m)^    sha256: .*$", f"    sha256: {sha256}", block)
    return manifest_text[:start] + block + manifest_text[end:]


def check_locked(source: dict[str, Any], digest: str, refresh: bool) -> str:
    locked = source.get("sha256")
    if not locked:
        return "locked_now"
    if digest == locked:
        return "ok"
    if refresh:
        return "refreshed"
    raise LoadError(f"{source['id']}: content differs from the locked sha256 "
                    f"({digest[:12]} vs {locked[:12]}); rerun with --refresh to accept it")


def verify_pdf(source: dict[str, Any], raw_dir: Path) -> Result:
    path = raw_dir / f"{source['id']}.pdf"
    result = Result(source["id"], source["access"], "missing", [path.name])
    if not path.exists():
        result.notes.append(f"place the PDF at {path}")
        return result
    if path.read_bytes()[:5] != b"%PDF-":
        raise LoadError(f"{source['id']}: {path.name} is not a PDF")
    result.sha256 = sha256_file(path)
    if not source.get("sha256"):
        result.status = "locked_now"
    elif result.sha256 == source["sha256"]:
        result.status = "ok"
    else:
        raise LoadError(f"{source['id']}: {path.name} does not match the locked sha256; "
                        "the file was replaced or corrupted")
    return result


def pmc_parts(source: dict[str, Any]) -> list[str]:
    if source.get("parts"):
        return [part["pmcid"] for part in source["parts"]]
    if source.get("pmcid"):
        return [source["pmcid"]]
    raise LoadError(f"{source['id']}: pmc_api source without pmcid or parts")


def fetch_pmc_xml(entrez: Entrez, pmcid: str) -> bytes:
    raw = entrez._call("efetch.fcgi", {"db": "pmc", "id": pmcid.removeprefix("PMC"),
                                        "retmode": "xml"})
    root = ET.fromstring(raw)
    if root.find(".//article") is None:
        raise LoadError(f"{pmcid}: efetch returned no <article>")
    return raw


def has_body(xml_bytes: bytes) -> bool:
    return ET.fromstring(xml_bytes).find(".//article/body") is not None


def load_pmc(source: dict[str, Any], raw_dir: Path, entrez: Entrez, refresh: bool) -> Result:
    folder = raw_dir / source["id"]
    paths = [folder / f"{pmcid}.xml" for pmcid in pmc_parts(source)]
    result = Result(source["id"], source["access"], "ok",
                    [p.relative_to(raw_dir).as_posix() for p in paths])
    have_all = all(p.exists() for p in paths)
    if have_all and source.get("sha256") == bundle_sha256(paths) and not refresh:
        result.sha256 = source["sha256"]
        return result
    fetched = {p: fetch_pmc_xml(entrez, p.stem) for p in paths}
    digest = sha256_bytes("\n".join(sha256_bytes(fetched[p]) for p in paths).encode())
    result.status = check_locked(source, digest, refresh)
    folder.mkdir(parents=True, exist_ok=True)
    for path, data in fetched.items():
        path.write_bytes(data)
        if not has_body(data):
            result.notes.append(f"{path.stem}: no <body>, full text not in the PMC record")
    result.sha256 = digest
    return result


def _text(element: ET.Element) -> str:
    return " ".join(" ".join(element.itertext()).split())


EMPTY_SEE = re.compile(r"\(\s*(?:see\s*:?\s*)?\)\.?", re.I)
HEADING_STYLES = {"italics", "bold", "underline"}
SPL_HEADING_MAX_WORDS = 8


def _text_without_links(element: ET.Element) -> str:
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if node.tag == f"{HL7}linkHtml":
            if node.tail:
                parts.append(node.tail)
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
        if node.tail:
            parts.append(node.tail)

    if element.text:
        parts.append(element.text)
    for child in element:
        walk(child)
    return " ".join(EMPTY_SEE.sub("", " ".join(parts)).split())


def _is_heading_paragraph(paragraph: ET.Element, text: str) -> bool:
    children = list(paragraph)
    if not children or (paragraph.text or "").strip():
        return False
    if any(c.tag != f"{HL7}content" or (c.tail or "").strip(" :.\n\t") for c in children):
        return False
    styles = {s for c in children for s in (c.get("styleCode") or "").lower().split()}
    return bool(styles & HEADING_STYLES) and len(text.split()) <= SPL_HEADING_MAX_WORDS


def section_text(section: ET.Element) -> str:
    lines: list[str] = []

    def walk(node: ET.Element, top: bool) -> None:
        for child in node:
            tag = child.tag.removeprefix(HL7)
            if tag == "title":
                title = _text_without_links(child)
                if title and not top:
                    lines.append(f"## {title}")
            elif tag == "paragraph":
                text = _text_without_links(child)
                if text:
                    lines.append(f"## {text.rstrip(' :')}" if _is_heading_paragraph(child, text)
                                 else text)
            elif tag == "list":
                for item in child.iter(f"{HL7}item"):
                    text = _text_without_links(item)
                    if text:
                        lines.append(f"- {text}")
            elif tag == "table":
                for row in child.iter(f"{HL7}tr"):
                    cells = [_text_without_links(c) for c in row
                             if c.tag in (f"{HL7}td", f"{HL7}th")]
                    if any(cells):
                        lines.append(" | ".join(cells))
            elif tag == "section":
                walk(child, False)
            elif tag in ("code", "id", "effectiveTime"):
                continue
            else:
                walk(child, top)

    walk(section, True)
    return "\n".join(lines)


def parse_spl(xml_bytes: bytes, section_codes: Iterable[str]) -> dict[str, Any]:
    root = ET.fromstring(xml_bytes)
    title_node = root.find(f"{HL7}title")
    forms = sorted({node.get("displayName", "").upper()
                    for node in root.iter(f"{HL7}formCode") if node.get("displayName")})
    wanted = set(section_codes)
    sections: dict[str, str] = {}
    for section in root.iter(f"{HL7}section"):
        code = section.find(f"{HL7}code")
        if code is None or code.get("codeSystem") != LOINC:
            continue
        if code.get("code") in wanted and code.get("code") not in sections:
            sections[code.get("code")] = section_text(section)
    version = root.find(f"{HL7}versionNumber")
    set_id = root.find(f"{HL7}setId")
    return {
        "setid": set_id.get("root") if set_id is not None else None,
        "version": version.get("value") if version is not None else None,
        "title": _text(title_node) if title_node is not None else "",
        "forms": forms,
        "sections": sections,
    }


def dedupe_sections(labels: list[dict[str, Any]], section_code: str) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for label in labels:
        text = label["sections"].get(section_code)
        if not text:
            continue
        entry = groups.setdefault(text, {"text": text, "labels": []})
        entry["labels"].append({key: label[key] for key in
                                ("setid", "version", "title", "groups")})
    out = sorted(groups.values(), key=lambda g: (-len(g["labels"]), g["text"]))
    for entry in out:
        entry["labels"].sort(key=lambda lab: (lab["setid"] or "", lab["version"] or ""))
    return out


def dailymed_setids(get: Callable[[str], bytes], query: dict[str, Any]
                    ) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for group, rxcuis in query["rxcui"].items():
        for rxcui in rxcuis:
            page = 1
            while True:
                url = (f"{DAILYMED}spls.json?rxcui={rxcui}&doctype={query['doctype']}"
                       f"&pagesize=100&page={page}")
                data = json.loads(get(url))
                for row in data["data"]:
                    groups.setdefault(row["setid"], [])
                    if group not in groups[row["setid"]]:
                        groups[row["setid"]].append(group)
                if page >= data["metadata"]["total_pages"]:
                    break
                page += 1
    return groups


def build_dailymed(get: Callable[[str], bytes], query: dict[str, Any]) -> dict[str, Any]:
    codes = [SECTION_CODES[name] for name in query["sections"]]
    excluded = {form.upper() for form in query.get("exclude_dosage_forms", [])}
    setids = dailymed_setids(get, query)
    labels, dropped = [], []
    for setid in sorted(setids):
        label = parse_spl(get(f"{DAILYMED}spls/{setid}.xml"), codes)
        label["groups"] = sorted(setids[setid])
        if excluded & set(label["forms"]):
            dropped.append({"setid": setid, "title": label["title"],
                            "forms": sorted(excluded & set(label["forms"]))})
            continue
        labels.append(label)
    sections = {name: dedupe_sections(labels, SECTION_CODES[name])
                for name in query["sections"]}
    return {"query": query, "labels_kept": len(labels), "labels_dropped": dropped,
            "labels_without_section": sorted(lab["setid"] for lab in labels
                                             if not all(lab["sections"].get(c) for c in codes)),
            "sections": sections}


def load_dailymed(source: dict[str, Any], raw_dir: Path, get: Callable[[str], bytes],
                  refresh: bool) -> Result:
    path = raw_dir / f"{source['id']}.json"
    result = Result(source["id"], source["access"], "ok", [path.name])
    if path.exists() and source.get("sha256") == sha256_file(path) and not refresh:
        result.sha256 = source["sha256"]
        return result
    payload = build_dailymed(get, source["query"])
    data = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
    result.status = check_locked(source, sha256_bytes(data), refresh)
    path.write_bytes(data)
    result.sha256 = sha256_bytes(data)
    for name, groups in payload["sections"].items():
        result.notes.append(f"{name}: {payload['labels_kept']} labels -> "
                            f"{len(groups)} distinct texts")
    if payload["labels_dropped"]:
        result.notes.append(f"dropped {len(payload['labels_dropped'])} labels by dosage form")
    if payload["labels_without_section"]:
        result.notes.append(f"{len(payload['labels_without_section'])} labels lack a "
                            "requested section")
    return result


def run(manifest_path: Path, only: Optional[set[str]], refresh: bool,
        entrez: Entrez, get: Callable[[str], bytes]) -> list[Result]:
    manifest = load_manifest(manifest_path)
    raw_dir = PACKAGE_ROOT / manifest["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    text = manifest_path.read_text(encoding="utf-8")
    today = dt.date.today().isoformat()
    results = []
    for source in manifest["sources"]:
        if only and source["id"] not in only:
            continue
        if source["access"] == "manual_pdf":
            result = verify_pdf(source, raw_dir)
        elif source["access"] == "pmc_api":
            result = load_pmc(source, raw_dir, entrez, refresh)
        else:
            result = load_dailymed(source, raw_dir, get, refresh)
        if result.status in ("locked_now", "refreshed"):
            text = set_lock(text, source["id"], today, result.sha256)
        results.append(result)
    manifest_path.write_text(text, encoding="utf-8", newline="\n")
    report = {"manifest": str(manifest_path), "run_on": today,
              "results": [vars(r) for r in results]}
    (raw_dir / "load_report.json").write_text(json.dumps(report, indent=2),
                                              encoding="utf-8", newline="\n")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Layer 1 loader: fetch and lock the guideline corpus")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--only", nargs="*", help="source ids to load; default all")
    ap.add_argument("--refresh", action="store_true",
                    help="accept upstream changes to API sources and re-lock them")
    args = ap.parse_args()

    api_key = env_value("NCBI_API_KEY")
    limiter = RateLimiter(per_min=NCBI_PER_MIN_WITH_KEY if api_key else NCBI_PER_MIN_NO_KEY,
                          window=60.0)
    entrez = Entrez(api_key=api_key, email=env_value("NCBI_EMAIL"), limiter=limiter)
    dailymed_limiter = RateLimiter(per_min=240, window=60.0)

    def get(url: str) -> bytes:
        dailymed_limiter.acquire()
        return _get(url)

    try:
        results = run(Path(args.manifest), set(args.only) if args.only else None,
                      args.refresh, entrez, get)
    except LoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for r in results:
        print(f"{r.id:36s} {r.access:13s} {r.status:11s} {(r.sha256 or '')[:12]}")
        for note in r.notes:
            print(f"{'':36s}   - {note}")
    missing = [r.id for r in results if r.status == "missing"]
    if missing:
        print(f"\nmissing PDFs: {missing}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
