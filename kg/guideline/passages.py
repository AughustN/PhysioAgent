#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifest.yaml"
DEFAULT_OUT = PACKAGE_ROOT / "kg" / "data" / "guideline_passages.jsonl"

MIN_WORDS = 150
MAX_WORDS = 400
HEADER_FOOTER_SHARE = 0.25
HEADING_MAX_WORDS = 20

EXCLUDED_SECTION = re.compile(
    r"^\W*(\d+(\.\d+)*\.?\s*)?(references?|bibliography|acknowledge?ments?|disclosures?"
    r"|conflicts? of interest|funding|author(s|'s)?( contributions| information)?"
    r"|abbreviations( and acronyms)?|supplementary (material|data)|appendix|"
    r"declaration of (competing )?interests?)\b", re.I)

ABBREVIATIONS = ("e.g.", "i.e.", "vs.", "i.v.", "et al.", "approx.", "Fig.", "Figs.",
                 "No.", "ca.", "cf.", "etc.", "Dr.", "min.", "max.")

SPOT_CHECKS = (
    ("acc_aha_hfsa_2022", ("Furosemide", "20–40 mg once or twice", "600 mg")),
    ("kdigo_aki_2012", ("not using diuretics to treat AKI",)),
    ("dailymed_furosemide", ("2 hours after the first",)),
    ("esc_hf_2021", ("DOSE",)),
)


class PassageError(RuntimeError):
    pass


@dataclass
class Block:
    kind: str
    text: str
    level: int = 0
    loc: dict[str, Any] = field(default_factory=dict)


@dataclass
class Passage:
    passage_id: str
    source_id: str
    region: str
    doc_type: str
    version: str
    heading_method: str
    section_path: list[str]
    kind: str
    location: dict[str, Any]
    text: str
    n_words: int


def words(text: str) -> int:
    return len(text.split())


LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def clean(text: str) -> str:
    for ligature, letters in LIGATURES.items():
        text = text.replace(ligature, letters)
    return " ".join(text.replace("­", "").split())


def _itertext(element: ET.Element, skip: tuple[str, ...] = ()) -> str:
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if node.tag in skip:
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
    return clean(" ".join(parts))


def linearise_table(rows: list[list[str]], caption: str = "") -> str:
    rows = [[clean(c or "") for c in row] for row in rows]
    rows = [row for row in rows if any(row)]
    lines = [caption] if caption else []
    while rows and sum(1 for c in rows[0] if c) == 1 and len(rows[0]) > 1:
        lines.append(next(c for c in rows[0] if c))
        rows = rows[1:]
    if not rows:
        return clean(" ".join(lines))
    header, body = rows[0], rows[1:]
    group = ""
    for row in body:
        filled = [c for c in row if c]
        if len(filled) == 1 and row[0] and len(row) > 1:
            group = row[0]
            continue
        cells = []
        for i, cell in enumerate(row):
            if not cell:
                continue
            name = header[i] if i < len(header) and header[i] else ""
            cells.append(f"{name}: {cell}" if name else cell)
        lines.append((f"[{group}] " if group else "") + " | ".join(cells))
    return "\n".join(lines)


def jats_table(table_wrap: ET.Element) -> str:
    label = clean(table_wrap.findtext("label") or "")
    caption_node = table_wrap.find("caption")
    caption = clean(f"{label} {_itertext(caption_node) if caption_node is not None else ''}")
    rows: list[list[str]] = []
    for tr in table_wrap.iter("tr"):
        row: list[str] = []
        for cell in tr:
            if cell.tag not in ("td", "th"):
                continue
            text = _itertext(cell)
            row.extend([text] + [""] * (int(cell.get("colspan", "1") or 1) - 1))
        rows.append(row)
    foot = table_wrap.find("table-wrap-foot")
    text = linearise_table(rows, caption)
    return text + (f"\n{_itertext(foot)}" if foot is not None else "")


def read_jats(path: Path, part_title: Optional[str] = None) -> Iterator[Block]:
    root = ET.parse(path).getroot()
    body = root.find(".//article/body")
    if body is None:
        raise PassageError(f"{path}: no <body>")
    base = 0
    if part_title:
        base = 1
        yield Block("heading", part_title, 1, {"file": path.name})

    def walk(node: ET.Element, depth: int) -> Iterator[Block]:
        for child in node:
            loc = {"file": path.name, "sec_id": node.get("id")}
            if child.tag == "sec":
                title = clean(child.findtext("title") or "")
                if title:
                    yield Block("heading", title, base + depth + 1, {"file": path.name,
                                                                     "sec_id": child.get("id")})
                yield from walk(child, depth + 1)
            elif child.tag == "p":
                text = _itertext(child, skip=("table-wrap", "fig", "disp-formula"))
                if text:
                    yield Block("para", text, 0, loc)
                for table in child.iter("table-wrap"):
                    yield Block("table", jats_table(table), 0, {**loc, "table": table.get("id")})
            elif child.tag == "table-wrap":
                yield Block("table", jats_table(child), 0, {**loc, "table": child.get("id")})
            elif child.tag in ("list", "boxed-text", "disp-quote"):
                text = _itertext(child)
                if text:
                    yield Block("para", text, 0, loc)

    yield from walk(body, 0)


MARGIN_BAND = 0.07
MARGIN_REPEAT_PAGES = 3
DOT_LEADER = re.compile(r"(\.\s?){4,}\s*\d*\s*$|\.\s\.\s\.")
HEADING_SIZE_STEP = 0.5
BODY_LOOKAHEAD = 3
BODY_MIN_WORDS = 15
BOLD_FONT = re.compile(r"(Bold|Black|Heavy|Semibold|Demi|[-.,]B$|[-.,]BI$)", re.I)
ITALIC_FONT = re.compile(r"(Italic|Oblique|[-.,]I$|[-.,]BI$)", re.I)
CLASS_LEVEL_NOTE = re.compile(r"^\W*([a-z]\s*)?(class of recommendation|level of evidence)\W*$", re.I)


def is_footnote(text: str) -> bool:
    parts = [part.strip() for part in re.split(r"[;.]\s+|[;.]$", text) if part.strip()]
    if parts and all(CLASS_LEVEL_NOTE.match(part) for part in parts):
        return True
    definitions = sum(1 for part in parts
                      if "=" in part and len(part.split("=", 1)[0].strip()) <= 40)
    return len(parts) >= 2 and definitions / len(parts) >= 0.8


def _norm_line(text: str) -> str:
    return re.sub(r"\d+", "#", clean(text).lower())


def _pdf_page(page: Any, number: int) -> tuple[list[dict[str, Any]], list[tuple[Any, list]]]:
    import fitz

    tables = []
    try:
        for table in page.find_tables().tables:
            rows = table.extract()
            if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                tables.append((fitz.Rect(table.bbox), rows))
    except Exception:
        tables = []
    height = page.rect.height
    lines = []
    for index, block in enumerate(page.get_text("dict")["blocks"]):
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            rect = fitz.Rect(line["bbox"])
            centre = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
            if any(r.contains(centre) for r, _ in tables):
                continue
            chars = sum(len(s["text"]) for s in spans)
            dominant = max(spans, key=lambda s: len(s["text"]))
            bold_chars = sum(len(s["text"]) for s in spans
                             if (s["flags"] & 16) or BOLD_FONT.search(s["font"]))
            italic_chars = sum(len(s["text"]) for s in spans
                               if (s["flags"] & 2) or ITALIC_FONT.search(s["font"]))
            lines.append({"text": "".join(s["text"] for s in line["spans"]),
                          "size": round(dominant["size"] * 2) / 2,
                          "bold": bold_chars / chars >= 0.8,
                          "italic": italic_chars / chars >= 0.8,
                          "margin": rect.y1 < height * MARGIN_BAND
                          or rect.y0 > height * (1 - MARGIN_BAND),
                          "block": index, "page": number, "y": rect.y0})
    return lines, tables


def _segments(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ln in lines:
        last = out[-1] if out else None
        if last and last["block"] == ln["block"] and last["page"] == ln["page"] \
                and last["size"] == ln["size"] and last["bold"] == ln["bold"]                 and last["italic"] == ln["italic"]:
            last["parts"].append(ln["text"])
        else:
            out.append({**ln, "parts": [ln["text"]]})
    for seg in out:
        seg["text"] = clean(re.sub(r"(\w)-\s+([a-z])", r"\1\2", " ".join(seg["parts"])))
    return [seg for seg in out if seg["text"]]


def _looks_like_heading(seg: dict[str, Any], following: list[dict[str, Any]]) -> bool:
    text = seg["text"]
    if not following or not re.search(r"[A-Za-z]{3}", text) or DOT_LEADER.search(text):
        return False
    if words(text) > HEADING_MAX_WORDS or text.endswith((".", ",", ";", ":")):
        return False
    if not any(words(f["text"]) >= BODY_MIN_WORDS for f in following):
        return False
    nxt = following[0]
    if seg["size"] >= nxt["size"] + HEADING_SIZE_STEP:
        return True
    if seg["size"] < nxt["size"] - 0.25:
        return False
    return (seg["bold"] and not nxt["bold"]) or (seg["italic"] and not nxt["italic"]
                                                 and not nxt["bold"])


def read_pdf(path: Path) -> tuple[list[Block], dict[str, Any]]:
    import fitz

    doc = fitz.open(path)
    pages = [(_pdf_page(page, number), number) for number, page in enumerate(doc, 1)]
    n_pages = len(pages)
    seen: dict[str, set[int]] = defaultdict(set)
    seen_margin: dict[str, set[int]] = defaultdict(set)
    for (lines, _), number in pages:
        for ln in lines:
            key = _norm_line(ln["text"])
            seen[key].add(number)
            if ln["margin"]:
                seen_margin[key].add(number)
    repeated = {k for k, ps in seen.items()
                if len(ps) >= max(3, HEADER_FOOTER_SHARE * n_pages) and words(k) <= 15}
    repeated |= {k for k, ps in seen_margin.items() if len(ps) >= MARGIN_REPEAT_PAGES}

    segments: list[dict[str, Any]] = []
    tables_by_page: dict[int, list] = {}
    lines_by_page: dict[int, list] = {}
    dropped_leaders = 0
    for (lines, tables), number in pages:
        kept = []
        for ln in lines:
            if _norm_line(ln["text"]) in repeated:
                continue
            if DOT_LEADER.search(ln["text"]):
                dropped_leaders += 1
                continue
            kept.append(ln)
        segments += _segments(kept)
        tables_by_page[number] = tables
        lines_by_page[number] = lines

    flags = [_looks_like_heading(seg, segments[i + 1:i + 1 + BODY_LOOKAHEAD])
             for i, seg in enumerate(segments)]
    styles = sorted({(seg["size"], seg["bold"], seg["italic"])
                     for seg, f in zip(segments, flags) if f},
                    key=lambda s: (-s[0], not s[1], not s[2]))
    level_of = {style: i + 1 for i, style in enumerate(styles)}

    blocks: list[Block] = []
    current_page = None

    def emit_tables(number: Optional[int]) -> None:
        if number is None:
            return
        for rect, rows in tables_by_page.get(number, []):
            caption = ""
            for ln in lines_by_page.get(number, []):
                if re.match(r"^\s*Table\s+\d+", ln["text"]) and 0 <= rect.y0 - ln["y"] <= 90:
                    caption = clean(ln["text"])
            blocks.append(Block("table", linearise_table(rows, caption), 0, {"page": number}))

    for seg, is_head in zip(segments, flags):
        if seg["page"] != current_page:
            emit_tables(current_page)
            current_page = seg["page"]
        loc = {"page": seg["page"]}
        if is_head:
            level = level_of[(seg["size"], seg["bold"], seg["italic"])]
            last = blocks[-1] if blocks else None
            if last and last.kind == "heading" and last.level == level \
                    and last.loc.get("page") == seg["page"]:
                last.text = clean(last.text + " " + seg["text"])
            else:
                blocks.append(Block("heading", seg["text"], level, loc))
        elif words(seg["text"]) >= 4 and not is_footnote(seg["text"]):
            blocks.append(Block("para", seg["text"], 0, loc))
    emit_tables(current_page)
    seen_pages = {seg["page"] for seg in segments}
    for number in sorted(set(tables_by_page) - seen_pages):
        emit_tables(number)

    stats = {"pages": n_pages, "heading_styles": [list(s) for s in styles],
             "removed_repeated_lines": len(repeated), "dropped_dot_leader_lines": dropped_leaders,
             "tables": sum(len(t) for t in tables_by_page.values())}
    return blocks, stats


def outline_recall(path: Path, blocks: list[Block]) -> Optional[float]:
    import fitz

    toc = [clean(t[1]) for t in fitz.open(path).get_toc()]
    toc = [t for t in toc if re.search(r"[a-z]{3,} [a-z]", t, re.I)]
    if not toc:
        return None
    found = {clean(b.text).lower() for b in blocks if b.kind == "heading"}
    hits = sum(1 for t in toc if any(t.lower()[:40] in h or h[:40] in t.lower() for h in found))
    return round(hits / len(toc), 3)


def read_dailymed(path: Path) -> Iterator[Block]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for section, groups in data["sections"].items():
        for i, group in enumerate(groups):
            routes = sorted({g for lab in group["labels"] for g in lab["groups"]})
            loc = {"group": i, "n_labels": len(group["labels"]),
                   "setids": [lab["setid"] for lab in group["labels"]]}
            yield Block("heading", f"{section.title()} ({', '.join(routes)})", 1, dict(loc))
            for line in group["text"].split("\n"):
                line = clean(line)
                if line.startswith("## "):
                    yield Block("heading", line[3:], 2, dict(loc))
                elif line:
                    yield Block("para", line, 0, dict(loc))


def split_sentences(text: str) -> list[str]:
    protected = text
    for abbr in ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", "․"))
    protected = re.sub(r"(?<=[a-z)][.!?])(\d+(?:[,–-]\d+)*)\s+(?=[A-Z])", r"\1\n", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[•])|\n", protected)
    return [p.replace("․", ".").strip() for p in parts if p.strip()]


def split_long(text: str, limit: int = MAX_WORDS) -> list[str]:
    if words(text) <= limit:
        return [text]
    pieces, current = [], []
    for sentence in split_sentences(text):
        if current and words(" ".join(current + [sentence])) > limit:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_table(text: str, limit: int = MAX_WORDS) -> list[str]:
    if words(text) <= limit:
        return [text]
    lines = text.split("\n")
    head = lines[0] if lines else ""
    pieces, current = [], []
    for line in lines[1:]:
        if current and words(" ".join([head] + current + [line])) > limit:
            pieces.append("\n".join([head] + current))
            current = []
        current.append(line)
    if current:
        pieces.append("\n".join([head] + current))
    return pieces


def chunk(blocks: Iterable[Block]) -> Iterator[tuple[list[str], str, str, dict[str, Any]]]:
    stack: list[tuple[int, str]] = []
    excluded_at: Optional[int] = None
    buffer: list[str] = []
    buffer_loc: dict[str, Any] = {}

    def path() -> list[str]:
        return [title for _, title in stack]

    def flush() -> Iterator[tuple[list[str], str, str, dict[str, Any]]]:
        if buffer:
            for piece in split_long(" ".join(buffer)):
                yield path(), "text", piece, dict(buffer_loc)
            buffer.clear()
            buffer_loc.clear()

    for block in blocks:
        if block.kind == "heading":
            yield from flush()
            if excluded_at is not None and block.level <= excluded_at:
                excluded_at = None
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text))
            if excluded_at is None and EXCLUDED_SECTION.match(block.text):
                excluded_at = block.level
            continue
        if excluded_at is not None:
            continue
        if block.kind == "table":
            yield from flush()
            if not block.text.strip():
                continue
            for piece in split_table(block.text):
                yield path(), "table", piece, dict(block.loc)
            continue
        if buffer and words(" ".join(buffer + [block.text])) > MAX_WORDS \
                and words(" ".join(buffer)) >= MIN_WORDS:
            yield from flush()
        if not buffer:
            buffer_loc.update(block.loc)
        else:
            for key, value in block.loc.items():
                if key == "page" and "page" in buffer_loc and value != buffer_loc["page"]:
                    buffer_loc["page_end"] = value
        buffer.append(block.text)
    yield from flush()


def build(manifest_path: Path) -> tuple[list[Passage], dict[str, Any]]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    raw_dir = PACKAGE_ROOT / manifest["raw_dir"]
    passages: list[Passage] = []
    report: dict[str, Any] = {"sources": {}}
    for source in manifest["sources"]:
        sid, access = source["id"], source["access"]
        stats: dict[str, Any] = {}
        if access == "pmc_api":
            parts = source.get("parts") or [{"pmcid": source["pmcid"], "title": None}]
            blocks: list[Block] = []
            for part in parts:
                blocks += list(read_jats(raw_dir / sid / f"{part['pmcid']}.xml",
                                         part.get("title")))
            method = "jats"
        elif access == "manual_pdf":
            pdf = raw_dir / f"{sid}.pdf"
            blocks, stats = read_pdf(pdf)
            stats["outline_recall"] = outline_recall(pdf, blocks)
            method = "font"
        else:
            blocks = list(read_dailymed(raw_dir / f"{sid}.json"))
            method = "label"
        made = []
        for n, (section_path, kind, text, loc) in enumerate(chunk(blocks)):
            made.append(Passage(f"{sid}:{n:04d}", sid, source["region"], source["doc_type"],
                                source.get("version") or "published", method, section_path,
                                kind, loc, text, words(text)))
        passages += made
        lengths = [p.n_words for p in made] or [0]
        stats.update({"heading_method": method, "blocks": len(blocks),
                      "headings": sum(1 for b in blocks if b.kind == "heading"),
                      "passages": len(made),
                      "kinds": dict(Counter(p.kind for p in made)),
                      "words_median": statistics.median(lengths),
                      "words_p10_p90": [sorted(lengths)[len(lengths) // 10],
                                        sorted(lengths)[(9 * len(lengths)) // 10]]})
        report["sources"][sid] = stats
    report["passages"] = len(passages)
    report["spot_checks"] = {
        f"{sid}: {' + '.join(needles)}": any(
            p.source_id == sid and all(n in p.text for n in needles) for p in passages)
        for sid, needles in SPOT_CHECKS}
    return passages, report


def main() -> int:
    ap = argparse.ArgumentParser(description="Layer 1: guideline corpus -> passages")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    passages, report = build(Path(args.manifest))
    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for p in passages:
            handle.write(json.dumps(vars(p), ensure_ascii=False) + "\n")
    report_path = out.with_name(out.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8", newline="\n")
    for sid, s in report["sources"].items():
        print(f"{sid:36s} {s['heading_method']:5s} passages={s['passages']:4d} "
              f"kinds={s['kinds']} median_words={s['words_median']} "
              f"p10/p90={s['words_p10_p90']}"
              + (f" outline_recall={s['outline_recall']}" if s.get("outline_recall") is not None
                 else ""))
    print(f"\n{report['passages']} passages")
    for check, ok in report["spot_checks"].items():
        print(f"  spot check {'OK  ' if ok else 'FAIL'} {check}")
    print(f"\n-> {out}\n-> {report_path}")
    return 0 if all(report["spot_checks"].values()) else 3


if __name__ == "__main__":
    sys.exit(main())
