"""Where every kg/data artifact lives (gitignored: patient-derived or copyrighted).

    ehr/         cohort build output (cohort/), probe and build reports
    guideline/   raw/ guideline sources and guideline_passages.jsonl
    kg/, embeddings/, retrieval/   outputs of the earlier knowledge-graph pipeline
"""
from __future__ import annotations

from pathlib import Path

KG_DIR = Path(__file__).resolve().parent
DATA = KG_DIR / "data"

EHR = DATA / "ehr"
KG = DATA / "kg"
UMLS = KG / "umls"
EMBEDDINGS = DATA / "embeddings"
RETRIEVAL = DATA / "retrieval"
GUIDELINE = DATA / "guideline"
GUIDELINE_RAW = GUIDELINE / "raw"
GUIDELINE_STALE = GUIDELINE / "_stale"
