"""Where every kg/data artifact lives, one folder per pipeline stage.

    ehr/         doses.json, visits, profiles, concept sets, scope and probe reports
    kg/          UMLS + PubMed flows, kg_raw, kg_refined, refine outputs; umls/ tables
    embeddings/  cached entity / relation / query embeddings (refine, trajectory)
    retrieval/   trajectories (+ KST) and their reports
    guideline/   raw/ sources, passages, dose records and gold; _stale/ outdated outputs
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
