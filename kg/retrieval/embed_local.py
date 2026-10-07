#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

KG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = KG_DIR / "data"
DEFAULT_MODEL = "NeuML/pubmedbert-base-embeddings"


def graph_vocabulary(path: Path) -> tuple[list[str], list[str]]:
    entities: dict[str, None] = {}
    relations: dict[str, None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        edge = json.loads(line)
        entities.setdefault(edge["head"], None)
        entities.setdefault(edge["tail"], None)
        relations.setdefault(edge["relation"], None)
    return list(entities), list(relations)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=str(DEFAULT_DATA / "kg_refined.jsonl"))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda / cpu; default: let torch pick")
    args = ap.parse_args()

    graph_path = Path(args.graph)
    if not graph_path.exists():
        raise SystemExit(f"{graph_path} not found -- run physioagent.kg.refine")
    entities, relations = graph_vocabulary(graph_path)
    print(f"graph      : {len(entities):,} entities, {len(relations):,} relations")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device=args.device)
    data_dir = Path(args.data_dir)
    slug = args.model.replace("/", "-")
    for kind, texts in (("entities", entities), ("relations", relations)):
        vectors = model.encode(texts, batch_size=args.batch, show_progress_bar=False,
                               convert_to_numpy=True, normalize_embeddings=False)
        payload: dict[str, Any] = {text: [round(float(x), 6) for x in vector]
                                   for text, vector in zip(texts, vectors)}
        out = data_dir / f"embeddings_{kind}_{slug}.json"
        out.write_text(json.dumps(payload), encoding="utf-8")
        print(f"{kind:10s} : {len(payload):,} vectors, dim {len(next(iter(payload.values())))}"
              f"  -> {out.name}")
    print("\nNow run retrieval with the SAME model:")
    print(f"  python -m physioagent.kg.retrieval.trajectory --similarity embedding "
          f"--embed-model {args.model} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
