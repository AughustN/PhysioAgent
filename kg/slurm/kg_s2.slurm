#!/bin/bash
# S2: a UMLS release -> the gated CUI graph -> triples between concepts that co-occur.
#
#   sbatch physioagent/kg/slurm/kg_s2.slurm
#
# Two programs, one job, because the second reads only what the first writes and the
# first is where all the I/O is: umls_prepare.py streams four RRF files (~20 GB of text
# for a full release) and reduces them to a few hundred MB of gated tables, then
# umls_source.py walks that in memory. Splitting them would mean reading the tables twice
# and holding a second allocation for the cheap half.
#
# NO GPU, NO LLM, NO NETWORK. Unlike the PubMed flow this one calls nothing out: every
# edge is a curated Metathesaurus assertion, so the job is pure CPU and disk. That is also
# why it can be re-run freely -- there is no bill and no rate limit, only wall clock.
#
# SIZING. Peak memory is the prepare step's edge set: one tuple of three interned CUI
# strings per surviving MRREL row. With the default --sab list (MSH, SNOMEDCT_US, NCI,
# RXNORM, ATC) and the strict gate that is single-digit millions of rows, comfortably
# inside 48G; widening --sab to the whole Metathesaurus is what makes this job die, not
# the file sizes. If it OOMs, narrow the gate before asking for more memory -- an edge set
# too big to hold is also a graph too unfocused to walk.
#
# S1 MUST HAVE RUN. umls_source.py walks between pairs of concepts that actually
# co-occurred in this cohort, and that list is kg/data/all_top_coexisting_concepts.json
# from kg_s0s1.slurm. Without it the flow falls back to every in-scope pair -- n^2/2
# instead of 20n, and no longer EHR-grounded. It warns rather than aborting, so check the
# log rather than assuming.
#
# WHAT TO READ WHEN IT FINISHES:
#   kg/data/umls/umls_prepare_report.json  seed_names_unresolved -> concepts the whole
#                                                                   flow never asks about
#                                          edge_rows_dropped     -> is out_of_gate ~100%?
#                                                                   then the gate is too
#                                                                   strict to connect
#                                                                   anything
#   kg/data/umls_report.json               pairs_no_path  -> gate too strict / graph too
#                                                            sparse
#                                          triples_by_hop -> what the far hops were worth
#
#SBATCH --job-name=physioagent-kg-s2
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# --- edit for your account ------------------------------------------------------------
WORK=/media02/tphung03
PROJECT="$WORK/PhysioAgent"
CONDA_SH="$WORK/miniconda3/etc/profile.d/conda.sh"
ENV_DIR="$WORK/conda/envs/physioagent"
# The META/ directory of an extracted UMLS release -- the one holding MRCONSO.RRF. A
# full release is ~30 G extracted; the subset produced by MetamorphoSys works too, as
# long as it kept MRSTY, MRSAT, MRCONSO and MRREL.
export UMLS_META="${UMLS_META:-$WORK/UMLS/2026AA/META}"
PROFILE="${PROFILE:-default}"
GATE_MODE="${GATE_MODE:-mesh-any}"
MAX_LENGTH="${MAX_LENGTH:-4}"
MIN_FREE_MB="${MIN_FREE_MB:-4000}"
# SKIP_PREPARE=1 when kg/data/umls/ was copied in from a box that already ran the prepare
# step. The tables are a pure function of (release, profile, gate mode, --sab), so copying
# them is not a shortcut that loses anything -- and the walk reads only three of them,
# 2.3 MB in total, against 17.6 GB of RRF. The licence travels with the files: they are
# derived from UMLS and go only to machines whose users hold their own UTS licence.
SKIP_PREPARE="${SKIP_PREPARE:-0}"
# --------------------------------------------------------------------------------------

# Same guard as kg_s0s1.slurm: /media02 is shared and unquota'd, and a full volume kills
# the job with an empty log 30 minutes in rather than in the first second.
AVAIL_MB=$(df -BM --output=avail /media02 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -z "$AVAIL_MB" ]]; then
    echo "WARNING: could not read free space on /media02; continuing anyway" >&2
elif (( AVAIL_MB < MIN_FREE_MB )); then
    echo "ABORT: /media02 has ${AVAIL_MB}M free, need >=${MIN_FREE_MB}M." >&2
    exit 1
fi

export HOME="$WORK"   # /home is read-only on this cluster; conda writes ~/.conda
source "$CONDA_SH"
conda activate "$ENV_DIR"
cd "$PROJECT"

echo "node=$(hostname)  umls=$UMLS_META  profile=$PROFILE  gate=$GATE_MODE" \
     "  skip_prepare=$SKIP_PREPARE"
if [[ "$SKIP_PREPARE" == "1" ]]; then
    for f in umls_graph.tsv seed_cui.tsv cui_term.tsv; do
        if [[ ! -f "physioagent/kg/data/umls/$f" ]]; then
            echo "ABORT: SKIP_PREPARE=1 but physioagent/kg/data/umls/$f is missing." >&2
            exit 1
        fi
    done
else
    for t in MRSTY MRSAT MRCONSO MRREL; do
        if [[ ! -f "$UMLS_META/$t.RRF" ]]; then
            echo "ABORT: $UMLS_META/$t.RRF missing." >&2
            echo "       MetamorphoSys subsets can omit MRSAT -- without it the MeSH half" >&2
            echo "       of the gate is empty and --gate-mode strict keeps nothing." >&2
            exit 1
        fi
    done
fi
if [[ ! -f "physioagent/kg/data/all_top_coexisting_concepts.json" ]]; then
    echo "WARNING: S1 output missing; the walk will fall back to every in-scope pair." >&2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Slurm ghi stdout vào file, nên Python đệm theo khối và bản log trông như treo cho tới
# lúc job kết thúc -- kể cả khi umls_source in tiến độ mỗi 200 cặp. Tắt đệm đi.
export PYTHONUNBUFFERED=1

if [[ "$SKIP_PREPARE" == "1" ]]; then
    echo "skipping prepare; using the tables already in kg/data/umls/"
else
    python -m physioagent.kg.sources.umls_prepare \
        --umls "$UMLS_META" --profile "$PROFILE" --gate-mode "$GATE_MODE"
fi

python -m physioagent.kg.sources.umls_source \
    --profile "$PROFILE" --max-length "$MAX_LENGTH"

echo
echo "=== umls_report.json ==="
python - <<'PY'
import json, pathlib
p = pathlib.Path("physioagent/kg/data")
prep = json.loads((p / "umls" / "umls_prepare_report.json").read_text())
run = json.loads((p / "umls_report.json").read_text())
print(f"{'CUIs through the gate':34s} {prep['cuis_kept']:,}")
print(f"{'edges':34s} {prep['edges']:,}")
print(f"{'seed names unresolved':34s} {len(prep['seed_names_unresolved'])}"
      f" / {prep['seed_names']}")
print(f"{'pairs walked':34s} {run['pairs']:,}")
print(f"{'  connected':34s} {run['pairs_with_path']:,}")
print(f"{'  no path':34s} {run['pairs_no_path']:,}")
print(f"{'triples':34s} {run['triples']:,}")
print(f"{'distinct edges':34s} {run['distinct_edges']:,}")
print(f"{'by hop':34s} {run['triples_by_hop']}")
PY

# S5 is cheap and reads whatever flows exist, so re-run it here: the point of S2 is the
# agreement count, and that number does not exist until combine has seen all three.
python -m physioagent.kg.combine
