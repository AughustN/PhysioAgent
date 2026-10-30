"""Score LLM arms on the T0 dose cohort (docs/cohort_spec.md, docs/feature_schema.md).

    plain        the model's own knowledge only
    plain_calib  plain + how often each dose was ordered in train (an ablation: does the
                 plain arm fail on calibration, or on reading the patient?)
    guideline    plain + the guideline rules that fire for this patient, values filled in
                 (kg/guideline/t0_rules.py); the reply names the rules it relies on

--dose-logprobs (E2) also asks the served model, once the reply has reached "dose_mg":,
how likely each menu dose is, digit by digit (vLLM only, no guided decoding: the dose is
read off the model's own distribution, not off the grammar). --score then reads the dose
out of that distribution as well: its median (the MAE-optimal choice), mean, mode, and
the same after batch calibration (Zhou et al., ICLR 2024: divide by the mean distribution
over the split, which uses no label). If the distribution orders patients while the
generated dose does not, the collapse to 20 mg is in the readout, not in the knowledge.

Each target is one row of features.csv.gz; the LLM sees those columns only, so the arm is
compared with B0 and B2 on the same inputs. Replies go to a JSONL checkpoint keyed by
hadm_id, which resumes after an interruption. The label is never written there: --score
joins it from cohort.csv.gz and prints aggregates only, cells under MIN_CELL suppressed.

    python -m physioagent.eval.run_cohort_dose --split dev --limit 5      # smoke
    python -m physioagent.eval.run_cohort_dose --split dev                 # full dev
    python -m physioagent.eval.run_cohort_dose --split dev --score
    python -m physioagent.eval.run_cohort_dose --arm plain_calib --split dev

Checkpoints go to $PHYSIOAGENT_DOSE_RESULTS (one folder per scoring model, outside the
repo: replies quote patient values and stay under the DUA).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from physioagent.config.loader import load_settings
from physioagent.kg.ehr.dataset import COHORT_DIR, GROUP_COLUMNS, load_dataset
from physioagent.llm.client import LLMClient
from physioagent.llm.cohort_prompts import (
    DOSE_SCHEMA,
    MENU,
    REPLY_KEYS,
    calibrated_system_prompt,
    fired_rules,
    guideline_system_prompt,
    guideline_user_prompt,
    parse_dose,
    plain_system_prompt,
    plain_user_prompt,
    prompt_signature,
    rules_signature,
)
from physioagent.llm.dose_prompts import DOSE_LADDER, snap_to_ladder
from physioagent.utils.rate_limit import RateLimiter

ARMS = ("plain", "plain_calib", "guideline")
RULE_ARMS = {"guideline"}
DOSE_KEY = re.compile(r'"dose_mg"\s*:\s*')
RULE_ID = re.compile(r"\bR[1-5]\b")
RESULTS_DIR = Path(os.environ.get("PHYSIOAGENT_DOSE_RESULTS")
                   or Path(__file__).resolve().parent / "results" / "cohort_dose")
MIN_CELL = 10
CONSTANTS = (10.0, 20.0, 40.0)
RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY_S = 2.0
# A local model behind vLLM queues requests rather than refusing them; a long wait is load.
BATCH_TIMEOUT_S = 180.0


def call_with_retry(client: LLMClient, system: str, user: str, attempts: int,
                    limiter: Optional[RateLimiter], meta: dict[str, Any]
                    ) -> tuple[Optional[dict[str, Any]], Optional[str], int]:
    delay, last = RETRY_BASE_DELAY_S, None
    for attempt in range(1, attempts + 1):
        if limiter is not None:
            limiter.acquire()
        try:
            return client.chat_json(system, user, meta, schema=DOSE_SCHEMA), None, attempt
        except Exception as exc:  # noqa: BLE001 - any failure is retried, then recorded
            last = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(delay)
                delay *= 2
    return None, last, attempts


def prompt_sha(system: str, arm: str = "plain") -> str:
    text = system + "\n" + prompt_signature()
    if arm in RULE_ARMS:
        text += "\n" + rules_signature()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def user_prompt_for(arm: str, features: pd.Series) -> str:
    return guideline_user_prompt(features) if arm in RULE_ARMS else plain_user_prompt(features)


def system_prompt_for(arm: str, split_col: str, cohort_dir: Path, tag: str = "") -> str:
    if arm == "plain":
        return plain_system_prompt()
    if arm == "guideline":
        return guideline_system_prompt()
    if arm == "plain_calib":
        # Train labels only, of the same split column the arm is scored on.
        _, y_train, _ = load_dataset("train", split_col=split_col, cohort_dir=cohort_dir,
                                     tag=tag)
        shares = y_train.value_counts(normalize=True)
        return calibrated_system_prompt({d: float(shares.get(d, 0.0)) for d in DOSE_LADDER},
                                        len(y_train))
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def checkpoint_path(arm: str, split: str, split_col: str, tag: str = "",
                    results_dir: Optional[Path] = None) -> Path:
    return Path(results_dir or RESULTS_DIR) / f"cohort_{arm}_{split_col}_{split}{tag}.jsonl"


def read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Last row per hadm_id; a rerun after an error replaces the error row."""
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row["status"] == "ok" or rows.get(row["hadm_id"], {}).get("status") != "ok":
            rows[row["hadm_id"]] = row
    return rows


def dose_prefix(content: str) -> str:
    """The reply up to where the dose value starts."""
    return content[:list(DOSE_KEY.finditer(content))[-1].end()]


DIGITS = set("0123456789")
# Every digit string a menu dose passes through: "", "1", "10", "100", "12", ... Gemma
# writes one digit per token, so log P(120) = log P(1|"") + log P(2|"1") + log P(0|"12")
# + log P(the number ends|"120"), the end being every next token that is not a digit.
DIGIT_NODES = sorted({str(d)[:i] for d in MENU for i in range(len(str(d)) + 1)})


def dose_logprobs(client: LLMClient, system: str, user: str, content: str
                  ) -> tuple[dict[str, float], int]:
    """log P(dose) for every menu dose, and how many digit tokens fell outside the top
    tokens returned (each is then given the smallest logprob returned, an upper bound)."""
    tops = dict(zip(DIGIT_NODES, client.next_token_logprobs(system, user, dose_prefix(content),
                                                            DIGIT_NODES)))
    out, missing = {}, 0
    for dose in MENU:
        text, lp = str(dose), 0.0
        for i, digit in enumerate(text):
            node = tops[text[:i]]
            if digit not in node:
                missing += 1
            lp += node.get(digit, min(node.values(), default=-30.0))
        p_digit = sum(math.exp(v) for k, v in tops[text].items() if k in DIGITS)
        lp += math.log(max(1.0 - p_digit, 1e-12))
        out[str(dose)] = round(lp, 5)
    return out, missing


def rule_ranges(features: pd.Series) -> dict[str, Optional[list[int]]]:
    """Rules that fire for this patient: their dose range, or None for a direction (R2)."""
    return {rule_id: None if f.low is None else [f.low, f.high]
            for rule_id, f in fired_rules(features).items()}


def one_dose(hadm_id: str, features: pd.Series, client: LLMClient, attempts: int,
             limiter: Optional[RateLimiter], arm: str, system: str,
             with_logprobs: bool = False) -> dict[str, Any]:
    started = time.time()
    meta: dict[str, Any] = {}
    user = user_prompt_for(arm, features)
    response, error, tries = call_with_retry(client, system, user, attempts, limiter, meta)
    row: dict[str, Any] = {"hadm_id": hadm_id, "arm": arm,
                           "prompt_sha": prompt_sha(system, arm),
                           "attempts": tries, "served_model": meta.get("model"),
                           "key_order": meta.get("key_order")}
    if arm in RULE_ARMS:
        row["fired_rules"] = rule_ranges(features)
    usage = meta.get("usage") or {}
    row["prompt_tokens"] = usage.get("prompt_tokens")
    row["completion_tokens"] = usage.get("completion_tokens")
    if response is None:
        row.update(status="error", pred_dose_mg=None, error=error)
    else:
        raw = parse_dose(response)
        if raw is None:
            row.update(status="unparsed", pred_dose_mg=None, raw=str(response)[:500])
        else:
            dose = snap_to_ladder(raw)
            reasoning = str(response.get("reasoning", ""))
            factor = str(response.get("primary_factor", ""))
            row.update(status="ok", pred_dose_mg=dose, raw_dose=raw, off_menu=raw != dose,
                       primary_factor=factor[:200], reasoning=reasoning[:800])
            if arm in RULE_ARMS:
                # From the whole text: the stored reasoning is cut at 800 characters.
                row["cited_rules"] = sorted(set(RULE_ID.findall(reasoning + " " + factor)))
            if with_logprobs:
                try:
                    row["dose_logprobs"], row["dose_logprobs_missing"] = dose_logprobs(
                        client, system, user, meta.get("content") or "")
                except Exception as exc:  # noqa: BLE001 - the dose itself is still scored
                    row["dose_logprobs_error"] = f"{type(exc).__name__}: {exc}"[:300]
    row["elapsed_s"] = round(time.time() - started, 2)
    return row


def run(X: pd.DataFrame, path: Path, settings: dict[str, Any], workers: int, rpm: int,
        fresh: bool, arm: str, system: str, with_logprobs: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fresh and path.exists():
        path.unlink()
    done = read_checkpoint(path)
    # One checkpoint answers one prompt: mixing rows of two prompts would score neither.
    stale = {r.get("prompt_sha") for r in done.values()} - {prompt_sha(system, arm)}
    if stale:
        raise SystemExit(f"{path.name} holds rows of another prompt {sorted(stale)}; "
                         "rerun with --fresh or point PHYSIOAGENT_DOSE_RESULTS elsewhere")
    todo = [h for h in X.index if done.get(h, {}).get("status") != "ok"]
    print(f"[{arm}] {len(todo)} to run, {len(X) - len(todo)} already in {path.name}")
    if not todo:
        return
    client = LLMClient.from_settings(settings)
    client.timeout = BATCH_TIMEOUT_S
    limiter = RateLimiter(per_min=rpm)
    print(f"[{arm}] model={client.model_name} at {client.base_url}, "
          f"{workers} workers, {rpm}/min")
    lock = threading.Lock()
    written, started = 0, time.time()
    with path.open("a", encoding="utf-8") as handle, \
            ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(one_dose, h, X.loc[h], client, RETRY_ATTEMPTS, limiter, arm,
                               system, with_logprobs) for h in todo]
        for future in as_completed(futures):
            row = future.result()
            with lock:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                written += 1
                if written == 1 or written % 25 == 0 or written == len(todo):
                    rate = written / max(time.time() - started, 1e-9) * 60
                    print(f"  [{arm}] {written}/{len(todo)}  {rate:.1f}/min  "
                          f"last={row['status']}", flush=True)


def _step(mg: np.ndarray) -> np.ndarray:
    return np.abs(mg[:, None] - np.asarray(DOSE_LADDER)[None, :]).argmin(axis=1)


def dose_metrics(y_true: np.ndarray, y_pred: np.ndarray, b0: float) -> dict[str, Any]:
    if not y_true.size:
        return {"n": 0}
    err = np.abs(y_pred - y_true)
    out: dict[str, Any] = {
        "n": int(y_true.size),
        "mae": round(float(err.mean()), 2),
        "medae": round(float(np.median(err)), 2),
        "exact": round(float((err == 0).mean()), 4),
        "within_1_step": round(float((np.abs(_step(y_pred) - _step(y_true)) <= 1).mean()), 4),
        "bias": round(float((y_pred - y_true).mean()), 2),
        "mean_pred": round(float(y_pred.mean()), 2),
        "mean_true": round(float(y_true.mean()), 2),
    }
    mae_b0 = float(np.abs(b0 - y_true).mean())
    out.update(mae_b0=round(mae_b0, 2),
               skill=round(1 - out["mae"] / mae_b0, 4) if mae_b0 else None,
               mape=round(float((err / y_true).mean()), 4),
               within_20pct=round(float((err <= 0.2 * y_true).mean()), 4))
    return out


def suppressed(m: dict[str, Any]) -> dict[str, Any]:
    """Metrics over fewer than MIN_CELL patients are not reported (smoke runs included)."""
    return m if m["n"] >= MIN_CELL else {"n": f"<{MIN_CELL}"}


def group_metrics(frame: pd.DataFrame, b0: float) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for col in GROUP_COLUMNS:
        cells: dict[str, Any] = {}
        for value, part in frame.groupby(col, dropna=False):
            name = str(value)
            if len(part) < MIN_CELL:
                cells[name] = {"n": f"<{MIN_CELL}"}
                continue
            m = dose_metrics(part["y"].to_numpy(float), part["pred"].to_numpy(float), b0)
            cells[name] = {k: m[k] for k in ("n", "mae", "mae_b0", "skill", "within_20pct",
                                             "bias")}
        report[col] = cells
    return report


def spearman(a: pd.Series, b: pd.Series) -> Optional[float]:
    # Ranks then Pearson: Series.corr(method="spearman") would need scipy.
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    if len(a) < MIN_CELL or a.nunique() < 2 or b.nunique() < 2:
        return None
    return round(float(a.rank().corr(b.rank())), 3)


def auroc(score: np.ndarray, positive: np.ndarray) -> Optional[float]:
    """Mann-Whitney AUROC with ties counted half."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos < MIN_CELL or n_neg < MIN_CELL:
        return None
    ranks = pd.Series(score).rank().to_numpy()
    return round(float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)), 4)


def _median_dose(p: np.ndarray) -> np.ndarray:
    menu = np.asarray(MENU, dtype=float)
    return menu[(np.cumsum(p, axis=1) >= 0.5 - 1e-9).argmax(axis=1)]


def distribution_readout(rows: dict[str, dict[str, Any]], y: pd.Series, b0: float
                         ) -> dict[str, Any]:
    """E2: the dose read out of the model's distribution over the menu, against the dose it
    generated, on the patients that have both."""
    keys = [str(d) for d in MENU]
    have = [h for h in y.index if rows.get(h, {}).get("status") == "ok"
            and all(k in (rows[h].get("dose_logprobs") or {}) for k in keys)]
    if len(have) < MIN_CELL:
        return {"n": f"<{MIN_CELL}"}
    menu = np.asarray(MENU, dtype=float)
    lp = np.array([[rows[h]["dose_logprobs"][k] for k in keys] for h in have], dtype=float)
    p = np.exp(lp - lp.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    # Batch calibration: the mean distribution over the split is the prompt's own prior.
    q = p / p.mean(axis=0, keepdims=True)
    q /= q.sum(axis=1, keepdims=True)
    y_true = y.loc[have].to_numpy(float)
    generated = np.array([rows[h]["pred_dose_mg"] for h in have], dtype=float)
    expected = p @ menu
    readouts = {"generated": generated, "mode": menu[p.argmax(axis=1)],
                "median": _median_dose(p), "mean_on_ladder": menu[_step(expected)],
                "batch_calibrated_mode": menu[q.argmax(axis=1)],
                "batch_calibrated_median": _median_dose(q)}
    out: dict[str, Any] = {
        "n": len(have),
        # How much of the unconstrained probability lands on a menu value at all.
        "menu_mass_mean": round(float(np.exp(lp).sum(axis=1).mean()), 4),
        "entropy_mean_nats": round(float(-(p * np.log(p + 1e-12)).sum(axis=1).mean()), 3),
        "mode_matches_generated": round(float((readouts["mode"] == generated).mean()), 4),
        "auroc_p_dose_ge60": auroc(p[:, menu >= 60].sum(axis=1), y_true >= 60),
        "spearman_expected_dose": spearman(pd.Series(expected), pd.Series(y_true)),
        "readouts": {},
    }
    for name, pred in readouts.items():
        m = dose_metrics(y_true, pred, b0)
        out["readouts"][name] = {**{k: m[k] for k in ("mae", "skill", "bias", "within_1_step")},
                                 "spearman": spearman(pd.Series(pred), pd.Series(y_true)),
                                 "pred_distribution": distribution(pd.Series(pred))}
    return out


def rule_use(rows: dict[str, dict[str, Any]], ok: list[str]) -> Optional[dict[str, Any]]:
    """Guideline arm: does the reply name the rules it was given, and stay in their range?"""
    ok = [h for h in ok if "fired_rules" in rows[h]]
    if not ok:
        return None
    fired = {h: rows[h]["fired_rules"] for h in ok}
    cited = {h: set(rows[h].get("cited_rules") or []) for h in ok}
    with_rule = [h for h in ok if fired[h]]
    ranged = [h for h in ok if any(r for r in fired[h].values())]

    def share(hs: list[str], test) -> Any:
        if len(hs) < MIN_CELL:
            return f"n<{MIN_CELL}"
        return {"n": len(hs), "pct": round(100 * float(np.mean([test(h) for h in hs])), 1)}

    return {"cites_a_fired_rule": share(with_rule, lambda h: bool(cited[h] & set(fired[h]))),
            "cites_a_rule_that_did_not_fire": share(ok, lambda h: bool(cited[h] - set(fired[h]))),
            "dose_inside_a_fired_range": share(ranged, lambda h: any(
                r and r[0] <= rows[h]["pred_dose_mg"] <= r[1] for r in fired[h].values()))}


def score(path: Path, split: str, split_col: str, cohort_dir: Path, tag: str = ""
          ) -> dict[str, Any]:
    rows = read_checkpoint(path)
    _, y, groups = load_dataset(split, split_col=split_col, cohort_dir=cohort_dir, tag=tag)
    # B0 is the train median of the same split column (docs/cohort_spec.md).
    _, y_train, _ = load_dataset("train", split_col=split_col, cohort_dir=cohort_dir, tag=tag)
    b0 = float(y_train.median())
    status = pd.Series({h: rows.get(h, {}).get("status", "missing") for h in y.index})
    ok = [h for h in y.index if status[h] == "ok"]
    frame = groups.loc[ok].assign(y=y.loc[ok],
                                  pred=[rows[h]["pred_dose_mg"] for h in ok])
    y_all = y.to_numpy(float)
    # Every target counts in the imputed score: a reply that failed falls back to B0.
    imputed = np.array([rows[h]["pred_dose_mg"] if status[h] == "ok" else b0
                        for h in y.index], dtype=float)
    models = sorted({str(r.get("served_model")) for r in rows.values() if r.get("served_model")})
    off_menu = [bool(rows[h].get("off_menu")) for h in ok]
    ordered = [rows[h]["key_order"] == REPLY_KEYS for h in ok if rows[h].get("key_order")]
    report = {
        "arm": sorted({str(r.get("arm")) for r in rows.values()}),
        "split": f"{split_col}={split}{tag}", "b0_mg": b0, "served_models": models,
        "prompt_sha": sorted({str(r.get("prompt_sha")) for r in rows.values()}),
        "status": {k: int(v) for k, v in status.value_counts().items()},
        "off_menu_rate": round(float(np.mean(off_menu)), 4) if off_menu else None,
        # Share of replies that wrote reasoning, then primary_factor, then the dose.
        "reasoning_first_rate": round(float(np.mean(ordered)), 4) if ordered else None,
        "answered": suppressed(dose_metrics(frame["y"].to_numpy(float),
                                            frame["pred"].to_numpy(float), b0)),
        # Does the arm order patients at all? MAE alone rewards collapsing on the mode.
        "spearman_answered": spearman(frame["pred"], frame["y"]),
        "all_targets_failed_as_b0": suppressed(dose_metrics(y_all, imputed, b0)),
        "by_group": group_metrics(frame, b0),
    }
    if any("dose_logprobs" in r or "dose_logprobs_error" in r for r in rows.values()):
        report["dose_logprobs_errors"] = sum("dose_logprobs_error" in r for r in rows.values())
        report["dose_logprobs_rows_with_a_digit_outside_top"] = sum(
            bool(r.get("dose_logprobs_missing")) for r in rows.values())
        report["distribution_readout"] = distribution_readout(rows, y, b0)
    uses = rule_use(rows, ok)
    if uses is not None:
        report["rule_use"] = uses
    band = report["by_group"]["dose_band"]
    maes = [c["mae"] for c in band.values() if "mae" in c]
    report["dose_band_macro_mae"] = round(float(np.mean(maes)), 2) if maes else None
    # A model that answers one dose for everyone scores like that constant: show both.
    report["constant_mae"] = {f"{c:g}": round(float(np.abs(c - y_all).mean()), 2)
                              for c in CONSTANTS}
    report["pred_distribution"] = distribution(frame["pred"])
    report["label_distribution"] = distribution(frame["y"])
    return report


def distribution(doses: pd.Series) -> dict[str, Any]:
    counts = doses.round().astype(int).value_counts().sort_index()
    return {str(k): int(v) if v >= MIN_CELL else f"<{MIN_CELL}" for k, v in counts.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="plain", choices=ARMS)
    ap.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    ap.add_argument("--split-col", default="split", choices=("split", "split_temporal"))
    ap.add_argument("--cohort-dir", type=Path, default=COHORT_DIR)
    ap.add_argument("--tag", default="", help='"_5d" for the dialysis stratum')
    ap.add_argument("--limit", type=int, help="first N targets only, for a smoke run")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=600)
    ap.add_argument("--fresh", action="store_true", help="delete the checkpoint first")
    ap.add_argument("--score", action="store_true", help="score what is on disk and exit")
    ap.add_argument("--save", type=Path, help="with --score: also write the report here")
    ap.add_argument("--dose-logprobs", action="store_true",
                    help="E2: also score every menu dose at dose_mg (vLLM only)")
    args = ap.parse_args()

    path = checkpoint_path(args.arm, args.split, args.split_col, args.tag)
    if args.score:
        report = score(path, args.split, args.split_col, args.cohort_dir, args.tag)
        text = json.dumps(report, indent=2)
        print(text)
        if args.save:
            args.save.write_text(text + "\n", encoding="utf-8")
        return 0
    X, _, _ = load_dataset(args.split, split_col=args.split_col, cohort_dir=args.cohort_dir,
                           tag=args.tag)
    X = X.sort_index()
    if args.limit:
        X = X.iloc[:args.limit]
    system = system_prompt_for(args.arm, args.split_col, args.cohort_dir, args.tag)
    run(X, path, load_settings(), args.workers, args.rpm, args.fresh, args.arm, system,
        args.dose_logprobs)
    print(f"[{args.arm}] checkpoint {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
