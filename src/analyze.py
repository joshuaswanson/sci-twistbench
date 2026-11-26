#!/usr/bin/env python3
"""Comprehensive statistical analysis and publication figure generator for Sci-TwistBench.

Addresses all methodological controls:
  - Excludes residual catch-all bucket F0 from framework diversity (SMR, N_eff, JS similarity).
  - Computes both all-assigned and valid-only + rarefied size-matched (k=5, k=10) SMR and N_eff.
  - Audits truncation (finish_reason == 'length') pre- and post-repair, reporting TVR on all vs completed outputs.
  - Computes matched-n split-half within-model JS similarity baseline alongside cross-model JS similarity
    and per-scenario monoculture breakdown across all 7 independent models (5 open-weight + 2 proprietary).
  - Computes Verbalized Sampling structural diagnostics (20% 1/5 floor, share in Direct's modal mechanism,
    within-call taxonomy merge rate across the 5 supposedly distinct hypotheses).
  - Applies Benjamini-Hochberg FDR correction across paired Wilcoxon signed-rank tests.
  - Persists every number cited in the paper deterministically into analysis/summary_metrics.json.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RES_DIR = ROOT / "results"
OUT_DIR = ROOT / "analysis"
FIG_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BAD_FW = {"F0", "F_PARSE_FAIL", "F_MISSING", "NA", "", None}

MODEL_DISPLAY = {
    "qwen3.8-27b": "Qwen3.8-27B",
    "gemma-4-31b-it": "Gemma-4-31B",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "mistral-medium-3.5-128b": "Mistral-3.5-128B",
    "gpt-oss-120b": "gpt-oss-120b",
    "claude-sonnet-5.5": "Claude Sonnet 5.5",
    "gpt-6.1-terra": "GPT-6.1 Terra",
    "olmo3-7b-base": "OLMo3-7B (Base)",
    "olmo3-32b-base": "OLMo3-32B (Base)",
    "olmo3-7b-inst-sft": "OLMo3-7B-Inst (SFT)",
    "olmo3-7b-inst-dpo": "OLMo3-7B-Inst (DPO)",
    "olmo3-7b-inst-rlvr": "OLMo3-7B-Inst (RLVR)",
    "olmo3-7b-think-sft": "OLMo3-7B-Think (SFT)",
    "olmo3-7b-think-dpo": "OLMo3-7B-Think (DPO)",
    "olmo3-7b-think-rlvr": "OLMo3-7B-Think (RLVR)",
    "olmo3-32b-think-sft": "OLMo3-32B-Think (SFT)",
    "olmo3-32b-think-dpo": "OLMo3-32B-Think (DPO)",
    "olmo3-32b-think-rlvr": "OLMo3-32B-Think (RLVR)",
    "olmo3.1-32b-think": "OLMo3.1-32B-Think",
}

OPEN_MODELS = [
    "qwen3.8-27b",
    "gemma-4-31b-it",
    "deepseek-v4-flash",
    "mistral-medium-3.5-128b",
    "gpt-oss-120b",
]
CLOSED_MODELS = [
    "claude-sonnet-5.5",
    "gpt-6.1-terra",
]
ALL_MAIN_MODELS = OPEN_MODELS + CLOSED_MODELS

OLMO_BASE = ["olmo3-7b-base", "olmo3-32b-base"]
OLMO_INST = ["olmo3-7b-inst-sft", "olmo3-7b-inst-dpo", "olmo3-7b-inst-rlvr"]
OLMO_THINK = ["olmo3-7b-think-sft", "olmo3-7b-think-dpo", "olmo3-7b-think-rlvr"]
OLMO_32B_THINK = ["olmo3-32b-think-sft", "olmo3-32b-think-dpo", "olmo3-32b-think-rlvr", "olmo3.1-32b-think"]

INTERV_MODES = ["direct", "high_temp", "vs", "icr", "axiom_neg"]
INTERV_DISPLAY = {
    "direct": "Baseline (T=1.0)",
    "high_temp": "High-Temp (T=1.3)",
    "vs": "Verbalized Sampling",
    "icr": "In-Context Regen.",
    "axiom_neg": "Axiom-Negation",
}

# Family mapping for strict cross-family judging (zero self-family judging, including OpenAI open/closed)
MODEL_FAMILY = {
    "qwen3.8-27b": "Qwen",
    "gemma-4-31b-it": "Gemma",
    "deepseek-v4-flash": "DeepSeek",
    "mistral-medium-3.5-128b": "Mistral",
    "gpt-oss-120b": "OpenAI",
    "gpt-6.1-terra": "OpenAI",
    "claude-sonnet-5.5": "Anthropic",
    "olmo3-7b-base": "OLMo",
    "olmo3-32b-base": "OLMo",
    "olmo3-7b-inst-sft": "OLMo",
    "olmo3-7b-inst-dpo": "OLMo",
    "olmo3-7b-inst-rlvr": "OLMo",
    "olmo3-7b-think-sft": "OLMo",
    "olmo3-7b-think-dpo": "OLMo",
    "olmo3-7b-think-rlvr": "OLMo",
    "olmo3-32b-think-sft": "OLMo",
    "olmo3-32b-think-dpo": "OLMo",
    "olmo3-32b-think-rlvr": "OLMo",
    "olmo3.1-32b-think": "OLMo",
}


def vendi_score(embeddings: np.ndarray) -> float:
    """Exact Vendi Score (Friedman & Dieng, 2023) using positive cosine similarity kernel."""
    n = len(embeddings)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb = embeddings / np.clip(norms, 1e-12, None)
    sim = np.clip(emb @ emb.T, 0.0, 1.0)
    K = sim / n
    eigvals = np.linalg.eigvalsh(K)
    eigvals = eigvals[eigvals > 1e-12]
    entropy = -np.sum(eigvals * np.log(eigvals))
    return float(np.exp(entropy))


def hill_effective_number(labels: List[str]) -> Tuple[float, float, int]:
    """Return (N_eff = exp(H), SMR = max_{c != F_OTHER} p_c, unique_count).
    Keeps F_OTHER in the denominator so novel tail mechanisms are not discarded,
    but never lets the F_OTHER catch-all bucket act as a single repeated modal mechanism.
    """
    if not labels:
        return 0.0, float("nan"), 0
    counts = Counter(labels)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    n_eff = float(np.exp(-np.sum(probs * np.log(probs + 1e-15))))
    named_counts = [c for k, c in counts.items() if k != "F_OTHER"]
    smr = float(max(named_counts) / total) if named_counts else 0.0
    return n_eff, smr, len(counts)


def rarefied_diversity(labels: List[str], k: int = 5, n_draws: int = 200, seed: int = 42) -> Tuple[float, float]:
    """Rarefied size-matched (N_eff, SMR) by sampling k items without replacement."""
    if len(labels) < k:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.array(labels, dtype=object)
    neffs, smrs = [], []
    for _ in range(n_draws):
        sub = arr[rng.choice(len(arr), size=k, replace=False)].tolist()
        ne, sm, _ = hill_effective_number(sub)
        neffs.append(ne)
        smrs.append(sm)
    return float(np.mean(neffs)), float(np.mean(smrs))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    sp = float(p.sum())
    sq = float(q.sum())
    if sp <= 0 or sq <= 0:
        return 1.0
    p = p / sp
    q = q / sq
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 1e-15
        return float(np.sum(a[mask] * np.log2(a[mask] / np.clip(b[mask], 1e-15, None))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def js_similarity_from_lists(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return float("nan")
    c1, c2 = Counter(a), Counter(b)
    vocab = sorted(set(c1.keys()) | set(c2.keys()))
    p = np.array([c1[v] for v in vocab], dtype=np.float64)
    q = np.array([c2[v] for v in vocab], dtype=np.float64)
    return 1.0 - js_divergence(p, q)


def bootstrap_ci(values: List[float], n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.mean(arr[idx], axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def bh_fdr(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR adjusted q-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    q = np.empty(m, dtype=np.float64)
    prev = 1.0
    for rank_idx in range(m - 1, -1, -1):
        i = order[rank_idx]
        rank = rank_idx + 1
        val = min(prev, p_values[i] * m / rank)
        q[i] = val
        prev = val
    return [float(min(1.0, x)) for x in q]


def load_all_data():
    scenarios = json.loads((DATA_DIR / "scenarios.json").read_text())
    contam_pairs = json.loads((DATA_DIR / "contamination_pairs.json").read_text())
    calibration = json.loads((DATA_DIR / "calibration.json").read_text())

    gen_records: Dict[str, Dict[str, Any]] = {}
    for p in sorted(RES_DIR.glob("gen_*.jsonl")):
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                gen_records[r["id"]] = r

    judge_scores: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for p in sorted(RES_DIR.glob("scores_judge_*.jsonl")):
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("parse_ok"):
                    judge_scores[r["judge"]][r["id"]] = r

    taxonomies: Dict[str, Dict[str, Any]] = {}
    for p in sorted(RES_DIR.glob("taxonomy_*.json")):
        d = json.loads(p.read_text())
        taxonomies[d["judge"]] = d

    emb_raw = {}
    emb_mech = {}
    if (RES_DIR / "embeddings_raw.npz").exists():
        z = np.load(RES_DIR / "embeddings_raw.npz", allow_pickle=True)
        ids = [str(x) for x in z["ids"]]
        q_arr = z["qwen3_emb"]
        m_arr = z["mpnet"]
        emb_raw = {
            "qwen3_emb": {rid: q_arr[i] for i, rid in enumerate(ids)},
            "mpnet": {rid: m_arr[i] for i, rid in enumerate(ids)},
        }
    if (RES_DIR / "embeddings_mech.npz").exists():
        z = np.load(RES_DIR / "embeddings_mech.npz", allow_pickle=True)
        ids = [str(x) for x in z["ids"]]
        q_arr = z["qwen3_emb"]
        m_arr = z["mpnet"]
        emb_mech = {
            "qwen3_emb": {rid: q_arr[i] for i, rid in enumerate(ids)},
            "mpnet": {rid: m_arr[i] for i, rid in enumerate(ids)},
        }

    return scenarios, contam_pairs, calibration, gen_records, judge_scores, taxonomies, emb_raw, emb_mech


def _quadrant_ok(arch: str, S: float, C: float, R: float) -> bool:
    if arch == "Q1_transformational":
        return (S >= 3.75) and (C >= 3.75) and (R >= 3.75)
    if arch == "Q2_incremental":
        return S <= 2.5
    if arch == "Q3_unrealistic_scifi":
        return R <= 2.5
    if arch == "Q4_data_contradicting":
        return C <= 2.5
    return False


def compute_calibration_report(calibration, judge_scores):
    judges = sorted(judge_scores.keys())
    report = {"per_judge": {}, "ensemble": {}}
    for j in judges:
        j_hits, j_total = 0, 0
        arch_scores = defaultdict(lambda: {"S": [], "C": [], "R": [], "pass": []})
        for item in calibration:
            cid = item["cal_id"]
            rec = judge_scores[j].get(cid)
            if not rec:
                continue
            j_total += 1
            arch = item["archetype"]
            ok = _quadrant_ok(arch, rec["surprise"], rec["coherence"], rec["realism"])
            j_hits += int(ok)
            arch_scores[arch]["S"].append(rec["surprise"])
            arch_scores[arch]["C"].append(rec["coherence"])
            arch_scores[arch]["R"].append(rec["realism"])
            arch_scores[arch]["pass"].append(int(ok))
        report["per_judge"][j] = {
            "accuracy": j_hits / max(1, j_total),
            "n_scored": j_total,
            "by_archetype": {
                a: {
                    "S": float(np.mean(v["S"])),
                    "C": float(np.mean(v["C"])),
                    "R": float(np.mean(v["R"])),
                    "acc": float(np.mean(v["pass"])),
                }
                for a, v in arch_scores.items()
            },
        }

    ens_hits, ens_total = 0, 0
    ens_arch = defaultdict(lambda: {"S": [], "C": [], "R": [], "pass": []})
    for item in calibration:
        cid = item["cal_id"]
        recs = [judge_scores[j][cid] for j in judges if cid in judge_scores[j]]
        if not recs:
            continue
        ens_total += 1
        S = float(np.mean([r["surprise"] for r in recs]))
        C = float(np.mean([r["coherence"] for r in recs]))
        R = float(np.mean([r["realism"] for r in recs]))
        arch = item["archetype"]
        ok = _quadrant_ok(arch, S, C, R)
        ens_hits += int(ok)
        ens_arch[arch]["S"].append(S)
        ens_arch[arch]["C"].append(C)
        ens_arch[arch]["R"].append(R)
        ens_arch[arch]["pass"].append(int(ok))

    report["ensemble"] = {
        "accuracy": ens_hits / max(1, ens_total),
        "n_scored": ens_total,
        "by_archetype": {
            a: {
                "S": float(np.mean(v["S"])),
                "C": float(np.mean(v["C"])),
                "R": float(np.mean(v["R"])),
                "acc": float(np.mean(v["pass"])),
            }
            for a, v in ens_arch.items()
        },
    }
    return report


def compute_inter_judge_agreement(judge_scores, live_ids: set):
    judges = sorted(judge_scores.keys())
    out = {}
    for i in range(len(judges)):
        for k in range(i + 1, len(judges)):
            j1, j2 = judges[i], judges[k]
            common = sorted((set(judge_scores[j1].keys()) & set(judge_scores[j2].keys())) & live_ids)
            pair_key = f"{j1}__vs__{j2}"
            pair_stats = {"n_common": len(common)}
            for dim in ("surprise", "coherence", "realism"):
                a = np.array([judge_scores[j1][x][dim] for x in common])
                b = np.array([judge_scores[j2][x][dim] for x in common])
                rho, _ = stats.spearmanr(a, b)
                r, _ = stats.pearsonr(a, b)
                qwk = cohen_kappa_score(a, b, weights="quadratic")
                exact_1 = float(np.mean(np.abs(a - b) <= 1))
                pair_stats[dim] = {
                    "spearman": float(rho),
                    "pearson": float(r),
                    "qwk": float(qwk),
                    "within_1": exact_1,
                }
            out[pair_key] = pair_stats
    return out


def aggregate_item_scores(gen_records, judge_scores, taxonomies):
    """Build unified DataFrame with strictly cross-family judge scores (excluding self-family)."""
    judges = sorted(judge_scores.keys())
    primary_tax_judge = "gpt-oss-120b" if "gpt-oss-120b" in taxonomies else sorted(taxonomies.keys())[0]
    secondary_tax_judge = next((j for j in sorted(taxonomies.keys()) if j != primary_tax_judge), None)

    rows = []
    for rid, g in gen_records.items():
        model = g["model"]
        mfam = MODEL_FAMILY.get(model, "Other")
        xfam_recs = [
            judge_scores[j][rid]
            for j in judges
            if rid in judge_scores[j] and MODEL_FAMILY.get(j, "") != mfam
        ]
        all_recs = [judge_scores[j][rid] for j in judges if rid in judge_scores[j]]
        recs = xfam_recs if xfam_recs else all_recs
        if not recs:
            continue

        S = float(np.mean([r["surprise"] for r in recs]))
        C = float(np.mean([r["coherence"] for r in recs]))
        R = float(np.mean([r["realism"] for r in recs]))
        contra = float(np.mean([1.0 if r.get("n_contradicted", 0) >= 1 else 0.0 for r in recs]))

        valid = (C >= 3.5) and (R >= 3.5) and (contra < 0.5)
        transformational = valid and (S >= 3.5)
        strict_trans = (S >= 4.0) and (C >= 4.0) and (R >= 4.0) and (contra == 0.0)

        fw1 = taxonomies[primary_tax_judge]["assignments"].get(rid, "F_MISSING")
        fw2 = taxonomies[secondary_tax_judge]["assignments"].get(rid, "F_MISSING") if secondary_tax_judge else "NA"

        rows.append(
            {
                "id": rid,
                "model": model,
                "family": mfam,
                "mode": g["mode"],
                "scenario_id": g["scenario_id"],
                "domain": g["domain"],
                "sample_idx": g.get("sample_idx", 0),
                "call_idx": g.get("call_idx", -1),
                "num_tokens": g.get("num_tokens", 0),
                "think_chars": len(g.get("thinking", "") or ""),
                "hyp_chars": len(g.get("hypothesis", "") or ""),
                "finish_reason": g.get("finish_reason", ""),
                "truncated": int(g.get("finish_reason", "") == "length"),
                "repaired": int(bool(g.get("repaired", False))),
                "surprise": S,
                "coherence": C,
                "realism": R,
                "contradicted": contra,
                "valid": int(valid),
                "transformational": int(transformational),
                "strict_trans": int(strict_trans),
                "fw_primary": fw1,
                "fw_secondary": fw2,
                "n_judges": len(recs),
            }
        )
    return pd.DataFrame(rows), primary_tax_judge, secondary_tax_judge


def compute_group_metrics(df_sub: pd.DataFrame, emb_raw: Dict, emb_mech: Dict) -> Dict[str, Any]:
    """Compute scenario-averaged Sci-TwistBench metrics excluding F0, plus valid-only & rarefied controls."""
    per_scen = []
    for sid, grp in df_sub.groupby("scenario_id"):
        n = len(grp)
        S_m = float(grp["surprise"].mean())
        C_m = float(grp["coherence"].mean())
        R_m = float(grp["realism"].mean())
        vr = float(grp["valid"].mean())
        tvr = float(grp["transformational"].mean())
        stvr = float(grp["strict_trans"].mean())
        ccr = float((grp["contradicted"] >= 0.5).mean())
        trunc_rate = float(grp["truncated"].mean())

        grp_comp = grp[grp["truncated"] == 0]
        tvr_comp = float(grp_comp["transformational"].mean()) if len(grp_comp) else float("nan")

        # Residual F0 catch-all rate (tracked transparently, EXCLUDED from SMR / N_eff)
        f0_rate = float((grp["fw_primary"] == "F0").mean())

        # Assigned (non-F0) frameworks across all runs
        fws_all = [x for x in grp["fw_primary"].tolist() if x not in BAD_FW]
        n_eff_all, smr_all, k_all = hill_effective_number(fws_all)

        # Top-2 cumulative share (named frameworks, non-F0/F_OTHER)
        if fws_all:
            named_cnts = [c for k, c in Counter(fws_all).most_common() if k != "F_OTHER"]
            top2_all = sum(named_cnts[:2]) / len(fws_all)
        else:
            top2_all = float("nan")

        # Secondary taxonomy (non-F0)
        fws_sec = [x for x in grp["fw_secondary"].tolist() if x not in BAD_FW]
        n_eff_sec, smr_sec, _ = hill_effective_number(fws_sec) if fws_sec else (float("nan"), float("nan"), 0)

        # Quality-gated (valid & transformational subsets, non-F0)
        grp_val = grp[grp["valid"] == 1]
        grp_tr = grp[grp["transformational"] == 1]
        fws_val = [x for x in grp_val["fw_primary"].tolist() if x not in BAD_FW]
        fws_tr = [x for x in grp_tr["fw_primary"].tolist() if x not in BAD_FW]
        n_eff_val, smr_val, k_val = hill_effective_number(fws_val) if fws_val else (float("nan"), float("nan"), 0)
        n_eff_tr, smr_tr, k_tr = hill_effective_number(fws_tr) if fws_tr else (float("nan"), float("nan"), 0)

        # Rarefied size-matched valid-only diversity (k=5 and k=10)
        n_eff_val_r5, smr_val_r5 = rarefied_diversity(fws_val, k=5, n_draws=200, seed=42)
        n_eff_val_r10, smr_val_r10 = rarefied_diversity(fws_val, k=10, n_draws=200, seed=42)

        # Secondary taxonomy valid-only
        fws_val_sec = [x for x in grp_val["fw_secondary"].tolist() if x not in BAD_FW]
        n_eff_val_sec, smr_val_sec, _ = hill_effective_number(fws_val_sec) if fws_val_sec else (float("nan"), float("nan"), 0)

        # Transformational Effective Diversity: TED = TVR * N_eff(T)
        ted = tvr * (0.0 if np.isnan(n_eff_tr) else n_eff_tr)

        ids_all = grp["id"].tolist()
        ids_val = grp_val["id"].tolist()
        q_raw_vecs = [emb_raw["qwen3_emb"][i] for i in ids_all if i in emb_raw.get("qwen3_emb", {})] if emb_raw else []
        m_raw_vecs = [emb_raw["mpnet"][i] for i in ids_all if i in emb_raw.get("mpnet", {})] if emb_raw else []
        q_mech_vecs = [emb_mech["qwen3_emb"][i] for i in ids_all if i in emb_mech.get("qwen3_emb", {})] if emb_mech else []
        q_val_vecs = [emb_raw["qwen3_emb"][i] for i in ids_val if i in emb_raw.get("qwen3_emb", {})] if emb_raw else []

        vs_qwen_raw = vendi_score(np.array(q_raw_vecs)) if q_raw_vecs else float("nan")
        vs_mpnet_raw = vendi_score(np.array(m_raw_vecs)) if m_raw_vecs else float("nan")
        vs_qwen_mech = vendi_score(np.array(q_mech_vecs)) if q_mech_vecs else float("nan")
        vs_qwen_val = vendi_score(np.array(q_val_vecs)) if q_val_vecs else float("nan")

        per_scen.append(
            {
                "scenario_id": sid,
                "n": n,
                "n_valid_assigned": len(fws_val),
                "n_assigned": len(fws_all),
                "n_judges_mean": float(grp["n_judges"].mean()),
                "trunc_rate": trunc_rate,
                "f0_rate": f0_rate,
                "surprise": S_m,
                "coherence": C_m,
                "realism": R_m,
                "vr": vr,
                "tvr": tvr,
                "tvr_comp": tvr_comp,
                "stvr": stvr,
                "ccr": ccr,
                "smr": smr_all,
                "top2": top2_all,
                "n_eff": n_eff_all,
                "k_unique": k_all,
                "smr_sec": smr_sec,
                "n_eff_sec": n_eff_sec,
                "smr_val": smr_val,
                "n_eff_val": n_eff_val,
                "smr_val_sec": smr_val_sec,
                "n_eff_val_sec": n_eff_val_sec,
                "smr_val_r5": smr_val_r5,
                "n_eff_val_r5": n_eff_val_r5,
                "smr_val_r10": smr_val_r10,
                "n_eff_val_r10": n_eff_val_r10,
                "smr_tr": smr_tr,
                "n_eff_tr": n_eff_tr,
                "ted": ted,
                "vs_qwen_raw": vs_qwen_raw,
                "vs_mpnet_raw": vs_mpnet_raw,
                "vs_qwen_mech": vs_qwen_mech,
                "vs_qwen_val": vs_qwen_val,
                "think_chars": float(grp["think_chars"].mean()),
            }
        )

    out = {
        "n_scenarios": len(per_scen),
        "n_samples_total": int(len(df_sub)),
        "n_judges": float(df_sub["n_judges"].mean()) if len(df_sub) else 0.0,
        "n_scenarios_val_ge1": sum(1 for x in per_scen if x["n_valid_assigned"] >= 1),
        "n_scenarios_val_ge5": sum(1 for x in per_scen if x["n_valid_assigned"] >= 5),
        "n_scenarios_val_ge10": sum(1 for x in per_scen if x["n_valid_assigned"] >= 10),
        "per_scenario": per_scen,
    }
    metric_keys = [
        "trunc_rate", "f0_rate", "surprise", "coherence", "realism", "vr", "tvr", "tvr_comp", "stvr", "ccr",
        "smr", "top2", "n_eff", "k_unique", "smr_sec", "n_eff_sec",
        "smr_val", "n_eff_val", "smr_val_sec", "n_eff_val_sec",
        "smr_val_r5", "n_eff_val_r5", "smr_val_r10", "n_eff_val_r10",
        "smr_tr", "n_eff_tr", "ted",
        "vs_qwen_raw", "vs_mpnet_raw", "vs_qwen_mech", "vs_qwen_val", "think_chars",
    ]
    for k in metric_keys:
        vals = [x[k] for x in per_scen]
        m, lo, hi = bootstrap_ci(vals)
        out[k] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
    return out


def compute_hivemind_analysis(df: pd.DataFrame, models: List[str], n_splits: int = 100) -> Dict[str, Any]:
    """Cross-model JS similarity, split-half within-model baseline (matched n=15), random baseline, and ensemble."""
    sub = df[(df["mode"] == "direct") & (df["model"].isin(models))]
    sids = sorted(sub["scenario_id"].unique())
    n_m = len(models)

    jss_full = np.ones((n_m, n_m), dtype=np.float64)
    jss_matched = np.zeros((n_m, n_m), dtype=np.float64)
    top1_mat = np.ones((n_m, n_m), dtype=np.float64)

    # Pre-extract non-F0 valid framework lists per (model, scenario)
    fw_lists: Dict[Tuple[str, str], List[str]] = {}
    for m in models:
        for sid in sids:
            g = sub[(sub["model"] == m) & (sub["scenario_id"] == sid)]
            fw_lists[(m, sid)] = [x for x in g["fw_primary"].tolist() if x not in BAD_FW]

    rng = np.random.default_rng(42)

    # Diagonal: within-model split-half JS similarity (n=15 vs n=15, averaged over n_splits)
    split_half_per_model = {}
    for i, m in enumerate(models):
        scen_sims = []
        for sid in sids:
            lst = fw_lists[(m, sid)]
            half = len(lst) // 2
            if half < 2:
                continue
            arr = np.array(lst, dtype=object)
            sims = []
            for _ in range(n_splits):
                perm = rng.permutation(len(arr))
                sims.append(js_similarity_from_lists(arr[perm[:half]].tolist(), arr[perm[half:2 * half]].tolist()))
            scen_sims.append(float(np.mean(sims)))
        val = float(np.mean(scen_sims)) if scen_sims else float("nan")
        jss_matched[i, i] = val
        split_half_per_model[m] = val

    # Off-diagonal: full-sample JS similarity, matched-n (n=15 vs n=15) JS similarity, and Top-1 modal agreement
    for i in range(n_m):
        for j in range(i + 1, n_m):
            m1, m2 = models[i], models[j]
            full_vals, match_vals, top1_hits = [], [], []
            for sid in sids:
                l1, l2 = fw_lists[(m1, sid)], fw_lists[(m2, sid)]
                if not l1 or not l2:
                    continue
                full_vals.append(js_similarity_from_lists(l1, l2))
                t1 = Counter(l1).most_common(1)[0][0]
                t2 = Counter(l2).most_common(1)[0][0]
                top1_hits.append(int(t1 == t2))

                k_draw = min(15, len(l1), len(l2))
                a1 = np.array(l1, dtype=object)
                a2 = np.array(l2, dtype=object)
                ms = []
                for _ in range(n_splits):
                    s1 = a1[rng.choice(len(a1), size=k_draw, replace=False)].tolist()
                    s2 = a2[rng.choice(len(a2), size=k_draw, replace=False)].tolist()
                    ms.append(js_similarity_from_lists(s1, s2))
                match_vals.append(float(np.mean(ms)))

            jss_full[i, j] = jss_full[j, i] = float(np.mean(full_vals))
            jss_matched[i, j] = jss_matched[j, i] = float(np.mean(match_vals))
            top1_mat[i, j] = top1_mat[j, i] = float(np.mean(top1_hits))

    # Permuted-attractor null baseline: if two models had the same degree of collapse but picked
    # independent random mechanisms from the scenario's active frameworks (matched n=15)
    perm_sims = []
    for sid in sids:
        active = sorted({fw for m in models for fw in fw_lists[(m, sid)]})
        if len(active) < 2:
            perm_sims.append(1.0)
            continue
        ps = []
        for _ in range(n_splits):
            m1, m2 = rng.choice(models, size=2, replace=False)
            l1, l2 = fw_lists[(m1, sid)], fw_lists[(m2, sid)]
            if not l1 or not l2:
                continue
            k_draw = min(15, len(l1), len(l2))
            s1 = rng.choice(l1, size=k_draw, replace=False).tolist()
            s2 = rng.choice(l2, size=k_draw, replace=False).tolist()
            perm_map = dict(zip(active, rng.permutation(active).tolist()))
            s2_perm = [perm_map[x] for x in s2]
            ps.append(js_similarity_from_lists(s1, s2_perm))
        if ps:
            perm_sims.append(float(np.mean(ps)))

    # Per-scenario monoculture breakdown across all models
    per_scenario_consensus = []
    for sid in sids:
        m_smrs = []
        m_top1s = []
        pooled_fws = []
        for m in models:
            lst = fw_lists[(m, sid)]
            if lst:
                _, sm, _ = hill_effective_number(lst)
                m_smrs.append(sm)
                m_top1s.append(Counter(lst).most_common(1)[0][0])
                pooled_fws.extend(lst)
        top_modal, top_modal_cnt = Counter(m_top1s).most_common(1)[0] if m_top1s else ("NONE", 0)
        pool_neff, pool_smr, pool_k = hill_effective_number(pooled_fws)
        per_scenario_consensus.append(
            {
                "scenario_id": sid,
                "mean_model_smr": float(np.mean(m_smrs)) if m_smrs else float("nan"),
                "pooled_smr": pool_smr,
                "pooled_n_eff": pool_neff,
                "pooled_k_unique": pool_k,
                "models_agreeing_on_modal": int(top_modal_cnt),
                "n_models": len(m_top1s),
                "modal_framework": top_modal,
            }
        )

    # Multi-model equal-budget ensemble (30 samples/scenario for 5 models, or 35 for 7 models)
    per_m_draw = max(1, round(30 / n_m))
    ens_neffs, ens_smrs, ens_top2s, ens_tvrs, ens_teds = [], [], [], [], []
    for sid in sids:
        pooled_rows = []
        for m in models:
            m_df = sub[(sub["model"] == m) & (sub["scenario_id"] == sid)]
            if len(m_df) >= per_m_draw:
                pooled_rows.append(m_df.sample(n=per_m_draw, random_state=42))
            else:
                pooled_rows.append(m_df)
        p_df = pd.concat(pooled_rows, ignore_index=True)
        fws = [x for x in p_df["fw_primary"].tolist() if x not in BAD_FW]
        ne, smr, _ = hill_effective_number(fws)
        cnts = [c for _, c in Counter(fws).most_common()]
        t2 = sum(cnts[:2]) / sum(cnts) if cnts else float("nan")
        tvr = float(p_df["transformational"].mean())
        tr = p_df[p_df["transformational"] == 1]
        tr_fws = [x for x in tr["fw_primary"].tolist() if x not in BAD_FW]
        ne_tr, _, _ = hill_effective_number(tr_fws) if tr_fws else (0.0, float("nan"), 0)
        ens_neffs.append(ne)
        ens_smrs.append(smr)
        ens_top2s.append(t2)
        ens_tvrs.append(tvr)
        ens_teds.append(tvr * (0.0 if np.isnan(ne_tr) else ne_tr))

    # Full-pool rank-frequency distribution (all runs pooled across all models)
    pool_top1, pool_top2 = [], []
    for sid in sids:
        all_fws = [fw for m in models for fw in fw_lists[(m, sid)]]
        cnts = [c for _, c in Counter(all_fws).most_common()]
        if cnts:
            pool_top1.append(cnts[0] / sum(cnts))
            pool_top2.append(sum(cnts[:2]) / sum(cnts))

    tri = np.triu_indices(n_m, 1)
    return {
        "models": models,
        "jss_matrix_full": jss_full.tolist(),
        "jss_matrix_matched15": jss_matched.tolist(),
        "top1_agreement_matrix": top1_mat.tolist(),
        "split_half_within_model_jss": split_half_per_model,
        "mean_within_model_jss_matched15": float(np.mean(np.diag(jss_matched))),
        "mean_offdiag_jss_full": float(np.mean(jss_full[tri])),
        "mean_offdiag_jss_matched15": float(np.mean(jss_matched[tri])),
        "permuted_attractor_jss_baseline15": float(np.mean(perm_sims)),
        "mean_offdiag_top1": float(np.mean(top1_mat[tri])),
        "per_scenario_consensus": per_scenario_consensus,
        "pooled_rank1_share": float(np.mean(pool_top1)),
        "pooled_top2_share": float(np.mean(pool_top2)),
        "ensemble_equal_budget": {
            "samples_per_model_per_scenario": per_m_draw,
            "total_per_scenario": per_m_draw * n_m,
            "n_eff": bootstrap_ci(ens_neffs),
            "smr": bootstrap_ci(ens_smrs),
            "top2": bootstrap_ci(ens_top2s),
            "tvr": bootstrap_ci(ens_tvrs),
            "ted": bootstrap_ci(ens_teds),
        },
    }


def compute_vs_diagnostics(df: pd.DataFrame, models: List[str]) -> Dict[str, Any]:
    """Compute structural Verbalized Sampling diagnostics: share in Direct's modal mechanism & within-call merge rate."""
    out = {}
    for m in models:
        d_dir = df[(df["model"] == m) & (df["mode"] == "direct")]
        d_vs = df[(df["model"] == m) & (df["mode"] == "vs")]
        if len(d_dir) == 0 or len(d_vs) == 0:
            continue

        direct_modal = {}
        for sid, g in d_dir.groupby("scenario_id"):
            fws = [x for x in g["fw_primary"].tolist() if x not in BAD_FW and x != "F_OTHER"]
            if fws:
                direct_modal[sid] = Counter(fws).most_common(1)[0][0]

        modal_shares = []
        n_calls_total = 0
        n_calls_5 = 0
        n_calls_merged = 0
        uniq_per_5call = []

        for sid, g in d_vs.groupby("scenario_id"):
            fws = [x for x in g["fw_primary"].tolist() if x not in BAD_FW]
            if fws and sid in direct_modal:
                modal_shares.append(sum(1 for x in fws if x == direct_modal[sid]) / len(fws))
            for ci, cg in g.groupby("call_idx"):
                if ci < 0:
                    continue
                n_calls_total += 1
                cfws = [x for x in cg["fw_primary"].tolist() if x not in BAD_FW]
                named_cfws = [x for x in cfws if x != "F_OTHER"]
                if len(cg) == 5:
                    n_calls_5 += 1
                    # Never count two F_OTHER tail items as merged into the same mechanism
                    if len(named_cfws) >= 2 and len(set(named_cfws)) < len(named_cfws):
                        n_calls_merged += 1
                    n_other = sum(1 for x in cfws if x == "F_OTHER")
                    uniq_per_5call.append(len(set(named_cfws)) + n_other)

        out[m] = {
            "n_vs_samples": int(len(d_vs)),
            "n_calls_total": n_calls_total,
            "n_calls_complete_5": n_calls_5,
            "direct_modal_share_in_vs": float(np.mean(modal_shares)) if modal_shares else float("nan"),
            "structural_floor_1_of_5": 0.20,
            "calls_with_within_call_merge": n_calls_merged,
            "within_call_merge_rate": n_calls_merged / max(1, n_calls_5),
            "mean_unique_frameworks_per_5call": float(np.mean(uniq_per_5call)) if uniq_per_5call else float("nan"),
        }
    return out


def compute_vendi_correlations(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically compute Spearman rank correlations between Vendi scores and N_eff across all conditions."""
    rows = []
    for m, d in summary["frontier_direct"].items():
        rows.append((f"{m}__direct", d["n_eff"]["mean"], d["vs_qwen_raw"]["mean"], d["vs_mpnet_raw"]["mean"], d["vs_qwen_mech"]["mean"]))
    for m, d in summary["olmo_lineage"].items():
        rows.append((f"{m}__direct", d["n_eff"]["mean"], d["vs_qwen_raw"]["mean"], d["vs_mpnet_raw"]["mean"], d["vs_qwen_mech"]["mean"]))
    for m, modes in summary["interventions"].items():
        for mode, d in modes.items():
            if mode == "direct":
                continue
            rows.append((f"{m}__{mode}", d["n_eff"]["mean"], d["vs_qwen_raw"]["mean"], d["vs_mpnet_raw"]["mean"], d["vs_qwen_mech"]["mean"]))

    def _corr(idx_a, idx_b):
        pts = [(r[idx_a], r[idx_b]) for r in rows if not np.isnan(r[idx_a]) and not np.isnan(r[idx_b])]
        if len(pts) < 3:
            return {"spearman": float("nan"), "p_value": float("nan"), "n": len(pts)}
        rho, p = stats.spearmanr([x[0] for x in pts], [x[1] for x in pts])
        return {"spearman": float(rho), "p_value": float(p), "n": len(pts)}

    return {
        "neff_vs_vendi_qwen_raw": _corr(1, 2),
        "neff_vs_vendi_mpnet_raw": _corr(1, 3),
        "neff_vs_vendi_qwen_mech": _corr(1, 4),
    }


def make_figures(summary: Dict[str, Any], df: pd.DataFrame):
    """Generate 5 single-claim golden-ratio plots matching ethz-spylab/modal-aphasia 1:1 physical styling."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import plot_util
    import matplotlib.pyplot as plt

    plot_util.setup_matplotlib()

    SHORT_MODEL = {
        "qwen3.8-27b": "Qwen3.8",
        "gemma-4-31b-it": "Gemma-4",
        "deepseek-v4-flash": "DeepSeek",
        "mistral-medium-3.5-128b": "Mistral",
        "gpt-oss-120b": "gpt-oss",
        "claude-sonnet-5.5": "Sonnet 5.5",
        "gpt-6.1-terra": "GPT-6.1",
    }

    main_models = [m for m in ALL_MAIN_MODELS if m in summary["frontier_direct"]]
    main_labels = [SHORT_MODEL.get(m, MODEL_DISPLAY[m]) for m in main_models]
    open_models = [m for m in OPEN_MODELS if m in summary["frontier_direct"]]
    closed_models = [m for m in CLOSED_MODELS if m in summary["frontier_direct"]]

    # ==================== FIGURE 1: Rank-Frequency Collapse Cliff (Half-width 1:1) ====================
    fig, ax = plt.subplots(figsize=plot_util.FIGURE_SIZE_HALF_IN)
    ranks_x = np.arange(1, 6)
    indiv_curves = []
    for m in main_models:
        sub = df[(df["model"] == m) & (df["mode"] == "direct")]
        r_vec = np.zeros(5)
        n_sc = 0
        for sid, g in sub.groupby("scenario_id"):
            fws = [x for x in g["fw_primary"].tolist() if x not in BAD_FW]
            if not fws:
                continue
            n_sc += 1
            cnts = [c for _, c in Counter(fws).most_common()]
            tot = sum(cnts)
            for r in range(min(4, len(cnts))):
                r_vec[r] += cnts[r] / tot
            if len(cnts) > 4:
                r_vec[4] += sum(cnts[4:]) / tot
        r_vec = 100.0 * r_vec / max(1, n_sc)
        indiv_curves.append(r_vec)
        col = "#999999" if m in OPEN_MODELS else "#7b5ea7"
        ax.plot(ranks_x, r_vec, color=col, lw=0.7, alpha=0.45)

    mean_curve = np.mean(indiv_curves, axis=0)
    sub_all = df[(df["model"].isin(main_models)) & (df["mode"] == "direct")]
    ens_vec = np.zeros(5)
    n_sc = 0
    for sid, g in sub_all.groupby("scenario_id"):
        fws = [x for x in g["fw_primary"].tolist() if x not in BAD_FW]
        if not fws:
            continue
        n_sc += 1
        cnts = [c for _, c in Counter(fws).most_common()]
        tot = sum(cnts)
        for r in range(min(4, len(cnts))):
            ens_vec[r] += cnts[r] / tot
        if len(cnts) > 4:
            ens_vec[4] += sum(cnts[4:]) / tot
    ens_vec = 100.0 * ens_vec / max(1, n_sc)

    disjoint_pct = 100.0 / max(1, len(main_models))
    ax.axhline(disjoint_pct, color="#666666", linestyle=":", lw=0.85, label=f"Disjoint (1/{len(main_models)})")
    ax.plot(ranks_x, mean_curve, "-o", color="C2", markersize=3.8, label="Single model")
    ax.plot(ranks_x, ens_vec, "--s", color="C0", markersize=3.6, label=f"{len(main_models)}-Model pool")
    ax.set_xticks(ranks_x)
    ax.set_xticklabels(["1st", "2nd", "3rd", "4th", "5th+"])
    ax.set_xlabel(r"Mechanism Rank")
    ax.set_ylabel(r"Share of Runs")
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([r"0\%", r"20\%", r"40\%", r"60\%", r"80\%", r"100\%"])
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, columnspacing=0.6, handlelength=1.0)
    plot_util.resize_figure_with_legend(fig)
    fig.savefig(FIG_DIR / "fig1_frontier_monoculture.pdf")
    plt.close(fig)

    # ==================== FIGURE 2: Cross-Model Similarity vs Split-Half Ceiling (Half-width 1:1) ====================
    fig, ax = plt.subplots(figsize=plot_util.FIGURE_SIZE_HALF_IN)
    hm = summary["hivemind_all7"] if "hivemind_all7" in summary else summary["hivemind_open5"]
    mat = np.array(hm["jss_matrix_matched15"])
    n_hm = len(hm["models"])
    hm_labels = [SHORT_MODEL.get(m, MODEL_DISPLAY[m]) for m in hm["models"]]
    im = ax.imshow(mat, vmin=0.35, vmax=1.0, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n_hm))
    ax.set_yticks(range(n_hm))
    ax.set_xticklabels(hm_labels, rotation=28, ha="right", fontsize=6.0)
    ax.set_yticklabels(hm_labels, fontsize=6.0)
    for i in range(n_hm):
        for j in range(n_hm):
            val = mat[i, j]
            weight = "bold" if i == j else "normal"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=5.2, fontweight=weight,
                color="white" if val > 0.72 else "black",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.03)
    cbar.set_label(r"Jensen--Shannon Similarity")
    cbar.ax.tick_params(labelsize=6.5)
    fig.savefig(FIG_DIR / "fig2_hivemind_heatmap.pdf")
    plt.close(fig)

    # ==================== FIGURE 3: Post-Training Trajectory (Rarefied Valid SMR vs TVR, Nine-col 1:1) ====================
    fig, ax = plt.subplots(figsize=(plot_util.FIGURE_SIZE_NINECOL_IN[0], 1.95))
    inst_keys = [m for m in OLMO_INST if m in summary["olmo_lineage"]]
    think_keys = [m for m in OLMO_THINK if m in summary["olmo_lineage"]]
    think32_keys = [m for m in OLMO_32B_THINK[:3] if m in summary["olmo_lineage"]]

    def _pts(keys):
        xs = [summary["olmo_lineage"][m]["smr_val_r5"]["mean"] * 100 for m in keys]
        ys = [summary["olmo_lineage"][m]["tvr"]["mean"] * 100 for m in keys]
        return xs, ys

    all_x, all_y = [], []
    if len(inst_keys) == 3:
        ix, iy = _pts(inst_keys)
        if "olmo3-7b-base" in summary["olmo_lineage"]:
            b7x = summary["olmo_lineage"]["olmo3-7b-base"]["smr_val_r5"]["mean"] * 100
            b7y = summary["olmo_lineage"]["olmo3-7b-base"]["tvr"]["mean"] * 100
            all_x.append(b7x); all_y.append(b7y)
            ax.scatter([b7x], [b7y], s=40, marker="D", color="C0", facecolors="none", lw=1.1, zorder=6)
            ax.annotate("7B-Base", (b7x, b7y), xytext=(6, -3), textcoords="offset points", fontsize=6.5, color="#0D6382")
            ax.annotate("", xy=(ix[0], iy[0]), xytext=(b7x, b7y), arrowprops=dict(arrowstyle="->", color="C0", lw=0.9, linestyle="--", shrinkA=5, shrinkB=5))
        all_x.extend(ix); all_y.extend(iy)
        ax.plot(ix, iy, "-o", color="C0", markersize=4.8, label="7B-Instruct")
        for lbl, px, py, dx, dy in zip(["SFT", "DPO", "RLVR"], ix, iy, [6, 5, 5], [-3, -10, -4]):
            ax.annotate(lbl, (px, py), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color="#0D6382")
        for k in range(2):
            ax.annotate("", xy=(ix[k + 1], iy[k + 1]), xytext=(ix[k], iy[k]), arrowprops=dict(arrowstyle="->", color="C0", lw=1.0, shrinkA=5, shrinkB=5))

    if len(think_keys) == 3:
        tx, ty = _pts(think_keys)
        all_x.extend(tx); all_y.extend(ty)
        ax.plot(tx, ty, "-s", color="C2", markersize=4.8, label="7B-Think")
        for lbl, px, py, dx, dy in zip(["SFT", "DPO", "RLVR"], tx, ty, [5, 5, -22], [-4, -2, -9]):
            ax.annotate(lbl, (px, py), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color="#9e0e30")
        for k in range(2):
            ax.annotate("", xy=(tx[k + 1], ty[k + 1]), xytext=(tx[k], ty[k]), arrowprops=dict(arrowstyle="->", color="C2", lw=1.0, shrinkA=5, shrinkB=5))

    if len(think32_keys) == 3:
        t32x, t32y = _pts(think32_keys)
        if "olmo3-32b-base" in summary["olmo_lineage"]:
            bx = summary["olmo_lineage"]["olmo3-32b-base"]["smr_val_r5"]["mean"] * 100
            by = summary["olmo_lineage"]["olmo3-32b-base"]["tvr"]["mean"] * 100
            all_x.append(bx); all_y.append(by)
            ax.scatter([bx], [by], s=42, marker="D", color="C3", facecolors="none", lw=1.1, zorder=6)
            ax.annotate("32B-Base", (bx, by), xytext=(-36, 4), textcoords="offset points", fontsize=6.5, color="C3")
            ax.annotate("", xy=(t32x[0], t32y[0]), xytext=(bx, by), arrowprops=dict(arrowstyle="->", color="C3", lw=0.9, linestyle="--", shrinkA=5, shrinkB=5))
        all_x.extend(t32x); all_y.extend(t32y)
        ax.plot(t32x, t32y, "-^", color="C3", markersize=5.0, label="32B-Think")
        for lbl, px, py, dx, dy in zip(["SFT", "DPO", "RLVR"], t32x, t32y, [-20, 5, -10], [4, 2, 5]):
            ax.annotate(lbl, (px, py), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color="C3")
        for k in range(2):
            ax.annotate("", xy=(t32x[k + 1], t32y[k + 1]), xytext=(t32x[k], t32y[k]), arrowprops=dict(arrowstyle="->", color="C3", lw=1.0, shrinkA=5, shrinkB=5))

    ax.set_xlabel(r"Valid Scientific Monoculture Rate ($\mathrm{SMR}_{\mathrm{val}@5}$)")
    ax.set_ylabel(r"Transformational Validity Rate")
    ax.set_xlim(47, 85)
    ax.set_ylim(8, 104)
    ax.set_xticks([50, 60, 70, 80])
    ax.set_xticklabels([r"50\%", r"60\%", r"70\%", r"80\%"])
    ax.set_yticks([10, 30, 50, 70, 90])
    ax.set_yticklabels([r"10\%", r"30\%", r"50\%", r"70\%", r"90\%"])
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3)
    plot_util.resize_figure_with_legend(fig)
    fig.savefig(FIG_DIR / "fig3_olmo_posttraining.pdf")
    plt.close(fig)

    # ==================== FIGURE 4: Inference-Time Shifts (5 Open + 2 Proprietary, Half-width 1:1) ====================
    fig, ax = plt.subplots(figsize=plot_util.FIGURE_SIZE_HALF_IN)
    open_exp3 = [m for m in OPEN_MODELS if m in summary["interventions"] and len(summary["interventions"][m]) == 5]
    closed_exp3 = [m for m in CLOSED_MODELS if m in summary["interventions"] and {"direct", "axiom_neg", "icr", "vs"}.issubset(summary["interventions"][m].keys())]

    mode_style = {
        "direct": ("s", 30, "Baseline"),
        "axiom_neg": ("o", 24, "Axiom-Neg"),
        "high_temp": ("v", 26, "High-T"),
        "icr": ("D", 22, "ICR"),
        "vs": ("*", 58, "VS"),
    }

    if open_exp3:
        mean_pts = {}
        for mode, (mk, sz, lbl) in mode_style.items():
            xs = [summary["interventions"][m][mode]["smr"]["mean"] * 100 for m in open_exp3]
            ys = [summary["interventions"][m][mode]["tvr"]["mean"] * 100 for m in open_exp3]
            mx, my = float(np.mean(xs)), float(np.mean(ys))
            mean_pts[mode] = (mx, my)
            ax.scatter(xs, ys, s=8, marker=mk, color="C0", alpha=0.16, linewidths=0, zorder=3)
            ax.scatter(
                [mx], [my], s=sz, marker=mk, color="C0",
                edgecolor="#1b2631" if mode == "direct" else "none", lw=0.6,
                zorder=6, label="Open-weight" if mode == "direct" else None,
            )

        dx0, dy0 = mean_pts["direct"]
        for mode in ("axiom_neg", "high_temp", "icr", "vs"):
            px, py = mean_pts[mode]
            ax.annotate("", xy=(px, py), xytext=(dx0, dy0), arrowprops=dict(arrowstyle="->", color="C0", lw=0.95, alpha=0.85, shrinkA=4, shrinkB=4))

        ax.annotate("Baseline", mean_pts["direct"], xytext=(5, -8), textcoords="offset points", fontsize=5.8, color="#0D6382", fontweight="bold")
        ax.annotate("Axiom-Neg", mean_pts["axiom_neg"], xytext=(-16, -15), textcoords="offset points", fontsize=5.6, color="#0D6382", bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.88), zorder=8)
        ax.annotate("High-T", mean_pts["high_temp"], xytext=(5, -2), textcoords="offset points", fontsize=5.6, color="#0D6382")
        ax.annotate("ICR", mean_pts["icr"], xytext=(6, -2), textcoords="offset points", fontsize=5.6, color="#0D6382", bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.82), zorder=8)
        ax.annotate("VS", mean_pts["vs"], xytext=(-12, -8), textcoords="offset points", fontsize=5.8, color="#0D6382", fontweight="bold")

    if closed_exp3:
        c_pts = {}
        for mode in ("direct", "axiom_neg", "icr", "vs"):
            mk, sz, lbl = mode_style[mode]
            xs = [summary["interventions"][m][mode]["smr"]["mean"] * 100 for m in closed_exp3]
            ys = [summary["interventions"][m][mode]["tvr"]["mean"] * 100 for m in closed_exp3]
            mx, my = float(np.mean(xs)), float(np.mean(ys))
            c_pts[mode] = (mx, my)
            ax.scatter(xs, ys, s=8, marker=mk, color="C2", alpha=0.20, linewidths=0, zorder=4)
            ax.scatter(
                [mx], [my], s=sz, marker=mk, color="C2",
                edgecolor="#1b2631" if mode == "direct" else "none", lw=0.6,
                zorder=7, label="Proprietary" if mode == "direct" else None,
            )
        cdx, cdy = c_pts["direct"]
        for mode in ("axiom_neg", "icr", "vs"):
            px, py = c_pts[mode]
            ax.annotate("", xy=(px, py), xytext=(cdx, cdy), arrowprops=dict(arrowstyle="->", color="C2", lw=0.95, linestyle="--", alpha=0.9, shrinkA=4, shrinkB=4))
        ax.annotate("Axiom-Neg", c_pts["axiom_neg"], xytext=(-14, 4), textcoords="offset points", fontsize=5.5, color="#9e0e30")
        ax.annotate("ICR", c_pts["icr"], xytext=(-5, 5), textcoords="offset points", fontsize=5.6, color="#9e0e30", fontweight="bold")
        ax.annotate("VS", c_pts["vs"], xytext=(-13, 3), textcoords="offset points", fontsize=5.6, color="#9e0e30", fontweight="bold")

    ax.axvline(20.0, color="#888888", linestyle=":", lw=0.75)
    ax.set_xlabel(r"Scientific Monoculture Rate")
    ax.set_ylabel("Transformational\nValidity Rate")
    ax.set_xlim(17, 96)
    ax.set_ylim(72, 106)
    ax.set_xticks([20, 40, 60, 80])
    ax.set_xticklabels([r"20\%", r"40\%", r"60\%", r"80\%"])
    ax.set_yticks([75, 85, 95])
    ax.set_yticklabels([r"75\%", r"85\%", r"95\%"])
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2)
    plot_util.resize_figure_with_legend(fig)
    fig.savefig(FIG_DIR / "fig4_interventions_pareto.pdf")
    plt.close(fig)

    # ==================== FIGURE 5: Per-Scenario Lock-In (All 7 Models, Half-width 1:1) ====================
    fig, ax = plt.subplots(figsize=plot_util.FIGURE_SIZE_HALF_IN)
    all_exp3 = [m for m in ALL_MAIN_MODELS if m in summary["interventions"] and {"direct", "axiom_neg", "vs"}.issubset(summary["interventions"][m].keys())]
    if all_exp3:
        sids = sorted({x["scenario_id"] for x in summary["interventions"][all_exp3[0]]["direct"]["per_scenario"]})
        xs_d, ys_a, ys_v = [], [], []
        for sid in sids:
            d_vals = [x["smr"] * 100 for m in all_exp3 for x in summary["interventions"][m]["direct"]["per_scenario"] if x["scenario_id"] == sid and not np.isnan(x["smr"])]
            a_vals = [x["smr"] * 100 for m in all_exp3 for x in summary["interventions"][m]["axiom_neg"]["per_scenario"] if x["scenario_id"] == sid and not np.isnan(x["smr"])]
            v_vals = [x["smr"] * 100 for m in all_exp3 for x in summary["interventions"][m]["vs"]["per_scenario"] if x["scenario_id"] == sid and not np.isnan(x["smr"])]
            if d_vals and a_vals and v_vals:
                xs_d.append(float(np.mean(d_vals)))
                ys_a.append(float(np.mean(a_vals)))
                ys_v.append(float(np.mean(v_vals)))

        ax.plot([18, 104], [18, 104], "--", color="#777777", lw=0.8, label="No change")
        ax.axhline(20.0, color="#999999", linestyle=":", lw=0.75)
        ax.scatter(xs_d, ys_a, s=22, marker="o", color="C2", alpha=0.88, label="Axiom-Neg")
        ax.scatter(xs_d, ys_v, s=24, marker="^", color="C0", alpha=0.90, label="VS")

    ax.set_xlabel(r"Baseline Scientific Monoculture Rate")
    ax.set_ylabel("Method Scientific\nMonoculture Rate")
    ax.set_xlim(22, 105)
    ax.set_ylim(15, 105)
    ax.set_xticks([25, 50, 75, 100])
    ax.set_xticklabels([r"25\%", r"50\%", r"75\%", r"100\%"])
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([r"20\%", r"40\%", r"60\%", r"80\%", r"100\%"])
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, columnspacing=0.6, handlelength=1.0)
    plot_util.resize_figure_with_legend(fig)
    fig.savefig(FIG_DIR / "fig5_attractor_reconvergence.pdf")
    plt.close(fig)

    # ==================== FIGURE 6: Experiment 4 Historical vs. Transferred Controls (Nine-col 1:1) ====================
    contam_models = [m for m in OPEN_MODELS if m in summary["contamination"]]
    if contam_models:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plot_util.FIGURE_SIZE_NINECOL_IN[0], 1.75))
        colors = ["C0", "C2", "C3", "C4", "C5"]
        xs = [0, 1]
        for idx, m in enumerate(contam_models):
            col = colors[idx % len(colors)]
            lbl = SHORT_MODEL.get(m, MODEL_DISPLAY[m])
            o_tvr = summary["contamination"][m]["orig"]["tvr"]["mean"] * 100
            i_tvr = summary["contamination"][m]["iso"]["tvr"]["mean"] * 100
            o_smr = summary["contamination"][m]["orig"]["smr"]["mean"] * 100
            i_smr = summary["contamination"][m]["iso"]["smr"]["mean"] * 100
            ax1.plot(xs, [o_tvr, i_tvr], "-o", color=col, markersize=3.8, lw=1.1, label=lbl)
            ax2.plot(xs, [o_smr, i_smr], "-o", color=col, markersize=3.8, lw=1.1)

        for ax, ylabel in [(ax1, "Transformational\nValidity Rate"), (ax2, "Scientific\nMonoculture Rate")]:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Historical", "Transferred"])
            ax.set_xlim(-0.25, 1.25)
            ax.set_ylabel(ylabel)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
        ax1.set_ylim(76, 103)
        ax1.set_yticks([80, 85, 90, 95, 100])
        ax1.set_yticklabels([r"80\%", r"85\%", r"90\%", r"95\%", r"100\%"])
        ax2.set_ylim(48, 99)
        ax2.set_yticks([50, 65, 80, 95])
        ax2.set_yticklabels([r"50\%", r"65\%", r"80\%", r"95\%"])
        fig.legend(*ax1.get_legend_handles_labels(), loc="outside upper center", ncol=5, columnspacing=0.6, handlelength=1.0)
        plot_util.resize_figure_with_legend(fig)
        fig.savefig(FIG_DIR / "fig6_contamination_controls.pdf")
        plt.close(fig)


def main():
    scenarios, contam_pairs, calibration, gen_records, judge_scores, taxonomies, emb_raw, emb_mech = load_all_data()
    print(f"Loaded {len(gen_records)} generated hypotheses, {len(judge_scores)} judges, {len(taxonomies)} taxonomies")

    cal_report = compute_calibration_report(calibration, judge_scores)
    ij_agree = compute_inter_judge_agreement(judge_scores, set(gen_records.keys()))
    df, pri_tax, sec_tax = aggregate_item_scores(gen_records, judge_scores, taxonomies)
    df.to_csv(OUT_DIR / "all_scored_items.csv", index=False)

    pre_audit = {}
    pre_audit_path = RES_DIR / "pre_repair_truncation_audit.json"
    if pre_audit_path.exists():
        pre_audit = json.loads(pre_audit_path.read_text())

    summary: Dict[str, Any] = {
        "counts": {
            "total_generated": len(gen_records),
            "total_scored": len(df),
            "total_truncated_post_repair": int(df["truncated"].sum()),
            "total_repaired": int(df["repaired"].sum()),
            "judges": sorted(judge_scores.keys()),
            "primary_taxonomy_judge": pri_tax,
            "secondary_taxonomy_judge": sec_tax,
        },
        "pre_repair_truncation_audit": pre_audit,
        "calibration": cal_report,
        "inter_judge_agreement": ij_agree,
        "frontier_direct": {},
        "olmo_lineage": {},
        "interventions": {},
        "contamination": {},
        "domain_breakdown": {},
    }

    # Exp 1: All 7 main models (5 open-weight + 2 closed-weight frontier)
    for m in ALL_MAIN_MODELS:
        sub = df[(df["model"] == m) & (df["mode"] == "direct")]
        if len(sub):
            summary["frontier_direct"][m] = compute_group_metrics(sub, emb_raw, emb_mech)

    avail_open = [m for m in OPEN_MODELS if m in summary["frontier_direct"]]
    avail_all7 = [m for m in ALL_MAIN_MODELS if m in summary["frontier_direct"]]
    summary["hivemind_open5"] = compute_hivemind_analysis(df, avail_open)
    summary["hivemind_all7"] = compute_hivemind_analysis(df, avail_all7)
    summary["hivemind"] = summary["hivemind_all7"]

    # Domain breakdown across all 7 main models
    for dom, grp in df[(df["mode"] == "direct") & (df["model"].isin(avail_all7))].groupby("domain"):
        smrs = []
        for _, g in grp.groupby(["model", "scenario_id"]):
            fws = [x for x in g["fw_primary"].tolist() if x not in BAD_FW]
            if fws:
                smrs.append(hill_effective_number(fws)[1])
        summary["domain_breakdown"][dom] = {
            "surprise": float(grp["surprise"].mean()),
            "coherence": float(grp["coherence"].mean()),
            "realism": float(grp["realism"].mean()),
            "tvr": float(grp["transformational"].mean()),
            "smr_mean": float(np.mean(smrs)) if smrs else float("nan"),
        }

    # Exp 2: OLMo-3 / 3.1 Post-Training Lineage (Base, 7B-Inst, 7B-Think, 32B-Think, 3.1-32B-Think)
    for m in OLMO_BASE + OLMO_INST + OLMO_THINK + OLMO_32B_THINK:
        sub = df[(df["model"] == m) & (df["mode"] == "direct")]
        if len(sub):
            summary["olmo_lineage"][m] = compute_group_metrics(sub, emb_raw, emb_mech)

    # Exp 3: Inference-time interventions across ALL 5 open-weight models + 2 closed-weight models (direct + vs)
    for m in ALL_MAIN_MODELS:
        m_modes = {}
        for mode in INTERV_MODES:
            sub = df[(df["model"] == m) & (df["mode"] == mode)]
            if len(sub):
                m_modes[mode] = compute_group_metrics(sub, emb_raw, emb_mech)
        if len(m_modes) >= 2:
            summary["interventions"][m] = m_modes

    # VS structural diagnostics (1/5 floor, direct modal share, within-call merge rate, length & rating ratio)
    vs_diag = compute_vs_diagnostics(df, list(summary["interventions"].keys()))
    for m, d in vs_diag.items():
        d_dir = df[(df["model"] == m) & (df["mode"] == "direct")]
        d_vs = df[(df["model"] == m) & (df["mode"] == "vs")]
        if len(d_dir) and len(d_vs):
            d["hyp_chars_baseline"] = float(d_dir["hyp_chars"].mean())
            d["hyp_chars_vs"] = float(d_vs["hyp_chars"].mean())
            d["hyp_chars_ratio_vs_to_base"] = float(d_vs["hyp_chars"].mean() / max(1.0, d_dir["hyp_chars"].mean()))
            d["S_base"] = float(d_dir["surprise"].mean()); d["S_vs"] = float(d_vs["surprise"].mean())
            d["C_base"] = float(d_dir["coherence"].mean()); d["C_vs"] = float(d_vs["coherence"].mean())
            d["R_base"] = float(d_dir["realism"].mean()); d["R_vs"] = float(d_vs["realism"].mean())
    summary["vs_diagnostics"] = vs_diag

    # Exact duplicate audit across all (model, mode, scenario_id)
    dup_by_cond = {}
    total_dups = 0
    by_cm_sc: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for rid, g in gen_records.items():
        by_cm_sc[(g["model"], g["mode"], g["scenario_id"])].append((g.get("hypothesis") or "").strip())
    by_cm: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
    for (m, mode, sid), hyps in by_cm_sc.items():
        n_d = len(hyps) - len(set(hyps))
        by_cm[(m, mode)].append((n_d, len(set(hyps))))
        total_dups += n_d
    for (m, mode), pairs in sorted(by_cm.items()):
        tot_d = sum(x[0] for x in pairs)
        if tot_d > 0:
            dup_by_cond[f"{m}__{mode}"] = {"duplicates": tot_d, "min_unique_in_scenario": min(x[1] for x in pairs)}
    summary["duplicate_audit"] = {"total_exact_duplicates": total_dups, "by_condition": dup_by_cond}

    # Per-judge strictness & 3-judge (without Mistral) vs 4-judge TVR sensitivity
    j_sens: Dict[str, Any] = {"per_judge_pass_rate": {}, "olmo_3judge_vs_4judge_tvr": {}}
    live_ids = set(gen_records.keys())
    for j, jmap in sorted(judge_scores.items()):
        all_pass, base_pass = [], []
        for rid, rec in jmap.items():
            if rid not in live_ids:
                continue
            ok = int((rec["surprise"] >= 4) and (rec["coherence"] >= 4) and (rec["realism"] >= 4) and (rec.get("n_contradicted", 0) == 0))
            all_pass.append(ok)
            g = gen_records[rid]
            if g["mode"] == "direct" and g["model"] in ALL_MAIN_MODELS:
                base_pass.append(ok)
        j_sens["per_judge_pass_rate"][j] = {
            "all_items_strict_pass": float(np.mean(all_pass)) if all_pass else 0.0,
            "main7_baseline_strict_pass": float(np.mean(base_pass)) if base_pass else 0.0,
        }
    three_judges = [j for j in sorted(judge_scores.keys()) if j != "mistral-medium-3.5-128b"]
    for m in OLMO_BASE + OLMO_INST + OLMO_THINK + OLMO_32B_THINK:
        sub_ids = [rid for rid, g in gen_records.items() if g["model"] == m and g["mode"] == "direct"]
        if not sub_ids:
            continue
        p3, p4 = [], []
        for rid in sub_ids:
            r3 = [judge_scores[j][rid] for j in three_judges if rid in judge_scores[j]]
            r4 = [judge_scores[j][rid] for j in sorted(judge_scores.keys()) if rid in judge_scores[j]]
            for recs_j, arr_j in ((r3, p3), (r4, p4)):
                if recs_j:
                    S = np.mean([x["surprise"] for x in recs_j])
                    C = np.mean([x["coherence"] for x in recs_j])
                    R = np.mean([x["realism"] for x in recs_j])
                    ct = np.mean([1.0 if x.get("n_contradicted", 0) >= 1 else 0.0 for x in recs_j])
                    arr_j.append(int((S >= 3.5) and (C >= 3.5) and (R >= 3.5) and (ct < 0.5)))
        j_sens["olmo_3judge_vs_4judge_tvr"][m] = {
            "tvr_3judge": float(np.mean(p3)) if p3 else 0.0,
            "tvr_4judge": float(np.mean(p4)) if p4 else 0.0,
        }
    summary["judge_sensitivity"] = j_sens

    # Exp 4: Contamination control (historical orig vs isomorphic transfer across 4 pairs)
    orig_ids = {p["orig"]["id"] for p in contam_pairs}
    iso_ids = {p["iso"]["id"] for p in contam_pairs}
    for m in OPEN_MODELS:
        sub = df[(df["model"] == m) & (df["mode"] == "contamination")]
        if len(sub):
            summary["contamination"][m] = {
                "orig": compute_group_metrics(sub[sub["scenario_id"].isin(orig_ids)], emb_raw, emb_mech),
                "iso": compute_group_metrics(sub[sub["scenario_id"].isin(iso_ids)], emb_raw, emb_mech),
            }

    # Paired Wilcoxon signed-rank tests with Benjamini-Hochberg FDR correction
    sig_tests: Dict[str, Dict[str, Any]] = {}
    exp3_keys = []
    for m, m_modes in summary["interventions"].items():
        if "direct" not in m_modes:
            continue
        d0 = {x["scenario_id"]: x for x in m_modes["direct"]["per_scenario"]}
        for other_mode in ("high_temp", "vs", "icr", "axiom_neg"):
            if other_mode not in m_modes:
                continue
            d1 = {x["scenario_id"]: x for x in m_modes[other_mode]["per_scenario"]}
            sids = sorted(set(d0) & set(d1))
            for k in ("smr", "smr_val", "n_eff", "ted", "tvr"):
                pairs = [(d0[s][k], d1[s][k]) for s in sids if not np.isnan(d0[s][k]) and not np.isnan(d1[s][k])]
                v0 = [x[0] for x in pairs]
                v1 = [x[1] for x in pairs]
                try:
                    stat, p = stats.wilcoxon(v0, v1)
                except ValueError:
                    stat, p = 0.0, 1.0
                tkey = f"{m}_direct_vs_{other_mode}_{k}"
                sig_tests[tkey] = {
                    "family": "exp3_interventions",
                    "n_pairs": len(pairs),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p),
                    "delta": float(np.mean(v1) - np.mean(v0)) if pairs else float("nan"),
                }
                exp3_keys.append(tkey)

    olmo_keys = []
    for m_a, m_b, label in [
        ("olmo3-7b-base", "olmo3-7b-inst-sft", "olmo_7b_base_vs_inst_sft"),
        ("olmo3-32b-base", "olmo3-32b-think-sft", "olmo_32b_base_vs_think_sft"),
        ("olmo3-7b-inst-sft", "olmo3-7b-inst-dpo", "olmo_inst_sft_vs_dpo"),
        ("olmo3-7b-inst-sft", "olmo3-7b-inst-rlvr", "olmo_inst_sft_vs_rlvr"),
        ("olmo3-7b-inst-sft", "olmo3-7b-think-sft", "olmo_inst_sft_vs_think_sft"),
        ("olmo3-7b-inst-rlvr", "olmo3-7b-think-rlvr", "olmo_inst_rlvr_vs_think_rlvr"),
        ("olmo3-7b-think-sft", "olmo3-7b-think-dpo", "olmo_think_sft_vs_dpo"),
        ("olmo3-7b-think-sft", "olmo3-7b-think-rlvr", "olmo_think_sft_vs_rlvr"),
        ("olmo3-32b-think-sft", "olmo3-32b-think-dpo", "olmo32b_think_sft_vs_dpo"),
        ("olmo3-32b-think-sft", "olmo3-32b-think-rlvr", "olmo32b_think_sft_vs_rlvr"),
        ("olmo3-7b-think-rlvr", "olmo3-32b-think-rlvr", "olmo_7b_vs_32b_think_rlvr"),
        ("olmo3-7b-think-sft", "olmo3-32b-think-sft", "olmo_7b_vs_32b_think_sft"),
    ]:
        if m_a in summary["olmo_lineage"] and m_b in summary["olmo_lineage"]:
            d0 = {x["scenario_id"]: x for x in summary["olmo_lineage"][m_a]["per_scenario"]}
            d1 = {x["scenario_id"]: x for x in summary["olmo_lineage"][m_b]["per_scenario"]}
            sids = sorted(set(d0) & set(d1))
            for k in ("tvr", "ccr", "smr", "smr_val", "smr_val_r5", "n_eff", "ted"):
                pairs = [(d0[s][k], d1[s][k]) for s in sids if not np.isnan(d0[s][k]) and not np.isnan(d1[s][k])]
                v0 = [x[0] for x in pairs]
                v1 = [x[1] for x in pairs]
                if len(pairs) < 4:
                    stat, p = 0.0, 1.0
                else:
                    try:
                        stat, p = stats.wilcoxon(v0, v1)
                    except ValueError:
                        stat, p = 0.0, 1.0
                tkey = f"{label}_{k}"
                sig_tests[tkey] = {
                    "family": "exp2_olmo_lineage",
                    "baseline": m_a,
                    "target": m_b,
                    "n_pairs": len(pairs),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p),
                    "delta": float(np.mean(v1) - np.mean(v0)) if pairs else float("nan"),
                }
                olmo_keys.append(tkey)

    # Apply BH-FDR separately within Exp 2 and Exp 3 AND globally across all tests
    all_keys = exp3_keys + olmo_keys
    q_exp3 = bh_fdr([sig_tests[k]["p_value"] for k in exp3_keys])
    q_olmo = bh_fdr([sig_tests[k]["p_value"] for k in olmo_keys])
    q_all = bh_fdr([sig_tests[k]["p_value"] for k in all_keys])
    for k, qv in zip(exp3_keys, q_exp3):
        sig_tests[k]["q_value_family"] = qv
    for k, qv in zip(olmo_keys, q_olmo):
        sig_tests[k]["q_value_family"] = qv
    for k, qv in zip(all_keys, q_all):
        sig_tests[k]["q_value_global"] = qv
    summary["significance_tests"] = sig_tests

    # Cross-taxonomy stability correlation (primary vs secondary taxonomy across all conditions)
    if sec_tax:
        p_smrs, s_smrs, p_neffs, s_neffs, pv_smrs, sv_smrs = [], [], [], [], [], []
        for grp_dict in (summary["frontier_direct"], summary["olmo_lineage"]):
            for d in grp_dict.values():
                p_smrs.append(d["smr"]["mean"]); s_smrs.append(d["smr_sec"]["mean"])
                p_neffs.append(d["n_eff"]["mean"]); s_neffs.append(d["n_eff_sec"]["mean"])
                pv_smrs.append(d["smr_val"]["mean"]); sv_smrs.append(d["smr_val_sec"]["mean"])
        for m in summary["interventions"]:
            for mode, d in summary["interventions"][m].items():
                if mode == "direct":
                    continue
                p_smrs.append(d["smr"]["mean"]); s_smrs.append(d["smr_sec"]["mean"])
                p_neffs.append(d["n_eff"]["mean"]); s_neffs.append(d["n_eff_sec"]["mean"])
                pv_smrs.append(d["smr_val"]["mean"]); sv_smrs.append(d["smr_val_sec"]["mean"])
        summary["cross_taxonomy_stability"] = {
            "smr_spearman": float(stats.spearmanr(p_smrs, s_smrs, nan_policy="omit")[0]),
            "neff_spearman": float(stats.spearmanr(p_neffs, s_neffs, nan_policy="omit")[0]),
            "smr_val_spearman": float(stats.spearmanr(pv_smrs, sv_smrs, nan_policy="omit")[0]),
            "n_conditions": len(p_smrs),
        }

    # Vendi vs N_eff Spearman correlations
    summary["vendi_correlations"] = compute_vendi_correlations(summary)

    (OUT_DIR / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
    if not getattr(args, "no_plots", False):
        make_figures(summary, df)
        print(f"Saved summary_metrics.json and 5 PDF figures to {FIG_DIR}")
    else:
        print(f"Saved summary_metrics.json to {OUT_DIR} (plots skipped)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()
    if args.plots_only:
        summary = json.loads((OUT_DIR / "summary_metrics.json").read_text())
        df = pd.read_csv(OUT_DIR / "all_scored_items.csv")
        make_figures(summary, df)
        print(f"Regenerated 5 PDF figures in {FIG_DIR}")
    else:
        main()
