#!/usr/bin/env python3
"""Eliminate exact-duplicate samples (from shared seeds in repairs and ICR chains) using unique per-sample seeds."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import (
    AXIOM_NEGATE_TEMPLATE,
    DIRECT_USER_TEMPLATE,
    ICR_USER_TEMPLATE,
    SYSTEM_SCIENTIST,
    VS_USER_TEMPLATE,
    extract_json_object,
    summarize_hypothesis_for_icr,
)
from vllm_runner import ChatAdapter, generate_batch, load_llm


def is_broken(rec: Dict[str, Any]) -> bool:
    if rec.get("finish_reason") == "length":
        return True
    hyp = (rec.get("hypothesis") or "").strip()
    if len(hyp) < 40:
        return True
    if rec.get("mode") == "vs" and not rec.get("parse_ok", True):
        return True
    return False


def find_duplicate_indices(recs: List[Dict[str, Any]]) -> List[int]:
    """Return indices of 2nd+ exact-duplicate hypotheses within the same scenario_id (or broken items)."""
    seen: Dict[str, Set[str]] = defaultdict(set)
    bad: List[int] = []
    for i, r in enumerate(recs):
        sid = r["scenario_id"]
        hyp = (r.get("hypothesis") or "").strip()
        if is_broken(r) or hyp in seen[sid]:
            bad.append(i)
        else:
            seen[sid].add(hyp)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", required=True, help="Comma-separated modes to deduplicate/rerun")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=24576)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    res_dir = Path(args.results_dir)
    scenarios = json.loads((data_dir / "scenarios.json").read_text())
    scen_by_id = {s["id"]: s for s in scenarios}
    contam_pairs = json.loads((data_dir / "contamination_pairs.json").read_text())
    for p in contam_pairs:
        scen_by_id[p["orig"]["id"]] = p["orig"]
        scen_by_id[p["iso"]["id"]] = p["iso"]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    llm, model_path, cfg = load_llm(
        args.model,
        tp_size=args.tp_size,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len_override=args.max_model_len,
    )
    adapter = ChatAdapter(model_path, cfg["kind"], llm.get_tokenizer(), enable_thinking=True)

    changed_records: List[Dict[str, Any]] = []

    for mode in modes:
        fp = res_dir / f"gen_{args.model}_{mode}.jsonl"
        t0 = time.time()

        if mode == "icr":
            # Full clean 3-chain x 10-turn regeneration with strictly unique per-(scenario, chain, turn) seeds
            n_chains = 3
            n_turns = 10
            histories: Dict[Tuple[int, int], List[str]] = {
                (si, ci): [] for si in range(len(scenarios)) for ci in range(n_chains)
            }
            new_recs: List[Dict[str, Any]] = []
            for turn in range(n_turns):
                turn_prompts = []
                turn_keys = []
                turn_seeds = []
                for si, s in enumerate(scenarios):
                    for ci in range(n_chains):
                        hist = histories[(si, ci)]
                        if not hist:
                            user_msg = DIRECT_USER_TEMPLATE.format(dossier=s["dossier"])
                        else:
                            prior_str = "\n".join(f"{k+1}. {h}" for k, h in enumerate(hist))
                            user_msg = ICR_USER_TEMPLATE.format(dossier=s["dossier"], prior_list=prior_str)
                        turn_prompts.append(adapter.format_prompt(SYSTEM_SCIENTIST, user_msg))
                        turn_keys.append((si, ci))
                        turn_seeds.append(100003 + si * 4001 + ci * 313 + turn * 17)

                outs = generate_batch(
                    llm, adapter, turn_prompts,
                    temperature=1.0, top_p=0.95, max_tokens=args.max_tokens, n=1,
                    seeds=turn_seeds,
                )
                for (si, ci), samples in zip(turn_keys, outs):
                    s = scenarios[si]
                    samp = samples[0]
                    histories[(si, ci)].append(summarize_hypothesis_for_icr(samp["text"]))
                    sample_idx = ci * n_turns + turn
                    rec = {
                        "id": f"{args.model}__icr__{s['id']}__s{sample_idx:02d}",
                        "model": args.model,
                        "family": cfg["family"],
                        "mode": "icr",
                        "scenario_id": s["id"],
                        "domain": s["domain"],
                        "sample_idx": sample_idx,
                        "chain_idx": ci,
                        "turn_idx": turn,
                        "hypothesis": samp["text"],
                        "thinking": samp["thinking"],
                        "finish_reason": samp["finish_reason"],
                        "num_tokens": samp["num_tokens"],
                        "repaired": True,
                    }
                    new_recs.append(rec)
                    changed_records.append(rec)

            # Verify zero exact duplicates
            by_sc = defaultdict(list)
            for r in new_recs:
                by_sc[r["scenario_id"]].append((r["hypothesis"] or "").strip())
            n_dups = sum(len(v) - len(set(v)) for v in by_sc.values())
            n_trunc = sum(1 for r in new_recs if r["finish_reason"] == "length")
            with open(fp, "w") as f:
                for r in new_recs:
                    f.write(json.dumps(r) + "\n")
            print(f"[dedup][icr] Rebuilt {fp.name} ({len(new_recs)} items, dups={n_dups}, trunc={n_trunc}) in {time.time()-t0:.1f}s", flush=True)

        elif mode in ("direct", "high_temp", "contamination", "axiom_neg"):
            recs = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
            temp = 1.3 if mode == "high_temp" else 1.0
            bad_idxs = find_duplicate_indices(recs)
            if not bad_idxs:
                print(f"[dedup][{mode}] {fp.name}: CLEAN (0 dups/broken)", flush=True)
                continue

            # Load stage-1 assumptions if axiom_neg
            assump_map = {}
            if mode == "axiom_neg":
                ap_file = fp.with_suffix(".assumptions.json")
                if ap_file.exists():
                    assump_map = json.loads(ap_file.read_text())

            prompts = []
            seeds = []
            for k, idx in enumerate(bad_idxs):
                r = recs[idx]
                s = scen_by_id[r["scenario_id"]]
                if mode == "axiom_neg":
                    ax_idx = int(r.get("axiom_idx", 0))
                    ax_list = assump_map.get(r["scenario_id"], [])
                    ax = ax_list[ax_idx] if ax_idx < len(ax_list) else {}
                    user_msg = AXIOM_NEGATE_TEMPLATE.format(
                        dossier=s["dossier"],
                        assumption=r.get("targeted_assumption") or ax.get("assumption", ""),
                        negation_prompt=r.get("negation_prompt") or ax.get("negation_prompt", ""),
                    )
                else:
                    user_msg = DIRECT_USER_TEMPLATE.format(dossier=s["dossier"])
                prompts.append(adapter.format_prompt(SYSTEM_SCIENTIST, user_msg))
                seeds.append(200009 + idx * 1307 + k * 53)

            outs = generate_batch(
                llm, adapter, prompts,
                temperature=temp, top_p=0.95, max_tokens=args.max_tokens, n=1,
                seeds=seeds,
            )
            for idx, samples in zip(bad_idxs, outs):
                samp = samples[0]
                recs[idx]["hypothesis"] = samp["text"]
                recs[idx]["thinking"] = samp["thinking"]
                recs[idx]["finish_reason"] = samp["finish_reason"]
                recs[idx]["num_tokens"] = samp["num_tokens"]
                recs[idx]["repaired"] = True
                changed_records.append(recs[idx])

            by_sc = defaultdict(list)
            for r in recs:
                by_sc[r["scenario_id"]].append((r.get("hypothesis") or "").strip())
            n_dups = sum(len(v) - len(set(v)) for v in by_sc.values())
            n_trunc = sum(1 for r in recs if r.get("finish_reason") == "length")
            with open(fp, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"[dedup][{mode}] Regenerated {len(bad_idxs)} items in {fp.name} (rem_dups={n_dups}, rem_trunc={n_trunc}) in {time.time()-t0:.1f}s", flush=True)

        elif mode == "vs":
            recs = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
            by_call: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
            for r in recs:
                if not is_broken(r):
                    by_call[(r["scenario_id"], int(r.get("call_idx", -1)))].append(r)

            # Keep unique valid 5-item calls; regenerate any duplicate call or incomplete call
            good_by_call: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
            missing_calls: List[Tuple[Dict[str, Any], int]] = []
            for s in scenarios:
                seen_sig: Set[str] = set()
                for ci in range(6):
                    items = by_call[(s["id"], ci)]
                    sig = "||".join((x.get("hypothesis") or "").strip() for x in items[:5])
                    if len(items) == 5 and sig not in seen_sig:
                        seen_sig.add(sig)
                        good_by_call[(s["id"], ci)] = items
                    else:
                        missing_calls.append((s, ci))

            if not missing_calls:
                print(f"[dedup][vs] {fp.name}: CLEAN (0 duplicate/missing calls)", flush=True)
                continue

            pending = list(missing_calls)
            for attempt in range(3):
                if not pending:
                    break
                prompts = [adapter.format_prompt(SYSTEM_SCIENTIST, VS_USER_TEMPLATE.format(dossier=s["dossier"])) for s, _ in pending]
                seeds = [300017 + attempt * 50021 + k * 1709 + ci * 97 for k, (s, ci) in enumerate(pending)]
                outs = generate_batch(
                    llm, adapter, prompts,
                    temperature=1.0, top_p=0.95, max_tokens=args.max_tokens, n=1,
                    seeds=seeds,
                )
                next_pending = []
                for (s, ci), samples in zip(pending, outs):
                    samp = samples[0]
                    parsed = extract_json_object(samp["text"])
                    hyps = parsed.get("hypotheses", []) if parsed else []
                    if not isinstance(hyps, list) or len(hyps) < 5:
                        next_pending.append((s, ci))
                        continue
                    call_recs = []
                    for hi, h in enumerate(hyps[:5]):
                        sidx = ci * 5 + hi
                        formatted = (
                            f"### Assumption Challenged\n{h.get('assumption_challenged', '')}\n\n"
                            f"### Core Hypothesis & Mechanism\n{h.get('mechanism', '')}\n\n"
                            f"### Reconciliation with Reported Data\n{h.get('reconciliation', '')}\n\n"
                            f"### Falsifiable Prediction\n{h.get('falsifiable_prediction', '')}"
                        )
                        rec = {
                            "id": f"{args.model}__vs__{s['id']}__s{sidx:02d}",
                            "model": args.model,
                            "family": cfg["family"],
                            "mode": "vs",
                            "scenario_id": s["id"],
                            "domain": s["domain"],
                            "sample_idx": sidx,
                            "call_idx": ci,
                            "vs_prob": h.get("probability"),
                            "hypothesis": formatted,
                            "thinking": samp["thinking"] if hi == 0 else "",
                            "finish_reason": samp["finish_reason"],
                            "num_tokens": samp["num_tokens"],
                            "parse_ok": True,
                            "repaired": True,
                        }
                        call_recs.append(rec)
                        changed_records.append(rec)
                    good_by_call[(s["id"], ci)] = call_recs
                pending = next_pending

            final_vs: List[Dict[str, Any]] = []
            for s in scenarios:
                sidx = 0
                for ci in range(6):
                    for r in good_by_call.get((s["id"], ci), [])[:5]:
                        old_id = r["id"]
                        r["sample_idx"] = sidx
                        r["id"] = f"{args.model}__vs__{s['id']}__s{sidx:02d}"
                        if r["id"] != old_id:
                            changed_records.append(r)
                        final_vs.append(r)
                        sidx += 1
            by_sc = defaultdict(list)
            for r in final_vs:
                by_sc[r["scenario_id"]].append((r.get("hypothesis") or "").strip())
            n_dups = sum(len(v) - len(set(v)) for v in by_sc.values())
            with open(fp, "w") as f:
                for r in final_vs:
                    f.write(json.dumps(r) + "\n")
            print(f"[dedup][vs] Rebuilt {fp.name} ({len(final_vs)} items, rem_failed_calls={len(pending)}, rem_dups={n_dups}) in {time.time()-t0:.1f}s", flush=True)

    out_rep = res_dir / f"dedup_{args.tag}.jsonl"
    by_id = {r["id"]: r for r in changed_records}
    with open(out_rep, "w") as f:
        for r in by_id.values():
            f.write(json.dumps(r) + "\n")
    print(f"[dedup] Wrote {len(by_id)} unique regenerated records to {out_rep}", flush=True)


if __name__ == "__main__":
    main()
