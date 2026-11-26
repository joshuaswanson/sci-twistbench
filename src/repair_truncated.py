#!/usr/bin/env python3
"""Repair truncated (finish_reason=='length' or empty) or failed-VS-parse records in-place at higher max_tokens."""

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", required=True, help="Comma-separated modes to inspect and repair")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=24576)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=137)
    ap.add_argument("--tag", default="", help="Optional output tag for repaired_<tag>.jsonl")
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

    # Pre-scan to see what actually needs repair
    plan: Dict[str, List[Dict[str, Any]]] = {}
    need_any = False
    for mode in modes:
        fp = res_dir / f"gen_{args.model}_{mode}.jsonl"
        if not fp.exists():
            print(f"[repair] Skip missing {fp}", flush=True)
            continue
        recs = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
        n_bad = sum(1 for r in recs if is_broken(r))
        if mode == "vs":
            by_sc = defaultdict(list)
            for r in recs:
                if not is_broken(r):
                    by_sc[(r["scenario_id"], r.get("call_idx"))].append(r)
            missing_calls = 0
            for s in scenarios:
                for ci in range(6):
                    if len(by_sc[(s["id"], ci)]) < 5:
                        missing_calls += 1
            if n_bad > 0 or missing_calls > 0:
                need_any = True
                plan[mode] = recs
                print(f"[repair] {fp.name}: n={len(recs)} bad={n_bad} missing_vs_calls={missing_calls}", flush=True)
            else:
                print(f"[repair] {fp.name}: CLEAN (n={len(recs)})", flush=True)
        else:
            if n_bad > 0:
                need_any = True
                plan[mode] = recs
                print(f"[repair] {fp.name}: n={len(recs)} bad={n_bad}", flush=True)
            else:
                print(f"[repair] {fp.name}: CLEAN (n={len(recs)})", flush=True)

    if not need_any:
        print(f"[repair] Nothing to repair for {args.model}!", flush=True)
        return

    llm, model_path, cfg = load_llm(
        args.model,
        tp_size=args.tp_size,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len_override=args.max_model_len,
    )
    adapter = ChatAdapter(model_path, cfg["kind"], llm.get_tokenizer(), enable_thinking=True)

    repaired_records: List[Dict[str, Any]] = []

    for mode, recs in plan.items():
        fp = res_dir / f"gen_{args.model}_{mode}.jsonl"
        t0 = time.time()

        if mode in ("direct", "high_temp", "contamination"):
            temp = 1.3 if mode == "high_temp" else 1.0
            bad_idxs = [i for i, r in enumerate(recs) if is_broken(r)]
            prompts = [
                adapter.format_prompt(SYSTEM_SCIENTIST, DIRECT_USER_TEMPLATE.format(dossier=scen_by_id[recs[i]["scenario_id"]]["dossier"]))
                for i in bad_idxs
            ]
            outs = generate_batch(llm, adapter, prompts, temperature=temp, top_p=0.95, max_tokens=args.max_tokens, n=1, seed=args.seed)
            still_bad = 0
            for idx, samples in zip(bad_idxs, outs):
                samp = samples[0]
                recs[idx]["hypothesis"] = samp["text"]
                recs[idx]["thinking"] = samp["thinking"]
                recs[idx]["finish_reason"] = samp["finish_reason"]
                recs[idx]["num_tokens"] = samp["num_tokens"]
                recs[idx]["repaired"] = True
                if is_broken(recs[idx]):
                    still_bad += 1
                repaired_records.append(recs[idx])
            with open(fp, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"[repair][{mode}] Repaired {len(bad_idxs)} items (still_bad={still_bad}) in {time.time()-t0:.1f}s", flush=True)

        elif mode == "axiom_neg":
            bad_idxs = [i for i, r in enumerate(recs) if is_broken(r)]
            prompts = []
            for idx in bad_idxs:
                r = recs[idx]
                s = scen_by_id[r["scenario_id"]]
                user_msg = AXIOM_NEGATE_TEMPLATE.format(
                    dossier=s["dossier"],
                    assumption=r.get("target_assumption", ""),
                    negation_prompt=r.get("negation_prompt", ""),
                )
                prompts.append(adapter.format_prompt(SYSTEM_SCIENTIST, user_msg))
            outs = generate_batch(llm, adapter, prompts, temperature=1.0, top_p=0.95, max_tokens=args.max_tokens, n=1, seed=args.seed)
            still_bad = 0
            for idx, samples in zip(bad_idxs, outs):
                samp = samples[0]
                recs[idx]["hypothesis"] = samp["text"]
                recs[idx]["thinking"] = samp["thinking"]
                recs[idx]["finish_reason"] = samp["finish_reason"]
                recs[idx]["num_tokens"] = samp["num_tokens"]
                recs[idx]["repaired"] = True
                if is_broken(recs[idx]):
                    still_bad += 1
                repaired_records.append(recs[idx])
            with open(fp, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"[repair][axiom_neg] Repaired {len(bad_idxs)} items (still_bad={still_bad}) in {time.time()-t0:.1f}s", flush=True)

        elif mode == "vs":
            good_by_call: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
            for r in recs:
                if not is_broken(r):
                    good_by_call[(r["scenario_id"], r.get("call_idx"))].append(r)

            missing_calls: List[Tuple[Dict[str, Any], int]] = []
            for s in scenarios:
                for ci in range(6):
                    if len(good_by_call[(s["id"], ci)]) < 5:
                        missing_calls.append((s, ci))

            # Retry up to 3 attempts for any JSON parse hiccup
            pending = list(missing_calls)
            for attempt in range(3):
                if not pending:
                    break
                prompts = [adapter.format_prompt(SYSTEM_SCIENTIST, VS_USER_TEMPLATE.format(dossier=s["dossier"])) for s, _ in pending]
                outs = generate_batch(llm, adapter, prompts, temperature=1.0, top_p=0.95, max_tokens=args.max_tokens, n=1, seed=args.seed + attempt * 99)
                next_pending = []
                for (s, ci), samples in zip(pending, outs):
                    samp = samples[0]
                    parsed = extract_json_object(samp["text"])
                    hyps = parsed.get("hypotheses", []) if parsed else []
                    if not isinstance(hyps, list) or len(hyps) < 5:
                        next_pending.append((s, ci))
                        if attempt == 2 and isinstance(hyps, list) and len(hyps) > 0:
                            # Keep partial on final attempt
                            pass
                        else:
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
                        repaired_records.append(rec)
                    good_by_call[(s["id"], ci)] = call_recs
                pending = next_pending

            final_vs: List[Dict[str, Any]] = []
            for s in scenarios:
                sidx = 0
                for ci in range(6):
                    for r in good_by_call[(s["id"], ci)][:5]:
                        old_id = r["id"]
                        r["sample_idx"] = sidx
                        r["id"] = f"{args.model}__vs__{s['id']}__s{sidx:02d}"
                        if r["id"] != old_id:
                            repaired_records.append(r)
                        final_vs.append(r)
                        sidx += 1
            with open(fp, "w") as f:
                for r in final_vs:
                    f.write(json.dumps(r) + "\n")
            print(f"[repair][vs] Rebuilt {fp.name} with {len(final_vs)} valid parsed items (remaining_failed_calls={len(pending)}) in {time.time()-t0:.1f}s", flush=True)

        elif mode == "icr":
            # Identify every (scenario_id, chain_idx) that has at least 1 broken turn
            by_chain: Dict[Tuple[str, int], Dict[int, Dict[str, Any]]] = defaultdict(dict)
            bad_chains: Set[Tuple[str, int]] = set()
            for r in recs:
                key = (r["scenario_id"], int(r["chain_idx"]))
                by_chain[key][int(r["turn_idx"])] = r
                if is_broken(r):
                    bad_chains.add(key)

            sorted_bad = sorted(bad_chains)
            print(f"[repair][icr] Rerunning {len(sorted_bad)}/{len(by_chain)} affected 10-turn chains at max_tokens={args.max_tokens}...", flush=True)
            histories: Dict[Tuple[str, int], List[str]] = {k: [] for k in sorted_bad}
            for turn in range(10):
                turn_prompts = []
                for sid, ci in sorted_bad:
                    s = scen_by_id[sid]
                    hist = histories[(sid, ci)]
                    if not hist:
                        user_msg = DIRECT_USER_TEMPLATE.format(dossier=s["dossier"])
                    else:
                        prior_str = "\n".join(f"{k+1}. {h}" for k, h in enumerate(hist))
                        user_msg = ICR_USER_TEMPLATE.format(dossier=s["dossier"], prior_list=prior_str)
                    turn_prompts.append(adapter.format_prompt(SYSTEM_SCIENTIST, user_msg))

                outs = generate_batch(llm, adapter, turn_prompts, temperature=1.0, top_p=0.95, max_tokens=args.max_tokens, n=1, seed=args.seed + turn * 17)
                for (sid, ci), samples in zip(sorted_bad, outs):
                    samp = samples[0]
                    histories[(sid, ci)].append(summarize_hypothesis_for_icr(samp["text"]))
                    old_r = by_chain[(sid, ci)][turn]
                    old_r["hypothesis"] = samp["text"]
                    old_r["thinking"] = samp["thinking"]
                    old_r["finish_reason"] = samp["finish_reason"]
                    old_r["num_tokens"] = samp["num_tokens"]
                    old_r["repaired"] = True
                    repaired_records.append(old_r)

            still_bad = sum(1 for r in recs if is_broken(r))
            with open(fp, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"[repair][icr] Repaired {len(sorted_bad)} chains (still_bad={still_bad}) in {time.time()-t0:.1f}s", flush=True)

    out_tag = args.tag.strip() if args.tag.strip() else args.model
    rep_path = res_dir / f"repaired_{out_tag}.jsonl"
    # Deduplicate repaired_records by id
    by_id = {r["id"]: r for r in repaired_records}
    with open(rep_path, "w") as f:
        for r in by_id.values():
            f.write(json.dumps(r) + "\n")
    print(f"[repair] Wrote {len(by_id)} repaired records to {rep_path}", flush=True)


if __name__ == "__main__":
    main()
