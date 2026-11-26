#!/usr/bin/env python3
"""Concurrent OpenRouter runner for closed-weight frontier models on Sci-TwistBench.

Outputs JSONL files matching src/vllm_runner.py schema so outputs drop straight into
the B200 3-judge ensemble, blind taxonomy clustering, embeddings, and analyze.py.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import (
    AXIOM_EXTRACT_TEMPLATE,
    AXIOM_NEGATE_TEMPLATE,
    DIRECT_USER_TEMPLATE,
    ICR_USER_TEMPLATE,
    SYSTEM_SCIENTIST,
    VS_USER_TEMPLATE,
    extract_json_object,
    summarize_hypothesis_for_icr,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_KEY_FILE = Path.home() / ".config" / "covert-channels" / "openrouter.key"
DOTENV_FILE = Path(__file__).resolve().parent.parent / ".env"

CLOSED_MODELS = {
    "gpt-6.1-terra": {
        "or_id": "gpt-6.1-terra",
        "provider": "openai",
        "family": "OpenAI",
        "supports_temp": False,
        "reasoning_effort": "low",
    },
    "gpt-6.2-sol": {
        "or_id": "gpt-6.2-sol",
        "provider": "openai",
        "family": "OpenAI",
        "supports_temp": False,
        "reasoning_effort": "low",
    },
    "claude-sonnet-5.5": {
        "or_id": "anthropic/claude-sonnet-5.5",
        "provider": "openrouter",
        "family": "Anthropic",
        "supports_temp": False,
        "reasoning": {"effort": "low"},
    },
    "gemini-4-carbon": {
        "or_id": "google/gemini-4-carbon",
        "provider": "openrouter",
        "family": "Google",
        "supports_temp": True,
        "reasoning": {"effort": "low"},
    },
    "gemini-4-neon": {
        "or_id": "google/gemini-4-neon",
        "provider": "openrouter",
        "family": "Google",
        "supports_temp": True,
        "reasoning": {"effort": "low"},
    },
}


def load_dotenv_key(name: str) -> str:
    if DOTENV_FILE.exists():
        for line in DOTENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def load_api_key(provider: str = "openrouter") -> str:
    if provider == "openai":
        k = os.environ.get("OPENAI_API_KEY", "").strip() or load_dotenv_key("OPENAI_API_KEY")
        if k:
            return k
        raise RuntimeError("No OPENAI_API_KEY found in env or .env")
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or load_dotenv_key("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    if DEFAULT_KEY_FILE.exists():
        return DEFAULT_KEY_FILE.read_text().strip()
    raise RuntimeError("No OPENROUTER_API_KEY found in env or ~/.config/covert-channels/openrouter.key")


def call_openrouter(
    api_key: str,
    model_cfg: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    max_retries: int = 6,
) -> Dict[str, Any]:
    is_openai = model_cfg.get("provider") == "openai"
    url = OPENAI_URL if is_openai else OPENROUTER_URL
    payload: Dict[str, Any] = {
        "model": model_cfg["or_id"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if is_openai:
        payload["max_completion_tokens"] = max_tokens
        if model_cfg.get("reasoning_effort"):
            payload["reasoning_effort"] = model_cfg["reasoning_effort"]
    else:
        payload["max_tokens"] = max_tokens
        if model_cfg.get("supports_temp"):
            payload["temperature"] = 1.0
            payload["top_p"] = 0.95
        if model_cfg.get("reasoning"):
            payload["reasoning"] = model_cfg["reasoning"]

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if not is_openai:
        headers["HTTP-Referer"] = "https://sci-twistbench.local"
        headers["X-Title"] = "Sci-TwistBench"

    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if "error" in data:
                raise RuntimeError(f"API error: {data['error']}")
            choice = data["choices"][0]
            msg = choice.get("message", {})
            text = (msg.get("content") or "").strip()
            thinking = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
            usage = data.get("usage", {})
            return {
                "text": text,
                "thinking": thinking,
                "finish_reason": choice.get("finish_reason", "stop"),
                "num_tokens": int(usage.get("completion_tokens", 0)),
                "cost": float(usage.get("cost", 0.0) or 0.0),
            }
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")[:300]
            last_err = f"HTTP {e.code}: {err_body}"
            if e.code == 400 and ("reasoning" in payload or "reasoning_effort" in payload) and "reason" in err_body.lower():
                payload.pop("reasoning", None)
                payload.pop("reasoning_effort", None)
                body = json.dumps(payload).encode("utf-8")
                continue
            sleep_s = min(30.0, 1.5 * (2 ** attempt))
            time.sleep(sleep_s)
        except Exception as e:
            last_err = str(e)
            sleep_s = min(30.0, 1.5 * (2 ** attempt))
            time.sleep(sleep_s)

    return {
        "text": "",
        "thinking": "",
        "finish_reason": f"error: {last_err}",
        "num_tokens": 0,
        "cost": 0.0,
    }


def run_model_mode(
    api_key: str,
    model_name: str,
    mode: str,
    scenarios: List[Dict[str, Any]],
    out_dir: Path,
    n_samples: int = 30,
    workers: int = 24,
) -> Path:
    cfg = CLOSED_MODELS[model_name]
    out_path = out_dir / f"gen_{model_name}_{mode}.jsonl"

    # Load already-completed IDs for safe resume
    existing: Dict[str, Dict[str, Any]] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("hypothesis"):
                    existing[r["id"]] = r

    t0 = time.time()
    total_cost = 0.0

    if mode == "direct":
        tasks: List[Tuple[Dict[str, Any], int, str]] = []
        for s in scenarios:
            for si in range(n_samples):
                rid = f"{model_name}__direct__{s['id']}__s{si:02d}"
                if rid not in existing:
                    tasks.append((s, si, rid))
        print(f"[{model_name}][direct] {len(existing)} cached, {len(tasks)} to run (workers={workers})", flush=True)

        def _do_direct(item: Tuple[Dict[str, Any], int, str]) -> Dict[str, Any]:
            s, si, rid = item
            prompt = DIRECT_USER_TEMPLATE.format(dossier=s["dossier"])
            res = call_openrouter(api_key, cfg, SYSTEM_SCIENTIST, prompt, max_tokens=3072)
            return {
                "id": rid,
                "model": model_name,
                "family": cfg["family"],
                "mode": "direct",
                "scenario_id": s["id"],
                "domain": s["domain"],
                "sample_idx": si,
                "temperature": 1.0,
                "hypothesis": res["text"],
                "thinking": res["thinking"],
                "finish_reason": res["finish_reason"],
                "num_tokens": res["num_tokens"],
                "cost": res["cost"],
            }

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_do_direct, t) for t in tasks]
            for fut in as_completed(futs):
                rec = fut.result()
                existing[rec["id"]] = rec
                total_cost += rec.pop("cost", 0.0)
                done += 1
                if done % 60 == 0 or done == len(tasks):
                    print(f"[{model_name}][direct] {done}/{len(tasks)} done ({time.time()-t0:.1f}s, ~${total_cost:.2f})", flush=True)

        # Write deterministic order
        ordered = []
        for s in scenarios:
            for si in range(n_samples):
                rid = f"{model_name}__direct__{s['id']}__s{si:02d}"
                if rid in existing:
                    ordered.append(existing[rid])
        with open(out_path, "w") as f:
            for r in ordered:
                f.write(json.dumps(r) + "\n")

    elif mode == "vs":
        n_calls = max(1, (n_samples + 4) // 5)
        # Check which (scenario, call_idx) pairs still need running
        have_calls = set()
        for r in existing.values():
            if r.get("parse_ok"):
                have_calls.add((r["scenario_id"], r.get("call_idx")))
        tasks_vs: List[Tuple[Dict[str, Any], int]] = []
        for s in scenarios:
            for ci in range(n_calls):
                if (s["id"], ci) not in have_calls:
                    tasks_vs.append((s, ci))
        print(f"[{model_name}][vs] {len(have_calls)} calls cached, {len(tasks_vs)} calls to run (workers={workers})", flush=True)

        def _do_vs(item: Tuple[Dict[str, Any], int]) -> Tuple[List[Dict[str, Any]], float]:
            s, ci = item
            prompt = VS_USER_TEMPLATE.format(dossier=s["dossier"])
            res = call_openrouter(api_key, cfg, SYSTEM_SCIENTIST, prompt, max_tokens=5120)
            parsed = extract_json_object(res["text"])
            hyps = parsed.get("hypotheses", []) if parsed else []
            out_recs = []
            base_idx = ci * 5
            if not isinstance(hyps, list) or not hyps:
                out_recs.append(
                    {
                        "id": f"{model_name}__vs__{s['id']}__s{base_idx:02d}",
                        "model": model_name,
                        "family": cfg["family"],
                        "mode": "vs",
                        "scenario_id": s["id"],
                        "domain": s["domain"],
                        "sample_idx": base_idx,
                        "call_idx": ci,
                        "vs_prob": None,
                        "hypothesis": res["text"],
                        "thinking": res["thinking"],
                        "finish_reason": res["finish_reason"],
                        "num_tokens": res["num_tokens"],
                        "parse_ok": False,
                    }
                )
            else:
                for hi, h in enumerate(hyps[:5]):
                    sidx = base_idx + hi
                    formatted = (
                        f"### Assumption Challenged\n{h.get('assumption_challenged', '')}\n\n"
                        f"### Core Hypothesis & Mechanism\n{h.get('mechanism', '')}\n\n"
                        f"### Reconciliation with Reported Data\n{h.get('reconciliation', '')}\n\n"
                        f"### Falsifiable Prediction\n{h.get('falsifiable_prediction', '')}"
                    )
                    out_recs.append(
                        {
                            "id": f"{model_name}__vs__{s['id']}__s{sidx:02d}",
                            "model": model_name,
                            "family": cfg["family"],
                            "mode": "vs",
                            "scenario_id": s["id"],
                            "domain": s["domain"],
                            "sample_idx": sidx,
                            "call_idx": ci,
                            "vs_prob": h.get("probability"),
                            "hypothesis": formatted,
                            "thinking": res["thinking"] if hi == 0 else "",
                            "finish_reason": res["finish_reason"],
                            "num_tokens": res["num_tokens"],
                            "parse_ok": True,
                        }
                    )
            return out_recs, res["cost"]

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_do_vs, t) for t in tasks_vs]
            for fut in as_completed(futs):
                recs, cost = fut.result()
                total_cost += cost
                for r in recs:
                    existing[r["id"]] = r
                done += 1
                if done % 18 == 0 or done == len(tasks_vs):
                    print(f"[{model_name}][vs] {done}/{len(tasks_vs)} calls done ({time.time()-t0:.1f}s, ~${total_cost:.2f})", flush=True)

        ordered = sorted(existing.values(), key=lambda r: (r["scenario_id"], r["sample_idx"]))
        with open(out_path, "w") as f:
            for r in ordered:
                f.write(json.dumps(r) + "\n")

    elif mode == "icr":
        n_chains = 3
        n_turns = max(1, n_samples // n_chains)
        chain_tasks: List[Tuple[Dict[str, Any], int]] = []
        for s in scenarios:
            for ci in range(n_chains):
                needed = any(
                    f"{model_name}__icr__{s['id']}__s{ci * n_turns + t:02d}" not in existing
                    for t in range(n_turns)
                )
                if needed:
                    chain_tasks.append((s, ci))
        print(
            f"[{model_name}][icr] {len(existing)} records cached, {len(chain_tasks)} chains to run (workers={workers})",
            flush=True,
        )

        def _do_chain(item: Tuple[Dict[str, Any], int]) -> Tuple[List[Dict[str, Any]], float]:
            s, ci = item
            hist: List[str] = []
            recs: List[Dict[str, Any]] = []
            cost_sum = 0.0
            for turn in range(n_turns):
                sample_idx = ci * n_turns + turn
                rid = f"{model_name}__icr__{s['id']}__s{sample_idx:02d}"
                if rid in existing:
                    rec = existing[rid]
                    hist.append(summarize_hypothesis_for_icr(rec.get("hypothesis", "")))
                    recs.append(rec)
                    continue
                if not hist:
                    user_msg = DIRECT_USER_TEMPLATE.format(dossier=s["dossier"])
                else:
                    prior_str = "\n".join(f"{k+1}. {h}" for k, h in enumerate(hist))
                    user_msg = ICR_USER_TEMPLATE.format(dossier=s["dossier"], prior_list=prior_str)
                res = call_openrouter(api_key, cfg, SYSTEM_SCIENTIST, user_msg, max_tokens=3072)
                cost_sum += res["cost"]
                hist.append(summarize_hypothesis_for_icr(res["text"]))
                recs.append(
                    {
                        "id": rid,
                        "model": model_name,
                        "family": cfg["family"],
                        "mode": "icr",
                        "scenario_id": s["id"],
                        "domain": s["domain"],
                        "sample_idx": sample_idx,
                        "chain_idx": ci,
                        "turn_idx": turn,
                        "hypothesis": res["text"],
                        "thinking": res["thinking"],
                        "finish_reason": res["finish_reason"],
                        "num_tokens": res["num_tokens"],
                    }
                )
            return recs, cost_sum

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_do_chain, t) for t in chain_tasks]
            for fut in as_completed(futs):
                recs, cost = fut.result()
                total_cost += cost
                for r in recs:
                    existing[r["id"]] = r
                done += 1
                if done % 6 == 0 or done == len(chain_tasks):
                    print(
                        f"[{model_name}][icr] {done}/{len(chain_tasks)} chains done ({time.time()-t0:.1f}s, ~${total_cost:.2f})",
                        flush=True,
                    )

        ordered = sorted(existing.values(), key=lambda r: (r["scenario_id"], r["sample_idx"]))
        with open(out_path, "w") as f:
            for r in ordered:
                f.write(json.dumps(r) + "\n")

    elif mode == "axiom_neg":
        assump_path = out_path.with_suffix(".assumptions.json")
        fallback_assumps = [
            {"axis": "Measurement/Construct", "assumption": "The reported outcome metric directly reflects the underlying causal quantity without observational or protocol distortion.", "negation_prompt": "Assume the observed anomaly stems from a hidden shift in what the measurement or protocol captures."},
            {"axis": "Causal Direction/Feedback", "assumption": "Causal influence flows unidirectionally from the intervention to the outcome without adaptive system feedback.", "negation_prompt": "Assume the system adapts or feeds back nonlinearly against the intervention, inverting the expected sign."},
            {"axis": "Spatiotemporal/Kinetic Scale", "assumption": "Steady-state or aggregate measurements capture the active dynamics regardless of timing, phase, or local transients.", "negation_prompt": "Assume timing, phase alignment, or transient kinetics govern the effect despite identical averages."},
            {"axis": "Hidden Boundary Coupling", "assumption": "The studied subsystem is effectively isolated from its surrounding environment, substrate, or institutional context.", "negation_prompt": "Assume an unmodeled physical, biological, or institutional boundary coupling interacts selectively with the treatment."},
            {"axis": "Latent Heterogeneity/Selection", "assumption": "The observed group reflects a homogeneous population without threshold switching or selection-induced stratification.", "negation_prompt": "Assume a hidden threshold regime switch or selection/composition shift separates subpopulations."},
            {"axis": "Secondary/Indirect Pathway", "assumption": "The primary target pathway dominates the outcome without activating competing or collateral pathways.", "negation_prompt": "Assume suppressing or altering the primary target triggers a dominant collateral or indirect pathway."},
        ]
        if assump_path.exists():
            extracted_assumptions: Dict[str, List[Dict[str, str]]] = json.loads(assump_path.read_text())
        else:
            extracted_assumptions = {}

        missing_s1 = [s for s in scenarios if s["id"] not in extracted_assumptions]
        if missing_s1:
            print(f"[{model_name}][axiom_neg] Stage 1: extracting assumptions for {len(missing_s1)} scenarios...", flush=True)

            def _do_extract(s: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]], float]:
                prompt = AXIOM_EXTRACT_TEMPLATE.format(dossier=s["dossier"])
                res = call_openrouter(api_key, cfg, SYSTEM_SCIENTIST, prompt, max_tokens=3072)
                parsed = extract_json_object(res["text"])
                assumps = parsed.get("assumptions", []) if parsed else []
                if not isinstance(assumps, list) or len(assumps) < 3:
                    assumps = fallback_assumps
                return s["id"], assumps[:6], res["cost"]

            with ThreadPoolExecutor(max_workers=min(workers, len(missing_s1))) as ex:
                for sid, assumps, cost in ex.map(_do_extract, missing_s1):
                    extracted_assumptions[sid] = assumps
                    total_cost += cost
            assump_path.write_text(json.dumps(extracted_assumptions, indent=2))
            print(f"[{model_name}][axiom_neg] Stage 1 done ({time.time()-t0:.1f}s)", flush=True)

        tasks_ax: List[Tuple[Dict[str, Any], int, int, Dict[str, str], str]] = []
        for s in scenarios:
            assumps = extracted_assumptions[s["id"]]
            per_ax = max(1, n_samples // len(assumps))
            rem = n_samples - per_ax * len(assumps)
            idx = 0
            for ai, ax in enumerate(assumps):
                n_here = per_ax + (1 if ai < rem else 0)
                for _ in range(n_here):
                    rid = f"{model_name}__axiom_neg__{s['id']}__s{idx:02d}"
                    if rid not in existing:
                        tasks_ax.append((s, idx, ai, ax, rid))
                    idx += 1
        print(
            f"[{model_name}][axiom_neg] Stage 2: {len(existing)} cached, {len(tasks_ax)} to run (workers={workers})",
            flush=True,
        )

        def _do_axiom(item: Tuple[Dict[str, Any], int, int, Dict[str, str], str]) -> Tuple[Dict[str, Any], float]:
            s, idx, ai, ax, rid = item
            user_msg = AXIOM_NEGATE_TEMPLATE.format(
                dossier=s["dossier"],
                assumption=ax.get("assumption", ""),
                negation_prompt=ax.get("negation_prompt", ""),
            )
            res = call_openrouter(api_key, cfg, SYSTEM_SCIENTIST, user_msg, max_tokens=3072)
            rec = {
                "id": rid,
                "model": model_name,
                "family": cfg["family"],
                "mode": "axiom_neg",
                "scenario_id": s["id"],
                "domain": s["domain"],
                "sample_idx": idx,
                "axiom_idx": ai,
                "axiom_axis": ax.get("axis", ""),
                "targeted_assumption": ax.get("assumption", ""),
                "hypothesis": res["text"],
                "thinking": res["thinking"],
                "finish_reason": res["finish_reason"],
                "num_tokens": res["num_tokens"],
            }
            return rec, res["cost"]

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_do_axiom, t) for t in tasks_ax]
            for fut in as_completed(futs):
                rec, cost = fut.result()
                total_cost += cost
                existing[rec["id"]] = rec
                done += 1
                if done % 60 == 0 or done == len(tasks_ax):
                    print(
                        f"[{model_name}][axiom_neg] {done}/{len(tasks_ax)} done ({time.time()-t0:.1f}s, ~${total_cost:.2f})",
                        flush=True,
                    )

        ordered = sorted(existing.values(), key=lambda r: (r["scenario_id"], r["sample_idx"]))
        with open(out_path, "w") as f:
            for r in ordered:
                f.write(json.dumps(r) + "\n")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"[{model_name}][{mode}] Saved {len(existing)} records to {out_path} in {time.time()-t0:.1f}s", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-6.2-sol,claude-sonnet-5.5,gemini-4-neon")
    ap.add_argument("--modes", default="direct,vs")
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    scenarios = json.loads((root / "data" / "scenarios.json").read_text())
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    if args.smoke:
        s0 = scenarios[0]
        prompt = DIRECT_USER_TEMPLATE.format(dossier=s0["dossier"])
        for m in models:
            api_key = load_api_key(CLOSED_MODELS[m].get("provider", "openrouter"))
            res = call_openrouter(api_key, CLOSED_MODELS[m], SYSTEM_SCIENTIST, prompt, max_tokens=2048)
            print(f"[smoke] {m:20s} ok={bool(res['text'])} tokens={res['num_tokens']} finish={res['finish_reason']} head={repr(res['text'][:80])}")
        return

    for m in models:
        api_key = load_api_key(CLOSED_MODELS[m].get("provider", "openrouter"))
        for mode in modes:
            run_model_mode(api_key, m, mode, scenarios, out_dir, n_samples=args.n_samples, workers=args.workers)


if __name__ == "__main__":
    main()
