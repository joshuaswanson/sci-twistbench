#!/usr/bin/env python3
"""Unified offline vLLM runner for Sci-TwistBench generation, interventions, judging, and embeddings.

Runs cleanly inside /fast/jswanson/vllm_qwen38_flash_next.sif on MPI-IS HTCondor GPU nodes.
Handles model-specific chat templates and thinking-channel parsing for:
  - Qwen3.8-27B
  - gemma-4-31B-it
  - DeepSeek-V4-Flash-0731 (official encoding_dsv4.py)
  - Mistral-Medium-3.5-128B
  - gpt-oss-120b (Harmony analysis/final channel parsing)
  - OLMo-3 Instruct & Think lineages
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompts import (
    AXIOM_EXTRACT_TEMPLATE,
    AXIOM_NEGATE_TEMPLATE,
    DIRECT_USER_TEMPLATE,
    ICR_USER_TEMPLATE,
    JUDGE_SYSTEM,
    JUDGE_USER_TEMPLATE,
    SYSTEM_SCIENTIST,
    TAXONOMY_ASSIGN_TEMPLATE,
    TAXONOMY_BUILD_TEMPLATE,
    VS_USER_TEMPLATE,
    extract_json_object,
    summarize_hypothesis_for_icr,
)


MODEL_REGISTRY = {
    # Frontier 5-family suite (cached on /fast/jswanson/models/)
    "qwen3.8-27b": {
        "path": "/fast/jswanson/models/Qwen3.8-27B",
        "family": "Qwen",
        "kind": "qwen38",
        "max_model_len": 16384,
        "limit_mm": True,
    },
    "gemma-4-31b-it": {
        "path": "/fast/jswanson/models/gemma-4-31B-it",
        "family": "Gemma",
        "kind": "gemma4",
        "max_model_len": 16384,
        "limit_mm": True,
    },
    "deepseek-v4-flash": {
        "path": "/fast/jswanson/models/DeepSeek-V4-Flash-0731",
        "family": "DeepSeek",
        "kind": "deepseek_v4",
        "max_model_len": 16384,
        "limit_mm": False,
    },
    "mistral-medium-3.5-128b": {
        "path": "/fast/jswanson/models/Mistral-Medium-3.5-128B",
        "family": "Mistral",
        "kind": "mistral3",
        "max_model_len": 16384,
        "limit_mm": True,
    },
    "gpt-oss-120b": {
        "path": "/fast/jswanson/models/gpt-oss-120b",
        "family": "OpenAI-OSS",
        "kind": "gpt_oss",
        "max_model_len": 16384,
        "limit_mm": False,
    },
    # OLMo-3 Post-Training Trajectory (HF cache snapshots)
    "olmo3-7b-base": {
        "hf_dir": "/fast/jswanson/hf_home/hub/models--allenai--Olmo-3-1025-7B",
        "family": "OLMo3-7B-Base",
        "kind": "olmo_base",
        "max_model_len": 8192,
        "limit_mm": False,
    },
    "olmo3-32b-base": {
        "hf_dir": "/fast/jswanson/hf_home/hub/models--allenai--Olmo-3-1125-32B",
        "family": "OLMo3-32B-Base",
        "kind": "olmo_base",
        "max_model_len": 8192,
        "limit_mm": False,
    },
    "olmo3-7b-inst-sft": {
        "hf_dir": "/fast/jswanson/hf_home/hub/models--allenai--Olmo-3-7B-Instruct-SFT",
        "family": "OLMo3-7B-Instruct",
        "kind": "olmo_instruct",
        "max_model_len": 8192,
        "limit_mm": False,
    },
    "olmo3-7b-inst-dpo": {
        "hf_dir": "/fast/jswanson/hf_home/hub/models--allenai--Olmo-3-7B-Instruct-DPO",
        "family": "OLMo3-7B-Instruct",
        "kind": "olmo_instruct",
        "max_model_len": 8192,
        "limit_mm": False,
    },
    "olmo3-7b-inst-rlvr": {
        "hf_dir": "/fast/jswanson/hf_home/hub/models--allenai--Olmo-3-7B-Instruct",
        "family": "OLMo3-7B-Instruct",
        "kind": "olmo_instruct",
        "max_model_len": 8192,
        "limit_mm": False,
    },
    "olmo3-7b-think-sft": {
        "hf_dir": "/fast/jswanson/hf_home/hub/models--allenai--Olmo-3-7B-Think-SFT",
        "family": "OLMo3-7B-Think",
        "kind": "olmo_think",
        "max_model_len": 12288,
        "limit_mm": False,
    },
    "olmo3-7b-think-dpo": {
        "hf_dir": "/fast/jswanson/hf_home/hub/models--allenai--Olmo-3-7B-Think-DPO",
        "family": "OLMo3-7B-Think",
        "kind": "olmo_think",
        "max_model_len": 12288,
        "limit_mm": False,
    },
    "olmo3-7b-think-rlvr": {
        "hf_dir": "/fast/jswanson/hf_home/hub/models--allenai--Olmo-3-7B-Think",
        "family": "OLMo3-7B-Think",
        "kind": "olmo_think",
        "max_model_len": 12288,
        "limit_mm": False,
    },
    "olmo3-32b-think-sft": {
        "hf_dir": "/fast/jswanson/hf_home/hub/models--allenai--Olmo-3-32B-Think-SFT",
        "family": "OLMo3-32B-Think",
        "kind": "olmo_think",
        "max_model_len": 24576,
        "limit_mm": False,
    },
    "olmo3-32b-think-dpo": {
        "hf_dir": "/fast/jswanson/hf_home/hub/models--allenai--Olmo-3-32B-Think-DPO",
        "family": "OLMo3-32B-Think",
        "kind": "olmo_think",
        "max_model_len": 24576,
        "limit_mm": False,
    },
    "olmo3-32b-think-rlvr": {
        "hf_dir": "/fast/jswanson/hf_home/hub/models--allenai--Olmo-3-32B-Think",
        "family": "OLMo3-32B-Think",
        "kind": "olmo_think",
        "max_model_len": 24576,
        "limit_mm": False,
    },
    "olmo3.1-32b-think": {
        "hf_dir": "/fast/jswanson/hf_home/hub/models--allenai--Olmo-3.1-32B-Think",
        "family": "OLMo3-32B-Think",
        "kind": "olmo_think",
        "max_model_len": 12288,
        "limit_mm": False,
    },
}


def resolve_model_path(model_key: str) -> Tuple[str, Dict[str, Any]]:
    cfg = MODEL_REGISTRY[model_key]
    if "path" in cfg:
        return cfg["path"], cfg
    snaps = sorted(Path(cfg["hf_dir"]).glob("snapshots/*"))
    assert snaps, f"No snapshot found in {cfg['hf_dir']}"
    return str(snaps[0]), cfg


class ChatAdapter:
    """Formats messages and splits thinking vs. final answer cleanly across model families."""

    def __init__(self, model_path: str, kind: str, tokenizer: Any, enable_thinking: bool = True):
        self.model_path = model_path
        self.kind = kind
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.dsv4_encode = None
        self.dsv4_parse = None
        if kind == "deepseek_v4":
            enc_dir = os.path.join(model_path, "encoding")
            if enc_dir not in sys.path:
                sys.path.insert(0, enc_dir)
            from encoding_dsv4 import encode_messages, parse_message_from_completion_text

            self.dsv4_encode = encode_messages
            self.dsv4_parse = parse_message_from_completion_text

    def format_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self.kind == "olmo_base":
            return f"{system}\n\n{user}\n\n### Assumption Challenged\n"
        if self.kind == "deepseek_v4":
            mode = "thinking" if self.enable_thinking else "chat"
            return self.dsv4_encode(messages, thinking_mode=mode)

        kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self.kind == "qwen38":
            kwargs["enable_thinking"] = self.enable_thinking
            if self.enable_thinking:
                kwargs["reasoning_effort"] = "low"
        elif self.kind == "gemma4":
            kwargs["enable_thinking"] = self.enable_thinking
        elif self.kind == "gpt_oss":
            kwargs["reasoning_effort"] = "low"
        elif self.kind == "mistral3":
            kwargs["reasoning_effort"] = "none"

        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            kwargs.pop("reasoning_effort", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def parse_output(self, raw_text: str) -> Tuple[str, str]:
        """Returns (thinking_text, final_text)."""
        if not raw_text:
            return "", ""

        if self.kind == "olmo_base":
            cleaned = raw_text
            for tok in ("<|endoftext|>", "<eos>", "</s>"):
                cleaned = cleaned.replace(tok, "")
            return "", ("### Assumption Challenged\n" + cleaned.strip())
        if self.kind == "deepseek_v4":
            mode = "thinking" if self.enable_thinking else "chat"
            try:
                parsed = self.dsv4_parse(raw_text, thinking_mode=mode)
                return (parsed.get("reasoning_content") or "").strip(), (parsed.get("content") or "").strip()
            except Exception:
                pass

        if self.kind == "gpt_oss":
            # Harmony channels: <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...
            m_final = re.search(
                r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|\Z)",
                raw_text,
                flags=re.DOTALL,
            )
            m_an = re.search(
                r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|\Z)",
                raw_text,
                flags=re.DOTALL,
            )
            if m_final:
                thinking = m_an.group(1).strip() if m_an else ""
                final = re.sub(r"<\|[a-zA-Z0-9_]+\|>", "", m_final.group(1)).strip()
                return thinking, final
            # Fallback if model emitted without explicit tags
            cleaned = re.sub(r"<\|[a-zA-Z0-9_]+\|>", "", raw_text).strip()
            return "", cleaned

        if self.kind == "gemma4":
            # Gemma 4 uses <|channel>thought\n...\n<channel|>
            if "<channel|>" in raw_text:
                parts = raw_text.split("<channel|>", 1)
                thinking = parts[0].replace("<|channel>thought", "").strip()
                final = parts[1]
                final = re.sub(r"<\|?[a-zA-Z0-9_]+>|<[a-zA-Z0-9_]+\|>", "", final).strip()
                return thinking, final

        # Standard <think>...</think> (Qwen3.8, OLMo-3 Think, DeepSeek fallback)
        if "</think>" in raw_text:
            pre, post = raw_text.split("</think>", 1)
            thinking = pre.replace("<think>", "").strip()
            final = post.strip()
            # Clean stray EOS tokens
            for tok in ("<|im_end|>", "<|endoftext|>", "<eos>", "<turn|>", "</s>"):
                final = final.replace(tok, "").strip()
            return thinking, final

        # Strip common EOS/special tokens if no thinking block
        cleaned = raw_text
        for tok in ("<|im_end|>", "<|endoftext|>", "<eos>", "<turn|>", "<|return|>", "</s>"):
            cleaned = cleaned.replace(tok, "")
        return "", cleaned.strip()


def load_llm(model_key: str, tp_size: int, gpu_mem_util: float, max_model_len_override: Optional[int] = None):
    from vllm import LLM

    model_path, cfg = resolve_model_path(model_key)
    max_len = max_model_len_override or cfg["max_model_len"]
    kwargs: Dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": tp_size,
        "dtype": "bfloat16",
        "max_model_len": max_len,
        "gpu_memory_utilization": gpu_mem_util,
        "trust_remote_code": True,
        "enable_prefix_caching": True,
    }
    if cfg.get("limit_mm"):
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if cfg["kind"] in ("deepseek_v4", "mistral3", "gpt_oss"):
        # Let quantized checkpoints (FP8 / MXFP4) use their native weight dtype
        kwargs["dtype"] = "auto"
    if cfg["kind"] == "deepseek_v4":
        kwargs["kv_cache_dtype"] = "fp8"

    print(f"[vllm_runner] Loading {model_key} from {model_path} (tp={tp_size}, max_len={max_len})...", flush=True)
    t0 = time.time()
    llm = LLM(**kwargs)
    print(f"[vllm_runner] Loaded {model_key} in {time.time() - t0:.1f}s", flush=True)
    return llm, model_path, cfg


def generate_batch(
    llm: Any,
    adapter: ChatAdapter,
    prompts: List[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
    n: int = 1,
    seed: int = 42,
    seeds: Optional[List[int]] = None,
) -> List[List[Dict[str, Any]]]:
    from vllm import SamplingParams

    # Keep special tokens in output when needed for channel parsing (gpt_oss, gemma4, qwen38, deepseek_v4, olmo_think)
    keep_special = adapter.kind in ("gpt_oss", "gemma4", "qwen38", "deepseek_v4", "olmo_think")
    if seeds is not None:
        sp: Any = [
            SamplingParams(
                temperature=temperature,
                top_p=top_p if temperature > 0 else 1.0,
                max_tokens=max_tokens,
                n=n,
                seed=int(s),
                skip_special_tokens=not keep_special,
            )
            for s in seeds
        ]
    elif temperature > 0 and len(prompts) > 1:
        sp = [
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=n,
                seed=int(seed + i * 1009),
                skip_special_tokens=not keep_special,
            )
            for i in range(len(prompts))
        ]
    else:
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p if temperature > 0 else 1.0,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            skip_special_tokens=not keep_special,
        )
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    results = []
    for out in outputs:
        sample_list = []
        for comp in out.outputs:
            thinking, final = adapter.parse_output(comp.text)
            sample_list.append(
                {
                    "raw_text": comp.text,
                    "thinking": thinking,
                    "text": final,
                    "finish_reason": comp.finish_reason,
                    "num_tokens": len(comp.token_ids),
                }
            )
        results.append(sample_list)
    return results


def run_single_mode(
    llm: Any,
    adapter: ChatAdapter,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    mode: str,
    out_path: Path,
):
    data_dir = Path(args.data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mode in ("direct", "high_temp", "vs", "icr", "axiom_neg"):
        scenarios = json.loads((data_dir / "scenarios.json").read_text())
    elif mode == "contamination":
        pairs = json.loads((data_dir / "contamination_pairs.json").read_text())
        scenarios = []
        for p in pairs:
            for side in ("orig", "iso"):
                item = dict(p[side])
                item["pair_id"] = p["pair_id"]
                item["concept"] = p["concept"]
                scenarios.append(item)
    else:
        raise ValueError(f"Unknown generation mode: {mode}")

    if args.limit_scenarios:
        scenarios = scenarios[: args.limit_scenarios]

    records: List[Dict[str, Any]] = []
    t_start = time.time()

    if mode in ("direct", "high_temp", "contamination"):
        temp = 1.3 if mode == "high_temp" else args.temperature
        prompts = [
            adapter.format_prompt(SYSTEM_SCIENTIST, DIRECT_USER_TEMPLATE.format(dossier=s["dossier"]))
            for s in scenarios
        ]
        batch_out = generate_batch(
            llm,
            adapter,
            prompts,
            temperature=temp,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=args.n_samples,
            seed=args.seed,
        )
        for s, samples in zip(scenarios, batch_out):
            for idx, samp in enumerate(samples):
                records.append(
                    {
                        "id": f"{args.model}__{mode}__{s['id']}__s{idx:02d}",
                        "model": args.model,
                        "family": cfg["family"],
                        "mode": mode,
                        "scenario_id": s["id"],
                        "domain": s["domain"],
                        "sample_idx": idx,
                        "temperature": temp,
                        "hypothesis": samp["text"],
                        "thinking": samp["thinking"],
                        "finish_reason": samp["finish_reason"],
                        "num_tokens": samp["num_tokens"],
                    }
                )

    elif mode == "vs":
        n_calls = max(1, args.n_samples // 5)
        prompts = [
            adapter.format_prompt(SYSTEM_SCIENTIST, VS_USER_TEMPLATE.format(dossier=s["dossier"]))
            for s in scenarios
        ]
        batch_out = generate_batch(
            llm,
            adapter,
            prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=max(args.max_tokens, 6144),
            n=n_calls,
            seed=args.seed,
        )
        for s, calls in zip(scenarios, batch_out):
            sample_idx = 0
            for call_i, samp in enumerate(calls):
                parsed = extract_json_object(samp["text"])
                hyps = parsed.get("hypotheses", []) if parsed else []
                if not isinstance(hyps, list) or not hyps:
                    records.append(
                        {
                            "id": f"{args.model}__vs__{s['id']}__s{sample_idx:02d}",
                            "model": args.model,
                            "family": cfg["family"],
                            "mode": "vs",
                            "scenario_id": s["id"],
                            "domain": s["domain"],
                            "sample_idx": sample_idx,
                            "call_idx": call_i,
                            "vs_prob": None,
                            "hypothesis": samp["text"],
                            "thinking": samp["thinking"],
                            "finish_reason": samp["finish_reason"],
                            "num_tokens": samp["num_tokens"],
                            "parse_ok": False,
                        }
                    )
                    sample_idx += 1
                    continue
                for h in hyps[:5]:
                    formatted = (
                        f"### Assumption Challenged\n{h.get('assumption_challenged', '')}\n\n"
                        f"### Core Hypothesis & Mechanism\n{h.get('mechanism', '')}\n\n"
                        f"### Reconciliation with Reported Data\n{h.get('reconciliation', '')}\n\n"
                        f"### Falsifiable Prediction\n{h.get('falsifiable_prediction', '')}"
                    )
                    records.append(
                        {
                            "id": f"{args.model}__vs__{s['id']}__s{sample_idx:02d}",
                            "model": args.model,
                            "family": cfg["family"],
                            "mode": "vs",
                            "scenario_id": s["id"],
                            "domain": s["domain"],
                            "sample_idx": sample_idx,
                            "call_idx": call_i,
                            "vs_prob": h.get("probability"),
                            "hypothesis": formatted,
                            "thinking": samp["thinking"] if h == hyps[0] else "",
                            "finish_reason": samp["finish_reason"],
                            "num_tokens": samp["num_tokens"],
                            "parse_ok": True,
                        }
                    )
                    sample_idx += 1

    elif mode == "icr":
        n_chains = 3
        n_turns = max(1, args.n_samples // n_chains)
        histories: Dict[Tuple[int, int], List[str]] = {
            (si, ci): [] for si in range(len(scenarios)) for ci in range(n_chains)
        }
        for turn in range(n_turns):
            turn_prompts = []
            turn_keys = []
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

            batch_out = generate_batch(
                llm,
                adapter,
                turn_prompts,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                n=1,
                seed=args.seed + turn * 101,
            )
            for (si, ci), samples in zip(turn_keys, batch_out):
                s = scenarios[si]
                samp = samples[0]
                histories[(si, ci)].append(summarize_hypothesis_for_icr(samp["text"]))
                sample_idx = ci * n_turns + turn
                records.append(
                    {
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
                    }
                )
            print(f"[icr] Completed turn {turn+1}/{n_turns} ({len(records)} total records)", flush=True)

    elif mode == "axiom_neg":
        s1_prompts = [
            adapter.format_prompt(SYSTEM_SCIENTIST, AXIOM_EXTRACT_TEMPLATE.format(dossier=s["dossier"]))
            for s in scenarios
        ]
        s1_out = generate_batch(
            llm,
            adapter,
            s1_prompts,
            temperature=0.4,
            top_p=0.95,
            max_tokens=3072,
            n=1,
            seed=args.seed,
        )
        extracted_assumptions: Dict[str, List[Dict[str, str]]] = {}
        for s, samples in zip(scenarios, s1_out):
            parsed = extract_json_object(samples[0]["text"])
            assumps = parsed.get("assumptions", []) if parsed else []
            if not isinstance(assumps, list) or len(assumps) < 3:
                assumps = [
                    {"axis": "Measurement/Construct", "assumption": "The reported outcome metric directly reflects the underlying causal quantity without observational or protocol distortion.", "negation_prompt": "Assume the observed anomaly stems from a hidden shift in what the measurement or protocol captures."},
                    {"axis": "Causal Direction/Feedback", "assumption": "Causal influence flows unidirectionally from the intervention to the outcome without adaptive system feedback.", "negation_prompt": "Assume the system adapts or feeds back nonlinearly against the intervention, inverting the expected sign."},
                    {"axis": "Spatiotemporal/Kinetic Scale", "assumption": "Steady-state or aggregate measurements capture the active dynamics regardless of timing, phase, or local transients.", "negation_prompt": "Assume timing, phase alignment, or transient kinetics govern the effect despite identical averages."},
                    {"axis": "Hidden Boundary Coupling", "assumption": "The studied subsystem is effectively isolated from its surrounding environment, substrate, or institutional context.", "negation_prompt": "Assume an unmodeled physical, biological, or institutional boundary coupling interacts selectively with the treatment."},
                    {"axis": "Latent Heterogeneity/Selection", "assumption": "The observed group reflects a homogeneous population without threshold switching or selection-induced stratification.", "negation_prompt": "Assume a hidden threshold regime switch or selection/composition shift separates subpopulations."},
                    {"axis": "Secondary/Indirect Pathway", "assumption": "The primary target pathway dominates the outcome without activating competing or collateral pathways.", "negation_prompt": "Assume suppressing or altering the primary target triggers a dominant collateral or indirect pathway."},
                ]
            extracted_assumptions[s["id"]] = assumps[:6]

        assump_path = out_path.with_suffix(".assumptions.json")
        assump_path.write_text(json.dumps(extracted_assumptions, indent=2))

        s2_prompts = []
        s2_meta = []
        for s in scenarios:
            assumps = extracted_assumptions[s["id"]]
            per_ax = max(1, args.n_samples // len(assumps))
            rem = args.n_samples - per_ax * len(assumps)
            for ai, ax in enumerate(assumps):
                n_here = per_ax + (1 if ai < rem else 0)
                user_msg = AXIOM_NEGATE_TEMPLATE.format(
                    dossier=s["dossier"],
                    assumption=ax.get("assumption", ""),
                    negation_prompt=ax.get("negation_prompt", ""),
                )
                s2_prompts.append(adapter.format_prompt(SYSTEM_SCIENTIST, user_msg))
                s2_meta.append((s, ai, ax, n_here))

        max_n = max(m[3] for m in s2_meta)
        s2_out = generate_batch(
            llm,
            adapter,
            s2_prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=max_n,
            seed=args.seed + 777,
        )
        counters: Dict[str, int] = {}
        for (s, ai, ax, n_here), samples in zip(s2_meta, s2_out):
            for samp in samples[:n_here]:
                idx = counters.get(s["id"], 0)
                counters[s["id"]] = idx + 1
                records.append(
                    {
                        "id": f"{args.model}__axiom_neg__{s['id']}__s{idx:02d}",
                        "model": args.model,
                        "family": cfg["family"],
                        "mode": "axiom_neg",
                        "scenario_id": s["id"],
                        "domain": s["domain"],
                        "sample_idx": idx,
                        "axiom_idx": ai,
                        "axiom_axis": ax.get("axis", ""),
                        "targeted_assumption": ax.get("assumption", ""),
                        "hypothesis": samp["text"],
                        "thinking": samp["thinking"],
                        "finish_reason": samp["finish_reason"],
                        "num_tokens": samp["num_tokens"],
                    }
                )

    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    non_empty = sum(1 for r in records if len((r.get("hypothesis") or "").strip()) > 40)
    print(
        f"[vllm_runner] DONE {args.model}/{mode}: {len(records)} records "
        f"({non_empty} non-empty) saved to {out_path} in {time.time() - t_start:.1f}s",
        flush=True,
    )


def run_generation_mode(args: argparse.Namespace):
    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    llm, model_path, cfg = load_llm(args.model, args.tp_size, args.gpu_mem_util, args.max_model_len)
    enable_thinking = not args.no_thinking
    if cfg["kind"] in ("mistral3", "olmo_instruct"):
        enable_thinking = False
    adapter = ChatAdapter(model_path, cfg["kind"], llm.get_tokenizer(), enable_thinking=enable_thinking)

    for mode in modes:
        if len(modes) == 1 and args.output.endswith(".jsonl"):
            out_path = Path(args.output)
        else:
            out_dir = Path(args.output)
            out_path = out_dir / f"gen_{args.model}_{mode}.jsonl"
        run_single_mode(llm, adapter, cfg, args, mode, out_path)


def run_judge_mode(args: argparse.Namespace):
    """Score one or more JSONL files (or calibration.json) using a specified judge model."""
    data_dir = Path(args.data_dir)
    scenario_map: Dict[str, Dict[str, Any]] = {}
    for s in json.loads((data_dir / "scenarios.json").read_text()):
        scenario_map[s["id"]] = s
    for p in json.loads((data_dir / "contamination_pairs.json").read_text()):
        for side in ("orig", "iso"):
            scenario_map[p[side]["id"]] = p[side]

    items: List[Dict[str, Any]] = []
    for inp in args.input_files:
        p = Path(inp)
        if p.suffix == ".json":
            arr = json.loads(p.read_text())
            for x in arr:
                items.append(
                    {
                        "id": x.get("cal_id") or x["id"],
                        "scenario_id": x["scenario_id"],
                        "hypothesis": x["hypothesis"],
                        "source_file": str(p),
                        "meta": {k: v for k, v in x.items() if k != "hypothesis"},
                    }
                )
        else:
            with p.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    x = json.loads(line)
                    items.append(
                        {
                            "id": x["id"],
                            "scenario_id": x["scenario_id"],
                            "hypothesis": x.get("hypothesis", ""),
                            "source_file": str(p),
                            "meta": {
                                k: x.get(k)
                                for k in ("model", "family", "mode", "domain", "sample_idx")
                            },
                        }
                    )

    print(f"[judge] Loaded {len(items)} items to score with judge={args.model}", flush=True)
    llm, model_path, cfg = load_llm(args.model, args.tp_size, args.gpu_mem_util, args.max_model_len)
    enable_thinking = not args.no_thinking
    if cfg["kind"] in ("mistral3", "olmo_instruct"):
        enable_thinking = False
    adapter = ChatAdapter(model_path, cfg["kind"], llm.get_tokenizer(), enable_thinking=enable_thinking)

    prompts = []
    for it in items:
        s = scenario_map[it["scenario_id"]]
        cblock = "\n".join(s["empirical_constraints"])
        user_msg = JUDGE_USER_TEMPLATE.format(
            dossier=s["dossier"],
            constraints_block=cblock,
            hypothesis=(it["hypothesis"] or "(empty response)")[:4500],
        )
        prompts.append(adapter.format_prompt(JUDGE_SYSTEM, user_msg))

    t0 = time.time()
    batch_out = generate_batch(
        llm,
        adapter,
        prompts,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        n=1,
        seed=42,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_ok = 0
    with out_path.open("w") as f:
        for it, samples in zip(items, batch_out):
            samp = samples[0]
            obj = extract_json_object(samp["text"])
            if obj and all(k in obj for k in ("surprise", "coherence", "realism")):
                parsed_ok += 1
                # Enforce hard constraint cap if judge ever violated rubric instruction
                checks = obj.get("constraint_checks", {})
                n_contra = sum(1 for v in checks.values() if str(v).upper() == "CONTRADICTED")
                coh = int(round(float(obj["coherence"])))
                if n_contra >= 1 and coh > 2:
                    coh = 2
                rec = {
                    "id": it["id"],
                    "judge": args.model,
                    "scenario_id": it["scenario_id"],
                    "meta": it["meta"],
                    "parse_ok": True,
                    "constraint_checks": checks,
                    "n_contradicted": n_contra,
                    "surprise": int(round(float(obj["surprise"]))),
                    "coherence": coh,
                    "realism": int(round(float(obj["realism"]))),
                    "assumption_dropped": str(obj.get("assumption_dropped", "")).strip(),
                    "mechanism_summary": str(obj.get("mechanism_summary", "")).strip(),
                    "brief_rationale": str(obj.get("brief_rationale", "")).strip(),
                }
            else:
                rec = {
                    "id": it["id"],
                    "judge": args.model,
                    "scenario_id": it["scenario_id"],
                    "meta": it["meta"],
                    "parse_ok": False,
                    "raw_judge_text": samp["text"][:1000],
                }
            f.write(json.dumps(rec) + "\n")
    print(
        f"[judge] DONE judge={args.model}: {parsed_ok}/{len(items)} parsed cleanly "
        f"saved to {out_path} in {time.time() - t0:.1f}s",
        flush=True,
    )

    if args.also_taxonomy:
        print(f"[judge->taxonomy] Reusing warm {args.model} engine for blind mechanistic taxonomy...", flush=True)
        run_taxonomy_with_llm(llm, adapter, scenario_map, args.model, [str(out_path)], Path(args.also_taxonomy), args.max_tokens)


def run_taxonomy_with_llm(
    llm: Any,
    adapter: ChatAdapter,
    scenario_map: Dict[str, Dict[str, Any]],
    model_name: str,
    input_files: List[str],
    out_path: Path,
    max_tokens: int,
):
    import random

    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for jf in input_files:
        with open(jf) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if not rec.get("parse_ok"):
                    continue
                # Exclude calibration items from model taxonomy counts
                if str(rec["id"]).startswith("cal_"):
                    continue
                sid = rec["scenario_id"]
                by_scenario.setdefault(sid, []).append(rec)

    rng = random.Random(42)
    sids = sorted(by_scenario.keys())
    build_prompts = []
    for sid in sids:
        items = list(by_scenario[sid])
        rng.shuffle(items)
        sample = items[:100]
        lines = [
            f"- [{i+1}] Assumption: {x.get('assumption_dropped','')} | Mechanism: {x.get('mechanism_summary','')}"
            for i, x in enumerate(sample)
        ]
        s = scenario_map[sid]
        user_msg = TAXONOMY_BUILD_TEMPLATE.format(
            title=s["title"],
            prevailing_paradigm=s.get("prevailing_paradigm", s["dossier"].split("\n\n")[0]),
            sample_list="\n".join(lines),
        )
        build_prompts.append(adapter.format_prompt(JUDGE_SYSTEM, user_msg))

    t0 = time.time()
    tax_out = generate_batch(
        llm, adapter, build_prompts, temperature=0.0, top_p=1.0, max_tokens=max(max_tokens, 3072), n=1, seed=42
    )
    taxonomies: Dict[str, List[Dict[str, str]]] = {}
    for sid, samples in zip(sids, tax_out):
        obj = extract_json_object(samples[0]["text"])
        fws = obj.get("frameworks", []) if obj else []
        taxonomies[sid] = fws
        print(f"[taxonomy] {sid}: {len(fws)} canonical frameworks built", flush=True)

    chunk_size = 25
    assign_prompts = []
    assign_meta = []
    for sid in sids:
        fws = taxonomies[sid]
        fw_block = "\n".join(f"- {fw['id']}: {fw['name']} — {fw['definition']}" for fw in fws)
        items = by_scenario[sid]
        s = scenario_map[sid]
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            items_block = "\n".join(
                f'- "item_{k}": Assumption: {x.get("assumption_dropped","")} | Mechanism: {x.get("mechanism_summary","")}'
                for k, x in enumerate(chunk)
            )
            user_msg = TAXONOMY_ASSIGN_TEMPLATE.format(
                title=s["title"], frameworks_block=fw_block, items_block=items_block
            )
            assign_prompts.append(adapter.format_prompt(JUDGE_SYSTEM, user_msg))
            assign_meta.append((sid, chunk))

    print(f"[taxonomy] Assigning {sum(len(v) for v in by_scenario.values())} items in {len(assign_prompts)} batches...", flush=True)
    assign_out = generate_batch(
        llm, adapter, assign_prompts, temperature=0.0, top_p=1.0, max_tokens=max(max_tokens, 3072), n=1, seed=42
    )
    assignments: Dict[str, str] = {}
    for (sid, chunk), samples in zip(assign_meta, assign_out):
        obj = extract_json_object(samples[0]["text"])
        amap = obj.get("assignments", obj) if obj else {}
        for k, x in enumerate(chunk):
            fw_id = amap.get(f"item_{k}", "F_PARSE_FAIL")
            assignments[x["id"]] = str(fw_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"judge": model_name, "taxonomies": taxonomies, "assignments": assignments},
            indent=2,
        )
    )
    print(f"[taxonomy] DONE: {len(assignments)} assignments saved to {out_path} in {time.time() - t0:.1f}s", flush=True)


def run_taxonomy_mode(args: argparse.Namespace):
    """Build blind canonical mechanistic taxonomies per scenario and classify ALL hypotheses."""
    data_dir = Path(args.data_dir)
    scenario_map: Dict[str, Dict[str, Any]] = {}
    for s in json.loads((data_dir / "scenarios.json").read_text()):
        scenario_map[s["id"]] = s
    for p in json.loads((data_dir / "contamination_pairs.json").read_text()):
        for side in ("orig", "iso"):
            scenario_map[p[side]["id"]] = p[side]

    llm, model_path, cfg = load_llm(args.model, args.tp_size, args.gpu_mem_util, args.max_model_len)
    enable_thinking = not args.no_thinking
    if cfg["kind"] in ("mistral3", "olmo_instruct"):
        enable_thinking = False
    adapter = ChatAdapter(model_path, cfg["kind"], llm.get_tokenizer(), enable_thinking=enable_thinking)
    run_taxonomy_with_llm(llm, adapter, scenario_map, args.model, args.input_files, Path(args.output), args.max_tokens)


def run_embed_mode(args: argparse.Namespace):
    """Compute L2-normalized embeddings for all hypotheses using cached Qwen3-Embedding-0.6B and all-mpnet-base-v2."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    items: List[Tuple[str, str]] = []
    seen = set()
    for inp in args.input_files:
        with open(inp) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = rec["id"]
                if rid in seen:
                    continue
                seen.add(rid)
                # Prefer judge-extracted canonical mechanism if available, else raw hypothesis
                txt = rec.get("mechanism_summary") or rec.get("hypothesis") or ""
                items.append((rid, txt[:2000]))

    print(f"[embed] Encoding {len(items)} unique texts...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embed_models = {
        "qwen3_emb": {
            "hf_dir": "/fast/jswanson/hf_home/hub/models--Qwen--Qwen3-Embedding-0.6B",
            "pooling": "last",
        },
        "mpnet": {
            "hf_dir": "/fast/jswanson/hf_home/hub/models--sentence-transformers--all-mpnet-base-v2",
            "pooling": "mean",
        },
    }

    ids = [x[0] for x in items]
    texts = [x[1] for x in items]
    out_dict: Dict[str, Any] = {"ids": np.array(ids)}

    for name, ecfg in embed_models.items():
        snaps = sorted(Path(ecfg["hf_dir"]).glob("snapshots/*"))
        assert snaps, f"Missing snapshot in {ecfg['hf_dir']}"
        mpath = str(snaps[0])
        print(f"[embed] Running {name} from {mpath} on {device}...", flush=True)
        tok = AutoTokenizer.from_pretrained(mpath, padding_side="left" if ecfg["pooling"] == "last" else "right")
        model = AutoModel.from_pretrained(mpath, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
        model.eval()
        vecs = []
        bs = 64
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch_t = texts[i : i + bs]
                enc = tok(batch_t, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                out = model(**enc)
                hs = out.last_hidden_state
                if ecfg["pooling"] == "last":
                    # Last non-padding token (padding_side='left' -> index -1)
                    emb = hs[:, -1, :]
                else:
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    emb = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                emb = F.normalize(emb.float(), p=2, dim=-1).cpu().numpy()
                vecs.append(emb)
        out_dict[name] = np.concatenate(vecs, axis=0).astype(np.float32)
        del model, tok
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out_dict)
    print(f"[embed] Saved embeddings to {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["generate", "judge", "taxonomy", "embed"], required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--mode", default="direct")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output", required=True)
    ap.add_argument("--input-files", nargs="*", default=[])
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--limit-scenarios", type=int, default=None)
    ap.add_argument("--also-taxonomy", default=None)
    args = ap.parse_args()

    if args.task == "generate":
        run_generation_mode(args)
    elif args.task == "judge":
        run_judge_mode(args)
    elif args.task == "taxonomy":
        run_taxonomy_mode(args)
    elif args.task == "embed":
        run_embed_mode(args)


if __name__ == "__main__":
    main()
