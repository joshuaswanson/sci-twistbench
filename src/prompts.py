#!/usr/bin/env python3
"""Prompt templates and structured parsers for Sci-TwistBench."""

import json
import re
from typing import Any, Dict, List, Optional

SYSTEM_SCIENTIST = (
    "You are an independent research scientist specializing in resolving anomalous empirical findings. "
    "Respond directly and concisely with rigorous, mechanistically grounded scientific reasoning."
)

DIRECT_USER_TEMPLATE = """Read the following empirical research dossier describing an unexplained scientific anomaly:

---
{dossier}
---

Task: Propose ONE concrete, paradigm-shifting scientific hypothesis that forces a re-interpretation of the prior observations while remaining strictly consistent with EVERY reported empirical finding and control measurement above. Avoid trivial explanations already ruled out by the controls, and avoid unphysical or unfalsifiable speculation.

Format your response using exactly these four headings:
### Assumption Challenged
(1-2 sentences identifying the specific implicit assumption of the prevailing paradigm that your hypothesis drops or inverts.)

### Core Hypothesis & Mechanism
(1 concise paragraph explaining the exact causal mechanism.)

### Reconciliation with Reported Data
(2-4 bullet points showing explicitly how your hypothesis accounts for every reported result and control check.)

### Falsifiable Prediction
(1-2 sentences specifying a concrete, discriminating experiment or measurement that would falsify your hypothesis.)"""


VS_USER_TEMPLATE = """Read the following empirical research dossier describing an unexplained scientific anomaly:

---
{dossier}
---

Task: Rather than giving only the single most obvious explanation, survey the full space of genuinely distinct explanatory paradigms. Generate 5 FUNDAMENTALLY DIFFERENT paradigm-shifting scientific hypotheses for this anomaly. Each of the 5 hypotheses must challenge a DIFFERENT underlying assumption and propose a qualitatively distinct causal mechanism—not minor wording variants of the same idea—while remaining strictly consistent with EVERY reported empirical finding and control measurement above.

For each hypothesis, estimate its relative prior plausibility probability (a number between 0.01 and 0.99; include both high-probability and creative lower-probability tail hypotheses).

Return ONLY a valid JSON object matching this exact schema (no markdown fences or extra prose outside JSON):
{{
  "hypotheses": [
    {{
      "index": 1,
      "probability": 0.30,
      "assumption_challenged": "1 sentence",
      "mechanism": "1 concise paragraph explaining the causal mechanism",
      "reconciliation": "Brief explanation of how it satisfies all reported data and controls",
      "falsifiable_prediction": "1 concrete discriminating test"
    }}
  ]
}}"""


ICR_USER_TEMPLATE = """Read the following empirical research dossier describing an unexplained scientific anomaly:

---
{dossier}
---

Previously proposed explanatory frameworks (DO NOT REUSE, PARAPHRASE, OR VARIANT-MODIFY ANY OF THESE):
{prior_list}

Task: Propose ONE NEW, fundamentally distinct paradigm-shifting scientific hypothesis that explores a completely different causal mechanism and challenges a different underlying assumption than ALL previously proposed frameworks above, while remaining strictly consistent with EVERY reported empirical finding and control measurement in the dossier.

Format your response using exactly these four headings:
### Assumption Challenged
(1-2 sentences identifying the specific implicit assumption dropped or inverted.)

### Core Hypothesis & Mechanism
(1 concise paragraph explaining the exact causal mechanism.)

### Reconciliation with Reported Data
(2-4 bullet points showing explicitly how your hypothesis accounts for every reported result and control check.)

### Falsifiable Prediction
(1-2 sentences specifying a concrete, discriminating experiment or measurement that would falsify your hypothesis.)"""


AXIOM_EXTRACT_TEMPLATE = """Read the following empirical research dossier describing an unexplained scientific anomaly:

---
{dossier}
---

Task: Before proposing any explanations, perform a structural audit of the prevailing paradigm's conceptual space. Identify exactly 6 DISTINCT, load-bearing implicit assumptions baked into how the prevailing paradigm interprets this system (covering different axes such as measurement/construct definition, causal direction/confounding, hidden system feedbacks, spatiotemporal/kinetic scales, population/selection structure, and boundary/environmental couplings).

Return ONLY a valid JSON object matching this exact schema:
{{
  "assumptions": [
    {{
      "id": 1,
      "axis": "short 2-4 word category",
      "assumption": "1 clear sentence stating the implicit assumption held by the prevailing paradigm",
      "negation_prompt": "1 sentence stating what happens conceptually if this specific assumption is false or inverted"
    }}
  ]
}}"""


AXIOM_NEGATE_TEMPLATE = """Read the following empirical research dossier describing an unexplained scientific anomaly:

---
{dossier}
---

Conceptual Transformation Directive:
Explicitly DROP or INVERT the following implicit assumption of the prevailing paradigm:
- Target Assumption: {assumption}
- Reframing Direction: {negation_prompt}

Task: Constructing strictly from the negation/inversion of THIS specific assumption, develop ONE concrete, scientifically realistic, paradigm-shifting hypothesis that resolves the anomaly while remaining strictly consistent with EVERY reported empirical finding and control measurement above.

Format your response using exactly these four headings:
### Assumption Challenged
(1-2 sentences stating how your hypothesis drops or inverts the targeted assumption.)

### Core Hypothesis & Mechanism
(1 concise paragraph explaining the exact causal mechanism.)

### Reconciliation with Reported Data
(2-4 bullet points showing explicitly how your hypothesis accounts for every reported result and control check.)

### Falsifiable Prediction
(1-2 sentences specifying a concrete, discriminating experiment or measurement that would falsify your hypothesis.)"""


JUDGE_SYSTEM = (
    "You are an impartial senior scientific referee evaluating candidate hypotheses for an unexplained empirical anomaly. "
    "Be strict, calibrated, and faithful to the empirical constraints reported in the dossier. "
    "Output ONLY valid JSON matching the requested schema."
)


JUDGE_USER_TEMPLATE = """Evaluate the candidate scientific hypothesis below against the empirical research dossier and its four hard empirical constraints.

=== EMPIRICAL DOSSIER ===
{dossier}

=== HARD EMPIRICAL CONSTRAINTS (MUST ALL BE SATISFIED) ===
{constraints_block}

=== CANDIDATE HYPOTHESIS ===
{hypothesis}

=== EVALUATION RUBRIC ===
1. Constraint Audit: For each constraint E1, E2, E3, E4, classify the hypothesis as:
   - "CONSISTENT": explicitly accounts for or is fully compatible with the constraint.
   - "PARTIAL": vaguely compatible or leaves the constraint unaddressed without directly contradicting it.
   - "CONTRADICTED": directly conflicts with or relies on a mechanism explicitly ruled out by the constraint.

2. Surprise / Reframing Depth (1-5):
   1 = Trivial restatement, empty speculation, or fails to propose a coherent mechanism.
   2 = Incremental/Exploratory: tweaks parameters, noise, or standard factors strictly inside the prevailing paradigm without changing its core assumptions.
   3 = Combinatorial: blends adjacent known variables or secondary confounders with modest conceptual shift.
   4 = Transformational: explicitly drops, inverts, or replaces a load-bearing implicit assumption of the prevailing paradigm, forcing a genuine re-interpretation of the prior observations.
   5 = Deep Paradigm Reframing: fundamentally restructures the conceptual framing (causal ontology, construct definition, or regime structure) so all seemingly paradoxical facts lock together unexpectedly in hindsight.

3. Empirical Coherence (1-5):
   1 = Directly contradicts two or more hard constraints (E1-E4) or is internally incoherent.
   2 = Directly contradicts one hard constraint (E1-E4), even if otherwise well-written. (STRICT RULE: if ANY constraint is CONTRADICTED, Coherence MUST be <= 2.)
   3 = No direct contradiction, but ignores or hand-waves away key quantitative features or controls.
   4 = Consistent with all four constraints (E1-E4) with sound causal logic.
   5 = Seamlessly and quantitatively reconciles all four constraints (E1-E4) with airtight backward coherence.

4. Scientific Realism & Falsifiability (1-5):
   1 = Violates established physical/biological/mathematical laws, invokes pseudoscience/sci-fi forces, or is completely absurd.
   2 = Scientifically implausible, relies on unmotivated ad-hoc mechanisms, or is unfalsifiable.
   3 = Loosely plausible in domain terms but hand-wavy mechanistically or offers only a weak/generic test.
   4 = Mechanistically plausible within established domain science and yields a clear falsifiable prediction.
   5 = Domain-expert-grade mechanistic realism with a sharp, directly discriminating falsifiable test.

5. Canonical Extraction (for blind clustering):
   - "assumption_dropped": 5-12 word phrase naming the core assumption challenged.
   - "mechanism_summary": 1 crisp neutral sentence (15-28 words) summarizing ONLY the core causal mechanism, stripped of rhetorical framing.

Return ONLY a single JSON object with these exact keys:
{{
  "constraint_checks": {{"E1": "CONSISTENT", "E2": "CONSISTENT", "E3": "CONSISTENT", "E4": "CONSISTENT"}},
  "surprise": 4,
  "coherence": 4,
  "realism": 4,
  "assumption_dropped": "...",
  "mechanism_summary": "...",
  "brief_rationale": "1-2 sentences justifying the scores."
}}"""


TAXONOMY_BUILD_TEMPLATE = """You are building a rigorous scientific taxonomy of explanatory frameworks proposed for an empirical anomaly.

=== ANOMALY SUMMARY ===
Title: {title}
Prevailing Paradigm: {prevailing_paradigm}

=== EXTRACTED CANDIDATE MECHANISMS (BLINDED SAMPLE) ===
{sample_list}

Task: Group these mechanisms into a clean, mutually exclusive set of distinct, high-level explanatory frameworks (typically 5 to 14 frameworks, plus "F0: Incoherent / Off-Topic / Non-Explanatory" if needed).
Two hypotheses belong to the SAME framework if and only if they rely on the SAME core causal mechanism or dropped assumption, even if phrased with different vocabulary. Two hypotheses belong to DIFFERENT frameworks if they invoke genuinely distinct causal mechanisms (e.g., circadian dosing timing vs. gut-microbiome drug conversion vs. hospital co-medication interaction are THREE distinct frameworks).

Return ONLY a valid JSON object:
{{
  "frameworks": [
    {{
      "id": "F1",
      "name": "Short 3-6 word mechanistic label",
      "definition": "1 concise sentence defining the core causal mechanism and assumption challenged."
    }}
  ]
}}"""


TAXONOMY_ASSIGN_TEMPLATE = """Assign each blinded scientific hypothesis mechanism below to the SINGLE best-matching explanatory framework from the canonical taxonomy for this anomaly. If a mechanism genuinely represents a clear scientific framework NOT covered by any listed framework, assign "F_OTHER"; if it is incoherent/empty/refusal, assign "F0".

=== ANOMALY ===
{title}

=== CANONICAL FRAMEWORKS ===
{frameworks_block}

=== HYPOTHESES TO CLASSIFY ===
{items_block}

Return ONLY a valid JSON object mapping each item_id string to its framework ID string (e.g. "F1", "F2", "F_OTHER", "F0"):
{{
  "assignments": {{
    "item_0": "F1"
  }}
}}"""


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract the first valid JSON object from model text."""
    if not text:
        return None
    cleaned = text.strip()
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Scan for balanced braces
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start : i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
        start = cleaned.find("{", start + 1)
    return None


def summarize_hypothesis_for_icr(text: str) -> str:
    """Extract concise 1-2 sentence mechanism summary from a structured hypothesis for ICR exclusion."""
    if not text:
        return "(empty)"
    # Try to pull Assumption Challenged + first sentence of Core Hypothesis
    m_assump = re.search(
        r"###\s*Assumption Challenged\s*(.+?)(?=###|\Z)", text, flags=re.IGNORECASE | re.DOTALL
    )
    m_mech = re.search(
        r"###\s*Core Hypothesis.*?\n(.+?)(?=###|\Z)", text, flags=re.IGNORECASE | re.DOTALL
    )
    parts = []
    if m_assump:
        a = " ".join(m_assump.group(1).strip().split())
        parts.append(a[:180])
    if m_mech:
        mc = " ".join(m_mech.group(1).strip().split())
        # Take first 1-2 sentences up to 220 chars
        parts.append(mc[:240])
    if parts:
        return " | ".join(parts)
    flat = " ".join(text.strip().split())
    return flat[:280]
