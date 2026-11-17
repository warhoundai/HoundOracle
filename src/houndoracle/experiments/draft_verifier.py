"""Local reasoning-first comparison; drafts never become independent evidence."""

import hashlib
import json
import time

from .generative import INSTRUCTIONS, SCHEMA, parse_output

COMMON = (
    INSTRUCTIONS.split("Return JSON")[0]
    + "Use only the supplied evidence, not memorized outside facts. "
)
FORMAT = "Return only JSON with decision YES/NO/MAYBE/INVALID and evidence_indices (zero-based passage indices). YES/NO requires sufficient cited evidence; otherwise MAYBE with []. No explanation or extra keys."


def predict(generate, question, evidence, *, mode, observer=None):
    if mode not in ["direct", "reasoning-first"]:
        raise ValueError("Unknown fixed comparison mode")
    start = time.perf_counter()
    if not evidence or not any(e.get("text", "").strip() for e in evidence):
        return {
            "candidate": "MAYBE",
            "failure": None,
            "evidence_indices": [],
            "calibrated_correctness": None,
            "production_qualified": False,
            "latency_seconds": time.perf_counter() - start,
            "usage": [],
        }
    payload = json.dumps(
        {"proposition": question, "permitted_evidence": evidence}, ensure_ascii=False
    )
    original = COMMON + "\n" + payload
    usage = []
    draft_hash = None
    truncated = False
    if mode == "reasoning-first":
        draft = generate(
            phase="analysis",
            messages=[
                {
                    "role": "user",
                    "content": original
                    + "\nCheck entity identity, temporal scope, quantities, comparisons and negation. Identify what the passages establish, contradict or leave missing. Analyze before deciding; the final JSON will be requested separately.",
                }
            ],
            max_tokens=1024,
            temperature=0.6,
            top_p=0.95,
            seed=1989,
        )
        text = draft["choices"][0]["message"].get("content")
        if not isinstance(text, str) or not text.strip():
            return {
                "candidate": "MAYBE",
                "failure": "EMPTY_DRAFT",
                "evidence_indices": [],
                "production_qualified": False,
                "calibrated_correctness": None,
                "latency_seconds": time.perf_counter() - start,
                "usage": [draft.get("usage", {})],
            }
        usage.append(draft.get("usage", {}))
        draft_hash = hashlib.sha256(text.encode()).hexdigest()
        truncated = draft["choices"][0].get("finish_reason") == "length" or draft.get(
            "time_stopped", False
        )
        if observer:
            observer(text)
        final_messages = [
            {"role": "user", "content": original},
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": original
                + "\nYour prior analysis is not an evidence source. Correct unsupported inferences. "
                + FORMAT,
            },
        ]
    else:
        final_messages = [{"role": "user", "content": original + "\n" + FORMAT}]
    result = generate(
        phase="final",
        messages=final_messages,
        max_tokens=128,
        temperature=0,
        top_p=1,
        seed=1989,
        response_format={"type": "json_object", "schema": SCHEMA},
    )
    usage.append(result.get("usage", {}))
    parsed = parse_output(result["choices"][0]["message"].get("content"), len(evidence))
    return parsed | {
        "latency_seconds": time.perf_counter() - start,
        "usage": usage,
        "draft_sha256": draft_hash,
        "reasoning_truncated": truncated,
        "final_time_stopped": result.get("time_stopped", False),
        "calibrated_correctness": None,
        "production_qualified": False,
    }
