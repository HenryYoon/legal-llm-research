"""R2 schema validation for tool_call dicts.

Key difference from R1 validator:
- Only z3.check_elements and sympy.calc are allowed.
- Any other (tool, method) pair is a hard FAIL.
- Direct-generation lanes (L06, L09, L10) pass without tool_call.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Optional, Tuple

import jsonschema

from .schemas_r2 import (
    ALLOWED_TOOL_METHODS_R2,
    SCHEMAS_R2,
    VALID_LANES_R2,
    is_direct_generation_lane,
    lane_needs_tool_call,
)
from .executor_r2 import execute_r2

# ── regex helpers ─────────────────────────────────────────────────────────────
_LANE_TAG = re.compile(r"<lane>(L\d{2})</lane>")
_TOOL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


# ── low-level helpers ─────────────────────────────────────────────────────────

def _is_allowed_tool_method(tool_call: Dict[str, Any]) -> bool:
    """Return True iff (tool, method) is in ALLOWED_TOOL_METHODS_R2."""
    pair = (tool_call.get("tool"), tool_call.get("method"))
    return pair in ALLOWED_TOOL_METHODS_R2


def validate_schema_r2(tool_call: Dict[str, Any], lane: Optional[str] = None) -> bool:
    """Validate *tool_call* against the R2 JSON Schema for *lane*.

    Returns False if:
    - (tool, method) not in ALLOWED_TOOL_METHODS_R2, OR
    - lane is not a valid R2 lane, OR
    - lane is a direct-generation lane (no tool_call expected), OR
    - JSON Schema validation fails.
    """
    if not _is_allowed_tool_method(tool_call):
        return False
    if lane is None:
        # Infer from (tool, method)
        tool, method = tool_call.get("tool"), tool_call.get("method")
        if (tool, method) == ("z3", "check_elements"):
            lane = "L01"
        elif (tool, method) == ("sympy", "calc"):
            lane = "L05"
        else:
            return False
    if lane not in SCHEMAS_R2:
        return False
    schema = SCHEMAS_R2[lane]
    if schema is None:
        # Direct-generation lane — tool_call should not appear
        return False
    try:
        jsonschema.validate(tool_call, schema)
        return True
    except jsonschema.ValidationError:
        return False


# ── message extraction ────────────────────────────────────────────────────────

def extract_from_messages_r2(messages) -> Tuple[Optional[str], Optional[Dict], Optional[Dict]]:
    """Extract (lane, tool_call_dict, tool_result_dict) from a messages list."""
    lane = None
    tool_call = None
    tool_result = None
    for m in messages:
        content = m.get("content", "")
        role = m.get("role")
        if role == "assistant":
            m_lane = _LANE_TAG.search(content)
            if m_lane:
                lane = m_lane.group(1).strip()
            m_tc = _TOOL_TAG.search(content)
            if m_tc:
                try:
                    tool_call = json.loads(m_tc.group(1))
                except json.JSONDecodeError:
                    tool_call = None
        elif role == "tool":
            try:
                tool_result = json.loads(content)
            except json.JSONDecodeError:
                tool_result = None
    return lane, tool_call, tool_result


def _compare(expected, actual) -> bool:
    """Shallow structural check: expected ⊆ actual."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(k in actual and _compare(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        return all(_compare(e, a) for e, a in zip(expected, actual))
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return math.isclose(expected, float(actual), rel_tol=1e-4, abs_tol=1e-4)
    return expected == actual


# ── final answer length check ─────────────────────────────────────────────────

def final_answer_length(messages) -> int:
    """Return the char length of the last assistant message.

    For tool-using lanes: the assistant message after the tool result.
    For direct-generation lanes: the last assistant message overall.
    """
    last_assistant = ""
    tool_seen = False
    last_assistant_overall = ""
    for m in messages:
        if m.get("role") == "tool":
            tool_seen = True
        if m.get("role") == "assistant":
            last_assistant_overall = m.get("content", "")
            if tool_seen:
                last_assistant = m.get("content", "")
    # If no tool was seen, use the last assistant message overall
    if not tool_seen:
        return len(last_assistant_overall)
    return len(last_assistant)


# ── roundtrip ─────────────────────────────────────────────────────────────────

def roundtrip_r2(sample: Dict[str, Any]) -> bool:
    """Full R2 roundtrip check.

    For tool-using lanes (L01, L05):
      (1) lane tag must be a valid R2 lane
      (2) tool_call (tool, method) must be in ALLOWED_TOOL_METHODS_R2
      (3) schema validates
      (4) executor returns no error
      (5) executor result is superset of expected tool content
      (6) final answer length >= 30 chars

    For direct-generation lanes (L06, L09, L10):
      (1) no tool_call present (or ignored)
      (2) final answer length >= 30 chars
    """
    ok, _ = roundtrip_r2_detailed(sample)
    return ok


def roundtrip_r2_detailed(sample: Dict[str, Any]) -> Tuple[bool, str]:
    messages = sample.get("messages", [])
    lane, tool_call, tool_result = extract_from_messages_r2(messages)

    # Fall back to sample-level lane field if not found in content
    if lane is None:
        lane = sample.get("lane")

    # Unknown lane
    if lane is not None and lane not in SCHEMAS_R2:
        return False, f"unknown R2 lane: {lane!r}"

    # Direct-generation lanes
    if lane is None or is_direct_generation_lane(lane):
        fa_len = final_answer_length(messages)
        if fa_len < 30:
            return False, f"final answer too short: {fa_len} chars (min 30)"
        return True, "direct-generation lane ok"

    # Tool-using lanes
    if tool_call is None:
        return False, "missing tool_call for tool-using lane"

    # Method hallucination check
    if not _is_allowed_tool_method(tool_call):
        return False, (
            f"disallowed method: tool={tool_call.get('tool')!r}, "
            f"method={tool_call.get('method')!r}"
        )

    if not validate_schema_r2(tool_call, lane):
        return False, "schema validation failed"

    result = execute_r2(tool_call)
    if "error" in result:
        return False, f"executor error: {result['error']}"

    if tool_result is not None and not _compare(tool_result, result):
        return False, f"result mismatch: expected={tool_result}, got={result}"

    fa_len = final_answer_length(messages)
    if fa_len < 30:
        return False, f"final answer too short: {fa_len} chars (min 30)"

    return True, "ok"
