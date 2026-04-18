"""Schema validation and roundtrip checking for training samples."""
from __future__ import annotations

from typing import Dict, Any, Tuple
import json
import re
import math

import jsonschema

from .schemas import SCHEMAS
from .executor import execute, lane_of


_LANE_TAG = re.compile(r"<lane>([^<]+)</lane>")
_TOOL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def validate_schema(tool_call: Dict[str, Any], lane: str = None) -> bool:
    """Validate a tool_call dict against its Lane's JSON Schema."""
    if lane is None:
        lane = lane_of(tool_call)
    schema = SCHEMAS.get(lane)
    if schema is None:
        return False
    try:
        jsonschema.validate(tool_call, schema)
        return True
    except jsonschema.ValidationError:
        return False


def extract_from_messages(messages):
    """Pull (lane, tool_call_dict, tool_result_dict) from a messages array."""
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
    """Shallow structural comparison of expected subset ⊆ actual."""
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


def roundtrip(sample: Dict[str, Any]) -> bool:
    """NL → tool_call → solver → expected-result match.

    sample: {"messages": [...]}  following plan.md §6.3.
    Returns True iff:
      (1) schema validates, (2) executor returns no error,
      (3) executor result is a superset of expected tool content.
    """
    messages = sample.get("messages", [])
    lane, tool_call, tool_result = extract_from_messages(messages)
    if tool_call is None or tool_result is None:
        # Direct-generation lanes (L06-L10) have no tool_call -> pass.
        return sample.get("lane", "").startswith("L06") or sample.get("lane", "").startswith("L07") or \
               sample.get("lane", "").startswith("L08") or sample.get("lane", "").startswith("L09") or \
               sample.get("lane", "").startswith("L10") or lane is None

    if not validate_schema(tool_call, lane):
        return False
    result = execute(tool_call)
    if "error" in result:
        return False
    return _compare(tool_result, result)


def roundtrip_detailed(sample: Dict[str, Any]) -> Tuple[bool, str]:
    messages = sample.get("messages", [])
    lane, tool_call, tool_result = extract_from_messages(messages)
    if tool_call is None or tool_result is None:
        return True, "direct-generation (no tool_call)"
    if not validate_schema(tool_call, lane):
        return False, "schema failed"
    result = execute(tool_call)
    if "error" in result:
        return False, f"executor error: {result.get('error')}"
    ok = _compare(tool_result, result)
    return ok, "ok" if ok else f"mismatch: expected={tool_result}, got={result}"
