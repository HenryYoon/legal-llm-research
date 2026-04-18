"""JSON Schemas for R2 Lane-specific tool_call payloads (5 lanes, 2 tools).

R2 design: Lane 10 -> 5, tools: z3.check_elements + sympy.calc only.
Schema-outside methods are strictly rejected.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Core tool call schemas (R2)
# ---------------------------------------------------------------------------

# L01 schema: z3.check_elements
_L01_SCHEMA = {
    "type": "object",
    "required": ["tool", "method", "elements", "facts", "matching"],
    "properties": {
        "tool": {"const": "z3"},
        "method": {"const": "check_elements"},
        "elements": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "facts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "matching": {
            "type": "object",
            "additionalProperties": {"type": "boolean"},
        },
        "mode": {"enum": ["and", "or"], "default": "and"},
    },
    "additionalProperties": False,
}

# L05 schema: sympy.calc  (method must be "calc")
_L05_SCHEMA = {
    "type": "object",
    "required": ["tool", "method", "expr", "vars"],
    "properties": {
        "tool": {"const": "sympy"},
        "method": {"const": "calc"},
        "expr": {"type": "string"},
        "vars": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
        "round_to": {"type": "integer"},
        "target": {"type": "string"},
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# R2 SCHEMAS registry
# ---------------------------------------------------------------------------

SCHEMAS_R2 = {
    # Lane keys match the <lane> tag emitted in messages
    "L01": _L01_SCHEMA,
    "L05": _L05_SCHEMA,
    # L06/L09/L10 are direct-generation lanes — no tool_call schema needed.
    # They are listed here as None so the validator can distinguish
    # "direct-gen lane" from "unknown lane".
    "L06": None,
    "L09": None,
    "L10": None,
}

# Allowed (tool, method) pairs — anything outside is rejected.
ALLOWED_TOOL_METHODS_R2 = {
    ("z3", "check_elements"),
    ("sympy", "calc"),
}

# Lane tags that are valid in R2
VALID_LANES_R2 = list(SCHEMAS_R2.keys())  # ["L01", "L05", "L06", "L09", "L10"]


def get_schema_r2(lane: str):
    """Return the JSON schema for *lane*, or None if direct-generation lane.

    Raises KeyError if lane is not a valid R2 lane.
    """
    if lane not in SCHEMAS_R2:
        raise KeyError(f"Unknown R2 lane: {lane!r}. Valid: {VALID_LANES_R2}")
    return SCHEMAS_R2[lane]


def is_direct_generation_lane(lane: str) -> bool:
    """Return True for lanes that do not use tool_call (L06, L09, L10)."""
    return SCHEMAS_R2.get(lane) is None and lane in SCHEMAS_R2


def lane_needs_tool_call(lane: str) -> bool:
    """Return True if this lane must produce a tool_call block."""
    return lane in SCHEMAS_R2 and SCHEMAS_R2[lane] is not None
