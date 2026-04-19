"""JSON Schemas for R2/R3 Lane-specific tool_call payloads (5 lanes, 2 tools).

R2 design: Lane 10 -> 5, tools: z3.check_elements + sympy.calc only.
R3 change: elements field uses enum whitelist to prevent hallucination.
Schema-outside methods are strictly rejected.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# R3: Whitelisted element names (P1-B)
# ---------------------------------------------------------------------------

ELEMENT_ENUM_R3 = [
    # English — hearsay (FRE 801(c))
    "out_of_court_statement",
    "offered_for_truth",
    # English — personal jurisdiction
    "minimum_contacts",
    "purposeful_availment",
    "fair_play_and_substantial_justice",
    # English — textualism
    "dictionary_definition_used",
    "plain_meaning_invoked",
    # Korean civil law
    "절도죄_구성요건",
    "점유이탈물횡령",
    "사기죄_구성요건",
    "소멸시효_완성",
    "시효중단",
    "계약해지권",
    "손해배상_요건",
    "과실_책임",
    "고의_책임",
]

# Per-task element whitelist for runtime validation
ELEMENT_WHITELIST_BY_TASK = {
    "hearsay": ["out_of_court_statement", "offered_for_truth"],
    "personal_jurisdiction": [
        "minimum_contacts", "purposeful_availment", "fair_play_and_substantial_justice"
    ],
    "textualism_tool": ["dictionary_definition_used", "plain_meaning_invoked"],
}

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

# ---------------------------------------------------------------------------
# R3: L01 schema with elements enum (whitelist enforced)
# ---------------------------------------------------------------------------

_L01_SCHEMA_R3 = {
    "type": "object",
    "required": ["tool", "method", "elements", "facts", "matching"],
    "properties": {
        "tool": {"const": "z3"},
        "method": {"const": "check_elements"},
        "elements": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ELEMENT_ENUM_R3,
            },
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

SCHEMAS_R3 = {
    "L01": _L01_SCHEMA_R3,
    "L05": _L05_SCHEMA,
    "L06": None,
    "L09": None,
    "L10": None,
}


def get_schema_r3(lane: str):
    """Return the R3 JSON schema for *lane* (with element enum whitelist)."""
    if lane not in SCHEMAS_R3:
        raise KeyError(f"Unknown R3 lane: {lane!r}. Valid: {list(SCHEMAS_R3)}")
    return SCHEMAS_R3[lane]

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
