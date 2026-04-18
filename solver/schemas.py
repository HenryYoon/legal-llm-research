"""JSON Schemas for Lane-specific tool_call payloads (L01~L05)."""
from __future__ import annotations

SCHEMAS = {
    # L01 — 요건 매칭: elements[], facts[], matching{element:bool}, mode(and|or)
    "L01_element_matching": {
        "type": "object",
        "required": ["tool", "method", "elements", "facts", "matching"],
        "properties": {
            "tool": {"const": "z3"},
            "method": {"const": "check_elements"},
            "elements": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "facts": {"type": "array", "items": {"type": "string"}},
            "matching": {
                "type": "object",
                "additionalProperties": {"type": "boolean"},
            },
            "mode": {"enum": ["and", "or"], "default": "and"},
        },
        "additionalProperties": True,
    },

    # L02 — 규칙 적용: conditions{cond:bool}, rule(list of condition names), conclusion_name
    "L02_rule_application": {
        "type": "object",
        "required": ["tool", "method", "rule", "conditions"],
        "properties": {
            "tool": {"const": "z3"},
            "method": {"const": "apply_rule"},
            "rule": {
                "type": "object",
                "required": ["antecedents", "consequent"],
                "properties": {
                    "antecedents": {"type": "array", "items": {"type": "string"}},
                    "consequent": {"type": "string"},
                    "mode": {"enum": ["and", "or"], "default": "and"},
                },
            },
            "conditions": {
                "type": "object",
                "additionalProperties": {"type": "boolean"},
            },
        },
        "additionalProperties": True,
    },

    # L03 — 포섭: 사실 → 규범 요건의 포섭 판단. norm with elements + fact_map.
    "L03_subsumption": {
        "type": "object",
        "required": ["tool", "method", "norm", "facts", "subsumption"],
        "properties": {
            "tool": {"const": "z3"},
            "method": {"const": "subsume"},
            "norm": {
                "type": "object",
                "required": ["name", "elements"],
                "properties": {
                    "name": {"type": "string"},
                    "elements": {"type": "array", "items": {"type": "string"}},
                },
            },
            "facts": {"type": "array", "items": {"type": "string"}},
            "subsumption": {
                "type": "object",
                "additionalProperties": {"type": "boolean"},
            },
        },
        "additionalProperties": True,
    },

    # L04 — 논리 판단: premises + conclusion, validity via Z3 unsat(¬(P→C)).
    "L04_logic_judgment": {
        "type": "object",
        "required": ["tool", "method", "premises", "conclusion"],
        "properties": {
            "tool": {"const": "z3"},
            "method": {"const": "check_validity"},
            "variables": {"type": "array", "items": {"type": "string"}},
            "premises": {"type": "array", "items": {"type": "string"}},
            "conclusion": {"type": "string"},
        },
        "additionalProperties": True,
    },

    # L05 — 계산: SymPy expression with variable bindings.
    "L05_calculation": {
        "type": "object",
        "required": ["tool", "method", "expr", "vars"],
        "properties": {
            "tool": {"const": "sympy"},
            "method": {"enum": ["eval", "solve"]},
            "expr": {"type": "string"},
            "vars": {
                "type": "object",
                "additionalProperties": {"type": "number"},
            },
            "round_to": {"type": "integer"},
            "target": {"type": "string"},  # for solve
        },
        "additionalProperties": True,
    },
}


LANE_NAMES = list(SCHEMAS.keys())


def get_schema(lane: str):
    return SCHEMAS.get(lane)
