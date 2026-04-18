"""Dispatch tool_call dicts to the appropriate solver backend."""
from __future__ import annotations

from typing import Dict, Any
from . import z3_legal, sympy_calc


def execute(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool_call JSON dict -> solver -> result dict.

    Expected keys: tool ("z3"|"sympy"), method, plus lane-specific payload.
    """
    tool = tool_call.get("tool")
    method = tool_call.get("method")

    try:
        if tool == "z3":
            if method == "check_elements":
                return z3_legal.check_elements(**tool_call)
            if method == "apply_rule":
                return z3_legal.apply_rule(**tool_call)
            if method == "subsume":
                return z3_legal.subsume(**tool_call)
            if method == "check_validity":
                return z3_legal.check_validity(**tool_call)
            raise ValueError(f"unknown z3 method: {method}")

        if tool == "sympy":
            return sympy_calc.calc(**tool_call)

        raise ValueError(f"unknown tool: {tool}")
    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__, "tool_call": tool_call}


def lane_of(tool_call: Dict[str, Any]) -> str:
    tool = tool_call.get("tool")
    method = tool_call.get("method")
    mapping = {
        ("z3", "check_elements"): "L01_element_matching",
        ("z3", "apply_rule"): "L02_rule_application",
        ("z3", "subsume"): "L03_subsumption",
        ("z3", "check_validity"): "L04_logic_judgment",
        ("sympy", "eval"): "L05_calculation",
        ("sympy", "solve"): "L05_calculation",
    }
    return mapping.get((tool, method), "unknown")
