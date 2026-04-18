"""R2 executor: dispatch tool_call dicts to solver backends.

R2 restriction: only z3.check_elements and sympy.calc are valid.
All other (tool, method) combinations return an error dict.
"""
from __future__ import annotations

from typing import Dict, Any
from . import z3_legal, sympy_calc


def execute_r2(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch an R2 tool_call JSON dict to the appropriate solver.

    Allowed:
      {"tool": "z3",    "method": "check_elements", ...}
      {"tool": "sympy", "method": "calc",            ...}

    Everything else returns an error dict (does NOT raise).
    """
    tool = tool_call.get("tool")
    method = tool_call.get("method")

    try:
        if tool == "z3":
            if method == "check_elements":
                return z3_legal.check_elements(**tool_call)
            # Any other z3 method is rejected in R2
            return {
                "error": f"R2: disallowed z3 method {method!r}. Only 'check_elements' allowed.",
                "error_type": "DisallowedMethod",
                "tool_call": tool_call,
            }

        if tool == "sympy":
            if method == "calc":
                # R2 uses method="calc"; internally dispatch to eval or solve
                # based on presence of "target" key.
                kw = {k: v for k, v in tool_call.items() if k not in ("tool", "method")}
                internal_method = "solve" if kw.get("target") else "eval"
                return sympy_calc.calc(method=internal_method, **kw)
            return {
                "error": f"R2: disallowed sympy method {method!r}. Only 'calc' allowed.",
                "error_type": "DisallowedMethod",
                "tool_call": tool_call,
            }

        return {
            "error": f"R2: unknown tool {tool!r}. Allowed: 'z3', 'sympy'.",
            "error_type": "UnknownTool",
            "tool_call": tool_call,
        }

    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "tool_call": tool_call,
        }


def lane_of_r2(tool_call: Dict[str, Any]) -> str:
    """Infer R2 lane from (tool, method) pair."""
    tool = tool_call.get("tool")
    method = tool_call.get("method")
    mapping = {
        ("z3", "check_elements"): "L01",
        ("sympy", "calc"): "L05",
    }
    return mapping.get((tool, method), "unknown")
