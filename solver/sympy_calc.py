"""SymPy-based L05 legal calculation (damages, interest, duration)."""
from __future__ import annotations

from typing import Dict, Any
import sympy as sp


def calc(
    expr: str,
    vars: Dict[str, float],
    method: str = "eval",
    round_to: int = 2,
    target: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate or solve a symbolic expression.

    method="eval": substitute vars and evaluate `expr` numerically.
        e.g. expr="principal*rate*years", vars={principal:1e7, rate:0.05, years:3}
    method="solve": solve `expr = 0` for `target`.
    """
    symbols = {name: sp.Symbol(name) for name in (vars or {}).keys()}
    if target and target not in symbols:
        symbols[target] = sp.Symbol(target)
    parsed = sp.sympify(expr, locals=symbols)

    if method == "eval":
        subs = {symbols[k]: v for k, v in (vars or {}).items()}
        value = float(parsed.evalf(subs=subs))
        rounded = round(value, round_to) if round_to is not None else value
        return {
            "lane": "L05_calculation",
            "method": "eval",
            "expr": expr,
            "value": rounded,
            "raw": value,
        }

    if method == "solve":
        if not target:
            raise ValueError("solve requires 'target'")
        subs = {symbols[k]: v for k, v in (vars or {}).items() if k != target}
        equation = parsed.subs(subs)
        sols = sp.solve(equation, symbols[target])
        sol_vals = [float(s.evalf()) if s.is_number else str(s) for s in sols]
        return {
            "lane": "L05_calculation",
            "method": "solve",
            "target": target,
            "solutions": sol_vals,
        }

    raise ValueError(f"unknown method: {method}")
