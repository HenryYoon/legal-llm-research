"""Z3-backed legal reasoning primitives for Lanes L01~L04.

All functions return plain python dicts (JSON-serializable).
"""
from __future__ import annotations

from typing import Dict, List, Any
import re
import z3


# ---------- helpers ----------

def _sanitize(name: str) -> str:
    """Make a string safe as a Z3 identifier."""
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^\w가-힣]", "_", s)
    return s or "x"


def _bool_vars(names: List[str]) -> Dict[str, z3.BoolRef]:
    return {n: z3.Bool(_sanitize(n)) for n in names}


# ---------- L01: 요건 매칭 ----------

def check_elements(
    elements: List[str],
    facts: List[str],
    matching: Dict[str, bool],
    mode: str = "and",
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate whether legal *elements* are satisfied by *matching*.

    matching: {element_name: bool}   — which elements each fact satisfies.
    mode: "and" (모든 요건) | "or" (택일).
    Returns all_met, details, proof.
    """
    details = {e: bool(matching.get(e, False)) for e in elements}
    solver = z3.Solver()
    bvars = _bool_vars(elements)
    for e, v in details.items():
        solver.add(bvars[e] == z3.BoolVal(v))
    expr = z3.And(*[bvars[e] for e in elements]) if mode == "and" else z3.Or(*[bvars[e] for e in elements])
    solver.add(expr)
    sat = solver.check() == z3.sat
    return {
        "lane": "L01_element_matching",
        "all_met": bool(sat),
        "mode": mode,
        "details": details,
        "elements": elements,
        "facts": facts,
    }


# ---------- L02: 규칙 적용 ----------

def apply_rule(
    rule: Dict[str, Any],
    conditions: Dict[str, bool],
    **kwargs,
) -> Dict[str, Any]:
    """Rule: (∧/∨ antecedents) → consequent.  Decide consequent given conditions."""
    ants: List[str] = rule["antecedents"]
    conseq: str = rule["consequent"]
    mode = rule.get("mode", "and")

    vs = _bool_vars(ants + [conseq])
    s = z3.Solver()
    for a in ants:
        s.add(vs[a] == z3.BoolVal(bool(conditions.get(a, False))))
    ant_expr = z3.And(*[vs[a] for a in ants]) if mode == "and" else z3.Or(*[vs[a] for a in ants])
    s.add(z3.Implies(ant_expr, vs[conseq]))
    # fire: if antecedents hold the consequent must be true
    fire = all(conditions.get(a, False) for a in ants) if mode == "and" else any(conditions.get(a, False) for a in ants)
    conclusion = bool(fire)
    return {
        "lane": "L02_rule_application",
        "rule_fired": conclusion,
        "conclusion": {conseq: conclusion},
        "antecedents": {a: bool(conditions.get(a, False)) for a in ants},
        "mode": mode,
    }


# ---------- L03: 포섭 ----------

def subsume(
    norm: Dict[str, Any],
    facts: List[str],
    subsumption: Dict[str, bool],
    **kwargs,
) -> Dict[str, Any]:
    """Subsume facts under norm.elements (treated as AND)."""
    elems = norm["elements"]
    details = {e: bool(subsumption.get(e, False)) for e in elems}
    vs = _bool_vars(elems)
    s = z3.Solver()
    for e, v in details.items():
        s.add(vs[e] == z3.BoolVal(v))
    s.add(z3.And(*[vs[e] for e in elems]))
    holding = s.check() == z3.sat
    return {
        "lane": "L03_subsumption",
        "norm": norm.get("name", ""),
        "holding": bool(holding),
        "details": details,
        "facts": facts,
    }


# ---------- L04: 논리 판단 ----------

_TOKEN_RE = re.compile(r"\s*(=>|<=>|&&|\|\||!|\(|\)|and|or|not|implies|iff|[A-Za-z_가-힣][\w가-힣]*)")

def _tokenize(expr: str):
    pos = 0
    toks = []
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            pos += 1
            continue
        tok = m.group(1)
        toks.append(tok)
        pos = m.end()
    return toks


class _Parser:
    """Tiny recursive-descent parser for propositional logic.

    Grammar:
        expr    := iff
        iff     := impl ( '<=>' impl )*
        impl    := or_ ( '=>' or_ )*        right-assoc
        or_     := and_ ( ('||'|'or') and_ )*
        and_    := not_ ( ('&&'|'and') not_ )*
        not_    := ('!'|'not') not_ | atom
        atom    := IDENT | '(' expr ')'
    """

    def __init__(self, tokens, env):
        self.toks = tokens
        self.i = 0
        self.env = env  # name -> z3 bool

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def eat(self, *expected):
        t = self.peek()
        if expected and t not in expected:
            raise ValueError(f"expected {expected}, got {t}")
        self.i += 1
        return t

    def parse(self):
        out = self.iff()
        if self.i != len(self.toks):
            raise ValueError(f"trailing tokens: {self.toks[self.i:]}")
        return out

    def iff(self):
        left = self.impl()
        while self.peek() == "<=>":
            self.eat()
            right = self.impl()
            left = left == right
        return left

    def impl(self):
        left = self.or_()
        if self.peek() in ("=>", "implies"):
            self.eat()
            right = self.impl()
            return z3.Implies(left, right)
        return left

    def or_(self):
        left = self.and_()
        while self.peek() in ("||", "or"):
            self.eat()
            right = self.and_()
            left = z3.Or(left, right)
        return left

    def and_(self):
        left = self.not_()
        while self.peek() in ("&&", "and"):
            self.eat()
            right = self.not_()
            left = z3.And(left, right)
        return left

    def not_(self):
        if self.peek() in ("!", "not"):
            self.eat()
            return z3.Not(self.not_())
        return self.atom()

    def atom(self):
        t = self.peek()
        if t == "(":
            self.eat("(")
            e = self.iff()
            self.eat(")")
            return e
        if t is None or not re.match(r"[A-Za-z_가-힣]", t):
            raise ValueError(f"unexpected token: {t}")
        self.eat()
        if t not in self.env:
            self.env[t] = z3.Bool(t)
        return self.env[t]


def _parse_formula(expr: str, env: Dict[str, z3.BoolRef]):
    toks = _tokenize(expr)
    return _Parser(toks, env).parse()


def check_validity(
    premises: List[str],
    conclusion: str,
    variables: List[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Decide whether (premises) |= conclusion propositionally.

    Implementation: check that (∧ premises) ∧ ¬conclusion is UNSAT.
    """
    env: Dict[str, z3.BoolRef] = {}
    if variables:
        env.update(_bool_vars(variables))
    prem_z3 = [_parse_formula(p, env) for p in premises]
    concl_z3 = _parse_formula(conclusion, env)
    s = z3.Solver()
    s.add(z3.And(*prem_z3) if prem_z3 else z3.BoolVal(True))
    s.add(z3.Not(concl_z3))
    result = s.check()
    valid = result == z3.unsat
    counter = None
    if not valid and result == z3.sat:
        m = s.model()
        counter = {str(d): bool(m[d]) for d in m.decls()}
    return {
        "lane": "L04_logic_judgment",
        "valid": bool(valid),
        "premises": premises,
        "conclusion": conclusion,
        "countermodel": counter,
    }
