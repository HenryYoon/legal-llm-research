"""Basic per-lane unit tests (pytest)."""
import json
import pytest

from solver.executor import execute
from solver.validator import validate_schema, roundtrip


# ---------- L01 ----------

def test_l01_all_met():
    tc = {
        "tool": "z3", "method": "check_elements",
        "elements": ["타인의 재물", "절취"],
        "facts": ["편의점 소유 음료수", "주머니에 넣고 퇴장"],
        "matching": {"타인의 재물": True, "절취": True},
        "mode": "and",
    }
    assert validate_schema(tc, "L01_element_matching")
    out = execute(tc)
    assert out["all_met"] is True
    assert out["details"]["절취"] is True


def test_l01_partial_fails_and():
    tc = {
        "tool": "z3", "method": "check_elements",
        "elements": ["유형력 행사", "사람의 신체에"],
        "facts": ["말다툼 중 멱살을 잡음"],
        "matching": {"유형력 행사": True, "사람의 신체에": False},
        "mode": "and",
    }
    out = execute(tc)
    assert out["all_met"] is False


# ---------- L02 ----------

def test_l02_rule_fires():
    tc = {
        "tool": "z3", "method": "apply_rule",
        "rule": {"antecedents": ["고의", "위법성"], "consequent": "형사책임", "mode": "and"},
        "conditions": {"고의": True, "위법성": True},
    }
    assert validate_schema(tc, "L02_rule_application")
    out = execute(tc)
    assert out["rule_fired"] is True
    assert out["conclusion"]["형사책임"] is True


def test_l02_rule_not_fires():
    tc = {
        "tool": "z3", "method": "apply_rule",
        "rule": {"antecedents": ["계약체결", "채무불이행"], "consequent": "손해배상청구"},
        "conditions": {"계약체결": True, "채무불이행": False},
    }
    out = execute(tc)
    assert out["rule_fired"] is False


# ---------- L03 ----------

def test_l03_subsume_met():
    tc = {
        "tool": "z3", "method": "subsume",
        "norm": {"name": "형법 제250조(살인)", "elements": ["사람을", "살해"]},
        "facts": ["피해자 A", "칼로 찔러 사망케 함"],
        "subsumption": {"사람을": True, "살해": True},
    }
    assert validate_schema(tc, "L03_subsumption")
    out = execute(tc)
    assert out["holding"] is True


def test_l03_subsume_fails():
    tc = {
        "tool": "z3", "method": "subsume",
        "norm": {"name": "형법 제329조(절도)", "elements": ["타인의 재물", "절취", "불법영득의사"]},
        "facts": ["자신의 물건을 가져감"],
        "subsumption": {"타인의 재물": False, "절취": True, "불법영득의사": False},
    }
    out = execute(tc)
    assert out["holding"] is False


# ---------- L04 ----------

def test_l04_valid_modus_ponens():
    tc = {
        "tool": "z3", "method": "check_validity",
        "variables": ["P", "Q"],
        "premises": ["P => Q", "P"],
        "conclusion": "Q",
    }
    assert validate_schema(tc, "L04_logic_judgment")
    out = execute(tc)
    assert out["valid"] is True


def test_l04_invalid_affirming_consequent():
    tc = {
        "tool": "z3", "method": "check_validity",
        "variables": ["P", "Q"],
        "premises": ["P => Q", "Q"],
        "conclusion": "P",
    }
    out = execute(tc)
    assert out["valid"] is False
    assert out["countermodel"] is not None


# ---------- L05 ----------

def test_l05_eval_damages():
    tc = {
        "tool": "sympy", "method": "eval",
        "expr": "principal*rate*years",
        "vars": {"principal": 10000000, "rate": 0.05, "years": 3},
        "round_to": 2,
    }
    assert validate_schema(tc, "L05_calculation")
    out = execute(tc)
    assert abs(out["value"] - 1500000.0) < 1e-3


def test_l05_solve_target():
    tc = {
        "tool": "sympy", "method": "solve",
        "expr": "principal*rate*years - 600000",
        "vars": {"principal": 10000000, "rate": 0.06},
        "target": "years",
    }
    out = execute(tc)
    assert abs(out["solutions"][0] - 1.0) < 1e-4


# ---------- roundtrip ----------

def test_roundtrip_l01_sample():
    sample = {
        "messages": [
            {"role": "system", "content": "legal AI"},
            {"role": "user", "content": "절도죄 성립 여부?"},
            {"role": "assistant", "content":
                "<lane>L01_element_matching</lane>\n"
                "<tool_call>" + json.dumps({
                    "tool": "z3", "method": "check_elements",
                    "elements": ["타인의 재물", "절취"],
                    "facts": ["편의점 음료"],
                    "matching": {"타인의 재물": True, "절취": True},
                    "mode": "and",
                }, ensure_ascii=False) + "</tool_call>"},
            {"role": "tool", "content": json.dumps({
                "all_met": True,
                "details": {"타인의 재물": True, "절취": True},
            }, ensure_ascii=False)},
            {"role": "assistant", "content": "절도죄 성립."},
        ]
    }
    assert roundtrip(sample) is True
