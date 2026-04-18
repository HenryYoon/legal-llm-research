"""R2 pytest suite — 5-lane schema, 2-method enforcement, roundtrip checks."""
from __future__ import annotations

import json
import pytest

from solver.executor_r2 import execute_r2, lane_of_r2
from solver.validator_r2 import validate_schema_r2, roundtrip_r2, roundtrip_r2_detailed
from solver.schemas_r2 import VALID_LANES_R2, ALLOWED_TOOL_METHODS_R2


# ── helpers ───────────────────────────────────────────────────────────────────

def _msg(lane: str, user: str, tool_call_dict: dict, tool_result_dict: dict,
         final_answer: str, system: str = "You are a legal reasoning AI."):
    tc_json = json.dumps(tool_call_dict, ensure_ascii=False)
    tr_json = json.dumps(tool_result_dict, ensure_ascii=False)
    return {
        "lane": lane,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant",
             "content": f"<lane>{lane}</lane>\n<tool_call>{tc_json}</tool_call>"},
            {"role": "tool", "content": tr_json},
            {"role": "assistant", "content": final_answer},
        ],
    }


def _direct_msg(lane: str, user: str, final_answer: str,
                system: str = "You are a legal reasoning AI."):
    return {
        "lane": lane,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": final_answer},
        ],
    }


# ── Test 1: schemas_r2 registry ───────────────────────────────────────────────

def test_valid_lanes_r2():
    assert set(VALID_LANES_R2) == {"L01", "L05", "L06", "L09", "L10"}


def test_allowed_tools():
    assert ("z3", "check_elements") in ALLOWED_TOOL_METHODS_R2
    assert ("sympy", "calc") in ALLOWED_TOOL_METHODS_R2
    assert len(ALLOWED_TOOL_METHODS_R2) == 2


# ── Test 2: L01 schema validation ────────────────────────────────────────────

def test_l01_schema_valid():
    tc = {
        "tool": "z3",
        "method": "check_elements",
        "elements": ["out-of-court statement", "offered for truth", "by declarant"],
        "facts": ["witness testified about defendant's prior statement"],
        "matching": {
            "out-of-court statement": True,
            "offered for truth": True,
            "by declarant": True,
        },
        "mode": "and",
    }
    assert validate_schema_r2(tc, "L01") is True


def test_l01_schema_missing_required():
    """Missing 'matching' field should fail."""
    tc = {
        "tool": "z3",
        "method": "check_elements",
        "elements": ["out-of-court statement"],
        "facts": ["witness said so"],
        # missing matching
    }
    assert validate_schema_r2(tc, "L01") is False


def test_l01_schema_extra_field_rejected():
    """additionalProperties: False — extra keys must fail."""
    tc = {
        "tool": "z3",
        "method": "check_elements",
        "elements": ["purposeful availment"],
        "facts": ["defendant signed contract in forum state"],
        "matching": {"purposeful availment": True},
        "mode": "and",
        "extra_key": "not_allowed",    # <-- should fail
    }
    assert validate_schema_r2(tc, "L01") is False


# ── Test 3: disallowed methods rejected ──────────────────────────────────────

def test_disallowed_z3_method():
    """R2 must reject apply_rule, subsume, check_validity."""
    for bad_method in ["apply_rule", "subsume", "check_validity", "equation_check",
                       "all_values_tac", "contains"]:
        tc = {
            "tool": "z3",
            "method": bad_method,
            "elements": ["A"],
            "facts": ["A is true"],
            "matching": {"A": True},
        }
        assert validate_schema_r2(tc) is False, f"should reject method={bad_method!r}"


def test_disallowed_sympy_method():
    """eval and solve are R1 methods; R2 only allows 'calc'."""
    for bad_method in ["eval", "solve", "simplify", "integrate"]:
        tc = {
            "tool": "sympy",
            "method": bad_method,
            "expr": "principal * rate",
            "vars": {"principal": 1000000, "rate": 0.05},
        }
        assert validate_schema_r2(tc) is False, f"should reject method={bad_method!r}"


# ── Test 4: L05 schema validation ────────────────────────────────────────────

def test_l05_schema_valid():
    tc = {
        "tool": "sympy",
        "method": "calc",
        "expr": "principal * rate * years",
        "vars": {"principal": 10000000, "rate": 0.05, "years": 3},
        "round_to": 0,
    }
    assert validate_schema_r2(tc, "L05") is True


def test_l05_missing_expr():
    tc = {
        "tool": "sympy",
        "method": "calc",
        "vars": {"principal": 1000000},
    }
    assert validate_schema_r2(tc, "L05") is False


# ── Test 5: executor_r2 ───────────────────────────────────────────────────────

def test_execute_r2_l01_all_met():
    tc = {
        "tool": "z3",
        "method": "check_elements",
        "elements": ["out-of-court statement", "offered for truth"],
        "facts": ["A said B"],
        "matching": {"out-of-court statement": True, "offered for truth": True},
        "mode": "and",
    }
    result = execute_r2(tc)
    assert "error" not in result
    assert result["all_met"] is True


def test_execute_r2_l01_not_met():
    tc = {
        "tool": "z3",
        "method": "check_elements",
        "elements": ["purposeful availment", "arising from contacts"],
        "facts": ["defendant advertised nationally"],
        "matching": {"purposeful availment": True, "arising from contacts": False},
        "mode": "and",
    }
    result = execute_r2(tc)
    assert "error" not in result
    assert result["all_met"] is False


def test_execute_r2_l05_eval():
    tc = {
        "tool": "sympy",
        "method": "calc",
        "expr": "principal * rate * years",
        "vars": {"principal": 10000000.0, "rate": 0.05, "years": 3.0},
        "round_to": 2,
    }
    result = execute_r2(tc)
    assert "error" not in result
    assert abs(result["value"] - 1500000.0) < 1.0


def test_execute_r2_rejects_disallowed():
    tc = {
        "tool": "z3",
        "method": "subsume",
        "norm": {"name": "test", "elements": ["A"]},
        "facts": ["B"],
        "subsumption": {"A": True},
    }
    result = execute_r2(tc)
    assert "error" in result
    assert "disallowed" in result["error"].lower()


# ── Test 6: roundtrip_r2 ─────────────────────────────────────────────────────

def test_roundtrip_l01_english():
    """Hearsay-style L01 roundtrip (English)."""
    tc = {
        "tool": "z3", "method": "check_elements",
        "elements": ["out-of-court statement", "offered for truth", "made by declarant"],
        "facts": ["witness testified that defendant said he ran the red light"],
        "matching": {
            "out-of-court statement": True,
            "offered for truth": True,
            "made by declarant": True,
        },
        "mode": "and",
    }
    from solver.executor_r2 import execute_r2 as _exec
    expected = _exec(tc)
    sample = _msg(
        lane="L01",
        user="Is this statement hearsay? Defendant told a witness he ran the red light.",
        tool_call_dict=tc,
        tool_result_dict=expected,
        final_answer=(
            "Yes, this is hearsay under FRE 801. The statement was made out of court "
            "by the declarant (defendant) and is offered to prove the truth of the matter "
            "asserted — that he ran the red light. All three hearsay elements are satisfied."
        ),
    )
    assert roundtrip_r2(sample) is True


def test_roundtrip_l05_korean():
    """L05 calculation roundtrip (Korean)."""
    tc = {
        "tool": "sympy", "method": "calc",
        "expr": "principal * rate * years",
        "vars": {"principal": 50000000.0, "rate": 0.05, "years": 2.0},
        "round_to": 0,
    }
    from solver.executor_r2 import execute_r2 as _exec
    expected = _exec(tc)
    sample = _msg(
        lane="L05",
        user="원금 5천만원, 연이율 5%, 기간 2년으로 단리 이자를 계산하세요.",
        tool_call_dict=tc,
        tool_result_dict=expected,
        final_answer=(
            "단리 이자는 5,000,000원입니다. 계산식: 원금(50,000,000) × 이율(0.05) × 기간(2년) = 5,000,000원. "
            "민법 제379조 기준 법정이율 5%를 적용한 결과입니다."
        ),
    )
    assert roundtrip_r2(sample) is True


def test_roundtrip_direct_l06():
    """L06 direct generation lane passes without tool_call."""
    sample = _direct_msg(
        lane="L06",
        user="What is the difference between hearsay and non-hearsay?",
        final_answer=(
            "Hearsay is an out-of-court statement offered to prove the truth of the matter "
            "asserted (FRE 801). Non-hearsay uses the statement for another purpose, such as "
            "showing notice or legally operative words, regardless of truth."
        ),
    )
    assert roundtrip_r2(sample) is True


def test_roundtrip_fail_disallowed_method():
    """Sample with schema-outside method must fail roundtrip."""
    tc = {
        "tool": "z3", "method": "equation_check",
        "elements": ["A"], "facts": ["B"], "matching": {"A": True}, "mode": "and",
    }
    sample = _msg(
        lane="L01",
        user="Does A apply?",
        tool_call_dict=tc,
        tool_result_dict={"result": True},
        final_answer=(
            "Yes, element A is satisfied. The evidence clearly establishes the required element "
            "under the applicable legal standard."
        ),
    )
    ok, reason = roundtrip_r2_detailed(sample)
    assert ok is False
    assert "disallowed" in reason.lower()


def test_roundtrip_fail_short_final_answer():
    """Short final answer (< 30 chars) must fail."""
    tc = {
        "tool": "z3", "method": "check_elements",
        "elements": ["purposeful availment"],
        "facts": ["defendant shipped goods into state"],
        "matching": {"purposeful availment": True},
        "mode": "and",
    }
    from solver.executor_r2 import execute_r2 as _exec
    expected = _exec(tc)
    sample = _msg(
        lane="L01",
        user="PJ?",
        tool_call_dict=tc,
        tool_result_dict=expected,
        final_answer="Yes.",  # too short
    )
    ok, reason = roundtrip_r2_detailed(sample)
    assert ok is False
    assert "short" in reason.lower()


# ── Test 7: lane_of_r2 ───────────────────────────────────────────────────────

def test_lane_of_r2_mapping():
    assert lane_of_r2({"tool": "z3", "method": "check_elements"}) == "L01"
    assert lane_of_r2({"tool": "sympy", "method": "calc"}) == "L05"
    assert lane_of_r2({"tool": "z3", "method": "apply_rule"}) == "unknown"
    assert lane_of_r2({"tool": "sympy", "method": "eval"}) == "unknown"


# ── Test 8: L09 and L10 direct generation ────────────────────────────────────

def test_roundtrip_l09_translation():
    sample = _direct_msg(
        lane="L09",
        user="Translate 'mens rea' into Korean legal terminology.",
        final_answer=(
            "'Mens rea'는 한국 형법에서 '고의(故意)'에 해당합니다. "
            "범죄 성립을 위해 객관적 구성요건(actus reus)과 함께 요구되는 주관적 요건입니다."
        ),
    )
    assert roundtrip_r2(sample) is True


def test_roundtrip_l10_uncertain():
    sample = _direct_msg(
        lane="L10",
        user="Is this contract enforceable given the ambiguous consideration clause?",
        final_answer=(
            "Uncertain. The consideration clause contains ambiguous language that courts "
            "have interpreted inconsistently. Outcome depends on jurisdiction and the specific "
            "facts presented at trial. Legal counsel is strongly recommended."
        ),
    )
    assert roundtrip_r2(sample) is True
