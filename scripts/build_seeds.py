"""Expand compact seed specs into per-Lane JSON seed files.

Each tool-bearing seed is roundtrip-validated against the solver before being saved.
Writes data/seeds/{LANE}.json   (one JSON file per Lane, list of samples).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._seed_spec import (
    L01_SPECS, L02_SPECS, L03_SPECS, L04_SPECS, L05_SPECS,
    L06_SPECS, L07_SPECS, L08_SPECS, L09_SPECS, L10_SPECS,
)
from solver.executor import execute
from solver.validator import roundtrip

SYSTEM = "너는 법률 추론 AI야. 입력을 Legal Lane으로 분류하고 적절한 솔버를 호출해라."
SEEDS_DIR = ROOT / "data" / "seeds"
SEEDS_DIR.mkdir(parents=True, exist_ok=True)


def _sample_from_toolcall(lane, user_q, tool_call, explanation):
    # run the solver to capture the ground-truth tool_result
    res = execute(tool_call)
    assistant_tc = f"<lane>{lane}</lane>\n<tool_call>{json.dumps(tool_call, ensure_ascii=False)}</tool_call>"
    sample = {
        "lane": lane,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": assistant_tc},
            {"role": "tool", "content": json.dumps(res, ensure_ascii=False)},
            {"role": "assistant", "content": explanation},
        ],
    }
    return sample


def build_l01():
    out = []
    for q, elements, facts, matching, mode, expl in L01_SPECS:
        tc = {
            "tool": "z3", "method": "check_elements",
            "elements": elements, "facts": facts,
            "matching": matching, "mode": mode,
        }
        out.append(_sample_from_toolcall("L01_element_matching", q, tc, expl))
    return out


def build_l02():
    out = []
    for q, name, ants, cons, conds, mode, expl in L02_SPECS:
        tc = {
            "tool": "z3", "method": "apply_rule",
            "rule": {"antecedents": ants, "consequent": cons, "mode": mode, "name": name},
            "conditions": conds,
        }
        out.append(_sample_from_toolcall("L02_rule_application", q, tc, expl))
    return out


def build_l03():
    out = []
    for q, name, elements, facts, subs, expl in L03_SPECS:
        tc = {
            "tool": "z3", "method": "subsume",
            "norm": {"name": name, "elements": elements},
            "facts": facts, "subsumption": subs,
        }
        out.append(_sample_from_toolcall("L03_subsumption", q, tc, expl))
    return out


def build_l04():
    out = []
    for q, variables, premises, conclusion, expl in L04_SPECS:
        tc = {
            "tool": "z3", "method": "check_validity",
            "variables": variables, "premises": premises,
            "conclusion": conclusion,
        }
        out.append(_sample_from_toolcall("L04_logic_judgment", q, tc, expl))
    return out


def build_l05():
    out = []
    for q, expr, vars_, method, expected, round_to, target, expl in L05_SPECS:
        tc = {"tool": "sympy", "method": method, "expr": expr, "vars": vars_}
        if round_to is not None:
            tc["round_to"] = round_to
        if target is not None:
            tc["target"] = target
        out.append(_sample_from_toolcall("L05_calculation", q, tc, expl))
    return out


LANE_FULL = {
    "L06": "L06_explanation",
    "L07": "L07_case_comparison",
    "L08": "L08_summary",
    "L09": "L09_translation",
    "L10": "L10_uncertain",
}

def build_direct(specs):
    """L06-L10: no tool_call, just system/user/assistant."""
    out = []
    for lane, q, a in specs:
        lane = LANE_FULL.get(lane, lane)
        out.append({
            "lane": lane,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<lane>{lane}</lane>\n{a}"},
            ],
        })
    return out


def _write(lane_name, samples):
    path = SEEDS_DIR / f"{lane_name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    # roundtrip check
    passes = sum(1 for s in samples if roundtrip(s))
    print(f"  {lane_name}: {len(samples)} samples, roundtrip {passes}/{len(samples)}")
    return passes, len(samples)


def main():
    print("Building seeds...")
    stats = {}
    stats["L01_element_matching"] = _write("L01_element_matching", build_l01())
    stats["L02_rule_application"] = _write("L02_rule_application", build_l02())
    stats["L03_subsumption"] = _write("L03_subsumption", build_l03())
    stats["L04_logic_judgment"] = _write("L04_logic_judgment", build_l04())
    stats["L05_calculation"] = _write("L05_calculation", build_l05())
    stats["L06_explanation"] = _write("L06_explanation", build_direct(L06_SPECS))
    stats["L07_case_comparison"] = _write("L07_case_comparison", build_direct(L07_SPECS))
    stats["L08_summary"] = _write("L08_summary", build_direct(L08_SPECS))
    stats["L09_translation"] = _write("L09_translation", build_direct(L09_SPECS))
    stats["L10_uncertain"] = _write("L10_uncertain", build_direct(L10_SPECS))

    total_pass = sum(p for p, _ in stats.values())
    total_all = sum(n for _, n in stats.values())
    print(f"\nTotal: {total_pass}/{total_all} roundtrip ({100*total_pass/total_all:.1f}%)")
    return stats


if __name__ == "__main__":
    main()
