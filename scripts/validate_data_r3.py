#!/usr/bin/env python3
"""R3 data validation script.

Checks:
  1. File exists and is valid JSONL
  2. Sample count ≥ 8000
  3. Language ratio: EN ≥ 70%
  4. Binary Yes/No start rate ≥ 95% for hearsay/PJ/textualism sources
  5. element enum whitelist compliance
  6. Tool_call JSON validity for L01/L05 samples
  7. Average final_answer length
  8. Lane distribution

Writes: reports/data_quality_r3.md
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ELEMENT_WHITELIST = {
    "out_of_court_statement", "offered_for_truth",
    "minimum_contacts", "purposeful_availment",
    "fair_play_and_substantial_justice",
    "dictionary_definition_used", "plain_meaning_invoked",
    "절도죄_구성요건", "점유이탈물횡령", "사기죄_구성요건",
    "소멸시효_완성", "시효중단", "계약해지권", "손해배상_요건",
    "과실_책임", "고의_책임",
}

BINARY_SOURCES = {
    "legalbench_hearsay_r3", "legalbench_pj_r3", "legalbench_textualism_r3",
    "legalbench_hearsay_test_r3", "legalbench_pj_test_r3",
    "legalbench_textualism_test_r3",
    "legalbench_hearsay", "legalbench_pj", "legalbench_textualism",
}


def load_jsonl(path: Path) -> List[Dict]:
    samples = []
    errors = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors += 1
                print(f"  JSON error at line {i+1}: {e}")
    if errors:
        print(f"  WARNING: {errors} JSON decode errors")
    return samples


def get_last_assistant(sample: Dict) -> str:
    """Return content of last assistant message."""
    msgs = [m for m in sample["messages"] if m["role"] == "assistant"]
    return msgs[-1]["content"] if msgs else ""


def is_english(sample: Dict) -> bool:
    sys_msg = next((m for m in sample["messages"] if m["role"] == "system"), None)
    if sys_msg and "You are" in sys_msg["content"]:
        return True
    return False


def validate(data_path: Path) -> Dict[str, Any]:
    print(f"Loading {data_path}...")
    samples = load_jsonl(data_path)
    n = len(samples)
    print(f"  Total samples: {n}")

    results: Dict[str, Any] = {"n_samples": n, "checks": {}}
    checks = results["checks"]

    # 1. Count check
    checks["sample_count"] = {"value": n, "pass": n >= 8000, "target": "≥ 8000"}

    # 2. Language ratio
    en_count = sum(1 for s in samples if is_english(s))
    ko_count = n - en_count
    en_pct = en_count / n * 100 if n else 0
    checks["language_ratio"] = {
        "en": en_count, "ko": ko_count,
        "en_pct": round(en_pct, 1),
        "pass": en_pct >= 70,
        "target": "EN ≥ 70%",
    }

    # 3. Binary Yes/No start rate
    binary_samples = [s for s in samples if s.get("source", "") in BINARY_SOURCES]
    yn_starts = 0
    yn_total = len(binary_samples)
    for s in binary_samples:
        fa = get_last_assistant(s)
        if re.match(r"^(Yes|No)[.\s,;:]", fa):
            yn_starts += 1
    yn_rate = yn_starts / yn_total if yn_total else 0
    checks["binary_yn_rate"] = {
        "yn_starts": yn_starts,
        "yn_total": yn_total,
        "rate": round(yn_rate, 3),
        "pass": yn_rate >= 0.95,
        "target": "≥ 95%",
    }

    # 4. Element whitelist compliance
    l01_samples = [s for s in samples if s.get("lane") == "L01"]
    whitelist_violations = []
    for s in l01_samples:
        for msg in s["messages"]:
            if msg["role"] == "assistant" and "<tool_call>" in msg["content"]:
                m = re.search(r"<tool_call>(.*?)</tool_call>", msg["content"], re.DOTALL)
                if m:
                    try:
                        tc = json.loads(m.group(1))
                        for elem in tc.get("elements", []):
                            if elem not in ELEMENT_WHITELIST:
                                whitelist_violations.append(elem)
                    except json.JSONDecodeError:
                        pass
    checks["element_whitelist"] = {
        "violations": len(whitelist_violations),
        "violation_examples": list(set(whitelist_violations))[:5],
        "pass": len(whitelist_violations) == 0,
        "target": "0 violations",
    }

    # 5. Tool_call JSON validity for L01/L05
    tc_valid = 0
    tc_invalid = 0
    tc_total = 0
    for s in samples:
        if s.get("lane") not in ("L01", "L05"):
            continue
        for msg in s["messages"]:
            if msg["role"] == "assistant" and "<tool_call>" in msg["content"]:
                tc_total += 1
                m = re.search(r"<tool_call>(.*?)</tool_call>", msg["content"], re.DOTALL)
                if m:
                    try:
                        json.loads(m.group(1))
                        tc_valid += 1
                    except json.JSONDecodeError:
                        tc_invalid += 1
    tc_rate = tc_valid / tc_total if tc_total else 0
    checks["tool_call_json_valid"] = {
        "valid": tc_valid, "invalid": tc_invalid, "total": tc_total,
        "rate": round(tc_rate, 3),
        "pass": tc_rate >= 0.95,
        "target": "≥ 95% valid JSON",
    }

    # 6. Final answer average length
    fa_lengths = [len(get_last_assistant(s)) for s in samples]
    avg_len = sum(fa_lengths) / len(fa_lengths) if fa_lengths else 0
    checks["avg_final_answer_length"] = {
        "value": round(avg_len, 1),
        "pass": avg_len >= 60,
        "target": "≥ 60 chars",
    }

    # 7. Lane distribution
    lane_dist = Counter(s.get("lane", "unknown") for s in samples)
    checks["lane_distribution"] = dict(lane_dist)

    # 8. Source distribution
    source_dist = Counter(s.get("source", "unknown") for s in samples)
    checks["source_distribution"] = dict(source_dist.most_common(15))

    # Overall pass/fail
    critical_checks = [
        "sample_count", "language_ratio", "binary_yn_rate",
        "tool_call_json_valid", "element_whitelist",
    ]
    results["overall_pass"] = all(
        checks[k]["pass"] for k in critical_checks
    )

    return results


def write_report(results: Dict, report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = results["checks"]

    def status(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    lines = [
        "# R3 Data Quality Report",
        "",
        f"**Overall: {'PASS' if results['overall_pass'] else 'FAIL'}**",
        "",
        "## Checks",
        "",
        f"| Check | Value | Target | Status |",
        f"|-------|-------|--------|--------|",
        f"| Sample count | {results['n_samples']} | ≥ 8000 | {status(checks['sample_count']['pass'])} |",
        f"| EN ratio | {checks['language_ratio']['en_pct']}% | ≥ 70% | {status(checks['language_ratio']['pass'])} |",
        f"| Binary Yes/No start | {checks['binary_yn_rate']['rate']:.3f} ({checks['binary_yn_rate']['yn_starts']}/{checks['binary_yn_rate']['yn_total']}) | ≥ 0.95 | {status(checks['binary_yn_rate']['pass'])} |",
        f"| Element whitelist violations | {checks['element_whitelist']['violations']} | 0 | {status(checks['element_whitelist']['pass'])} |",
        f"| Tool_call JSON valid | {checks['tool_call_json_valid']['rate']:.3f} | ≥ 0.95 | {status(checks['tool_call_json_valid']['pass'])} |",
        f"| Avg final_answer length | {checks['avg_final_answer_length']['value']:.1f} chars | ≥ 60 | {status(checks['avg_final_answer_length']['pass'])} |",
        "",
        "## Lane Distribution",
        "",
    ]
    for lane, count in sorted(checks["lane_distribution"].items()):
        pct = count / results["n_samples"] * 100
        lines.append(f"- {lane}: {count} ({pct:.1f}%)")

    lines += [
        "",
        "## Source Distribution (top 15)",
        "",
    ]
    for source, count in checks["source_distribution"].items():
        lines.append(f"- {source}: {count}")

    lines += [
        "",
        "## Language Split",
        "",
        f"- EN: {checks['language_ratio']['en']} ({checks['language_ratio']['en_pct']}%)",
        f"- KO: {checks['language_ratio']['ko']} ({100 - checks['language_ratio']['en_pct']:.1f}%)",
    ]

    if checks["element_whitelist"]["violation_examples"]:
        lines += [
            "",
            "## Element Whitelist Violation Examples",
            "",
        ]
        for v in checks["element_whitelist"]["violation_examples"]:
            lines.append(f"- `{v}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written → {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="R3 data validation")
    parser.add_argument(
        "--data", default="data/sft_r3.jsonl",
        help="Path to SFT JSONL (relative to repo root)"
    )
    parser.add_argument(
        "--report", default="reports/data_quality_r3.md",
        help="Path to write quality report"
    )
    args = parser.parse_args()

    data_path = ROOT / args.data
    report_path = ROOT / args.report

    results = validate(data_path)

    print("\n=== Summary ===")
    for k, v in results["checks"].items():
        if isinstance(v, dict) and "pass" in v:
            mark = "OK" if v["pass"] else "FAIL"
            print(f"  [{mark}] {k}")

    print(f"\nOverall: {'PASS' if results['overall_pass'] else 'FAIL'}")

    write_report(results, report_path)
