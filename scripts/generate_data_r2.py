#!/usr/bin/env python3
"""R2 data generation: seed × template augmentation → SFT + DPO datasets.

Output:
  data/sft_r2.jsonl  — L01 4K, L05 1K, L06 3K, L09 500, L10 500 = ~9K
  data/dpo_r2.jsonl  — 10K pairs
                       rejected: 50% bad-method, 30% bad-matching, 20% short-answer

Usage:
  python scripts/generate_data_r2.py [--sft-only] [--dpo-only] [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from solver.validator_r2 import roundtrip_r2_detailed, validate_schema_r2
from solver.executor_r2 import execute_r2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS_DIR = ROOT / "data" / "seeds_r2"
DATA_DIR = ROOT / "data"

SFT_TARGET = {"L01": 4000, "L05": 1000, "L06": 3000, "L09": 500, "L10": 500}
DPO_TARGET = 10000

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Template substitution tables
# ---------------------------------------------------------------------------

NAMES_EN = [
    "Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry",
    "Isabella", "James", "Karen", "Liam", "Mary", "Nathan", "Olivia", "Paul",
    "Quinn", "Rachel", "Samuel", "Tina", "Ursula", "Victor", "Wendy", "Xavier",
    "Yvonne", "Zachary", "Andrea", "Brian", "Chloe", "Derek",
]

NAMES_KO = [
    "김철수", "이영희", "박민준", "최수진", "정동욱", "강지혜", "윤성호",
    "임소연", "오준서", "한미경", "신태양", "류지원", "조민아", "서현준",
    "배나리", "허재원", "남궁민", "전보라", "고태경", "문서진",
]

STATES = [
    "California", "Texas", "New York", "Florida", "Illinois",
    "Pennsylvania", "Ohio", "Georgia", "North Carolina", "Michigan",
    "Arizona", "Colorado", "Washington", "Oregon", "Nevada",
]

AMOUNTS_USD = [
    10000, 25000, 50000, 75000, 100000, 150000, 200000, 250000,
    500000, 750000, 1000000, 2000000, 5000000,
]

AMOUNTS_KRW = [
    1000000, 3000000, 5000000, 10000000, 30000000, 50000000,
    100000000, 200000000, 500000000,
]

RATES = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
YEARS = [1, 2, 3, 5, 7, 10]
MONTHS = [1, 2, 3, 4, 6, 9, 12, 18, 24, 36]

LEGAL_TERMS_EN = [
    "breach of contract", "negligence", "fraud", "misrepresentation",
    "defamation", "assault", "battery", "trespass", "conversion",
    "unjust enrichment", "quantum meruit", "promissory estoppel",
]

DISALLOWED_METHODS = [
    ("z3", "equation_check"),
    ("z3", "apply_rule"),
    ("z3", "subsume"),
    ("z3", "check_validity"),
    ("z3", "contains"),
    ("z3", "all_values_tac"),
    ("sympy", "eval"),
    ("sympy", "solve"),
    ("sympy", "simplify"),
    ("sympy", "integrate"),
    ("sympy", "differentiate"),
]

SHORT_ANSWERS = [
    "Yes.", "No.", "절도죄 성립.", "성립하지 않음.", "Yes, hearsay.",
    "No, not hearsay.", "Uncertain.", "계산 완료.", "Translation done.",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_seeds(lane: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load all seed files, optionally filtered by lane."""
    seeds = []
    for f in sorted(SEEDS_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for item in data:
            if lane is None or item.get("lane") == lane:
                seeds.append(item)
    return seeds


def _detect_language(text: str) -> str:
    """Rough language detection: Korean or English."""
    korean_chars = len(re.findall(r"[가-힣]", text))
    return "ko" if korean_chars > len(text) * 0.05 else "en"


def _replace_name(text: str, new_name: str, lang: str = "en") -> str:
    """Replace the first uppercase name-like token."""
    if lang == "en":
        return re.sub(r"\b[A-Z][a-z]+\b", new_name, text, count=1)
    return text


def _replace_amount_en(text: str, new_amount: int) -> str:
    """Replace a dollar amount in text."""
    return re.sub(r"\$[\d,]+", f"${new_amount:,}", text, count=1)


def _replace_amount_ko(text: str, new_amount: int) -> str:
    """Replace a Korean KRW amount."""
    formatted = f"{new_amount:,}원"
    return re.sub(r"[\d,]+원", formatted, text, count=1)


def _replace_rate(text: str, new_rate: float) -> str:
    """Replace a percentage in text."""
    return re.sub(r"\d+(\.\d+)?%", f"{new_rate*100:.0f}%", text, count=1)


def _replace_years(text: str, new_years: int) -> str:
    """Replace 'N years' or 'N년'."""
    text = re.sub(r"\d+ years?", f"{new_years} years", text, count=1)
    text = re.sub(r"\d+년", f"{new_years}년", text, count=1)
    return text


def _augment_sample(sample: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    """Apply random template substitutions to a seed sample."""
    s = copy.deepcopy(sample)
    messages = s.get("messages", [])
    if not messages:
        return None

    # Detect language from user message
    user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
    lang = _detect_language(user_content)

    # Apply 1-3 random substitutions
    num_subs = rng.randint(1, 3)
    for _ in range(num_subs):
        sub_type = rng.choice(["name", "amount", "rate", "years"])
        for m in messages:
            content = m["content"]
            if sub_type == "name":
                new_name = rng.choice(NAMES_KO if lang == "ko" else NAMES_EN)
                m["content"] = _replace_name(content, new_name, lang)
            elif sub_type == "amount":
                if lang == "ko":
                    m["content"] = _replace_amount_ko(content, rng.choice(AMOUNTS_KRW))
                else:
                    m["content"] = _replace_amount_en(content, rng.choice(AMOUNTS_USD))
            elif sub_type == "rate":
                m["content"] = _replace_rate(content, rng.choice(RATES))
            elif sub_type == "years":
                m["content"] = _replace_years(content, rng.choice(YEARS))

    # Re-execute tool_call to update tool result
    from solver.validator_r2 import extract_from_messages_r2
    lane, tool_call, _ = extract_from_messages_r2(messages)
    if tool_call is not None:
        real_result = execute_r2(tool_call)
        if "error" in real_result:
            return None  # discard
        for m in messages:
            if m["role"] == "tool":
                m["content"] = json.dumps(real_result, ensure_ascii=False)
                break

    # Validate roundtrip
    ok, reason = roundtrip_r2_detailed(s)
    if not ok:
        log.debug("Augmented sample failed roundtrip: %s", reason)
        return None

    return s


def _generate_sft_for_lane(
    lane: str, target: int, rng: random.Random
) -> List[Dict[str, Any]]:
    """Generate SFT samples for one lane via augmentation."""
    seeds = _load_seeds(lane)
    if not seeds:
        log.warning("No seeds found for lane %s", lane)
        return []

    samples = []
    # Include all seeds directly first
    for s in seeds:
        ok, _ = roundtrip_r2_detailed(s)
        if ok:
            samples.append(s)

    attempts = 0
    max_attempts = target * 20
    while len(samples) < target and attempts < max_attempts:
        seed = rng.choice(seeds)
        aug = _augment_sample(seed, rng)
        if aug is not None:
            samples.append(aug)
        attempts += 1

    if len(samples) < target:
        # Oversample by repeating if needed
        while len(samples) < target:
            samples.append(rng.choice(samples[:len(seeds) or 1]))

    rng.shuffle(samples)
    result = samples[:target]
    log.info("Lane %s: generated %d/%d samples (after %d attempts)", lane, len(result), target, attempts)
    return result


# ---------------------------------------------------------------------------
# DPO rejected sample generators
# ---------------------------------------------------------------------------

def _make_rejected_bad_method(
    sample: Dict[str, Any], rng: random.Random
) -> Optional[Dict[str, Any]]:
    """Create a rejected sample with a schema-outside method."""
    from solver.validator_r2 import extract_from_messages_r2
    import re as _re

    s = copy.deepcopy(sample)
    messages = s.get("messages", [])
    lane, tool_call, _ = extract_from_messages_r2(messages)

    if tool_call is None:
        return None  # can't make bad method for direct-gen lanes

    bad_tool, bad_method = rng.choice(DISALLOWED_METHODS)
    bad_tc = dict(tool_call)
    bad_tc["tool"] = bad_tool
    bad_tc["method"] = bad_method
    bad_tc_json = json.dumps(bad_tc, ensure_ascii=False)

    for m in messages:
        if m["role"] == "assistant" and "<tool_call>" in m["content"]:
            m["content"] = _re.sub(
                r"<tool_call>.*?</tool_call>",
                f"<tool_call>{bad_tc_json}</tool_call>",
                m["content"],
                flags=_re.DOTALL,
            )
            break
    # Tool result stays as-is (doesn't matter for rejected)
    return s


def _make_rejected_bad_matching(
    sample: Dict[str, Any], rng: random.Random
) -> Optional[Dict[str, Any]]:
    """Create a rejected sample with wrong element matching values."""
    from solver.validator_r2 import extract_from_messages_r2
    import re as _re

    s = copy.deepcopy(sample)
    messages = s.get("messages", [])
    lane, tool_call, _ = extract_from_messages_r2(messages)

    if tool_call is None or tool_call.get("method") != "check_elements":
        return None

    bad_tc = dict(tool_call)
    matching = dict(bad_tc.get("matching", {}))
    if not matching:
        return None

    # Flip one or two matching values
    keys = list(matching.keys())
    flip_count = rng.randint(1, min(2, len(keys)))
    for k in rng.sample(keys, flip_count):
        matching[k] = not matching[k]
    bad_tc["matching"] = matching

    bad_tc_json = json.dumps(bad_tc, ensure_ascii=False)
    for m in messages:
        if m["role"] == "assistant" and "<tool_call>" in m["content"]:
            m["content"] = _re.sub(
                r"<tool_call>.*?</tool_call>",
                f"<tool_call>{bad_tc_json}</tool_call>",
                m["content"],
                flags=_re.DOTALL,
            )
            break
    return s


def _make_rejected_short_answer(
    sample: Dict[str, Any], rng: random.Random
) -> Optional[Dict[str, Any]]:
    """Create a rejected sample with a too-short final answer."""
    s = copy.deepcopy(sample)
    messages = s.get("messages", [])

    # Replace the last assistant message with a very short answer
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            messages[i]["content"] = rng.choice(SHORT_ANSWERS)
            break
    return s


def _generate_dpo(
    sft_samples: List[Dict[str, Any]],
    target: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Generate DPO pairs from SFT chosen samples + constructed rejecteds."""
    # Filter SFT samples that have tool_call (needed for bad-method/bad-matching)
    tool_samples = [
        s for s in sft_samples
        if any("<tool_call>" in m.get("content", "") for m in s.get("messages", []))
    ]
    all_samples = sft_samples

    pairs = []
    bad_method_target = int(target * 0.50)
    bad_matching_target = int(target * 0.30)
    short_answer_target = target - bad_method_target - bad_matching_target

    def _add_pairs(chosen_pool, rejected_fn, count, label):
        added = 0
        attempts = 0
        while added < count and attempts < count * 10:
            chosen = rng.choice(chosen_pool)
            rejected = rejected_fn(chosen, rng)
            if rejected is not None:
                pairs.append({
                    "chosen": chosen["messages"],
                    "rejected": rejected["messages"],
                    "lane": chosen.get("lane"),
                    "rejected_type": label,
                })
                added += 1
            attempts += 1
        log.info("DPO %s: %d/%d pairs", label, added, count)

    _add_pairs(tool_samples or all_samples, _make_rejected_bad_method,
               bad_method_target, "bad_method")
    _add_pairs(tool_samples or all_samples, _make_rejected_bad_matching,
               bad_matching_target, "bad_matching")
    _add_pairs(all_samples, _make_rejected_short_answer,
               short_answer_target, "short_answer")

    rng.shuffle(pairs)
    return pairs[:target]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate R2 SFT and DPO data")
    parser.add_argument("--sft-only", action="store_true")
    parser.add_argument("--dpo-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate but don't write files; print stats")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sft_samples: List[Dict[str, Any]] = []

    if not args.dpo_only:
        log.info("=== Generating SFT data ===")
        for lane, target in SFT_TARGET.items():
            lane_samples = _generate_sft_for_lane(lane, target, rng)
            for s in lane_samples:
                s["split"] = "sft"
            sft_samples.extend(lane_samples)

        log.info("SFT total: %d samples", len(sft_samples))

        if not args.dry_run:
            sft_path = DATA_DIR / "sft_r2.jsonl"
            with open(sft_path, "w", encoding="utf-8") as f:
                for s in sft_samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            log.info("Written: %s", sft_path)
        else:
            log.info("[dry-run] Would write %d SFT samples", len(sft_samples))

    if not args.sft_only:
        log.info("=== Generating DPO data ===")
        if not sft_samples:
            # Load seeds as proxy
            sft_samples = _load_seeds()

        dpo_pairs = _generate_dpo(sft_samples, DPO_TARGET, rng)
        log.info("DPO total: %d pairs", len(dpo_pairs))

        if not args.dry_run:
            dpo_path = DATA_DIR / "dpo_r2.jsonl"
            with open(dpo_path, "w", encoding="utf-8") as f:
                for p in dpo_pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            log.info("Written: %s", dpo_path)
        else:
            log.info("[dry-run] Would write %d DPO pairs", len(dpo_pairs))

    # --- Stats ---
    log.info("=== Summary ===")
    if sft_samples:
        lang_dist: Dict[str, int] = {}
        lane_dist: Dict[str, int] = {}
        final_lengths = []
        schema_pass = 0; schema_total = 0

        for s in sft_samples:
            lane = s.get("lane", "?")
            lane_dist[lane] = lane_dist.get(lane, 0) + 1

            user_content = next(
                (m["content"] for m in s.get("messages", []) if m["role"] == "user"), ""
            )
            lang = _detect_language(user_content)
            lang_dist[lang] = lang_dist.get(lang, 0) + 1

            # Final answer length
            for m in reversed(s.get("messages", [])):
                if m["role"] == "assistant":
                    final_lengths.append(len(m["content"]))
                    break

            # Schema validation
            from solver.validator_r2 import extract_from_messages_r2
            _, tool_call, _ = extract_from_messages_r2(s.get("messages", []))
            if tool_call is not None:
                schema_total += 1
                if validate_schema_r2(tool_call):
                    schema_pass += 1

        total = len(sft_samples)
        en_count = lang_dist.get("en", 0)
        ko_count = lang_dist.get("ko", 0)
        avg_len = sum(final_lengths) / len(final_lengths) if final_lengths else 0
        min_len = min(final_lengths) if final_lengths else 0
        max_len = max(final_lengths) if final_lengths else 0

        log.info("Language dist: EN=%d (%.1f%%) KO=%d (%.1f%%)",
                 en_count, 100*en_count/total, ko_count, 100*ko_count/total)
        log.info("Lane dist: %s", lane_dist)
        log.info("Final answer: avg=%.0f min=%d max=%d", avg_len, min_len, max_len)
        if schema_total > 0:
            log.info("Schema pass rate: %d/%d (%.1f%%)",
                     schema_pass, schema_total, 100*schema_pass/schema_total)


if __name__ == "__main__":
    main()
