"""Augment seed data into full SFT / DPO corpora.

Targets (Director's L01-boost):
  L01 3000, L02/L03/L04/L05 1500 each, L06-L10 ~2000 each.
  DPO pairs ~20000.

Strategy for tool-bearing lanes (L01-L04):
  1. Load seed as template.
  2. Randomly flip some matching/conditions booleans.
  3. Run solver to compute ground-truth tool_result (always correct).
  4. Keep samples whose roundtrip validates.

For L05: randomise numeric vars in supported ranges, recompute expected value.

For L06-L10 direct lanes: template-vary user phrasing with prefixes/suffixes,
shuffle answer boilerplate. (No solver involvement — always passes.)

DPO (chosen/rejected):
  chosen  = correct tool_call produced by augmenter
  rejected = mutated tool_call (wrong lane, wrong matching, malformed JSON, etc.)
"""
from __future__ import annotations
import json
import random
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solver.executor import execute
from solver.validator import roundtrip, validate_schema

random.seed(20260418)

SEEDS_DIR = ROOT / "data" / "seeds"
DATA_DIR = ROOT / "data"
OUT_SFT = DATA_DIR / "sft_merged.jsonl"
OUT_DPO = DATA_DIR / "dpo_pairs.jsonl"

TARGETS = {
    "L01_element_matching": 3000,
    "L02_rule_application": 1500,
    "L03_subsumption": 1500,
    "L04_logic_judgment": 1500,
    "L05_calculation": 1500,
    "L06_explanation": 2000,
    "L07_case_comparison": 2000,
    "L08_summary": 2000,
    "L09_translation": 2000,
    "L10_uncertain": 2000,
}

USER_PREFIXES = ["", "문의드립니다. ", "안녕하세요. ", "질문: ", "변호사님, ", "법률상담 요청: "]
USER_SUFFIXES = ["", " 답변 부탁드립니다.", " 설명해주세요.", " 알려주세요.", " 판단이 가능할까요?"]


def load_seeds(lane):
    path = SEEDS_DIR / f"{lane}.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _clone(x):
    return copy.deepcopy(x)


def _tool_call(sample):
    """Extract the assistant's tool_call dict."""
    import re
    for m in sample["messages"]:
        if m["role"] == "assistant":
            match = re.search(r"<tool_call>(.*?)</tool_call>", m["content"], re.DOTALL)
            if match:
                return json.loads(match.group(1))
    return None


def _rebuild_sample(seed, new_tc, new_expl=None):
    """Re-run solver and rebuild messages from new tool_call."""
    import re
    res = execute(new_tc)
    if "error" in res:
        return None
    lane = seed["lane"]
    # vary user phrasing
    user_q = seed["messages"][1]["content"]
    user_q = random.choice(USER_PREFIXES) + user_q + random.choice(USER_SUFFIXES)
    explanation = new_expl or seed["messages"][-1]["content"]
    asst1 = f"<lane>{lane}</lane>\n<tool_call>{json.dumps(new_tc, ensure_ascii=False)}</tool_call>"
    return {
        "lane": lane,
        "messages": [
            seed["messages"][0],
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": asst1},
            {"role": "tool", "content": json.dumps(res, ensure_ascii=False)},
            {"role": "assistant", "content": explanation},
        ],
    }


def aug_boolean_lane(seed, key_path):
    """Generic mutation: flip ~30% of booleans in the keyed dict."""
    tc = _clone(_tool_call(seed))
    obj = tc
    for k in key_path[:-1]:
        obj = obj[k]
    target = obj[key_path[-1]]
    if not isinstance(target, dict) or not target:
        return None
    keys = list(target.keys())
    flip = random.sample(keys, k=max(1, len(keys) // 3))
    for k in flip:
        target[k] = not target[k]
    return _rebuild_sample(seed, tc)


def aug_l01(seed): return aug_boolean_lane(seed, ["matching"])
def aug_l02(seed): return aug_boolean_lane(seed, ["conditions"])
def aug_l03(seed): return aug_boolean_lane(seed, ["subsumption"])


def aug_l04(seed):
    """Perturb conclusion or swap a premise (may flip validity); solver recomputes."""
    tc = _clone(_tool_call(seed))
    # optionally negate the conclusion with 50% chance
    if random.random() < 0.5:
        c = tc["conclusion"].strip()
        if c.startswith("not "):
            tc["conclusion"] = c[4:]
        else:
            tc["conclusion"] = f"not ({c})"
    return _rebuild_sample(seed, tc)


L05_SCALE = {
    "principal": (1_000_000, 1_000_000_000),
    "rate": (0.01, 0.2),
    "years": (1, 30),
    "days": (1, 365),
    "months": (1, 120),
    "fault": (0.0, 0.8),
    "ratio": (0.1, 0.9),
    "count": (1, 10),
    "heirs": (1, 10),
    "n": (1, 10),
    "r": (0.01, 0.2),
    "r1": (0.01, 0.15), "r2": (0.01, 0.15), "r3": (0.01, 0.15),
    "med": (100_000, 100_000_000), "income": (100_000, 100_000_000), "solatium": (100_000, 100_000_000),
    "fv": (1_000_000, 1_000_000_000), "pv": (1_000_000, 1_000_000_000),
    "amount": (10_000, 1_000_000_000), "daily": (50_000, 1_000_000),
    "unused": (1, 30), "hours": (1, 12), "wage": (8000, 50000),
    "total": (1_000_000, 1_000_000_000), "end": (2000, 2050), "start": (1900, 2025),
    "period": (1, 30), "monthly": (100_000, 10_000_000),
    "base": (1, 1_000_000_000),
    "earnest": (100_000, 100_000_000), "fee": (100_000, 100_000_000), "stamp": (10_000, 1_000_000),
    "deposit": (1_000_000, 1_000_000_000), "avg": (1_000_000, 20_000_000),
    "sale": (10_000_000, 10_000_000_000), "cost": (1_000_000, 10_000_000_000),
    "premium": (0.01, 0.2),
    "supply": (1_000_000, 1_000_000_000), "defect": (0.0, 0.9), "price": (100_000, 1_000_000_000),
    "interest": (100_000, 100_000_000),
}


def aug_l05(seed):
    tc = _clone(_tool_call(seed))
    # only randomise numeric vars for 'eval' (solve keeps expected symbolic target)
    if tc.get("method") != "eval":
        return None
    for k in list(tc.get("vars", {}).keys()):
        lo, hi = L05_SCALE.get(k, (1, 100))
        if isinstance(lo, float) or isinstance(hi, float):
            tc["vars"][k] = round(random.uniform(lo, hi), 4)
        else:
            tc["vars"][k] = random.randint(int(lo), int(hi))
    return _rebuild_sample(seed, tc)


# -------- direct lanes --------

L06_PREFIXES = ["", "간단히 ", "한 단락으로 ", "초보자에게 설명하듯 "]
L06_SUFFIXES = ["", " 감사합니다.", " 부탁드립니다."]

def aug_direct(seed):
    s = _clone(seed)
    q = s["messages"][1]["content"]
    s["messages"][1]["content"] = random.choice(USER_PREFIXES) + q + random.choice(USER_SUFFIXES)
    return s


AUG = {
    "L01_element_matching": aug_l01,
    "L02_rule_application": aug_l02,
    "L03_subsumption": aug_l03,
    "L04_logic_judgment": aug_l04,
    "L05_calculation": aug_l05,
    "L06_explanation": aug_direct,
    "L07_case_comparison": aug_direct,
    "L08_summary": aug_direct,
    "L09_translation": aug_direct,
    "L10_uncertain": aug_direct,
}


def generate_lane(lane, target):
    seeds = load_seeds(lane)
    if not seeds:
        return []
    samples = list(seeds)  # include original seeds
    tries = 0
    max_tries = target * 5
    aug_fn = AUG[lane]
    while len(samples) < target and tries < max_tries:
        tries += 1
        seed = random.choice(seeds)
        try:
            new = aug_fn(seed)
        except Exception:
            new = None
        if new is None:
            continue
        if not roundtrip(new):
            continue
        samples.append(new)
    return samples[:target]


def make_dpo_pair(sample):
    """Construct (chosen, rejected) from one correct sample.

    chosen   = current sample
    rejected = mutated — wrong lane tag or malformed matching/conditions values
    """
    if "<tool_call>" not in sample["messages"][2]["content"]:
        # direct lane: swap correct answer with a generic wrong one
        bad = _clone(sample)
        bad["messages"][-1]["content"] = "<lane>L01_element_matching</lane>\n<tool_call>{}</tool_call>"
        return sample, bad

    bad = _clone(sample)
    # replace lane tag with a wrong lane
    wrong_lanes = [l for l in TARGETS if l != sample["lane"] and l.startswith("L")]
    wrong = random.choice(wrong_lanes)
    bad["messages"][2]["content"] = bad["messages"][2]["content"].replace(
        f"<lane>{sample['lane']}</lane>", f"<lane>{wrong}</lane>"
    )
    # also corrupt the tool_call json slightly so roundtrip fails
    bad["messages"][2]["content"] = bad["messages"][2]["content"].replace(
        "<tool_call>{", "<tool_call>{\"BAD\":true,"
    )
    return sample, bad


def main():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    all_samples = []
    lane_counts = {}
    for lane, tgt in TARGETS.items():
        samples = generate_lane(lane, tgt)
        lane_counts[lane] = len(samples)
        all_samples.extend(samples)
        print(f"  {lane}: {len(samples)}")

    random.shuffle(all_samples)
    with OUT_SFT.open("w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nSFT written: {OUT_SFT} ({len(all_samples)} samples)")

    # DPO pairs
    tool_bearing = [s for s in all_samples if s["lane"].startswith(("L01", "L02", "L03", "L04", "L05"))]
    random.shuffle(tool_bearing)
    dpo_target = 20000
    dpo_source = tool_bearing * (dpo_target // max(1, len(tool_bearing)) + 1)
    dpo_source = dpo_source[:dpo_target]
    with OUT_DPO.open("w", encoding="utf-8") as f:
        for s in dpo_source:
            chosen, rejected = make_dpo_pair(s)
            f.write(json.dumps({"chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
    print(f"DPO written: {OUT_DPO} ({dpo_target} pairs)")

    return lane_counts


if __name__ == "__main__":
    main()
