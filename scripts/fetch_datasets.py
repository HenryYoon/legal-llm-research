"""Fetch LegalBench CSVs and probe KLAC schema.

Outputs:
  data/legalbench/<task>.csv
  data/legalbench_status.json
  data/klac_status.json
"""
from __future__ import annotations
import json
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LB_DIR = ROOT / "data" / "legalbench"
LB_DIR.mkdir(parents=True, exist_ok=True)
LB_STATUS = ROOT / "data" / "legalbench_status.json"
KLAC_STATUS = ROOT / "data" / "klac_status.json"

LB_TASKS = ["hearsay", "personal_jurisdiction", "rule_qa", "textualism_tool_dictionaries"]


def fetch_legalbench():
    status = {}
    try:
        from datasets import load_dataset
    except Exception as e:
        status["_import_error"] = str(e)
        LB_STATUS.write_text(json.dumps(status, indent=2, ensure_ascii=False))
        return status

    # textualism_tool has multiple variants in LegalBench; try fallbacks
    task_candidates = {
        "hearsay": ["hearsay"],
        "personal_jurisdiction": ["personal_jurisdiction"],
        "rule_qa": ["rule_qa"],
        "textualism_tool": ["textualism_tool_dictionaries", "textualism_tool_plain"],
    }

    for out_name, candidates in task_candidates.items():
        ok = False
        for cand in candidates:
            try:
                ds = load_dataset("nguha/legalbench", cand)
                # save each split to its own csv
                for split, sub in ds.items():
                    out = LB_DIR / f"{out_name}__{split}.csv"
                    sub.to_csv(str(out), index=False)
                status[out_name] = {
                    "variant": cand,
                    "splits": {split: len(ds[split]) for split in ds.keys()},
                }
                ok = True
                break
            except Exception as e:
                status[f"{out_name}__{cand}__error"] = f"{type(e).__name__}: {e}"
        if not ok:
            status[out_name] = "FAILED"

    LB_STATUS.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    print(f"LegalBench status: {LB_STATUS}")
    return status


def probe_klac():
    status = {}
    try:
        from datasets import load_dataset
        ds = load_dataset("jihye-moon/klac_legal_aid_counseling")
        status["splits"] = {k: len(v) for k, v in ds.items()}
        first_split = next(iter(ds))
        sample = ds[first_split][0]
        status["fields"] = list(sample.keys())
        # don't store full sample (may be long) — first 200 chars per field
        status["sample_excerpt"] = {
            k: (str(v)[:200] + ("..." if len(str(v)) > 200 else ""))
            for k, v in sample.items()
        }
        # license — try to grab from dataset info
        try:
            info = ds[first_split].info
            status["license"] = getattr(info, "license", None)
            status["description"] = (info.description or "")[:300]
        except Exception:
            status["license"] = "unknown"
    except Exception as e:
        status["error"] = f"{type(e).__name__}: {e}"
        status["traceback"] = traceback.format_exc()[:2000]
    KLAC_STATUS.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    print(f"KLAC status: {KLAC_STATUS}")
    return status


if __name__ == "__main__":
    print("Fetching LegalBench…")
    fetch_legalbench()
    print("\nProbing KLAC…")
    probe_klac()
