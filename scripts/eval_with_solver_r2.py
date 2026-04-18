#!/usr/bin/env python3
"""R2 evaluation with Outlines constrained decoding.

Changes from eval_with_solver.py:
- Lane 5 (L01, L05, L06, L09, L10) instead of 10
- Constrained lane generation via outlines.regex (choice)
- Constrained tool_call JSON via outlines.json_schema
- Fallback: regex-based extraction if Outlines unavailable

Usage:
  python scripts/eval_with_solver_r2.py --model qwen25 --task hearsay --variant lane_solver
  python scripts/eval_with_solver_r2.py --smoke-test  # single-inference test
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from solver.executor_r2 import execute_r2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_IDS = {
    "qwen25": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3": "Qwen/Qwen3-1.7B",
}

VALID_LANES_R2 = ["L01", "L05", "L06", "L09", "L10"]
LANE_CHOICE_PATTERN = r"(L01|L05|L06|L09|L10)"

TASK_CONFIG = {
    "hearsay": {
        "csv_path": "data/legalbench/hearsay__test.csv",
        "type": "binary",
        "label_col": "answer",
        "question_col": "text",
        "solver_compatible": True,
    },
    "personal_jurisdiction": {
        "csv_path": "data/legalbench/personal_jurisdiction__test.csv",
        "type": "binary",
        "label_col": "answer",
        "question_col": "text",
        "solver_compatible": True,
    },
    "rule_qa": {
        "csv_path": "data/legalbench/rule_qa__test.csv",
        "type": "generation",
        "label_col": "answer",
        "question_col": "text",
        "solver_compatible": False,
    },
    "textualism_tool": {
        "csv_path": "data/legalbench/textualism_tool__test.csv",
        "type": "binary",
        "label_col": "answer",
        "question_col": "text",
        "solver_compatible": False,
    },
}

# R2 JSON Schema for tool_call (z3.check_elements)
L01_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["z3"]},
        "method": {"type": "string", "enum": ["check_elements"]},
        "elements": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "facts": {"type": "array", "items": {"type": "string"}},
        "matching": {"type": "object", "additionalProperties": {"type": "boolean"}},
        "mode": {"type": "string", "enum": ["and", "or"]},
    },
    "required": ["tool", "method", "elements", "facts", "matching"],
}

L05_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["sympy"]},
        "method": {"type": "string", "enum": ["calc"]},
        "expr": {"type": "string"},
        "vars": {"type": "object", "additionalProperties": {"type": "number"}},
        "round_to": {"type": "integer"},
    },
    "required": ["tool", "method", "expr", "vars"],
}

LANE_SYSTEM_PROMPT_R2 = """You are a legal reasoning AI (R2). Classify the task into one of five Legal Lanes:
- L01: Element matching — use z3.check_elements for Boolean element satisfaction
- L05: Calculation — use sympy.calc for numeric computation
- L06: Explanation — explain legal concepts directly (no tool_call)
- L09: Translation — translate legal terms (no tool_call)
- L10: Uncertain — state uncertainty (no tool_call)

For L01 tasks, generate:
<lane>L01</lane>
<tool_call>{"tool": "z3", "method": "check_elements", "elements": [...], "facts": [...], "matching": {...}, "mode": "and"}</tool_call>
[Use solver result to write a final answer of at least 80 characters explaining the legal conclusion.]

For L05 tasks, generate:
<lane>L05</lane>
<tool_call>{"tool": "sympy", "method": "calc", "expr": "...", "vars": {...}}</tool_call>
[Use solver result to write a final answer explaining the calculation.]

For L06/L09/L10, answer directly after <lane>L06</lane> (or L09/L10)."""


# ---------------------------------------------------------------------------
# Outlines integration
# ---------------------------------------------------------------------------

class OutlinesConstrainedDecoder:
    """Wrapper that uses Outlines for constrained lane and tool_call generation.

    Falls back to regex extraction if Outlines is unavailable or fails.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._outlines_model = None
        self._outlines_available = False
        self._lane_generator = None
        self._l01_generator = None
        self._l05_generator = None

        self._init_outlines()

    def _init_outlines(self):
        """Initialize Outlines generators. Log and fall back on failure."""
        try:
            import outlines
            # Wrap the HuggingFace model
            self._outlines_model = outlines.from_transformers(
                self.model, self.tokenizer
            )
            # Lane choice generator: constrain to exactly one of the 5 lanes
            self._lane_generator = outlines.Generator(
                self._outlines_model,
                output_type=outlines.regex(LANE_CHOICE_PATTERN),
            )
            # L01 tool_call JSON generator
            self._l01_generator = outlines.Generator(
                self._outlines_model,
                output_type=outlines.json_schema(L01_TOOL_SCHEMA),
            )
            # L05 tool_call JSON generator
            self._l05_generator = outlines.Generator(
                self._outlines_model,
                output_type=outlines.json_schema(L05_TOOL_SCHEMA),
            )
            self._outlines_available = True
            log.info("Outlines constrained decoding: ENABLED")
        except Exception as e:
            log.warning(
                "Outlines initialization failed (%s). Using regex fallback.", e
            )
            self._outlines_available = False

    def generate_lane(self, prompt: str, max_new_tokens: int = 8) -> str:
        """Generate lane tag with Outlines choice constraint, or regex fallback."""
        if self._outlines_available and self._lane_generator is not None:
            try:
                lane = self._lane_generator(
                    prompt, max_tokens=max_new_tokens, sampling_params={"temperature": 0.0}
                )
                if lane in VALID_LANES_R2:
                    return lane
            except Exception as e:
                log.debug("Outlines lane generation failed: %s", e)

        # Regex fallback: generate free text and parse
        return self._generate_free_and_extract_lane(prompt, max_new_tokens=50)

    def generate_tool_call(
        self, prompt: str, lane: str, max_new_tokens: int = 256
    ) -> Optional[Dict[str, Any]]:
        """Generate structured tool_call JSON with Outlines json_schema, or regex fallback."""
        if self._outlines_available:
            try:
                generator = (
                    self._l01_generator if lane == "L01" else
                    self._l05_generator if lane == "L05" else
                    None
                )
                if generator is not None:
                    raw = generator(
                        prompt,
                        max_tokens=max_new_tokens,
                        sampling_params={"temperature": 0.0},
                    )
                    return json.loads(raw) if isinstance(raw, str) else raw
            except Exception as e:
                log.debug("Outlines tool_call generation failed: %s", e)

        # Regex fallback
        return self._generate_free_and_extract_tool_call(prompt, max_new_tokens)

    def generate_free(
        self, prompt: str, max_new_tokens: int = 512
    ) -> str:
        """Unconstrained generation for final answer and direct lanes."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with __import__("torch").no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # --- internal helpers ---

    def _generate_free_and_extract_lane(
        self, prompt: str, max_new_tokens: int = 80
    ) -> str:
        text = self.generate_free(prompt, max_new_tokens)
        m = re.search(r"<lane>(L\d{2})</lane>", text)
        if m and m.group(1) in VALID_LANES_R2:
            return m.group(1)
        # Try bare pattern
        m = re.search(r"\b(L01|L05|L06|L09|L10)\b", text)
        if m:
            return m.group(1)
        log.debug("Lane extraction failed from: %r → defaulting L06", text[:80])
        return "L06"

    def _generate_free_and_extract_tool_call(
        self, prompt: str, max_new_tokens: int = 256
    ) -> Optional[Dict[str, Any]]:
        text = self.generate_free(prompt, max_new_tokens)
        m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
        return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_key: str, adapter_path: Optional[str] = None):
    """Load model with 4-bit quantization."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_id = MODEL_IDS[model_key]
    log.info("Loading %s from %s", model_key, model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    if adapter_path:
        from peft import PeftModel
        log.info("Loading adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
    log.info("Model loaded.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

def run_lane_solver_r2(
    decoder: OutlinesConstrainedDecoder,
    question: str,
) -> Tuple[str, Optional[Dict], Optional[Dict], str]:
    """
    Full R2 lane+solver pipeline for a single question.

    Returns: (lane, tool_call, tool_result, final_answer)
    """
    # Step 1: Generate lane (constrained)
    lane_prompt = (
        f"{LANE_SYSTEM_PROMPT_R2}\n\nQuestion: {question}\n\n"
        "Lane classification: <lane>"
    )
    lane = decoder.generate_lane(lane_prompt)
    log.debug("Lane: %s", lane)

    tool_call = None
    tool_result = None

    # Step 2: For L01/L05, generate tool_call (constrained JSON)
    if lane in ("L01", "L05"):
        tc_prompt = (
            f"{LANE_SYSTEM_PROMPT_R2}\n\nQuestion: {question}\n\n"
            f"<lane>{lane}</lane>\n<tool_call>"
        )
        tool_call = decoder.generate_tool_call(tc_prompt, lane)
        if tool_call is not None:
            tool_result = execute_r2(tool_call)
            if "error" in tool_result:
                log.debug("Executor error: %s", tool_result["error"])
                tool_result = None

    # Step 3: Generate final answer (unconstrained)
    if tool_result is not None:
        fa_prompt = (
            f"{LANE_SYSTEM_PROMPT_R2}\n\nQuestion: {question}\n\n"
            f"<lane>{lane}</lane>\n"
            f"Solver result: {json.dumps(tool_result, ensure_ascii=False)}\n\n"
            "Final answer (at least 80 characters, explain conclusion, cite relevant rule):\n"
        )
    else:
        fa_prompt = (
            f"{LANE_SYSTEM_PROMPT_R2}\n\nQuestion: {question}\n\n"
            f"<lane>{lane}</lane>\n"
            "Answer:\n"
        )
    final_answer = decoder.generate_free(fa_prompt, max_new_tokens=300)
    return lane, tool_call, tool_result, final_answer.strip()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_task(
    decoder: OutlinesConstrainedDecoder,
    task: str,
    max_samples: int = 50,
) -> Dict[str, Any]:
    """Run evaluation on a LegalBench task, return metrics dict."""
    import pandas as pd

    cfg = TASK_CONFIG[task]
    csv_path = ROOT / cfg["csv_path"]
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)

    predictions, labels, tool_call_rates, parse_rates = [], [], [], []

    for _, row in df.iterrows():
        question = str(row[cfg["question_col"]])
        label = str(row[cfg["label_col"]]).strip()
        labels.append(label)

        lane, tc, tr, fa = run_lane_solver_r2(decoder, question)

        tool_call_rates.append(1 if tc is not None else 0)

        if cfg["type"] == "binary":
            m = re.search(r"\b(Yes|No)\b", fa, re.IGNORECASE)
            if m:
                pred = m.group(1).capitalize()
                parse_rates.append(1)
            else:
                pred = "No"
                parse_rates.append(0)
            predictions.append(pred)
        else:
            predictions.append(fa)
            parse_rates.append(1)

    metrics: Dict[str, Any] = {
        "task": task,
        "n_samples": len(labels),
        "tool_call_rate": sum(tool_call_rates) / len(tool_call_rates),
        "parse_rate": sum(parse_rates) / len(parse_rates),
    }

    if cfg["type"] == "binary":
        from sklearn.metrics import balanced_accuracy_score
        metrics["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)
    else:
        from rouge_score import rouge_scorer as rs_module
        scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, hyp)["rougeL"].fmeasure
                  for ref, hyp in zip(labels, predictions)]
        metrics["rouge_l"] = sum(scores) / len(scores)

    log.info("Task %s: %s", task, {k: f"{v:.3f}" if isinstance(v, float) else v
                                    for k, v in metrics.items()})
    return metrics


# ---------------------------------------------------------------------------
# Smoke test (single inference)
# ---------------------------------------------------------------------------

def smoke_test(model_key: str = "qwen25", adapter_path: Optional[str] = None):
    """Quick single-inference test to verify Outlines integration."""
    log.info("=== Outlines smoke test ===")
    model, tokenizer = load_model_and_tokenizer(model_key, adapter_path)
    decoder = OutlinesConstrainedDecoder(model, tokenizer)

    test_question = (
        "On the issue of whether Sarah was present at the meeting, "
        "the fact that Sarah told her assistant she planned to attend. Is this hearsay?"
    )
    log.info("Test question: %s", test_question)
    lane, tc, tr, fa = run_lane_solver_r2(decoder, test_question)

    log.info("Lane: %s", lane)
    log.info("Tool call: %s", tc)
    log.info("Tool result: %s", tr)
    log.info("Final answer (%d chars): %s", len(fa), fa[:200])

    outlines_status = "ENABLED" if decoder._outlines_available else "FALLBACK (regex)"
    log.info("Outlines status: %s", outlines_status)

    return {
        "lane": lane,
        "tool_call": tc,
        "tool_result": tr,
        "final_answer": fa,
        "outlines_available": decoder._outlines_available,
        "final_answer_length": len(fa),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="R2 evaluation with Outlines")
    parser.add_argument("--model", choices=list(MODEL_IDS), default="qwen25")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--task", choices=list(TASK_CONFIG), default=None)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run single-inference Outlines smoke test")
    parser.add_argument("--output", default=None, help="Path to write results JSON")
    args = parser.parse_args()

    if args.smoke_test:
        result = smoke_test(args.model, args.adapter)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)
    decoder = OutlinesConstrainedDecoder(model, tokenizer)

    tasks = list(TASK_CONFIG) if args.all_tasks else ([args.task] if args.task else [])
    if not tasks:
        parser.error("Specify --task <name> or --all-tasks")

    all_metrics = []
    for task in tasks:
        m = evaluate_task(decoder, task, max_samples=args.max_samples)
        all_metrics.append(m)

    results = {
        "model": args.model,
        "adapter": args.adapter,
        "timestamp": datetime.now().isoformat(),
        "outlines_available": decoder._outlines_available,
        "metrics": all_metrics,
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        log.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
