#!/usr/bin/env python3
"""Evaluation with solver integration: A2 (solver), A3 (lane+solver)."""
import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import balanced_accuracy_score
from rouge_score import rouge_scorer

# Import solver
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from solver import executor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model mapping
MODEL_IDS = {
    "qwen25": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3": "Qwen/Qwen3-1.7B",
}

# Task configuration
TASK_CONFIG = {
    "hearsay": {
        "csv_path": "data/legalbench/hearsay__test.csv",
        "type": "binary",
        "label_col": "answer",
        "question_col": "text",
        "true_labels": ["Yes", "No"],
        "solver_compatible": True,  # L01 L02 tasks
    },
    "personal_jurisdiction": {
        "csv_path": "data/legalbench/personal_jurisdiction__test.csv",
        "type": "binary",
        "label_col": "answer",
        "question_col": "text",
        "true_labels": ["Yes", "No"],
        "solver_compatible": True,  # L01 tasks
    },
    "rule_qa": {
        "csv_path": "data/legalbench/rule_qa__test.csv",
        "type": "generation",
        "label_col": "answer",
        "question_col": "text",
        "solver_compatible": False,  # L06 direct generation
    },
    "textualism_tool": {
        "csv_path": "data/legalbench/textualism_tool__test.csv",
        "type": "binary",
        "label_col": "answer",
        "question_col": "text",
        "true_labels": ["Yes", "No"],
        "solver_compatible": False,  # L06 direct generation
    },
}

# Lane system prompt
LANE_SYSTEM_PROMPT = """You are a legal reasoning AI. Classify the task into one of 10 Legal Lanes:
- L01: Element matching (identify matching legal elements)
- L02: Rule application (apply legal rules to conditions)
- L03: Subsumption (check if facts fit legal standards)
- L04: Logic judgment (evaluate logical validity)
- L05: Calculation (solve numeric/mathematical problems)
- L06: Explanation (explain legal concepts)
- L07: Case comparison (compare with precedents)
- L08: Summarization (summarize legal content)
- L09: Translation (translate legal terms)
- L10: Uncertainty (handle ambiguous cases)

For L01-L05 tasks, generate a tool_call in this format:
<tool_call>{"tool": "z3"|"sympy", "method": "...", ...other_fields...}</tool_call>

Then use the solver result to formulate your final answer.
For L06-L10 tasks, answer directly without tool_call.

Format your response as:
<lane>L##_description</lane>
[tool_call if applicable]
[final answer]"""


def load_model_and_tokenizer(model_key: str, adapter_path: str = None):
    """Load model with 4-bit quantization. Optionally apply PEFT adapter."""
    model_id = MODEL_IDS[model_key]

    logger.info(f"Loading {model_key} from {model_id}")

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
        dtype=torch.bfloat16,
    )

    if adapter_path:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    logger.info(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    return model, tokenizer


def build_solver_prompt(variant: str, task: str, question: str) -> str:
    """Build prompt for solver variant (A2 or A3)."""
    if variant == "solver":
        # A2: No Lane classification, direct tool_call attempt
        template = f"""You are a legal reasoning AI.
For the following question, first generate a tool_call to use the solver, then provide your final answer.

Question: {question}

Generate a tool_call like: <tool_call>{{"tool": "z3"|"sympy", "method": "...", ...}}</tool_call>

Final Answer: """
    else:  # lane_solver (A3)
        # A3: Lane classification + tool_call
        template = f"""{LANE_SYSTEM_PROMPT}

Question: {question}

Response: """

    return template


def parse_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """Parse <tool_call>...</tool_call> from response."""
    pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(pattern, response, re.DOTALL)

    if not match:
        return None

    try:
        tool_call_str = match.group(1)
        tool_call = json.loads(tool_call_str)
        return tool_call
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse tool_call JSON: {e}")
        return None


def execute_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Execute tool_call via solver.executor."""
    try:
        result = executor.execute(tool_call)
        return result
    except Exception as e:
        logger.warning(f"Solver execution failed: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


def generate_with_solver(
    model, tokenizer, variant: str, task: str, question: str, max_new_tokens: int = 256
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Generate answer with optional solver integration.

    Returns: (final_answer, tool_call_dict, solver_result)
    """
    prompt = build_solver_prompt(variant, task, question)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    response = response.strip()

    # Try to parse tool_call
    tool_call = parse_tool_call(response)
    solver_result = None

    if tool_call:
        # Execute solver
        solver_result = execute_tool_call(tool_call)

        # If solver succeeded, use result in context for final answer
        if "error" not in solver_result:
            logger.debug(f"Solver result: {solver_result}")
            # In a full pipeline, re-inject solver result and regenerate final answer
            # For now, extract answer from response after tool_call
        else:
            logger.warning(f"Solver error: {solver_result}")
    else:
        logger.debug("No tool_call found in response, using direct answer")

    # Extract final answer (last line or after tool_call)
    final_answer = response
    if "<tool_call>" in response:
        # Take text after tool_call as final answer
        parts = response.split("</tool_call>")
        if len(parts) > 1:
            final_answer = parts[-1].strip()

    return final_answer, tool_call, solver_result


def parse_binary_answer(response: str, true_labels: List[str]) -> str:
    """Parse binary answer from response."""
    response = response.strip().upper()

    # Try exact match first
    for label in true_labels:
        if label.upper() in response:
            return label

    # Fallback to first word if it's Yes/No
    first_word = response.split()[0].upper() if response.split() else ""
    if first_word in [l.upper() for l in true_labels]:
        for label in true_labels:
            if label.upper() == first_word:
                return label

    return "PARSE_FAIL"


def compute_balanced_accuracy(
    predictions: List[str], references: List[str]
) -> float:
    """Compute balanced accuracy, handling parse failures."""
    valid_preds = []
    valid_refs = []

    for pred, ref in zip(predictions, references):
        if pred != "PARSE_FAIL":
            valid_preds.append(1 if pred == "Yes" else 0)
            valid_refs.append(1 if ref == "Yes" else 0)

    if not valid_preds:
        return 0.0

    return balanced_accuracy_score(valid_refs, valid_preds)


def evaluate_task_with_solver(
    model, tokenizer, variant: str, task: str, model_key: str, max_samples: int = None
) -> Dict[str, Any]:
    """Evaluate on a single task with solver."""
    config = TASK_CONFIG[task]
    csv_path = Path(config["csv_path"])

    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return {"error": f"CSV not found: {csv_path}"}

    # Load dataset
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)

    logger.info(f"Evaluating {task} with {len(df)} samples (variant={variant})")

    predictions = []
    references = []
    parse_successes = 0
    tool_call_successes = 0

    for idx, row in df.iterrows():
        question = row[config["question_col"]]
        reference = row[config["label_col"]]

        try:
            final_answer, tool_call, solver_result = generate_with_solver(
                model, tokenizer, variant, task, question
            )

            if tool_call:
                tool_call_successes += 1

            if config["type"] == "binary":
                pred = parse_binary_answer(final_answer, config["true_labels"])
                if pred != "PARSE_FAIL":
                    parse_successes += 1
            else:
                pred = final_answer

            predictions.append(pred)
            references.append(reference)
        except Exception as e:
            logger.error(f"Error on sample {idx}: {e}")
            predictions.append("ERROR")
            references.append(reference)

        if (idx + 1) % 10 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)}")

    # Compute metrics
    parse_rate = parse_successes / len(df) if config["type"] == "binary" else 1.0
    tool_call_rate = tool_call_successes / len(df) if len(df) > 0 else 0

    if config["type"] == "binary":
        accuracy = compute_balanced_accuracy(predictions, references)
    else:
        # For generation tasks, use ROUGE-L score
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_scores = []
            for pred, ref in zip(predictions, references):
                if pred and ref:
                    score = scorer.score(ref, pred)
                    rouge_scores.append(score['rougeL'].fmeasure)
            accuracy = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
        except Exception as e:
            logger.warning(f"ROUGE-L computation failed: {e}, using exact match")
            accuracy = sum(1 for p, r in zip(predictions, references) if p.lower() == r.lower()) / len(df)

    return {
        "task": task,
        "model": model_key,
        "variant": variant,
        "accuracy": accuracy,
        "n": len(df),
        "parse_rate": parse_rate,
        "tool_call_rate": tool_call_rate,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluation with solver integration (A2, A3)")
    parser.add_argument(
        "--model",
        choices=["qwen25", "qwen3"],
        default="qwen25",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--variant",
        choices=["solver", "lane_solver"],
        default="solver",
        help="Variant: solver=A2 (no Lane), lane_solver=A3 (with Lane)",
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_CONFIG.keys()) + ["all"],
        default="all",
        help="Task to evaluate",
    )
    parser.add_argument(
        "--smoke-test",
        type=int,
        default=None,
        help="Max samples for smoke test",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional LoRA adapter path (for A3 SFT variant)",
    )
    args = parser.parse_args()

    # Setup directories
    results_dir = Path("results")
    log_dir = results_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = log_dir / f"experiment_{args.model}_{args.variant}_{args.task}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    logger.info(f"Starting solver evaluation: model={args.model}, variant={args.variant}, task={args.task}")

    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(args.model, adapter_path=args.adapter)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Determine tasks to evaluate
    tasks = [args.task] if args.task != "all" else list(TASK_CONFIG.keys())

    # Evaluate tasks
    results = []
    for task in tasks:
        try:
            result = evaluate_task_with_solver(
                model, tokenizer, args.variant, task, args.model, max_samples=args.smoke_test
            )
            results.append(result)
            logger.info(f"Task {task} result: {result}")
        except Exception as e:
            logger.error(f"Error evaluating task {task}: {e}", exc_info=True)

    # Save results
    results_csv = results_dir / "experiment_matrix.csv"
    results_df = pd.DataFrame(results)

    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        results_df = pd.concat([existing, results_df], ignore_index=True)

    results_df.to_csv(results_csv, index=False)
    logger.info(f"Results saved to {results_csv}")

    # Print summary
    print("\n" + "="*60)
    print("SOLVER EVALUATION SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)


if __name__ == "__main__":
    main()
