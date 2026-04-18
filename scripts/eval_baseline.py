#!/usr/bin/env python3
"""Baseline evaluation: A1 (qwen25 CoT), B1 (qwen3 CoT), C1 (qwen3 thinking)."""
import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import balanced_accuracy_score
from rouge_score import rouge_scorer

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
    "qwen3-thinking": "Qwen/Qwen3-1.7B",
}

# Task configuration
TASK_CONFIG = {
    "hearsay": {
        "csv_path": "data/legalbench/hearsay__test.csv",
        "type": "binary",
        "label_col": "answer",
        "question_col": "text",
        "true_labels": ["Yes", "No"],
    },
    "personal_jurisdiction": {
        "csv_path": "data/legalbench/personal_jurisdiction__test.csv",
        "type": "binary",
        "label_col": "answer",
        "question_col": "text",
        "true_labels": ["Yes", "No"],
    },
    "rule_qa": {
        "csv_path": "data/legalbench/rule_qa__test.csv",
        "type": "generation",
        "label_col": "answer",
        "question_col": "text",
    },
    "textualism_tool": {
        "csv_path": "data/legalbench/textualism_tool__test.csv",
        "type": "binary",
        "label_col": "answer",
        "question_col": "text",
        "true_labels": ["Yes", "No"],
    },
}


def load_model_and_tokenizer(model_key: str, enable_thinking: bool = False):
    """Load model with 4-bit quantization."""
    model_id = MODEL_IDS[model_key]

    logger.info(f"Loading {model_key} from {model_id}")

    # 4-bit quantization config
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

    logger.info(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    return model, tokenizer


def build_prompt(task: str, question: str) -> str:
    """Build task-specific prompt."""
    config = TASK_CONFIG[task]

    if config["type"] == "binary":
        template = f"""You are a legal reasoning AI. Answer the following question with ONLY "Yes" or "No".

Question: {question}

Answer: """
    else:  # generation
        template = f"""You are a legal reasoning AI. Answer the following question with a clear, concise explanation.

Question: {question}

Answer: """

    return template


def generate_answer(
    model, tokenizer, prompt: str, max_new_tokens: int = 128,
    enable_thinking: bool = False,
) -> str:
    """Generate answer from model using chat template."""
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)

    gen_tokens = max_new_tokens * 8 if enable_thinking else max_new_tokens

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=gen_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    if enable_thinking and "</think>" in response:
        response = response.split("</think>", 1)[1]
    return response.strip()


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
    # Convert to numeric (0/1 for binary, or direct string match for generation)
    valid_preds = []
    valid_refs = []

    for pred, ref in zip(predictions, references):
        if pred != "PARSE_FAIL":
            valid_preds.append(1 if pred == "Yes" else 0)
            valid_refs.append(1 if ref == "Yes" else 0)

    if not valid_preds:
        return 0.0

    return balanced_accuracy_score(valid_refs, valid_preds)


def evaluate_task(
    model, tokenizer, task: str, model_key: str, max_samples: int = None,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    """Evaluate on a single task."""
    config = TASK_CONFIG[task]
    csv_path = Path(config["csv_path"])

    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return {"error": f"CSV not found: {csv_path}"}

    # Load dataset
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)

    logger.info(f"Evaluating {task} with {len(df)} samples")

    predictions = []
    references = []
    parse_successes = 0

    for idx, row in df.iterrows():
        question = row[config["question_col"]]
        reference = row[config["label_col"]]

        prompt = build_prompt(task, question)
        response = generate_answer(model, tokenizer, prompt, enable_thinking=enable_thinking)

        if config["type"] == "binary":
            pred = parse_binary_answer(response, config["true_labels"])
            if pred != "PARSE_FAIL":
                parse_successes += 1
        else:
            pred = response

        predictions.append(pred)
        references.append(reference)

        if (idx + 1) % 10 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)}")

    # Compute metrics
    parse_rate = parse_successes / len(df) if config["type"] == "binary" else 1.0

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
        "accuracy": accuracy,
        "n": len(df),
        "parse_rate": parse_rate,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation (A1, B1, C1)")
    parser.add_argument(
        "--model",
        choices=["qwen25", "qwen3", "qwen3-thinking"],
        default="qwen25",
        help="Model to evaluate",
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
        help="Max samples for smoke test (default: full evaluation)",
    )
    args = parser.parse_args()

    # Setup directories
    results_dir = Path("results")
    log_dir = results_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = log_dir / f"baseline_{args.model}_{args.task}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    logger.info(f"Starting baseline evaluation: model={args.model}, task={args.task}")

    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Determine tasks to evaluate
    tasks = [args.task] if args.task != "all" else list(TASK_CONFIG.keys())

    # Evaluate tasks
    results = []
    for task in tasks:
        try:
            result = evaluate_task(
                model, tokenizer, task, args.model, max_samples=args.smoke_test,
                enable_thinking=(args.model == "qwen3-thinking"),
            )
            results.append(result)
            logger.info(f"Task {task} result: {result}")
        except Exception as e:
            logger.error(f"Error evaluating task {task}: {e}", exc_info=True)

    # Save results
    results_csv = results_dir / "baseline.csv"
    results_df = pd.DataFrame(results)

    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        results_df = pd.concat([existing, results_df], ignore_index=True)

    results_df.to_csv(results_csv, index=False)
    logger.info(f"Results saved to {results_csv}")

    # Print summary
    print("\n" + "="*60)
    print("BASELINE EVALUATION SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)


if __name__ == "__main__":
    main()
