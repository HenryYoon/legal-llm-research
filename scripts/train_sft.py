#!/usr/bin/env python3
"""SFT training script using transformers + PEFT LoRA (fallback from Unsloth)."""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

# Try Unsloth first, fallback to transformers + PEFT
UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("Unsloth not available, using transformers + PEFT fallback")
    # Check bfloat16 support
    def is_bfloat16_supported():
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_for_training(config: Dict[str, Any]):
    """Load model with LoRA (Unsloth if available, otherwise transformers + PEFT)."""
    model_name = config.get("model_name")
    max_seq_length = config.get("max_seq_length", 2048)
    load_in_4bit = config.get("load_in_4bit", True)

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Using Unsloth: {UNSLOTH_AVAILABLE}")

    if UNSLOTH_AVAILABLE:
        # Use Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )

        # Add LoRA adapters
        lora_target_modules = config.get("lora_target_modules", [])
        lora_r = config.get("lora_r", 16)
        lora_alpha = config.get("lora_alpha", 32)

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing=config.get("gradient_checkpointing", True),
            use_rslora=False,
        )
    else:
        # Fallback: transformers + PEFT
        lora_target_modules = config.get("lora_target_modules", [])
        lora_r = config.get("lora_r", 16)
        lora_alpha = config.get("lora_alpha", 32)

        # Setup 4bit quantization
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",
        )

        # Setup LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)

        # Enable gradient checkpointing
        if config.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()

    logger.info(f"Model loaded with LoRA (r={lora_r}, alpha={lora_alpha})")
    return model, tokenizer


def load_dataset_for_sft(dataset_path: str) -> Any:
    """Load JSONL dataset with messages format for SFT."""
    logger.info(f"Loading dataset from {dataset_path}")

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    return dataset


def format_chat_template(example: Dict[str, Any], tokenizer, max_seq_length: int = 2048) -> Dict[str, Any]:
    """Format example with tokenizer's chat template and tokenize."""
    messages = example.get("messages", [])

    # Use tokenizer's chat template to format
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize the text
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def train_sft(config_path: str):
    """Main SFT training function."""
    config = load_config(config_path)

    logger.info(f"Starting SFT training with config: {config_path}")

    # Setup output directory
    output_dir = config.get("output_dir", "models/qwen25-1.5b-legal-sft")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    try:
        model, tokenizer = load_model_for_training(config)
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return

    # Setup dataset
    try:
        dataset = load_dataset_for_sft(config.get("dataset_path"))
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        return

    # Format dataset
    logger.info("Formatting dataset with chat template...")
    max_seq_length = config.get("max_seq_length", 2048)
    dataset = dataset.map(
        lambda ex: format_chat_template(ex, tokenizer, max_seq_length),
        remove_columns=dataset.column_names,
    )

    # Setup training arguments (ensure proper types)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(config.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        warmup_steps=100,
        max_grad_norm=1.0,
        weight_decay=0.01,
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=100,
        save_strategy="steps",
        eval_strategy="no",  # No validation in baseline
        seed=42,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    logger.info(f"Training args:\n{training_args}")

    # Initialize data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # No masked language modeling
    )

    # Initialize trainer (using Trainer instead of SFTTrainer for compatibility)
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        logger.info("Starting training...")
        trainer.train()

        # Save final model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info("Training completed successfully!")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("OOM detected, attempting fallback: batch//2 + grad_accum*2")
            # Implement fallback
            batch_size = config.get("per_device_train_batch_size", 4) // 2
            grad_accum = config.get("gradient_accumulation_steps", 4) * 2

            training_args.per_device_train_batch_size = batch_size
            training_args.gradient_accumulation_steps = grad_accum

            logger.info(f"Fallback: batch_size={batch_size}, grad_accum={grad_accum}")

            try:
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=dataset,
                    data_collator=data_collator,
                )
                trainer.train()
                trainer.save_model(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info("Training completed with fallback settings!")
            except Exception as e2:
                logger.error(f"Training failed even with fallback: {e2}", exc_info=True)
        else:
            logger.error(f"Training failed: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="SFT training with Unsloth + TRL")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    # Verify config exists
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return

    train_sft(args.config)


if __name__ == "__main__":
    main()
