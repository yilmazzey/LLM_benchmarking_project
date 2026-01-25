# -*- coding: utf-8 -*-
"""
Llama 3.2 Fine-Tuning Script for Abstract Generation

Fine-tunes Llama 3.2 (1B or 3B) on academic papers for abstract generation.
Uses Unsloth for efficient training with LoRA adapters.

Usage:
    python llama-ft.py --model 3b --epochs 3 --data_path ./veri.json
    python llama-ft.py --model 1b --epochs 3 --data_path ./veri.json
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt    # still needed

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from datasets import Dataset
from transformers import TrainerCallback, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer


# Model configurations
MODEL_CONFIGS = {
    "1b": {"model_name": "unsloth/Llama-3.2-1B-Instruct", "output_name": "llama-3.2-1b-abstract-lora"},
    "3b": {"model_name": "unsloth/Llama-3.2-3B-Instruct", "output_name": "llama-3.2-3b-abstract-lora"},
    "8b": {"model_name": "unsloth/Meta-Llama-3.1-8B-Instruct", "output_name": "llama-3.1-8b-abstract-lora"},
}

# Prompt templates
SYSTEM_PROMPT = """You are an expert scientific abstract writer with deep knowledge of academic writing conventions."""


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 for abstract generation")
    parser.add_argument("--model", choices=["1b", "3b", "8b"], default="3b",
                        help="Model size to fine-tune: 1b, 3b, or 8b (default: 3b)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--output_dir", default="./lora_adapters",
                        help="Directory to save LoRA adapters (default: ./lora_adapters)")
    parser.add_argument("--data_path", default="./veri.json",
                        help="Path to the training data JSON file (default: ./veri.json)")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                        help="Maximum sequence length (default: 8192)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size (default: 1)")
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token for gated models (optional)")
    parser.add_argument("--eval_split", type=float, default=0.0,
                        help="Fraction of data to use for evaluation (default: 0.0, no eval)")
    parser.add_argument("--use_processed", action="store_true",
                        help="Use preprocessed dataset from disk (faster for parallel training)")
    parser.add_argument("--processed_path", default="./processed_dataset",
                        help="Path to preprocessed dataset directory (default: ./processed_dataset)")
    return parser.parse_args()


def create_user_prompt(title, content):
    """Create the user prompt for abstract generation."""
    return f"""Create a concise and informative abstract for the following scientific paper:

Title: {title}

Paper content:
<content>
{content}
</content>

Your abstract should:
1. Begin with a clear statement of the research problem or objective
2. Briefly describe the methodology used
3. Summarize the key findings and results
4. State the main conclusions and implications
5. Be self-contained and understandable on its own
6. Use formal academic language and avoid first-person pronouns
7. Be between 150-250 words
8. Avoid citations, references, or detailed numerical data
9. Focus on the most significant aspects of the research

Output ONLY the abstract text."""


def load_and_convert_dataset(data_path):
    """Load local JSON dataset and convert to ShareGPT format."""
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading dataset from {data_path}...")

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON file: {e}")
        sys.exit(1)

    print(f"Loaded {len(articles)} articles")

    conversations = []
    skipped = 0

    for idx, article in enumerate(articles):
        title = article.get("title", "").replace("_", " ").title()
        content = article.get("content", "")
        abstract = article.get("abstract", "")

        if not content or not content.strip():
            skipped += 1
            continue
        if not abstract or not abstract.strip():
            skipped += 1
            continue
        if not title or not title.strip():
            title = "Untitled Paper"

        conversation = [
            {"from": "human", "value": create_user_prompt(title, content)},
            {"from": "gpt", "value": abstract}
        ]
        conversations.append({"conversations": conversation})

    if skipped > 0:
        print(f"WARNING: Skipped {skipped} articles with empty content or abstract")

    if len(conversations) == 0:
        print("ERROR: No valid articles found in dataset")
        sys.exit(1)

    print(f"Valid articles for training: {len(conversations)}")
    return Dataset.from_list(conversations)


class LossLoggingCallback(TrainerCallback):
    """Callback to print loss at each step."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            lr = logs.get("learning_rate", 0)
            print(f"  Step {step:4d} | Loss: {loss:.4f} | LR: {lr:.2e}")


def check_cuda():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        print("If you have a GPU, make sure CUDA drivers are installed.")
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"\nGPU detected: {gpu_name}")
    print(f"GPU count: {gpu_count}")
    print(f"GPU memory: {gpu_memory:.1f} GB")

    return gpu_name, gpu_memory


def main():
    args = parse_args()

    print("=" * 60)
    print("LLAMA 3.2 FINE-TUNING FOR ABSTRACT GENERATION")
    print("=" * 60)

    gpu_name, gpu_memory = check_cuda()

    model_config = MODEL_CONFIGS[args.model]
    print(f"\nModel: {model_config['model_name']}")
    print(f"Epochs: {args.epochs}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"Data path: {args.data_path}")
    if args.eval_split > 0:
        print(f"Eval split: {args.eval_split * 100:.1f}%")

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = None
    load_in_4bit = True

    print(f"\nLoading model: {model_config['model_name']}...")
    model_kwargs = {
        "model_name": model_config['model_name'],
        "max_seq_length": args.max_seq_length,
        "dtype": dtype,
        "load_in_4bit": load_in_4bit,
    }
    if args.hf_token:
        model_kwargs["token"] = args.hf_token

    model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print(f"Using native chat template")
    print(f"eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

    if args.use_processed:
        if not os.path.exists(args.processed_path):
            print(f"ERROR: Preprocessed dataset not found at {args.processed_path}")
            print("Run: python data_process.py --input ./veri.json --output ./processed_dataset")
            sys.exit(1)
        print(f"\nLoading preprocessed dataset from {args.processed_path}...")
        dataset = Dataset.load_from_disk(args.processed_path)
        print(f"Loaded {len(dataset)} samples (preprocessed)")
    else:
        print("\nWARNING: Processing dataset from scratch. This is slow for parallel training.")
        print("Consider using: python data_process.py --input ./veri.json")
        print("Then use: --use_processed flag\n")
        dataset = load_and_convert_dataset(args.data_path)
        dataset = standardize_sharegpt(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    eval_dataset = None
    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=3407)
        dataset = split["train"]
        eval_dataset = split["test"]
        print(f"\nDataset split: {len(dataset)} train, {len(eval_dataset)} eval")
    else:
        print(f"\nDataset prepared: {len(dataset)} samples (no eval split)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{model_config['output_name']}_ep{args.epochs}_{timestamp}"
    output_path = os.path.join(args.output_dir, output_name)
    os.makedirs(output_path, exist_ok=True)

    print(f"Output path: {output_path}")

    total_samples = len(dataset)
    steps_per_epoch = total_samples // (args.batch_size * args.gradient_accumulation)
    total_steps = steps_per_epoch * args.epochs
    print(f"\nTraining steps: ~{steps_per_epoch} per epoch, ~{total_steps} total")

    sft_config_kwargs = {
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "warmup_ratio": 0.03,
        "num_train_epochs": args.epochs,
        "learning_rate": 2e-4,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": 3407,
        "output_dir": output_path,
        "report_to": "none",
        "bf16": True,
        "eos_token": None,
    }

    if eval_dataset is not None:
        sft_config_kwargs["eval_strategy"] = "epoch"
        sft_config_kwargs["per_device_eval_batch_size"] = args.batch_size
        sft_config_kwargs["load_best_model_at_end"] = True
        sft_config_kwargs["metric_for_best_model"] = "eval_loss"
        sft_config_kwargs["greater_is_better"] = False

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": dataset,
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_length,
        "data_collator": DataCollatorForSeq2Seq(tokenizer=tokenizer),
        "dataset_num_proc": 2,
        "packing": False,
        "args": SFTConfig(**sft_config_kwargs),
        "callbacks": [LossLoggingCallback()],
    }

    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = SFTTrainer(**trainer_kwargs)

    # ────────────────────────────────────────────────────────────────
    #   Removed: train_on_responses_only(...)
    #   Reason: Causes over-masking (-100 everywhere) on Llama-3.2-Instruct
    #   Native chat template + TRL already handles assistant-only loss correctly
    # ────────────────────────────────────────────────────────────────

    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"\nReserved memory before training: {start_gpu_memory} GB")

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / gpu_memory * 100, 3)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Training time: {trainer_stats.metrics['train_runtime']/60:.2f} minutes")
    print(f"Final train loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")
    if eval_dataset is not None and 'eval_loss' in trainer_stats.metrics:
        print(f"Final eval loss: {trainer_stats.metrics['eval_loss']}")
    print(f"Peak reserved memory: {used_memory} GB ({used_percentage:.1f}%)")
    print(f"Peak memory for training: {used_memory_for_lora} GB")

    final_output_path = os.path.join(output_path, "final")
    os.makedirs(final_output_path, exist_ok=True)
    print(f"\nSaving final LoRA adapters to {final_output_path}...")
    model.save_pretrained(final_output_path)
    tokenizer.save_pretrained(final_output_path)

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"\nCheckpoints saved at: {output_path}")
    print(f"Final model saved at: {final_output_path}")
    print("\nTo use the fine-tuned model:")
    print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{final_output_path}')")


if __name__ == "__main__":
    main()