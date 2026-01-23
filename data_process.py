# -*- coding: utf-8 -*-
"""
Dataset Preprocessing Script for Llama 3.2 Fine-Tuning

Preprocesses veri.json once and saves to disk for faster parallel training.
Avoids redundant processing when running multiple models simultaneously.

Usage:
    python data_process.py --input ./veri.json --output ./processed_dataset
    python data_process.py --input ./veri.json --output ./processed_dataset --num_proc 16
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from datasets import Dataset
from unsloth.chat_templates import standardize_sharegpt


# Prompt templates (same as llama-ft.py)
SYSTEM_PROMPT = """You are an expert scientific abstract writer with deep knowledge of academic writing conventions."""


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess dataset for Llama 3.2 fine-tuning")
    parser.add_argument("--input", default="./veri.json",
                        help="Path to input JSON file (default: ./veri.json)")
    parser.add_argument("--output", default="./processed_dataset",
                        help="Output directory for processed dataset (default: ./processed_dataset)")
    parser.add_argument("--num_proc", type=int, default=16,
                        help="Number of processes for standardization (default: 16)")
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
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading dataset from {data_path}...")
    start_time = time.time()

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON file: {e}")
        sys.exit(1)

    load_time = time.time() - start_time
    print(f"Loaded {len(articles)} articles in {load_time:.2f} seconds")

    conversations = []
    skipped = 0

    print("Converting to ShareGPT format...")
    for idx, article in enumerate(articles):
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(articles)} articles...")

        # Safe field extraction with defaults
        title = article.get("title", "").replace("_", " ").title()
        content = article.get("content", "")
        abstract = article.get("abstract", "")

        # Skip articles with empty content or abstract
        if not content or not content.strip():
            skipped += 1
            continue
        if not abstract or not abstract.strip():
            skipped += 1
            continue
        if not title or not title.strip():
            title = "Untitled Paper"

        conversation = [
            {"from": "system", "value": SYSTEM_PROMPT},
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


def main():
    args = parse_args()

    print("=" * 60)
    print("DATASET PREPROCESSING FOR LLAMA 3.2 FINE-TUNING")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Num processes: {args.num_proc}")
    print()

    overall_start = time.time()

    # Load and convert dataset
    dataset = load_and_convert_dataset(args.input)

    # Standardize ShareGPT format (this is the slow part)
    print(f"\nStandardizing ShareGPT format (num_proc={args.num_proc})...")
    print("This may take ~10-15 minutes depending on your CPU...")
    standardize_start = time.time()

    dataset = standardize_sharegpt(dataset, num_proc=args.num_proc)

    standardize_time = time.time() - standardize_start
    print(f"Standardization complete in {standardize_time:.2f} seconds ({standardize_time/60:.2f} minutes)")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Save to disk
    print(f"\nSaving processed dataset to {args.output}...")
    save_start = time.time()
    dataset.save_to_disk(args.output)
    save_time = time.time() - save_start

    overall_time = time.time() - overall_start

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(dataset)}")
    print(f"Output directory: {args.output}")
    print(f"\nTiming:")
    print(f"  Standardization: {standardize_time:.2f}s ({standardize_time/60:.2f} min)")
    print(f"  Save to disk: {save_time:.2f}s")
    print(f"  Total time: {overall_time:.2f}s ({overall_time/60:.2f} min)")

    # Calculate directory size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(args.output):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)

    size_mb = total_size / (1024 * 1024)
    print(f"  Directory size: {size_mb:.2f} MB")

    print("\nTo use the preprocessed dataset:")
    print(f"  python llama-ft.py --model 3b --use_processed --processed_path {args.output}")


if __name__ == "__main__":
    main()
