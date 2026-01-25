#!/usr/bin/env python3
"""
Debug script to inspect formatted examples and verify train_on_responses_only markers.
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt, get_chat_template
import json
import sys
from datasets import Dataset

def load_and_convert_dataset(data_path):
    """Load local JSON dataset and convert to ShareGPT format."""
    print(f"Loading dataset from {data_path}...")

    with open(data_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    print(f"Loaded {len(articles)} articles")

    conversations = []

    # Take only first 3 for debugging
    for idx, article in enumerate(articles[:3]):
        title = article.get("title", "").replace("_", " ").title()
        content = article.get("content", "")
        abstract = article.get("abstract", "")

        if not content or not abstract:
            continue
        if not title:
            title = "Untitled Paper"

        # Create simple user prompt
        user_prompt = f"Create an abstract for: {title}\n\nContent: {content[:200]}..."

        conversation = [
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": abstract}
        ]
        conversations.append({"conversations": conversation})

    print(f"Created {len(conversations)} conversations for debugging")
    return Dataset.from_list(conversations)


def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_formatting.py <model_size> <data_path>")
        print("Example: python debug_formatting.py 3b ./veri.json")
        sys.exit(1)

    model_size = sys.argv[1]
    data_path = sys.argv[2]

    print("=" * 80)
    print(f"DEBUGGING CHAT TEMPLATE FOR {model_size.upper()} MODEL")
    print("=" * 80)

    # Model configs
    MODEL_CONFIGS = {
        "1b": "unsloth/Llama-3.2-1B-Instruct",
        "3b": "unsloth/Llama-3.2-3B-Instruct",
        "8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
    }

    if model_size not in MODEL_CONFIGS:
        print(f"ERROR: Invalid model size. Choose: 1b, 3b, or 8b")
        sys.exit(1)

    # Load tokenizer only (fast)
    print(f"\nLoading tokenizer for {MODEL_CONFIGS[model_size]}...")
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIGS[model_size],
        max_seq_length=2048,
        load_in_4bit=False,  # Don't need to load model weights
        dtype=None,
    )

    # Configure template based on model size
    if model_size == "8b":
        print(f"\n8B Model - Using Alpaca template (skipping this debug)")
        print("For 8B, we don't use train_on_responses_only, so no issue expected")
        return
    else:
        print(f"\n{model_size.upper()} Model - Applying Llama-3.1 chat template...")
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3.1",
        )

    print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

    # Load dataset
    dataset = load_and_convert_dataset(data_path)
    dataset = standardize_sharegpt(dataset)

    # Format prompts
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Print first example in detail
    print("\n" + "=" * 80)
    print("FORMATTED EXAMPLE #1")
    print("=" * 80)
    example_text = dataset[0]["text"]
    print(example_text)
    print("=" * 80)

    # Check for markers
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    print("\n" + "=" * 80)
    print("MARKER CHECK")
    print("=" * 80)
    print(f"Instruction part: {repr(instruction_part)}")
    print(f"Response part: {repr(response_part)}")
    print()
    print(f"Instruction part found in text: {instruction_part in example_text}")
    print(f"Response part found in text: {repr(response_part in example_text)}")

    if instruction_part in example_text:
        print(f"✓ Instruction marker FOUND at position: {example_text.index(instruction_part)}")
    else:
        print(f"✗ Instruction marker NOT FOUND")
        print(f"Searching for variations...")
        if "<|start_header_id|>user<|end_header_id|>" in example_text:
            print(f"  Found without \\n\\n: <|start_header_id|>user<|end_header_id|>")
            # Find what comes after
            idx = example_text.index("<|start_header_id|>user<|end_header_id|>")
            print(f"  Next 50 chars: {repr(example_text[idx:idx+80])}")

    if response_part in example_text:
        print(f"✓ Response marker FOUND at position: {example_text.index(response_part)}")
    else:
        print(f"✗ Response marker NOT FOUND")
        print(f"Searching for variations...")
        if "<|start_header_id|>assistant<|end_header_id|>" in example_text:
            print(f"  Found without \\n\\n: <|start_header_id|>assistant<|end_header_id|>")
            # Find what comes after
            idx = example_text.index("<|start_header_id|>assistant<|end_header_id|>")
            print(f"  Next 50 chars: {repr(example_text[idx:idx+80])}")

    # Show all header markers in the text
    print("\n" + "=" * 80)
    print("ALL HEADER MARKERS IN TEXT")
    print("=" * 80)
    lines = example_text.split('\n')
    for i, line in enumerate(lines):
        if 'header_id' in line:
            print(f"Line {i}: {repr(line)}")
            if i + 1 < len(lines):
                print(f"Line {i+1}: {repr(lines[i+1])}")

    # Tokenize and show label masking
    print("\n" + "=" * 80)
    print("TOKENIZATION CHECK")
    print("=" * 80)

    # Tokenize the text
    tokenized = tokenizer(example_text, return_tensors="pt")
    print(f"Total tokens: {len(tokenized['input_ids'][0])}")
    print(f"First 20 tokens: {tokenized['input_ids'][0][:20].tolist()}")

    # Simulate what train_on_responses_only would do
    print("\n" + "=" * 80)
    print("SIMULATING train_on_responses_only")
    print("=" * 80)

    # Split by response_part
    if response_part in example_text:
        parts = example_text.split(response_part)
        print(f"Split into {len(parts)} parts by response marker")
        print(f"Part 0 (should be masked): {len(parts[0])} chars")
        if len(parts) > 1:
            print(f"Part 1 (should be trained): {len(parts[1])} chars")
            print(f"Preview of training part: {parts[1][:200]}...")
    else:
        print("❌ PROBLEM: Cannot split by response_part - this is why all labels are -100!")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if instruction_part not in example_text or response_part not in example_text:
        print("❌ The instruction_part or response_part markers don't match the template!")
        print("   You need to update the markers in llama-ft.py to match what's actually generated.")
        print("\n   Try these in train_on_responses_only:")

        # Suggest correct markers
        if "<|start_header_id|>user<|end_header_id|>" in example_text:
            idx = example_text.index("<|start_header_id|>user<|end_header_id|>")
            next_chars = example_text[idx+len("<|start_header_id|>user<|end_header_id|>"):idx+len("<|start_header_id|>user<|end_header_id|>")+10]
            print(f"   instruction_part = '<|start_header_id|>user<|end_header_id|>{repr(next_chars)}'")

        if "<|start_header_id|>assistant<|end_header_id|>" in example_text:
            idx = example_text.index("<|start_header_id|>assistant<|end_header_id|>")
            next_chars = example_text[idx+len("<|start_header_id|>assistant<|end_header_id|>"):idx+len("<|start_header_id|>assistant<|end_header_id|>")+10]
            print(f"   response_part = '<|start_header_id|>assistant<|end_header_id|>{repr(next_chars)}'")
    else:
        print("✓ Markers look correct! The issue might be elsewhere.")
        print("  Check if the dataset has valid content after the response marker.")


if __name__ == "__main__":
    main()
