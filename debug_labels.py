#!/usr/bin/env python3
"""
Debug script to check actual label masking in the training process.
This will show us EXACTLY what's being masked and what's being trained on.
"""

import json
import sys
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset

# Prompt template
def create_user_prompt(title, content):
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

def load_model_and_tokenizer(model_size="3b"):
    """Load model and tokenizer."""
    model_map = {
        "1b": "unsloth/Llama-3.2-1B-Instruct",
        "3b": "unsloth/Llama-3.2-3B-Instruct",
        "8b": "unsloth/Meta-Llama-3.1-8B-Instruct"
    }

    model_name = model_map.get(model_size)
    if not model_name:
        print(f"ERROR: Invalid model size '{model_size}'")
        sys.exit(1)

    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    # Apply chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    return model, tokenizer

def test_single_example(model_size="3b", data_path="./veri.json"):
    """Test label masking on a single example."""

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_size)

    # Load one example from dataset
    print(f"\nLoading data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    # Use first valid article
    article = articles[0]
    title = article.get("title", "").replace("_", " ").title()
    content = article.get("content", "")
    abstract = article.get("abstract", "")

    # Create conversation (NO SYSTEM MESSAGE)
    conversation = [
        {"from": "human", "value": create_user_prompt(title, content)},
        {"from": "gpt", "value": abstract}
    ]

    print("\n" + "="*80)
    print("CONVERSATION STRUCTURE")
    print("="*80)
    print(f"Number of messages: {len(conversation)}")
    for i, msg in enumerate(conversation):
        print(f"\nMessage {i+1}:")
        print(f"  From: {msg['from']}")
        print(f"  Value length: {len(msg['value'])} chars")
        print(f"  First 100 chars: {msg['value'][:100]}...")

    # Create dataset
    dataset = Dataset.from_list([{"conversations": conversation}])

    # Apply chat template
    print("\n" + "="*80)
    print("APPLYING CHAT TEMPLATE")
    print("="*80)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Get the formatted text
    formatted_text = dataset[0]["text"]

    print(f"Formatted text length: {len(formatted_text)} chars")
    print("\nFormatted text (first 1000 chars):")
    print("-"*80)
    print(formatted_text[:1000])
    print("-"*80)

    # Check for markers
    print("\n" + "="*80)
    print("CHECKING FOR MASKING MARKERS")
    print("="*80)

    user_marker = "<|start_header_id|>user<|end_header_id|>\n\n"
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    print(f"User marker present: {user_marker in formatted_text}")
    print(f"Assistant marker present: {assistant_marker in formatted_text}")

    if user_marker in formatted_text:
        user_pos = formatted_text.find(user_marker)
        print(f"User marker position: {user_pos}")

    if assistant_marker in formatted_text:
        assistant_pos = formatted_text.find(assistant_marker)
        print(f"Assistant marker position: {assistant_pos}")

    # Now apply train_on_responses_only
    print("\n" + "="*80)
    print("APPLYING train_on_responses_only")
    print("="*80)

    trainer = train_on_responses_only(
        tokenizer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # Tokenize and check labels
    print("\nTokenizing example...")
    tokens = tokenizer(formatted_text, return_tensors="pt")
    input_ids = tokens["input_ids"][0]

    print(f"Number of tokens: {len(input_ids)}")

    # Apply the data collator to see what gets masked
    print("\nApplying data collator (train_on_responses_only masking)...")

    # Create a fake batch
    batch = [{"input_ids": input_ids.tolist(), "attention_mask": [1]*len(input_ids)}]

    # Apply collator
    result = trainer(batch)

    labels = result["labels"][0]

    # Count masked vs unmasked tokens
    masked_count = sum(1 for label in labels if label == -100)
    unmasked_count = sum(1 for label in labels if label != -100)

    print(f"\nLabel statistics:")
    print(f"  Total tokens: {len(labels)}")
    print(f"  Masked tokens (-100): {masked_count} ({100*masked_count/len(labels):.1f}%)")
    print(f"  Training tokens: {unmasked_count} ({100*unmasked_count/len(labels):.1f}%)")

    if unmasked_count == 0:
        print("\n" + "="*80)
        print("ERROR: ALL TOKENS ARE MASKED!")
        print("="*80)
        print("This means train_on_responses_only is not finding the assistant response.")
        print("\nDebugging info:")
        print(f"  - Instruction marker: {repr(trainer.instruction_part)}")
        print(f"  - Response marker: {repr(trainer.response_part)}")
        print(f"\nSearching for markers in formatted text...")
        if trainer.instruction_part in formatted_text:
            print(f"  ✓ Instruction marker found at position {formatted_text.find(trainer.instruction_part)}")
        else:
            print(f"  ✗ Instruction marker NOT FOUND")

        if trainer.response_part in formatted_text:
            print(f"  ✓ Response marker found at position {formatted_text.find(trainer.response_part)}")
        else:
            print(f"  ✗ Response marker NOT FOUND")

        return False
    else:
        print("\n" + "="*80)
        print("SUCCESS: Labels are being created correctly")
        print("="*80)

        # Show what's being trained on
        unmasked_tokens = [token_id for token_id, label in zip(input_ids.tolist(), labels) if label != -100]
        print(f"\nFirst 20 training tokens (unmasked):")
        print(tokenizer.decode(unmasked_tokens[:20]))

        return True

if __name__ == "__main__":
    model_size = sys.argv[1] if len(sys.argv) > 1 else "3b"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "./veri.json"

    print("="*80)
    print("LABEL MASKING DEBUG SCRIPT")
    print("="*80)
    print(f"Model size: {model_size}")
    print(f"Data path: {data_path}")

    success = test_single_example(model_size, data_path)

    if not success:
        print("\n" + "="*80)
        print("FAILED: The masking is not working correctly")
        print("="*80)
        sys.exit(1)
    else:
        print("\n" + "="*80)
        print("SUCCESS: The masking is working correctly")
        print("="*80)
        sys.exit(0)
