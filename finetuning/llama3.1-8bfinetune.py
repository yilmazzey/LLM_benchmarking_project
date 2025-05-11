#!/usr/bin/env python
# -- coding: utf-8 --
"""
Fine-tune Llama-3.2-3B-Instruct with LoRA on local ShareGPT JSON dataset for academic abstract generation.
Usage: python llama3_2_alpacaft_local_v5.py --dataset data.json [--output output_dir] [--epochs 3] [--batch-size 4]
"""

import os
import json
import torch
import argparse
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer
from transformers.trainer_callback import EarlyStoppingCallback

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.2-3B with LoRA")
    parser.add_argument("--dataset", type=str, required=True, help="Path to ShareGPT JSON file")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    return parser.parse_args()

def main():
    args = parse_args()

    max_seq_length = 8192
    model_name = "unsloth/Meta-Llama-3.1-8B"
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Unified system prompt from abstract_generation.py
    system_instruction = """You are an expert scientific abstract writer with deep knowledge of academic writing conventions. Your task is to create a comprehensive, accurate, and professional abstract for the given research paper."""

    with open(args.dataset, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    for item in raw_data:
        if "conversations" in item:
            conv = item["conversations"]
            for i, msg in enumerate(conv):
                if msg["role"] == "user" and (i == len(conv) - 1 or conv[i + 1]["role"] != "assistant"):
                    conv.insert(i + 1, {"role": "assistant", "content": ""})

    dataset = Dataset.from_list(raw_data)

    def apply_chat_template(example):
        current_conv = example["conversations"]
        
        # Start with the unified system prompt
        processed_conv = [{"role": "system", "content": system_instruction}]

        # Add user and assistant turns, skipping any original system messages
        for message in current_conv:
            if message["role"] == "user":
                # NOTE: For best results, the content of this user message in your
                # output_new_sharegpt.json should ideally match the structure of the
                # user_prompt in abstract_generation.py (including title, sections, and detailed instructions).
                processed_conv.append(message)
            elif message["role"] == "assistant":
                processed_conv.append(message)
        
        return {"text": tokenizer.apply_chat_template(processed_conv, tokenize=False)}

    dataset = dataset.train_test_split(test_size=0.2, seed=3407)
    train_dataset = dataset["train"].map(apply_chat_template, num_proc=1)
    val_dataset = dataset["test"].map(apply_chat_template, num_proc=1)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=150,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("Starting training...")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name}, Total Memory: {gpu.total_memory / 1024**3:.2f} GB")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    FastLanguageModel.for_inference(model)
    try:
        with open("test.json", "r", encoding="utf-8") as f:
            paper = json.load(f)
            if isinstance(paper, list):
                paper = paper[0]
    except:
        paper = {
            "title": "Advances in NLP for Scientific Literature",
            "content": "This paper explores transformer-based methods for extracting information from research papers."
        }

    # User prompt for inference, aligned with abstract_generation.py's structure
    user_prompt_for_inference = f"""
Create a concise and informative abstract for the following scientific paper:

Title: {paper['title']}

Paper content:
<content>
{paper['content']}
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

Output ONLY the abstract text with no additional comments, explanations, or metadata.
"""

    messages = [
        {"role": "system", "content": system_instruction}, # This will now use the new unified system_instruction
        {"role": "user", "content": user_prompt_for_inference},
    ]

    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    print("Generating abstract:")
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    model.generate(input_ids=inputs, streamer=streamer, max_new_tokens=512, temperature=0.7)

    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
