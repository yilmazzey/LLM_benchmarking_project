#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen 2.5 7B Fine-tuning Script (Local Dataset Version)

This script fine-tunes Qwen 2.5 7B model on conversational data using an Alpaca-style approach.
This version supports loading local JSON files in ShareGPT format.

Usage:
  python qwen_2.57b.py [--dataset FILE.json] [--output OUTPUT] [--steps STEPS] [--epochs EPOCHS] [--batch-size BATCH_SIZE]

Example:
  python qwen_2.57b.py --dataset papers_sharegpt.json --epochs 10 --batch-size 2
"""

#==============================================================================
# IMPORTS
#==============================================================================
import os
import json
import torch
import argparse
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer

#==============================================================================
# ARGUMENT PARSING
#==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 7B model on local conversational data")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Path to local JSON file in ShareGPT format")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory for saving model (default: outputs)")
    parser.add_argument("--steps", type=int, default=300,
                        help="Number of training steps (default: 300)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (overrides steps if specified)")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Per-device training batch size (default: 12) You can try 10-8-6")
    return parser.parse_args()
# Overfitting'i önlemek için validation loss'u takip etmeniz ve early stopping kullanmanız faydalı olacaktır.

#==============================================================================
# MAIN FUNCTION
#==============================================================================
def main():
    #==========================================================================
    # CONFIGURATION
    #==========================================================================
    # Parse arguments
    args = parse_args()
    
    # Model configuration
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
    model_name = "unsloth/Qwen2.5-7B"
    dataset_path = args.dataset
    output_dir = args.output
    chat_template = "llama-3.1"  # Format to use for chat templates

    # Optional HF token for gated models
    hf_token = os.environ.get("HF_TOKEN", None)  # Set via env var or replace with "hf_..."

    # Training parameters
    per_device_train_batch_size = args.batch_size
    gradient_accumulation_steps = 4
    warmup_steps = 5
    max_steps = args.steps if args.epochs is None else None
    num_train_epochs = args.epochs
    learning_rate = 2e-4

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    #==========================================================================
    # MODEL LOADING
    #==========================================================================
    print(f"Loading model {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=hf_token,
    )

    #==========================================================================
    # LORA CONFIGURATION
    #==========================================================================
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    #==========================================================================
    # DATA PREPARATION
    #==========================================================================
    print("Preparing tokenizer...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    print(f"Loading dataset from {dataset_path}...")
    try:
        # Load the local JSON file
        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(raw_data)
        
        # Standardize and format dataset
        dataset = standardize_sharegpt(dataset)
        dataset = dataset.map(formatting_prompts_func, batched=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure your dataset is a JSON file containing records with 'conversations' field in ShareGPT format")
        return

    #==========================================================================
    # TRAINING SETUP
    #==========================================================================
    print("Setting up training...")
    # Initialize with either max_steps or num_train_epochs but not both
    if num_train_epochs is not None:
        # If epochs is defined, use it and set max_steps to -1 (= run for full epochs)
        training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=-1,  # -1 means use num_train_epochs instead
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",  # Use "wandb" for Weights & Biases
        )
    else:
        # Otherwise, use max_steps and default num_train_epochs
        training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps if max_steps is not None else 60,  # Ensure max_steps is not None
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",  # Use "wandb" for Weights & Biases
        )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=training_args,
    )

    # Train on responses only
    print("Setting up response-only training...")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    #==========================================================================
    # TRAINING EXECUTION
    #==========================================================================
    print("Starting training...")
    # Show GPU info
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
    else:
        print("Warning: No GPU detected. Training will be very slow.")
        start_gpu_memory = 0
        max_memory = 0

    trainer_stats = trainer.train()

    # Show training stats
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3) if max_memory > 0 else 0
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3) if max_memory > 0 else 0
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    #==========================================================================
    # MODEL SAVING
    #==========================================================================
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    #==========================================================================
    # INFERENCE EXAMPLE
    #==========================================================================
    print("Running inference example...")
    # Enable 2x faster inference
    FastLanguageModel.for_inference(model)
    print("Loading input...")
    try:
        with open('test.json', 'r', encoding='utf-8') as f:
            paper = json.load(f)
            # If it's a list with one paper, extract it
            if isinstance(paper, list) and len(paper) > 0:
                paper = paper[0]
    except Exception as e:
        print(f"Error loading input file: {e}")
    
    # Extract title and sections from the paper
    try:
        title = paper.get('title', '')
        sections = paper.get('content', '')
        title_and_sections = f"Title: {title}\n\nSections: {sections}"
    except Exception as e:
        print(f"Error extracting title and sections: {e}")

    # Example generation
    messages = [
        {"role": "user", "content": f" Write me an abstract for this article, <input>{title_and_sections}</input>"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    # Stream the output
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    print("Model output:")
    _ = model.generate(
        input_ids=inputs, 
        streamer=text_streamer,
        max_new_tokens=2048,
        use_cache=True,
        temperature=1.5,
        min_p=0.1
    )

    print("\nDone! Model saved to:", output_dir)
    print("\nYou can load this model with:")
    print(f"  from unsloth import FastLanguageModel")
    print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{output_dir}')")


if __name__ == "__main__":
    main() 
