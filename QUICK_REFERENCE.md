# Quick Reference: Chat Templates by Model Size

## TL;DR

```python
# 8B Model
- Template: Alpaca string format
- get_chat_template: NO
- train_on_responses_only: NO
- Training: Full sequence

# 1B/3B Models
- Template: Llama-3.1 chat format
- get_chat_template: YES (chat_template="llama-3.1")
- train_on_responses_only: YES
- Training: Assistant responses only
```

## Example Output Formats

### 8B Model (Alpaca)
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Create an abstract...

### Input:


### Response:
This paper presents...<|end_of_text|>
```

### 1B/3B Models (Llama-3.1)
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Create an abstract...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

This paper presents...<|eot_id|>
```

## Code Flow in llama-ft.py

```python
# Step 1: Load model and tokenizer (line 255)
model, tokenizer = FastLanguageModel.from_pretrained(...)

# Step 2: Configure template based on model size (line 273)
if args.model == "8b":
    use_alpaca = True
    # No get_chat_template call
else:
    use_alpaca = False
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Step 3: Format dataset (line 313)
if use_alpaca:
    dataset = dataset.map(formatting_prompts_func_alpaca, ...)
else:
    dataset = dataset.map(formatting_prompts_func_chat, ...)

# Step 4: Configure trainer (line 383, 395)
data_collator = DataCollatorForSeq2Seq(...) if not use_alpaca else None
trainer = SFTTrainer(...)

# Step 5: Apply response masking (line 395)
if not use_alpaca:
    trainer = train_on_responses_only(trainer, ...)
else:
    # Full sequence training
```

## Testing Commands

```bash
# Verify templates visually
python3 verify_templates.py

# Test 8B
python llama-ft.py --model 8b --epochs 1 --data_path ./veri.json

# Test 3B
python llama-ft.py --model 3b --epochs 1 --data_path ./veri.json

# Test 1B
python llama-ft.py --model 1b --epochs 1 --data_path ./veri.json
```

## What to Look For

### 8B Output
```
Using Alpaca prompt template for 8B model
eos_token: <|end_of_text|> (id: 128001)
...
Using full sequence training (Alpaca template)
```

### 1B/3B Output
```
Using Llama-3.1 chat template for 3B model
eos_token: <|end_of_text|> (id: 128001)
...
Applying train_on_responses_only...
  instruction_part: '<|start_header_id|>user<|end_header_id|>\n\n'
  response_part: '<|start_header_id|>assistant<|end_header_id|>\n\n'
```

## Common Issues

❌ **Wrong**: Using `get_chat_template` for 8B
✅ **Correct**: 8B uses Alpaca string template only

❌ **Wrong**: Not using `train_on_responses_only` for 1B/3B
✅ **Correct**: 1B/3B should mask user prompts

❌ **Wrong**: Using chat template for 8B
✅ **Correct**: Each model size has its own template approach

## Reference Notebooks

- `unsloth8b.py` - Lines 114-139 (Alpaca approach)
- `unsloth1bve3b.py` - Lines 126-209 (Chat template approach)
