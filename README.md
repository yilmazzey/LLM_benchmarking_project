# Llama 3.2 Fine-Tuning for Abstract Generation

Fine-tune Llama 3.2 models (3B and 1B) on academic papers for automatic abstract generation using QLoRA and Unsloth.

## Requirements

- **GPU**: NVIDIA B200 (180GB VRAM) or similar
- **CUDA**: 12.0+
- **Python**: 3.10+

---

## Quick Setup (uv - Recommended)

### 1. Transfer Data to Server

From your **local machine**, transfer `veri.json` to the server:

```bash
# Replace with your server details
rsync -avz --progress ./veri.json username@server-ip:/path/to/LLM_benchmarking_project/
```

Example:
```bash
rsync -avz --progress ./veri.json mreative@10.0.0.100:~/LLM_benchmarking_project/
```

### 2. SSH into Server

```bash
ssh username@server-ip
cd ~/LLM_benchmarking_project
```

### 3. Install Dependencies with uv

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install deps (FAST!)
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 4. Run Training

```bash
# 3B first, then 1B
python llama-ft.py --model 3b --epochs 2 --batch_size 8 --gradient_accumulation 8 --eval_split 0.05 && \
python llama-ft.py --model 1b --epochs 2 --batch_size 8 --gradient_accumulation 8 --eval_split 0.05
```

---

## Alternative: pip install

```bash
pip install unsloth
pip install --no-deps trl==0.22.2
pip install transformers==4.56.2
```

Or use Unsloth Docker image:
```bash
docker pull unsloth/unsloth
```

## Dataset

The script expects a JSON file (`veri.json`) with the following structure:

```json
[
  {
    "title": "Paper Title",
    "content": "Full paper content...",
    "abstract": "Ground truth abstract..."
  },
  ...
]
```

- **Total samples**: 4,234 academic papers
- **Train/Eval split**: 95% / 5% (4,022 train, 212 eval)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Models | Llama 3.2 3B-Instruct, Llama 3.2 1B-Instruct |
| Quantization | 4-bit (QLoRA) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 16 |
| Max Sequence Length | 8192 tokens |
| Batch Size | 8 |
| Gradient Accumulation | 8 |
| Effective Batch Size | 64 |
| Epochs | 2 |
| Learning Rate | 2e-4 |
| LR Scheduler | Cosine |
| Warmup | 3% of total steps |
| Optimizer | AdamW 8-bit |
| Precision | bfloat16 |
| Eval Split | 5% |

## Quick Start

### Train 3B Model (First)

```bash
python llama-ft.py \
    --model 3b \
    --epochs 2 \
    --batch_size 8 \
    --gradient_accumulation 8 \
    --eval_split 0.05 \
    --data_path ./veri.json \
    --output_dir ./lora_adapters
```

### Train 1B Model (Second)

```bash
python llama-ft.py \
    --model 1b \
    --epochs 2 \
    --batch_size 8 \
    --gradient_accumulation 8 \
    --eval_split 0.05 \
    --data_path ./veri.json \
    --output_dir ./lora_adapters
```

### Run Both Sequentially

```bash
# 3B first, then 1B
python llama-ft.py --model 3b --epochs 2 --batch_size 8 --gradient_accumulation 8 --eval_split 0.05 && \
python llama-ft.py --model 1b --epochs 2 --batch_size 8 --gradient_accumulation 8 --eval_split 0.05
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `3b` | Model size: `3b` or `1b` |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `1` | Per-device batch size |
| `--gradient_accumulation` | `8` | Gradient accumulation steps |
| `--eval_split` | `0.0` | Fraction for evaluation (0.05 = 5%) |
| `--max_seq_length` | `8192` | Maximum sequence length |
| `--data_path` | `./veri.json` | Path to training data |
| `--output_dir` | `./lora_adapters` | Output directory for adapters |
| `--hf_token` | `None` | HuggingFace token (optional) |

## Output Structure

```
./lora_adapters/
├── llama-3.2-3b-abstract-lora_ep2_YYYYMMDD_HHMMSS/
│   ├── checkpoint-XXX/           # Epoch 1 checkpoint
│   ├── checkpoint-XXX/           # Epoch 2 checkpoint
│   └── final/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── special_tokens_map.json
│
└── llama-3.2-1b-abstract-lora_ep2_YYYYMMDD_HHMMSS/
    └── ...
```

## Training Steps Calculation

With the default configuration:
- **Samples**: ~4,022 (after 5% eval split)
- **Effective batch size**: 8 × 8 = 64
- **Steps per epoch**: 4,022 ÷ 64 ≈ 63
- **Total steps (2 epochs)**: ~126
- **Warmup steps**: ~4 (3% of 126)

## Inference

After training, use the fine-tuned model:

```python
from unsloth import FastLanguageModel

# Load the fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    "./lora_adapters/llama-3.2-3b-abstract-lora_ep2_XXXXXXXX_XXXXXX/final",
    max_seq_length=8192,
    load_in_4bit=True,
)

# Enable fast inference
FastLanguageModel.for_inference(model)

# Create prompt
messages = [
    {"role": "system", "content": "You are an expert scientific abstract writer with deep knowledge of academic writing conventions."},
    {"role": "user", "content": """Create a concise and informative abstract for the following scientific paper:

Title: Your Paper Title

Paper content:
<content>
Your paper content here...
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

Output ONLY the abstract text."""}
]

# Generate
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

# Decode
generated_abstract = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_abstract)
```

## Expected Results

| Model | Train Loss | Eval Loss | Training Time* |
|-------|------------|-----------|----------------|
| 3B | ~1.0-1.5 | ~1.2-1.8 | ~30-45 min |
| 1B | ~1.2-1.8 | ~1.4-2.0 | ~15-25 min |

*Estimated on B200 GPU

## Troubleshooting

### OOM (Out of Memory)

If you encounter OOM errors, reduce batch size:
```bash
python llama-ft.py --model 3b --batch_size 4 --gradient_accumulation 16
```

### CUDA Not Available

Ensure CUDA is properly installed:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### File Not Found

Make sure `veri.json` exists in the current directory or provide full path:
```bash
python llama-ft.py --data_path /full/path/to/veri.json
```

## License

This project uses Unsloth which is licensed under LGPL-3.0.
