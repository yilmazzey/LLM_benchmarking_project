#!/usr/bin/env python3
"""
Verify chat template formatting for different model sizes.
Shows the difference between 8B (Alpaca) and 1B/3B (Llama-3.1 chat) formatting.
"""

# Alpaca prompt template (for 8B model)
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Mock tokenizer for demonstration
class MockTokenizer:
    def __init__(self, model_type):
        self.eos_token = "<|end_of_text|>"
        self.model_type = model_type

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        """Mock Llama-3.1 chat template."""
        result = "<|begin_of_text|>"
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            result += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        return result

def format_alpaca(user_msg, assistant_msg, tokenizer):
    """Format using Alpaca template (8B)."""
    text = ALPACA_PROMPT.format(user_msg, "", assistant_msg) + tokenizer.eos_token
    return text

def format_chat(conversation, tokenizer):
    """Format using Llama-3.1 chat template (1B/3B)."""
    return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)

# Sample conversation
user_message = "Create a concise abstract for a paper about quantum computing."
assistant_message = "This paper presents a novel approach to quantum error correction using topological codes. We demonstrate a 30% improvement in qubit stability and propose practical implementations for near-term quantum computers."

# Convert to conversation format
conversation = [
    {"role": "user", "content": user_message},
    {"role": "assistant", "content": assistant_message}
]

print("=" * 80)
print("CHAT TEMPLATE VERIFICATION")
print("=" * 80)

# 8B Model (Alpaca format)
print("\n" + "=" * 80)
print("8B MODEL - ALPACA TEMPLATE")
print("=" * 80)
tokenizer_8b = MockTokenizer("8b")
formatted_8b = format_alpaca(user_message, assistant_message, tokenizer_8b)
print(formatted_8b)
print("\n[Note: Full sequence training - no masking of user prompts]")

# 1B/3B Model (Llama-3.1 chat format)
print("\n" + "=" * 80)
print("1B/3B MODEL - LLAMA-3.1 CHAT TEMPLATE")
print("=" * 80)
tokenizer_chat = MockTokenizer("3b")
formatted_chat = format_chat(conversation, tokenizer_chat)
print(formatted_chat)
print("\n[Note: train_on_responses_only will mask everything before '<|start_header_id|>assistant<|end_header_id|>']")

print("\n" + "=" * 80)
print("KEY DIFFERENCES")
print("=" * 80)
print("""
8B Model (Alpaca):
  - Uses Alpaca prompt template with ### markers
  - NO get_chat_template() call
  - NO train_on_responses_only()
  - Full sequence training (trains on instruction + response)
  - Simple string formatting

1B/3B Models (Llama-3.1):
  - Uses <|start_header_id|> format
  - REQUIRES get_chat_template(tokenizer, chat_template="llama-3.1")
  - USES train_on_responses_only() to mask user prompts
  - Only trains on assistant responses (more efficient)
  - Uses tokenizer.apply_chat_template()
""")

print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
