import json
import os
import requests
import sys
from datetime import datetime
import curses

# Ollama API endpoint (default local address)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Available models with their Ollama names
MODELS = {
    1: {"name": "smollm2:1.7b", "display_name": "smollm2 1.7B"},
    2: {"name": "qwen2.5:1.5b", "display_name": "qwen 2.5 1.5B"},
    3: {"name": "qwen2.5:3b", "display_name": "qwen 2.5 3B"},
    4: {"name": "llama3.2:3b", "display_name": "llama3.2 3B"},
    5: {"name": "gemma2:2b", "display_name": "Gemma 2:2B"},
    6: {"name": "phi4-mini", "display_name": "Phi4-mini 3.8B"},
    7: {"name": "qwen2.5:7b", "display_name": "qwen 2.5 7B"},
    8: {"name": "llama3.1:8b", "display_name": "llama3.1 8B"},
    9: {"name": "qwen2.5:14b", "display_name": "qwen 2.5 14B"},
    10: {"name": "deepseek-r1:8b", "display_name": "deepseek-r1 8B"},
    11: {"name": "qwen2.5:32b", "display_name": "qwen 2.5 32B"},
}

# Function to generate text using Ollama
def generate_with_ollama(prompt, model_name):
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=data)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            print(f"Ollama API error: {response.status_code}")
            print(f"Response content: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None

# Function to pull model using Ollama
def pull_model(model_name):
    try:
        print(f"\nPulling model {model_name}...")
        response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model_name},
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "status" in data:
                    print(f"Status: {data['status']}")
                if "completed" in data and data["completed"]:
                    print(f"✓ Model {model_name} successfully pulled!")
                    return True
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error pulling model: {e}")
        return False

# Function to display model selection menu and get user choice
def select_model():
    print("\n===== MODEL SELECTION =====")
    for key, model in MODELS.items():
        print(f"{key}. {model['display_name']} (ollama name: {model['name']})")
    
    while True:
        try:
            choice = int(input("\nSelect a model (1-6): "))
            if choice in MODELS:
                model_name = MODELS[choice]["name"]
                # Pull the model before proceeding
                if pull_model(model_name):
                    return model_name
                else:
                    print(f"Failed to pull model {model_name}. Please try again or select a different model.")
            else:
                print("Invalid choice. Please select a number between 1 and 6.")
        except ValueError:
            print("Please enter a valid number.")

def select_models():
    """Use curses to allow multi-selection of models. Navigate with arrow keys, toggle with space, confirm with Enter."""
    models_list = [(v['name'], v['display_name']) for k, v in sorted(MODELS.items())]
    selected = [False] * len(models_list)
    current_row = 0

    def print_menu(stdscr):
        stdscr.clear()
        stdscr.addstr("Use arrow keys to navigate, space to select, Enter to confirm selection.\n\n")
        for idx, (model, display) in enumerate(models_list):
            checkbox = "[x]" if selected[idx] else "[ ]"
            if idx == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(f"{checkbox} {display} (ollama name: {model})\n")
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(f"{checkbox} {display} (ollama name: {model})\n")
        stdscr.refresh()

    def main_curses(stdscr):
        nonlocal current_row, selected
        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
        print_menu(stdscr)

        while True:
            key = stdscr.getch()
            if key == curses.KEY_UP and current_row > 0:
                current_row -= 1
            elif key == curses.KEY_DOWN and current_row < len(models_list) - 1:
                current_row += 1
            elif key == ord(' '):
                selected[current_row] = not selected[current_row]
            elif key in [curses.KEY_ENTER, 10, 13]:
                break
            print_menu(stdscr)

    curses.wrapper(main_curses)
    chosen_models = [models_list[i][0] for i, sel in enumerate(selected) if sel]
    # If no model selected, default to the first one
    if not chosen_models:
        chosen_models = [models_list[0][0]]
    return chosen_models

# Function to create a checkpoint file name based on model and timestamp
def get_checkpoint_filename(model_name):
    # Replace special characters in model name to make a valid filename
    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    return f"checkpoint_{safe_model_name}.json"

# Function to save checkpoint
def save_checkpoint(abstracts, current_idx, model_name):
    checkpoint_data = {
        "model_name": model_name,
        "last_processed_idx": current_idx,
        "abstracts": abstracts
    }
    
    checkpoint_file = get_checkpoint_filename(model_name)
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=4, ensure_ascii=False)
    
    print(f"Checkpoint saved: {checkpoint_file}")

# Function to load checkpoint
def load_checkpoint(model_name):
    checkpoint_file = get_checkpoint_filename(model_name)
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            return checkpoint_data.get("abstracts", []), checkpoint_data.get("last_processed_idx", -1)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    return [], -1

# Function to create a high-quality prompt for abstract generation
def create_abstract_prompt(title, sections):
    system_prompt = """You are an expert scientific abstract writer with deep knowledge of academic writing conventions. Your task is to create a comprehensive, accurate, and professional abstract for the given research paper."""
    
    user_prompt = f"""
Create a concise and informative abstract for the following scientific paper:

Title: {title}

Paper content:
<content>
{sections}
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
    
    return f"{system_prompt}\n\n{user_prompt}"

# Function to check if checkpoint exists and offer to resume
def check_for_checkpoint():
    checkpoint_files = [f for f in os.listdir() if f.startswith("checkpoint_") and f.endswith(".json")]
    
    if not checkpoint_files:
        return None, -1
    
    print("\n===== CHECKPOINTS FOUND =====")
    print("0. Start a new processing run")
    
    for idx, checkpoint_file in enumerate(checkpoint_files, 1):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
                model_name = checkpoint_data.get("model_name", "Unknown")
                last_idx = checkpoint_data.get("last_processed_idx", -1)
                print(f"{idx}. Resume '{model_name}' (processed {last_idx+1} articles)")
        except:
            print(f"{idx}. {checkpoint_file} (corrupted)")
    
    while True:
        try:
            choice = int(input("\nSelect an option: "))
            if choice == 0:
                return None, -1
            if 1 <= choice <= len(checkpoint_files):
                with open(checkpoint_files[choice-1], "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                return checkpoint_data.get("model_name"), checkpoint_data.get("last_processed_idx", -1)
            print(f"Invalid choice. Please select a number between 0 and {len(checkpoint_files)}.")
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error: {e}")

# Function to process articles and generate abstracts
def process_articles(articles, model_name, start_idx=-1):
    total_articles = len(articles)
    abstracts = []
    
    # Load existing abstracts if resuming from checkpoint
    if start_idx > -1:
        abstracts, _ = load_checkpoint(model_name)
        print(f"Resuming from article {start_idx+1}/{total_articles}")
    else:
        print(f"Starting new processing run with model: {model_name}")
    
    try:
        # Process each article
        for idx, article in enumerate(articles):
            # Skip already processed articles if resuming
            if idx <= start_idx:
                continue
                
            title = article.get("title", "No Title")
            original_abstract = article.get("abstract", "")
            sections = article.get("content", "")
            
            print(f"\nProcessing article {idx + 1}/{total_articles}: {title}")
            
            # Create prompt for generating the abstract
            prompt = create_abstract_prompt(title, sections)
            
            # Generate the abstract using Ollama
            generated_text = generate_with_ollama(prompt, model_name)
            
            if generated_text:
                abstracts.append({
                    "title": title,
                    "original_abstract": original_abstract,
                    "generated_abstract": generated_text.strip()
                })
                
                print(f"✓ Abstract generated ({len(generated_text.strip())} chars)")
                
                # Save checkpoint after each successful processing
                save_checkpoint(abstracts, idx, model_name)
            else:
                print(f"✗ Failed to generate abstract for article {idx + 1}")
                # Save checkpoint even when generation fails
                save_checkpoint(abstracts, idx-1, model_name)
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user!")
        print(f"Progress saved: {len(abstracts)}/{total_articles} articles processed")
        raise
    
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print(f"Progress saved: {len(abstracts)}/{total_articles} articles processed")
        return abstracts
    
    return abstracts

# Function to save final results
def save_results(abstracts, model_name, total_articles):
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Replace special characters in model name to make a valid filename
    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    
    output_file = f"abstracts_{safe_model_name}_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(abstracts, f, indent=4, ensure_ascii=False)
    
    print(f"\nProcessing complete. Results saved to {output_file}")
    
    # Only remove checkpoint file if all articles were processed
    if len(abstracts) == total_articles:
        checkpoint_file = get_checkpoint_filename(model_name)
        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                print(f"Checkpoint file removed: {checkpoint_file}")
            except Exception as e:
                print(f"Note: Could not remove checkpoint file: {checkpoint_file}. Error: {e}")

def main():
    print("=" * 50)
    print("ABSTRACT GENERATION APP")
    print("=" * 50)

    # Use multi-selection to choose models
    selected_models = select_models()
    print(f"Selected models: {selected_models}\n")

    # JSON file path for the articles
    json_file_path = "/Users/zeynep_yilmaz/Desktop/sdpgithub/LLM_benchmarking_project/abstract_generation/processed_data.json"
    if not os.path.exists(json_file_path):
        print(f"Error: File not found: {json_file_path}")
        return

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            articles = json.load(f)
            print(f"Loaded {len(articles)} articles from {json_file_path}")
    except Exception as e:
        print(f"Error loading articles: {e}")
        return

    total_articles = len(articles)

    # Process articles for each selected model sequentially
    try:
        for model_name in selected_models:
            print("\n" + "=" * 50)
            print(f"Processing with model: {model_name}")
            print("=" * 50 + "\n")

            # Check for existing checkpoint for this model
            checkpoint_abstracts, last_idx = load_checkpoint(model_name)
            if last_idx != -1:
                checkpoint_file = get_checkpoint_filename(model_name)
                print(f"Last checkpoint file for model {model_name}: {checkpoint_file}")
                resume_answer = input(f"Checkpoint for model {model_name} found (up to article {last_idx+1}/{total_articles}). Do you want to resume? (y/n): ")
                if resume_answer.lower().startswith('y'):
                    print(f"Resuming for model {model_name} from article {last_idx+1}/{total_articles}")
                    start_idx = last_idx
                else:
                    start_idx = -1
            else:
                start_idx = -1

            # Pull the model before processing
            if not pull_model(model_name):
                print(f"Failed to pull model {model_name}. Skipping...")
                continue

            # Process the articles for the current model
            abstracts = process_articles(articles, model_name, start_idx)

            # If processing is complete, save results and delete checkpoint; if not, leave checkpoint intact
            if abstracts and len(abstracts) == total_articles:
                save_results(abstracts, model_name, total_articles)
            else:
                print(f"Processing incomplete for model {model_name}. Checkpoint saved for resuming later.")
    except KeyboardInterrupt:
        print("Exiting program after interruption.")
        return

if __name__ == "__main__":
     main()
