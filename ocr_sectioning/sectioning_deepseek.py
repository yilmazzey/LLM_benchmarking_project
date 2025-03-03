import os
import json
import glob
import re
import time
import logging
import argparse
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('article_processing.log'),
        logging.StreamHandler()
    ]
)

@retry(
    wait=wait_fixed(5),
    stop=stop_after_attempt(3)
)
def extract_sections_with_deepseek(client, text):
    """
    Extract title, abstract, and content using DeepSeek model via OpenRouter
    """
    prompt = """ANALYZE THIS ACADEMIC PAPER AND EXACTLY FOLLOW THESE INSTRUCTIONS:

1. Identify the FULL PAPER TITLE
2. Extract the COMPLETE ABSTRACT (do not truncate)
3. Extract the MAIN CONTENT that comes AFTER THE ABSTRACT ENDS

FORMAT YOUR RESPONSE *EXACTLY* LIKE THIS WITHOUT ANY ADDITIONAL TEXT:

###TITLE###
[Full paper title here]

###ABSTRACT###
[Complete abstract text here. Include all paragraphs.]

###CONTENT###
[Main body content here. This must start AFTER the abstract ends. 
Do NOT include any abstract text in this section.]

PAPER TEXT TO ANALYZE: """
    
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://yourwebsite.com",  # Replace with your site URL
            "X-Title": "Academic Paper Processing",     # Replace with your site name
        },
        model="deepseek/deepseek-r1-distill-llama-70b:free",
        messages=[
            {
                "role": "user", 
                "content": prompt + text
            }
        ],
        temperature=0.1,
        max_tokens=4000
    )
    
    return completion.choices[0].message.content

def parse_response(raw_response):
    """Parse the response to extract title, abstract, and content"""
    sections = {
        'title': [],
        'abstract': [],
        'content': []
    }
    current_section = None
    
    # Normalize response
    text = raw_response.replace('**', '').replace('__', '')
    lines = [line.strip() for line in text.split('\n')]
    
    section_pattern = re.compile(
        r'^###(TITLE|ABSTRACT|CONTENT)###\s*$',
        re.IGNORECASE
    )
    
    for line in lines:
        # Check for section headers
        match = section_pattern.match(line)
        if match:
            current_section = match.group(1).lower()
            continue
        
        # Add content to current section
        if current_section and line:
            sections[current_section].append(line)
    
    # Join lines
    cleaned = {
        'title': ' '.join(sections['title']).strip(),
        'abstract': ' '.join(sections['abstract']).strip(),
        'content': ' '.join(sections['content']).strip()
    }
    
    # Remove excessive section markers that might remain
    for key in cleaned:
        if cleaned[key]:
            cleaned[key] = re.sub(r'(?i)###(title|abstract|content)###', '', cleaned[key]).strip()
    
    return cleaned

def preprocess_ocr_text(ocr_file_path):
    """Remove line numbers, OCR markers, and clean the text"""
    with open(ocr_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove line numbers and OCR markers
    clean_lines = []
    for line in lines:
        # Skip OCR Start/End markers
        if "--- OCR Start ---" in line or "--- OCR End ---" in line:
            continue
        
        # Remove line numbers (pattern like "123| ")
        line = re.sub(r'^\d+\|\s*', '', line)
        clean_lines.append(line)
    
    return ''.join(clean_lines)

def read_markdown_sections(md_file_path):
    """Read title, abstract, and content from a markdown file"""
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title
        title_match = re.search(r'# (.*?)\n', content)
        title = title_match.group(1).strip() if title_match else None
        
        # Extract abstract
        abstract_match = re.search(r'## Abstract\n\n(.*?)\n\n## Content', content, re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else None
        
        # Extract content
        content_match = re.search(r'## Content\n\n(.*?)(?:\n\n---|$)', content, re.DOTALL)
        content_text = content_match.group(1).strip() if content_match else None
        
        return {
            'title': title,
            'abstract': abstract,
            'content': content_text
        }
    except Exception as e:
        logging.error(f"Error reading markdown file {md_file_path}: {str(e)}")
        return {'title': None, 'abstract': None, 'content': None}

def phase1_create_initial_sections(input_dir, output_dir, force_reprocess=None):
    """Phase 1: Create initial markdown files with sectioning"""
    logging.info("=== PHASE 1: Creating initial markdown files with sectioning ===")
    
    # Get all text files in the input directory
    ocr_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not ocr_files:
        logging.error(f"No text files found in {input_dir}")
        return False
    
    logging.info(f"Found {len(ocr_files)} files to process")
    
    # Filter files that need to be processed
    files_to_process = []
    already_processed = []
    
    for ocr_file in ocr_files:
        file_name = os.path.basename(ocr_file)
        base_name = os.path.splitext(file_name)[0]
        output_file = os.path.join(output_dir, f"{base_name}.md")
        
        # Check if file should be processed
        should_process = False
        
        # Force reprocessing if specified
        if force_reprocess and base_name in force_reprocess:
            should_process = True
        # Otherwise, check if the file exists and has valid content
        elif not os.path.exists(output_file):
            should_process = True
        else:
            # File exists, check if it has valid content
            sections = read_markdown_sections(output_file)
            if not sections['title'] or not sections['abstract'] or sections['abstract'] == "None":
                should_process = True
        
        if should_process:
            files_to_process.append(ocr_file)
        else:
            already_processed.append(file_name)
    
    logging.info(f"Already processed: {len(already_processed)} files")
    logging.info(f"To be processed: {len(files_to_process)} files")
    
    if force_reprocess:
        logging.info(f"Forcing reprocessing of: {force_reprocess}")
    
    if not files_to_process:
        logging.info("No files need processing. All done!")
        return True
    
    # Initialize OpenAI client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",  # Replace with your actual OpenRouter API key
    )
    
    # Process files in batches to avoid rate limits
    processed_count = 0
    failed_files = []
    batch_size = 10
    
    try:
        for i, ocr_file in enumerate(tqdm(files_to_process, desc="Phase 1: Creating sectioned files")):
            try:
                file_name = os.path.basename(ocr_file)
                base_name = os.path.splitext(file_name)[0]
                output_file = os.path.join(output_dir, f"{base_name}.md")
                
                # Process the OCR text
                ocr_text = preprocess_ocr_text(ocr_file)
                if not ocr_text.strip():
                    logging.warning(f"File {file_name} is empty after preprocessing")
                    failed_files.append(file_name)
                    continue
                
                # Extract sections
                raw_response = extract_sections_with_deepseek(client, ocr_text)
                sections = parse_response(raw_response)
                
                # Write to markdown file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {sections['title']}\n\n")
                    f.write(f"## Abstract\n\n{sections['abstract']}\n\n")
                    f.write(f"## Content\n\n{sections['content']}\n\n")
                
                logging.info(f"Successfully created sections for: {file_name}")
                processed_count += 1
                
                # Add delay after each batch of API calls
                if (i + 1) % batch_size == 0:
                    logging.info(f"Processed {i + 1}/{len(files_to_process)} files, pausing...")
                    time.sleep(5)  # Pause between batches
                
            except Exception as e:
                logging.error(f"Error processing {ocr_file}: {str(e)}")
                failed_files.append(os.path.basename(ocr_file))
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        logging.warning("\nProcess interrupted by user.")
        logging.info(f"Progress so far: {processed_count}/{len(files_to_process)} files processed in this run.")
        logging.info(f"You can resume from where you left off by running the script again.")
        return False
    
    # Print summary
    logging.info(f"Phase 1 completed!")
    logging.info(f"Processed in this run: {processed_count}/{len(files_to_process)} files")
    logging.info(f"Total processed: {len(already_processed) + processed_count}/{len(ocr_files)} files")
    
    if failed_files:
        logging.warning(f"Failed files in this run: {len(failed_files)}")
        for file in sorted(failed_files):
            logging.warning(f"  - {file}")
    
    return True

def process_folder(input_dir="LLM_benchmarking_project/ocr_sectioning/ocr_output_100", 
                  output_dir="LLM_benchmarking_project/ocr_sectioning/sectioning_output_100",
                  force_reprocess=None):
    """Process all OCR files in a folder using Phase 1 only"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Execute Phase 1 only
    phase1_success = phase1_create_initial_sections(input_dir, output_dir, force_reprocess)
    if not phase1_success:
        logging.error("Phase 1 did not complete successfully.")
        return
    
    logging.info("Phase 1 completed successfully!")
    logging.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Process OCR files and create sectioned markdown files")
    parser.add_argument("--input", default="LLM_benchmarking_project/ocr_sectioning/ocr_output_100", 
                        help="Directory containing OCR text files")
    parser.add_argument("--output", default="LLM_benchmarking_project/ocr_sectioning/sectioning_output_100", 
                        help="Directory to save markdown files")
    parser.add_argument("--force", nargs="+", 
                        help="Force reprocessing of specific files (base names without extension)")
    
    args = parser.parse_args()
    
    # Process folder with provided arguments
    process_folder(
        input_dir=args.input,
        output_dir=args.output,
        force_reprocess=args.force
    )