#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert Papers to ShareGPT Format

This script converts the formatted_papersv2.json file containing academic papers into
ShareGPT-format conversations suitable for finetuning LLama 3.2 models.

Usage:
  python convert_papers_to_sharegpt.py [--input INPUT] [--output OUTPUT] [--format FORMAT]
"""

import json
import argparse
import os
from typing import List, Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Convert papers to ShareGPT format")
    parser.add_argument("--input", type=str, default="input.json",
                        help="Input JSON file with papers (default: input.json)")
    parser.add_argument("--output", type=str, default="output.json",
                        help="Output file in ShareGPT format (default: output.json)")
    parser.add_argument("--max-papers", type=int, default=None,
                        help="Maximum number of papers to convert (default: all)")
    return parser.parse_args()



def create_abstract_instruction_conversation(paper: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    Create an instruction-following format where:
    - User provides title and sections and asks for an abstract
    - Assistant responds with the abstract
    
    Format:
    Simple ShareGPT format compatible with Llama 3.1 chat template and train_on_responses_only
    """
    # Combine title and sections as input
    title_and_sections = f"Title: {paper['title']}\n\nSections:\n{paper['sections']}"
    
    return {
        "conversations": [
            {"role": "user", "content": f"Write me an abstract for this article: {title_and_sections}"},
            {"role": "assistant", "content": paper['abstract']}
        ]
    }

def main():
    args = parse_args()
    
    print(f"Loading papers from {args.input}...")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            papers = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    # Limit number of papers if specified
    if args.max_papers:
        papers = papers[:args.max_papers]
        print(f"Limited to {args.max_papers} papers")
    
    print(f"Converting {len(papers)} papers to ShareGPT format using 'abstract_instruction' strategy...")
    
    conversion_func = create_abstract_instruction_conversation
    converted_data = [conversion_func(paper) for paper in papers]
    
    print(f"Saving {len(converted_data)} conversations to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print("Conversion complete!")
    print(f"You can now use this dataset with llama3_2_alpacaft.py:")
    print(f"python llama3_2_alpacaft.py --dataset {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main() 
