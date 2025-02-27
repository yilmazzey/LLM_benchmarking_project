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
    parser.add_argument("--input", type=str, default="formatted_papersv2.json",
                        help="Input JSON file with papers (default: formatted_papersv2.json)")
    parser.add_argument("--output", type=str, default="papers_sharegpt.json",
                        help="Output file in ShareGPT format (default: papers_sharegpt.json)")
    parser.add_argument("--format", type=str, default="title_abstract",
                        choices=["title_abstract", "abstract_sections", "sections_qa", "abstract_instruction"],
                        help="Conversion format (default: title_abstract)")
    parser.add_argument("--max-papers", type=int, default=None,
                        help="Maximum number of papers to convert (default: all)")
    return parser.parse_args()

def create_title_abstract_conversation(paper: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    Create a conversation where:
    - User asks for information about the paper title
    - Assistant responds with the abstract
    """
    return {
        "conversations": [
            {"role": "user", "content": f"What is the paper '{paper['title']}' about?"},
            {"role": "assistant", "content": paper['abstract']}
        ]
    }

def create_abstract_sections_conversation(paper: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    Create a conversation where:
    - User provides the abstract and asks for more details
    - Assistant responds with the full paper sections
    """
    return {
        "conversations": [
            {"role": "user", "content": f"I read this abstract: \"{paper['abstract']}\"\n\nCan you provide me with more details about this research?"},
            {"role": "assistant", "content": paper['sections']}
        ]
    }

def create_sections_qa_conversation(paper: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    Create a conversation with multiple turns:
    - User asks about the paper title
    - Assistant responds with the abstract
    - User asks for methodology and findings
    - Assistant responds with relevant sections
    """
    # Extract a simplified version of sections for the second response
    sections = paper['sections']
    # Limit to a reasonable length for a response
    if len(sections) > 4000:
        sections = sections[:4000] + "..."
    
    return {
        "conversations": [
            {"role": "user", "content": f"What is the paper '{paper['title']}' about?"},
            {"role": "assistant", "content": paper['abstract']},
            {"role": "user", "content": "Can you explain the methodology and key findings of this research?"},
            {"role": "assistant", "content": sections}
        ]
    }

def create_abstract_instruction_conversation(paper: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    Create an instruction-following format where:
    - User provides title and sections and asks for an abstract
    - Assistant responds with the abstract
    
    Format:
    Instruction: Write me an abstract for this article, <input>title and sections</input>
    Output: abstract
    """
    # Combine title and sections as input
    title_and_sections = f"Title: {paper['title']}\n\nSections:\n{paper['sections']}"
    
    # Limit input length if needed
    if len(title_and_sections) > 6000:
        title_and_sections = title_and_sections[:6000] + "..."
    
    return {
        "conversations": [
            {"role": "user", "content": f"Instruction: Write me an abstract for this article, <input>{title_and_sections}</input>"},
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
    
    print(f"Converting {len(papers)} papers to ShareGPT format using '{args.format}' strategy...")
    
    # Choose conversion strategy
    if args.format == "title_abstract":
        conversion_func = create_title_abstract_conversation
    elif args.format == "abstract_sections":
        conversion_func = create_abstract_sections_conversation
    elif args.format == "sections_qa":
        conversion_func = create_sections_qa_conversation
    elif args.format == "abstract_instruction":
        conversion_func = create_abstract_instruction_conversation
    else:
        print(f"Unknown format: {args.format}")
        return
    
    # Convert papers
    converted_data = [conversion_func(paper) for paper in papers]
    
    # Save output
    print(f"Saving {len(converted_data)} conversations to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print("Conversion complete!")
    print(f"You can now use this dataset with llama3_2_alpacaft.py:")
    print(f"python llama3_2_alpacaft.py --dataset {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main() 
