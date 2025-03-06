# LLM Benchmarking Project (sdp_llm)

This repository contains the code for the LLM Benchmarking Project (sdp_llm), organized into folders representing different stages of the project.

## Todo

### Data

- [x] Raw Data Creation -> 1487 Articles with Gemini 2.0 Flash OCR, Marker Library
- [ ] Raw Data Check -> 100 Manuel Human Bench, 1000 Randomized 3 serial words check 100 of them between the articles, 
- [x] Raw Data To Sectioned Data -> Later we will do sectioning with Gemini-2.0-Flash or Regex

### Abstract Generation

- [ ] For the choosen models;

#### Open Source

#### Low Parameter; ollama & colab -> They Are on the generate file.

- [x] smollm2 1.7B*
- [x] qwen 2.5 1.5B*
- [ ] qwen 2.5 3B
- [x] llama3.2 3B*
- [ ] Gemma 2:2B
- [ ] FlanT5
- [ ] Phi4-mini 3.8B

#### Medium Parameter; ollama They Are on the generate file.
- [ ] qwen 2.5 7B
- [ ] Llama 3.1 8B*
- [ ] qwen 2.5 14B
- [ ] r1 8b*

#### High Parameter;

- [ ] llama3.3 70B
- [ ] deepseek r1 70b
- [ ] qwen 2.5 32B [ It is on the generate file ]
- [ ] mistral-large*

#### Closed Source
- [ ] gpt-4o-mini
- [ ] gpt-4o
- [ ] gemini-2.0-flash*
- [ ] deepseek-r1

#### Evaluations

- [ ] all-MiniLM-L12-v2
- [ ] paraphrase-multilingual-MiniLM-L12-v2
- [ ] all-mpnet-base-v2
- [ ] Bge-large
- [ ] nomic-embed-text
- [ ] BERTscore F1
- [ ] ROUGE 1-2-L score
- [ ] UniEval
- [ ] G-eval

#### Fine-tuning

- [ ] Fine tuning data creation

- [ ] gpt-4o-mini
- [ ] gemini-2.0-flash

- [ ] llama3.2 3B
- [ ] smollm2 1.7B

- [ ] Llama 3.1 8B
- [ ] qwen 2.5 7B

- [ ] llama3.3 70B

## Project Stages

The project is structured into the following stages:

1.  **Web Scraping (`web_scraping/`)**
    * Code for extracting articles from ACL Anthology using selenium.
    * Notebooks used in checkpoint and initial database creation.

2.  **Gemini OCR + Section Division (`ocr_sectioning/`)**
    * Code utilizing Gemini for Optical Character Recognition (OCR).
    * Implementation of section division logic to structure extracted text.
    * Includes necessary scripts to form the correct format of dataset (resulting json file)

3.  **Abstract Generation (`abstract_generation/`)**
    * Code for generating abstracts using Large Language Models (LLMs).
    * Includes scripts for processing input text and generating concise summaries.

4.  **Evaluations (`evaluations/`)** -> https://colab.research.google.com/drive/1Sy1S7LTowmzSiYdLjD6p_Pzi6j9eLYSL?usp=sharing
    * Code for evaluating the performance of generated abstracts. Which includes embeddings, n-grams and model based methods for now.

5.  **Fine-tuning (`fine_tuning/`)** -> Branch FineTuning
    * Code for fine-tuning LLMs for specific tasks.
    * Includes scripts for data preparation, model training, and evaluation.


## Repository Structure
