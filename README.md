# LLM Benchmarking Project (sdp_llm)

This repository contains the code for the LLM Benchmarking Project (sdp_llm), organized into folders representing different stages of the project.

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
