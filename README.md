# Ethnographic Feature Extraction Using LLMs

Can large language models (LLMs) can perform ethnographic feature extraction at a level comparable to human coders? Advanced NLP techniques are combined with transformer-based embeddings and multiple LLM APIs (e.g., GPT-4, Claude-2, Mistral-7B) to annotate sociocultural features in ethnographic texts.

## Considerations:
- Test across multiple LLMs
- Inter-reliability score comparison per ritual or across all rituals? (Could do both: Aggregated and per-ritual metrics) 

## Project Structure

- _main.py_
  Orchestrates the pipeline:
  - Extracts and cleans text from PDF ethnographies.
  - Uses transformer-based embeddings for topic modeling.
  - Queries multiple LLMs for annotation.
  - Aggregates the results and exports them to CSV.

- _extract_text.py_  
  - Contains functions to extract text from PDF files (using `pdfplumber` and OCR via `pytesseract`), and clean the extracted text.

- _topic_modelling.py_  
  - Uses Latent Dirichlet Allocation (LDA) or Sentence‑BERT to encode paragraphs and clusters them into topics using KMeans.

- _llm_annotate.py_
  - Handles LLM queries for feature annotation.  

- _evaluation.py_
  - Evaluates model outputs against human-coded data.  
  - Computes metrics (accuracy, Cohen’s kappa, macro F1, MCC).  
  - Provides per-feature error analysis, including confusion matrices and McNemar’s tests for statistical significance.

- _visualisation.py_
  - Visualise model performance and the correlation between evaluation metrics.

## Setup

1. **Dependencies:**  
   Ensure you have the following installed:
   - Python 3.8+
   - `pdfplumber`, `pytesseract`, `pdf2image`
   - `sentence-transformers`, `scikit-learn`
   - `openai`, `requests`
   - `pandas`, `numpy`
   - `matplotlib`, `seaborn`
   - `statsmodels`

   You can install dependencies using:
   ```bash
   pip install -r requirements.txt