# Ethnographic Feature Extraction Using LLMs

Can large language models (LLMs) perform ethnographic feature extraction at a level comparable to human coders? 

## Data Format

### ethnographic_texts.csv
The file should contain two columns:

- `ethnography_number`: Index
- `paragraph`: Ethnographic text

### features.csv
The file should contain three columns:

- `feature_name`: The name of the feature to be annotated
- `feature_description`: A detailed description of the feature
- `feature_options`: Valid options/values for this feature (e.g., "low, medium, high" or "0, 1" for binary features)

### human_coded.csv
The file should contain at least 2 columns:

- `ethnography_number`: Index
- All feature names

## Project Structure

```
├── data/
│   ├── ethnographic_texts.csv   # Source ethnographic texts with ethnography_number and paragraph
│   ├── features.csv             # Features to be annotated with descriptions
│   └── human_coded.csv          # Human annotations (ground truth)
├── results/
│   ├── {model}_annotations.csv  # LLM annotations
│   ├── {model}_logprobs.json    # LLM "certainty" measure for each response
├── src/
│   ├── main.py                  # Main script to run the annotation pipeline
│   ├── llm_annotate.py          # LLM API interaction and annotation logic
│   ├── llm_client.py            # Unified client for multiple LLM providers
│   ├── batch_processing.py      # Send batch to API
│   ├── evaluation.py            # Statistical evaluation of model performance
│   ├── cost_estimator.py        # Generate estimations of cost for each model
│   ├── config.py                # API keys and model configuration
│   └── utils.py                 # Utility functions for data processing
└── requirements.txt             # Project dependencies
```

## Workflow

1. **Data Loading**: Ethnographic texts are loaded from `ethnographic_texts.csv`
2. **Feature Definition**: ethnography features are loaded from `features.csv`
3. **LLM Annotation**: Each text is processed by multiple LLM models
4. **(Un)certainty Estimates**: Using OpenAI's logprobs, certainty estimates are given
5. **Evaluation**: Model predictions are compared against human annotations (if provided)

<!-- ## Statistical Analysis

The project implements several statistical analyses:

- **Performance Metrics**: Accuracy, precision, recall, F1 score, Cohen's Kappa, and Matthews Correlation Coefficient
- **Bootstrap Confidence Intervals**: Non-parametric estimation of uncertainty in performance metrics
- **Friedman Test**: Non-parametric test to detect differences across multiple models
- **Wilcoxon Signed-Rank Test**: Pairwise comparison of models with Bonferroni correction
- **McNemar's Test**: Evaluates whether models differ in their error patterns
- **ROC Curve Analysis**: For binary features, evaluates true positive vs. false positive rates -->

## Setup and Usage

1. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Configure API keys**:
   Set the following environment variables with your API keys:
   ```bash
   # On Linux/Mac
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export HUGGINGFACE_API_KEY="your-huggingface-key"

   # On Windows (Command Prompt)
   set OPENAI_API_KEY=your-openai-key
   set ANTHROPIC_API_KEY=your-anthropic-key
   set HUGGINGFACE_API_KEY=your-huggingface-key

   # On Windows (PowerShell)
   $env:OPENAI_API_KEY="your-openai-key"
   $env:ANTHROPIC_API_KEY="your-anthropic-key"
   $env:HUGGINGFACE_API_KEY="your-huggingface-key"
   ```

3. **Prepare data**:
   - Place your ethnographic texts in `data/ethnographic_texts.csv`
   - Define features in `data/features.csv`
   - Add human annotations in `data/human_coded.csv`

4. **Run the annotation pipeline**:
   ```
   python src/main.py
   ```

5. **Evaluate model performance**:
   ```
   python src/evaluation.py
   ```