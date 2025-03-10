# Ethnographic Feature Extraction Using LLMs

Can large language models (LLMs) perform ethnographic feature extraction at a level comparable to human coders? 

## Considerations:
- LLM sampling (just single sample + bootstrap (perhaps temp=0); or multiple samples and take mode)?

## Data Format

### ritual_features.csv
The file should contain three columns:

- `feature_name`: The name of the feature to be annotated
- `feature_description`: A detailed description of the feature
- `feature_options`: Valid options/values for this feature (e.g., "low, medium, high" or "0, 1" for binary features)


## Key Features

- **Multi-model comparison**: Evaluates annotations from OpenAI, Anthropic, and Hugging Face models
- **Comprehensive evaluation**: Calculates accuracy, precision, recall, F1 score, Cohen's Kappa, and Matthews Correlation Coefficient
- **Statistical significance testing**: Uses Friedman and Wilcoxon signed-rank tests to determine if performance differences between models are statistically significant
- **Robust error handling**: Implements retry mechanisms with exponential backoff for API rate limits and timeouts
- **Detailed visualisations**: Generates performance heatmaps, confusion matrices, and statistical significance matrices

## Project Structure

```
├── data/
│   ├── ritual_texts.csv         # Source ethnographic texts with ritual_number and paragraph
│   ├── ritual_features.csv      # Features to be annotated with descriptions
│   ├── human_coded.csv          # Human annotations (ground truth)
│   └── model_predictions.csv    # Generated model predictions
├── results/
│   ├── figures/                 # Generated visualisations
│   ├── final_coded_ethnography.json # Nested results structure
│   ├── overall_prediction_agreement.csv # Overall model performance metrics
│   ├── per_feature_metrics.csv  # Performance metrics by feature
│   ├── feature_analysis_detailed.csv # Detailed feature analysis with confusion matrices
│   ├── pairwise_model_tests.csv # Statistical comparisons between model pairs
│   └── statistical_tests.json   # Results of statistical significance tests
├── logs/
│   ├── evaluation.log           # Logs from the evaluation process
│   └── visualization.log        # Logs from the visualisation process
├── src/
│   ├── main.py                  # Main script to run the annotation pipeline
│   ├── config.py                # API keys and model configuration
│   ├── llm_annotate.py          # LLM API interaction and annotation logic
│   ├── utils.py                 # Utility functions for data processing
│   ├── evaluation.py            # Statistical evaluation of model performance
│   ├── vis.py                   # Visualisation of results
│   ├── visualisation.py         # Additional visualisation utilities
│   ├── topic_modelling.py       # Topic modelling for ritual texts
│   └── extract_text.py          # Text extraction utilities
└── requirements.txt             # Project dependencies
```

## Workflow

1. **Data Loading**: Ethnographic texts are loaded from `ritual_texts.csv`
2. **Feature Definition**: Ritual features are loaded from `ritual_features.csv`
3. **LLM Annotation**: Each text is processed by multiple LLM models
4. **Result Aggregation**: Results are aggregated and normalised
5. **Evaluation**: Model predictions are compared against human annotations
6. **Statistical Analysis**: Performance differences are tested for statistical significance
7. **Visualisation**: Results are visualised through various plots and figures

## Statistical Analysis

The project implements several statistical analyses:

- **Performance Metrics**: Accuracy, precision, recall, F1 score, Cohen's Kappa, and Matthews Correlation Coefficient
- **Bootstrap Confidence Intervals**: Non-parametric estimation of uncertainty in performance metrics
- **Friedman Test**: Non-parametric test to detect differences across multiple models
- **Wilcoxon Signed-Rank Test**: Pairwise comparison of models with Bonferroni correction
- **McNemar's Test**: Evaluates whether models differ in their error patterns
- **ROC Curve Analysis**: For binary features, evaluates true positive vs. false positive rates

## Visualisations

The project generates several visualisations:

- **Model Performance Comparison**: Bar charts with error bars showing performance across metrics
- **Feature Performance Heatmap**: Shows which models perform best on which features
- **Confusion Matrices**: For each model-feature combination
- **Statistical Significance Matrix**: Shows which model differences are statistically significant
- **Summary Figure**: Comprehensive overview of key findings

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
   - Place your ethnographic texts in `data/ritual_texts.csv`
   - Define ritual features in `data/ritual_features.csv`
   - Add human annotations in `data/human_coded.csv`

4. **Run the annotation pipeline**:
   ```
   python src/main.py
   ```

5. **Evaluate model performance**:
   ```
   python src/evaluation.py
   ```

6. **Generate visualisations**:
   ```
   python src/vis.py
   ```

## Error Handling

The system implements robust error handling for API interactions:
- Rate limit detection with exponential backoff
- Timeout handling with retries
- JSON parsing error recovery
- Comprehensive logging

## Future Improvements

Potential enhancements to the project:
- Implement multiple LLM runs per ritual to estimate model uncertainty
- Implement active learning to improve model performance over time