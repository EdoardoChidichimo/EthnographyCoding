import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, 
                            cohen_kappa_score, matthews_corrcoef, precision_score, 
                            recall_score, roc_curve, auc)
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import friedmanchisquare, wilcoxon
from pathlib import Path
from config import LLM_MODELS

def load_data():
    """Load human-coded and model prediction data"""
    # Load human-coded data
    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "results"
    
    human_data = pd.read_csv(data_dir / "human_coded.csv")
    print(f"Loaded {len(human_data)} human-coded records")
    
    # Load each model's predictions
    model_data = []
    for model in LLM_MODELS:
        model_file = results_dir / f"{model}_annotations.csv"
        if model_file.exists():
            df = pd.read_csv(model_file)
            df['model'] = model  # Add model identifier
            model_data.append(df)
            print(f"Loaded {len(df)} predictions for {model}")
        else:
            print(f"No predictions file found for {model}")
    
    if not model_data:
        raise FileNotFoundError("No model prediction files found")
        
    # Combine all model predictions
    model_predictions = pd.concat(model_data, ignore_index=True)
    
    return human_data, model_predictions

def evaluate_model_predictions(human_data, model_predictions, model):
    """Evaluate predictions for a specific model"""
    # Get predictions for this model
    model_df = model_predictions[model_predictions['model'] == model].copy()
    
    # Ensure ritual_number is first column and both DataFrames are sorted
    human_data = human_data.sort_values('ritual_number')
    model_df = model_df.sort_values('ritual_number')
    
    # Get common ritual numbers
    common_rituals = set(human_data['ritual_number']) & set(model_df['ritual_number'])
    
    if not common_rituals:
        print(f"No common rituals found for {model}")
        return None
    
    # Filter to common rituals
    human_subset = human_data[human_data['ritual_number'].isin(common_rituals)]
    model_subset = model_df[model_df['ritual_number'].isin(common_rituals)]
    
    # Sort both by ritual_number to ensure alignment
    human_subset = human_subset.sort_values('ritual_number')
    model_subset = model_subset.sort_values('ritual_number')
    
    # Get feature columns (excluding ritual_number and model)
    feature_columns = [col for col in model_subset.columns 
                      if col not in ['ritual_number', 'model']]
    
    # Calculate metrics for each feature
    feature_metrics = {}
    for feature in feature_columns:
        if feature in human_subset.columns:
            try:
                # Get values and handle missing data
                y_true = human_subset[feature].values
                y_pred = model_subset[feature].values
                
                # Handle missing values for both numeric and string data
                valid_mask = np.ones(len(y_true), dtype=bool)
                for i, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
                    # Check for pandas NA, None, or empty string
                    if pd.isna(true_val) or pd.isna(pred_val) or true_val == '' or pred_val == '':
                        valid_mask[i] = False
                
                if not valid_mask.any():
                    print(f"No valid predictions for feature {feature}")
                    continue
                    
                y_true = y_true[valid_mask]
                y_pred = y_pred[valid_mask]
                
                # Calculate metrics only if we have valid data
                if len(y_true) > 0:
                    metrics = calculate_metrics(y_true, y_pred)
                    if metrics:
                        metrics['n_samples'] = len(y_true)  # Add sample size info
                        metrics['n_missing'] = sum(~valid_mask)  # Add count of missing values
                        feature_metrics[feature] = metrics
                else:
                    print(f"No valid samples for feature {feature}")
            except Exception as e:
                print(f"Error calculating metrics for {feature}: {e}")
    
    return feature_metrics

def calculate_metrics(y_true, y_pred, n_bootstrap=1000, confidence=0.95):
    """Calculate metrics with bootstrap confidence intervals"""
    # Try to convert to numeric, but keep as strings if not possible
    try:
        y_true_num = pd.to_numeric(y_true, errors='raise')
        y_pred_num = pd.to_numeric(y_pred, errors='raise')
        y_true = y_true_num
        y_pred = y_pred_num
        is_numeric = True
    except:
        is_numeric = False
    
    # Get unique values
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    unique_labels = np.unique(np.concatenate([unique_true, unique_pred]))
    is_binary = len(unique_labels) <= 2
    
    # Handle edge case where all samples are same class
    if len(unique_true) == 1 and len(unique_pred) == 1 and unique_true[0] == unique_pred[0]:
        if is_numeric:
            return {
                "accuracy": 1.0,
                "precision": 1.0 if unique_true[0] == 1 else 0.0,
                "recall": 1.0 if unique_true[0] == 1 else 0.0,
                "f1": 1.0 if unique_true[0] == 1 else 0.0,
                "cohen_kappa": 0.0,
                "mcc": 0.0,
                "roc_auc": None
            }
        else:
            return {
                "accuracy": 1.0,
                "precision": None,  # Not meaningful for non-binary data
                "recall": None,
                "f1": None,
                "cohen_kappa": 0.0,
                "mcc": 0.0,
                "roc_auc": None
            }
    
    try:
        # Calculate base metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred)
        }
        
        # Only calculate these metrics if we have multiple classes and numeric data
        if len(unique_true) > 1 and is_numeric:
            metrics.update({
                "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
                "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
                "cohen_kappa": cohen_kappa_score(y_true, y_pred),
                "mcc": matthews_corrcoef(y_true, y_pred)
            })
        else:
            metrics.update({
                "precision": None,
                "recall": None,
                "f1": None,
                "cohen_kappa": cohen_kappa_score(y_true, y_pred) if len(unique_true) > 1 else 0.0,
                "mcc": None
            })
            
        # Bootstrap confidence intervals if enough samples
        if len(y_true) > 10 and len(unique_true) > 1:
            bootstrap_metrics = {metric: [] for metric in metrics.keys()}
            indices = np.arange(len(y_true))
            
            for _ in range(n_bootstrap):
                bootstrap_indices = np.random.choice(indices, size=len(indices), replace=True)
                bootstrap_true = y_true[bootstrap_indices]
                bootstrap_pred = y_pred[bootstrap_indices]
                
                # Only calculate bootstrap if we have variation in the sample
                if len(np.unique(bootstrap_true)) > 1:
                    bootstrap_metrics["accuracy"].append(accuracy_score(bootstrap_true, bootstrap_pred))
                    
                    if is_numeric:
                        bootstrap_metrics["precision"].append(precision_score(bootstrap_true, bootstrap_pred, average='macro', zero_division=0))
                        bootstrap_metrics["recall"].append(recall_score(bootstrap_true, bootstrap_pred, average='macro', zero_division=0))
                        bootstrap_metrics["f1"].append(f1_score(bootstrap_true, bootstrap_pred, average='macro', zero_division=0))
                        bootstrap_metrics["cohen_kappa"].append(cohen_kappa_score(bootstrap_true, bootstrap_pred))
                        bootstrap_metrics["mcc"].append(matthews_corrcoef(bootstrap_true, bootstrap_pred))
            
            # Calculate confidence intervals
            alpha = (1 - confidence) / 2
            for metric in metrics.keys():
                if len(bootstrap_metrics.get(metric, [])) > 0:
                    metrics[f"{metric}_lower"] = np.quantile(bootstrap_metrics[metric], alpha)
                    metrics[f"{metric}_upper"] = np.quantile(bootstrap_metrics[metric], 1 - alpha)
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

def mcnemar_test_feature(human, model):
    """
    Performs McNemar's test for a binary feature.
    Expects two pandas Series (human and model predictions).
    Returns statistic and p-value.
    """
    n00 = sum((human == 0) & (model == 0))
    n01 = sum((human == 0) & (model == 1))
    n10 = sum((human == 1) & (model == 0))
    n11 = sum((human == 1) & (model == 1))
    table = [[n00, n01], [n10, n11]]
    
    try:
        result = mcnemar(table, exact=True)
        return result.statistic, result.pvalue
    except Exception as e:
        print(f"McNemar test failed: {e}. Attempting with correction.")
        # Apply correction if we have zeros in the table
        table = [[max(cell, 0.5) for cell in row] for row in table]
        result = mcnemar(table, exact=False, correction=True)
        return result.statistic, result.pvalue

def statistical_tests(human_data, model_data):
    """
    Performs statistical tests to compare models:
    1. Friedman test to determine if there are significant differences
    2. Pairwise Wilcoxon signed-rank tests if Friedman test is significant
    """
    feature_columns = human_data.columns.difference(["id", "ritual"])
    models = model_data["model"].unique()
    
    if len(models) < 2:
        print("Fewer than 2 models, skipping statistical tests")
        return None
    
    # Prepare data for statistical tests
    model_accuracies = {model: [] for model in models}
    
    for feature in feature_columns:
        human_feature = human_data.set_index("ritual")[feature]
        
        for model in models:
            model_subset = model_data[model_data["model"] == model].set_index("ritual")
            common_indices = human_feature.index.intersection(model_subset.index)
            
            if len(common_indices) > 0:
                model_feature = model_subset.loc[common_indices, feature]
                accuracy = (human_feature.loc[common_indices] == model_feature).mean()
                model_accuracies[model].append(accuracy)
    
    # Check if we have enough data
    if all(len(accs) >= 5 for model, accs in model_accuracies.items()):
        # Friedman test
        friedman_data = [model_accuracies[model] for model in models]
        stat, p_value = friedmanchisquare(*friedman_data)
        
        results = {
            "friedman_statistic": stat,
            "friedman_p_value": p_value,
            "significant_difference": p_value < 0.05
        }
        
        # If Friedman test is significant, perform pairwise Wilcoxon tests
        if p_value < 0.05 and len(models) > 1:
            pairwise_results = []
            
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    stat, p_val = wilcoxon(model_accuracies[model1], model_accuracies[model2])
                    pairwise_results.append({
                        "model1": model1,
                        "model2": model2,
                        "wilcoxon_statistic": stat,
                        "wilcoxon_p_value": p_val,
                        "significant_difference": p_val < 0.05 / (len(models) * (len(models) - 1) / 2)  # Bonferroni correction
                    })
            
            results["pairwise_tests"] = pairwise_results
            
            # Save pairwise results
            pd.DataFrame(pairwise_results).to_csv("../results/pairwise_model_tests.csv", index=False)
        
        return results
    else:
        print("Insufficient data for statistical tests")
        return None

def per_feature_analysis(human_data, model_data):
    """
    Computes confusion matrices and McNemar's test for each feature.
    Returns a dictionary with feature names as keys and results as values.
    """
    feature_columns = human_data.columns.difference(["id", "ritual"])
    models = model_data["model"].unique()
    analysis_results = {}
    
    for feature in feature_columns:
        human_feature = human_data.set_index("ritual")[feature]
        feature_results = {}
        
        for model in models:
            model_subset = model_data[model_data["model"] == model].set_index("ritual")
            common_indices = human_feature.index.intersection(model_subset.index)
            
            if len(common_indices) > 0:
                model_feature = model_subset.loc[common_indices, feature]
                
                try:
                    conf_matrix = confusion_matrix(human_feature.loc[common_indices], model_feature)
                    # Add precision, recall, F1 for this feature
                    precision = precision_score(human_feature.loc[common_indices], model_feature, zero_division=0)
                    recall = recall_score(human_feature.loc[common_indices], model_feature, zero_division=0)
                    f1 = f1_score(human_feature.loc[common_indices], model_feature, average='binary', zero_division=0)
                    
                    # Calculate ROC curve if feature is binary
                    unique_values = pd.concat([human_feature, model_feature]).unique()
                    if len(unique_values) == 2:
                        try:
                            fpr, tpr, _ = roc_curve(human_feature.loc[common_indices], model_feature)
                            roc_auc = auc(fpr, tpr)
                        except Exception as e:
                            print(f"ROC calculation failed for {feature}, {model}: {e}")
                            roc_auc = None
                    else:
                        roc_auc = None
                    
                    stat, pval = mcnemar_test_feature(human_feature.loc[common_indices], model_feature)
                    
                    feature_results[model] = {
                        "confusion_matrix": conf_matrix,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "roc_auc": roc_auc,
                        "mcnemar_stat": stat,
                        "p_value": pval,
                        "significant_difference": pval < 0.05
                    }
                except Exception as e:
                    print(f"Error in per_feature_analysis for {feature}, {model}: {e}")
        
        analysis_results[feature] = feature_results
    
    # Save feature results as a flattened DataFrame
    flattened_results = []
    for feature, models_dict in analysis_results.items():
        for model, metrics in models_dict.items():
            row = {"feature": feature, "model": model}
            for metric, value in metrics.items():
                if metric != "confusion_matrix":  # Skip confusion matrix in flattened output
                    row[metric] = value
            flattened_results.append(row)
    
    pd.DataFrame(flattened_results).to_csv("../results/feature_analysis_detailed.csv", index=False)
    
    return analysis_results

def load_results():
    """Load results from CSV files"""
    base_path = Path(__file__).parent.parent
    results_path = base_path / "results"
    eval_path = base_path / "evaluation"
    
    files = {
        "model_evaluations": [],  # Will hold individual model evaluation files
        "summary": eval_path / "evaluation_summary.csv",
        "feature_analysis": results_path / "feature_analysis_detailed.csv",
        "pairwise": results_path / "pairwise_model_tests.csv"
    }
    
    data = {}
    
    # Load model-specific evaluation files
    from config import LLM_MODELS
    for model in LLM_MODELS:
        model_file = eval_path / f"{model}_evaluation.csv"
        if model_file.exists():
            df = pd.read_csv(model_file)
            df['model'] = model
            data.setdefault('model_evaluations', []).append(df)
            print(f"Loaded evaluation data for {model}")
    
    # Combine model evaluations if any exist
    if 'model_evaluations' in data and data['model_evaluations']:
        data['model_evaluations'] = pd.concat(data['model_evaluations'], ignore_index=True)
    
    # Load other result files
    for key, filepath in files.items():
        if key != 'model_evaluations':  # Skip as we handled this above
            if filepath.exists():
                data[key] = pd.read_csv(filepath)
                print(f"Loaded {key} data")
            else:
                print(f"Optional file not found: {filepath}")
    
    if not data:
        raise FileNotFoundError("No evaluation results found")
    
    return data


def final_evaluation():

    data = load_results()
        
    results_dir = Path(__file__).parent.parent / "results"
    
    # Generate model performance plots if we have model evaluations
    if 'model_evaluations' in data and not data['model_evaluations'].empty:
        
        # Calculate mean metrics and preserve confidence intervals
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'cohen_kappa', 'mcc']
        agg_dict = {}
        for metric in metrics:
            agg_dict[metric] = 'mean'
            if f"{metric}_lower" in data['model_evaluations'].columns:
                agg_dict[f"{metric}_lower"] = 'mean'
            if f"{metric}_upper" in data['model_evaluations'].columns:
                agg_dict[f"{metric}_upper"] = 'mean'
        
        model_summary = data['model_evaluations'].groupby('model').agg(agg_dict).reset_index()

        # Save model summary to CSV
        model_summary.to_csv(results_dir / "final_aggregated_model_summary.csv", index=False)

def main():
    # Create evaluation directory
    eval_dir = Path(__file__).parent.parent / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        human_data, model_predictions = load_data()
        
        # Evaluate each model
        all_results = {}
        for model in LLM_MODELS:
            print(f"Evaluating {model}...")
            metrics = evaluate_model_predictions(human_data, model_predictions, model)
            if metrics:
                all_results[model] = metrics
                
                # Convert metrics to DataFrame format
                model_results = []
                for feature, feature_metrics in metrics.items():
                    row = {'feature': feature}
                    row.update(feature_metrics)
                    model_results.append(row)
                
                model_results_df = pd.DataFrame(model_results)
                
                # Save model-specific results with confidence intervals
                model_results_df.to_csv(eval_dir / f"{model}_evaluation.csv", index=False)
        
        print("Evaluation complete")
        
    except Exception as e:
        print(f"Error in evaluation process: {e}")
        raise

    final_evaluation()

if __name__ == "__main__":
    main()