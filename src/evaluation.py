import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, 
                            cohen_kappa_score, matthews_corrcoef, precision_score, 
                            recall_score, roc_curve, auc)
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import friedmanchisquare, wilcoxon
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("../logs/evaluation.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

def load_data():
    """Load human-coded and model prediction data"""
    try:
        human_data = pd.read_csv("../data/human_coded.csv")
        model_data = pd.read_csv("../data/model_predictions.csv")
        logger.info(f"Loaded data: {len(human_data)} human records, {len(model_data)} model predictions")
        return human_data, model_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def prediction_agreement(human_data, model_data):
    """
    Creates a binary correctness dataset (1 = correct, 0 = incorrect) for all features,
    and computes overall metrics with confidence intervals using bootstrap.
    """
    feature_columns = human_data.columns.difference(["id", "ritual"])
    models = model_data["model"].unique()
    results = []
    
    # For storing per-feature metrics
    feature_metrics = {model: {} for model in models}
    
    logger.info(f"Computing agreement metrics for {len(models)} models across {len(feature_columns)} features")
    
    for model in models:
        model_subset = model_data[model_data["model"] == model].set_index("ritual")[feature_columns]
        human_subset = human_data.set_index("ritual")[feature_columns]
        
        # Match indices to ensure we're comparing the same rituals
        common_indices = human_subset.index.intersection(model_subset.index)
        if len(common_indices) < len(human_subset):
            logger.warning(f"Only {len(common_indices)}/{len(human_subset)} rituals found for model {model}")
        
        human_matched = human_subset.loc[common_indices]
        model_matched = model_subset.loc[common_indices]
        
        # Overall correctness
        correctness_df = (human_matched == model_matched).astype(int)
        
        # Calculate overall metrics
        metrics = calculate_metrics(correctness_df.values.flatten(), np.ones(correctness_df.size))
        metrics["model"] = model
        results.append(metrics)
        
        # Per-feature metrics
        for feature in feature_columns:
            feature_correctness = correctness_df[feature].values
            feature_metrics[model][feature] = calculate_metrics(
                feature_correctness, 
                np.ones(len(feature_correctness))
            )
    
    # Save per-feature metrics
    feature_results = []
    for model in models:
        for feature, metrics in feature_metrics[model].items():
            metrics["model"] = model
            metrics["feature"] = feature
            feature_results.append(metrics)
    
    feature_df = pd.DataFrame(feature_results)
    feature_df.to_csv("../results/per_feature_metrics.csv", index=False)
    logger.info(f"Saved per-feature metrics for {len(models)} models")
    
    return pd.DataFrame(results)

def calculate_metrics(y_true, y_pred, n_bootstrap=1000, confidence=0.95):
    """Calculate metrics with bootstrap confidence intervals"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }
    
    # Bootstrap confidence intervals
    if len(y_true) > 10:  # Only bootstrap if we have enough samples
        bootstrap_metrics = {metric: [] for metric in metrics.keys()}
        indices = np.arange(len(y_true))
        
        for _ in range(n_bootstrap):
            bootstrap_indices = np.random.choice(indices, size=len(indices), replace=True)
            bootstrap_true = np.array(y_true)[bootstrap_indices]
            bootstrap_pred = np.array(y_pred)[bootstrap_indices]
            
            bootstrap_metrics["accuracy"].append(accuracy_score(bootstrap_true, bootstrap_pred))
            bootstrap_metrics["precision"].append(precision_score(bootstrap_true, bootstrap_pred, zero_division=0))
            bootstrap_metrics["recall"].append(recall_score(bootstrap_true, bootstrap_pred, zero_division=0))
            bootstrap_metrics["f1"].append(f1_score(bootstrap_true, bootstrap_pred, average="macro", zero_division=0))
            bootstrap_metrics["cohen_kappa"].append(cohen_kappa_score(bootstrap_true, bootstrap_pred))
            bootstrap_metrics["mcc"].append(matthews_corrcoef(bootstrap_true, bootstrap_pred))
        
        # Calculate confidence intervals
        alpha = (1 - confidence) / 2
        for metric in metrics.keys():
            metrics[f"{metric}_lower"] = np.quantile(bootstrap_metrics[metric], alpha)
            metrics[f"{metric}_upper"] = np.quantile(bootstrap_metrics[metric], 1 - alpha)
    
    return metrics

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
        logger.warning(f"McNemar test failed: {e}. Attempting with correction.")
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
        logger.info("Fewer than 2 models, skipping statistical tests")
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
        logger.warning("Insufficient data for statistical tests")
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
                            logger.warning(f"ROC calculation failed for {feature}, {model}: {e}")
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
                    logger.error(f"Error in per_feature_analysis for {feature}, {model}: {e}")
        
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

def main():
    try:
        human_data, model_data = load_data()

        logger.info("Computing prediction agreement metrics...")
        prediction_agreement_metrics = prediction_agreement(human_data, model_data)
        prediction_agreement_metrics.to_csv("../results/overall_prediction_agreement.csv", index=False)
        
        logger.info("Performing statistical tests between models...")
        stats_results = statistical_tests(human_data, model_data)
        if stats_results:
            with open("../results/statistical_tests.json", "w") as f:
                import json
                json.dump(stats_results, f, indent=4)
        
        logger.info("Performing per-feature analysis...")
        feature_analysis = per_feature_analysis(human_data, model_data)
        
        logger.info("Evaluation complete.")
        return prediction_agreement_metrics, feature_analysis, stats_results
    
    except Exception as e:
        logger.error(f"Error in evaluation process: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()