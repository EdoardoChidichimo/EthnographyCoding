# evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, cohen_kappa_score, 
                             matthews_corrcoef)
from statsmodels.stats.contingency_tables import mcnemar

def load_data():
    """Loads human-coded and LLM-coded data."""
    human_data = pd.read_csv("../data/human_coded.csv")  # Ground truth
    model_data = pd.read_csv("../data/model_predictions.csv")  # LLM outputs
    return human_data, model_data

def prediction_agreement(human_data, model_data):
    """
    Creates a binary correctness dataset (1 = correct, 0 = incorrect) for all features,
    and computes overall metrics.
    """
    feature_columns = human_data.columns.difference(["id", "model"])
    results = []
    for model in model_data["model"].unique():
        model_subset = model_data[model_data["model"] == model].set_index("id")[feature_columns]
        correctness_df = (human_data[feature_columns] == model_subset).astype(int)
        accuracy = correctness_df.mean().mean()
        cohen_kappa_val = cohen_kappa_score(correctness_df.values.flatten(), [1] * correctness_df.size)
        macro_f1_val = f1_score(correctness_df.values.flatten(), [1] * correctness_df.size, average="macro")
        mcc_val = matthews_corrcoef(correctness_df.values.flatten(), [1] * correctness_df.size)
        results.append({
            "model": model,
            "accuracy": accuracy,
            "cohen_kappa": cohen_kappa_val,
            "macro_f1": macro_f1_val,
            "mcc": mcc_val
        })
    return pd.DataFrame(results)

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
    result = mcnemar(table, exact=True)
    return result.statistic, result.pvalue

def per_feature_analysis(human_data, model_data):
    """
    Computes confusion matrices and McNemar's test for each feature.
    Returns a dictionary with feature names as keys and results as values.
    """
    feature_columns = human_data.columns.difference(["id", "model"])
    analysis_results = {}
    for feature in feature_columns:
        human_feature = human_data[feature]
        feature_results = {}
        for model in model_data["model"].unique():
            model_feature = model_data[model_data["model"] == model].set_index("id")[feature]
            conf_matrix = confusion_matrix(human_feature, model_feature)
            stat, pval = mcnemar_test_feature(human_feature, model_feature)
            feature_results[model] = {"confusion_matrix": conf_matrix, "mcnemar_stat": stat, "p_value": pval}
        analysis_results[feature] = feature_results
    return analysis_results

def main():
    human_data, model_data = load_data()
    prediction_agreement_metrics = prediction_agreement(human_data, model_data)
    prediction_agreement_metrics.to_csv("../data/prediction_agreement_metrics.csv", index=False)
    
    # Optionally, run per-feature error analysis
    feature_analysis = per_feature_analysis(human_data, model_data)
    # Save or log feature_analysis as needed
    print("Per-feature analysis:", feature_analysis)


if __name__ == "__main__":
    main()