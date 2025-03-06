import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, classification_report, matthews_corrcoef, jaccard_score, confusion_matrix, accuracy_score, f1_score
from scipy.stats import chi2_contingency
from statsmodels.stats.inter_rater import fleiss_kappa, cohen_kappa


def load_data():
    """Loads human-coded and LLM-coded data."""
    human_data = pd.read_csv("data/human_coded.csv")  # Ground truth
    model_data = pd.read_csv("data/model_predictions.csv")  # LLM outputs
    return human_data, model_data

def prediction_agreement(human_data, model_data):
    """Creates a binary correctness dataset (1 = correct, 0 = incorrect) for all features."""
    feature_columns = human_data.columns.difference(["id", "model"])

    results = []
    for model in model_data["model"].unique():
        model_subset = model_data[model_data["model"] == model].set_index("id")[feature_columns]

        # Compute binary correctness: 1 if prediction matches human label, else 0
        correctness_df = (human_data[feature_columns] == model_subset).astype(int)

        accuracy = correctness_df.mean().mean()  # Average correctness across all features (first average by feature, then across features)
        cohen_kappa = cohen_kappa_score(correctness_df.values.flatten(), [1] * correctness_df.size)  # Compare against perfect agreement (accounts for chance)
        macro_f1 = f1_score(correctness_df.values.flatten(), [1] * correctness_df.size, average="macro")  # Compare to ideal scenario
        mcc = matthews_corrcoef(correctness_df.values.flatten(), [1] * correctness_df.size)

        results.append({
            "model": model,
            "accuracy": accuracy,
            "cohen_kappa": cohen_kappa,
            "macro_f1": macro_f1,
            "mcc": mcc
        })

    return pd.DataFrame(results)

def main():
    human_data, model_data = load_data()
    prediction_agreement_metrics = prediction_agreement(human_data, model_data)
    prediction_agreement_metrics.to_csv("data/prediction_agreement_metrics.csv", index=False)

if __name__ == "__main__":
    main()