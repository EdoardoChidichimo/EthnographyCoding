# visualisation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (10, 6),
    "savefig.dpi": 300
})

def load_results():
    return pd.read_csv("data/overall_prediction_agreement.csv")

def plot_model_performance(df):
    """
    Plots model performance across evaluation metrics.
    If error bar columns (e.g., accuracy_err) are present, they will be added.
    """
    metrics = ["accuracy", "cohen_kappa", "macro_f1", "mcc"]
    df_melted = df.melt(id_vars=["model"], value_vars=metrics, var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Metric", y="Score", hue="model", data=df_melted, palette="deep", edgecolor="black")
    
    # Optionally add error bars if error columns exist in df
    for i, row in df.iterrows():
        for metric in metrics:
            err_col = f"{metric}_err"
            if err_col in df.columns:
                # Compute the x coordinate for the bar
                x = list(df_melted["Metric"]).index(metric) + i * 0.2  # adjust based on bar positions
                ax.errorbar(x, row[metric], yerr=row[err_col], fmt='none', c='black')
    
    plt.title("Model Performance Across Evaluation Metrics", fontsize=18, weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model", loc="lower right")
    plt.ylabel("Score")
    plt.xlabel("")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/model_performance.png", dpi=300)
    plt.show()

def plot_correlation_matrix(df):
    metrics = ["accuracy", "cohen_kappa", "macro_f1", "mcc"]
    corr_matrix = df[metrics].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True, cbar=True)
    
    plt.title("Correlation Between Performance Metrics", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig("results/performance_correlation.png", dpi=300)
    plt.show()

def main():
    df = load_results()
    print("Generating model performance plots...")
    plot_model_performance(df)
    plot_correlation_matrix(df)

if __name__ == "__main__":
    main()