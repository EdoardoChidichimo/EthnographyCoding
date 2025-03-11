import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_dir / "visualization.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Set visualization styles
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (10, 6),
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def ensure_directory(path):
    """Ensure that directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def load_results():
    """Load results from CSV files"""
    base_path = Path(__file__).parent.parent
    results_path = base_path / "results"
    eval_path = base_path / "evaluation"
    ensure_directory(eval_path)
    
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

def plot_model_performance(df, output_dir="../results/figures"):
    """
    Plots model performance across evaluation metrics with error bars.
    """
    ensure_directory(output_dir)
    
    metrics = ["accuracy", "precision", "recall", "f1", "cohen_kappa", "mcc"]
    metric_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "cohen_kappa": "Cohen's Kappa",
        "mcc": "Matthews Correlation"
    }
    
    # Check which metrics are available
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        logger.warning("No metrics found in dataframe")
        return
    
    # Create a melted dataframe for plotting
    df_melted = df.melt(id_vars=["model"], 
                        value_vars=available_metrics, 
                        var_name="Metric", 
                        value_name="Score")
    
    # Map metric codes to display names
    df_melted["Metric"] = df_melted["Metric"].map(lambda x: metric_names.get(x, x))
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Plot bars
    ax = sns.barplot(x="Metric", y="Score", hue="model", data=df_melted, 
                     palette="deep", edgecolor="black", linewidth=1)
    
    # Add error bars if confidence intervals are available
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        
        for j, metric in enumerate(available_metrics):
            lower_col = f"{metric}_lower"
            upper_col = f"{metric}_upper"
            
            if lower_col in model_data.columns and upper_col in model_data.columns:
                # Calculate error bar sizes
                central = model_data[metric].values[0]
                lower = model_data[lower_col].values[0]
                upper = model_data[upper_col].values[0]
                yerr = np.array([[central - lower], [upper - central]])
                
                # Find the x position of the bar
                bar_index = j + (i * 0.8 / len(df["model"].unique())) - 0.4  # Adjust based on number of models
                
                ax.errorbar(bar_index, central, yerr=yerr.flatten(), fmt='none', c='black', capsize=5)
    
    # Enhance the plot
    plt.title("Model Performance Across Evaluation Metrics", fontsize=18, weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model", loc="upper right", frameon=True, fancybox=True, framealpha=0.9)
    plt.ylabel("Score")
    plt.xlabel("")
    plt.ylim(0, 1.05)  # Leave room for error bars
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = f"{output_dir}/model_performance.png"
    plt.savefig(output_path)
    logger.info(f"Saved model performance plot to {output_path}")
    plt.close()

# def plot_model_performance(df, output_dir="../results/figures"):
#     """
#     Plots model performance across evaluation metrics with error bars.
#     """
#     ensure_directory(output_dir)
    
#     metrics = ["accuracy", "precision", "recall", "f1", "cohen_kappa", "mcc"]
#     metric_names = {
#         "accuracy": "Accuracy",
#         "precision": "Precision",
#         "recall": "Recall",
#         "f1": "F1 Score",
#         "cohen_kappa": "Cohen's Kappa",
#         "mcc": "Matthews Correlation"
#     }
    
#     # Check which metrics are available
#     available_metrics = [m for m in metrics if m in df.columns]
#     if not available_metrics:
#         logger.warning("No metrics found in dataframe")
#         return
    
#     # Create a melted dataframe for plotting
#     df_melted = df.melt(id_vars=["model"], 
#                         value_vars=available_metrics, 
#                         var_name="Metric", 
#                         value_name="Score")
    
#     # Map metric codes to display names
#     df_melted["Metric"] = df_melted["Metric"].map(lambda x: metric_names.get(x, x))
    
#     # Create the figure
#     plt.figure(figsize=(12, 8))
    
#     # Plot bars
#     ax = sns.barplot(x="Metric", y="Score", hue="model", data=df_melted, 
#                      palette="deep", edgecolor="black", linewidth=1)
    
#     # Add error bars if confidence intervals are available
#     for i, model in enumerate(df["model"].unique()):
#         model_data = df[df["model"] == model]
        
#         for j, metric in enumerate(available_metrics):
#             lower_col = f"{metric}_lower"
#             upper_col = f"{metric}_upper"
            
#             if lower_col in model_data.columns and upper_col in model_data.columns:
#                 # Calculate error bar sizes
#                 central = model_data[metric].values[0]
#                 lower = model_data[lower_col].values[0]
#                 upper = model_data[upper_col].values[0]
#                 yerr = np.array([[central - lower], [upper - central]])
                
#                 # Find the x position of the bar
#                 bar_index = j + i/len(df["model"].unique())*0.8 - 0.4  # Adjust based on number of models
                
#                 ax.errorbar(bar_index, central, yerr=yerr, fmt='none', c='black', capsize=5)
    
#     # Enhance the plot
#     plt.title("Model Performance Across Evaluation Metrics", fontsize=18, weight="bold")
#     plt.xticks(rotation=45, ha="right")
#     plt.legend(title="Model", loc="upper right", frameon=True, fancybox=True, framealpha=0.9)
#     plt.ylabel("Score")
#     plt.xlabel("")
#     plt.ylim(0, 1.05)  # Leave room for error bars
    
#     # Add a grid for better readability
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
    
#     plt.tight_layout()
#     output_path = f"{output_dir}/model_performance.png"
#     plt.savefig(output_path)
#     logger.info(f"Saved model performance plot to {output_path}")
#     plt.close()





def main():
    """Generate model performance visualization from evaluation results."""
    print("Starting visualization generation...")
    
    try:
        # Load results
        data = load_results()
        
        # Create figures directory
        figures_dir = Path(__file__).parent.parent / "results/figures"
        ensure_directory(figures_dir)
        
        # Generate model performance plots if we have model evaluations
        if 'model_evaluations' in data and not data['model_evaluations'].empty:
            print("Generating model performance plot...")
            
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
            
            # Generate overall performance plot with confidence intervals
            plot_model_performance(model_summary, figures_dir)
            
            print("Visualization generation complete. Plot saved in:", figures_dir)
        else:
            print("No model evaluation data found.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run evaluation first to generate the necessary data files.")
    except Exception as e:
        print(f"Error generating visualization: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging

if __name__ == "__main__":
    main()