import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("../logs/visualization.log"),
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
    results_path = Path("../results")
    ensure_directory(results_path)
    
    files = {
        "overall": results_path / "overall_prediction_agreement.csv",
        "feature": results_path / "per_feature_metrics.csv",
        "feature_detailed": results_path / "feature_analysis_detailed.csv",
        "pairwise": results_path / "pairwise_model_tests.csv"
    }
    
    data = {}
    for key, filepath in files.items():
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
            logger.info(f"Loaded {key} data from {filepath}")
        else:
            logger.warning(f"File not found: {filepath}")
    
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
                bar_index = j + i/len(df["model"].unique())*0.8 - 0.4  # Adjust based on number of models
                
                ax.errorbar(bar_index, central, yerr=yerr, fmt='none', c='black', capsize=5)
    
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

def plot_feature_performance(feature_df, output_dir="../results/figures"):
    """
    Creates a heatmap showing model performance across different features.
    """
    ensure_directory(output_dir)
    
    if feature_df is None or feature_df.empty:
        logger.warning("No feature data available for plotting")
        return
    
    # Pivot the data for the heatmap
    pivot_df = feature_df.pivot_table(index="feature", columns="model", values="f1")
    
    # Sort features by average F1 score
    avg_scores = pivot_df.mean(axis=1)
    pivot_df = pivot_df.loc[avg_scores.sort_values(ascending=False).index]
    
    plt.figure(figsize=(12, max(6, len(pivot_df) * 0.4)))
    
    # Create a custom colormap from red to green
    cmap = sns.color_palette("RdYlGn", 10)
    
    # Plot the heatmap
    ax = sns.heatmap(pivot_df, annot=True, cmap=cmap, fmt=".2f", 
                     linewidths=0.5, cbar_kws={"label": "F1 Score"})
    
    plt.title("Model Performance by Feature (F1 Score)", fontsize=18, weight="bold")
    plt.ylabel("Feature")
    plt.xlabel("Model")
    plt.tight_layout()
    
    output_path = f"{output_dir}/feature_performance_heatmap.png"
    plt.savefig(output_path)
    logger.info(f"Saved feature performance heatmap to {output_path}")
    plt.close()
    
    # Also create a boxplot of performance by feature
    plt.figure(figsize=(10, max(6, len(pivot_df) * 0.3)))
    
    # Melt the pivot table for the boxplot
    melted = pivot_df.reset_index().melt(id_vars=["feature"], var_name="model", value_name="F1 Score")
    
    # Sort by median score
    order = melted.groupby("feature")["F1 Score"].median().sort_values(ascending=False).index
    
    # Create the boxplot
    sns.boxplot(x="F1 Score", y="feature", data=melted, order=order, palette="deep", orient="h")
    
    plt.title("Distribution of F1 Scores Across Models by Feature", fontsize=16, weight="bold")
    plt.tight_layout()
    
    output_path = f"{output_dir}/feature_performance_boxplot.png"
    plt.savefig(output_path)
    logger.info(f"Saved feature performance boxplot to {output_path}")
    plt.close()

def plot_confusion_matrices(feature_detailed_df, output_dir="../results/figures/confusion_matrices"):
    """
    Creates confusion matrices for each model-feature combination.
    """
    ensure_directory(output_dir)
    
    if feature_detailed_df is None or feature_detailed_df.empty:
        logger.warning("No detailed feature data available for plotting confusion matrices")
        return
    
    models = feature_detailed_df["model"].unique()
    features = feature_detailed_df["feature"].unique()
    
    for model in models:
        model_dir = f"{output_dir}/{model}"
        ensure_directory(model_dir)
        
        for feature in features:
            row = feature_detailed_df[(feature_detailed_df["model"] == model) & 
                                      (feature_detailed_df["feature"] == feature)]
            
            if row.empty:
                continue
            
            # Extract confusion matrix (stored as a string)
            conf_matrix_str = row["confusion_matrix"].iloc[0]
            if pd.isna(conf_matrix_str):
                continue
                
            try:
                # Parse the confusion matrix from string representation
                conf_matrix_str = conf_matrix_str.replace('[', '').replace(']', '')
                rows = conf_matrix_str.split()
                
                if len(rows) == 4:  # 2x2 matrix
                    conf_matrix = np.array([[int(rows[0]), int(rows[1])], 
                                            [int(rows[2]), int(rows[3])]])
                else:
                    # Try to parse more complex matrices
                    conf_matrix = np.array([int(x) for x in conf_matrix_str.split()]).reshape(2, 2)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
                            xticklabels=["Negative", "Positive"],
                            yticklabels=["Negative", "Positive"])
                
                # Add labels and title
                plt.ylabel("Human Label (Ground Truth)")
                plt.xlabel("Model Prediction")
                plt.title(f"Confusion Matrix: {model} - {feature}", fontsize=16)
                
                # Add metrics
                metrics_text = (
                    f"Precision: {row['precision'].iloc[0]:.2f}\n"
                    f"Recall: {row['recall'].iloc[0]:.2f}\n"
                    f"F1: {row['f1'].iloc[0]:.2f}"
                )
                
                plt.figtext(0.95, 0.5, metrics_text, fontsize=12, 
                           bbox=dict(facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                
                output_path = f"{model_dir}/{feature}_confusion.png"
                plt.savefig(output_path)
                logger.info(f"Saved confusion matrix to {output_path}")
                plt.close()
                
            except Exception as e:
                logger.error(f"Error creating confusion matrix for {model}-{feature}: {e}")

def plot_statistical_significance(pairwise_df, output_dir="../results/figures"):
    """
    Creates a matrix showing which model differences are statistically significant.
    """
    ensure_directory(output_dir)
    
    if pairwise_df is None or pairwise_df.empty:
        logger.warning("No pairwise statistical test data available")
        return
    
    # Get unique models
    models = sorted(set(pairwise_df["model1"].unique()) | set(pairwise_df["model2"].unique()))
    
    # Create a DataFrame for the significance matrix
    significance_matrix = pd.DataFrame(index=models, columns=models)
    
    # Fill with p-values
    for _, row in pairwise_df.iterrows():
        significance_matrix.loc[row["model1"], row["model2"]] = row["wilcoxon_p_value"]
        significance_matrix.loc[row["model2"], row["model1"]] = row["wilcoxon_p_value"]
    
    # Fill diagonal with 1.0 (no difference)
    for model in models:
        significance_matrix.loc[model, model] = 1.0
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    
    # Custom diverging colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(significance_matrix, dtype=bool))
    
    # Plot
    ax = sns.heatmap(significance_matrix, mask=mask, cmap=cmap, vmax=0.05, 
                     square=True, linewidths=.5, cbar_kws={"shrink": .5, "label": "p-value"})
    
    # Add stars for significance
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:  # Upper triangle only
                val = significance_matrix.loc[model1, model2]
                if not pd.isna(val) and val < 0.05:
                    stars = "***" if val < 0.001 else "**" if val < 0.01 else "*"
                    ax.text(j, i, stars, ha="center", va="center", color="white")
    
    plt.title("Statistical Significance of Model Differences\n(Wilcoxon Signed-Rank Test)", fontsize=16)
    plt.tight_layout()
    
    output_path = f"{output_dir}/model_significance_matrix.png"
    plt.savefig(output_path)
    logger.info(f"Saved statistical significance matrix to {output_path}")
    plt.close()

def create_summary_figure(overall_df, feature_df, output_dir="../results/figures"):
    """
    Creates a comprehensive summary figure with key findings.
    """
    ensure_directory(output_dir)
    
    if overall_df is None or overall_df.empty:
        logger.warning("No overall metrics available for summary figure")
        return
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    
    # 1. Overall model performance
    ax1 = fig.add_subplot(gs[0, 0])
    
    models = overall_df["model"].unique()
    metrics = ["accuracy", "f1"]
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, overall_df["accuracy"], width, label="Accuracy", color="skyblue")
    rects2 = ax1.bar(x + width/2, overall_df["f1"], width, label="F1 Score", color="lightcoral")
    
    ax1.set_ylabel("Score")
    ax1.set_title("Overall Model Performance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Add value labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=10)
    
    add_labels(rects1)
    add_labels(rects2)
    
    # 2. Top performing features
    ax2 = fig.add_subplot(gs[0, 1])
    
    if feature_df is not None and not feature_df.empty:
        # Group by feature and calculate mean F1 score
        feature_means = feature_df.groupby("feature")["f1"].mean().sort_values(ascending=False)
        top_features = feature_means.head(10)  # Top 10 features
        
        # Plot horizontal bars
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_features)))
        bars = ax2.barh(range(len(top_features)), top_features.values, color=colors)
        
        # Add labels
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features.index)
        ax2.set_xlabel("Average F1 Score")
        ax2.set_title("Top Performing Features")
        
        # Add value labels to bars
        for i, v in enumerate(top_features.values):
            ax2.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No feature data available", 
                ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Feature performance heatmap
    ax3 = fig.add_subplot(gs[1, :])
    
    if feature_df is not None and not feature_df.empty:
        # Pivot the data for the heatmap
        pivot_df = feature_df.pivot_table(index="feature", columns="model", values="f1")
        
        # Sort features by average F1 score and select top 15
        avg_scores = pivot_df.mean(axis=1)
        top_features_df = pivot_df.loc[avg_scores.sort_values(ascending=False).index[:15]]
        
        # Plot the heatmap
        cmap = sns.color_palette("RdYlGn", 10)
        sns.heatmap(top_features_df, ax=ax3, annot=True, cmap=cmap, fmt=".2f", 
                    linewidths=0.5, cbar_kws={"label": "F1 Score"})
        
        ax3.set_title("Model Performance Across Top Features (F1 Score)")
    else:
        ax3.text(0.5, 0.5, "No feature data available", 
                ha='center', va='center', transform=ax3.transAxes)
    
    plt.tight_layout()
    fig.suptitle("Model Performance Summary", fontsize=20, y=1.02)
    
    output_path = f"{output_dir}/summary_figure.png"
    plt.savefig(output_path)
    logger.info(f"Saved summary figure to {output_path}")
    plt.close()

def generate_all_visualizations():
    """
    Generate all visualizations from the results data.
    """
    logger.info("Starting visualization generation...")
    
    # Load all result data
    data = load_results()
    
    # Generate visualizations
    if "overall" in data:
        plot_model_performance(data["overall"])
    else:
        logger.warning("Overall metrics data not available")
    
    if "feature" in data:
        plot_feature_performance(data["feature"])
    else:
        logger.warning("Feature metrics data not available")
    
    if "feature_detailed" in data:
        plot_confusion_matrices(data["feature_detailed"])
    else:
        logger.warning("Detailed feature data not available")
    
    if "pairwise" in data:
        plot_statistical_significance(data["pairwise"])
    else:
        logger.warning("Pairwise statistical test data not available")
    
    # Create summary figure with key findings
    create_summary_figure(
        data.get("overall"),
        data.get("feature")
    )
    
    logger.info("Visualization generation completed")

def plot_model_learning_curves(history_df, output_dir="../results/figures"):
    """
    Plot learning curves for model training history.
    
    Parameters:
    -----------
    history_df : pandas.DataFrame
        DataFrame containing training history with columns:
        'epoch', 'model', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
    output_dir : str
        Directory to save the output figures
    """
    ensure_directory(output_dir)
    
    if history_df is None or history_df.empty:
        logger.warning("No training history data available")
        return
    
    # Get unique models
    models = history_df["model"].unique()
    
    # Plot learning curves for each model
    for model in models:
        model_data = history_df[history_df["model"] == model]
        
        if model_data.empty:
            continue
            
        # Create a figure with two subplots (loss and accuracy)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot loss curves
        ax1.plot(model_data["epoch"], model_data["train_loss"], 'b-', label='Training Loss')
        ax1.plot(model_data["epoch"], model_data["val_loss"], 'r-', label='Validation Loss')
        ax1.set_title(f"{model}: Loss During Training")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot accuracy curves
        ax2.plot(model_data["epoch"], model_data["train_accuracy"], 'b-', label='Training Accuracy')
        ax2.plot(model_data["epoch"], model_data["val_accuracy"], 'r-', label='Validation Accuracy')
        ax2.set_title(f"{model}: Accuracy During Training")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        output_path = f"{output_dir}/{model}_learning_curves.png"
        plt.savefig(output_path)
        logger.info(f"Saved learning curves for {model} to {output_path}")
        plt.close()

def plot_feature_correlation_matrix(data, output_dir="../results/figures"):
    """
    Plot a correlation matrix of feature presence.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with columns for each feature (binary indicators)
    output_dir : str
        Directory to save the output figures
    """
    ensure_directory(output_dir)
    
    if data is None or data.empty:
        logger.warning("No feature data available for correlation analysis")
        return
    
    # Compute correlation matrix
    corr_matrix = data.corr()
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, vmin=-.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")
    
    plt.title("Feature Correlation Matrix", fontsize=18, weight="bold")
    plt.tight_layout()
    
    output_path = f"{output_dir}/feature_correlation_matrix.png"
    plt.savefig(output_path)
    logger.info(f"Saved feature correlation matrix to {output_path}")
    plt.close()

def plot_model_agreement_network(pairwise_df, output_dir="../results/figures"):
    """
    Plot a network diagram showing agreement between models.
    Requires networkx and matplotlib.
    
    Parameters:
    -----------
    pairwise_df : pandas.DataFrame
        DataFrame with columns 'model1', 'model2', 'agreement_score'
    output_dir : str
        Directory to save the output figures
    """
    try:
        import networkx as nx
        
        ensure_directory(output_dir)
        
        if pairwise_df is None or pairwise_df.empty:
            logger.warning("No pairwise data available for network diagram")
            return
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (models)
        models = set(pairwise_df["model1"].unique()) | set(pairwise_df["model2"].unique())
        for model in models:
            G.add_node(model)
        
        # Add edges with agreement scores as weights
        for _, row in pairwise_df.iterrows():
            # Only add edges for pairs with agreement score
            if "agreement_score" in row and not pd.isna(row["agreement_score"]):
                G.add_edge(row["model1"], row["model2"], 
                          weight=row["agreement_score"],
                          width=row["agreement_score"] * 5)  # Scale for visualization
        
        # Get position layout
        pos = nx.spring_layout(G, seed=42)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="lightblue", alpha=0.8)
        
        # Draw edges with width based on agreement score
        edges = G.edges(data=True)
        weights = [data["weight"] for _, _, data in edges]
        widths = [data["width"] for _, _, data in edges]
        
        # Create a colormap for the edges
        edge_cmap = plt.cm.YlGnBu
        
        # Draw the edges
        nx.draw_networkx_edges(G, pos, width=widths, edge_color=weights, 
                              edge_cmap=edge_cmap, edge_vmin=0.5, edge_vmax=1.0)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
        
        # Add a colorbar for reference
        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(0.5, 1.0))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label("Agreement Score")
        
        plt.title("Model Agreement Network", fontsize=18, weight="bold")
        plt.axis("off")
        plt.tight_layout()
        
        output_path = f"{output_dir}/model_agreement_network.png"
        plt.savefig(output_path)
        logger.info(f"Saved model agreement network to {output_path}")
        plt.close()
        
    except ImportError:
        logger.warning("networkx not available - skipping model agreement network plot")

if __name__ == "__main__":
    generate_all_visualizations()