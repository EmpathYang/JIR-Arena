import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from glob import glob

def analyze_evaluation_metrics(directory_path, metrics_to_analyze=None, key_metrics=["duration", "recall", "precision", "weighted_ndcg", "weighted_time_match"], decimal_places=6):
    """
    Analyze evaluation metrics from JSON files in the specified directory.
    
    Args:
        directory_path (str): Path to directory containing JSON evaluation files
        metrics_to_analyze (list): List of metrics to analyze. If None, use default metrics
    """
    # Default metrics to analyze
    default_metrics = {
        "duration": ["duration"],
        "recall": ["recall", "recall"],
        "precision": ["precision", "precision"],
        "weighted_ndcg": ["relevance", "weighted_ndcg"],
        "weighted_start_match": ["timeliness", "weighted_start_match"],
        "weighted_end_match": ["timeliness", "weighted_end_match"],
        "weighted_time_match": ["timeliness", "weighted_time_match"],
        "matched_weighted_ndcg": ["relevance", "matched_weighted_ndcg"],
        "matched_weighted_start_match": ["timeliness", "matched_weighted_start_match"],
        "matched_weighted_end_match": ["timeliness", "matched_weighted_end_match"],
        "matched_weighted_time_match": ["timeliness", "matched_weighted_time_match"],
    }
    
    # Use provided metrics or default ones
    metrics_config = metrics_to_analyze if metrics_to_analyze else default_metrics
    
    # Get all JSON files in the directory
    json_files = glob(os.path.join(directory_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return
    
    # Initialize a dictionary to store metrics
    metrics = {metric_name: [] for metric_name in metrics_config.keys()}
    
    # Process each JSON file
    filenames = []
    for json_file in json_files:
        filename = os.path.basename(json_file).replace('.json', '')
        filenames.append(filename)
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract metrics
        for metric_name, path in metrics_config.items():
            if len(path) == 1:
                # Metric is at the top level
                metrics[metric_name].append(data[path[0]])
            elif len(path) == 2:
                # Metric is nested one level deep
                metrics[metric_name].append(data[path[0]][path[1]])
            elif len(path) == 3:
                # Metric is nested two levels deep
                metrics[metric_name].append(data[path[0]][path[1]][path[2]])
            else:
                print(f"Warning: Invalid path depth for metric {metric_name}")
                metrics[metric_name].append(None)
    
    print(metrics)

    # Create DataFrame for easier analysis
    df = pd.DataFrame(metrics, index=filenames)    

    weights = df['duration']
    numeric_cols = df.select_dtypes(include='number').columns.drop('duration')

    weighted_mean = df[numeric_cols].multiply(weights, axis=0).sum() / weights.sum()

    # Weighted std (per column)
    def weighted_std(values, weights):
        avg = np.average(values, weights=weights)
        return np.sqrt(np.average((values - avg) ** 2, weights=weights))

    weighted_std_series = df[numeric_cols].apply(lambda col: weighted_std(col, weights))

    # Combine into a summary DataFrame (each row is a statistic)
    weighted_summary = pd.DataFrame(
        [weighted_mean, weighted_std_series],
        index=['weighted_mean', 'weighted_std']
    ).round(decimal_places)
    
    original_summary = df[numeric_cols].describe().round(decimal_places)

    full_summary = pd.concat([original_summary, weighted_summary])
    full_summary.to_csv(os.path.join(directory_path, "summary_statistics.csv"))

    print("Saved summary_with_weighted_stats.csv with original and weighted stats:")
    print(full_summary)

    # Create visualizations
    create_visualizations(df, directory_path, key_metrics)
    
    return df

def create_visualizations(df, output_dir, key_metrics):
    """Create visualizations for the metrics"""
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))
    
    # Box plot for all metrics
    plt.subplot(2, 2, 1)
    df.boxplot()
    plt.title('Distribution of Performance Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Bar chart for mean values
    plt.subplot(2, 2, 2)
    df.mean().plot(kind='bar', yerr=df.std())
    plt.title('Mean Value of Each Metric (with Standard Deviation)')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Heatmap for correlations
    plt.subplot(2, 2, 3)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Between Metrics')
    plt.tight_layout()
    
    # Scatter plot matrix for selected metrics (if we have 4 or fewer metrics)
    if len(df.columns) <= 4:
        plt.subplot(2, 2, 4)
        pd.plotting.scatter_matrix(df, figsize=(10, 10), alpha=0.8)
        plt.tight_layout()
    else:
        # If too many metrics, just use the first 4
        plt.subplot(2, 2, 4)
        pd.plotting.scatter_matrix(df[key_metrics], figsize=(10, 10), alpha=0.8)
        plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'metrics_analysis.png'), dpi=300, bbox_inches='tight')
    
    # Individual metric distributions
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df.columns):
        plt.subplot(3, 3, i+1 if i < 9 else 9)  # Limit to 9 subplots
        if i < 9:  # Only show up to 9 metrics to avoid overcrowding
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
        if i == 8 and len(df.columns) > 9:
            plt.text(0.5, 0.5, f"{len(df.columns) - 9} more metrics not shown", 
                    horizontalalignment='center', verticalalignment='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distributions.png'), dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_output_dir', type=str, required=True)

    args = parser.parse_args()

    for model_id in ["claude-3-7", "DeepSeek-V3-0324", "gemini", "gpt-4o"]:
        analyze_evaluation_metrics(f"{args.evaluation_output_dir}_{model_id}")