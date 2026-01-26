"""
Threshold Analysis for Chronos-2 Anomaly Detection Results
Evaluates which threshold configuration performs best across all datasets
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_path: str):
    """Load results from JSON file"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_thresholds(results_dict: dict):
    """
    Analyze and aggregate metrics for each threshold configuration
    
    Args:
        results_dict: Dictionary with results from runChronos2Eval.py
        
    Returns:
        DataFrame with aggregated metrics per threshold
    """
    
    threshold_metrics = {}
    
    # Iterate through each dataset
    for filename, result in results_dict.items():
        if 'metrics' not in result:
            continue
            
        # Iterate through each threshold configuration
        for metric_item in result['metrics']:
            threshold_key = str(metric_item['thresholds'])
            
            if threshold_key not in threshold_metrics:
                threshold_metrics[threshold_key] = {
                    'accuracies': [],
                    'precisions': [],
                    'recalls': [],
                    'f1_scores': [],
                    'auc_pr': [],
                    'auc_roc': [],
                    'vus_pr': [],
                    'vus_roc': [],
                    'standard_f1': [],
                    'pa_f1': [],
                    'event_based_f1': [],
                    'r_based_f1': [],
                    'affiliation_f': [],
                    'datasets': []
                }
            
            threshold_metrics[threshold_key]['accuracies'].append(metric_item.get('accuracy', 0))
            threshold_metrics[threshold_key]['precisions'].append(metric_item.get('precision', 0))
            threshold_metrics[threshold_key]['recalls'].append(metric_item.get('recall', 0))
            threshold_metrics[threshold_key]['f1_scores'].append(metric_item.get('f1_score', 0))
            threshold_metrics[threshold_key]['auc_pr'].append(metric_item.get('AUC-PR', 0))
            threshold_metrics[threshold_key]['auc_roc'].append(metric_item.get('AUC-ROC', 0))
            threshold_metrics[threshold_key]['vus_pr'].append(metric_item.get('VUS-PR', 0))
            threshold_metrics[threshold_key]['vus_roc'].append(metric_item.get('VUS-ROC', 0))
            threshold_metrics[threshold_key]['standard_f1'].append(metric_item.get('Standard-F1', 0))
            threshold_metrics[threshold_key]['pa_f1'].append(metric_item.get('PA-F1', 0))
            threshold_metrics[threshold_key]['event_based_f1'].append(metric_item.get('Event-based-F1', 0))
            threshold_metrics[threshold_key]['r_based_f1'].append(metric_item.get('R-based-F1', 0))
            threshold_metrics[threshold_key]['affiliation_f'].append(metric_item.get('Affiliation-F', 0))
            threshold_metrics[threshold_key]['datasets'].append(filename)
    
    # Calculate aggregate statistics
    aggregated = []
    for threshold, metrics in threshold_metrics.items():
        aggregated.append({
            'threshold': threshold,
            'avg_accuracy': np.mean(metrics['accuracies']),
            'std_accuracy': np.std(metrics['accuracies']),
            'avg_precision': np.mean(metrics['precisions']),
            'std_precision': np.std(metrics['precisions']),
            'avg_recall': np.mean(metrics['recalls']),
            'std_recall': np.std(metrics['recalls']),
            'avg_f1': np.mean(metrics['f1_scores']),
            'std_f1': np.std(metrics['f1_scores']),
            'avg_auc_pr': np.mean(metrics['auc_pr']),
            'std_auc_pr': np.std(metrics['auc_pr']),
            'avg_auc_roc': np.mean(metrics['auc_roc']),
            'std_auc_roc': np.std(metrics['auc_roc']),
            'avg_vus_pr': np.mean(metrics['vus_pr']),
            'std_vus_pr': np.std(metrics['vus_pr']),
            'avg_vus_roc': np.mean(metrics['vus_roc']),
            'std_vus_roc': np.std(metrics['vus_roc']),
            'avg_standard_f1': np.mean(metrics['standard_f1']),
            'std_standard_f1': np.std(metrics['standard_f1']),
            'avg_pa_f1': np.mean(metrics['pa_f1']),
            'std_pa_f1': np.std(metrics['pa_f1']),
            'avg_event_based_f1': np.mean(metrics['event_based_f1']),
            'std_event_based_f1': np.std(metrics['event_based_f1']),
            'avg_r_based_f1': np.mean(metrics['r_based_f1']),
            'std_r_based_f1': np.std(metrics['r_based_f1']),
            'avg_affiliation_f': np.mean(metrics['affiliation_f']),
            'std_affiliation_f': np.std(metrics['affiliation_f']),
            'num_datasets': len(metrics['datasets'])
        })
    
    return pd.DataFrame(aggregated)


def print_analysis(df: pd.DataFrame):
    """Print analysis results in a readable format"""
    
    print("\n" + "="*120)
    print("THRESHOLD ANALYSIS - RESULTS SUMMARY")
    print("="*120 + "\n")
    
    # Sort by AUC-ROC (primary metric)
    df_sorted = df.sort_values('avg_auc_roc', ascending=False)
    
    print("Top 5 Thresholds by Average AUC-ROC Score:\n")
    for idx, row in df_sorted.head(5).iterrows():
        print(f"Threshold: {row['threshold']}")
        print(f"  Standard Metrics:")
        print(f"    Accuracy:  {row['avg_accuracy']:.4f} (±{row['std_accuracy']:.4f})")
        print(f"    Precision: {row['avg_precision']:.4f} (±{row['std_precision']:.4f})")
        print(f"    Recall:    {row['avg_recall']:.4f} (±{row['std_recall']:.4f})")
        print(f"    F1 Score:  {row['avg_f1']:.4f} (±{row['std_f1']:.4f})")
        print(f"  TSB-AD Metrics:")
        print(f"    AUC-PR:        {row['avg_auc_pr']:.4f} (±{row['std_auc_pr']:.4f})")
        print(f"    AUC-ROC:       {row['avg_auc_roc']:.4f} (±{row['std_auc_roc']:.4f})")
        print(f"    VUS-PR:        {row['avg_vus_pr']:.4f} (±{row['std_vus_pr']:.4f})")
        print(f"    VUS-ROC:       {row['avg_vus_roc']:.4f} (±{row['std_vus_roc']:.4f})")
        print(f"    Standard-F1:   {row['avg_standard_f1']:.4f} (±{row['std_standard_f1']:.4f})")
        print(f"    PA-F1:         {row['avg_pa_f1']:.4f} (±{row['std_pa_f1']:.4f})")
        print(f"    Event-based-F1: {row['avg_event_based_f1']:.4f} (±{row['std_event_based_f1']:.4f})")
        print(f"    R-based-F1:    {row['avg_r_based_f1']:.4f} (±{row['std_r_based_f1']:.4f})")
        print(f"    Affiliation-F: {row['avg_affiliation_f']:.4f} (±{row['std_affiliation_f']:.4f})")
        print(f"  Datasets evaluated: {row['num_datasets']}\n")
    
    print("\n" + "="*120)
    print("BEST THRESHOLDS BY METRIC")
    print("="*120 + "\n")
    
    best_accuracy = df.loc[df['avg_accuracy'].idxmax()]
    best_precision = df.loc[df['avg_precision'].idxmax()]
    best_recall = df.loc[df['avg_recall'].idxmax()]
    best_f1 = df.loc[df['avg_f1'].idxmax()]
    best_auc_pr = df.loc[df['avg_auc_pr'].idxmax()]
    best_auc_roc = df.loc[df['avg_auc_roc'].idxmax()]
    best_vus_pr = df.loc[df['avg_vus_pr'].idxmax()]
    best_vus_roc = df.loc[df['avg_vus_roc'].idxmax()]
    best_pa_f1 = df.loc[df['avg_pa_f1'].idxmax()]
    best_affiliation_f = df.loc[df['avg_affiliation_f'].idxmax()]
    
    print("Standard Metrics:")
    print(f"  Best Accuracy:   {best_accuracy['threshold']} ({best_accuracy['avg_accuracy']:.4f})")
    print(f"  Best Precision:  {best_precision['threshold']} ({best_precision['avg_precision']:.4f})")
    print(f"  Best Recall:     {best_recall['threshold']} ({best_recall['avg_recall']:.4f})")
    print(f"  Best F1 Score:   {best_f1['threshold']} ({best_f1['avg_f1']:.4f})")
    print("\nTSB-AD Metrics:")
    print(f"  Best AUC-PR:        {best_auc_pr['threshold']} ({best_auc_pr['avg_auc_pr']:.4f})")
    print(f"  Best AUC-ROC:       {best_auc_roc['threshold']} ({best_auc_roc['avg_auc_roc']:.4f})")
    print(f"  Best VUS-PR:        {best_vus_pr['threshold']} ({best_vus_pr['avg_vus_pr']:.4f})")
    print(f"  Best VUS-ROC:       {best_vus_roc['threshold']} ({best_vus_roc['avg_vus_roc']:.4f})")
    print(f"  Best PA-F1:         {best_pa_f1['threshold']} ({best_pa_f1['avg_pa_f1']:.4f})")
    print(f"  Best Affiliation-F: {best_affiliation_f['threshold']} ({best_affiliation_f['avg_affiliation_f']:.4f})")
    print("\n")


def save_analysis(df: pd.DataFrame, output_path: str):
    """Save analysis to CSV file"""
    df_sorted = df.sort_values('avg_f1', ascending=False)
    df_sorted.to_csv(output_path, index=False)
    print(f"Analysis saved to: {output_path}")


def plot_threshold_comparison(df: pd.DataFrame, output_dir: str = "./Nunzio/results/univariate/plots/"):
    """Create visualization plots comparing thresholds"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    
    # Sort by AUC-ROC
    df_sorted = df.sort_values('avg_auc_roc', ascending=False).head(10)
    
    # Plot 1: Standard Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Standard Metrics Comparison across Top 10 Thresholds', fontsize=16, fontweight='bold')
    
    metrics = ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        x_pos = np.arange(len(df_sorted))
        ax.bar(x_pos, df_sorted[metric], alpha=0.7, color='steelblue')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(t)[:15] for t in df_sorted['threshold']], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Plot 2: TSB-AD Metrics Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('TSB-AD Metrics Comparison across Top 10 Thresholds', fontsize=16, fontweight='bold')
    
    tsb_metrics = ['avg_auc_pr', 'avg_auc_roc', 'avg_vus_pr', 'avg_vus_roc', 'avg_pa_f1', 'avg_affiliation_f']
    tsb_titles = ['AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC', 'PA-F1', 'Affiliation-F']
    
    axes = axes.flatten()
    for idx, (metric, title) in enumerate(zip(tsb_metrics, tsb_titles)):
        ax = axes[idx]
        x_pos = np.arange(len(df_sorted))
        ax.bar(x_pos, df_sorted[metric], alpha=0.7, color='coral')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(t)[:15] for t in df_sorted['threshold']], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Plot 3: All F1 Variants
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(df_sorted))
    width = 0.15
    
    ax.bar(x_pos - 1.5*width, df_sorted['avg_f1'], width, label='F1', alpha=0.8)
    ax.bar(x_pos - 0.5*width, df_sorted['avg_standard_f1'], width, label='Standard-F1', alpha=0.8)
    ax.bar(x_pos + 0.5*width, df_sorted['avg_pa_f1'], width, label='PA-F1', alpha=0.8)
    ax.bar(x_pos + 1.5*width, df_sorted['avg_event_based_f1'], width, label='Event-based-F1', alpha=0.8)
    
    ax.set_xlabel('Threshold', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('All F1 Score Variants Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(t)[:15] for t in df_sorted['threshold']], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Plot 4: Heatmap of all metrics
    fig, ax = plt.subplots(figsize=(14, 8))
    
    metrics_to_plot = [
        'avg_accuracy', 'avg_precision', 'avg_recall', 'avg_f1',
        'avg_auc_pr', 'avg_auc_roc', 'avg_vus_pr', 'avg_vus_roc',
        'avg_standard_f1', 'avg_pa_f1', 'avg_event_based_f1', 'avg_affiliation_f'
    ]
    
    heatmap_data = df_sorted[metrics_to_plot].copy()
    heatmap_data.index = [str(t)[:20] for t in df_sorted['threshold']]
    heatmap_data.columns = ['Accuracy', 'Precision', 'Recall', 'F1', 
                             'AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC',
                             'Std-F1', 'PA-F1', 'Event-F1', 'Aff-F']
    
    sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                cbar_kws={'label': 'Score'}, linewidths=0.5)
    ax.set_title('All Metrics Heatmap for Top 10 Thresholds', fontsize=14, fontweight='bold')
    ax.set_xlabel('Threshold', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Plot 5: Error bars for top metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Metrics with Standard Deviation (Top 10 Thresholds)', fontsize=16, fontweight='bold')
    
    error_metrics = [
        ('avg_auc_roc', 'std_auc_roc', 'AUC-ROC'),
        ('avg_pa_f1', 'std_pa_f1', 'PA-F1'),
        ('avg_affiliation_f', 'std_affiliation_f', 'Affiliation-F'),
        ('avg_f1', 'std_f1', 'F1 Score')
    ]
    
    for idx, (metric, std_metric, title) in enumerate(error_metrics):
        ax = axes[idx // 2, idx % 2]
        x_pos = np.arange(len(df_sorted))
        ax.errorbar(x_pos, df_sorted[metric], yerr=df_sorted[std_metric], 
                   fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(t)[:15] for t in df_sorted['threshold']], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(f'{title} with Error Bars')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    print(f"\nAll plots saved to: {output_dir}")


def load_all_json_results(results_dirs):
    """Load and merge results from all JSON files in specified directories"""
    all_results = {}
    
    for results_dir in results_dirs:
        path = Path(results_dir)
        if not path.exists():
            print(f"Warning: Directory not found: {results_dir}")
            continue
        
        # Find all JSON files
        json_files = list(path.glob('*.json'))
        print(f"Found {len(json_files)} JSON files in {results_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Merge results
                    if isinstance(data, dict):
                        all_results.update(data)
                    print(f"  Loaded: {json_file.name} ({len(data) if isinstance(data, dict) else 'N/A'} datasets)")
            except Exception as e:
                print(f"  Error loading {json_file.name}: {e}")
    
    return all_results


def main():
    """Main execution"""
    
    # Define all results directories
    results_dirs = [
        "./Nunzio/results/univariate",
        "./Sara"
    ]
    
    # Load all JSON results
    print("Loading all JSON results from:")
    all_results = load_all_json_results(results_dirs)
    
    if not all_results:
        print("Error: No results loaded from any directory")
        return
    
    print(f"\nTotal datasets loaded: {len(all_results)}\n")
    
    # Analyze thresholds
    analysis_df = analyze_thresholds(all_results)
    
    # Print results
    print_analysis(analysis_df)
    
    # Save analysis
    output_csv = "./Nunzio/results/univariate/threshold_analysis_all.csv"
    save_analysis(analysis_df, output_csv)
    
    # Generate plots (display but don't save)
    plot_threshold_comparison(analysis_df)


if __name__ == "__main__":
    main()
