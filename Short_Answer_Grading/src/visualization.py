"""
Visualization tools module
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional

def set_plotting_style():
    """
    Set plotting style
    """
    # Set seaborn style
    sns.set(style="whitegrid")
    # Set English font
    plt.rcParams['font.sans-serif'] = ['Arial']  # For displaying English labels correctly
    plt.rcParams['axes.unicode_minus'] = False  # For displaying minus sign correctly

def plot_similarity_distribution(df: pd.DataFrame, similarity_col: str = 'similarity', 
                                score_col: str = 'mapped_score', save_path: Optional[str] = None):
    """
    Plot similarity score distribution
    
    Args:
        df: Dataframe containing similarity scores
        similarity_col: Similarity column name
        score_col: True score column name
        save_path: If provided, the figure will be saved to this path
    """
    set_plotting_style()
    
    plt.figure(figsize=(10, 6))
    
    # Group by true score
    for score in sorted(df[score_col].unique()):
        subset = df[df[score_col] == score]
        sns.kdeplot(subset[similarity_col], label=f'Score {score}', fill=True, alpha=0.3)
    
    # Draw threshold lines
    plt.axvline(x=0.60, color='r', linestyle='--', label='Threshold 0.60')
    plt.axvline(x=0.85, color='g', linestyle='--', label='Threshold 0.85')
    
    plt.title('Similarity Distribution by Score Category')
    plt.xlabel('Similarity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_similarity_boxplot(df: pd.DataFrame, similarity_col: str = 'similarity', 
                          score_col: str = 'mapped_score', save_path: Optional[str] = None):
    """
    Plot similarity boxplot
    
    Args:
        df: Dataframe containing similarity scores
        similarity_col: Similarity column name
        score_col: True score column name
        save_path: If provided, the figure will be saved to this path
    """
    set_plotting_style()
    
    plt.figure(figsize=(10, 6))
    
    # Create boxplot
    sns.boxplot(x=score_col, y=similarity_col, data=df)
    
    # Draw threshold lines
    plt.axhline(y=0.60, color='r', linestyle='--', label='Threshold 0.60')
    plt.axhline(y=0.85, color='g', linestyle='--', label='Threshold 0.85')
    
    plt.title('Similarity Distribution by Score Category')
    plt.xlabel('Actual Score')
    plt.ylabel('Similarity')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix_heatmap(cm: np.ndarray, class_names: List[str] = None, 
                               save_path: Optional[str] = None):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: If provided, the figure will be saved to this path
    """
    set_plotting_style()
    
    if class_names is None:
        class_names = ['Incorrect(0)', 'Partially Correct(1)', 'Completely Correct(2)']
    
    plt.figure(figsize=(8, 6))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotated heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_score_comparison(y_true: List[int], y_pred: List[int], save_path: Optional[str] = None):
    """
    Plot true score vs predicted score comparison
    
    Args:
        y_true: List of true scores
        y_pred: List of predicted scores
        save_path: If provided, the figure will be saved to this path
    """
    set_plotting_style()
    
    # Create dataframe
    df = pd.DataFrame({'True Score': y_true, 'Predicted Score': y_pred})
    
    plt.figure(figsize=(10, 6))
    
    # Use jitter to avoid overlapping points
    sns.scatterplot(x='True Score', y='Predicted Score', data=df, alpha=0.5, s=100)
    
    # Draw ideal line
    x = np.array([min(y_true), max(y_true)])
    plt.plot(x, x, 'r--', label='Ideal Prediction')
    
    plt.title('True Score vs. Predicted Score')
    plt.xlabel('True Score')
    plt.ylabel('Predicted Score')
    plt.legend()
    plt.grid(True)
    
    # Set axis ticks
    plt.xticks(range(min(y_true), max(y_true)+1))
    plt.yticks(range(min(y_pred), max(y_pred)+1))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_results(df: pd.DataFrame, true_score_col: str = 'mapped_score', 
                    pred_score_col: str = 'predicted_score', similarity_col: str = 'similarity',
                    output_dir: Optional[str] = None):
    """
    Visualize results
    
    Args:
        df: Dataframe containing evaluation results
        true_score_col: True score column name
        pred_score_col: Predicted score column name
        similarity_col: Similarity column name
        output_dir: Output directory, if provided, figures will be saved to this directory
    """
    # Get true and predicted labels
    y_true = df[true_score_col].astype(int).tolist()
    y_pred = df[pred_score_col].astype(int).tolist()
    
    # Plot similarity distribution
    plot_similarity_distribution(df, similarity_col, true_score_col, 
                               save_path=f"{output_dir}/similarity_distribution.png" if output_dir else None)
    
    # Plot similarity boxplot
    plot_similarity_boxplot(df, similarity_col, true_score_col,
                          save_path=f"{output_dir}/similarity_boxplot.png" if output_dir else None)
    
    # Plot confusion matrix heatmap
    cm = np.zeros((3, 3))  # Assume there are 3 categories
    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]] += 1
    
    plot_confusion_matrix_heatmap(cm.astype(int),
                               save_path=f"{output_dir}/confusion_matrix.png" if output_dir else None)
    
    # Plot score comparison
    plot_score_comparison(y_true, y_pred,
                        save_path=f"{output_dir}/score_comparison.png" if output_dir else None)

if __name__ == "__main__":
    # Test visualization functions
    # Create example data
    np.random.seed(42)
    
    # Generate random similarity scores
    n_samples = 100
    similarities = np.concatenate([
        np.random.normal(0.4, 0.15, size=30),  # Score 0 similarities
        np.random.normal(0.7, 0.1, size=40),   # Score 1 similarities
        np.random.normal(0.9, 0.05, size=30)   # Score 2 similarities
    ])
    
    # Clip values to [0, 1] range
    similarities = np.clip(similarities, 0, 1)
    
    # Create true scores
    true_scores = np.concatenate([
        np.zeros(30),
        np.ones(40),
        np.ones(30) * 2
    ]).astype(int)
    
    # Generate predicted scores based on thresholds
    pred_scores = np.array([
        2 if sim >= 0.85 else (1 if sim >= 0.6 else 0)
        for sim in similarities
    ])
    
    # Create dataframe
    test_df = pd.DataFrame({
        'mapped_score': true_scores,
        'predicted_score': pred_scores,
        'similarity': similarities
    })
    
    # Visualize
    visualize_results(test_df) 