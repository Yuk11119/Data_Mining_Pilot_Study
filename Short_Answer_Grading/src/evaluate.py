"""
Evaluation metrics and functions module
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, fbeta_score

def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
    
    Returns:
        Dictionary containing various evaluation metrics
    """
    # Ensure inputs are valid
    if len(y_true) != len(y_pred):
        raise ValueError("True labels and predicted labels must have the same length")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate macro-averaged F1 score
    f1 = f1_score(y_true, y_pred, average='macro')
    
    # Calculate macro-averaged F2 score (beta=2, emphasizes recall more)
    f2 = fbeta_score(y_true, y_pred, beta=2, average='macro')
    
    # Return metrics dictionary
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'f2_score': f2
    }

def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print evaluation metrics
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print("\n========== Evaluation Metrics ==========")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Macro-averaged F1 score: {metrics['f1_score']:.4f}")
    print(f"Macro-averaged F2 score: {metrics['f2_score']:.4f}")
    print("===============================\n")

def generate_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """
    Generate confusion matrix
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
    
    Returns:
        Confusion matrix
    """
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    return cm

def print_confusion_matrix(cm: np.ndarray, class_names: List[str] = None) -> None:
    """
    Print confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    # Default class names
    default_class_names = ['Incorrect(0)', 'Partially Correct(1)', 'Completely Correct(2)']
    
    # Adjust class names based on confusion matrix dimensions
    if class_names is None:
        # Only use class names that match matrix dimensions
        class_names = default_class_names[:cm.shape[0]]
    
    print("\n========== Confusion Matrix ==========")
    print("Predicted →")
    print("↓ Actual | " + " | ".join(class_names))
    
    for i, row in enumerate(cm):
        print(f"{class_names[i]}  | " + " | ".join([str(val) for val in row]))
    print("===============================\n")

def print_classification_report(y_true: List[int], y_pred: List[int], class_names: List[str] = None) -> None:
    """
    Print classification report
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        class_names: List of class names
    """
    # Get classes that actually exist in the data
    unique_classes = sorted(set(y_true) | set(y_pred))
    
    # Default class names
    default_class_names = ['Incorrect(0)', 'Partially Correct(1)', 'Completely Correct(2)']
    
    # If class names are not provided, create based on actual classes
    if class_names is None:
        # Only use class names corresponding to classes that exist
        class_names = [default_class_names[i] for i in unique_classes if i < len(default_class_names)]
    
    try:
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        
        print("\n========== Classification Report ==========")
        print(report)
        print("===============================\n")
    except Exception as e:
        print(f"\nError generating classification report: {e}")
        print("Some classes may be missing in the data.")
        print("Skipping detailed classification report.")
        print("===============================\n")

def evaluate_model(df: pd.DataFrame, true_score_col: str = 'mapped_score', 
                  pred_score_col: str = 'predicted_score') -> Dict[str, float]:
    """
    Evaluate model performance
    
    Args:
        df: Dataframe containing true and predicted scores
        true_score_col: Column name for true scores
        pred_score_col: Column name for predicted scores
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Ensure necessary columns exist
    if true_score_col not in df.columns or pred_score_col not in df.columns:
        raise ValueError(f"Dataframe must contain '{true_score_col}' and '{pred_score_col}' columns")
    
    # Get true and predicted labels
    y_true = df[true_score_col].astype(int).tolist()
    y_pred = df[pred_score_col].astype(int).tolist()
    
    # Calculate evaluation metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print evaluation metrics
    print_metrics(metrics)
    
    # Generate and print confusion matrix
    cm = generate_confusion_matrix(y_true, y_pred)
    print_confusion_matrix(cm)
    
    # Print classification report
    print_classification_report(y_true, y_pred)
    
    return metrics

def analyze_errors(df: pd.DataFrame, true_score_col: str = 'mapped_score', 
                 pred_score_col: str = 'predicted_score') -> pd.DataFrame:
    """
    Analyze prediction errors
    
    Args:
        df: Dataframe containing true and predicted scores
        true_score_col: Column name for true scores
        pred_score_col: Column name for predicted scores
    
    Returns:
        Subset of instances with prediction errors
    """
    # Find instances with prediction errors
    error_mask = df[true_score_col] != df[pred_score_col]
    error_df = df[error_mask].copy()
    
    # Add error type information
    error_df['error_type'] = error_df.apply(
        lambda row: f"True:{row[true_score_col]}, Predicted:{row[pred_score_col]}", 
        axis=1
    )
    
    print(f"\nFound {len(error_df)} prediction errors (out of {len(df)} instances).")
    print(f"Error rate: {len(error_df) / len(df) * 100:.2f}%")
    
    # Count error types
    error_counts = error_df['error_type'].value_counts()
    print("\nError type distribution:")
    for error_type, count in error_counts.items():
        print(f"{error_type}: {count} instances ({count/len(error_df)*100:.1f}%)")
    
    return error_df

if __name__ == "__main__":
    # Test evaluation metrics calculation
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 0, 2, 1, 1, 2]
    
    # Calculate and print evaluation metrics
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics)
    
    # Generate and print confusion matrix
    cm = generate_confusion_matrix(y_true, y_pred)
    print_confusion_matrix(cm)
    
    # Print classification report
    print_classification_report(y_true, y_pred) 