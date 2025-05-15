"""
Model evaluation script
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

from src.data_utils import load_processed_dataset
from src.model import SimilarityModel
from src.scoring import ScoringSystem
from src.evaluate import evaluate_model, analyze_errors
from src.visualization import visualize_results

def evaluate_on_dataset(split: str = 'test', 
                       threshold_high: float = 0.85, 
                       threshold_low: float = 0.60,
                       model_name: str = 'paraphrase-MiniLM-L6-v2',
                       data_dir: str = '../data/processed',
                       results_dir: str = '../results',
                       save_visuals: bool = True,
                       sample_size: Union[int, float] = None,
                       random_state: int = 42) -> pd.DataFrame:
    """
    Evaluate model on a specific dataset
    
    Args:
        split: Dataset split name
        threshold_high: Similarity threshold for completely correct answers
        threshold_low: Similarity threshold for partially correct answers
        model_name: Pretrained model name
        data_dir: Data directory
        results_dir: Results directory
        save_visuals: Whether to save visualization results
        sample_size: Sample size; if an integer, represents the number of samples;
                    if a float < 1, represents the proportion; None means use all data
        random_state: Random seed for sampling
    
    Returns:
        Dataframe containing evaluation results
    """
    print(f"\n========== Evaluating model on {split} set ==========")
    print(f"Using model: {model_name}")
    print(f"Threshold settings: High={threshold_high}, Low={threshold_low}")
    if sample_size is not None:
        sample_desc = f"{sample_size} samples" if isinstance(sample_size, int) else f"{sample_size*100:.1f}% samples"
        print(f"Using sampling: {sample_desc}")
    
    # Load dataset
    try:
        df = load_processed_dataset(split, data_dir, sample_size=sample_size, random_state=random_state)
        print(f"Successfully loaded {split} set, {len(df)} instances")
    except Exception as e:
        print(f"Failed to load {split} set: {e}")
        return pd.DataFrame()
    
    # Load model
    try:
        similarity_model = SimilarityModel(model_name=model_name)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return pd.DataFrame()
    
    # Create scoring system
    scoring_system = ScoringSystem(
        similarity_model=similarity_model,
        threshold_high=threshold_high,
        threshold_low=threshold_low
    )
    
    # Score the dataset
    print("\nStarting scoring...")
    result_df = scoring_system.score_dataset(df)
    print("Scoring completed")
    
    # Evaluate model
    print("\nEvaluating model performance:")
    metrics = evaluate_model(result_df)
    
    # Analyze errors
    print("\nError analysis:")
    error_df = analyze_errors(result_df)
    
    # Visualize results
    if save_visuals:
        os.makedirs(results_dir, exist_ok=True)
        print("\nGenerating visualization results...")
        visualize_results(result_df, output_dir=results_dir)
        print(f"Visualization results saved to {results_dir}")
    
    # Save results
    results_file = os.path.join(results_dir, f"{split}_results.csv")
    result_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    return result_df

def threshold_tuning(df: pd.DataFrame, 
                   high_thresholds: List[float] = None, 
                   low_thresholds: List[float] = None,
                   model_name: str = 'paraphrase-MiniLM-L6-v2') -> Tuple[float, float, Dict[str, float]]:
    """
    Tune optimal thresholds
    
    Args:
        df: Dataframe
        high_thresholds: List of high thresholds
        low_thresholds: List of low thresholds
        model_name: Pretrained model name
        
    Returns:
        Best high threshold, best low threshold, and corresponding metrics
    """
    print("\n========== Threshold Tuning ==========")
    
    if high_thresholds is None:
        high_thresholds = [0.80, 0.82, 0.85, 0.87, 0.90]
    
    if low_thresholds is None:
        low_thresholds = [0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
    
    # Load model
    try:
        similarity_model = SimilarityModel(model_name=model_name)
        
        # Generate embeddings for reference and student answers, avoid repeated computation
        reference_answers = df['reference_answer'].tolist()
        provided_answers = df['provided_answer'].tolist()
        similarities = similarity_model.batch_compute_similarity(reference_answers, provided_answers)
        df = df.copy()
        df['similarity'] = similarities
        
        # Store results
        results = []
        
        # Try different threshold combinations
        for high in high_thresholds:
            for low in low_thresholds:
                if low >= high:
                    continue  # Skip invalid threshold combinations
                
                # Apply thresholds
                df['predicted_score'] = df['similarity'].apply(
                    lambda x: 2 if x >= high else (1 if x >= low else 0)
                )
                
                # Calculate evaluation metrics
                y_true = df['mapped_score'].astype(int).tolist()
                y_pred = df['predicted_score'].astype(int).tolist()
                
                from sklearn.metrics import accuracy_score, f1_score, fbeta_score
                accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                f2 = fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0)
                
                results.append({
                    'high_threshold': high,
                    'low_threshold': low,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'f2_score': f2
                })
        
        # If no valid results, return default values
        if not results:
            print("No valid threshold combinations found. Using default values.")
            return 0.85, 0.60, {'accuracy': 0.0, 'f1_score': 0.0, 'f2_score': 0.0}
            
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Find the threshold combination with the highest F2 score
        best_result = results_df.loc[results_df['f2_score'].idxmax()]
        best_high = best_result['high_threshold']
        best_low = best_result['low_threshold']
        best_metrics = {
            'accuracy': best_result['accuracy'],
            'f1_score': best_result['f1_score'],
            'f2_score': best_result['f2_score']
        }
        
        print("\nBest threshold combination:")
        print(f"High threshold = {best_high}")
        print(f"Low threshold = {best_low}")
        print(f"Accuracy = {best_metrics['accuracy']:.4f}")
        print(f"F1 score = {best_metrics['f1_score']:.4f}")
        print(f"F2 score = {best_metrics['f2_score']:.4f}")
        
        # Output all results
        print("\nAll threshold combination results:")
        print(results_df.sort_values('f2_score', ascending=False).head(10).to_string(index=False))
        
        return best_high, best_low, best_metrics
        
    except Exception as e:
        print(f"Error during threshold tuning: {e}")
        # Return default thresholds
        return 0.85, 0.60, {'accuracy': 0.0, 'f1_score': 0.0, 'f2_score': 0.0}

def main():
    """主函数"""
    # 确保结果目录存在
    os.makedirs('../results', exist_ok=True)
    
    # 评估模型在测试集上的性能
    result_df = evaluate_on_dataset(split='test')
    
    # 阈值调整
    print("\n是否进行阈值调整？(y/n)")
    choice = input()
    if choice.lower() == 'y':
        best_high, best_low, _ = threshold_tuning(result_df)
        
        # 使用最佳阈值重新评估
        print("\n使用最佳阈值重新评估:")
        evaluate_on_dataset(
            split='test',
            threshold_high=best_high,
            threshold_low=best_low
        )

if __name__ == "__main__":
    main() 