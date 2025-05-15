"""
Automated Short Answer Grading System Based on Pre-trained BERT Embeddings - Main Program
"""
import os
import argparse
import pandas as pd
from typing import Dict

from src.data_utils import load_saf_dataset, prepare_dataset, save_processed_dataset
from src.model import SimilarityModel
from src.scoring import ScoringSystem
from src.evaluate import evaluate_model, analyze_errors
from src.visualization import visualize_results
from src.model_evaluation import evaluate_on_dataset, threshold_tuning

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Short Answer Grading System')
    
    # Data processing parameters
    parser.add_argument('--data_dir', type=str, default='../Dataset/saf_communication_networks_english/data',
                       help='Original dataset directory')
    parser.add_argument('--processed_data_dir', type=str, default='data/processed',
                       help='Processed dataset directory')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Results directory')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='paraphrase-MiniLM-L6-v2',
                       help='Pretrained model name')
    parser.add_argument('--threshold_high', type=float, default=0.85,
                       help='Similarity threshold for completely correct answers')
    parser.add_argument('--threshold_low', type=float, default=0.60,
                       help='Similarity threshold for partially correct answers')
    
    # Sampling parameters
    parser.add_argument('--sample_size', type=str, default=None,
                       help='Sample size, can be an integer (e.g., "100") or a decimal proportion (e.g., "0.1")')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for sampling')
    
    # Feature selection
    parser.add_argument('--preprocess', action='store_true',
                       help='Whether to preprocess data')
    parser.add_argument('--evaluate', action='store_true',
                       help='Whether to evaluate the model')
    parser.add_argument('--tune_thresholds', action='store_true',
                       help='Whether to tune thresholds')
    parser.add_argument('--analyze_errors', action='store_true',
                       help='Whether to analyze errors')
    parser.add_argument('--visualize', action='store_true',
                       help='Whether to visualize results')
    
    # Evaluation parameters
    parser.add_argument('--split', type=str, default='test',
                       help='Data split to use for evaluation')
    
    return parser.parse_args()

def preprocess_data(args):
    """Preprocess data"""
    print("\n========== Data Preprocessing ==========")
    
    # Load original dataset
    datasets = load_saf_dataset(args.data_dir)
    
    if not datasets:
        print("Failed to load dataset. Cannot continue.")
        return False
    
    print("Dataset loaded successfully.")
    for split_name, df in datasets.items():
        print(f"{split_name} set: {len(df)} rows")
    
    # Process dataset
    print("\nProcessing dataset...")
    processed_datasets = {
        split_name: prepare_dataset(df) 
        for split_name, df in datasets.items()
    }
    
    # Save processed dataset
    save_processed_dataset(processed_datasets, args.processed_data_dir)
    
    return True

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Process sample_size parameter
    sample_size = None
    if args.sample_size is not None:
        try:
            # Try to convert to integer
            sample_size = int(args.sample_size)
        except ValueError:
            try:
                # Try to convert to float
                sample_size = float(args.sample_size)
            except ValueError:
                print(f"Warning: Invalid sample_size format: {args.sample_size}, will use the full dataset")
    
    # Create necessary directories
    os.makedirs(args.processed_data_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # If no features are specified, display help information
    if not (args.preprocess or args.evaluate or args.tune_thresholds or args.analyze_errors or args.visualize):
        print("No features specified. Use --help to see available options.")
        
        print("\nExample usage:")
        print("1. Preprocess data:")
        print("   python main.py --preprocess")
        print("2. Evaluate model:")
        print("   python main.py --evaluate")
        print("3. Tune thresholds:")
        print("   python main.py --tune_thresholds")
        print("4. Analyze errors:")
        print("   python main.py --analyze_errors")
        print("5. Visualize results:")
        print("   python main.py --visualize")
        print("6. Execute complete workflow:")
        print("   python main.py --preprocess --evaluate --visualize")
        return
    
    # Preprocess data
    if args.preprocess:
        success = preprocess_data(args)
        if not success:
            print("Data preprocessing failed. Cannot continue.")
            return
    
    # Evaluate model
    if args.evaluate:
        print("\n========== Model Evaluation ==========")
        result_df = evaluate_on_dataset(
            split=args.split,
            threshold_high=args.threshold_high,
            threshold_low=args.threshold_low,
            model_name=args.model_name,
            data_dir=args.processed_data_dir,
            results_dir=args.results_dir,
            save_visuals=args.visualize,
            sample_size=sample_size,
            random_state=args.random_state
        )
    
    # Tune thresholds
    if args.tune_thresholds:
        print("\n========== Threshold Tuning ==========")
        
        # Load result data
        results_file = os.path.join(args.results_dir, f"{args.split}_results.csv")
        if not os.path.exists(results_file):
            print(f"Results file does not exist: {results_file}")
            print("Please first evaluate the model with --evaluate to generate results")
            return
        
        try:
            result_df = pd.read_csv(results_file)
            
            # Tune thresholds
            best_high, best_low, _ = threshold_tuning(
                df=result_df,
                model_name=args.model_name
            )
            
            # Re-evaluate with best thresholds
            print("\nRe-evaluating model with best thresholds:")
            evaluate_on_dataset(
                split=args.split,
                threshold_high=best_high,
                threshold_low=best_low,
                model_name=args.model_name,
                data_dir=args.processed_data_dir,
                results_dir=args.results_dir,
                save_visuals=args.visualize,
                sample_size=sample_size,
                random_state=args.random_state
            )
        except Exception as e:
            print(f"Error during threshold tuning: {e}")
    
    # Analyze errors
    if args.analyze_errors:
        print("\n========== Error Analysis ==========")
        
        # Import error analysis module
        from src.error_analysis import display_error_examples, categorize_error_patterns, analyze_error_by_length, suggest_improvements
        
        # Load result data
        results_file = os.path.join(args.results_dir, f"{args.split}_results.csv")
        if not os.path.exists(results_file):
            print(f"Results file does not exist: {results_file}")
            print("Please first evaluate the model with --evaluate to generate results")
            return
        
        try:
            result_df = pd.read_csv(results_file)
            
            # Analyze errors
            error_df = analyze_errors(result_df)
            
            if len(error_df) == 0:
                print("No prediction errors found.")
                return
            
            # Display error examples
            display_error_examples(error_df)
            
            # Categorize error patterns
            error_patterns_df = categorize_error_patterns(error_df)
            
            # Analyze the relationship between errors and text length
            analyze_error_by_length(error_patterns_df, output_dir=args.results_dir)
            
            # Suggest improvements
            suggest_improvements(error_patterns_df)
        except Exception as e:
            print(f"Error during error analysis: {e}")
    
    # Visualize results
    if args.visualize and not args.evaluate:
        print("\n========== Result Visualization ==========")
        
        # Load result data
        results_file = os.path.join(args.results_dir, f"{args.split}_results.csv")
        if not os.path.exists(results_file):
            print(f"Results file does not exist: {results_file}")
            print("Please first evaluate the model with --evaluate to generate results")
            return
        
        try:
            result_df = pd.read_csv(results_file)
            
            # Visualize results
            visualize_results(result_df, output_dir=args.results_dir)
            print(f"Visualization results have been saved to {args.results_dir}")
        except Exception as e:
            print(f"Error during result visualization: {e}")

if __name__ == "__main__":
    main() 