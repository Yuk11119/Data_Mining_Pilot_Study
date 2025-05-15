"""
Error analysis script
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from src.data_utils import load_processed_dataset
from src.evaluate import analyze_errors

def display_error_examples(error_df: pd.DataFrame, n_examples: int = 5):
    """
    Display examples of prediction errors
    
    Args:
        error_df: Dataframe containing prediction errors
        n_examples: Number of examples to display
    """
    print(f"\n========== Error Prediction Examples ({min(n_examples, len(error_df))}) ==========")
    
    # Group by error type
    error_types = error_df['error_type'].unique()
    
    for error_type in error_types:
        type_df = error_df[error_df['error_type'] == error_type]
        samples = type_df.sample(min(n_examples, len(type_df)))
        
        print(f"\nError Type: {error_type} (Total: {len(type_df)})")
        
        for i, (_, row) in enumerate(samples.iterrows()):
            print(f"\nSample {i+1}:")
            print(f"Question: {row['question']}")
            print(f"Reference Answer: {row['reference_answer']}")
            print(f"Student Answer: {row['provided_answer']}")
            print(f"True Score: {row['mapped_score']}")
            print(f"Predicted Score: {row['predicted_score']}")
            print(f"Similarity: {row['similarity']:.4f}")

def categorize_error_patterns(error_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and categorize error patterns
    
    Args:
        error_df: Dataframe containing prediction errors
    
    Returns:
        Dataframe with error pattern categorization
    """
    # Copy dataframe
    df = error_df.copy()
    
    # Calculate length of reference and student answers
    df['ref_length'] = df['reference_answer'].str.len()
    df['provided_length'] = df['provided_answer'].str.len()
    df['length_ratio'] = df['provided_length'] / df['ref_length']
    
    # Define error pattern categories
    def determine_error_pattern(row):
        # If student answer is very short (relative to reference)
        if row['length_ratio'] < 0.3:
            return "Too Short Answer"
        
        # If student answer is very long (relative to reference)
        elif row['length_ratio'] > 3:
            return "Too Long Answer"
        
        # If similarity is close to threshold
        elif abs(row['similarity'] - 0.60) < 0.05 or abs(row['similarity'] - 0.85) < 0.05:
            return "Borderline Case"
        
        # If true score is high but similarity is low
        elif row['mapped_score'] > row['predicted_score'] and row['similarity'] < 0.6:
            return "Underestimated Correct Answer"
        
        # If true score is low but similarity is high
        elif row['mapped_score'] < row['predicted_score'] and row['similarity'] > 0.85:
            return "Overestimated Incorrect Answer"
        
        # Other errors
        else:
            return "Other Errors"
    
    # Apply error pattern categorization
    df['error_pattern'] = df.apply(determine_error_pattern, axis=1)
    
    # Count instances of each error pattern
    pattern_counts = df['error_pattern'].value_counts()
    
    print("\n========== Error Pattern Analysis ==========")
    print(f"Found {len(df)} errors in total.")
    
    for pattern, count in pattern_counts.items():
        print(f"{pattern}: {count} instances ({count/len(df)*100:.1f}%)")
    
    return df

def analyze_error_by_length(error_df: pd.DataFrame, output_dir: Optional[str] = None):
    """
    Analyze the relationship between errors and text length
    
    Args:
        error_df: Dataframe containing prediction errors
        output_dir: Output directory
    """
    # Set plotting style
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial']  # For displaying English labels correctly
    plt.rcParams['axes.unicode_minus'] = False  # For displaying minus sign correctly
    
    # Calculate text length
    error_df['ref_length'] = error_df['reference_answer'].str.len()
    error_df['provided_length'] = error_df['provided_answer'].str.len()
    
    # Plot scatter plot: reference answer length vs student answer length, colored by error type
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='ref_length', 
        y='provided_length', 
        hue='error_type', 
        data=error_df, 
        alpha=0.7
    )
    
    plt.title('Length Relationship Between Reference and Student Answers in Errors')
    plt.xlabel('Reference Answer Length (characters)')
    plt.ylabel('Student Answer Length (characters)')
    plt.legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'error_length_analysis.png'), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Plot box plot of length differences
    plt.figure(figsize=(10, 6))
    error_df['length_diff'] = error_df['provided_length'] - error_df['ref_length']
    sns.boxplot(x='error_type', y='length_diff', data=error_df)
    
    plt.title('Length Differences by Error Type')
    plt.xlabel('Error Type')
    plt.ylabel('Length Difference (Student Answer - Reference Answer)')
    plt.xticks(rotation=45)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'error_length_diff.png'), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()

def suggest_improvements(error_patterns_df: pd.DataFrame):
    """
    Suggest improvements based on error pattern analysis
    
    Args:
        error_patterns_df: Dataframe containing error patterns
    """
    # Count instances of each error pattern
    pattern_counts = error_patterns_df['error_pattern'].value_counts()
    total_errors = len(error_patterns_df)
    
    print("\n========== Improvement Suggestions ==========")
    
    # Provide suggestions for specific error patterns
    if "Too Short Answer" in pattern_counts and pattern_counts["Too Short Answer"] / total_errors > 0.1:
        print("\n1. For handling too short answers:")
        print("   - Consider using special scoring logic for short answers, such as keyword matching")
        print("   - Set a minimum threshold for answer length, below which a different scoring strategy is used")
        print("   - For extremely short responses like 'yes', 'no', consider integrating question context into evaluation")
    
    if "Too Long Answer" in pattern_counts and pattern_counts["Too Long Answer"] / total_errors > 0.1:
        print("\n2. For handling too long answers:")
        print("   - Consider implementing sliding window mechanism to break long answers into smaller segments for comparison")
        print("   - Extract key sentences or information from long answers for comparison")
        print("   - Explore sentence-level rather than document-level similarity calculation")
    
    if "Borderline Case" in pattern_counts and pattern_counts["Borderline Case"] / total_errors > 0.1:
        print("\n3. For threshold borderline cases:")
        print("   - Consider introducing more score levels, rather than just 0/1/2")
        print("   - Implement fuzzy thresholds, where cases near thresholds can receive scores between two levels")
        print("   - Fine-tune thresholds with more training data, or use different thresholds for different question types")
    
    if "Underestimated Correct Answer" in pattern_counts and pattern_counts["Underestimated Correct Answer"] / total_errors > 0.1:
        print("\n4. For underestimated correct answers:")
        print("   - Consider domain-specific fine-tuning of the model to better understand synonymous expressions")
        print("   - Add synonym expansion to automatically enrich reference answer expressions")
        print("   - Explore question-type specific scoring strategies")
    
    if "Overestimated Incorrect Answer" in pattern_counts and pattern_counts["Overestimated Incorrect Answer"] / total_errors > 0.1:
        print("\n5. For overestimated incorrect answers:")
        print("   - Add checking for correctness of key concepts, not just overall semantic similarity")
        print("   - Incorporate more features such as keyword presence, numeric accuracy, etc.")
        print("   - Consider adversarial training for the model to better distinguish semantically similar but factually incorrect answers")
    
    print("\nComprehensive improvement suggestions:")
    print("1. Hybrid model strategy: Combine BERT-based semantic similarity with traditional keyword matching methods")
    print("2. Multi-feature scoring: Include features like answer length, keyword presence, and numeric accuracy in scoring")
    print("3. Domain adaptation: Fine-tune the model on domain-specific data for better understanding of technical terminology")
    print("4. Question-aware scoring: Adjust scoring strategies based on question types or difficulty levels")
    print("5. Ensemble methods: Use multiple models and methods to cross-validate scoring results")

def main():
    """Main function"""
    # Load result data
    try:
        results_file = '../results/test_results.csv'
        if os.path.exists(results_file):
            result_df = pd.read_csv(results_file)
            print(f"Successfully loaded results file, {len(result_df)} instances")
        else:
            print(f"Results file does not exist: {results_file}")
            return
    except Exception as e:
        print(f"Failed to load results file: {e}")
        return
    
    # Analyze errors
    error_df = analyze_errors(result_df)
    
    if len(error_df) == 0:
        print("No prediction errors found.")
        return
    
    # Display error examples
    display_error_examples(error_df)
    
    # Categorize error patterns
    error_patterns_df = categorize_error_patterns(error_df)
    
    # Analyze the relationship between errors and length
    analyze_error_by_length(error_patterns_df, output_dir='../results')
    
    # Suggest improvements
    suggest_improvements(error_patterns_df)

if __name__ == "__main__":
    main() 