"""
Data exploration script
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple

from src.data_utils import load_saf_dataset, prepare_dataset

def set_plotting_style():
    """Set plotting style"""
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial']  # For displaying English labels correctly
    plt.rcParams['axes.unicode_minus'] = False  # For displaying minus sign correctly

def explore_dataset_info(datasets: Dict[str, pd.DataFrame]):
    """
    Explore basic dataset information
    
    Args:
        datasets: Dictionary of dataframes
    """
    print("\n========== Dataset Information ==========")
    
    for split_name, df in datasets.items():
        print(f"\n{split_name} set:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {', '.join(df.columns)}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("\n  Missing values:")
            for col, count in missing_values.items():
                if count > 0:
                    print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print("\n  No missing values found.")
        
        # Display a sample
        print("\n  Sample entry:")
        sample = df.iloc[0]
        for col in ['question', 'reference_answer', 'provided_answer', 'score']:
            if col in sample:
                print(f"    {col}: {sample[col]}")

def explore_score_distribution(datasets: Dict[str, pd.DataFrame]):
    """
    Explore score distribution in datasets
    
    Args:
        datasets: Dictionary of dataframes
    """
    print("\n========== Score Distribution ==========")
    
    # Set plotting style
    set_plotting_style()
    
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 5*len(datasets)))
    if len(datasets) == 1:
        axes = [axes]
    
    for i, (split_name, df) in enumerate(datasets.items()):
        if 'score' in df.columns:
            # Calculate score distribution
            score_counts = df['score'].value_counts().sort_index()
            
            # Print score distribution
            print(f"\n{split_name} set score distribution:")
            for score, count in score_counts.items():
                print(f"  Score {score}: {count} ({count/len(df)*100:.1f}%)")
            
            # Plot score distribution
            ax = axes[i]
            score_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(f"{split_name} Set Score Distribution")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.grid(axis='y')
            
            # Add count labels on bars
            for j, v in enumerate(score_counts):
                ax.text(j, v+5, str(v), ha='center')
    
    plt.tight_layout()
    plt.show()

def explore_text_length(datasets: Dict[str, pd.DataFrame], processed_datasets: Dict[str, pd.DataFrame]):
    """
    Explore text length
    
    Args:
        datasets: Original dataset dictionary
        processed_datasets: Processed dataset dictionary
    """
    print("\n========== Text Length Analysis ==========")
    
    # Analyze text length in columns
    columns_to_analyze = ['question', 'reference_answer', 'provided_answer']
    
    for split_name in datasets.keys():
        print(f"\n{split_name} set text length statistics:")
        
        original_df = datasets[split_name]
        processed_df = processed_datasets[split_name]
        
        for col in columns_to_analyze:
            # Calculate character count
            original_lens = original_df[col].str.len()
            processed_lens = processed_df[col].str.len()
            
            # Calculate word count
            original_word_counts = original_df[col].str.split().str.len()
            processed_word_counts = processed_df[col].str.split().str.len()
            
            print(f"\n  {col}:")
            print(f"    Original - Avg characters: {original_lens.mean():.1f}, Avg words: {original_word_counts.mean():.1f}")
            print(f"    Processed - Avg characters: {processed_lens.mean():.1f}, Avg words: {processed_word_counts.mean():.1f}")
            print(f"    Shortest: {processed_lens.min()} characters, Longest: {processed_lens.max()} characters")

def explore_answer_similarity(df: pd.DataFrame, n_examples: int = 5):
    """
    Explore similarity between reference and provided answers
    
    Args:
        df: Dataframe containing reference and provided answers
        n_examples: Number of examples to show
    """
    print("\n========== Answer Similarity Examples ==========")
    
    # Group by score
    scores = sorted(df['score'].unique())
    
    for score in scores:
        score_df = df[df['score'] == score]
        samples = score_df.sample(min(n_examples, len(score_df)))
        
        print(f"\nExamples with score {score}:")
        
        for i, (_, row) in enumerate(samples.iterrows()):
            print(f"\nExample {i+1}:")
            print(f"  Question: {row['question']}")
            print(f"  Reference Answer: {row['reference_answer']}")
            print(f"  Student Answer: {row['provided_answer']}")

def explore_data():
    """Main data exploration function"""
    # Load dataset
    print("Loading dataset...")
    datasets = load_saf_dataset()
    
    if not datasets:
        print("Failed to load dataset.")
        return
    
    # Basic information exploration
    explore_dataset_info(datasets)
    
    # Score distribution exploration
    explore_score_distribution(datasets)
    
    # Process datasets
    print("\nProcessing dataset...")
    processed_datasets = {
        split_name: prepare_dataset(df) 
        for split_name, df in datasets.items()
    }
    
    # Text length analysis
    explore_text_length(datasets, processed_datasets)
    
    # Answer examples
    explore_answer_similarity(datasets['train'])
    
    print("\nData exploration complete!")

if __name__ == "__main__":
    explore_data() 