"""
Data loading and preprocessing module
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Union, Optional

def load_saf_dataset(data_dir: str = "../Dataset/saf_communication_networks_english/data") -> Dict[str, pd.DataFrame]:
    """
    Load SAF dataset
    
    Args:
        data_dir: Data directory path
    
    Returns:
        Dictionary containing train, validation, and test sets
    """
    try:
        # Use pandas to directly read parquet files
        train_file = os.path.join(data_dir, "train-00000-of-00001-33368fd062630adb.parquet")
        val_file = os.path.join(data_dir, "validation-00000-of-00001-ac83a9f5b20af433.parquet")
        test_unseen_answers_file = os.path.join(data_dir, "test_unseen_answers-00000-of-00001-934b6dd7b400658f.parquet")
        test_unseen_questions_file = os.path.join(data_dir, "test_unseen_questions-00000-of-00001-c4d530c0df70ed3d.parquet")
        
        # Check if files exist
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training set file does not exist: {train_file}")
        if not os.path.exists(val_file):
            raise FileNotFoundError(f"Validation set file does not exist: {val_file}")
        if not os.path.exists(test_unseen_answers_file):
            raise FileNotFoundError(f"Test set (unseen answers) file does not exist: {test_unseen_answers_file}")
        if not os.path.exists(test_unseen_questions_file):
            raise FileNotFoundError(f"Test set (unseen questions) file does not exist: {test_unseen_questions_file}")
            
        # Read parquet files
        train_df = pd.read_parquet(train_file)
        val_df = pd.read_parquet(val_file)
        test_unseen_answers_df = pd.read_parquet(test_unseen_answers_file)
        test_unseen_questions_df = pd.read_parquet(test_unseen_questions_file)
        
        # Merge test datasets
        test_df = pd.concat([test_unseen_answers_df, test_unseen_questions_df], ignore_index=True)
        
        return {
            "train": train_df,
            "validation": val_df,
            "test": test_df
        }
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}

def preprocess_text(text: str) -> str:
    """
    Preprocess text data
    
    Args:
        text: Input text
    
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Basic text cleaning
    text = text.strip()
    
    # Remove extra spaces
    text = " ".join(text.split())
    
    return text

def map_scores_to_scale(df: pd.DataFrame, score_col: str = 'score', 
                      target_col: str = 'mapped_score') -> pd.DataFrame:
    """
    Map scores to a 0-2 scale
    
    Args:
        df: Input dataframe
        score_col: Column containing original scores
        target_col: Column to store mapped scores
    
    Returns:
        Dataframe with mapped scores
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Map scores
    # Assuming the original scores are in [0, 1, 2, 3, 4, 5]
    # Map to [0, 1, 2] where:
    # 0 = incorrect (original 0, 1)
    # 1 = partially correct (original 2, 3)
    # 2 = completely correct (original 4, 5)
    
    # Create the mapping function
    def map_score(score):
        if score <= 1:
            return 0  # Incorrect
        elif score <= 3:
            return 1  # Partially correct
        else:
            return 2  # Completely correct
    
    # Apply the mapping
    result_df[target_col] = result_df[score_col].apply(map_score)
    
    return result_df

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataset for model training and evaluation
    
    Args:
        df: Original dataframe
    
    Returns:
        Processed dataframe
    """
    # Ensure necessary columns exist
    required_cols = ['question', 'reference_answer', 'provided_answer', 'score']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Dataset is missing required column: {col}")
    
    # Copy dataframe to avoid modifying the original
    processed_df = df.copy()
    
    # Preprocess text
    for col in ['question', 'reference_answer', 'provided_answer']:
        processed_df[col] = processed_df[col].apply(preprocess_text)
    
    # Map scores to 0-2 scale
    processed_df = map_scores_to_scale(processed_df)
    
    # Filter out missing values
    processed_df = processed_df.dropna(subset=['reference_answer', 'provided_answer', 'mapped_score'])
    
    return processed_df

def save_processed_dataset(datasets: Dict[str, pd.DataFrame], output_dir: str = 'data/processed') -> None:
    """
    Save processed dataset to disk
    
    Args:
        datasets: Dictionary of dataframes to save
        output_dir: Output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, df in datasets.items():
        output_file = os.path.join(output_dir, f"{split_name}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {split_name} set to {output_file}, {len(df)} rows")

def load_processed_dataset(split: str, data_dir: str = "../data/processed", sample_size: Union[int, float] = None, random_state: int = 42) -> pd.DataFrame:
    """
    Load processed dataset
    
    Args:
        split: Dataset split name ('train', 'validation', 'test')
        data_dir: Data directory
        sample_size: Sample size; if an integer, represents the number of samples;
                    if a float < 1, represents the proportion; None means use all data
        random_state: Random seed for sampling
    
    Returns:
        Loaded dataframe
    """
    file_path = os.path.join(data_dir, f"{split}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed dataset file not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Sampling
    if sample_size is not None:
        if isinstance(sample_size, int) and sample_size > 0:
            # If sample_size is a positive integer, take a fixed number of samples
            sample_size = min(sample_size, len(df))
            df = df.sample(n=sample_size, random_state=random_state)
        elif isinstance(sample_size, float) and 0 < sample_size < 1:
            # If sample_size is a float between 0 and 1, take that proportion
            df = df.sample(frac=sample_size, random_state=random_state)
        else:
            print(f"Invalid sample_size: {sample_size}. Using full dataset.")
    
    return df

if __name__ == "__main__":
    # Test data loading
    datasets = load_saf_dataset()
    if datasets:
        print("Successfully loaded dataset")
        for split_name, df in datasets.items():
            print(f"{split_name}: {len(df)} rows, columns: {df.columns.tolist()}")
            
            # Sample data
            print(f"\nSample from {split_name}:")
            sample = df.head(1)
            for col in ['question', 'reference_answer', 'provided_answer', 'score']:
                if col in sample:
                    print(f"{col}: {sample[col].values[0]}")
            
            # Process dataset
            processed_df = prepare_dataset(df)
            print(f"\nProcessed {split_name}: {len(processed_df)} rows")
            print(f"Score distribution: {processed_df['mapped_score'].value_counts().to_dict()}")
    else:
        print("Failed to load dataset") 