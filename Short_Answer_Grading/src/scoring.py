"""
Scoring system implementation module
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional

from src.model import SimilarityModel

class ScoringSystem:
    """
    Threshold-based short answer scoring system
    """
    
    def __init__(self, 
                 similarity_model: SimilarityModel,
                 threshold_high: float = 0.85,
                 threshold_low: float = 0.60):
        """
        Initialize scoring system
        
        Args:
            similarity_model: Similarity model
            threshold_high: Similarity threshold for completely correct answers
            threshold_low: Similarity threshold for partially correct answers
        """
        self.model = similarity_model
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
    
    def score_answer(self, reference_answer: str, provided_answer: str) -> Tuple[int, float]:
        """
        Score a single answer
        
        Args:
            reference_answer: Reference answer
            provided_answer: Student provided answer
        
        Returns:
            Tuple of score and similarity
        """
        # Calculate similarity
        similarity = self.model.compute_similarity(reference_answer, provided_answer)
        
        # Assign score based on similarity thresholds
        if similarity >= self.threshold_high:
            score = 2  # Completely correct
        elif similarity >= self.threshold_low:
            score = 1  # Partially correct
        else:
            score = 0  # Incorrect
            
        return score, similarity
    
    def batch_score_answers(self, 
                          reference_answers: List[str], 
                          provided_answers: List[str]) -> Tuple[List[int], List[float]]:
        """
        Batch score answers
        
        Args:
            reference_answers: List of reference answers
            provided_answers: List of student answers
        
        Returns:
            Tuple of score list and similarity list
        """
        # Batch calculate similarities
        similarities = self.model.batch_compute_similarity(reference_answers, provided_answers)
        
        # Assign scores based on similarity thresholds
        scores = []
        for similarity in similarities:
            if similarity >= self.threshold_high:
                scores.append(2)  # Completely correct
            elif similarity >= self.threshold_low:
                scores.append(1)  # Partially correct
            else:
                scores.append(0)  # Incorrect
        
        return scores, similarities
    
    def score_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score an entire dataset
        
        Args:
            df: Dataframe containing reference answers and student answers
        
        Returns:
            Dataframe with scores and similarities
        """
        # Ensure necessary columns exist
        if 'reference_answer' not in df.columns or 'provided_answer' not in df.columns:
            raise ValueError("Dataframe must contain 'reference_answer' and 'provided_answer' columns")
        
        # Create a copy of the result dataframe
        result_df = df.copy()
        
        # Prepare inputs for batch scoring
        reference_answers = result_df['reference_answer'].tolist()
        provided_answers = result_df['provided_answer'].tolist()
        
        # Batch scoring
        predicted_scores, similarities = self.batch_score_answers(reference_answers, provided_answers)
        
        # Add result columns
        result_df['predicted_score'] = predicted_scores
        result_df['similarity'] = similarities
        
        return result_df

if __name__ == "__main__":
    # Test scoring system
    from src.model import SimilarityModel
    
    # Load model
    similarity_model = SimilarityModel()
    
    # Create scoring system
    scoring_system = ScoringSystem(similarity_model)
    
    # Test scoring a single answer
    reference = "Water boils at 100 degrees."
    student1 = "Water boils at 100 degrees Celsius."
    student2 = "Water turns into gas when it reaches 100 degrees Celsius."
    student3 = "I don't know at what temperature water boils."
    
    score1, sim1 = scoring_system.score_answer(reference, student1)
    score2, sim2 = scoring_system.score_answer(reference, student2)
    score3, sim3 = scoring_system.score_answer(reference, student3)
    
    print("\nScoring System Test:")
    print(f"Reference answer: {reference}")
    print(f"Student answer 1: {student1}, Similarity: {sim1:.4f}, Score: {score1}")
    print(f"Student answer 2: {student2}, Similarity: {sim2:.4f}, Score: {score2}")
    print(f"Student answer 3: {student3}, Similarity: {sim3:.4f}, Score: {score3}")
    
    # Test batch scoring
    references = [reference] * 3
    students = [student1, student2, student3]
    
    scores, sims = scoring_system.batch_score_answers(references, students)
    
    print("\nBatch Scoring Test:")
    for i in range(3):
        print(f"Student answer {i+1}, Similarity: {sims[i]:.4f}, Score: {scores[i]}") 