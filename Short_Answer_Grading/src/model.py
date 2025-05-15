"""
Model loading and similarity calculation module
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityModel:
    """
    Similarity calculator based on pre-trained BERT embedding models
    """
    
    def __init__(self, model_name: str = 'paraphrase-MiniLM-L6-v2'):
        """
        Initialize the model
        
        Args:
            model_name: Pre-trained model name
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded successfully: {model_name}, Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Convert text to embedding vectors
        
        Args:
            text: Input text or list of texts
        
        Returns:
            Embedding vectors
        """
        # Ensure input is in list form
        if isinstance(text, str):
            text = [text]
        
        # Generate embeddings
        embeddings = self.model.encode(text, convert_to_numpy=True)
        return embeddings
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Cosine similarity score
        """
        # Encode texts
        embedding1 = self.encode_text(text1)
        embedding2 = self.encode_text(text2)
        
        # Calculate cosine similarity
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
            
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return similarity
    
    def batch_compute_similarity(self, reference_answers: List[str], provided_answers: List[str]) -> List[float]:
        """
        Batch calculate similarities between reference answers and student answers
        
        Args:
            reference_answers: List of reference answers
            provided_answers: List of student answers
        
        Returns:
            List of similarity scores
        """
        if len(reference_answers) != len(provided_answers):
            raise ValueError("The number of reference answers and student answers must be the same")
        
        # Encode all texts
        ref_embeddings = self.encode_text(reference_answers)
        provided_embeddings = self.encode_text(provided_answers)
        
        # Calculate similarity between each pair of answers
        similarities = []
        for i in range(len(reference_answers)):
            ref_emb = ref_embeddings[i].reshape(1, -1)
            prov_emb = provided_embeddings[i].reshape(1, -1)
            similarity = cosine_similarity(ref_emb, prov_emb)[0][0]
            similarities.append(similarity)
        
        return similarities

if __name__ == "__main__":
    # Test model loading and similarity calculation
    model = SimilarityModel()
    
    # Test simple similarity calculation
    reference = "Water boils at 100 degrees."
    student1 = "Water boils at 100 degrees Celsius."
    student2 = "Water turns into gas when it reaches 100 degrees Celsius."
    student3 = "I don't know at what temperature water boils."
    
    print("\nSimilarity Calculation Test:")
    print(f"Reference answer: {reference}")
    print(f"Student answer 1: {student1}, Similarity: {model.compute_similarity(reference, student1):.4f}")
    print(f"Student answer 2: {student2}, Similarity: {model.compute_similarity(reference, student2):.4f}")
    print(f"Student answer 3: {student3}, Similarity: {model.compute_similarity(reference, student3):.4f}")
    
    # Test batch calculation
    refs = [reference] * 3
    students = [student1, student2, student3]
    
    print("\nBatch Similarity Calculation Test:")
    similarities = model.batch_compute_similarity(refs, students)
    for i, sim in enumerate(similarities):
        print(f"Student answer {i+1} similarity: {sim:.4f}") 