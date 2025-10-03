import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from src.config import (
    WINDOW_SIZE, 
    OVERLAP, 
    MIN_CHUNK_DISTANCE, 
    MAX_CHUNK_DISTANCE
)

def find_local_minima(similarity_scores: List[float]) -> List[int]:
    """
    Find local minima in similarity scores while respecting minimum and maximum distance constraints.
    
    Args:
        similarity_scores (List[float]): List of similarity scores between consecutive chunks
        
    Returns:
        List[int]: Indices of local minima that satisfy the distance constraints
    """
    if len(similarity_scores) < 3:  # Need at least 3 points to find local minima
        return []
    
    minima_indices = []
    
    # Find all local minima
    for i in range(1, len(similarity_scores) - 1):
        if (similarity_scores[i] < similarity_scores[i - 1] and 
            similarity_scores[i] < similarity_scores[i + 1]):
            minima_indices.append(i)
    
    # If no minima found, return empty list
    if not minima_indices:
        return []
    
    # Filter minima based on distance constraints
    filtered_minima = [minima_indices[0]]
    
    for idx in minima_indices[1:]:
        # Check if current minimum is far enough from the last accepted minimum
        distance = idx - filtered_minima[-1]
        if MIN_CHUNK_DISTANCE <= distance <= MAX_CHUNK_DISTANCE:
            filtered_minima.append(idx)
    
    return filtered_minima

def create_chunks_with_sliding_window(sentences: List[str]) -> Tuple[List[List[str]], List[float]]:
    """
    Create overlapping chunks from a list of sentences using sliding window approach.
    
    Args:
        sentences (List[str]): List of input sentences to be chunked
        
    Returns:
        Tuple[List[List[str]], List[float]]: Tuple containing:
            - List of chunks where each chunk is a list of sentences
            - List of similarity scores between consecutive chunks
    """
    if not sentences:
        return []
    
    if len(sentences) <= WINDOW_SIZE:
        return [sentences]
    
    # Initialize sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    chunks = []
    chunk_embeddings = []
    
    # Create initial chunks with sliding window
    for i in range(0, len(sentences) - WINDOW_SIZE + 1, WINDOW_SIZE - OVERLAP):
        end_idx = min(i + WINDOW_SIZE, len(sentences))
        current_chunk = sentences[i:end_idx]
        
        # Create embedding for the current chunk by concatenating sentences
        chunk_text = " ".join(current_chunk)
        chunk_embedding = model.encode([chunk_text])[0]
        
        chunks.append(current_chunk)
        chunk_embeddings.append(chunk_embedding)
        
        if end_idx == len(sentences):
            break
    
    # Convert embeddings to numpy array for easier computation
    chunk_embeddings = np.array(chunk_embeddings)
    
    # Calculate similarity between consecutive chunks
    similarity_scores = []
    for i in range(len(chunk_embeddings) - 1):
        sim = cosine_similarity([chunk_embeddings[i]], [chunk_embeddings[i + 1]])[0][0]
        similarity_scores.append(sim)
    
    return chunks, similarity_scores

if __name__ == "__main__":
    # Test the functions with sample sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A lazy dog sleeps in the sun.",
        "The sun is very bright today.",
        "Today is a beautiful day.",
        "The weather is perfect for a walk.",
        "Walking is good for health.",
        "Health is wealth they say.",
        "They say morning walks are the best.",
        "Walking in nature is refreshing.",
        "Nature has its own beauty.",
        "Beauty lies in the eyes of the beholder.",
        "The beholder must have a pure heart."
    ]
    
    # Get chunks and similarity scores
    chunks, similarity_scores = create_chunks_with_sliding_window(test_sentences)
    
    print("Generated Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print("\n".join(chunk))
        if i < len(similarity_scores):
            print(f"\nSimilarity with next chunk: {similarity_scores[i]:.4f}")
        print("-" * 50)
    
    # Find and print local minima
    minima_indices = find_local_minima(similarity_scores)
    print("\nLocal Minima Indices (potential chunk boundaries):")
    print(minima_indices)
    
    if minima_indices:
        print("\nChunk Boundaries at Local Minima:")
        for idx in minima_indices:
            print(f"Break point after chunk {idx + 1} (similarity score: {similarity_scores[idx]:.4f})")