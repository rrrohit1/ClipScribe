import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from src.config import WINDOW_SIZE, OVERLAP

def create_chunks_with_sliding_window(sentences: List[str]) -> List[List[str]]:
    """
    Create overlapping chunks from a list of sentences using sliding window approach.
    
    Args:
        sentences (List[str]): List of input sentences to be chunked
        
    Returns:
        List[List[str]]: List of chunks where each chunk is a list of sentences
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
    similarities = []
    for i in range(len(chunk_embeddings) - 1):
        sim = cosine_similarity([chunk_embeddings[i]], [chunk_embeddings[i + 1]])[0][0]
        similarities.append(sim)
    
    return chunks, similarities

if __name__ == "__main__":
    # Test the function with sample sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A lazy dog sleeps in the sun.",
        "The sun is very bright today.",
        "Today is a beautiful day.",
        "The weather is perfect for a walk.",
        "Walking is good for health.",
        "Health is wealth they say.",
        "They say morning walks are the best."
    ]
    
    chunks, similarities = create_chunks_with_sliding_window(test_sentences)
    
    print("Generated Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print("\n".join(chunk))
        if i < len(similarities):
            print(f"\nSimilarity with next chunk: {similarities[i]:.4f}")
        print("-" * 50)