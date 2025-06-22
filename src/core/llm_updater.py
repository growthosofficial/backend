import numpy as np
from typing import Optional, Dict
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import get_embedding
from database.supabase_manager import supabase_manager

def SSC(text: str, threshold: float, database_folder: str = None) -> Optional[Dict]:
    """
    Calculate semantic similarity and return the MOST similar knowledge category
    
    Since categories are MECE, returns only the single most relevant category
    
    Args:
        text: Input text to compare
        threshold: Minimum similarity threshold
        database_folder: Ignored (kept for compatibility)
        
    Returns:
        Most similar knowledge item with similarity score, or None if no match above threshold
    """
    text_embedding = get_embedding(text)
    emb1 = np.array(text_embedding).reshape(1, -1)

    best_match = None
    highest_similarity = 0.0

    # Load all knowledge from Supabase
    existing_knowledge = supabase_manager.load_all_knowledge()
    
    for knowledge_item in existing_knowledge:
        # Use pre-computed embedding if available
        if 'embedding' in knowledge_item and knowledge_item['embedding']:
            emb2 = np.array(knowledge_item['embedding']).reshape(1, -1)
        else:
            # Fallback: compute embedding from content
            content = knowledge_item.get('content', '')
            if not content:
                continue
            emb2 = np.array(get_embedding(content)).reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        # Keep track of the highest similarity
        if similarity > highest_similarity and similarity > threshold:
            highest_similarity = similarity
            best_match = knowledge_item.copy()
            best_match['similarity_score'] = float(similarity)
    
    return best_match