import json
import logging
from typing import List, Optional, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from database import DatabaseManager
from models import KnowledgeItem

logger = logging.getLogger(__name__)


class SemanticSimilarityCore:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1).reshape(1, -1)
            vec2 = np.array(embedding2).reshape(1, -1)
            
            similarity = cosine_similarity(vec1, vec2)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0

    def SSC(self, input_embedding: List[float], threshold: float = 0.8) -> Optional[Dict]:
        """
        Semantic Similarity Core function.
        
        Args:
            input_embedding: Embedding vector for input text
            threshold: Minimum similarity threshold (default 0.8)
            
        Returns:
            Dictionary with most similar item and similarity score if above threshold
        """
        try:
            # Get all knowledge items
            items = self.db_manager.get_all_knowledge_items()
            
            if not items:
                logger.info("No existing knowledge items found")
                return None
            
            best_match = None
            best_score = 0.0
            
            # Compare input embedding with all stored embeddings
            for item in items:
                try:
                    # Parse stored embedding
                    stored_embedding = json.loads(item.embedding)
                    
                    if not stored_embedding:
                        continue
                    
                    # Calculate similarity
                    similarity = self.calculate_cosine_similarity(input_embedding, stored_embedding)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = item
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid embedding format for item {item.id}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing item {item.id}: {e}")
                    continue
            
            # Check if best match meets threshold
            if best_match and best_score >= threshold:
                logger.info(f"Found similar item (ID: {best_match.id}) with score: {best_score:.3f}")
                
                # Parse tags
                try:
                    tags = json.loads(best_match.tags)
                except json.JSONDecodeError:
                    tags = []
                
                return {
                    "id": best_match.id,
                    "category": best_match.category,
                    "content": best_match.content,
                    "tags": tags,
                    "similarity_score": best_score,
                    "created_at": best_match.created_at,
                    "last_updated": best_match.last_updated
                }
            else:
                logger.info(f"No similar items found above threshold {threshold}")
                return None
                
        except Exception as e:
            logger.error(f"SSC function failed: {e}")
            raise Exception(f"Semantic similarity search failed: {str(e)}")

    def find_all_similar(self, input_embedding: List[float], 
                        threshold: float = 0.7, limit: int = 5) -> List[Tuple[Dict, float]]:
        """
        Find all similar items above threshold.
        
        Args:
            input_embedding: Embedding vector for input text
            threshold: Minimum similarity threshold
            limit: Maximum number of results to return
            
        Returns:
            List of tuples (item_dict, similarity_score) sorted by similarity
        """
        try:
            items = self.db_manager.get_all_knowledge_items()
            
            if not items:
                return []
            
            similarities = []
            
            for item in items:
                try:
                    stored_embedding = json.loads(item.embedding)
                    if not stored_embedding:
                        continue
                    
                    similarity = self.calculate_cosine_similarity(input_embedding, stored_embedding)
                    
                    if similarity >= threshold:
                        try:
                            tags = json.loads(item.tags)
                        except json.JSONDecodeError:
                            tags = []
                        
                        item_dict = {
                            "id": item.id,
                            "category": item.category,
                            "content": item.content,
                            "tags": tags,
                            "created_at": item.created_at,
                            "last_updated": item.last_updated
                        }
                        
                        similarities.append((item_dict, similarity))
                        
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Error processing item {item.id}: {e}")
                    continue
            
            # Sort by similarity score (descending) and limit results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Failed to find similar items: {e}")
            raise Exception(f"Similar items search failed: {str(e)}")