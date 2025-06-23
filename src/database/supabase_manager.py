import json
from typing import Any, List, Dict, Optional
from supabase import create_client, Client
from datetime import datetime, timedelta

from config.settings import settings

class SupabaseManager:
    """Read-only Supabase manager for data retrieval and analysis only"""
    
    def __init__(self):
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("Supabase URL and KEY must be configured")
        
        self.supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    # READ-ONLY METHODS FOR DATA RETRIEVAL
    
    def get_knowledge_by_id(self, item_id: int) -> Optional[Dict]:
        """Get a specific knowledge item by ID"""
        result = self.supabase.table('knowledge_items').select('*').eq('id', item_id).execute()
        return result.data[0] if result.data else None
    
    def get_knowledge_by_sub_category(self, sub_category: str) -> Optional[Dict]:
        """
        Get a specific knowledge item by sub-category
        
        Args:
            sub_category: Sub-category name to retrieve
            
        Returns:
            Knowledge item dict or None if not found
        """
        result = self.supabase.table('knowledge_items').select('*').eq('sub_category', sub_category).execute()
        return result.data[0] if result.data else None
    
    def get_knowledge_by_main_category(self, main_category: str) -> List[Dict]:
        """
        Get all knowledge items for a specific main category
        
        Args:
            main_category: Main category name to retrieve
            
        Returns:
            List of knowledge items in that main category
        """
        result = self.supabase.table('knowledge_items').select('*').eq('main_category', main_category).execute()
        return result.data
    
    def load_all_knowledge(self) -> List[Dict]:
        """
        Load all knowledge items from Supabase for similarity analysis
        
        Returns:
            List of all knowledge items with main_category and sub_category
        """
        result = self.supabase.table('knowledge_items').select('*').order('created_at', desc=True).execute()
        
        print(f"📚 Loaded {len(result.data)} knowledge items from Supabase for analysis")
        return result.data

    def load_knowledge_by_ids(self, ids: List[int]) -> List[Dict]:
        """
        Load knowledge items by IDs
        
        Args:
            ids: List of knowledge item IDs
        """
        result = self.supabase.table('knowledge_items').select('*').in_('id', ids).execute()
        return result.data
    
    def search_knowledge_by_content(self, search_term: str, limit: int = 10) -> List[Dict]:
        """
        Search knowledge items by content (text search)
        
        Args:
            search_term: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching knowledge items
        """
        result = self.supabase.table('knowledge_items')\
            .select('*')\
            .ilike('content', f'%{search_term}%')\
            .limit(limit)\
            .execute()
        
        return result.data
    
    def search_knowledge_by_tags(self, tags: List[str]) -> List[Dict]:
        """
        Search knowledge items by tags
        
        Args:
            tags: List of tags to search for
            
        Returns:
            List of matching knowledge items
        """
        results = []
        for tag in tags:
            result = self.supabase.table('knowledge_items')\
                .select('*')\
                .contains('tags', [tag])\
                .execute()
            results.extend(result.data)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for item in results:
            if item['id'] not in seen:
                seen.add(item['id'])
                unique_results.append(item)
        
        return unique_results
    
    # ANALYTICS AND STATISTICS (READ-ONLY)
    
    def get_database_stats(self) -> Dict:
        """
        Get comprehensive statistics about the knowledge database
        
        Returns:
            Dictionary with database statistics
        """
        # Get total count
        count_result = self.supabase.table('knowledge_items').select('id', count='exact').execute()
        total_items = count_result.count
        
        # Get all items for analysis
        all_items = self.load_all_knowledge()
        
        # Collect statistics
        all_tags = []
        main_categories = []
        sub_categories = []
        main_category_counts = {}
        strength_scores = []
        sources = []
        
        for item in all_items:
            # Tags
            tags = item.get('tags', [])
            if isinstance(tags, list):
                all_tags.extend(tags)
            
            # Categories
            main_cat = item.get('main_category', 'General Studies')
            sub_cat = item.get('sub_category', 'unknown')
            
            main_categories.append(main_cat)
            sub_categories.append(sub_cat)
            
            # Count main categories
            main_category_counts[main_cat] = main_category_counts.get(main_cat, 0) + 1
            
            # Strength scores for SRS
            if item.get('strength_score') is not None:
                strength_scores.append(float(item['strength_score']))
            
            # Sources
            source = item.get('source', 'text')
            sources.append(source)
        
        # Count unique items
        unique_main_categories = list(set(main_categories))
        unique_sub_categories = list(set(sub_categories))
        unique_tags = list(set(all_tags))
        unique_sources = list(set(sources))
        
        tag_counts = {tag: all_tags.count(tag) for tag in unique_tags}
        source_counts = {source: sources.count(source) for source in unique_sources}
        
        stats = {
            'total_knowledge_items': total_items,
            'unique_main_categories': len(unique_main_categories),
            'unique_sub_categories': len(unique_sub_categories),
            'unique_tags': len(unique_tags),
            'unique_sources': len(unique_sources),
            'main_category_distribution': main_category_counts,
            'source_distribution': source_counts,
            'main_categories': unique_main_categories,
            'sub_categories': unique_sub_categories,
            'most_common_tags': sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'database_type': 'Supabase (Read-Only)',
            # SRS statistics
            'avg_strength_score': sum(strength_scores) / len(strength_scores) if strength_scores else 0,
            'items_with_strength_score': len(strength_scores),
            'strong_items': len([s for s in strength_scores if s >= 0.8]),
            'weak_items': len([s for s in strength_scores if s < 0.5]),
            'total_reviews_completed': sum(item.get('review_count', 0) for item in all_items)
        }
        
        return stats
    
    def get_categories_grouped(self) -> Dict:
        """
        Get all categories grouped by main category (for frontend display)
        
        Returns:
            Dictionary with main categories as keys and sub-categories as values
        """
        all_items = self.load_all_knowledge()
        grouped = {}
        
        for item in all_items:
            main_cat = item.get('main_category', 'General Studies')
            sub_cat = item.get('sub_category', 'unknown')
            
            if main_cat not in grouped:
                grouped[main_cat] = {
                    'main_category': main_cat,
                    'sub_categories': [],
                    'total_items': 0,
                    'last_updated': None
                }
            
            # Add sub-category info
            sub_cat_info = {
                'id': item.get('id'),
                'sub_category': sub_cat,
                'content_preview': item.get('content', '')[:100] + '...' if len(item.get('content', '')) > 100 else item.get('content', ''),
                'tags': item.get('tags', []),
                'source': item.get('source', 'text'),
                'strength_score': item.get('strength_score'),
                'next_review_due': item.get('next_review_due'),
                'created_at': item.get('created_at'),
                'last_updated': item.get('last_updated')
            }
            
            grouped[main_cat]['sub_categories'].append(sub_cat_info)
            grouped[main_cat]['total_items'] += 1
            
            # Update last_updated
            item_updated = item.get('last_updated')
            if item_updated and (not grouped[main_cat]['last_updated'] or item_updated > grouped[main_cat]['last_updated']):
                grouped[main_cat]['last_updated'] = item_updated
        
        return grouped
    
    def get_items_due_for_review(self, limit: int = 50) -> List[Dict]:
        """
        Get knowledge items that are due for review (for analytics only)
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of items due for review, ordered by due date
        """
        current_time = datetime.utcnow().isoformat()
        
        result = self.supabase.table('knowledge_items')\
            .select('*')\
            .lte('next_review_due', current_time)\
            .order('next_review_due', desc=False)\
            .limit(limit)\
            .execute()
        
        return result.data
    
    def get_knowledge_strength_distribution(self) -> Dict:
        """
        Get distribution of knowledge strength scores for analytics
        
        Returns:
            Dictionary with strength score ranges and counts
        """
        all_items = self.load_all_knowledge()
        
        distribution = {
            'very_weak': 0,    # 0.0 - 0.2
            'weak': 0,         # 0.2 - 0.4
            'medium': 0,       # 0.4 - 0.6
            'strong': 0,       # 0.6 - 0.8
            'very_strong': 0,  # 0.8 - 1.0
            'no_score': 0      # None/null
        }
        
        for item in all_items:
            score = item.get('strength_score')
            if score is None:
                distribution['no_score'] += 1
            elif score < 0.2:
                distribution['very_weak'] += 1
            elif score < 0.4:
                distribution['weak'] += 1
            elif score < 0.6:
                distribution['medium'] += 1
            elif score < 0.8:
                distribution['strong'] += 1
            else:
                distribution['very_strong'] += 1
        
        return distribution
    
    def get_category_strength_analysis(self) -> Dict:
        """
        Get strength analysis by academic category
        
        Returns:
            Dictionary with categories and their average strength scores
        """
        all_items = self.load_all_knowledge()
        category_scores = {}
        
        for item in all_items:
            main_cat = item.get('main_category', 'General Studies')
            score = item.get('strength_score')
            
            if main_cat not in category_scores:
                category_scores[main_cat] = []
            
            if score is not None:
                category_scores[main_cat].append(score)
        
        # Calculate averages
        category_analysis = {}
        for category, scores in category_scores.items():
            if scores:
                category_analysis[category] = {
                    'avg_strength': sum(scores) / len(scores),
                    'total_items': len(scores),
                    'strong_items': len([s for s in scores if s >= 0.8]),
                    'weak_items': len([s for s in scores if s < 0.5])
                }
            else:
                category_analysis[category] = {
                    'avg_strength': 0,
                    'total_items': 0,
                    'strong_items': 0,
                    'weak_items': 0
                }
        
        return category_analysis

    def create_evaluation(self, evaluation_data: Dict) -> Optional[Dict[str, Any]]:
        """Create a new evaluation record"""
        try:
            evaluation = {
                "knowledge_id": evaluation_data["knowledge_id"],
                "question_text": evaluation_data["question_text"],
                "answer_text": evaluation_data["answer_text"],
                "score": evaluation_data["score"],
                "feedback": evaluation_data["feedback"],
                "improvement_suggestions": evaluation_data["improvement_suggestions"],
                "correct_points": evaluation_data["correct_points"],
                "incorrect_points": evaluation_data["incorrect_points"],
            }
            # Insert evaluation with points as arrays
            result = self.supabase.table('evaluations').insert(evaluation).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error in create_evaluation: {e}")
            return None

    def get_all_evaluations(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Get all evaluations from the database with optional pagination
        
        Args:
            limit: Maximum number of evaluations to return (default: 100)
            offset: Number of evaluations to skip (default: 0)
            
        Returns:
            List of evaluation records with knowledge item details
        """
        try:
            # Get evaluations with knowledge item details using a join
            result = self.supabase.table('evaluations')\
                .select('''
                    *,
                    knowledge_items!inner(
                        id,
                        main_category,
                        sub_category,
                        content
                    )
                ''')\
                .order('created_at', desc=True)\
                .range(offset, offset + limit - 1)\
                .execute()
            
            return result.data
        except Exception as e:
            print(f"Error getting evaluations: {e}")
            return []

    def get_evaluations_by_knowledge_id(self, knowledge_id: int) -> List[Dict]:
        """
        Get all evaluations for a specific knowledge item
        
        Args:
            knowledge_id: ID of the knowledge item
            
        Returns:
            List of evaluations for that knowledge item
        """
        try:
            result = self.supabase.table('evaluations')\
                .select('*')\
                .eq('knowledge_id', knowledge_id)\
                .order('created_at', desc=True)\
                .execute()
            
            return result.data
        except Exception as e:
            print(f"Error getting evaluations for knowledge_id {knowledge_id}: {e}")
            return []

    def get_evaluations_by_category(self, main_category: str) -> List[Dict]:
        """
        Get all evaluations for a specific academic category
        
        Args:
            main_category: Main academic category
            
        Returns:
            List of evaluations for that category
        """
        try:
            result = self.supabase.table('evaluations')\
                .select('''
                    *,
                    knowledge_items!inner(
                        id,
                        main_category,
                        sub_category,
                        content
                    )
                ''')\
                .eq('knowledge_items.main_category', main_category)\
                .order('created_at', desc=True)\
                .execute()
            
            return result.data
        except Exception as e:
            print(f"Error getting evaluations for category {main_category}: {e}")
            return []

    def get_evaluation_statistics(self) -> Dict:
        """
        Get comprehensive statistics about evaluations
        
        Returns:
            Dictionary with evaluation statistics
        """
        try:
            # Get all evaluations
            all_evaluations = self.supabase.table('evaluations')\
                .select('''
                    *,
                    knowledge_items!inner(
                        main_category,
                        sub_category
                    )
                ''')\
                .execute()
            
            evaluations = all_evaluations.data
            
            if not evaluations:
                return {
                    'total_evaluations': 0,
                    'average_score': 0,
                    'score_distribution': {},
                    'category_performance': {},
                    'recent_evaluations': 0
                }
            
            # Calculate statistics
            scores = [eval_item['score'] for eval_item in evaluations]
            total_evaluations = len(evaluations)
            average_score = sum(scores) / total_evaluations if scores else 0
            
            # Score distribution
            score_distribution = {i: scores.count(i) for i in range(1, 6)}
            
            # Category performance
            category_scores = {}
            for eval_item in evaluations:
                category = eval_item['knowledge_items']['main_category']
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(eval_item['score'])
            
            category_performance = {}
            for category, cat_scores in category_scores.items():
                category_performance[category] = {
                    'average_score': sum(cat_scores) / len(cat_scores),
                    'total_evaluations': len(cat_scores),
                    'score_distribution': {i: cat_scores.count(i) for i in range(1, 6)}
                }
            
            # Recent evaluations (last 7 days)
            recent_date = (datetime.now() - timedelta(days=7)).isoformat()
            recent_evaluations = len([
                e for e in evaluations 
                if e.get('created_at', '') >= recent_date
            ])
            
            return {
                'total_evaluations': total_evaluations,
                'average_score': round(average_score, 2),
                'score_distribution': score_distribution,
                'category_performance': category_performance,
                'recent_evaluations': recent_evaluations,
                'highest_scoring_category': max(category_performance.items(), key=lambda x: x[1]['average_score'])[0] if category_performance else None,
                'lowest_scoring_category': min(category_performance.items(), key=lambda x: x[1]['average_score'])[0] if category_performance else None
            }
            
        except Exception as e:
            print(f"Error getting evaluation statistics: {e}")
            return {
                'total_evaluations': 0,
                'average_score': 0,
                'score_distribution': {},
                'category_performance': {},
                'recent_evaluations': 0
            }


# Create global instance
supabase_manager = SupabaseManager()