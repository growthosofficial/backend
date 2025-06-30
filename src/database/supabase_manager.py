import json
from typing import Any, List, Dict, Optional
from supabase import create_client, Client
from datetime import datetime, timedelta, timezone
import random

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
    
    def load_all_knowledge(self, main_category: str | None = None) -> List[Dict]:
        """
        Load all knowledge items from Supabase for similarity analysis
        
        Args:
            main_category: Optional main category to filter by
            
        Returns:
            List of all knowledge items with main_category and sub_category
        """
        query = self.supabase.table('knowledge_items').select('*').order('created_at', desc=True)
        
        if main_category:
            query = query.eq('main_category', main_category)
            
        result = query.execute()
        
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
        count_result = self.supabase.table('knowledge_items').select('id').execute()
        total_items = len(count_result.data)
        
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

    def update_mastery(self, knowledge_id: int, evaluation_id: int | None, mastery: float, mastery_explanation: str = "") -> None:
        """
        Update the mastery level and explanation of a knowledge item
        
        Args:
            knowledge_id: ID of the knowledge item
            mastery: Mastery level between 0 and 1
            mastery_explanation: Explanation of how the mastery was calculated
        """
        try:
            mastery = max(0, min(1, mastery))
            mastery = round(mastery, 2)
            
            self.supabase.table('knowledge_items')\
                .update({
                    'mastery': mastery,
                    'mastery_explanation': mastery_explanation
                })\
                .eq('id', knowledge_id)\
                .execute()
            
            if evaluation_id:
                self.supabase.table('evaluations')\
                    .update({
                        'mastery': mastery,
                        'mastery_explanation': mastery_explanation
                    })\
                    .eq('id', evaluation_id)\
                    .execute()
                
        except Exception as e:
            print(f"Error updating mastery for knowledge item {knowledge_id}: {e}")
    
    def create_evaluation_groups(self, count: int, test_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Create evaluation groups for batch evaluations
        
        Args:
            count: Number of evaluation groups to create
            test_id: Optional test ID to associate with the groups
            
        Returns:
            List of created evaluation group records
        """
        groups = []
        for _ in range(count):
            data = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "test_id": test_id
            }
            result = self.supabase.table('evaluation_groups').insert(data).execute()
            if result.data:
                groups.append(result.data[0])
        return groups

    def create_evaluation(self, evaluation_data: Dict) -> Optional[Dict[str, Any]]:
        """
        Create a new evaluation record
        
        Args:
            evaluation_data: Dictionary containing evaluation data including:
                - knowledge_id: ID of the knowledge item being evaluated
                - evaluation_group_id: Optional ID of the evaluation group this belongs to
                - questions: List of questions or content
                - answers: User's answers
                - feedback: Feedback on the answers
                - points: Points earned
                - mastery: Mastery level achieved
                - mastery_explanation: Explanation of mastery calculation
            
        Returns:
            Created evaluation record or None if failed
        """
        try:
            result = self.supabase.table('evaluations').insert(evaluation_data).execute()
            return result.data[0] if result.data else None
            
        except Exception as e:
            print(f"Error in create_evaluation: {e}")
            return None

    def get_evaluations(self, knowledge_id: int) -> List[Dict]:
        """
        Get  most recent evaluations for a specific knowledge item
        
        Args:
            knowledge_id: ID of the knowledge item
            
        Returns:
            Dictionary containing list of most recent evaluations with their questions, answers, feedback and points
        """
        try:
            result = self.supabase.table('evaluations')\
                .select('*')\
                .eq('knowledge_id', knowledge_id)\
                .order('created_at', desc=True)\
                .limit(10)\
                .execute()
            return result.data if result.data else []

        except Exception as e:
            print(f"Error getting evaluations: {e}")
            return []

    def get_evaluations_by_knowledge_ids(self, knowledge_ids: List[int]) -> List[Dict]:
        """
        Get evaluations by knowledge IDs
        
        Args:
            knowledge_ids: List of knowledge IDs to retrieve
        """
        try:
            result = self.supabase.table('evaluations')\
                .select('*')\
                .in_('knowledge_id', knowledge_ids)\
                .execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting evaluations by knowledge IDs: {e}")
            return []

    def create_multiple_choice_questions(self, questions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create multiple choice questions in batch
        
        Args:
            questions_data: List of question data dictionaries, each containing:
                - question_text: The question text
                - options: List of options
                - correct_answer_index: Index of correct answer
                - explanation: Explanation of correct answer
                - knowledge_id: ID of related knowledge item
            
        Returns:
            List of created question records
        """
        try:
            # Add timestamps to each question
            now = datetime.now(timezone.utc).isoformat()
            for question in questions_data:
                question['created_at'] = now
                question['updated_at'] = now
                
            # Insert all questions in one batch
            result = self.supabase.table('multiple_choice_questions').insert(questions_data).execute()
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error creating multiple choice questions: {e}")
            return []

    def get_multiple_choice_questions(self, question_ids: List[int]) -> List[Dict]:
        """
        Get multiple choice questions by IDs
        
        Args:
            question_ids: List of question IDs to retrieve
            
        Returns:
            List of question data
        """
        try:
            if not question_ids:
                return []
                
            result = self.supabase.table('multiple_choice_questions')\
                .select('*')\
                .in_('id', question_ids)\
                .execute()
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error getting multiple choice questions: {e}")
            return []

    def create_test(self, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Create a new test record
        
        Args:
            category: Optional category for the test
            
        Returns:
            Created test record if successful, None otherwise
        """
        try:
            data = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "category": category,
                "score": 0,
                "total_score": 0
            }
            result = self.supabase.table('tests').insert(data).execute()
            return result.data[0] if result.data else None
            
        except Exception as e:
            print(f"Error creating test: {e}")
            return None

    def update_test_scores(self, test_id: int, score: int, total_score: int) -> Optional[Dict[str, Any]]:
        """
        Update the scores for a test
        
        Args:
            test_id: ID of the test to update
            score: Current score achieved
            total_score: Total possible score
            
        Returns:
            Updated test record or None if failed
        """
        try:
            result = self.supabase.table('tests')\
                .update({
                    "score": score,
                    "total_score": total_score,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                })\
                .eq('id', test_id)\
                .execute()
            return result.data[0] if result.data else None
            
        except Exception as e:
            print(f"Error updating test scores: {e}")
            return None

    def get_latest_tests(self, limit: int = 10, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the latest tests with their scores
        
        Args:
            limit: Maximum number of tests to return
            category: Optional category to filter by
            
        Returns:
            List of test records ordered by creation date
        """
        try:
            query = self.supabase.table('tests')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(limit)
            
            if category:
                query = query.eq('category', category)
                
            result = query.execute()
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error getting latest tests: {e}")
            return []

    def get_random_knowledge_items(self, count: int, main_category: Optional[str] = None, sub_category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get random knowledge items from the database
        
        Args:
            count: Number of items to return
            main_category: Optional main category to filter by
            sub_category: Optional sub category to filter by
            
        Returns:
            List of knowledge items
        """
        try:
            # Build query with filters
            query = self.supabase.table('knowledge_items').select('*')
            
            if main_category:
                query = query.filter('main_category', 'eq', main_category)
            if sub_category:
                query = query.filter('sub_category', 'eq', sub_category)
            
            # Get all matching items first
            result = query.execute()
            if not result.data:
                return []
            
            # Randomly select 'count' items
            items = random.sample(result.data, min(count, len(result.data)))
            return items
            
        except Exception as e:
            print(f"Error getting random knowledge: {e}")
            return []

# Create global instance
supabase_manager = SupabaseManager()