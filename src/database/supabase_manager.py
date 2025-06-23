import json
from typing import List, Dict, Optional
from supabase import create_client, Client
from datetime import datetime

from config.settings import settings
from utils.category_mapping import get_main_category

class SupabaseManager:
    """Manages knowledge items in Supabase database with two-tier categorization"""
    
    def __init__(self):
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("Supabase URL and KEY must be configured")
        
        self.supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the knowledge_items table exists with updated schema"""
        # Note: You'll need to update your Supabase table schema to include:
        # - main_category (text)
        # - sub_category (text) 
        # - Rename existing 'category' to 'sub_category' if needed
        # - Add main_category column
        pass
    
    def add_knowledge_item(self, knowledge_item: Dict) -> Dict:
        """
        Add or update a knowledge item in Supabase with two-tier categorization
        
        Args:
            knowledge_item: Knowledge item dictionary with main_category and sub_category
            
        Returns:
            The inserted/updated record
        """
        # Ensure main_category is set
        sub_category = knowledge_item.get('sub_category', knowledge_item.get('category', 'general'))
        main_category = knowledge_item.get('main_category')
        
        if not main_category:
            main_category = get_main_category(sub_category)
        
        # Prepare data for insertion
        data = {
            'main_category': main_category,
            'sub_category': sub_category,
            'content': knowledge_item['content'],
            'tags': knowledge_item.get('tags', []),
            'embedding': knowledge_item.get('embedding', []),
            'created_at': datetime.utcnow().isoformat(),
            'last_updated': datetime.utcnow().isoformat()
        }
        
        # Check if sub_category already exists (since sub-categories should be unique)
        existing = self.get_knowledge_by_sub_category(sub_category)
        
        if existing:
            # Update existing record
            result = self.supabase.table('knowledge_items').update(data).eq('sub_category', sub_category).execute()
            print(f"Updated existing sub-category: {sub_category} (Main: {main_category})")
        else:
            # Insert new record
            result = self.supabase.table('knowledge_items').insert(data).execute()
            print(f"Added new sub-category: {sub_category} (Main: {main_category})")
        
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
        
        if result.data:
            return result.data[0]
        return None
    
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
        Load all knowledge items from Supabase
        
        Returns:
            List of all knowledge items with main_category and sub_category
        """
        result = self.supabase.table('knowledge_items').select('*').execute()
        
        # Ensure backward compatibility for existing data
        for item in result.data:
            # If old schema, migrate on-the-fly
            if 'category' in item and 'sub_category' not in item:
                item['sub_category'] = item['category']
            if 'main_category' not in item and 'sub_category' in item:
                item['main_category'] = get_main_category(item['sub_category'])
        
        print(f"📚 Loaded {len(result.data)} knowledge items from Supabase")
        return result.data
    
    def delete_knowledge_item(self, sub_category: str) -> bool:
        """
        Delete a knowledge item by sub-category
        
        Args:
            sub_category: Sub-category name to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        result = self.supabase.table('knowledge_items').delete().eq('sub_category', sub_category).execute()
        
        if result.data:
            print(f"Deleted sub-category: {sub_category}")
            return True
        else:
            print(f"Sub-category not found: {sub_category}")
            return False
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the knowledge database with two-tier categorization
        
        Returns:
            Dictionary with database statistics including main/sub category breakdowns
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
        
        for item in all_items:
            # Tags
            all_tags.extend(item.get('tags', []))
            
            # Categories
            main_cat = item.get('main_category', 'General')
            sub_cat = item.get('sub_category', 'unknown')
            
            main_categories.append(main_cat)
            sub_categories.append(sub_cat)
            
            # Count main categories
            main_category_counts[main_cat] = main_category_counts.get(main_cat, 0) + 1
        
        # Count unique items
        unique_main_categories = list(set(main_categories))
        unique_sub_categories = list(set(sub_categories))
        unique_tags = list(set(all_tags))
        tag_counts = {tag: all_tags.count(tag) for tag in unique_tags}
        
        stats = {
            'total_knowledge_items': total_items,
            'unique_main_categories': len(unique_main_categories),
            'unique_sub_categories': len(unique_sub_categories),
            'unique_tags': len(unique_tags),
            'main_category_distribution': main_category_counts,
            'main_categories': unique_main_categories,
            'sub_categories': unique_sub_categories,
            'most_common_tags': sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'database_type': 'Supabase'
        }
        
        return stats
    
    def get_categories_grouped(self) -> Dict:
        """
        Get all categories grouped by main category
        
        Returns:
            Dictionary with main categories as keys and sub-categories as values
        """
        all_items = self.load_all_knowledge()
        grouped = {}
        
        for item in all_items:
            main_cat = item.get('main_category', 'General')
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

# Create global instance
supabase_manager = SupabaseManager()