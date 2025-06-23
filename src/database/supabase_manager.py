
import json
from typing import List, Dict, Optional
from supabase import create_client, Client
from datetime import datetime

from config.settings import settings

class SupabaseManager:
    """Manages knowledge items in Supabase database"""
    
    def __init__(self):
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("Supabase URL and KEY must be configured")
        
        self.supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the knowledge_items table exists"""
        # Note: You'll need to create this table in Supabase manually
        # or use the SQL provided below
        pass
    
    def add_knowledge_item(self, knowledge_item: Dict) -> Dict:
        """
        Add or update a knowledge item in Supabase
        
        Args:
            knowledge_item: Knowledge item dictionary
            
        Returns:
            The inserted/updated record
        """
        # Prepare data for insertion
        data = {
            'category': knowledge_item['category'],
            'content': knowledge_item['content'],
            'tags': knowledge_item.get('tags', []),
            'embedding': knowledge_item.get('embedding', []),
            'created_at': datetime.utcnow().isoformat(),
            'last_updated': datetime.utcnow().isoformat()
        }
        
        # Check if category already exists
        existing = self.get_knowledge_by_category(knowledge_item['category'])
        
        if existing:
            # Update existing record
            result = self.supabase.table('knowledge_items').update(data).eq('category', knowledge_item['category']).execute()
            print(f"Updated existing category: {knowledge_item['category']}")
        else:
            # Insert new record
            result = self.supabase.table('knowledge_items').insert(data).execute()
            print(f"Added new category: {knowledge_item['category']}")
        
        return result.data[0] if result.data else None
    
    def get_knowledge_by_category(self, category: str) -> Optional[Dict]:
        """
        Get a specific knowledge item by category
        
        Args:
            category: Category name to retrieve
            
        Returns:
            Knowledge item dict or None if not found
        """
        result = self.supabase.table('knowledge_items').select('*').eq('category', category).execute()
        
        if result.data:
            return result.data[0]
        return None
    
    def load_all_knowledge(self) -> List[Dict]:
        """
        Load all knowledge items from Supabase
        
        Returns:
            List of all knowledge items
        """
        result = self.supabase.table('knowledge_items').select('*').execute()
        
        print(f"📚 Loaded {len(result.data)} knowledge categories from Supabase")
        return result.data
    
    def delete_knowledge_item(self, category: str) -> bool:
        """
        Delete a knowledge item by category
        
        Args:
            category: Category name to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        result = self.supabase.table('knowledge_items').delete().eq('category', category).execute()
        
        if result.data:
            print(f"Deleted category: {category}")
            return True
        else:
            print(f"Category not found: {category}")
            return False
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the knowledge database
        
        Returns:
            Dictionary with database statistics
        """
        # Get total count
        count_result = self.supabase.table('knowledge_items').select('id', count='exact').execute()
        total_items = count_result.count
        
        # Get all items for tag analysis
        all_items = self.load_all_knowledge()
        
        # Collect all tags
        all_tags = []
        categories = []
        
        for item in all_items:
            all_tags.extend(item.get('tags', []))
            categories.append(item.get('category', 'unknown'))
        
        # Count unique tags
        unique_tags = list(set(all_tags))
        tag_counts = {tag: all_tags.count(tag) for tag in unique_tags}
        
        stats = {
            'total_knowledge_items': total_items,
            'unique_tags': len(unique_tags),
            'categories': categories,
            'most_common_tags': sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'database_type': 'Supabase'
        }
        
        return stats

# Create global instance
supabase_manager = SupabaseManager()