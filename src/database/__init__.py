"""
Database package for Second Brain Knowledge Management System
"""

from .supabase_manager import supabase_manager, SupabaseManager

__all__ = [
    "supabase_manager",
    "SupabaseManager"
]