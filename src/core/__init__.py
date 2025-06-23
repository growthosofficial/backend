"""
Core processing modules for Second Brain Knowledge Management System
"""

from .embeddings import get_embedding
from .similarity import SSC
from .llm_updater import LLMUpdater, call_azure_openai_llm

__all__ = [
    "get_embedding",
    "SSC", 
    "LLMUpdater",
    "call_azure_openai_llm"
]