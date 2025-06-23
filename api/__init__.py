"""
FastAPI package for Second Brain Knowledge Management System
"""

from .main import app
from .service import KnowledgeService
from .models import *

__all__ = ["app", "KnowledgeService"]