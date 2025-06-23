from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# REQUEST MODELS (for processing only)

class ProcessTextRequest(BaseModel):
    """Request model for processing text input"""
    text: str = Field(..., min_length=1, max_length=10000, description="Input text to process")
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold for matching")


# RESPONSE MODELS (read-only data from database)

class RecommendationResponse(BaseModel):
    """Single recommendation response"""
    option_number: int = Field(..., description="Recommendation option number (1-3)")
    change: str = Field(..., description="Explanation of what changes will be made")
    updated_text: str = Field(..., description="The complete updated/new text content")
    main_category: str = Field(..., description="Main academic subject category")
    sub_category: str = Field(..., description="Sub-category within the academic subject")
    tags: List[str] = Field(default_factory=list, description="Content-based tags")
    preview: str = Field(..., description="Short preview of the content")


class ProcessTextResponse(BaseModel):
    """Response model for text processing"""
    recommendations: List[RecommendationResponse] = Field(..., description="List of 3 recommendations")
    similar_main_category: Optional[str] = Field(None, description="Most similar existing main category if found")
    similar_sub_category: Optional[str] = Field(None, description="Most similar existing sub-category if found")
    similarity_score: Optional[float] = Field(None, description="Similarity score if match found")
    status: str = Field("success", description="Request status")


class KnowledgeItemResponse(BaseModel):
    """Response model for knowledge items (read-only)"""
    id: int
    main_category: str
    sub_category: str
    content: str
    tags: List[str]
    source: Optional[str] = "text"
    strength_score: Optional[float] = None
    last_reviewed: Optional[datetime] = None
    next_review_due: Optional[datetime] = None
    review_count: Optional[int] = 0
    ease_factor: Optional[float] = 2.5
    interval_days: Optional[int] = 1
    created_at: datetime
    last_updated: datetime


class CategoryResponse(BaseModel):
    """Single knowledge category response"""
    main_category: str = Field(..., description="Main academic subject")
    sub_categories: List[Dict[str, Any]] = Field(default_factory=list, description="Sub-categories within this subject")
    total_items: int = Field(0, description="Total number of items in this category")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")


class CategoriesResponse(BaseModel):
    """Response model for listing categories"""
    categories: List[CategoryResponse] = Field(..., description="List of all academic subject categories")
    total_main_categories: int = Field(..., description="Total number of main academic subjects")
    total_sub_categories: int = Field(..., description="Total number of sub-categories")
    total_items: int = Field(..., description="Total number of knowledge items")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Overall system status")
    database: str = Field(..., description="Database connection status")
    openai_api: str = Field(..., description="OpenAI API status")
    timestamp: datetime = Field(..., description="Health check timestamp")


class StatsResponse(BaseModel):
    """Response model for database statistics"""
    total_items: int = Field(..., description="Total number of knowledge items")
    unique_main_categories: int = Field(..., description="Number of unique main academic subjects")
    unique_sub_categories: int = Field(..., description="Number of unique sub-categories")
    unique_tags: int = Field(..., description="Number of unique tags")
    unique_sources: int = Field(..., description="Number of unique sources")
    main_category_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution of items across main categories")
    source_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution of items by source")
    most_common_tags: List[tuple] = Field(default_factory=list, description="Most frequently used tags")
    avg_strength_score: float = Field(0.0, description="Average knowledge strength score")
    items_with_strength_score: int = Field(0, description="Number of items with strength scores")
    strong_items: int = Field(0, description="Number of items with strength >= 0.8")
    weak_items: int = Field(0, description="Number of items with strength < 0.5")
    total_reviews_completed: int = Field(0, description="Total number of reviews completed")
    database_type: str = Field("Supabase (Read-Only)", description="Type of database")


class StrengthDistributionResponse(BaseModel):
    """Response model for knowledge strength distribution"""
    very_weak: int = Field(0, description="Items with strength 0.0-0.2")
    weak: int = Field(0, description="Items with strength 0.2-0.4")
    medium: int = Field(0, description="Items with strength 0.4-0.6")
    strong: int = Field(0, description="Items with strength 0.6-0.8")
    very_strong: int = Field(0, description="Items with strength 0.8-1.0")
    no_score: int = Field(0, description="Items without strength scores")


class CategoryStrengthResponse(BaseModel):
    """Response model for category strength analysis"""
    category: str
    avg_strength: float
    total_items: int
    strong_items: int
    weak_items: int


class ItemsDueResponse(BaseModel):
    """Response model for items due for review (analytics only)"""
    items_due: List[KnowledgeItemResponse]
    total_due: int
    status: str = "success"


class SearchResponse(BaseModel):
    """Response model for search results"""
    results: List[KnowledgeItemResponse]
    total_results: int
    search_term: Optional[str] = None
    status: str = "success"


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")