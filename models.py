from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from sqlmodel import SQLModel


class ProcessTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Input text to process")
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold for matching")


class RecommendationResponse(BaseModel):
    option_number: int
    change: str
    updated_text: str
    main_category: str = Field(..., description="Main category (hardcoded)")
    sub_category: str = Field(..., description="Sub-category (LLM generated)")
    tags: List[str]
    preview: str


class ProcessTextResponse(BaseModel):
    recommendations: List[RecommendationResponse]
    similar_main_category: Optional[str] = None
    similar_sub_category: Optional[str] = None
    similarity_score: Optional[float] = None
    status: str = "success"


class KnowledgeItemBase(SQLModel):
    main_category: str = Field(..., description="Main knowledge category (hardcoded)")
    sub_category: str = Field(..., description="Sub-category (LLM generated)")
    content: str = Field(..., description="Main content")
    tags: List[str] = Field(default_factory=list, description="Content tags")


class KnowledgeItem(KnowledgeItemBase, table=True):
    __tablename__ = "knowledge_items"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    main_category: str
    sub_category: str
    content: str
    tags: str = Field(default="[]")  # Store as JSON string
    embedding: str = Field(default="[]")  # Store as JSON string
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)


class KnowledgeItemCreate(KnowledgeItemBase):
    main_category: Optional[str] = Field(None, description="Main category (will be auto-determined if not provided)")
    
    @validator('main_category', pre=True, always=True)
    def set_main_category(cls, v, values):
        if v is None and 'sub_category' in values:
            # Import here to avoid circular imports
            from src.utils.category_mapping import get_main_category
            return get_main_category(values['sub_category'])
        return v


class KnowledgeItemUpdate(BaseModel):
    main_category: Optional[str] = None
    sub_category: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    
    @validator('main_category', pre=True, always=True)
    def update_main_category(cls, v, values):
        if v is None and 'sub_category' in values and values['sub_category'] is not None:
            # Import here to avoid circular imports
            from src.utils.category_mapping import get_main_category
            return get_main_category(values['sub_category'])
        return v


class KnowledgeItemResponse(KnowledgeItemBase):
    id: int
    created_at: datetime
    last_updated: datetime


class CategoryResponse(BaseModel):
    main_category: str
    sub_categories: List[Dict[str, Any]] = Field(default_factory=list)
    total_items: int = 0
    last_updated: Optional[datetime] = None


class CategoriesResponse(BaseModel):
    categories: List[CategoryResponse]
    total_main_categories: int
    total_sub_categories: int
    total_items: int


class HealthResponse(BaseModel):
    status: str
    database: str
    openai_api: str
    timestamp: datetime


class StatsResponse(BaseModel):
    total_items: int
    unique_main_categories: int
    unique_sub_categories: int
    unique_tags: int
    latest_update: Optional[datetime]
    main_category_distribution: Dict[str, int] = Field(default_factory=dict)
    most_common_tags: List[tuple] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)