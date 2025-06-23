from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from sqlmodel import SQLModel


class ProcessTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Input text to process")
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold for matching")


class RecommendationResponse(BaseModel):
    option_number: int
    change: str
    updated_text: str
    category: str
    tags: List[str]
    preview: str


class ProcessTextResponse(BaseModel):
    recommendations: List[RecommendationResponse]
    similar_category: Optional[str] = None
    similarity_score: Optional[float] = None
    status: str = "success"


class KnowledgeItemBase(SQLModel):
    category: str = Field(..., description="Knowledge category")
    content: str = Field(..., description="Main content")
    tags: List[str] = Field(default_factory=list, description="Content tags")


class KnowledgeItem(KnowledgeItemBase, table=True):
    __tablename__ = "knowledge_items"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    category: str
    content: str
    tags: str = Field(default="[]")  # Store as JSON string
    embedding: str = Field(default="[]")  # Store as JSON string
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)


class KnowledgeItemCreate(KnowledgeItemBase):
    pass


class KnowledgeItemUpdate(KnowledgeItemBase):
    category: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None


class KnowledgeItemResponse(KnowledgeItemBase):
    id: int
    created_at: datetime
    last_updated: datetime


class HealthResponse(BaseModel):
    status: str
    database: str
    openai_api: str
    timestamp: datetime


class StatsResponse(BaseModel):
    total_items: int
    unique_categories: int
    unique_tags: int
    latest_update: Optional[datetime]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)