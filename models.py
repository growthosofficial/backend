from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from pydantic import validator


# SELF-TEST MODELS

class QuestionType(str, Enum):
    """Type of quiz question"""
    MULTIPLE_CHOICE = "multiple_choice"
    FREE_TEXT = "free_text"

class Question(BaseModel):
    """Question model for self-test feature"""
    question_text: str = Field(..., description="The question text")
    main_category: str = Field(..., description="Main knowledge category")
    sub_category: str = Field(..., description="Sub category within the main category")
    knowledge_id: int = Field(..., gt=0, description="ID of the knowledge item this question is based on")
    answer: str = Field("", description="The answer to evaluate (empty for question generation)")
    question_type: QuestionType = Field(default=QuestionType.FREE_TEXT, description="Type of question (multiple choice or free text)")

class MultipleChoiceQuestion(BaseModel):
    """Internal model for multiple choice questions with all fields"""
    question_id: int = Field(..., description="Database ID of the question")
    question_text: str = Field(..., description="The question text")
    options: List[str] = Field(..., description="List of 4 options (indexed 0-3)")
    correct_answer_index: int = Field(..., ge=0, le=3, description="Index of correct answer (0-3)")
    explanation: str = Field(..., description="Explanation of correct answer and distractors")
    knowledge_id: int = Field(..., gt=0, description="ID of the knowledge item this question is based on")

    @validator('options')
    def validate_options(cls, v):
        if len(v) != 4:
            raise ValueError('Must have exactly 4 options')
        return v

class MultipleChoiceQuestionResponse(BaseModel):
    """Public response model for multiple choice questions (without answers)"""
    question_id: int = Field(..., description="Database ID of the question")
    question_text: str = Field(..., description="The question text")
    options: List[str] = Field(..., description="List of 4 options (indexed 0-3)")
    knowledge_id: int = Field(..., gt=0, description="ID of the knowledge item this question is based on")
    selected_answer_index: int | None = Field(None, description="Index of selected option (0-3)")

class GenerateQuestionsResponse(BaseModel):
    """Response model for question generation"""
    questions: List[Question] = Field(..., description="List of generated questions")
    total_questions: int = Field(..., description="Total number of questions generated")

class GenerateMultipleChoiceResponse(BaseModel):
    """Response model for multiple choice question generation"""
    questions: List[MultipleChoiceQuestionResponse] = Field(..., description="List of generated multiple choice questions")
    total_questions: int = Field(..., description="Total number of questions generated")

class AnswerRequest(BaseModel):
    """Request model for submitting an answer"""
    knowledge_id: int = Field(..., gt=0, description="ID of the knowledge item")
    question_text: str = Field(..., min_length=1, description="The question being answered")
    answer: str = Field(..., min_length=1, description="The user's answer to evaluate")

class MultipleChoiceAnswerRequest(BaseModel):
    """Request model for submitting a multiple choice answer"""
    question_id: int = Field(..., gt=0, description="ID of the multiple choice question")
    selected_answer_index: int = Field(..., ge=0, le=3, description="Index of selected option (0-3)")

class MultipleChoiceBatchAnswerRequest(BaseModel):
    """Request model for submitting multiple multiple choice answers"""
    answers: List[MultipleChoiceAnswerRequest] = Field(..., description="List of answers to evaluate")

class EvaluationResponse(BaseModel):
    """Response model for evaluation results"""
    question_text: str
    answer: str
    score: float | None = None
    feedback: str
    correct_points: List[str]
    incorrect_points: List[str]
    evaluation_id: int | None = None
    knowledge_id: int
    mastery: float
    mastery_explanation: str
    sample_answer: str | None = None
    is_correct: bool | None = None
    multiple_choice_question_id: int | None = None
    # Category fields
    main_category: str | None = None
    sub_category: str | None = None
    # Multiple choice specific fields
    options: List[str] | None = None
    selected_index: int | None = None
    correct_answer_index: int | None = None

class BatchAnswerRequest(BaseModel):
    """Request model for submitting multiple answers"""
    answers: List[AnswerRequest] = Field(..., description="List of answers to evaluate")

class BatchEvaluationResponse(BaseModel):
    """Response model for batch answer evaluation"""
    evaluations: List[EvaluationResponse] = Field(..., description="List of evaluations for each answer")
    total_evaluated: int = Field(..., description="Total number of answers evaluated")

class EvaluationHistoryResponse(BaseModel):
    """Response model for evaluation history"""
    evaluations: List[EvaluationResponse]
    knowledge_id: int
    total_evaluations: int
    average_score: float
    current_mastery: float
    mastery_explanation: str


# REQUEST MODELS (for processing only)

class ProcessTextRequest(BaseModel):
    """Request model for processing text input"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to process")
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold")
    goal: Optional[str] = Field(None, max_length=500, description="Learning goal for relevance analysis")


# RESPONSE MODELS (read-only data from database)

class RecommendationResponse(BaseModel):
    option_number: int = Field(..., description="Recommendation option number (1-4)")
    change: str = Field(..., description="Detailed explanation of changes and benefits")
    updated_text: str = Field(..., description="Complete updated/new text content")
    main_category: str = Field(..., description="Main academic category")
    sub_category: str = Field(..., description="Specific sub-category")
    tags: list[str] = Field(default_factory=list, description="Semantic-optimized tags")
    preview: str = Field(..., description="Short preview of content")
    action_type: str = Field(..., description="Action type: merge/update/create_new")
    reasoning: str = Field(..., description="Reasoning for this approach")
    semantic_coherence: str = Field(..., description="Coherence assessment: high/medium/low")
    is_goal_aware: bool = Field(..., description="Whether this considered the goal")
    
    # Goal-specific fields (optional)
    relevance_score: Optional[int] = Field(None, ge=1, le=10, description="Goal relevance (1-10)")
    goal_alignment: Optional[str] = Field(None, description="How this aligns with the goal")
    goal_priority: Optional[str] = Field(None, description="Priority level: high/medium/low")


class ProcessTextResponse(BaseModel):
    """Response model for text processing"""
    recommendations: list[RecommendationResponse] = Field(..., description="List of 4 recommendations")
    similar_main_category: str | None = Field(None, description="Most similar existing main category")
    similar_sub_category: str | None = Field(None, description="Most similar existing sub-category")
    similarity_score: float | None = Field(None, description="Similarity score if match found")
    goal_provided: bool = Field(..., description="Whether a learning goal was provided")
    goal_summary: Optional[str] = Field(None, description="Summary of goal relevance analysis")
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


class MultipleChoiceEvaluationDetail(BaseModel):
    """Detailed evaluation for a multiple choice question"""
    question_text: str
    options: List[str]
    selected_index: int
    correct_answer_index: int
    is_correct: bool
    feedback: str

class MultipleChoiceBatchEvaluationResponse(BaseModel):
    """Response model for batch multiple choice answer evaluation"""
    evaluations: List[EvaluationResponse]
    total_evaluated: int