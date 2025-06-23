"""
Self-test generation and assessment logic for the Second Brain Knowledge Management System.
"""
import sys
from pathlib import Path
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field
import random
import json
from fastapi import APIRouter, HTTPException, status, Query

# Add parent directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

try:
    from database.supabase_manager import supabase_manager
    from core.self_test import generate_question, evaluate_answer
except ImportError as e:
    print(f"❌ Import error in self_test.py: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path}")
    raise

# Create router
router = APIRouter(
    prefix="/api/self-test",
    tags=["Knowledge Assessment"]
)

# Models
class QuestionType(str, Enum):
    """Type of quiz question"""
    MULTIPLE_CHOICE = "multiple_choice"
    OPEN_ENDED = "open_ended"

class Question(BaseModel):
    """Question model for self-test feature"""
    question_text: str = Field(..., description="The question text")
    category: str = Field(..., description="Knowledge category this question is from")
    knowledge_id: int = Field(..., gt=0, description="ID of the knowledge item this question is based on")
    answer: str = Field("", description="The answer to evaluate (empty for question generation)")

class GenerateQuestionsResponse(BaseModel):
    """Response model for question generation"""
    questions: List[Question] = Field(..., description="List of generated questions")
    categories_covered: List[str] = Field(..., description="List of knowledge categories covered")
    total_questions: int = Field(..., description="Total number of questions generated")

class AnswerRequest(BaseModel):
    """Request model for submitting an answer"""
    knowledge_id: int = Field(..., gt=0, description="ID of the knowledge item")
    question_text: str = Field(..., min_length=1, description="The question being answered")
    answer: str = Field(..., min_length=1, description="The user's answer to evaluate")

class EvaluationResponse(BaseModel):
    """Response model for answer evaluation"""
    question_text: str = Field(..., description="The original question that was asked")
    answer: str = Field(..., description="Your submitted answer")
    score: int = Field(..., ge=1, le=5, description="Score from 1-5")
    feedback: str = Field(..., description="Overall feedback on the answer")
    correct_points: List[str] = Field(default_factory=list, description="Points that were correct")
    incorrect_points: List[str] = Field(default_factory=list, description="Points that were incorrect or missing")
    improvement_suggestions: str = Field(..., description="Suggestions for improvement")
    evaluation_id: int | None = Field(None, description="Database ID of the stored evaluation")

class BatchAnswerRequest(BaseModel):
    """Request model for submitting multiple answers"""
    answers: List[AnswerRequest] = Field(..., description="List of answers to evaluate")

class BatchEvaluationResponse(BaseModel):
    """Response model for batch answer evaluation"""
    evaluations: List[EvaluationResponse] = Field(..., description="List of evaluations for each answer")
    total_evaluated: int = Field(..., description="Total number of answers evaluated")
    average_score: float = Field(..., ge=1.0, le=5.0, description="Average score across all answers")

@router.post("/generate", response_model=GenerateQuestionsResponse)
async def generate_questions(
    num_questions: int = Query(default=5, ge=1, le=20, description="Number of questions to generate")
):
    """
    Generate questions from the knowledge base.
    Questions will be evaluated by ChatGPT later.
    
    This endpoint:
    1. Retrieves random knowledge items from the database
    2. Uses Azure OpenAI to generate thought-provoking questions
    3. Returns questions with their knowledge_id for later evaluation
    """
    try:
        # Get all knowledge items
        knowledge_items = supabase_manager.load_all_knowledge()
        if not knowledge_items:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No knowledge items found in database"
            )
        
        # Filter out items without valid numeric IDs
        valid_items = []
        for item in knowledge_items:
            try:
                item['id'] = int(item['id'])  # Convert ID to integer
                if item['id'] > 0:  # Ensure positive ID
                    valid_items.append(item)
            except (ValueError, TypeError, KeyError):
                continue
        
        if not valid_items:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No knowledge items with valid IDs found"
            )
        
        # Randomly select items to generate questions from
        selected_items = random.sample(valid_items, min(num_questions, len(valid_items)))
        
        # Generate questions for each selected item
        questions = []
        categories_covered = set()
        
        for item in selected_items:
            # Prepare knowledge text
            category = item.get('category', 'Unknown')
            content = item.get('content', '')
            knowledge_id = item['id']  # Already validated as positive integer
            
            # Generate question using our dedicated function
            question_data = generate_question(category, content)
            
            if question_data:
                # Create Question object
                question = Question(
                    question_text=question_data['question_text'],
                    category=category,
                    knowledge_id=knowledge_id,
                    answer=""
                )
                questions.append(question)
                categories_covered.add(category)
        
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate any valid questions"
            )
        
        return GenerateQuestionsResponse(
            questions=questions,
            categories_covered=list(categories_covered),
            total_questions=len(questions)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating questions: {str(e)}"
        )

@router.post("/evaluate", response_model=BatchEvaluationResponse)
async def evaluate_answers(request: BatchAnswerRequest):
    """
    Evaluate multiple answers to questions and store results.
    
    This endpoint:
    1. Takes a list of answers with their corresponding knowledge IDs and questions
    2. Efficiently loads all required knowledge items at once
    3. Evaluates each answer using Azure OpenAI
    4. Stores evaluation results in the database
    5. Returns detailed feedback and scores for all answers
    """
    try:
        # Extract unique knowledge IDs
        knowledge_ids = list(set(answer.knowledge_id for answer in request.answers))
        
        # Load all required knowledge items efficiently
        knowledge_items = supabase_manager.load_knowledge_by_ids(knowledge_ids)
        if not knowledge_items:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No knowledge items found for the provided IDs"
            )
        
        # Index knowledge items by ID for efficient lookup
        knowledge_map = {int(item['id']): item for item in knowledge_items}
        
        # Evaluate each answer
        evaluations = []
        total_score = 0
        
        for answer_request in request.answers:
            knowledge_id = answer_request.knowledge_id
            
            # Get the knowledge item
            knowledge_item = knowledge_map.get(knowledge_id)
            if not knowledge_item:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Knowledge item with ID {knowledge_id} not found"
                )
            
            # Get the knowledge content and category
            knowledge_content = knowledge_item.get('content', '')
            category = knowledge_item.get('category', 'Unknown')
            
            if not knowledge_content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Knowledge item {knowledge_id} has no content"
                )
            
            if category == 'Unknown':
                print(f"Warning: Knowledge item {knowledge_id} has no category, using 'Unknown'")
            
            # Evaluate the answer
            evaluation = evaluate_answer(
                question_text=answer_request.question_text,
                answer=answer_request.answer,
                knowledge_content=knowledge_content,
                category=category
            )
            
            # Store evaluation in database
            stored_eval = supabase_manager.create_evaluation({
                "knowledge_id": knowledge_id,
                "question_text": answer_request.question_text,
                "answer_text": answer_request.answer,
                "score": evaluation['score'],
                "feedback": evaluation['feedback'],
                "improvement_suggestions": evaluation['improvement_suggestions'],
                "correct_points": evaluation['correct_points'],
                "incorrect_points": evaluation['incorrect_points']
            })
            
            # Create evaluation response
            evaluation_response = EvaluationResponse(
                question_text=answer_request.question_text,
                answer=answer_request.answer,
                score=evaluation['score'],
                feedback=evaluation['feedback'],
                correct_points=evaluation['correct_points'],
                incorrect_points=evaluation['incorrect_points'],
                improvement_suggestions=evaluation['improvement_suggestions'],
                evaluation_id=stored_eval['id'] if stored_eval else None
            )
            
            evaluations.append(evaluation_response)
            total_score += evaluation['score']
        
        # Calculate average score (now on 1-5 scale)
        average_score = total_score / len(evaluations) if evaluations else 1.0
        
        return BatchEvaluationResponse(
            evaluations=evaluations,
            total_evaluated=len(evaluations),
            average_score=round(average_score, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluating answers: {str(e)}"
        )