"""
Self-test generation and assessment logic for the Second Brain Knowledge Management System.
"""
from typing import List
import random
from fastapi import APIRouter, HTTPException, status, Query, Path

from database.supabase_manager import supabase_manager
from core.self_test import (
    generate_free_text_question, 
    generate_multiple_choice_question,
    evaluate_answer, 
    get_knowledge_mastery
)
from models import (
    QuestionType, Question, GenerateQuestionsResponse, AnswerRequest, EvaluationResponse,
    BatchAnswerRequest, BatchEvaluationResponse, EvaluationHistoryResponse,
    MultipleChoiceQuestion, GenerateMultipleChoiceResponse
)

# Create router
router = APIRouter(
    prefix="/api/self-test",
    tags=["Knowledge Assessment"]
)

# Free Text Question Endpoints
@router.post("/free-text/generate", response_model=GenerateQuestionsResponse)
async def generate_free_text_questions(
    num_questions: int = Query(default=3, ge=1, le=20, description="Number of questions to generate"),
    main_category: str | None = Query(default=None, description="Optional main category to filter questions by")
):
    """
    Generate free text questions from the knowledge base.
    Questions will be evaluated by ChatGPT later.
    
    This endpoint:
    1. Retrieves random knowledge items from the database
    2. Uses Azure OpenAI to generate thought-provoking questions
    3. Returns questions with their knowledge_id for later evaluation
    
    Args:
        num_questions: Number of questions to generate (1-20)
        main_category: Optional main category to filter questions by
    """
    try:
        # Get knowledge items, optionally filtered by main category
        knowledge_items = supabase_manager.load_all_knowledge(main_category=main_category)
        if not knowledge_items:
            error_msg = "No knowledge items found in database"
            if main_category:
                error_msg = f"No knowledge items found for category: {main_category}"
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg
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
        
        for item in selected_items:
            # Get categories and content
            item_main_category = item.get('main_category', 'Unknown')
            sub_category = item.get('sub_category', 'Unknown')
            content = item.get('content', '')
            knowledge_id = item['id']  # Already validated as positive integer
            
            # Generate question using our dedicated function
            question_data = generate_free_text_question(
                category=item_main_category,  # Pass main category for backwards compatibility
                content=content,
                knowledge_id=knowledge_id
            )
            
            if question_data:
                # Create Question object
                question = Question(
                    question_text=question_data['question_text'],
                    main_category=item_main_category,
                    sub_category=sub_category,
                    knowledge_id=knowledge_id,
                    answer=""
                )
                questions.append(question)
        
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate any valid questions"
            )
        
        return GenerateQuestionsResponse(
            questions=questions,
            total_questions=len(questions)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating questions: {str(e)}"
        )

@router.post("/free-text/evaluate", response_model=BatchEvaluationResponse)
async def evaluate_free_text_answers(request: BatchAnswerRequest):
    """
    Evaluate multiple free text answers to questions and store results.
    
    This endpoint:
    1. Takes a list of answers with their corresponding knowledge IDs and questions
    2. Efficiently loads all required knowledge items at once
    3. Evaluates each answer using Azure OpenAI
    4. Stores evaluation results and updates mastery in the database
    5. Returns detailed feedback, scores, mastery levels, and sample answers
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
            main_category = knowledge_item.get('main_category', 'Unknown')
            sub_category = knowledge_item.get('sub_category', 'Unknown')
            
            if not knowledge_content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Knowledge item {knowledge_id} has no content"
                )
            
            # Evaluate the answer
            evaluation = evaluate_answer(
                question_text=answer_request.question_text,
                answer=answer_request.answer,
                knowledge_content=knowledge_content,
                main_category=main_category,
                sub_category=sub_category
            )
            
            # Create evaluation data
            evaluation_data = {
                "knowledge_id": knowledge_id,
                "question_text": answer_request.question_text,
                "answer_text": answer_request.answer,
                "score": evaluation['score'],
                "feedback": evaluation['feedback'],
                "correct_points": evaluation['correct_points'],
                "incorrect_points": evaluation['incorrect_points']
            }
            
            # Store evaluation in database
            stored_eval = supabase_manager.create_evaluation(evaluation_data)
            
            # Calculate and update mastery
            mastery_result = get_knowledge_mastery(
                knowledge_id=knowledge_id,
                current_evaluation=evaluation_data,
                supabase_manager=supabase_manager
            )
            
            # Update knowledge item mastery and explanation
            supabase_manager.update_mastery(
                knowledge_id=knowledge_id,
                evaluation_id=stored_eval['id'] if stored_eval else None,
                mastery=mastery_result['mastery'],
                mastery_explanation=mastery_result['explanation']
            )
            
            # Create evaluation response
            evaluation_response = EvaluationResponse(
                question_text=answer_request.question_text,
                answer=answer_request.answer,
                score=evaluation['score'],
                feedback=evaluation['feedback'],
                correct_points=evaluation['correct_points'],
                incorrect_points=evaluation['incorrect_points'],
                evaluation_id=stored_eval['id'] if stored_eval else None,
                knowledge_id=knowledge_id,
                mastery=mastery_result['mastery'],
                mastery_explanation=mastery_result['explanation'],
                sample_answer=evaluation.get('sample_answer')
            )
            
            evaluations.append(evaluation_response)
        
        return BatchEvaluationResponse(
            evaluations=evaluations,
            total_evaluated=len(evaluations)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluating answers: {str(e)}"
        )

# Multiple Choice Question Endpoints
@router.post("/multiple-choice/generate", response_model=GenerateMultipleChoiceResponse)
async def generate_multiple_choice_questions(
    num_questions: int = Query(default=3, ge=1, le=20, description="Number of questions to generate"),
    main_category: str | None = Query(default=None, description="Optional main category to filter questions by")
):
    """
    Generate multiple choice questions from the knowledge base.
    Questions will be evaluated by ChatGPT later.
    
    This endpoint:
    1. Retrieves random knowledge items from the database
    2. Uses Azure OpenAI to generate thought-provoking multiple choice questions
    3. Returns questions with their knowledge_id for later evaluation
    
    Args:
        num_questions: Number of questions to generate (1-20)
        main_category: Optional main category to filter questions by
    """
    try:
        # Get knowledge items, optionally filtered by main category
        knowledge_items = supabase_manager.load_all_knowledge(main_category=main_category)
        if not knowledge_items:
            error_msg = "No knowledge items found in database"
            if main_category:
                error_msg = f"No knowledge items found for category: {main_category}"
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg
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
        questions_to_create = []
        
        for item in selected_items:
            # Get categories and content
            item_main_category = item.get('main_category', 'Unknown')
            content = item.get('content', '')
            knowledge_id = item['id']  # Already validated as positive integer
            
            # Generate multiple choice question using our dedicated function
            question_data = generate_multiple_choice_question(
                category=item_main_category,
                content=content,
                knowledge_id=knowledge_id
            )
            
            if question_data:
                questions_to_create.append({
                    "question_text": question_data['question_text'],
                    "options": question_data['options'],
                    "correct_answer_index": question_data['correct_answer_index'],
                    "explanation": question_data['explanation'],
                    "knowledge_id": knowledge_id
                })
        
        if not questions_to_create:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate any valid multiple choice questions"
            )
        
        # Store questions in database in batch
        db_questions = supabase_manager.create_multiple_choice_questions(questions_to_create)
        if not db_questions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store multiple choice questions"
            )
        
        # Create response objects
        questions = [
            MultipleChoiceQuestion(
                id=q['id'],
                question_text=q['question_text'],
                options=q['options'],
                correct_answer_index=q['correct_answer_index'],
                explanation=q['explanation'],
                knowledge_id=q['knowledge_id']
            )
            for q in db_questions
        ]
        
        return GenerateMultipleChoiceResponse(
            questions=questions,
            total_questions=len(questions)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating multiple choice questions: {str(e)}"
        )

@router.get("/evaluations/{knowledge_id}", response_model=EvaluationHistoryResponse)
async def get_evaluations_by_knowledge_id(
    knowledge_id: int = Path(gt=0, title="Knowledge ID", description="ID of the knowledge item to get evaluations for"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of evaluations to return")
):
    """
    Get evaluation history for a specific knowledge item.
    
    Args:
        knowledge_id: ID of the knowledge item
        limit: Maximum number of evaluations to return (1-100)
        
    Returns:
        List of evaluations with scores, feedback, and mastery progression
    """
    try:
        # Get knowledge item to verify it exists and get current mastery
        knowledge_item = supabase_manager.get_knowledge_by_id(knowledge_id)
        if not knowledge_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge item {knowledge_id} not found"
            )
        
        # Get evaluations for this knowledge item
        eval_result = supabase_manager.supabase.table('evaluations')\
            .select('*')\
            .eq('knowledge_id', knowledge_id)\
            .order('created_at', desc=False)\
            .limit(limit)\
            .execute()
            
        evaluations = []
        total_score = 0
        
        for eval_data in eval_result.data:
            evaluation = EvaluationResponse(
                question_text=eval_data.get('question_text', ''),
                answer=eval_data.get('answer_text', ''),
                score=eval_data.get('score', 1),
                feedback=eval_data.get('feedback', ''),
                correct_points=eval_data.get('correct_points', []),
                incorrect_points=eval_data.get('incorrect_points', []),
                evaluation_id=eval_data.get('id'),
                knowledge_id=knowledge_id,
                mastery=eval_data.get('mastery', 0.0),
                mastery_explanation=eval_data.get('mastery_explanation', ''),
                sample_answer=eval_data.get('sample_answer', None)
            )
            evaluations.append(evaluation)
            total_score += evaluation.score
        
        # Calculate average score
        average_score = total_score / len(evaluations) if evaluations else 0
        
        return EvaluationHistoryResponse(
            evaluations=evaluations,
            knowledge_id=knowledge_id,
            total_evaluations=len(evaluations),
            average_score=round(average_score, 2),
            current_mastery=knowledge_item.get('mastery', 0.0),
            mastery_explanation=knowledge_item.get('mastery_explanation', '')
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving evaluations: {str(e)}"
        )