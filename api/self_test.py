"""
Self-test generation and assessment logic for the Second Brain Knowledge Management System.
"""
from typing import List, Dict, Optional
import random
import json
import time
from fastapi import APIRouter, HTTPException, status, Query, Path
from datetime import datetime, timezone

from database.supabase_manager import supabase_manager
from src.core.self_test import (
    generate_free_text_questions,
    generate_multiple_choice_questions_batch,
    evaluate_free_text_answer,
    calculate_free_text_mastery,
    calculate_multiple_choice_mastery
)
from models import (
    QuestionType, Question, GenerateQuestionsResponse, AnswerRequest, EvaluationResponse,
    BatchAnswerRequest, BatchEvaluationResponse, EvaluationHistoryResponse,
    MultipleChoiceQuestion, GenerateMultipleChoiceResponse, MultipleChoiceQuestionResponse,
    MultipleChoiceAnswerRequest, MultipleChoiceBatchAnswerRequest,
    MultipleChoiceBatchEvaluationResponse, MultipleChoiceEvaluationDetail, EvaluationGroupResponse,
    TestResponse, TestListResponse
)

def distribute_questions_randomly(total_questions: int, num_items: int) -> List[int]:
    """
    Distribute questions randomly and evenly across knowledge items.
    
    Args:
        total_questions: Total number of questions to distribute
        num_items: Number of knowledge items to distribute across
        
    Returns:
        List of question counts for each item (index corresponds to item index)
    """
    print("total_questions: ", total_questions, "num_items: ", num_items)
    if num_items == 0:
        return []
    
    if total_questions <= num_items:
        # If we have fewer questions than items, randomly select items
        distribution = [0] * num_items
        selected_indices = random.sample(range(num_items), total_questions)
        for idx in selected_indices:
            distribution[idx] = 1
        return distribution
    
    # Calculate base questions per item and remaining questions
    base_questions = total_questions // num_items
    remaining_questions = total_questions % num_items
    
    # Start with base distribution
    distribution = [base_questions] * num_items
    
    # Randomly distribute remaining questions
    if remaining_questions > 0:
        # Get random indices to add extra questions to
        extra_indices = random.sample(range(num_items), remaining_questions)
        for idx in extra_indices:
            distribution[idx] += 1
    
    # Shuffle the distribution to make it more random
    random.shuffle(distribution)
    
    return distribution

# Create router
router = APIRouter(
    prefix="/api/self-test",
    tags=["Knowledge Assessment"]
)

# Free Text Question Endpoints
@router.post("/free-text/generate", response_model=GenerateQuestionsResponse)
async def generate_free_text_questions_endpoint(
    num_questions: int = Query(default=3, ge=1, le=10, description="Total number of questions to generate"),
    main_category: Optional[str] = Query(default=None, description="Main category to filter by")
):
    """
    Generate free text questions from knowledge base.
    Creates a new test record and returns questions with test_id.
    Questions are distributed evenly and randomly across knowledge items.
    
    Args:
        num_questions: Total number of questions to generate (1-10)
        main_category: Optional main category filter
        
    Returns:
        List of generated questions with test_id
    """
    try:
        # Calculate total possible score (each free text question is worth 5 points)
        total_possible_score = num_questions * 5
        
        # Create a new test record
        test_record = supabase_manager.create_test(
            category=main_category or "All Categories",
            total_score=total_possible_score
        )
        if not test_record:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create test record"
            )
        
        # Get knowledge items - get more than needed to allow for random distribution
        knowledge_items = supabase_manager.get_random_knowledge_items(
            count=num_questions,
            main_category=main_category
        )
        
        if not knowledge_items:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No knowledge items found matching the criteria"
            )
            
        # Randomly shuffle knowledge items for better distribution
        random.shuffle(knowledge_items)
        
        # Use random distribution for questions
        total_questions = num_questions
        num_items = len(knowledge_items)
        question_distribution = distribute_questions_randomly(total_questions, num_items)
        
        all_questions = []
        questions_generated = 0
        
        for i, item in enumerate(knowledge_items):
            if questions_generated >= total_questions:
                break
                
            # Get questions for this item from distribution
            item_questions = question_distribution[i]
            
            if item_questions == 0:
                continue
                
            # Generate questions for this item
            questions = generate_free_text_questions(
                knowledge_content=item.get('content', ''),
                main_category=item.get('main_category', 'Unknown'),
                sub_category=item.get('sub_category', 'Unknown'),
                num_questions=item_questions
            )
            
            if questions:
                for q in questions:
                    all_questions.append(Question(
                        question_text=q['question_text'],
                        knowledge_id=item['id'],
                        knowledge_content=item.get('content', ''),
                        main_category=item.get('main_category', 'Unknown'),
                        sub_category=item.get('sub_category', 'Unknown')
                    ))
                    questions_generated += 1
                    
                    if questions_generated >= total_questions:
                        break
        
        # Shuffle all questions for random order
        random.shuffle(all_questions)
        
        return GenerateQuestionsResponse(
            questions=all_questions,
            total_questions=len(all_questions),
            test_id=test_record['id']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating questions: {str(e)}"
        )

@router.post("/free-text/evaluate", response_model=BatchEvaluationResponse)
async def evaluate_free_text_answers(request: BatchAnswerRequest):
    """
    Evaluate a batch of free text answers using Azure OpenAI.
    Updates mastery levels for each knowledge item based on evaluation.
    
    This endpoint:
    1. Retrieves knowledge items for each answer
    2. Uses Azure OpenAI to evaluate answers
    3. Calculates and updates mastery levels
    4. Updates test scores (each question scored 0-5, total possible 5 per question)
    5. Returns detailed evaluation feedback
    """
    try:
        evaluations = []
        total_score = 0
        total_possible = 0
        
        # Create evaluation groups
        evaluation_groups = supabase_manager.create_evaluation_groups(
            count=len(request.answers),
            test_id=request.test_id if request.test_id else None
        )
        
        for i, answer_request in enumerate(request.answers):
            # Get knowledge item to verify it exists and get current mastery
            knowledge_item = supabase_manager.get_knowledge_by_id(answer_request.knowledge_id)
            if not knowledge_item:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Knowledge item {answer_request.knowledge_id} not found"
                )
            
            knowledge_content = knowledge_item.get('content', '')
            current_mastery = knowledge_item.get('mastery', 0.0)
            
            # Evaluate the answer
            evaluation = evaluate_free_text_answer(
                question_text=answer_request.question_text,
                answer=answer_request.answer,
                knowledge_content=knowledge_content,
                main_category=knowledge_item.get('main_category', 'Unknown'),
                sub_category=knowledge_item.get('sub_category', 'Unknown')
            )
            
            # Store evaluation in database
            evaluation_data = {
                "knowledge_id": answer_request.knowledge_id,
                "question_text": answer_request.question_text,
                "answer_text": answer_request.answer,
                "score": evaluation['score'],
                "feedback": evaluation['feedback'],
                "correct_points": evaluation['correct_points'],
                "incorrect_points": evaluation['incorrect_points'],
                "question_type": QuestionType.FREE_TEXT,
                "evaluation_group_id": evaluation_groups[i]['id'] if evaluation_groups else None
            }
            
            stored_eval = supabase_manager.create_evaluation(evaluation_data)
            
            # Calculate mastery
            mastery_result = calculate_free_text_mastery(
                knowledge_content=knowledge_content,
                new_evaluation={
                    'question_text': answer_request.question_text,
                    'answer_text': answer_request.answer,
                    'score': evaluation['score'],
                    'feedback': evaluation['feedback'],
                    'correct_points': evaluation['correct_points'],
                    'incorrect_points': evaluation['incorrect_points'],
                    'question_type': QuestionType.FREE_TEXT
                },
                previous_evaluations=[],
                current_mastery=current_mastery
            )

            if stored_eval:
                # Update knowledge item mastery and explanation
                supabase_manager.update_mastery(
                    knowledge_id=answer_request.knowledge_id,
                    evaluation_id=stored_eval['id'] if stored_eval else None,
                    mastery=mastery_result['mastery'],
                    mastery_explanation=mastery_result['explanation']
                )

                knowledge_item['mastery'] = mastery_result['mastery']
                knowledge_item['mastery_explanation'] = mastery_result['explanation']

                # Create evaluation response
                evaluation_response = EvaluationResponse(
                    question_text=answer_request.question_text,
                    answer=answer_request.answer,
                    score=evaluation['score'],
                    feedback=evaluation['feedback'],
                    correct_points=evaluation['correct_points'],
                    incorrect_points=evaluation['incorrect_points'],
                    evaluation_id=stored_eval['id'] if stored_eval else None,
                    knowledge_id=answer_request.knowledge_id,
                    mastery=mastery_result['mastery'],
                    previous_mastery=current_mastery,
                    mastery_explanation=mastery_result['explanation'],
                    sample_answer=evaluation.get('sample_answer', ''),
                    is_correct=None,  # Not applicable for free text
                    multiple_choice_question_id=None,  # Not applicable for free text
                    main_category=knowledge_item.get('main_category', 'Unknown'),
                    sub_category=knowledge_item.get('sub_category', 'Unknown')
                )
                
                evaluations.append(evaluation_response)
            
            # After evaluation is complete, update scores
            total_score += evaluation['score']  # Score is already 0-5
            total_possible += 5  # Each free text question is worth 5 points
        
        # Update test scores if test_id is provided
        if request.test_id:
            supabase_manager.update_test_scores(
                test_id=request.test_id,
                score=total_score
            )
        
        return BatchEvaluationResponse(
            evaluations=evaluations,
            total_evaluated=len(evaluations),
            test_id=request.test_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluating answers: {str(e)}"
        )

# Multiple Choice Question Endpoints
@router.post("/multiple-choice/generate", response_model=GenerateMultipleChoiceResponse)
async def generate_multiple_choice_questions(
    num_questions: int = Query(default=3, ge=1, le=10, description="Total number of questions to generate"),
    main_category: Optional[str] = Query(default=None, description="Main category to filter by")
):
    """
    Generate multiple choice questions from knowledge base.
    Creates a new test record and returns questions with test_id.
    Questions are distributed evenly and randomly across knowledge items.
    
    Args:
        num_questions: Total number of questions to generate (1-10)
        main_category: Optional main category filter
        
    Returns:
        List of generated multiple choice questions with test_id
    """
    try:
        # Calculate total possible score (each multiple choice question is worth 1 point)
        total_possible_score = num_questions
        
        # Create a new test record
        test_record = supabase_manager.create_test(
            category=main_category or "All Categories",
            total_score=total_possible_score
        )
        if not test_record:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create test record"
            )
        
        # Get knowledge items - get more than needed to allow for random distribution
        knowledge_items = supabase_manager.get_random_knowledge_items(
            count=num_questions,  # Get more items than needed for better distribution
            main_category=main_category
        )
        
        if not knowledge_items:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No knowledge items found matching the criteria"
            )
            
        # Randomly shuffle knowledge items for better distribution
        random.shuffle(knowledge_items)
        
        # Use random distribution for questions
        total_questions = num_questions
        num_items = len(knowledge_items)
        question_distribution = distribute_questions_randomly(total_questions, num_items)
        
        # Generate questions and prepare for batch creation
        questions_to_create = []
        generated_questions = []
        questions_generated = 0

        print("Distribution: ", question_distribution)
        
        for i, item in enumerate(knowledge_items):
            if questions_generated >= total_questions:
                break
                
            # Get questions for this item from distribution
            item_questions = question_distribution[i]
            
            if item_questions == 0:
                continue
                
            # Generate questions for this item
            questions = generate_multiple_choice_questions_batch(
                knowledge_content=item.get('content', ''),
                main_category=item.get('main_category', 'Unknown'),
                sub_category=item.get('sub_category', 'Unknown'),
                num_questions=item_questions
            )
            
            if questions:
                for q in questions:
                    # Add to batch creation list
                    questions_to_create.append({
                        'question_text': q['question_text'],
                        'options': q['options'],
                        'correct_answer_index': q['correct_answer_index'],
                        'explanation': q['explanation'],
                        'knowledge_id': item['id']
                    })
                    
                    # Keep track of generated questions and their knowledge items
                    generated_questions.append((q, item))
                    questions_generated += 1
                    
                    if questions_generated >= total_questions:
                        break
        
        # Batch create questions in database
        stored_questions = supabase_manager.create_multiple_choice_questions(questions_to_create)
        
        # Create response objects with stored question IDs
        all_questions = []
        for i, (question, item) in enumerate(generated_questions):
            if i < len(stored_questions):  # Safety check
                stored_question = stored_questions[i]
                all_questions.append(MultipleChoiceQuestion(
                    question_text=question['question_text'],
                    options=question['options'],
                    knowledge_id=item['id'],
                    knowledge_content=item.get('content', ''),
                    main_category=item.get('main_category', 'Unknown'),
                    sub_category=item.get('sub_category', 'Unknown'),
                    question_id=stored_question['id']
                ))
        
        # Shuffle all questions for random order
        random.shuffle(all_questions)
        
        return GenerateMultipleChoiceResponse(
            questions=all_questions,
            total_questions=len(all_questions),
            test_id=test_record['id']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating questions: {str(e)}"
        )

@router.post("/multiple-choice/evaluate", response_model=MultipleChoiceBatchEvaluationResponse)
async def evaluate_multiple_choice_answers(request: MultipleChoiceBatchAnswerRequest):
    """
    Evaluate a batch of multiple choice answers.
    Updates mastery levels for each knowledge item based on evaluation.
    
    This endpoint:
    1. Retrieves questions and knowledge items
    2. Evaluates answers against correct answers
    3. Calculates and updates mastery levels
    4. Updates test scores (each correct answer = 1 point, wrong = 0, total possible = 1 per question)
    5. Returns evaluation results
    """
    try:
        # Get all question IDs and knowledge IDs
        question_ids = [answer.question_id for answer in request.answers]
        knowledge_ids = [answer.knowledge_id for answer in request.answers]
        
        # Get questions and knowledge items
        questions = supabase_manager.get_multiple_choice_questions(question_ids)
        knowledges = supabase_manager.load_knowledge_by_ids(knowledge_ids)
        
        if not questions or not knowledges:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Questions or knowledge items not found"
            )
            
        # Create maps for easier lookup
        question_map = {q['id']: q for q in questions}
        knowledge_map = {k['id']: k for k in knowledges}
        
        # Get previous evaluations for mastery calculation
        knowledge_evaluations = supabase_manager.get_evaluations_by_knowledge_ids(knowledge_ids)
        knowledge_evaluations_map = {}
        for evaluation in knowledge_evaluations:
            if evaluation['knowledge_id'] not in knowledge_evaluations_map:
                knowledge_evaluations_map[evaluation['knowledge_id']] = []
            knowledge_evaluations_map[evaluation['knowledge_id']].append(evaluation)
        
        # Create evaluation groups
        evaluation_groups = supabase_manager.create_evaluation_groups(
            count=len(request.answers),
            test_id=request.test_id
        )
        
        # Process each answer
        all_evaluations = []
        total_score = 0
        total_possible = len(request.answers)  # Each multiple choice is worth 1 point
        
        for i, answer_request in enumerate(request.answers):
            question = question_map.get(answer_request.question_id)
            knowledge_item = knowledge_map.get(answer_request.knowledge_id)
            
            if not question or not knowledge_item:
                continue
                
            # Check if answer is correct
            is_correct = answer_request.selected_index == question['correct_answer_index']
            if is_correct:
                total_score += 1  # 1 point for correct answer, 0 for wrong
            
            # Format answer text for evaluation
            answer_text = f"""Question: {question['question_text']}
Selected Answer: {question['options'][answer_request.selected_index]}
Correct Answer: {question['options'][question['correct_answer_index']]}
Is Correct: {is_correct}"""
            
            # Get previous evaluations for this knowledge item
            previous_evaluations = knowledge_evaluations_map.get(answer_request.knowledge_id, [])
            
            # Calculate mastery
            current_mastery = knowledge_item.get('mastery', 0.0)
            mastery_result = calculate_multiple_choice_mastery(
                knowledge_content=knowledge_item['content'],
                new_evaluation={
                    'question_text': question['question_text'],
                    'answer_text': answer_text,
                    'is_correct': is_correct,
                    'question_type': QuestionType.MULTIPLE_CHOICE
                },
                previous_evaluations=previous_evaluations,
                current_mastery=current_mastery
            )
            
            # Store evaluation
            stored_eval = supabase_manager.create_evaluation({
                'knowledge_id': answer_request.knowledge_id,
                'question_text': question['question_text'],
                'answer_text': answer_text,
                'feedback': question['explanation'],
                'is_correct': is_correct,
                'score': 1 if is_correct else 0,  # Store binary score in database
                'question_type': QuestionType.MULTIPLE_CHOICE,
                'evaluation_group_id': evaluation_groups[i]['id'] if evaluation_groups else None,
                'multiple_choice_question_id': question['id'],
                'correct_answer_index': question['correct_answer_index']
            })
            
            if stored_eval:
                # Update mastery
                supabase_manager.update_mastery(
                    knowledge_id=answer_request.knowledge_id,
                    evaluation_id=stored_eval['id'],
                    mastery=mastery_result['mastery'],
                    mastery_explanation=mastery_result['explanation']
                )
                
                # Create evaluation response
                evaluation = MultipleChoiceEvaluationDetail(
                    question_text=question['question_text'],
                    options=question['options'],
                    selected_index=answer_request.selected_index,
                    correct_answer_index=question['correct_answer_index'],
                    is_correct=is_correct,
                    feedback=question['explanation'],
                    evaluation_id=stored_eval['id'],
                    knowledge_id=answer_request.knowledge_id,
                    multiple_choice_question_id=question['id'],
                    mastery=mastery_result['mastery'],
                    previous_mastery=current_mastery,
                    mastery_explanation=mastery_result['explanation'],
                    main_category=knowledge_item.get('main_category', 'Unknown'),
                    sub_category=knowledge_item.get('sub_category', 'Unknown')
                )
                all_evaluations.append(evaluation)
        
        # Update test scores if test_id is provided
        if request.test_id:
            supabase_manager.update_test_scores(
                test_id=request.test_id,
                score=total_score
            )
        
        return MultipleChoiceBatchEvaluationResponse(
            evaluations=all_evaluations,
            total_evaluated=len(all_evaluations),
            test_id=request.test_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluating answers: {str(e)}"
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
        List of evaluations grouped by evaluation_group_id
    """
    try:
        # Get knowledge item to verify it exists and get current mastery
        knowledge_item = supabase_manager.get_knowledge_by_id(knowledge_id)
        if not knowledge_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge item {knowledge_id} not found"
            )
        
        # Get evaluations for this knowledge item with evaluation group info
        eval_result = supabase_manager.supabase.table('evaluations')\
            .select('*, evaluation_group_id')\
            .eq('knowledge_id', knowledge_id)\
            .order('created_at', desc=True)\
            .limit(limit)\
            .execute()
            
        if not eval_result.data:
            return EvaluationHistoryResponse(evaluation_groups=[])
            
        # Group evaluations by evaluation_group_id
        grouped_evaluations = {}
        total_score = 0
        
        for eval_data in eval_result.data:
            group_id = eval_data.get('evaluation_group_id')
            if not group_id:
                continue
                
            created_at = datetime.fromisoformat(eval_data.get('created_at'))
            
            # Create evaluation response
            evaluation = EvaluationResponse(
                question_text=eval_data.get('question_text', ''),
                answer=eval_data.get('answer_text', ''),
                score=eval_data.get('score'),
                feedback=eval_data.get('feedback', ''),
                correct_points=eval_data.get('correct_points', []),
                incorrect_points=eval_data.get('incorrect_points', []),
                evaluation_id=eval_data.get('id'),
                knowledge_id=eval_data.get('knowledge_id'),
                mastery=eval_data.get('mastery', 0.0),
                previous_mastery=eval_data.get('previous_mastery', 0.0),
                mastery_explanation=eval_data.get('mastery_explanation', ''),
                sample_answer=eval_data.get('sample_answer', ''),
                is_correct=eval_data.get('is_correct'),
                multiple_choice_question_id=eval_data.get('multiple_choice_question_id'),
                main_category=eval_data.get('main_category', 'Unknown'),
                sub_category=eval_data.get('sub_category', 'Unknown'),
                correct_answer_index=eval_data.get('correct_answer_index')
            )
            
            if group_id not in grouped_evaluations:
                grouped_evaluations[group_id] = EvaluationGroupResponse(
                    evaluation_group_id=group_id,
                    created_at=created_at,
                    evaluations=[],
                    mastery=eval_data.get('mastery', 0.0),
                    mastery_explanation=eval_data.get('mastery_explanation', '')
                )
            grouped_evaluations[group_id].evaluations.append(evaluation)
            
            if evaluation.score is not None:
                total_score += evaluation.score
        
        #sort by created_at
        evaluation_groups = sorted(
            grouped_evaluations.values(),
            key=lambda x: x.created_at
        )

        return EvaluationHistoryResponse(
            evaluation_groups=evaluation_groups,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving evaluations: {str(e)}"
        )

@router.get("/tests", response_model=TestListResponse)
async def get_latest_tests(
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of tests to return"),
    category: Optional[str] = Query(default=None, description="Optional category to filter by")
):
    """
    Get the latest tests with their scores.
    
    Args:
        limit: Maximum number of tests to return (1-50)
        category: Optional category to filter by
        
    Returns:
        List of tests with their scores and metadata
    """
    try:
        # Get latest tests from database
        tests = supabase_manager.get_latest_tests(limit=limit, category=category)
        if not tests:
            return TestListResponse(tests=[], total_tests=0)
        
        # Format response
        test_responses = []
        for test in tests:
            # Calculate percentage score
            percentage = (test['score'] / test['total_score'] * 100) if test['total_score'] > 0 else 0
            
            test_response = TestResponse(
                id=test['id'],
                category=test['category'],
                score=test['score'],
                total_score=test['total_score'],
                percentage=round(percentage, 2),
                created_at=test['created_at'],
                updated_at=test.get('updated_at') or test['created_at']
            )
            test_responses.append(test_response)
        
        return TestListResponse(
            tests=test_responses,
            total_tests=len(test_responses)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching tests: {str(e)}"
        )