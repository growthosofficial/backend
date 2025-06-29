"""
Self-test generation and assessment logic for the Second Brain Knowledge Management System.
"""
from typing import List, Dict, Optional
import random
import json
import time
from fastapi import APIRouter, HTTPException, status, Query, Path

from database.supabase_manager import supabase_manager
from src.core.self_test import (
    generate_free_text_question, 
    generate_multiple_choice_question,
    evaluate_free_text_answer,
    evaluate_multiple_choice_answers,
    calculate_free_text_mastery,
    calculate_multiple_choice_mastery
)
from models import (
    QuestionType, Question, GenerateQuestionsResponse, AnswerRequest, EvaluationResponse,
    BatchAnswerRequest, BatchEvaluationResponse, EvaluationHistoryResponse,
    MultipleChoiceQuestion, GenerateMultipleChoiceResponse, MultipleChoiceQuestionResponse,
    MultipleChoiceAnswerRequest, MultipleChoiceBatchAnswerRequest,
    MultipleChoiceBatchEvaluationResponse, MultipleChoiceEvaluationDetail
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
    Questions will be distributed evenly across randomly selected knowledge items.
    Questions will be evaluated by ChatGPT later.
    
    This endpoint:
    1. Retrieves random knowledge items from the database
    2. Distributes total questions evenly across selected items
    3. Uses Azure OpenAI to generate thought-provoking questions
    4. Returns questions with their knowledge_id for later evaluation
    
    Args:
        num_questions: Total number of questions to generate (1-20)
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
        
        # Calculate how many knowledge items to use based on total questions
        # Use at most num_questions items, but at least num_questions/5 items
        # This ensures 1-5 questions per item while using as many items as reasonable
        min_items = max(1, num_questions // 5)  # At most 5 questions per item
        max_items = min(num_questions, len(valid_items))  # At least 1 question per item
        num_items = random.randint(min_items, max_items)
        
        # Randomly select items to generate questions from
        selected_items = random.sample(valid_items, num_items)
        
        # Distribute questions across items
        # First, give each item the minimum number of questions
        questions_per_item = [num_questions // num_items] * num_items
        # Then distribute remaining questions randomly
        remaining = num_questions - sum(questions_per_item)
        if remaining > 0:
            # Randomly select items to receive extra questions
            lucky_items = random.sample(range(num_items), remaining)
            for idx in lucky_items:
                questions_per_item[idx] += 1
        
        # Generate questions for each selected item
        questions = []
        
        for item, num_item_questions in zip(selected_items, questions_per_item):
            # Get categories and content
            item_main_category = item.get('main_category', 'Unknown')
            sub_category = item.get('sub_category', 'Unknown')
            content = item.get('content', '')
            knowledge_id = item['id']  # Already validated as positive integer
            
            # Generate questions for this item - pass num_questions directly to prompt
            questions_data = generate_free_text_question(
                category=item_main_category,
                content=content,
                knowledge_id=knowledge_id,
                num_questions=num_item_questions  # Pass the calculated number for this item
            )
            
            if questions_data:
                # Process each question in the returned list
                for question_data in questions_data:
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
        
        # Shuffle all questions before returning
        random.shuffle(questions)
        
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
    Evaluate a batch of free text answers and calculate mastery levels
    
    Args:
        request: BatchAnswerRequest containing list of answers to evaluate
        
    Returns:
        BatchEvaluationResponse with evaluation results and mastery updates
    """
    try:
        evaluations = []

        knowledge_ids = [answer.knowledge_id for answer in request.answers]
        knowledges = supabase_manager.load_knowledge_by_ids(knowledge_ids)
        knowledge_map = {item['id']: item for item in knowledges}
        knowledge_evaluations = supabase_manager.get_evaluations_by_knowledge_ids(knowledge_ids)
        knowledge_evaluations_map = {}

        for evaluation in knowledge_evaluations:
            if evaluation['knowledge_id'] not in knowledge_evaluations_map:
                knowledge_evaluations_map[evaluation['knowledge_id']] = []
            knowledge_evaluations_map[evaluation['knowledge_id']].append(evaluation)

        evaluation_groups = supabase_manager.create_evaluation_groups(len(request.answers))

        for i, answer_request in enumerate(request.answers):
            knowledge_id = answer_request.knowledge_id
            
            knowledge_result = knowledge_map.get(knowledge_id)
            if not knowledge_result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Knowledge item {knowledge_id} not found"
                )

            previous_evaluations = knowledge_evaluations_map.get(knowledge_id)
            if not previous_evaluations:
                previous_evaluations = []

            knowledge_content = knowledge_result.get('content', '')
            current_mastery = knowledge_result.get('mastery', 0.0)
            
            # Evaluate the answer
            evaluation = evaluate_free_text_answer(
                question_text=answer_request.question_text,
                answer=answer_request.answer,
                knowledge_content=knowledge_content,
                main_category=knowledge_result.get('main_category', 'Unknown'),
                sub_category=knowledge_result.get('sub_category', 'Unknown')
            )
            
            # Store evaluation in database
            evaluation_data = {
                "knowledge_id": knowledge_id,
                "question_text": answer_request.question_text,
                "answer_text": answer_request.answer,
                "score": evaluation['score'],
                "feedback": evaluation['feedback'],
                "correct_points": evaluation['correct_points'],
                "incorrect_points": evaluation['incorrect_points'],
                "question_type": QuestionType.FREE_TEXT,
                "sample_answer": evaluation.get('sample_answer'),
                "evaluation_group_id": evaluation_groups[i]['id']
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
                previous_evaluations=previous_evaluations,
                current_mastery=current_mastery
            )

            if stored_eval:
                previous_evaluations.append(stored_eval)
            
            # Update knowledge item mastery and explanation
            supabase_manager.update_mastery(
                knowledge_id=knowledge_id,
                evaluation_id=stored_eval['id'] if stored_eval else None,
                mastery=mastery_result['mastery'],
                mastery_explanation=mastery_result['explanation']
            )

            knowledge_map[knowledge_id]['mastery'] = mastery_result['mastery']
            knowledge_map[knowledge_id]['mastery_explanation'] = mastery_result['explanation']

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
                previous_mastery=current_mastery,
                mastery_explanation=mastery_result['explanation'],
                sample_answer=evaluation.get('sample_answer'),
                is_correct=None,  # Not applicable for free text
                multiple_choice_question_id=None  # Not applicable for free text
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
    num_questions: int = Query(default=3, ge=1, le=20, description="Total number of questions to generate"),
    main_category: str | None = Query(default=None, description="Optional main category to filter questions by")
):
    """
    Generate multiple choice questions from the knowledge base.
    Questions will be distributed evenly across randomly selected knowledge items.
    Questions will be evaluated by ChatGPT later.
    
    This endpoint:
    1. Retrieves random knowledge items from the database
    2. Distributes total questions evenly across selected items
    3. Uses Azure OpenAI to generate thought-provoking multiple choice questions
    4. Returns questions with their knowledge_id for later evaluation
    
    Args:
        num_questions: Total number of questions to generate (1-20)
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
        
        # Calculate how many knowledge items to use based on total questions
        # Use at most num_questions items, but at least num_questions/5 items
        # This ensures 1-5 questions per item while using as many items as reasonable
        min_items = max(1, num_questions // 5)  # At most 5 questions per item
        max_items = min(num_questions, len(valid_items))  # At least 1 question per item
        num_items = random.randint(min_items, max_items)
        
        # Randomly select items to generate questions from
        selected_items = random.sample(valid_items, num_items)
        
        # Distribute questions across items
        # First, give each item the minimum number of questions
        questions_per_item = [num_questions // num_items] * num_items
        # Then distribute remaining questions randomly
        remaining = num_questions - sum(questions_per_item)
        if remaining > 0:
            # Randomly select items to receive extra questions
            lucky_items = random.sample(range(num_items), remaining)
            for idx in lucky_items:
                questions_per_item[idx] += 1

        # Generate questions for each selected item
        all_questions = []
        
        for item, num_item_questions in zip(selected_items, questions_per_item):
            # Get categories and content
            item_main_category = item.get('main_category', 'Unknown')
            item_sub_category = item.get('sub_category', 'Unknown')
            content = item.get('content', '')
            knowledge_id = item['id']  # Already validated as positive integer
            
            # Generate multiple choice questions using our dedicated function
            questions_data = generate_multiple_choice_question(
                category=item_main_category,
                content=content,
                knowledge_id=knowledge_id,
                num_questions=num_item_questions
            )
            
            if questions_data:
                # Create response objects for each question
                for q in questions_data:
                    # Ensure question_id is an integer, skip if not available
                    question_id = q.get('question_id')
                    if question_id is None:
                        continue
                        
                    question = MultipleChoiceQuestionResponse(
                        question_id=question_id,
                        question_text=q['question_text'],
                        options=q['options'],
                        knowledge_id=knowledge_id,
                        selected_answer_index=0,
                        main_category=item_main_category,
                        sub_category=item_sub_category
                    )
                    all_questions.append(question)
        
        if not all_questions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate any valid multiple choice questions"
            )
        
        return GenerateMultipleChoiceResponse(
            questions=all_questions,
            total_questions=len(all_questions)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating multiple choice questions: {str(e)}"
        )

@router.post("/multiple-choice/evaluate", response_model=MultipleChoiceBatchEvaluationResponse)
async def evaluate_multiple_choice_answers_endpoint(request: MultipleChoiceBatchAnswerRequest):
    """
    Evaluate a batch of multiple choice answers and calculate mastery levels
    
    Args:
        request: MultipleChoiceBatchAnswerRequest containing list of answers to evaluate
        
    Returns:
        MultipleChoiceBatchEvaluationResponse with evaluation results and mastery updates
    """
    try:
        # Get all question IDs
        question_ids = [answer.question_id for answer in request.answers]
        
        # Get questions from database
        questions = supabase_manager.get_multiple_choice_questions(question_ids)
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No questions found for the provided IDs"
            )
            
        # Index questions by ID for easy lookup
        questions_map = {q['id']: q for q in questions}
        
        # Group answers by knowledge ID for efficient evaluation
        knowledge_groups = {}
        for answer in request.answers:
            question = questions_map.get(answer.question_id)
            if not question:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Question {answer.question_id} not found"
                )
            
            knowledge_id = question['knowledge_id']
            if knowledge_id not in knowledge_groups:
                knowledge_groups[knowledge_id] = []
            
            knowledge_groups[knowledge_id].append({
                'question': question,
                'selected_index': answer.selected_answer_index
            })
        
        # Get knowledge items for each group
        knowledge_items = supabase_manager.load_knowledge_by_ids(list(knowledge_groups.keys()))
        knowledge_map = {item['id']: item for item in knowledge_items}
        
        # Process each knowledge group
        all_evaluations = []
        
        count = len(knowledge_groups)
        evaluation_groups = supabase_manager.create_evaluation_groups(count)

        print(evaluation_groups)

        for i, (knowledge_id, answers) in enumerate(knowledge_groups.items()):
            knowledge_item = knowledge_map.get(knowledge_id)
            if not knowledge_item:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Knowledge item {knowledge_id} not found"
                )
            
            previous_evaluations = supabase_manager.get_evaluations(knowledge_id)
            
            # Format answers for evaluation
            answer_texts = []
            for answer in answers:
                question = answer['question']
                selected_index = answer['selected_index']
                is_correct = selected_index == question['correct_answer_index']
                
                answer_text = f"""
Question {i+1}: {question['question_text']}
Selected Answer: {question['options'][selected_index]}
Correct Answer: {question['options'][question['correct_answer_index']]}
Is Correct: {is_correct}
"""
                answer_texts.append(answer_text)
            
            # Join answers with a newline separator
            answers_text = "\n".join(answer_text.strip() for answer_text in answer_texts)
            
            # Get evaluation feedback for all answers in this group
            evaluation = evaluate_multiple_choice_answers(
                answers_text=answers_text,
                knowledge_content=knowledge_item.get('content', ''),
                main_category=knowledge_item.get('main_category', 'Unknown'),
                sub_category=knowledge_item.get('sub_category', 'Unknown')
            )
            
            # Calculate mastery based on all answers in the group
            mastery_result = calculate_multiple_choice_mastery(
                knowledge_content=knowledge_item.get('content', ''),
                new_evaluation={
                    'question_text': 'Multiple Choice Questions',
                    'answer_text': answers_text,
                    'feedback': evaluation['feedback'],
                    'question_type': QuestionType.MULTIPLE_CHOICE
                },
                previous_evaluations=previous_evaluations,
                current_mastery=knowledge_item.get('mastery', 0.0)
            )
            
            # Create evaluation records for each answer
            for answer, eval_result in zip(answers, evaluation['evaluations']):
                question = answer['question']
                selected_index = answer['selected_index']
                is_correct = selected_index == question['correct_answer_index']
                
                # Store evaluation in database
                evaluation_data = {
                    "knowledge_id": knowledge_id,
                    "question_text": question['question_text'],
                    "answer_text": question['options'][selected_index],
                    "feedback": eval_result['explanation'],  # Use specific explanation for this answer
                    "correct_points": [],  # No points for multiple choice
                    "incorrect_points": [],  # No points for multiple choice
                    "question_type": QuestionType.MULTIPLE_CHOICE,
                    "is_correct": is_correct,
                    "multiple_choice_question_id": question['id'],
                    "mastery": mastery_result['mastery'],  # Add mastery from the calculation
                    "mastery_explanation": mastery_result['explanation'],  # Add mastery explanation,
                    "correct_answer": question['options'][question['correct_answer_index']],
                    "evaluation_group_id": evaluation_groups[i]['id']
                }
                
                stored_eval = supabase_manager.create_evaluation(evaluation_data)
                
                # Create evaluation response
                eval_response = EvaluationResponse(
                    question_text=question['question_text'],
                    answer=question['options'][selected_index],
                    score=None,  # No score for multiple choice
                    feedback=eval_result['explanation'],  # Use specific explanation for this answer
                    correct_points=[],  # No points for multiple choice
                    incorrect_points=[],  # No points for multiple choice
                    knowledge_id=knowledge_id,
                    evaluation_id=stored_eval['id'] if stored_eval else None,
                    mastery=mastery_result['mastery'],
                    previous_mastery=knowledge_item.get('mastery', 0.0),  # Add previous mastery
                    mastery_explanation=mastery_result['explanation'],
                    sample_answer=None,  # No sample answer for multiple choice
                    is_correct=is_correct,
                    multiple_choice_question_id=question['id'],
                    # Add multiple choice specific fields
                    options=question['options'],
                    selected_index=selected_index,
                    correct_answer_index=question['correct_answer_index']
                )
                
                all_evaluations.append(eval_response)
            
            # Update knowledge item mastery
            supabase_manager.update_mastery(
                knowledge_id=knowledge_id,
                evaluation_id=None,  # No single evaluation ID for batch
                mastery=mastery_result['mastery'],
                mastery_explanation=mastery_result['explanation']
            )
        
        return MultipleChoiceBatchEvaluationResponse(
            evaluations=all_evaluations,
            total_evaluated=len(all_evaluations)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluating multiple choice answers: {str(e)}"
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
                score=eval_data.get('score'),  # May be None for multiple choice
                feedback=eval_data.get('feedback', ''),
                correct_points=eval_data.get('correct_points', []),
                incorrect_points=eval_data.get('incorrect_points', []),
                evaluation_id=eval_data.get('id'),
                knowledge_id=knowledge_id,
                mastery=eval_data.get('mastery', 0.0),
                previous_mastery=eval_data.get('previous_mastery', 0.0),  # Get previous mastery from evaluation record
                mastery_explanation=eval_data.get('mastery_explanation', ''),
                sample_answer=eval_data.get('sample_answer', None),
                is_correct=eval_data.get('is_correct'),
                multiple_choice_question_id=eval_data.get('multiple_choice_question_id')
            )
            evaluations.append(evaluation)
            # Only add score to total if it exists (free text answers)
            if evaluation.score is not None:
                total_score += evaluation.score
        
        # Calculate average score only for free text answers
        scored_evaluations = [e for e in evaluations if e.score is not None]
        average_score = total_score / len(scored_evaluations) if scored_evaluations else 0
        
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