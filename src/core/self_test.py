"""
Self-test generation and evaluation using Azure OpenAI for the Second Brain Knowledge Management System.
"""
import json
import time
import openai
from typing import Dict, Optional, List, Any
import os
from datetime import datetime
import random

from models import QuestionType
from config.settings import settings

def call_azure_openai(prompt: str, prompt_name: str) -> str:
    """
    Call Azure OpenAI API with error handling and retries
    
    Args:
        prompt: The prompt to send to the API
        prompt_name: Name of the prompt for logging
        
    Returns:
        Response text from the API
    """
    try:
        # Validate required settings
        if not settings.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY not found")
        if not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_ENDPOINT not found")
        if not settings.AZURE_OPENAI_API_VERSION:
            raise ValueError("AZURE_OPENAI_API_VERSION not found")
        if not settings.AZURE_OPENAI_DEPLOYMENT_NAME:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME not found")
        
        # Configure Azure OpenAI client
        client = openai.AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )
        
        response = client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates thought-provoking questions for knowledge assessment. Always respond with valid JSON in the specified format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        # create file into tmp/prompts with timestamp and prompt_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{prompt_name.replace(' ', '_')}.txt"
        filepath = os.path.join("tmp/prompts", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
            
        raise ValueError(f"No valid response from Azure OpenAI for {prompt_name}")
        
    except Exception as e:
        print(f"Error calling Azure OpenAI for {prompt_name}: {e}")
        raise

def format_evaluation_history(evaluations: List[Dict]) -> str:
    """
    Format evaluation history for mastery calculation in a structured way
    
    Args:
        evaluations: List of evaluation records
        
    Returns:
        Formatted evaluation history string with clear structure and separation
    """
    if not evaluations:
        return ""

    eval_history = ""
    for i, eval in enumerate(evaluations, 1):
        eval_type = eval.get('question_type')
        if eval_type == QuestionType.MULTIPLE_CHOICE:
            eval_history += f"""
Evaluation #{i} - Multiple Choice
Question: {eval.get('question_text', '')}
Your Answer: {eval.get('answer_text', '')}
Correct Answer: {eval.get('correct_answer', '')}
"""
        else:
            eval_history += f"""
Evaluation #{i} - Free Text
Question: {eval.get('question_text', '')}
Your Answer: {eval.get('answer_text', '')}
Feedback: {eval.get('feedback', '')}
"""

    return eval_history

def generate_free_text_questions(
    knowledge_content: str,
    main_category: str,
    sub_category: str,
    num_questions: int = 3
) -> Optional[List[Dict[str, Any]]]:
    """
    Generate multiple free text questions from knowledge content using Azure OpenAI
    
    Args:
        knowledge_content: The knowledge content to generate questions from
        main_category: Main category of the knowledge
        sub_category: Sub category of the knowledge
        num_questions: Number of questions to generate (default: 3)
        
    Returns:
        List of dictionaries, each containing question text and sample answer
    """
    try:
        prompt = f"""You are a knowledgeable tutor creating test questions.
Generate {num_questions} thought-provoking questions based on this knowledge content.
Each question should test deep understanding and require detailed explanation.

REQUIREMENTS FOR EACH QUESTION:
1. Focus on the most important concepts in order of relevance
2. Combine multiple related concepts when they naturally fit together
3. Skip less important details if including them would make the question too complex
4. Keep questions focused and concise while still being thought-provoking
5. Should require explanation, analysis, or application of concepts
6. Should not be answerable with just a single word or simple fact
7. Should encourage connecting ideas and demonstrating understanding
8. Each question should be answerable in about 3-5 sentences
9. Questions should be diverse and cover different aspects of the content
10. Avoid redundant or very similar questions

Main Category: {main_category}
Sub Category: {sub_category}
Knowledge Content:
{knowledge_content}

Respond with valid JSON only, in this format:
{{
    "questions": [
        {{
            "question_text": "First thought-provoking question here...",
            "sample_answer": "A detailed sample answer that would get full marks"
        }},
        {{
            "question_text": "Second thought-provoking question here...",
            "sample_answer": "Another detailed sample answer"
        }},
        // ... more questions ...
    ]
}}"""

        response = call_azure_openai(prompt, "Free Text Questions Generation")
        result = json.loads(response)
        
        # Validate response format
        if not isinstance(result, dict) or 'questions' not in result:
            print("Error: Invalid response format - missing 'questions' array")
            return None
            
        questions = result['questions']
        if not isinstance(questions, list):
            print("Error: Invalid response format - 'questions' is not an array")
            return None
            
        # Validate each question
        valid_questions = []
        for q in questions:
            if not isinstance(q, dict) or 'question_text' not in q:
                print("Error: Invalid question format - missing 'question_text'")
                continue
                
            # Ensure sample_answer exists
            if 'sample_answer' not in q:
                q['sample_answer'] = ''
                
            valid_questions.append(q)
            
        return valid_questions if valid_questions else None
        
    except Exception as e:
        print(f"Error generating free text questions: {e}")
        return None

def evaluate_free_text_answer(question_text: str, answer: str, knowledge_content: str, main_category: str, sub_category: str) -> Dict:
    """
    Evaluate a user's free text answer to a question using Azure OpenAI
    
    Args:
        question_text: The original question text
        answer: The user's answer to evaluate
        knowledge_content: The knowledge content the question was based on
        main_category: The main knowledge category (e.g. Physics)
        sub_category: The sub category (e.g. Quantum Mechanics)
        
    Returns:
        Dictionary with evaluation results including score (0-5), feedback, etc.
    """
    # PRE-CHECK: Immediately score single words, numbers, or irrelevant responses as 0
    answer_trimmed = answer.strip()
    if (len(answer_trimmed) <= 3 or 
        answer_trimmed.isdigit() or 
        answer_trimmed.lower() in ['yes', 'no', 'ok', 'hi', 'hello', 'test', '1', '2', '3', '4', '5'] or
        len(answer_trimmed.split()) <= 1):
        return {
            "score": 0,
            "feedback": "Your answer is too brief and does not address the question. Please provide a detailed explanation that demonstrates your understanding of the topic.",
            "correct_points": [],
            "incorrect_points": ["Answer too short", "No attempt to address the question", "Single word/number response"],
            "sample_answer": ""
        }

    prompt_template = '''You are a VERY STRICT evaluator. You MUST follow these rules exactly.

Main Category: {main_category}
Sub Category: {sub_category}
Question: {question}
Answer: {answer}
Reference Knowledge Content: {knowledge}

CRITICAL - Initial Checks:
1. If the answer is irrelevant, off-topic, or just a greeting/single word, score it 0 immediately
2. If the answer shows no attempt to address the question's specific points, score it 0
3. Only proceed with detailed evaluation if the answer makes a genuine attempt to address the question

CRITICAL - Answer Completeness Check:
1. First identify ALL parts of what was asked in the question
2. Check if EACH part was properly addressed in the answer
3. Example: If question asks "explain X and give example", both parts must be present
4. Missing ANY major part of the question should result in score ≤ 2

Score the answer on a scale of 0-5 using these strict guidelines:
0 = Any of these conditions:
   - Not understood / Incorrect / Completely off-topic
   - Single word or greeting only
   - No attempt to address the question
   - Irrelevant response
1 = Very basic attempt but mostly incorrect or missing key points
2 = Basic understanding but significant parts missing or incorrect
3 = Moderate understanding, some key parts missing or superficial
4 = Good understanding with minor gaps or imperfections
5 = ONLY if ALL of these are true:
   - ALL parts of the question were fully addressed
   - ALL explanations are thorough and accurate
   - ALL requested examples/applications provided
   - Shows deep understanding beyond basic facts

CRITICAL: The answer "{answer}" appears to be very short. If it's a single word, number, or doesn't address the question, you MUST score it 0.

Format your response as a JSON object with this structure:
{{
    "score": <0-5>,
    "feedback": "Clear explanation of strengths/weaknesses. Use you to refer to the user.",
    "correct_points": ["Point 1", "Point 2"],
    "incorrect_points": ["Missing/wrong point 1", "Missing/wrong point 2"],
    "sample_answer": "What a good answer would look like"
}}'''

    try:
        # Format prompt with actual content
        prompt = prompt_template.format(
            question=question_text,
            answer=answer,
            knowledge=knowledge_content,
            main_category=main_category,
            sub_category=sub_category
        )
        
        # Get response from Azure OpenAI
        response = call_azure_openai(prompt, "Free Text Answer Evaluation")
        
        # Parse response
        result = json.loads(response)
        
        # Validate evaluation response
        required_fields = ["score", "feedback", "correct_points", "incorrect_points", "sample_answer"]
        for field in required_fields:
            if field not in result:
                print(f"Error: Response missing {field} field")
                return {
                    "score": 0,
                    "feedback": "Error in evaluation response",
                    "correct_points": [],
                    "incorrect_points": ["Error processing response"],
                    "sample_answer": ""
                }
        
        # Ensure score is integer 0-5
        result["score"] = max(0, min(5, int(result["score"])))
        return result
        
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {e}")
        return {
            "score": 0,
            "feedback": f"Error processing response: {str(e)}",
            "correct_points": [],
            "incorrect_points": ["Error processing response"],
            "sample_answer": ""
        }
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {
            "score": 0,
            "feedback": f"Error: {str(e)}",
            "correct_points": [],
            "incorrect_points": ["Error processing response"],
            "sample_answer": ""
        }

def calculate_free_text_mastery(knowledge_content: str, new_evaluation: Dict, previous_evaluations: List[Dict], current_mastery: float) -> Dict:
    """
    Calculate mastery level using LLM by analyzing evaluation history
    
    Args:
        knowledge_content: The full content of the knowledge item
        new_evaluation: The current evaluation being analyzed
        previous_evaluations: List of previous evaluation records
        current_mastery: Current mastery level of the knowledge item
        
    Returns:
        Dictionary with mastery level (0-1) and explanation
    """
    prompt_template = '''Assess understanding and provide feedback. Respond with valid JSON only.

Current Mastery: {current_mastery}

CRITICAL MASTERY GUIDELINES:
1. Free Text Answer Evaluation:
   - Good explanations increase mastery (0.1-0.2 increase)
   - Poor answers decrease mastery (0.1-0.2 decrease)
   - "I don't know" responses significantly decrease mastery (0.2-0.3 decrease)
   - Partial understanding can be recognized in free text answers

2. Multiple Choice History Evaluation (if any):
   - Multiple choice answers are binary - either fully correct or wrong
   - NO partial credit for wrong multiple choice answers
   - Each wrong multiple choice decreases mastery by 0.1-0.2
   - Each correct multiple choice increases mastery by 0.05-0.1
   - Multiple consecutive correct answers needed to show mastery
   - A single wrong answer indicates gaps in understanding

3. General Rules:
   - Recent performance has more weight than older answers
   - Free text answers have more impact than multiple choice
   - Consider answer quality and depth of explanation
   - Look for patterns of understanding vs. misconceptions

Knowledge Content:
{knowledge_content}

Question Type: {question_type}
Question: {new_eval_question}
Your Answer: {new_eval_answer}

Previous Answers (newest to oldest):
{evaluation_history}

Response Format:
{{
    "mastery": <float 0-1>,
    "explanation": "2-3 sentences describing: 1) How well you understand this topic 2) What you've demonstrated good knowledge of 3) What you need to work on"
}}

Example responses:
"Your free text answer shows good understanding of database concepts, but your multiple choice history reveals gaps in normalization forms. While you can explain relationships well, the pattern of wrong multiple choice answers about normalization indicates areas needing review."

"Your detailed explanation of neural networks demonstrates strong understanding. However, recent multiple choice answers about activation functions were incorrect. Focus on connecting your theoretical knowledge with specific technical details."

"Your answers show declining mastery. The current 'I don't know' response and previous wrong multiple choice answers suggest fundamental gaps in understanding. You need to review the basic concepts before proceeding."'''

    try:
        # Format evaluation history
        eval_history = format_evaluation_history(previous_evaluations)
        
        # Format the current answer based on type
        current_answer = new_evaluation.get('answer_text', '')
        
        # Format prompt with actual content
        prompt = prompt_template.format(
            knowledge_content=knowledge_content,
            current_mastery=current_mastery,
            question_type=new_evaluation.get('question_type', QuestionType.FREE_TEXT),
            new_eval_question=new_evaluation.get('question_text', ''),
            new_eval_answer=current_answer,
            evaluation_history=eval_history
        )
        
        # Get response from Azure OpenAI
        response = call_azure_openai(prompt, "Free Text Mastery Calculation")
        if not response:
            print("Empty response from Azure OpenAI")
            return {
                "mastery": max(0.0, current_mastery - 0.1),  # Slight decrease on empty response
                "explanation": "No response from LLM analysis. Mastery adjusted based on current performance."
            }
        
        # Parse response
        try:
            # Clean the response string
            response = response.strip()
            response = ''.join(char for char in response if ord(char) >= 32 or char in '\n\r\t')
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]
            response = response.replace('\\n', '')
            
            # Parse JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1:
                    json_str = response[start:end + 1].replace('\\"', '"')
                    result = json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found")
            
            # Validate and adjust mastery
            if "mastery" not in result or "explanation" not in result:
                raise ValueError("Missing required fields")
            
            result["mastery"] = float(result["mastery"])
            result["mastery"] = max(0, min(1, result["mastery"]))

            return result
            
        except Exception as e:
            print(f"Error processing response: {e}")
            print(f"Raw response: {response}")
            # Return a default mastery value based on current performance
            return {
                "mastery": max(0.0, current_mastery - 0.1),  # Slight decrease on parsing error
                "explanation": f"Error processing LLM response: {str(e)}. Mastery adjusted based on current performance."
            }
        
    except Exception as e:
        print(f"Error in mastery calculation: {e}")
        # Return a default mastery value instead of keeping current_mastery
        # This ensures we always have a valid mastery value
        return {
            "mastery": max(0.0, current_mastery - 0.1),  # Slight decrease on error
            "explanation": f"Error in LLM analysis: {str(e)}. Mastery adjusted based on current performance."
        }

def calculate_multiple_choice_mastery(knowledge_content: str, new_evaluation: Dict, previous_evaluations: List[Dict], current_mastery: float) -> Dict:
    """
    Calculate mastery level using LLM by analyzing multiple choice evaluation history
    
    Args:
        knowledge_content: The full content of the knowledge item
        new_evaluation: The current evaluation being analyzed
        previous_evaluations: List of previous evaluation records
        current_mastery: Current mastery level of the knowledge item
        
    Returns:
        Dictionary with mastery level (0-1) and explanation
    """
    prompt_template = '''Assess understanding and provide feedback for multiple choice answers. Respond with valid JSON only.

Current Mastery: {current_mastery}

CRITICAL MASTERY GUIDELINES:
1. Multiple choice answers are binary - either fully correct or wrong
2. NO partial credit for wrong answers, even if reasoning shows some understanding
3. Wrong answers should DECREASE mastery more significantly than correct answers increase it
4. Mastery calculation rules:
   - Each wrong answer should decrease mastery by 0.1-0.2
   - Each correct answer should increase mastery by 0.05-0.1
   - Multiple consecutive correct answers needed to show mastery
   - A single wrong answer indicates gaps in understanding
5. Recent performance has more weight than older answers
6. Free text answers (if any) should have more impact than multiple choice

Knowledge Content:
{knowledge_content}

Current Evaluation:
{current_evaluation}

Previous Answers (newest to oldest, free text weighs more than multiple choice):
{evaluation_history}

Response Format:
{{
    "mastery": <float 0-1>,
    "explanation": "2-3 sentences describing: 1) Pattern of multiple choice performance 2) Areas of demonstrated understanding 3) Specific concepts needing review"
}}

Example responses:
"Your multiple choice performance shows inconsistent understanding. While you correctly identified database relationships in 2 questions, the wrong answer about normalization forms indicates a fundamental gap. Focus on reviewing the specific criteria for different normalization forms."

"Recent answers show a pattern of incorrect responses about quantum mechanics principles. Each wrong answer suggests gaps in core understanding. You need to review the fundamental concepts before moving to more complex applications."

"Multiple choice results demonstrate solid grasp of machine learning basics. You've consistently answered correctly about algorithm types and their applications. Continue practicing with more advanced concepts to further strengthen understanding."'''

    try:
        # Format evaluation history
        eval_history = format_evaluation_history(previous_evaluations)
        
        # Format the current evaluation
        current_eval_text = f"""
Questions and Answers:
{new_evaluation.get('answer_text', '')}

Overall Feedback:
{new_evaluation.get('feedback', '')}"""
        
        # Format prompt with actual content
        prompt = prompt_template.format(
            knowledge_content=knowledge_content,
            current_mastery=current_mastery,
            current_evaluation=current_eval_text,
            evaluation_history=eval_history
        )
        
        # Get response from Azure OpenAI
        response = call_azure_openai(prompt, "Multiple Choice Mastery Calculation")
        if not response:
            print("Empty response from Azure OpenAI")
            return {
                "mastery": current_mastery,  # Keep current mastery on error
                "explanation": "Error getting LLM analysis, maintaining current mastery level"
            }
        
        # Parse response
        try:
            # Clean the response string
            response = response.strip()
            response = ''.join(char for char in response if ord(char) >= 32 or char in '\n\r\t')
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]
            response = response.replace('\\n', '')
            
            # Parse JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1:
                    json_str = response[start:end + 1].replace('\\"', '"')
                    result = json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found")
            
            # Validate and adjust mastery
            if "mastery" not in result or "explanation" not in result:
                raise ValueError("Missing required fields")
            
            result["mastery"] = float(result["mastery"])
            result["mastery"] = max(0, min(1, result["mastery"]))

            return result
            
        except Exception as e:
            print(f"Error processing response: {e}")
            print(f"Raw response: {response}")
            # Return a default mastery value based on current performance
            return {
                "mastery": max(0.0, current_mastery - 0.1),  # Slight decrease on parsing error
                "explanation": f"Error processing LLM response: {str(e)}. Mastery adjusted based on current performance."
            }
        
    except Exception as e:
        print(f"Error in multiple choice mastery calculation: {e}")
        return {
            "mastery": current_mastery,  # Keep current mastery on error
            "explanation": "Error in LLM analysis, maintaining current mastery level"
        }

def generate_multiple_choice_questions_batch(
    knowledge_content: str,
    main_category: str,
    sub_category: str,
    num_questions: int = 3
) -> Optional[List[Dict[str, Any]]]:
    """
    Generate multiple multiple choice questions from knowledge content using Azure OpenAI
    
    Args:
        knowledge_content: The knowledge content to generate questions from
        main_category: Main category of the knowledge
        sub_category: Sub category of the knowledge
        num_questions: Number of questions to generate (default: 3)
        
    Returns:
        List of dictionaries, each containing question text, options, correct answer index, and explanation
    """
    try:
        prompt = f"""You are a knowledgeable tutor creating multiple choice test questions.
Generate {num_questions} thought-provoking questions with 4 options each based on this knowledge content.
Each question should test deep understanding and have plausible but clearly incorrect distractors.

REQUIREMENTS FOR EACH QUESTION:
1. Focus on the most important concepts in order of relevance
2. Combine multiple related concepts when they naturally fit together
3. Skip less important details if including them would make the question too complex
4. Keep questions focused and concise while still being thought-provoking
5. Should test understanding, analysis, or application of concepts
6. Each question should have exactly 4 options (A, B, C, D)
7. Only one option should be clearly correct
8. Other options should be plausible but clearly wrong
9. Questions should be diverse and cover different aspects of the content
10. Avoid redundant or very similar questions

Main Category: {main_category}
Sub Category: {sub_category}
Knowledge Content:
{knowledge_content}

Respond with valid JSON only, in this format:
{{
    "questions": [
        {{
            "question_text": "First thought-provoking question here...",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer_index": 0,
            "explanation": "Detailed explanation of why the correct answer is right and others are wrong"
        }},
        {{
            "question_text": "Second thought-provoking question here...",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer_index": 2,
            "explanation": "Another detailed explanation"
        }},
        // ... more questions ...
    ]
}}"""

        response = call_azure_openai(prompt, "Multiple Choice Questions Generation")
        result = json.loads(response)
        
        # Validate response format
        if not isinstance(result, dict) or 'questions' not in result:
            print("Error: Invalid response format - missing 'questions' array")
            return None
            
        questions = result['questions']
        if not isinstance(questions, list):
            print("Error: Invalid response format - 'questions' is not an array")
            return None
            
        # Validate each question
        valid_questions = []
        for q in questions:
            if not isinstance(q, dict) or 'question_text' not in q or 'options' not in q:
                print("Error: Invalid question format - missing required fields")
                continue
                
            # Ensure all required fields exist
            if 'correct_answer_index' not in q:
                print("Error: Invalid question format - missing 'correct_answer_index'")
                continue
                
            if 'explanation' not in q:
                q['explanation'] = ''
                
            # Validate options array
            if not isinstance(q['options'], list) or len(q['options']) != 4:
                print("Error: Invalid question format - must have exactly 4 options")
                continue
                
            # Validate correct_answer_index
            if not isinstance(q['correct_answer_index'], int) or q['correct_answer_index'] < 0 or q['correct_answer_index'] >= 4:
                print("Error: Invalid question format - correct_answer_index must be 0-3")
                continue
                
            valid_questions.append(q)
            
        return valid_questions if valid_questions else None
        
    except Exception as e:
        print(f"Error generating multiple choice questions: {e}")
        return None