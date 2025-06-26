"""
Self-test generation and evaluation using Azure OpenAI for the Second Brain Knowledge Management System.
"""
import json
import time
import openai
from typing import Dict, Optional, List
import os
from datetime import datetime

from models import QuestionType
from config.settings import settings

def call_azure_openai(prompt: str, prompt_name: str) -> str:
    """Call Azure OpenAI API for question generation"""
    if not settings.AZURE_OPENAI_API_KEY:
        raise ValueError("AZURE_OPENAI_API_KEY not found")
    if not settings.AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT not found")
        
    # Save prompt to file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{prompt_name.replace(' ', '_')}.txt"
    filepath = os.path.join("tmp/prompts", filename)
    
    # Save prompt to file
    with open(filepath, 'w') as f:
        f.write(prompt)
        
    # Configure Azure OpenAI client
    client = openai.AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
    )
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates thought-provoking questions for knowledge assessment. Always respond with valid JSON in the specified format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    end_time = time.time()
    print(f"[TIMING] ⏱️ {prompt_name}: {(end_time - start_time):.2f}s")
    
    return response.choices[0].message.content or ""

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
-------------------------------
Question: {eval.get('question_text', '')}
Your Answer: {eval.get('answer_text', '')}
Correct Answer: {eval.get('correct_answer', '')}
-------------------------------"""
        else:
            eval_history += f"""
Evaluation #{i} - Free Text
-------------------------------
Question: {eval.get('question_text', '')}
Your Answer: {eval.get('answer_text', '')}
Feedback: {eval.get('feedback', '')}
-------------------------------"""

    return eval_history

def generate_free_text_question(category: str, content: str, knowledge_id: Optional[int] = None) -> Optional[Dict]:
    """
    Generate a thought-provoking free text question for a given knowledge item
    
    Args:
        category: Knowledge category
        content: Knowledge content to generate question from
        knowledge_id: Optional ID to analyze past evaluation patterns
        
    Returns:
        Dictionary with question_text if successful, None if failed
    """
    evaluation_context = ""
    if knowledge_id:
        try:
            from database.supabase_manager import supabase_manager
            evaluations = supabase_manager.get_evaluations(knowledge_id)
            evaluation_context = format_evaluation_history(evaluations)
        except Exception as e:
            print(f"Error getting evaluation patterns: {e}")
            # Continue without evaluation data if there's an error
            pass

    # Question generation prompt template
    prompt_template = '''Generate 1 thought-provoking, open-ended question based on this knowledge content. 
The question should test deep understanding and critical thinking, not just memorization.

Knowledge Category: {category}
Content: {content}

Previous Answers (newest to oldest):
{evaluation_context}

QUESTION GENERATION PRIORITIES:
1. HIGHEST PRIORITY: If there are incorrect/missing points from previous answers, create a question that specifically addresses these gaps in understanding
2. SECOND PRIORITY: Look for important concepts in the knowledge content that haven't been covered by the previous questions shown above
3. Only if no incorrect points or uncovered content: Create a question that approaches previously covered concepts from a new angle

Requirements for the question:
1. Focus on the most important concepts in order of relevance to the user's understanding
2. Combine multiple related concepts when they naturally fit together
3. Skip less important details if including them would make the question too complex
4. Keep the question focused and concise while still being thought-provoking
5. Should require explanation, analysis, or application of concepts
6. Should not be answerable with just a single word or simple fact
7. Should encourage connecting ideas and demonstrating understanding
8. IMPORTANT: Question should be answerable in about 3 sentences - don't try to cover too many points if it would require a longer explanation

Example approach:
- First check the incorrect/missing points from previous answers and prioritize those topics
- Then scan the knowledge content for important concepts not yet questioned
- If the content has multiple concepts, pick the 1-2 most important ones that can be explained together concisely
- If there are examples in the content, use them to ground the question but don't make the question solely about the example
- If there are technical details, focus on their practical implications rather than memorization
- Better to have a focused question about key concepts than a broad question trying to cover everything

Format your response as a JSON object with this structure:
{{
    "question_text": "Your thought-provoking question here that requires deep analysis and understanding..."
}}'''

    try:
        # Format prompt with actual content
        prompt = prompt_template.format(
            category=category,
            content=content,
            evaluation_context=evaluation_context
        )
        
        # Get response from Azure OpenAI
        response = call_azure_openai(prompt, "Question Generation")
        
        # Parse response
        question_data = json.loads(response)
        
        # Validate response has required field
        if "question_text" not in question_data:
            print(f"Error: Response missing question_text field for category {category}")
            return None
            
        return question_data
        
    except json.JSONDecodeError as e:
        print(f"Error parsing question response for category {category}: {e}")
        return None
    except Exception as e:
        print(f"Error generating question for category {category}: {e}")
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
        Dictionary with evaluation results including score (1-5), feedback, etc.
    """
    prompt_template = '''You are an evaluator assessing answers to knowledge questions. Respond with valid JSON only.

Main Category: {main_category}
Sub Category: {sub_category}
Question: {question}
Answer: {answer}
Reference Knowledge Content: {knowledge}

CRITICAL - Initial Checks:
1. If the answer is irrelevant, off-topic, or just a greeting/single word, score it 1 immediately
2. If the answer shows no attempt to address the question's specific points, score it 1
3. Only proceed with detailed evaluation if the answer makes a genuine attempt to address the question

CRITICAL - Answer Completeness Check:
1. First identify ALL parts of what was asked in the question
2. Check if EACH part was properly addressed in the answer
3. Example: If question asks "explain X and give example", both parts must be present
4. Missing ANY major part of the question should result in score ≤ 3

Score the answer on a scale of 1-5 using these strict guidelines:
1 = Any of these conditions:
   - Not understood / Incorrect / Completely off-topic
   - Single word or greeting only
   - No attempt to address the question
   - Irrelevant response
2 = Basic understanding but significant parts missing or incorrect
3 = Moderate understanding, some key parts missing or superficial
4 = Good understanding with minor gaps or imperfections
5 = ONLY if ALL of these are true:
   - ALL parts of the question were fully addressed
   - ALL explanations are thorough and accurate
   - ALL requested examples/applications provided
   - Shows deep understanding beyond basic facts

Also provide a sample answer that would score 5/5.

Format your response as a JSON object with this structure:
{{
    "score": <1-5>,
    "feedback": "Clear explanation of strengths/weaknesses. Use you to refer to the user.",
    "correct_points": ["Point 1", "Point 2"],
    "incorrect_points": ["Missing/wrong point 1", "Missing/wrong point 2"],
    "sample_answer": "Brief but complete answer showing mastery"
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
        required_fields = ["score", "feedback", "correct_points", "incorrect_points"]
        for field in required_fields:
            if field not in result:
                print(f"Error: Response missing {field} field")
                return {
                    "score": 1,
                    "feedback": "Error in evaluation response",
                    "correct_points": [],
                    "incorrect_points": ["Error processing response"],
                    "sample_answer": ""
                }
        
        # Ensure score is integer 1-5
        result["score"] = max(1, min(5, int(result["score"])))
        return result
        
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {e}")
        return {
            "score": 1,
            "feedback": f"Error processing response: {str(e)}",
            "correct_points": [],
            "incorrect_points": ["Error processing response"],
            "sample_answer": ""
        }
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {
            "score": 1,
            "feedback": f"Error: {str(e)}",
            "correct_points": [],
            "incorrect_points": ["Error processing response"],
            "sample_answer": ""
        }

def evaluate_multiple_choice_answers(answers_text: str, knowledge_content: str, main_category: str, sub_category: str) -> Dict:
    """
    Evaluate multiple choice answers for a knowledge item using Azure OpenAI
    
    Args:
        answers_text: Formatted text containing questions, selected answers, and correct answers
        knowledge_content: The knowledge content the questions were based on
        main_category: The main knowledge category
        sub_category: The sub category
        
    Returns:
        Dictionary with evaluation results including feedback and array of evaluations
    """
    prompt_template = '''You are an evaluator assessing multiple choice answers. Respond with valid JSON only.

Main Category: {main_category}
Sub Category: {sub_category}
Knowledge Content: {knowledge}

Multiple Choice Answers:
{answers}

EVALUATION GUIDELINES:
1. Analyze each answer individually and provide specific feedback
2. Identify why the student chose each answer (correct or incorrect)
3. For incorrect answers, explain the misconception that led to that choice
4. For correct answers, reinforce the understanding demonstrated
5. Connect the answers to the core concepts in the knowledge content

Format your response as a JSON object with this structure:
{{
    "feedback": "Overall analysis of understanding based on answer patterns. Use you to refer to the user.",
    "evaluations": [
        {{
            "question": "The original question text",
            "selected_answer": "What the user selected",
            "is_correct": true/false,
            "explanation": "Why this answer was chosen and what it reveals about understanding"
        }},
        // ... more evaluations ...
    ]
}}

Example feedback structure:
"You demonstrate good understanding of concept X in questions 1 and 3, but seem to have some confusion about Y in question 2. Focus on reviewing the relationship between..."

Example evaluation:
{{
    "question": "What is the primary role of mitochondria?",
    "selected_answer": "Energy production",
    "is_correct": true,
    "explanation": "You correctly identified the key function of mitochondria. This shows you understand cellular energy processes."
}}'''

    try:
        # Format prompt with actual content
        prompt = prompt_template.format(
            answers=answers_text,
            knowledge=knowledge_content,
            main_category=main_category,
            sub_category=sub_category
        )
        
        # Get response from Azure OpenAI
        response = call_azure_openai(prompt, "Multiple Choice Evaluation")
        
        # Parse response
        result = json.loads(response)
        
        # Validate evaluation response
        required_fields = ["feedback", "evaluations"]
        for field in required_fields:
            if field not in result:
                print(f"Error: Response missing {field} field")
                return {
                    "feedback": "Error in evaluation response",
                    "evaluations": []
                }
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {e}")
        return {
            "feedback": f"Error processing response: {str(e)}",
            "evaluations": []
        }
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {
            "feedback": f"Error: {str(e)}",
            "evaluations": []
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

Guidelines:
- Free text answers show deeper understanding than multiple choice
- Good explanations increase mastery more than correct recognition
- Poor answers (especially "I don't know") decrease mastery
- Consider answer quality and type when adjusting mastery

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
"You have a good understanding of basic circuit components and Ohm's law. You can explain voltage and current flow clearly. You need to work on understanding parallel circuits and power calculations."

"You show basic grasp of SQL queries and table joins. You can write simple SELECT statements but struggle with complex joins and subqueries. You should focus on practicing GROUP BY and aggregate functions."

"You're still developing understanding of neural networks. You recognize basic terminology but can't explain backpropagation or activation functions. You need to review the fundamental concepts."'''

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
            return {
                "mastery": current_mastery,  # Keep current mastery on error
                "explanation": "Error in LLM analysis, maintaining current mastery level"
            }
        
    except Exception as e:
        print(f"Error in mastery calculation: {e}")
        return {
            "mastery": current_mastery,  # Keep current mastery on error
            "explanation": "Error in LLM analysis, maintaining current mastery level"
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

Guidelines:
- Multiple choice answers show recognition and basic understanding
- Consistent correct answers indicate good grasp of fundamentals
- Pattern of incorrect answers may reveal specific misconceptions
- Consider answer patterns and explanations when adjusting mastery
- Recent performance should have more weight than older answers
- Quick, correct answers suggest stronger mastery than hesitant ones
- Look for improvement trends over time

Knowledge Content:
{knowledge_content}

Current Evaluation:
{current_evaluation}

Previous Answers (newest to oldest):
{evaluation_history}

Response Format:
{{
    "mastery": <float 0-1>,
    "explanation": "2-3 sentences describing: 1) Pattern of multiple choice performance 2) Areas of demonstrated understanding 3) Specific concepts needing review"
}}

Example responses:
"Your multiple choice performance shows strong recognition of key concepts with 80% accuracy. You consistently identify correct database relationships but sometimes confuse normalization forms. Focus on reviewing the specific criteria for 2NF vs 3NF."

"Recent answers demonstrate improved understanding of quantum mechanics principles. You correctly identify wave-particle duality examples but still have difficulty with uncertainty principle applications. Continue practicing problems involving position-momentum uncertainty."

"Multiple choice results indicate basic familiarity with machine learning concepts. While you can recognize supervised vs unsupervised approaches, you often mix up specific algorithms like k-means and kNN. Review the key characteristics that distinguish clustering from classification algorithms."'''

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
            return {
                "mastery": current_mastery,  # Keep current mastery on error
                "explanation": "Error in LLM analysis, maintaining current mastery level"
            }
        
    except Exception as e:
        print(f"Error in multiple choice mastery calculation: {e}")
        return {
            "mastery": current_mastery,  # Keep current mastery on error
            "explanation": "Error in LLM analysis, maintaining current mastery level"
        }

def generate_multiple_choice_question(category: str, content: str, knowledge_id: Optional[int] = None) -> Optional[Dict]:
    """
    Generate a multiple choice question for a given knowledge item
    
    Args:
        category: Knowledge category
        content: Knowledge content to generate question from
        knowledge_id: Optional ID to analyze past evaluation patterns
        
    Returns:
        Dictionary with question_text, options, correct_answer_index, explanation, and question_id if successful, None if failed
    """
    evaluation_context = ""
    if knowledge_id:
        try:
            from database.supabase_manager import supabase_manager
            evaluations = supabase_manager.get_evaluations(knowledge_id)
            evaluation_context = format_evaluation_history(evaluations)
        except Exception as e:
            print(f"Error getting evaluation patterns: {e}")
            pass

    prompt_template = '''Generate 1 multiple choice question based on this knowledge content.
The question should test understanding and application, not just memorization.

Knowledge Category: {category}
Content: {content}

Previous Answers (newest to oldest):
{evaluation_context}

QUESTION GENERATION PRIORITIES:
1. HIGHEST PRIORITY: If there are incorrect/missing points from previous answers, create a question that specifically addresses these gaps
2. SECOND PRIORITY: Look for important concepts that haven't been covered by previous questions
3. Only if no incorrect points or uncovered content: Create a question that approaches previously covered concepts from a new angle

Requirements:
1. Question should test understanding and application of concepts
2. Options should be plausible but clearly distinguishable
3. Include exactly 4 options (indexed 0-3)
4. One and only one option should be correct
5. Distractors (wrong options) should be:
   - Plausible but clearly incorrect
   - Based on common misconceptions
   - Similar in length and style to correct answer
   - Not obviously wrong or joke answers
6. All options should be:
   - Similar in length
   - Grammatically consistent with question
   - Clear and unambiguous
   - Independent of each other

Format your response as a JSON object with this structure:
{{
    "question_text": "Your thought-provoking multiple choice question here...",
    "options": [
        "First option text here",
        "Second option text here",
        "Third option text here",
        "Fourth option text here"
    ],
    "correct_answer_index": <number 0-3>,
    "explanation": "Detailed explanation of why the correct answer is correct and why each distractor is incorrect"
}}'''

    try:
        # Format prompt with actual content
        prompt = prompt_template.format(
            category=category,
            content=content,
            evaluation_context=evaluation_context
        )
        
        # Get response from Azure OpenAI
        response = call_azure_openai(prompt, "Multiple Choice Question Generation")
        
        # Parse response
        question_data = json.loads(response)
        
        # Validate response has required fields
        required_fields = ["question_text", "options", "correct_answer_index", "explanation"]
        for field in required_fields:
            if field not in question_data:
                print(f"Error: Response missing {field} field for category {category}")
                return None
                
        # Validate options and correct answer
        if len(question_data["options"]) != 4:
            print(f"Error: Expected 4 options, got {len(question_data['options'])}")
            return None
            
        if not isinstance(question_data["correct_answer_index"], int) or \
           question_data["correct_answer_index"] < 0 or \
           question_data["correct_answer_index"] > 3:
            print(f"Error: Invalid correct_answer_index: {question_data['correct_answer_index']}")
            return None
            
        # Store question in database if knowledge_id is provided
        if knowledge_id:
            try:
                from database.supabase_manager import supabase_manager
                db_question = supabase_manager.create_multiple_choice_question({
                    "question_text": question_data["question_text"],
                    "options": question_data["options"],
                    "correct_answer_index": question_data["correct_answer_index"],
                    "explanation": question_data["explanation"],
                    "knowledge_id": knowledge_id
                })
                if db_question:
                    # Add database ID to response
                    question_data["question_id"] = db_question["id"]
            except Exception as e:
                print(f"Error storing multiple choice question: {e}")
                # Continue without storing if there's an error
                pass
            
        return question_data
        
    except json.JSONDecodeError as e:
        print(f"Error parsing question response for category {category}: {e}")
        return None
    except Exception as e:
        print(f"Error generating question for category {category}: {e}")
        return None