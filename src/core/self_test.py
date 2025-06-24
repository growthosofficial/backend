"""
Self-test generation and evaluation using Azure OpenAI for the Second Brain Knowledge Management System.
"""
import json
import openai
from typing import Dict, Optional, List

from config.settings import settings

def call_azure_openai(prompt: str) -> str:
    """Call Azure OpenAI API for question generation"""
    if not settings.AZURE_OPENAI_API_KEY:
        raise ValueError("AZURE_OPENAI_API_KEY not found")
    if not settings.AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT not found")
        
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
        temperature=0.7
    )
    
    return response.choices[0].message.content or ""

def generate_question(category: str, content: str, knowledge_id: Optional[int] = None) -> Optional[Dict]:
    """
    Generate a thought-provoking question for a given knowledge item
    
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
            patterns = supabase_manager.get_evaluations(knowledge_id)
            if patterns and patterns["evaluations"]:
                # Format previous evaluations for the prompt
                eval_examples = patterns["evaluations"]  # Already limited to 3 most recent
                evaluation_context = "Previous question-answer examples:\n"
                for eval in eval_examples:
                    evaluation_context += f"""
Question: {eval['question_text']}
Answer: {eval['answer_text']}
Feedback: {eval['feedback']}
What was correct:
{chr(10).join(f"- {point}" for point in eval['correct_points'])}
What was incorrect/missing:
{chr(10).join(f"- {point}" for point in eval['incorrect_points'])}
"""
        except Exception as e:
            print(f"Error getting evaluation patterns: {e}")
            # Continue without evaluation data if there's an error
            pass

    # Question generation prompt template
    prompt_template = '''Generate 1 thought-provoking, open-ended question based on this knowledge content. 
The question should test deep understanding and critical thinking, not just memorization.

Knowledge Category: {category}
Content: {content}

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
        response = call_azure_openai(prompt)
        
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

def evaluate_answer(question_text: str, answer: str, knowledge_content: str, main_category: str, sub_category: str) -> Dict:
    """
    Evaluate a user's answer to a question using Azure OpenAI
    
    Args:
        question_text: The original question text
        answer: The user's answer to evaluate
        knowledge_content: The knowledge content the question was based on
        main_category: The main knowledge category (e.g. Physics)
        sub_category: The sub category (e.g. Quantum Mechanics)
        
    Returns:
        Dictionary with evaluation results including score (1-5), feedback, etc.
    """
    # Evaluation prompt with 1-5 scale definition
    prompt_template = '''You are evaluating an answer to a knowledge assessment question.
Please evaluate the answer based on both the provided knowledge content AND your general knowledge.
Be flexible - if the answer provides correct information that's not in the knowledge content but is accurate, it should get credit.

Main Category: {main_category}
Sub Category: {sub_category}
Question: {question}
Your Answer: {answer}
Reference Knowledge Content: {knowledge}

Score the answer on a scale of 1-5:
1 = Not understood / Incorrect
2 = Basic understanding with significant gaps
3 = Moderate understanding
4 = Good understanding with minor gaps
5 = Complete mastery

Requirements for evaluation:
1. Score must be an integer from 1 to 5
2. Provide specific feedback on strengths and weaknesses
3. Consider both factual accuracy and depth of understanding
4. Give partial credit for partially correct answers
5. Reward demonstration of understanding over memorization
6. Consider additional correct information even if not in reference content
7. Explain the scoring rationale clearly
8. Consider the knowledge category context when evaluating domain-specific answers

Format your response as a JSON object with this structure:
{{
    "score": <integer 1-5>,
    "feedback": "Detailed feedback explaining strengths, weaknesses, and specific suggestions for improvement...",
    "correct_points": ["Point 1 that was correct", "Point 2 that was correct", ...],
    "incorrect_points": ["Point 1 that was incorrect/missing", "Point 2 that was incorrect/missing", ...]
}}'''

    try:
        # Format prompt with actual content
        prompt = prompt_template.format(
            main_category=main_category,
            sub_category=sub_category,
            question=question_text,
            answer=answer,
            knowledge=knowledge_content
        )
        
        # Get response from Azure OpenAI
        response = call_azure_openai(prompt)
        
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
                    "incorrect_points": ["Error processing response"]
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
            "incorrect_points": ["Error processing response"]
        }
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {
            "score": 1,
            "feedback": f"Error: {str(e)}",
            "correct_points": [],
            "incorrect_points": ["Error processing response"]
        }

def calculate_mastery_with_llm(knowledge_content: str, evaluations: List[Dict]) -> Dict:
    """
    Calculate mastery level using LLM by analyzing evaluation history
    
    Args:
        knowledge_content: The full content of the knowledge item
        evaluations: List of evaluation records, each containing question, answer, score, feedback, etc.
        
    Returns:
        Dictionary with mastery level (0-1) and explanation
    """
    prompt_template = '''You are a helpful assistant that analyzes student mastery levels based on their evaluation history.
Always respond with valid JSON in the specified format.

Knowledge Content:
{knowledge_content}

Evaluation History (from most recent to oldest):
{evaluation_history}

Requirements for mastery calculation:
1. Output a single mastery number between 0 and 1
2. Consider both breadth (concept coverage) and depth (quality of understanding)
3. Recent evaluations should matter more than older ones
4. Perfect scores (5/5) on multiple evaluations indicate strong mastery
5. IMPORTANT - Concept Coverage Analysis:
   - First identify all key concepts in the knowledge content
   - Check if the questions and answers together cover ALL these concepts
   - Even with perfect scores, if some key concepts were never tested, mastery cannot be 1.0
   - Example: If knowledge has concepts A, B, C but questions only tested A and B, max mastery should be around 0.67
6. To achieve mastery = 1.0, ALL of these must be true:
   - Questions and answers must cover ALL key concepts from the knowledge content
   - Must demonstrate understanding of each concept
   - Must have consistently high scores
   - Recent performance must be strong

Format your response as a JSON object with this structure:
{{
    "mastery": <float between 0 and 1>,
    "explanation": "Brief explanation of why this mastery level was assigned"
}}

Example response:
{{
    "mastery": 0.85,
    "explanation": "Strong performance (5/5) on recent tests covering concepts X and Y. However, concept Z from the knowledge content hasn't been tested yet, preventing full mastery score."
}}'''

    try:
        # Format evaluation history
        eval_history = ""
        for i, eval in enumerate(evaluations):
            eval_history += f"""
Evaluation {i+1}:
Question: {eval.get('question_text', '')}
Answer: {eval.get('answer_text', '')}
Score: {eval.get('score', 0)}/5
Feedback: {eval.get('feedback', '')}
Correct Points:
{chr(10).join(f"- {point}" for point in eval.get('correct_points', []))}
Incorrect/Missing Points:
{chr(10).join(f"- {point}" for point in eval.get('incorrect_points', []))}
"""
        
        # Format prompt with actual content
        prompt = prompt_template.format(
            knowledge_content=knowledge_content,
            evaluation_history=eval_history
        )
        
        # Get response from Azure OpenAI
        response = call_azure_openai(prompt)
        if not response:
            print("Empty response from Azure OpenAI")
            return {
                "mastery": 0.8 if evaluations and evaluations[0].get('score', 0) == 5 else 0.4,
                "explanation": "Error getting LLM analysis, using score-based fallback"
            }
        
        # Parse response
        try:
            # Clean the response string
            response = response.strip()
            
            # Remove any potential BOM or hidden characters
            response = ''.join(char for char in response if ord(char) >= 32 or char in '\n\r\t')
            
            # Remove any leading/trailing quotes if present
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]
            
            # Remove any escaped newlines
            response = response.replace('\\n', '')
            
            # Try to find JSON in the response
            try:
                # First try direct parsing
                result = json.loads(response)
            except json.JSONDecodeError:
                # If that fails, try to find JSON object markers
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1:
                    json_str = response[start:end + 1]
                    # Replace any escaped quotes
                    json_str = json_str.replace('\\"', '"')
                    result = json.loads(json_str)
                else:
                    raise ValueError("No valid JSON object found in response")
            
            # Validate required fields
            if "mastery" not in result or "explanation" not in result:
                print("Error: Response missing required fields")
                print(f"Raw response: {response}")
                raise ValueError("Missing required fields in response")
            
            # Ensure mastery is float 0-1
            try:
                result["mastery"] = float(result["mastery"])
                result["mastery"] = max(0, min(1, result["mastery"]))
                result["mastery"] = round(result["mastery"], 2)
            except (TypeError, ValueError):
                print(f"Error converting mastery to float: {result['mastery']}")
                raise ValueError("Invalid mastery value")
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing mastery calculation response: {e}")
            print(f"Raw response: {response}")
            # Fallback: Use a score-based calculation
            latest_score = evaluations[0].get('score', 0) if evaluations else 0
            return {
                "mastery": min(1.0, latest_score / 5.0),  # Convert 5-point score to 0-1 scale
                "explanation": "Error in LLM analysis, using score-based fallback"
            }
        
    except Exception as e:
        print(f"Error in mastery calculation: {e}")
        # Fallback: Use a score-based calculation if we have evaluations
        if evaluations:
            latest_score = evaluations[0].get('score', 0)
            return {
                "mastery": min(1.0, latest_score / 5.0),
                "explanation": "Error in LLM analysis, using score-based fallback"
            }
        return {
            "mastery": 0.0,
            "explanation": f"Error: {str(e)}"
        }

def get_knowledge_mastery(knowledge_id: int, current_evaluation: Dict, supabase_manager) -> Dict:
    """
    Calculate mastery level for a knowledge item based on current and past evaluations
    
    Args:
        knowledge_id: ID of the knowledge item
        current_evaluation: Current evaluation data
        supabase_manager: Instance of SupabaseManager for database access
        
    Returns:
        Dictionary with mastery level (0-1) and explanation
    """
    try:
        # Get knowledge item content
        knowledge_result = supabase_manager.supabase.table('knowledge_items')\
            .select('content')\
            .eq('id', knowledge_id)\
            .single()\
            .execute()
        
        if not knowledge_result.data:
            print(f"Knowledge item {knowledge_id} not found")
            score_based_mastery = min(1.0, current_evaluation.get('score', 0) / 5.0)
            return {
                "mastery": score_based_mastery,
                "explanation": "Using current evaluation score as mastery (knowledge item not found)"
            }
        
        knowledge_content = knowledge_result.data.get('content', '')
        
        # Get recent evaluations ordered by date
        eval_result = supabase_manager.supabase.table('evaluations')\
            .select('*')\
            .eq('knowledge_id', knowledge_id)\
            .order('created_at', desc=True)\
            .limit(5)\
            .execute()
        
        # Prepare evaluations list with current evaluation first
        evaluations = [current_evaluation]  # Current evaluation
        if eval_result.data:
            evaluations.extend(eval_result.data)  # Previous evaluations
        
        # Calculate mastery using LLM
        mastery_result = calculate_mastery_with_llm(
            knowledge_content=knowledge_content,
            evaluations=evaluations
        )
        
        return mastery_result
        
    except Exception as e:
        print(f"Error calculating mastery for knowledge item {knowledge_id}: {e}")
        # Fallback to using the current evaluation score
        score_based_mastery = min(1.0, current_evaluation.get('score', 0) / 5.0)
        return {
            "mastery": score_based_mastery,
            "explanation": f"Using current evaluation score as mastery due to error: {str(e)}"
        }