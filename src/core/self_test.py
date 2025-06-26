"""
Self-test generation and evaluation using Azure OpenAI for the Second Brain Knowledge Management System.
"""
import json
import time
import openai
from typing import Dict, Optional, List

from config.settings import settings

def call_azure_openai(prompt: str, prompt_name: str) -> str:
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
    prompt_template = '''You are an evaluator assessing answers to knowledge questions. Respond with valid JSON only.

Main Category: {main_category}
Sub Category: {sub_category}
Question: {question}
Answer: {answer}
Reference Knowledge Content: {knowledge}

CRITICAL - Answer Completeness Check:
1. First identify ALL parts of what was asked in the question
2. Check if EACH part was properly addressed in the answer
3. Example: If question asks "explain X and give example", both parts must be present
4. Missing ANY major part of the question should result in score ≤ 3

Score the answer on a scale of 1-5 using these strict guidelines:
1 = Not understood / Incorrect / Completely off-topic
2 = Basic understanding but significant parts missing or incorrect
3 = Moderate understanding, some key parts missing or superficial
4 = Good understanding with minor gaps or imperfections
5 = ONLY if ALL of these are true:
   - ALL parts of the question were fully addressed
   - ALL explanations are thorough and accurate
   - ALL requested examples/applications provided
   - Shows deep understanding beyond basic facts

Scoring Guidelines:
- Missing any major asked-for component: Maximum score 3
- Only basic definitions without asked-for explanations: Maximum score 2
- No examples when examples were requested: Maximum score 3
- Incorrect or missing real-world applications: Reduce score by at least 1
- Superficial answers to multi-part questions: Maximum score 3

Also provide a sample answer that would score 5/5. The sample answer should:
1. Directly address all parts of the question
2. Show clear understanding of the concepts
3. Use specific examples or applications where relevant
4. Connect ideas logically
5. Stay focused and avoid unnecessary details

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
        response = call_azure_openai(prompt, "Answer Evaluation")
        
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

def calculate_mastery_with_llm(knowledge_content: str, new_evaluation: Dict, previous_evaluations: List[Dict], current_mastery: float) -> Dict:
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
        eval_history = ""
        for i, eval in enumerate(previous_evaluations, 1):
            eval_type = eval.get('question_type', 'free_text')
            if eval_type == 'multiple_choice':
                eval_history += f"""
Q{i} (Multiple Choice): {eval.get('question_text', '')}
Selected: {eval.get('selected_answer', '')} | Correct: {eval.get('correct_answer', '')}"""
            else:
                eval_history += f"""
Q{i} (Free Text): {eval.get('question_text', '')}
A{i}: {eval.get('answer_text', '')}"""
        
        # Format the current answer based on type
        if new_evaluation.get('question_type') == 'multiple_choice':
            current_answer = f"Selected: {new_evaluation.get('selected_answer', '')} | Correct: {new_evaluation.get('correct_answer', '')}"
        else:
            current_answer = new_evaluation.get('answer_text', '')
        
        # Format prompt with actual content
        prompt = prompt_template.format(
            knowledge_content=knowledge_content,
            current_mastery=current_mastery,
            question_type=new_evaluation.get('question_type', 'free_text'),
            new_eval_question=new_evaluation.get('question_text', ''),
            new_eval_answer=current_answer,
            evaluation_history=eval_history
        )
        
        # Get response from Azure OpenAI
        response = call_azure_openai(prompt, "Mastery Calculation")
        if not response:
            print("Empty response from Azure OpenAI")
            return {
                "mastery": 0.0,
                "explanation": "Error getting LLM analysis, using score-based fallback"
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
                "mastery": 0.0,
                "explanation": "Error in LLM analysis, using score-based fallback"
            }
        
    except Exception as e:
        print(f"Error in mastery calculation: {e}")
        return {
            "mastery": 0.0,
            "explanation": "Error in LLM analysis, using score-based fallback"
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
        # Get knowledge item content and current mastery
        knowledge_result = supabase_manager.supabase.table('knowledge_items')\
            .select('content,mastery')\
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
        current_mastery = knowledge_result.data.get('mastery', 0.0)
        
        # Get recent evaluations ordered by date
        eval_result = supabase_manager.supabase.table('evaluations')\
            .select('*')\
            .eq('knowledge_id', knowledge_id)\
            .order('created_at', desc=True)\
            .limit(10)\
            .execute()
        
        # Get previous evaluations
        previous_evaluations = eval_result.data if eval_result.data else []
        
        mastery_result = calculate_mastery_with_llm(
            knowledge_content=knowledge_content,
            new_evaluation=current_evaluation,
            previous_evaluations=previous_evaluations,
            current_mastery=current_mastery
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