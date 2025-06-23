"""
Self-test generation and evaluation using Azure OpenAI for the Second Brain Knowledge Management System.
"""
import json
import openai
from typing import Dict, Optional

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

def generate_question(category: str, content: str) -> Optional[Dict]:
    """
    Generate a thought-provoking question for a given knowledge item
    
    Args:
        category: Knowledge category
        content: Knowledge content to generate question from
        
    Returns:
        Dictionary with question_text if successful, None if failed
    """
    # Question generation prompt template
    prompt_template = '''Generate 1 thought-provoking, open-ended question based on this knowledge content. 
The question should test deep understanding and critical thinking, not just memorization.

Knowledge Category: {category}
Content: {content}

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
            content=content
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

def evaluate_answer(question_text: str, answer: str, knowledge_content: str, category: str) -> Dict:
    """
    Evaluate a user's answer to a question using Azure OpenAI
    
    Args:
        question_text: The original question text
        answer: The user's answer to evaluate
        knowledge_content: The knowledge content the question was based on
        category: The knowledge category for additional context
        
    Returns:
        Dictionary with evaluation results including score (1-5), feedback, etc.
    """
    # Evaluation prompt with 1-5 scale definition
    prompt_template = '''You are evaluating an answer to a knowledge assessment question.
Please evaluate the answer based on both the provided knowledge content AND your general knowledge.
Be flexible - if the answer provides correct information that's not in the knowledge content but is accurate, it should get credit.

Knowledge Category: {category}
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
    "feedback": "Detailed feedback explaining strengths and weaknesses...",
    "correct_points": ["Point 1 that was correct", "Point 2 that was correct", ...],
    "incorrect_points": ["Point 1 that was incorrect/missing", "Point 2 that was incorrect/missing", ...],
    "improvement_suggestions": "Specific suggestions for improvement..."
}}'''

    try:
        # Format prompt with actual content
        prompt = prompt_template.format(
            category=category,
            question=question_text,
            answer=answer,
            knowledge=knowledge_content
        )
        
        # Get response from Azure OpenAI
        response = call_azure_openai(prompt)
        
        # Parse response
        result = json.loads(response)
        
        # Validate evaluation response
        required_fields = ["score", "feedback", "correct_points", "incorrect_points", 
                         "improvement_suggestions"]
        for field in required_fields:
            if field not in result:
                print(f"Error: Response missing {field} field")
                return {
                    "score": 1,
                    "feedback": "Error in evaluation response",
                    "correct_points": [],
                    "incorrect_points": ["Error processing response"],
                    "improvement_suggestions": "Please try again"
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
            "improvement_suggestions": "Please try again"
        }
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {
            "score": 1,
            "feedback": f"Error: {str(e)}",
            "correct_points": [],
            "incorrect_points": ["Error processing response"],
            "improvement_suggestions": "Please try again"
        }