import openai
from typing import List, Dict, Optional
from config.settings import settings
from utils.category_mapping import get_main_category

def call_azure_openai_llm(prompt: str) -> str:
    """
    Call Azure OpenAI GPT-4 with the given prompt
    
    Args:
        prompt: The prompt to send to the LLM
        
    Returns:
        The LLM's response text
    """
    if not settings.AZURE_OPENAI_API_KEY:
        raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
    if not settings.AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")
    
    # Configure Azure OpenAI client
    client = openai.AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
    )
    
    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant for knowledge management. Provide structured, actionable recommendations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1500
    )
    
    return response.choices[0].message.content

def LLMUpdater(input_text: str, existing_knowledge: Optional[Dict], llm_type: str = "azure_openai") -> List[Dict]:
    """
    Generate 3 different recommendations for updating knowledge using Azure OpenAI
    
    Args:
        input_text: New text input from user
        existing_knowledge: Most similar existing knowledge item (from SSC)
        llm_type: Type of LLM to use (only "azure_openai" supported)
        
    Returns:
        List of 3 recommendation dictionaries with main_category and sub_category
    """
    if llm_type != "azure_openai":
        raise ValueError("Only 'azure_openai' is supported for llm_type")
    
    # Build context for the LLM
    context = f"New input text: {input_text}\n\n"
    
    if existing_knowledge:
        context += f"Most similar existing knowledge:\n"
        context += f"- Main Category: {existing_knowledge.get('main_category', 'Unknown')}\n"
        context += f"- Sub-Category: {existing_knowledge.get('sub_category', existing_knowledge.get('category', 'Unknown'))}\n"
        context += f"- Content: {existing_knowledge.get('content', '')[:200]}...\n"
        context += f"- Similarity Score: {existing_knowledge.get('similarity_score', 0):.3f}\n\n"
    else:
        context += "No similar existing knowledge found.\n\n"
    
    # Create the prompt for generating recommendations
    prompt = f"""{context}

Please provide exactly 3 different recommendations for handling this new input text. Each recommendation should be a different approach:

1. **Merge/Update Approach**: If similar knowledge exists, merge or update it
2. **Replace/Revise Approach**: If similar knowledge exists, replace or significantly revise it  
3. **New Category Approach**: Create entirely new knowledge in a new sub-category

For each recommendation, provide:
- A clear explanation of what changes will be made
- The complete updated/new text content
- A specific sub-category name (be descriptive and specific, e.g., "machine learning algorithms", "startup funding strategies", "productivity techniques")
- 3-5 relevant tags
- A brief preview (first 100 characters)

Format your response as JSON with this exact structure:
[
  {{
    "option_number": 1,
    "change": "Description of what changes will be made",
    "updated_text": "Complete updated or new content text",
    "sub_category": "specific sub-category name",
    "tags": ["tag1", "tag2", "tag3"],
    "preview": "Brief preview of content..."
  }},
  {{
    "option_number": 2,
    "change": "Description of what changes will be made",
    "updated_text": "Complete updated or new content text", 
    "sub_category": "specific sub-category name",
    "tags": ["tag1", "tag2", "tag3"],
    "preview": "Brief preview of content..."
  }},
  {{
    "option_number": 3,
    "change": "Description of what changes will be made",
    "updated_text": "Complete updated or new content text",
    "sub_category": "specific sub-category name", 
    "tags": ["tag1", "tag2", "tag3"],
    "preview": "Brief preview of content..."
  }}
]

Make sure the sub-categories are specific and descriptive. The content should be comprehensive and well-structured."""

    try:
        # Get response from Azure OpenAI
        llm_response = call_azure_openai_llm(prompt)
        
        # Parse the JSON response
        import json
        recommendations = json.loads(llm_response)
        
        # Validate and enhance each recommendation
        enhanced_recommendations = []
        for rec in recommendations:
            # Determine main category from sub-category
            sub_category = rec.get('sub_category', 'general')
            main_category = get_main_category(sub_category)
            
            enhanced_rec = {
                'option_number': rec.get('option_number', len(enhanced_recommendations) + 1),
                'change': rec.get('change', 'Update knowledge'),
                'updated_text': rec.get('updated_text', input_text),
                'main_category': main_category,
                'sub_category': sub_category,
                'tags': rec.get('tags', []),
                'preview': rec.get('preview', rec.get('updated_text', '')[:100] + '...')
            }
            enhanced_recommendations.append(enhanced_rec)
        
        return enhanced_recommendations
        
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response as JSON: {e}")
        # Fallback recommendations
        return create_fallback_recommendations(input_text, existing_knowledge)
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
        return create_fallback_recommendations(input_text, existing_knowledge)

def create_fallback_recommendations(input_text: str, existing_knowledge: Optional[Dict]) -> List[Dict]:
    """
    Create fallback recommendations if LLM call fails
    
    Args:
        input_text: The input text
        existing_knowledge: Existing knowledge item if any
        
    Returns:
        List of 3 fallback recommendations
    """
    # Determine sub-category from input text (simple keyword matching)
    input_lower = input_text.lower()
    
    if any(word in input_lower for word in ['ai', 'machine learning', 'neural', 'algorithm']):
        sub_category = 'artificial intelligence'
    elif any(word in input_lower for word in ['business', 'startup', 'entrepreneur', 'marketing']):
        sub_category = 'business strategy'
    elif any(word in input_lower for word in ['productivity', 'time management', 'habits']):
        sub_category = 'productivity techniques'
    elif any(word in input_lower for word in ['programming', 'code', 'software', 'development']):
        sub_category = 'software development'
    else:
        sub_category = 'general knowledge'
    
    main_category = get_main_category(sub_category)
    
    recommendations = [
        {
            'option_number': 1,
            'change': 'Merge with existing knowledge' if existing_knowledge else 'Create new knowledge entry',
            'updated_text': input_text,
            'main_category': main_category,
            'sub_category': sub_category,
            'tags': ['new', 'unprocessed'],
            'preview': input_text[:100] + '...' if len(input_text) > 100 else input_text
        },
        {
            'option_number': 2,
            'change': 'Create enhanced version with additional context',
            'updated_text': f"Enhanced: {input_text}",
            'main_category': main_category,
            'sub_category': f"enhanced {sub_category}",
            'tags': ['enhanced', 'expanded'],
            'preview': f"Enhanced: {input_text[:90]}..." if len(input_text) > 90 else f"Enhanced: {input_text}"
        },
        {
            'option_number': 3,
            'change': 'Create new specialized category',
            'updated_text': input_text,
            'main_category': 'General',
            'sub_category': 'specialized knowledge',
            'tags': ['specialized', 'new-category'],
            'preview': input_text[:100] + '...' if len(input_text) > 100 else input_text
        }
    ]
    
    return recommendations