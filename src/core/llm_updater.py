import openai
import json
from typing import List, Dict, Optional
from config.settings import settings
from utils.category_mapping import get_all_subject_categories, get_subject_categories_prompt_text, validate_subject_category, get_subject_category_fallback

def _ensure_string(value):
    """Ensure a value is a string, converting lists to space-separated strings."""
    if isinstance(value, list):
        return ' '.join(str(item) for item in value)
    elif isinstance(value, (str, int, float)):
        return str(value)
    elif value is None:
        return ""
    else:
        return str(value)

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
            {"role": "system", "content": """You are a Knowledge Organization Strategist. Your role is to analyze input text and generate 3 different strategic approaches for organizing it.

CRITICAL: You generate INSTRUCTIONS for a second LLM that will actually transform the input text. Your instructions must be:
- Specific and actionable
- Clear step-by-step commands
- Detailed enough for another LLM to execute precisely
- Focused on text transformation, restructuring, and content organization

You do NOT transform the text yourself - you create detailed instructions for how another LLM should transform it.

Your communication style is methodical, explanatory, and system-aware. You avoid ambiguity, vague tags, mechanical merging, rigid taxonomy, content dilution, and goal-irrelevant recommendations."""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1000 # Reduced for faster responses
    )
    
    content = response.choices[0].message.content
    if content is None:
        return ""
    return content

def LLMUpdater(input_text: str, existing_knowledge: Optional[Dict], goal: Optional[str] = None, llm_type: str = "azure_openai") -> Dict:
    """
    Generate 4 different recommendations using Azure OpenAI with goal-based semantic analysis
    
    Args:
        input_text: New text input from user
        existing_knowledge: Most similar existing knowledge item (from SSC)
        goal: User's current learning/knowledge goal for contextual relevance analysis
        llm_type: Type of LLM to use (only "azure_openai" supported)
        
    Returns:
        Dictionary with goal_relevance_score and recommendations
    """
    if llm_type != "azure_openai":
        raise ValueError("Only 'azure_openai' is supported for llm_type")
    
    # Build semantic context for analysis
    context_sections = []
    
    # Input text analysis
    context_sections.append(f"INPUT TEXT:\n{input_text}\n")
    
    # Goal context if provided
    if goal:
        context_sections.append(f"LEARNING GOAL: {goal}\n")
    
    # Existing knowledge context
    if existing_knowledge:
        similarity_score = existing_knowledge.get('similarity_score', 0)
        context_sections.append(f"SIMILAR EXISTING KNOWLEDGE (Score: {similarity_score:.3f}):")
        context_sections.append(f"Category: {existing_knowledge.get('main_category', 'Unknown')} > {existing_knowledge.get('sub_category', 'Unknown')}")
        context_sections.append(f"Content: {existing_knowledge.get('content', '')[:200]}...")
        context_sections.append("")
    else:
        context_sections.append("SIMILAR EXISTING KNOWLEDGE: None found\n")
    
    # Academic categories context
    categories_text = """Available academic categories: Mathematics, Physics, Chemistry, Biology, Computer Science, Engineering, Medicine, Environmental Science, Astronomy, Geology, Psychology, Sociology, Anthropology, Political Science, Economics, Geography, Linguistics, Archaeology, History, Philosophy, Literature, Languages, Religious Studies, Art History, Music Theory, Theater Arts, Business Administration, Law, Education, Communications, Journalism, Architecture, Agriculture, Nutrition, Visual Arts, Music, Creative Writing, Film Studies, Photography, Design, Public Health, Physical Education, Sports Science, Therapy & Rehabilitation, Data Science, Cybersecurity, Biotechnology, Cognitive Science, International Studies, Gender Studies, Urban Planning, General Studies"""
    context_sections.append(categories_text)
    
    # Construct the enhanced semantic prompt
    goal_instructions = ""
    goal_fields_example = ""
    if goal:
        goal_instructions = f"""
GOAL: "{goal}"

For options 1 & 2:
1. Evaluate input relevance to goal (input text relevance to goal should be the same for both options 1 and 2)
2. Perform semantic search for similar knowledge
3. Add ONLY goal-relevant content
4. Include: relevance_score (1-10), goal_alignment, goal_priority (high/medium/low)

Option 3: Regular semantic approach (no goal bias)
"""
    else:
        goal_instructions = """
No goal provided. All 3 options use semantic organization.
"""

    prompt = f"""Analyze this knowledge input and provide 3 recommendations for optimal organization.

INPUT: {input_text}

{chr(10).join(context_sections)}

{goal_instructions}

INSTRUCTIONS:
1. First, evaluate the input's relevance to the goal (if provided) and assign a goal_relevance_score (1-10)
   GOAL RELEVANCE SCORING GUIDELINES:
   - 1-2: Content is completely unrelated to the goal (e.g., history content when goal is gaming)
   - 3-4: Content has very weak or tangential connection to the goal
   - 5-6: Content has some relevance but requires significant transformation to align with goal
   - 7-8: Content is moderately relevant and can be adapted to the goal
   - 9-10: Content is highly relevant and directly supports the goal
2. Provide a brief explanation for the goal_relevance_score
3. Provide exactly 3 distinct recommendations
4. Preserve all content detail - never summarize or paraphrase
5. Use semantic relationships with existing knowledge
6. Select main_category from academic subjects list (exact match required)
7. Generate 3-4 meaningful tags per recommendation (tags should be related to the content and category)
8. CRITICAL: Write instructions for transforming the input text into updated text

OUTPUT FORMAT (MUST BE VALID JSON):
{{
  "goal_relevance_score": 7,
  "goal_relevance_explanation": "Brief explanation of why this score was assigned",
  "recommendations": [
    {{
      "option_number": 1,
      "change": "2-3 sentences explaining this approach",
      "instructions": "Transform the input text by: [specific text transformation steps like restructure paragraphs, add headings, modify language style, combine related content, etc.]. Focus only on text transformation.",
      "main_category": "EXACT academic subject name",
      "sub_category": "specific sub-topic",
      "tags": ["tag1", "tag2", "tag3"],
      "action_type": "merge/update/create_new"{goal_fields_example}
    }},
    {{
      "option_number": 2,
      "change": "2-3 sentences explaining an alternative approach",
      "instructions": "Transform the input text by: [specific text transformation steps like restructure paragraphs, add headings, modify language style, combine related content, etc.]. Focus only on text transformation.",
      "main_category": "EXACT academic subject name",
      "sub_category": "specific sub-topic",
      "tags": ["tag1", "tag2", "tag3"],
      "action_type": "merge/update/create_new"{goal_fields_example}
    }},
    {{
      "option_number": 3,
      "change": "2-3 sentences explaining regular semantic approach.",
      "instructions": "Transform the input text by: [specific text transformation steps like restructure paragraphs, add headings, modify language style, combine related content, etc.]. Focus only on text transformation.",
      "main_category": "EXACT academic subject name",
      "sub_category": "specific sub-topic",
      "tags": ["tag1", "tag2", "tag3"],
      "action_type": "merge/update/create_new"
    }}
  ]
}}

REQUIREMENTS:
- Academic subject must exactly match provided list
- Never reduce content detail
- Instructions should focus ONLY on text transformation (restructure, add sections, modify language, combine content)
- Do NOT include semantic search, external operations, or citations in instructions
- Include goal fields (relevance_score, goal_alignment, goal_priority) ONLY in options 1 & 2 when goal provided
- Ensure valid JSON formatting"""

    try:
        # Get response from Azure OpenAI
        llm_response = call_azure_openai_llm(prompt)
        
        # Clean the response - remove markdown code blocks if present
        cleaned_response = llm_response.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        if cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]   # Remove ```
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]  # Remove trailing ```
        cleaned_response = cleaned_response.strip()
        
        # Parse the JSON response
        response_data = json.loads(cleaned_response)
        
        # Handle both new nested structure and old list format
        if isinstance(response_data, dict):
            # New format with goal_relevance_score and recommendations
            goal_relevance_score = response_data.get('goal_relevance_score', 5)
            goal_relevance_explanation = response_data.get('goal_relevance_explanation', '')
            recommendations = response_data.get('recommendations', [])
        else:
            # Old format - direct list of recommendations
            goal_relevance_score = 5
            goal_relevance_explanation = 'No goal relevance analysis available'
            recommendations = response_data if isinstance(response_data, list) else []
        
        # Add goal-specific fields to options 1 & 2 when goal is provided
        if goal:
            for rec in recommendations:
                if rec.get('option_number', 0) in [1, 2]:
                    rec.update({
                        'relevance_score': rec.get('relevance_score', 5),
                        'goal_alignment': _ensure_string(rec.get('goal_alignment', 'General knowledge contribution to learning goal')),
                        'goal_priority': rec.get('goal_priority', 'medium'),
                        'is_goal_aware': True
                    })
                else:
                    rec['is_goal_aware'] = False
        
        print(f"✅ Generated {len(recommendations)} semantic knowledge recommendations")
        if goal:
            goal_aware_count = sum(1 for rec in recommendations if rec.get('is_goal_aware'))
            print(f"🎯 Goal-aligned recommendations: {goal_aware_count}/3")
            print(f"🎯 Overall goal relevance score: {goal_relevance_score}/10")
        
        return {
            "goal_relevance_score": goal_relevance_score,
            "goal_relevance_explanation": goal_relevance_explanation,
            "recommendations": recommendations
        }
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing LLM response as JSON: {e}")
        print(f"Raw response: {llm_response}")
        return {}
    except Exception as e:
        print(f"❌ Error calling Azure OpenAI: {e}")
        return {}