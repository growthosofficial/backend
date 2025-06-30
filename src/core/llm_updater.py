import openai
import json
import pickle
import os
from typing import List, Dict, Optional
from config.settings import settings
from utils.category_mapping import get_all_subject_categories, get_subject_categories_prompt_text, validate_subject_category, get_subject_category_fallback
from core.similarity import SSC

# Cache for main category embeddings to avoid recalculating
_MAIN_CATEGORY_EMBEDDINGS_CACHE = {}
_EMBEDDINGS_CACHE_FILE = "main_category_embeddings.pkl"

def _clean_sub_category(sub_category: str) -> str:
    """
    Clean sub-category to ensure it only contains the sub-category name, not the full path
    Removes main category prefix if present (e.g., 'Engineering > naval operations' -> 'naval operations')
    """
    if not sub_category:
        return "General"
    
    # If sub_category contains ' > ', split and take the part after it
    if ' > ' in sub_category:
        parts = sub_category.split(' > ')
        if len(parts) >= 2:
            return parts[-1].strip()  # Take the last part (the actual sub-category)
    
    return sub_category.strip()

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

def precompute_main_category_embeddings():
    """
    Pre-compute and save embeddings for all main categories
    This should be run once during setup/deployment
    """
    from core.embeddings import get_embedding
    main_categories = get_all_subject_categories()
    
    print("🔄 Pre-computing main category embeddings...")
    embeddings = {}
    
    for i, category in enumerate(main_categories, 1):
        print(f"  Computing embedding {i}/{len(main_categories)}: {category}")
        embeddings[category] = get_embedding(category)
    
    # Save to file
    with open(_EMBEDDINGS_CACHE_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"✅ Saved embeddings for {len(main_categories)} main categories to {_EMBEDDINGS_CACHE_FILE}")
    return embeddings

def _get_main_category_embeddings():
    """
    Get embeddings for all main categories, using cache for efficiency
    OPTIMIZED: Loads from pre-computed file if available
    """
    global _MAIN_CATEGORY_EMBEDDINGS_CACHE
    
    if not _MAIN_CATEGORY_EMBEDDINGS_CACHE:
        # Try to load from pre-computed file first
        if os.path.exists(_EMBEDDINGS_CACHE_FILE):
            try:
                print(f"📂 Loading pre-computed embeddings from {_EMBEDDINGS_CACHE_FILE}...")
                with open(_EMBEDDINGS_CACHE_FILE, 'rb') as f:
                    _MAIN_CATEGORY_EMBEDDINGS_CACHE = pickle.load(f)
                print(f"✅ Loaded {len(_MAIN_CATEGORY_EMBEDDINGS_CACHE)} cached embeddings")
            except Exception as e:
                print(f"⚠️ Failed to load cached embeddings: {e}")
                print("🔄 Computing embeddings on first use...")
                _MAIN_CATEGORY_EMBEDDINGS_CACHE = precompute_main_category_embeddings()
        else:
            print("🔄 No pre-computed embeddings found. Computing on first use...")
            _MAIN_CATEGORY_EMBEDDINGS_CACHE = precompute_main_category_embeddings()
    
    return _MAIN_CATEGORY_EMBEDDINGS_CACHE

def _map_subcategory_to_main_category_fast(sub_category: str) -> str:
    """
    Fast keyword-based mapping for immediate results
    Used as fallback while embeddings are being computed
    """
    if not sub_category:
        return "General Studies"
    
    sub_lower = sub_category.lower()
    
    # Quick keyword matching (much faster than embeddings)
    keyword_mapping = {
        # STEM
        'math': 'Mathematics', 'algebra': 'Mathematics', 'calculus': 'Mathematics', 'geometry': 'Mathematics',
        'physics': 'Physics', 'quantum': 'Physics', 'mechanics': 'Physics', 'relativity': 'Physics',
        'chemistry': 'Chemistry', 'chemical': 'Chemistry', 'molecule': 'Chemistry', 'reaction': 'Chemistry',
        'biology': 'Biology', 'biological': 'Biology', 'cell': 'Biology', 'genetic': 'Biology', 'organism': 'Biology',
        'computer': 'Computer Science', 'programming': 'Computer Science', 'software': 'Computer Science', 'algorithm': 'Computer Science',
        'engineering': 'Engineering', 'mechanical': 'Engineering', 'electrical': 'Engineering', 'civil': 'Engineering',
        'medicine': 'Medicine', 'medical': 'Medicine', 'health': 'Medicine', 'clinical': 'Medicine', 'treatment': 'Medicine',
        'environment': 'Environmental Science', 'climate': 'Environmental Science', 'ecology': 'Environmental Science',
        'astronomy': 'Astronomy', 'space': 'Astronomy', 'planet': 'Astronomy', 'star': 'Astronomy',
        'geology': 'Geology', 'earth': 'Geology', 'rock': 'Geology', 'mineral': 'Geology',
        
        # Social Sciences
        'psychology': 'Psychology', 'psychological': 'Psychology', 'behavior': 'Psychology', 'mental': 'Psychology',
        'sociology': 'Sociology', 'social': 'Sociology', 'society': 'Sociology', 'culture': 'Sociology',
        'anthropology': 'Anthropology', 'ethnography': 'Anthropology', 'cultural': 'Anthropology',
        'politics': 'Political Science', 'political': 'Political Science', 'government': 'Political Science',
        'economics': 'Economics', 'economic': 'Economics', 'economy': 'Economics', 'market': 'Economics',
        'geography': 'Geography', 'geographical': 'Geography', 'map': 'Geography', 'location': 'Geography',
        'language': 'Linguistics', 'linguistic': 'Linguistics', 'grammar': 'Linguistics', 'syntax': 'Linguistics',
        'archaeology': 'Archaeology', 'artifact': 'Archaeology', 'excavation': 'Archaeology',
        
        # Humanities
        'history': 'History', 'historical': 'History', 'past': 'History', 'ancient': 'History',
        'philosophy': 'Philosophy', 'philosophical': 'Philosophy', 'ethics': 'Philosophy', 'logic': 'Philosophy',
        'literature': 'Literature', 'literary': 'Literature', 'novel': 'Literature', 'poetry': 'Literature',
        'religion': 'Religious Studies', 'religious': 'Religious Studies', 'theology': 'Religious Studies',
        'art': 'Art History', 'artistic': 'Art History', 'painting': 'Art History', 'sculpture': 'Art History',
        'music': 'Music Theory', 'musical': 'Music Theory', 'composition': 'Music Theory', 'melody': 'Music Theory',
        'theater': 'Theater Arts', 'drama': 'Theater Arts', 'performance': 'Theater Arts', 'acting': 'Theater Arts',
        
        # Applied & Professional
        'business': 'Business Administration', 'management': 'Business Administration', 'strategy': 'Business Administration',
        'law': 'Law', 'legal': 'Law', 'court': 'Law', 'justice': 'Law',
        'education': 'Education', 'teaching': 'Education', 'learning': 'Education', 'pedagogy': 'Education',
        'communication': 'Communications', 'media': 'Communications', 'journalism': 'Communications',
        'architecture': 'Architecture', 'architectural': 'Architecture', 'building': 'Architecture',
        'agriculture': 'Agriculture', 'farming': 'Agriculture', 'crop': 'Agriculture',
        'nutrition': 'Nutrition', 'diet': 'Nutrition', 'food': 'Nutrition',
        
        # Arts & Creative
        'visual': 'Visual Arts', 'graphic': 'Visual Arts', 'design': 'Visual Arts', 'illustration': 'Visual Arts',
        'writing': 'Creative Writing', 'creative': 'Creative Writing', 'story': 'Creative Writing', 'narrative': 'Creative Writing',
        'film': 'Film Studies', 'movie': 'Film Studies', 'cinema': 'Film Studies', 'video': 'Film Studies',
        'photography': 'Photography', 'photo': 'Photography', 'camera': 'Photography',
        
        # Health & Physical
        'public health': 'Public Health', 'epidemiology': 'Public Health',
        'exercise': 'Physical Education', 'fitness': 'Physical Education', 'sport': 'Physical Education',
        'therapy': 'Therapy & Rehabilitation', 'rehabilitation': 'Therapy & Rehabilitation',
        
        # Modern/Interdisciplinary
        'data': 'Data Science', 'analytics': 'Data Science', 'big data': 'Data Science',
        'security': 'Cybersecurity', 'cyber': 'Cybersecurity', 'encryption': 'Cybersecurity',
        'biotechnology': 'Biotechnology', 'biotech': 'Biotechnology',
        'cognitive': 'Cognitive Science', 'neuroscience': 'Cognitive Science', 'brain': 'Cognitive Science',
        'international': 'International Studies', 'global': 'International Studies',
        'gender': 'Gender Studies', 'feminist': 'Gender Studies',
        'urban': 'Urban Planning', 'city planning': 'Urban Planning'
    }
    
    # Check for exact matches first
    for keyword, category in keyword_mapping.items():
        if keyword in sub_lower:
            return category
    
    return "General Studies"

def _map_subcategory_to_main_category(sub_category: str, use_fast_fallback: bool = True) -> str:
    """
    Map a sub-category to the most semantically similar main category using embeddings
    OPTIMIZED: Uses cached main category embeddings for efficiency
    FALLBACK: Uses fast keyword matching if embeddings not available
    """
    if not sub_category:
        return "General Studies"
    
    # Try to get cached embeddings first
    try:
        main_category_embeddings = _get_main_category_embeddings()
        
        # If we have embeddings, use semantic similarity
        if main_category_embeddings:
            # Get embedding for the sub-category
            from core.embeddings import get_embedding
            sub_category_embedding = get_embedding(sub_category)
            
            best_match = "General Studies"
            highest_similarity = 0.0
            
            # Calculate cosine similarity
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            emb1 = np.array(sub_category_embedding).reshape(1, -1)
            
            # Compare with each main category
            for main_cat, main_cat_embedding in main_category_embeddings.items():
                emb2 = np.array(main_cat_embedding).reshape(1, -1)
                similarity = cosine_similarity(emb1, emb2)[0][0]
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = main_cat
            
            return best_match
    
    except Exception as e:
        print(f"⚠️ Semantic mapping failed: {e}")
    
    # Fallback to fast keyword matching
    if use_fast_fallback:
        print("🔄 Using fast keyword-based fallback mapping...")
        return _map_subcategory_to_main_category_fast(sub_category)
    
    return "General Studies"

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
- Specific and actionable TEXT TRANSFORMATION steps
- Clear commands for modifying, restructuring, or combining existing text
- Detailed enough for another LLM to execute precisely
- Focused ONLY on text transformation, restructuring, and content organization

TEXT TRANSFORMATION FOCUS:
- How to modify existing sentences and paragraphs
- How to restructure the flow and organization
- How to combine or separate content
- How to improve readability and clarity
- How to remove redundancy or add transitions

AVOID:
- Content creation (adding new information)
- Analysis or interpretation
- Suggestions for external resources
- Structural recommendations without specific text changes

You do NOT transform the text yourself - you create detailed instructions for how another LLM should transform it.
"""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1000 # Reduced for faster responses
    )
    
    content = response.choices[0].message.content
    if content is None:
        return ""
    return content

def LLMUpdater(input_text: str, existing_knowledge: Optional[Dict], goal: Optional[str] = None, llm_type: str = "azure_openai", similarity_threshold: float = 0.8) -> Dict:
    """
    Generate 3 different recommendations using Azure OpenAI with goal-based semantic analysis
    OPTIMIZED: LLM only generates sub-categories, main categories are mapped via semantic similarity
    
    Args:
        input_text: New text input from user
        existing_knowledge: Most similar existing knowledge item (from SSC)
        goal: User's current learning/knowledge goal for contextual relevance analysis
        llm_type: Type of LLM to use (only "azure_openai" supported)
        similarity_threshold: Threshold for determining when to use exact existing categories (default: 0.8)
        
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
        context_sections.append(f"SIMILAR EXISTING KNOWLEDGE (Score: {similarity_score:.3f}):, Threshold: {similarity_threshold}")
        context_sections.append(f"Category: {existing_knowledge.get('main_category', 'Unknown')} > {existing_knowledge.get('sub_category', 'Unknown')}")
        context_sections.append(f"Content: {existing_knowledge.get('content', '')[:200]}...")
        context_sections.append("")
    else:
        context_sections.append("SIMILAR EXISTING KNOWLEDGE: None found\n")
    
    # Construct the enhanced semantic prompt
    goal_instructions = ""
    if goal:
        goal_instructions = f"""

For options 1 & 2:
1. Evaluate input relevance to goal (input text relevance to goal should be the same for both options 1 and 2)
2. Add ONLY goal-relevant content transformation instructions
3. Include: relevance_score (1-10), goal_alignment, goal_priority (high/medium/low)

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

CRITICAL CATEGORY RULES:
1. If SIMILAR EXISTING KNOWLEDGE is found (similarity score > {similarity_threshold}), you MUST use the EXACT existing main_category and sub_category for at least 2 out of 3 recommendations
2. DO NOT create new sub-categories when similar existing knowledge is provided - use the exact ones provided
3. Only create new sub-categories if no similar existing knowledge is found

INSTRUCTIONS:
1. Evaluate the input's relevance to the goal (if provided) and assign a goal_relevance_score (1-10)
2. Evaluate the input's relevance to the similar existing knowledge (if provided)

SPECIFIC RULES:
1. sub_category: If SIMILAR EXISTING KNOWLEDGE exists, use the EXACT sub_category provided. Only create new sub-categories if no similar knowledge exists. New sub-categories should be based on the main idea of the input text. 
2. action_type: 
   - If SIMILAR EXISTING KNOWLEDGE NOT found, **create_new** (always)
   - If SIMILAR EXISTING KNOWLEDGE found, **merge** if instructions are to append to existing knowledge, **update** if input is mostly redundant to existing knowledge
   - CRITICAL: When creating new sub-categories, action_type MUST be **create_new**
3. tags: 3-4 meaningful tags per recommendation (tags should be related to the content and category)
4. instructions: Transform the input text by performing SPECIFIC TEXT TRANSFORMATION STEPS. Focus ONLY on how to modify, restructure, or combine the actual text content.

IMPORTANT: Make instructions as specific and actionable as possible. Instead of "add bullet points," say "add bullet points to list [specific items]." Instead of "restructure paragraphs," say "restructure paragraphs by combining sentences about [specific topic]."

VALID TEXT TRANSFORMATION INSTRUCTIONS:
- "Restructure paragraphs by combining related sentences about [specific topic]"
- "Add bullet points to list [specific items] in chronological order"
- "Convert passive voice sentences to active voice throughout the text"
- "Merge duplicate information about [specific topic] and remove redundancy"
- "Add transitional phrases like 'Furthermore,' 'Additionally,' between paragraphs"
- "Break long paragraphs about [specific topic] into shorter ones of 2-3 sentences"
- "Reorganize content chronologically, starting with [event] and ending with [event]"
- "Combine sentences about [topic A] and [topic B] into single paragraphs"
- "Add subheadings: '[Heading 1]', '[Heading 2]', '[Heading 3]' to organize content"
- "Remove unnecessary details about [specific topic] and focus on key points"
- "Split compound sentences about [topic] into shorter, clearer sentences"
- "Group related information about [topic] into dedicated paragraphs"
- "Add introductory sentences to each paragraph for better flow"
- "Convert long lists into bullet points for [specific items]"

INVALID INSTRUCTIONS (DO NOT USE):
- "Create a section titled..." (content creation)
- "Include diagrams or illustrations" (not text transformation)
- "Add references to other ships" (content addition)
- "Compare and contrast with..." (analysis, not transformation)
- "Highlight the significance of..." (interpretation, not transformation)
- "Summarize the importance of..." (content creation)

5. change: 2-3 sentences explaining the instruction approach of the recommendation

OUTPUT FORMAT (MUST BE VALID JSON):
{{
 "goal_relevance_score": 1-10,
 "goal_relevance_explanation": "Brief explanation of why this score was assigned",
 "recommendations": [
   {{
     "option_number": 1,
     "sub_category": "",
     "action_type": "",
     "tags": ["tag1", "tag2", "tag3"],
     "instructions": "",
     "change": ""
   }},
   {{
     "option_number": 2,
     "sub_category": "",
     "action_type": "",
     "tags": ["tag1", "tag2", "tag3"],
     "instructions": "",
     "change": ""
   }},
   {{
     "option_number": 3,
     "sub_category": "",
     "action_type": "",
     "tags": ["tag1", "tag2", "tag3"],
     "instructions": "",
     "change": ""
   }}
 ]
}}
"""

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
        
        # OPTIMIZATION: Map sub-categories to main categories using semantic similarity
        # BUT: If similar existing knowledge is found, use exact categories for at least 2 recommendations
        existing_main_category = existing_knowledge.get('main_category') if existing_knowledge else None
        existing_sub_category = existing_knowledge.get('sub_category') if existing_knowledge else None
        similarity_score = existing_knowledge.get('similarity_score', 0) if existing_knowledge else 0
        
        # Clean existing sub-category to ensure it's just the sub-category name
        if existing_sub_category:
            existing_sub_category = _clean_sub_category(existing_sub_category)
        
        # If we have high similarity existing knowledge, use exact categories for first 2 recommendations
        if existing_knowledge and similarity_score > similarity_threshold:
            exact_categories_used = 0
            for i, rec in enumerate(recommendations):
                sub_category = rec.get('sub_category', 'General')
                
                # Clean the sub-category from LLM response
                cleaned_sub_category = _clean_sub_category(sub_category)
                rec['sub_category'] = cleaned_sub_category
                
                # For first 2 recommendations, try to use exact existing categories
                if exact_categories_used < 2 and existing_sub_category and cleaned_sub_category.lower() == existing_sub_category.lower():
                    rec['main_category'] = existing_main_category
                    rec['sub_category'] = existing_sub_category  # Ensure exact match
                    exact_categories_used += 1
                    print(f"✅ Using exact existing categories for option {i+1}: {existing_main_category} > {existing_sub_category}")
                else:
                    # Use semantic mapping for other cases
                    main_category = _map_subcategory_to_main_category(cleaned_sub_category)
                    rec['main_category'] = main_category
        else:
            # No similar existing knowledge, use semantic mapping for all
            for rec in recommendations:
                sub_category = rec.get('sub_category', 'General')
                cleaned_sub_category = _clean_sub_category(sub_category)
                rec['sub_category'] = cleaned_sub_category
                main_category = _map_subcategory_to_main_category(cleaned_sub_category)
                rec['main_category'] = main_category
        
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
        
        # VALIDATION: Ensure action_type is consistent with existing knowledge and sub-categories
        UPDATE_SIMILARITY_THRESHOLD = 0.7  # Higher threshold required for UPDATE operations
        
        for rec in recommendations:
            action_type = rec.get('action_type', 'create_new')
            sub_category = rec.get('sub_category', 'General')
            
            # If no similar existing knowledge found, action_type should be create_new
            if not existing_knowledge:
                if action_type != 'create_new':
                    print(f"⚠️ Fixing inconsistent action_type: {action_type} -> create_new (no existing knowledge)")
                    rec['action_type'] = 'create_new'
            
            # If using new sub-categories (not exact existing ones), action_type should be create_new
            elif existing_sub_category and sub_category.lower() != existing_sub_category.lower():
                if action_type != 'create_new':
                    print(f"⚠️ Fixing inconsistent action_type: {action_type} -> create_new (new sub-category)")
                    rec['action_type'] = 'create_new'
            
            # If using exact existing sub-category, validate action_type based on similarity threshold
            elif existing_sub_category and sub_category.lower() == existing_sub_category.lower():
                if action_type == 'create_new':
                    # Determine if it should be merge or update based on instructions
                    instructions = rec.get('instructions', '').lower()
                    if 'append' in instructions or 'add' in instructions or 'incorporate' in instructions:
                        new_action_type = 'merge'
                    else:
                        new_action_type = 'update'
                    print(f"⚠️ Fixing inconsistent action_type: {action_type} -> {new_action_type} (existing sub-category)")
                    rec['action_type'] = new_action_type
                
                # Additional validation: UPDATE operations require higher similarity threshold
                elif action_type == 'update' and similarity_score < UPDATE_SIMILARITY_THRESHOLD:
                    print(f"⚠️ Fixing UPDATE action_type: {action_type} -> merge (similarity {similarity_score:.3f} < {UPDATE_SIMILARITY_THRESHOLD} threshold)")
                    rec['action_type'] = 'merge'
                
                # Log the final action_type decision
                if action_type in ['merge', 'update']:
                    print(f"✅ Action type {action_type} validated for similarity {similarity_score:.3f} (threshold: {similarity_threshold})")
        
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