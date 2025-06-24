import openai
import json
from typing import List, Dict, Optional
from config.settings import settings
from utils.category_mapping import get_all_subject_categories, get_subject_categories_prompt_text, validate_subject_category, get_subject_category_fallback

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
            {"role": "system", "content": """You are a Goal-Oriented Semantic Knowledge Organizer.

Your role is to interpret new knowledge inputs and determine their optimal structural placement by comparing semantic similarity, generating actionable reorganization proposals, and enhancing categorization through MECE-based reasoning. You excel at goal-aware knowledge curation that aligns learning materials with specific objectives.

You interpret every task through these core principles:
- Structure reveals meaning
- Similarity invites nuance  
- Clarity supports reuse
- Coherence improves recall
- Categories evolve with context
- Goals shape relevance

You draw expertise from:
- Ontology Design
- Semantic Search
- Content Strategy
- Information Architecture
- Learning Sciences
- Goal-Oriented Knowledge Management

Your communication style is methodical, explanatory, and system-aware. You avoid ambiguity, vague tags, mechanical merging, rigid taxonomy, content dilution, and goal-irrelevant recommendations."""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=3500  # Increased for detailed responses
    )
    
    return response.choices[0].message.content

def LLMUpdater(input_text: str, existing_knowledge: Optional[Dict], goal: Optional[str] = None, llm_type: str = "azure_openai") -> List[Dict]:
    """
    Generate 4 different recommendations using Azure OpenAI with goal-based semantic analysis
    
    Args:
        input_text: New text input from user
        existing_knowledge: Most similar existing knowledge item (from SSC)
        goal: User's current learning/knowledge goal for contextual relevance analysis
        llm_type: Type of LLM to use (only "azure_openai" supported)
        
    Returns:
        List of 4 recommendation dictionaries with goal-based relevance analysis
    """
    if llm_type != "azure_openai":
        raise ValueError("Only 'azure_openai' is supported for llm_type")
    
    # Build semantic context for analysis
    context_sections = []
    
    # Input text analysis
    context_sections.append(f"INPUT TEXT (New Knowledge):\n{input_text}\n")
    
    # Goal context if provided
    if goal:
        context_sections.append(f"LEARNING GOAL:\n{goal}\n")
        context_sections.append("🎯 GOAL-RELEVANCE ANALYSIS REQUIRED for Options 1 & 2\n")
    
    # Existing knowledge context
    if existing_knowledge:
        similarity_score = existing_knowledge.get('similarity_score', 0)
        context_sections.append(f"EXISTING KNOWLEDGE (Similarity: {similarity_score:.3f}):")
        context_sections.append(f"Main Category: {existing_knowledge.get('main_category', 'Unknown')}")
        context_sections.append(f"Sub-Category: {existing_knowledge.get('sub_category', 'Unknown')}")
        context_sections.append(f"Content Preview: {existing_knowledge.get('content', '')[:300]}...")
        context_sections.append("")
    else:
        context_sections.append("EXISTING KNOWLEDGE: No similar content found (semantic distance > threshold)\n")
    
    # Academic categories context
    categories_text = get_subject_categories_prompt_text()
    context_sections.append(categories_text)
    
    # Construct the enhanced semantic prompt
    goal_instructions = ""
    goal_fields_example = ""
    if goal:
        goal_instructions = f"""
##### GOAL-ORIENTED ANALYSIS (Required for Options 1 & 2):

For recommendations 1 and 2, you MUST perform goal-relevance analysis:
- Assess how this knowledge supports achieving: "{goal}"
- Assign relevance_score (1-10) based on direct utility toward the goal
- Explain goal_alignment: WHY this knowledge matters for the goal
- Determine goal_priority (high/medium/low) for learning sequence
- Consider knowledge gaps, prerequisite concepts, and practical applications
- Evaluate if this knowledge accelerates, supports, or tangentially relates to goal achievement

RECOMMENDATION STRATEGY:
- Options 1 & 2: Goal-optimized approaches with relevance scoring
- Options 3 & 4: Semantic-optimal approaches without goal bias
"""
    else:
        goal_instructions = """
##### STANDARD SEMANTIC ANALYSIS:
No learning goal provided. All 4 recommendations will use semantic-optimal organization principles.
"""

    prompt = f"""##### CONTEXT:
You are a Goal-Oriented Semantic Knowledge Organizer analyzing new knowledge input.

{chr(10).join(context_sections)}

{goal_instructions}

##### INSTRUCTIONS:

1. **Examine the INPUT TEXT** as a new piece of information requiring optimal structural placement.

2. **Analyze semantic relationships** with EXISTING KNOWLEDGE using the similarity score. Consider whether to merge, append, restructure, or create new categories.

3. **Provide exactly 4 distinct recommendations** for handling this knowledge:
   - **Option 1**: {"Goal-optimized merge/update approach" if goal else "Semantic merge/update approach"}
   - **Option 2**: {"Goal-optimized replace/restructure approach" if goal else "Semantic replace/restructure approach"}  
   - **Option 3**: Comprehensive semantic organization (goal-agnostic)
   - **Option 4**: Specialized categorization approach (goal-agnostic)

4. **Ensure meaningful distinction** between all recommendations. Each must offer different structural benefits and trade-offs.

5. **Preserve content integrity**: Never summarize, paraphrase, or reduce detail. Maintain all nuance and specificity from the input text.

6. **Smart structural decisions**:
   - Only merge when semantic coherence is maintained
   - Create new sections/categories when content is sufficiently distinct
   - Suggest appropriate action_type: merge/update/create_new
   - Ensure logical information architecture

7. **Generate calibrated tags** (3-5 per recommendation):
   - Accurately describe subject matter (e.g., "machine-learning", "cognitive-psychology")
   - Use lowercase, hyphenated format for multi-word tags
   - Optimize for categorization and semantic search
   - Avoid generic terms like "new", "updated", "content"

8. **Academic category compliance**: Select main_category from the provided academic subjects list (exact match required).

##### OUTPUT FORMAT:

Respond with valid JSON in this exact structure:

[
  {{
    "option_number": 1,
    "change": "3-5 sentences explaining the recommended approach, rationale, and structural impact. Focus on goal-relevance and semantic optimization.",
    "updated_text": "Complete updated/merged/new content preserving all detail and nuance",
    "main_category": "EXACT academic subject name from categories list",
    "sub_category": "specific descriptive sub-topic within that academic subject",
    "tags": ["tag1", "tag2", "tag3", "tag4"],
    "preview": "First 100 characters of updated content...",
    "action_type": "merge/update/create_new",
    "reasoning": "Detailed explanation of why this structural approach optimizes knowledge organization",{goal_fields_example}
    "semantic_coherence": "high/medium/low - assessment of how well this maintains semantic relationships"
  }},
  {{
    "option_number": 2,
    "change": "3-5 sentences explaining the second approach with different structural trade-offs and benefits.",
    "updated_text": "Complete updated/restructured/new content maintaining full detail",
    "main_category": "EXACT academic subject name from categories list",
    "sub_category": "specific descriptive sub-topic within that academic subject",
    "tags": ["tag1", "tag2", "tag3", "tag4"],
    "preview": "First 100 characters of updated content...",
    "action_type": "merge/update/create_new",
    "reasoning": "Explanation of alternative structural benefits and organization logic",{goal_fields_example}
    "semantic_coherence": "high/medium/low - coherence assessment for this approach"
  }},
  {{
    "option_number": 3,
    "change": "3-5 sentences explaining comprehensive semantic organization without goal bias.",
    "updated_text": "Complete content organized for optimal semantic relationships",
    "main_category": "EXACT academic subject name from categories list",
    "sub_category": "specific descriptive sub-topic within that academic subject",
    "tags": ["tag1", "tag2", "tag3", "tag4"],
    "preview": "First 100 characters of updated content...",
    "action_type": "merge/update/create_new",
    "reasoning": "Pure semantic optimization rationale",
    "semantic_coherence": "high/medium/low - assessment of semantic relationship quality"
  }},
  {{
    "option_number": 4,
    "change": "3-5 sentences explaining specialized categorization approach with unique structural benefits.",
    "updated_text": "Complete content organized with specialized categorical focus",
    "main_category": "EXACT academic subject name from categories list",
    "sub_category": "specific descriptive sub-topic within that academic subject",
    "tags": ["tag1", "tag2", "tag3", "tag4"],
    "preview": "First 100 characters of updated content...",
    "action_type": "merge/update/create_new",
    "reasoning": "Specialized organization benefits and use case rationale",
    "semantic_coherence": "high/medium/low - coherence assessment for specialized approach"
  }}
]

##### CRITICAL REQUIREMENTS:

- **Academic Compliance**: main_category must exactly match academic subjects list
- **Content Preservation**: Never reduce detail, summarize, or paraphrase input content
- **Structural Integrity**: Ensure logical information architecture and semantic coherence
- **Goal Fields**: Include relevance_score, goal_alignment, goal_priority ONLY in options 1 & 2 when goal provided
- **Tag Quality**: Generate meaningful, searchable tags avoiding generic terms
- **JSON Validity**: Ensure proper formatting and escaped characters
- **Distinct Approaches**: Each recommendation must offer meaningfully different structural benefits"""

    try:
        # Get response from Azure OpenAI
        llm_response = call_azure_openai_llm(prompt)
        
        # Parse the JSON response
        recommendations = json.loads(llm_response)
        
        # Validate and enhance each recommendation
        enhanced_recommendations = []
        for rec in recommendations:
            main_category = rec.get('main_category', '')
            sub_category = rec.get('sub_category', 'general knowledge')
            
            # Validate main category (academic subject)
            if not validate_subject_category(main_category):
                print(f"⚠️  Warning: Invalid academic subject '{main_category}', using fallback")
                main_category = get_subject_category_fallback(sub_category)
            
            enhanced_rec = {
                'option_number': rec.get('option_number', len(enhanced_recommendations) + 1),
                'change': rec.get('change', 'Semantic knowledge organization'),
                'updated_text': rec.get('updated_text', input_text),
                'main_category': main_category,
                'sub_category': sub_category,
                'tags': rec.get('tags', []),
                'preview': rec.get('preview', rec.get('updated_text', '')[:100] + '...'),
                'action_type': rec.get('action_type', 'create_new'),
                'reasoning': rec.get('reasoning', 'Semantic optimization approach'),
                'semantic_coherence': rec.get('semantic_coherence', 'medium')
            }
            
            # Add goal-specific fields only for options 1 & 2 when goal is provided
            if goal and rec.get('option_number', 0) in [1, 2]:
                enhanced_rec.update({
                    'relevance_score': rec.get('relevance_score', 5),
                    'goal_alignment': rec.get('goal_alignment', 'General knowledge contribution to learning goal'),
                    'goal_priority': rec.get('goal_priority', 'medium'),
                    'is_goal_aware': True
                })
            else:
                enhanced_rec['is_goal_aware'] = False
            
            enhanced_recommendations.append(enhanced_rec)
        
        # Ensure we have exactly 4 recommendations
        while len(enhanced_recommendations) < 4:
            enhanced_recommendations.append(create_fallback_recommendation(
                input_text, existing_knowledge, len(enhanced_recommendations) + 1, goal
            ))
        
        print(f"✅ Generated {len(enhanced_recommendations)} semantic knowledge recommendations")
        if goal:
            goal_aware_count = sum(1 for rec in enhanced_recommendations if rec.get('is_goal_aware'))
            print(f"🎯 Goal-aligned recommendations: {goal_aware_count}/4")
        
        # Log semantic coherence assessment
        coherence_scores = [rec.get('semantic_coherence', 'medium') for rec in enhanced_recommendations]
        high_coherence = coherence_scores.count('high')
        print(f"🔗 High semantic coherence: {high_coherence}/4 recommendations")
        
        return enhanced_recommendations
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing LLM response as JSON: {e}")
        print(f"Raw response: {llm_response[:300]}...")
        return create_fallback_recommendations(input_text, existing_knowledge, goal)
    except Exception as e:
        print(f"❌ Error calling Azure OpenAI: {e}")
        return create_fallback_recommendations(input_text, existing_knowledge, goal)

def create_fallback_recommendation(input_text: str, existing_knowledge: Optional[Dict], option_number: int, goal: Optional[str] = None) -> Dict:
    """
    Create a single fallback recommendation with semantic awareness
    
    Args:
        input_text: The input text
        existing_knowledge: Existing knowledge item if any
        option_number: The recommendation number
        goal: Learning goal if provided
        
    Returns:
        Single fallback recommendation dictionary
    """
    main_category = get_subject_category_fallback(input_text)
    sub_category = f'general knowledge {option_number}'
    
    # Determine action type based on semantic similarity
    if existing_knowledge and existing_knowledge.get('similarity_score', 0) > 0.85:
        action_type = 'merge' if option_number % 2 == 1 else 'update'
    elif existing_knowledge and existing_knowledge.get('similarity_score', 0) > 0.7:
        action_type = 'update'
    else:
        action_type = 'create_new'
    
    # Generate semantic-aware tags
    basic_tags = ['knowledge-management', f'option-{option_number}']
    if goal:
        basic_tags.append('goal-oriented')
    
    recommendation = {
        'option_number': option_number,
        'change': f'Fallback semantic organization approach {option_number}. This recommendation provides basic knowledge structuring while maintaining content integrity and semantic relationships.',
        'updated_text': input_text,
        'main_category': main_category,
        'sub_category': sub_category,
        'tags': basic_tags,
        'preview': input_text[:100] + '...' if len(input_text) > 100 else input_text,
        'action_type': action_type,
        'reasoning': f'Fallback recommendation applying semantic organization principles with {action_type} strategy',
        'semantic_coherence': 'medium',
        'is_goal_aware': False
    }
    
    # Add goal-specific fields for options 1 & 2 when goal is provided
    if goal and option_number in [1, 2]:
        recommendation.update({
            'relevance_score': 5,  # Neutral score for fallback
            'goal_alignment': f'Fallback assessment: Knowledge may contribute to goal "{goal}" but requires manual evaluation',
            'goal_priority': 'medium',
            'is_goal_aware': True
        })
    
    return recommendation

def create_fallback_recommendations(input_text: str, existing_knowledge: Optional[Dict], goal: Optional[str] = None) -> List[Dict]:
    """
    Create 4 fallback recommendations with semantic organization principles
    
    Args:
        input_text: The input text
        existing_knowledge: Existing knowledge item if any
        goal: Learning goal if provided
        
    Returns:
        List of 4 fallback recommendations
    """
    recommendations = []
    
    for i in range(1, 5):
        recommendation = create_fallback_recommendation(input_text, existing_knowledge, i, goal)
        recommendations.append(recommendation)
    
    main_category = get_subject_category_fallback(input_text)
    print(f"⚠️  Using semantic fallback recommendations with academic subject: {main_category}")
    if goal:
        print(f"🎯 Goal-aware fallback recommendations: 2/4 (options 1 & 2)")
    print(f"🔗 Semantic organization: All recommendations maintain content integrity")
    
    return recommendations