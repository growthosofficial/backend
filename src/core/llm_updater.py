import json
import logging
from typing import List, Dict, Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMUpdater:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4"

    def create_recommendation_prompt(self, input_text: str, existing_item: Optional[Dict]) -> str:
        """Create a structured prompt for generating recommendations."""
        
        if existing_item:
            existing_section = f"""
EXISTING SIMILAR KNOWLEDGE:
Category: {existing_item.get('category', 'N/A')}
Content: {existing_item.get('content', 'N/A')}
Tags: {', '.join(existing_item.get('tags', []))}
Similarity Score: {existing_item.get('similarity_score', 'N/A'):.3f}
"""
        else:
            existing_section = "EXISTING SIMILAR KNOWLEDGE: None found"

        prompt = f"""You are helping manage a MECE (Mutually Exclusive, Collectively Exhaustive) knowledge database.

INPUT TEXT: {input_text}

{existing_section}

Please provide exactly 3 different recommendations for how to handle this new information. Each recommendation should follow a different strategic approach:

1. **Merge/Update Approach**: How to combine with existing knowledge (if any exists)
2. **Replace/Revise Approach**: How to improve or replace existing content (if any exists)  
3. **New Category Approach**: Create a separate new category for this information

For each recommendation, provide:
- Clear explanation of what changes would be made
- Complete updated/new text content (well-formatted and comprehensive)
- Appropriate category name (descriptive and specific)
- 3-5 relevant content-based tags (focus on key concepts and topics)
- A brief preview/summary of the content
- Ensure MECE principles are maintained

CRITICAL: Respond with ONLY a valid JSON array containing exactly 3 objects. Each object must have these exact keys:
- "change": string explaining the approach
- "updated_text": string with complete content
- "category": string with category name
- "tags": array of strings (3-5 tags)
- "preview": string with brief summary

Example format:
[
  {{
    "change": "Detailed explanation of the merge approach...",
    "updated_text": "Complete merged content here...",
    "category": "Specific Category Name",
    "tags": ["tag1", "tag2", "tag3", "tag4"],
    "preview": "Brief summary of the merged content..."
  }},
  ... (2 more objects)
]

Respond with JSON only, no additional text or formatting."""

        return prompt

    async def LLMUpdater(self, input_text: str, existing_item: Optional[Dict], 
                        llm_type: str = "openai") -> List[Dict]:
        """
        Generate 3 AI-powered recommendations for handling input text.
        
        Args:
            input_text: New text to process
            existing_item: Most similar existing knowledge item (if any)
            llm_type: LLM provider (currently only "openai" supported)
            
        Returns:
            List of 3 recommendation dictionaries
        """
        try:
            if llm_type != "openai":
                raise ValueError(f"Unsupported LLM type: {llm_type}")

            # Create prompt
            prompt = self.create_recommendation_prompt(input_text, existing_item)
            
            logger.info("Generating LLM recommendations...")
            
            # Call OpenAI GPT-4
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert knowledge management assistant. Always respond with valid JSON only."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            try:
                recommendations = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"Raw response: {content}")
                raise Exception("LLM returned invalid JSON format")
            
            # Validate response structure
            if not isinstance(recommendations, list):
                raise Exception("LLM response must be a JSON array")
            
            if len(recommendations) != 3:
                raise Exception(f"Expected 3 recommendations, got {len(recommendations)}")
            
            # Validate each recommendation
            required_keys = ["change", "updated_text", "category", "tags", "preview"]
            for i, rec in enumerate(recommendations):
                if not isinstance(rec, dict):
                    raise Exception(f"Recommendation {i+1} must be an object")
                
                for key in required_keys:
                    if key not in rec:
                        raise Exception(f"Recommendation {i+1} missing required key: {key}")
                
                # Ensure tags is a list
                if not isinstance(rec["tags"], list):
                    raise Exception(f"Recommendation {i+1} 'tags' must be an array")
                
                # Add option number
                rec["option_number"] = i + 1
            
            logger.info(f"Successfully generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"LLMUpdater failed: {e}")
            
            # Return fallback recommendations if LLM fails
            fallback_recommendations = self.create_fallback_recommendations(input_text, existing_item)
            logger.warning("Using fallback recommendations due to LLM failure")
            return fallback_recommendations

    def create_fallback_recommendations(self, input_text: str, existing_item: Optional[Dict]) -> List[Dict]:
        """Create basic fallback recommendations if LLM fails."""
        
        # Generate basic category name from first few words
        words = input_text.split()[:3]
        basic_category = " ".join(words).title() if words else "General Knowledge"
        
        # Generate basic tags from input text
        common_words = [word.lower().strip('.,!?;:') for word in input_text.split() 
                       if len(word) > 3 and word.lower() not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'she', 'use', 'your', 'said', 'each', 'make', 'most', 'over', 'said', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were']]
        basic_tags = list(set(common_words[:5]))
        
        if len(basic_tags) < 3:
            basic_tags.extend(['knowledge', 'information', 'data'])
        
        recommendations = [
            {
                "option_number": 1,
                "change": "Store as new knowledge item with automatic categorization",
                "updated_text": input_text,
                "category": basic_category,
                "tags": basic_tags[:4],
                "preview": input_text[:100] + "..." if len(input_text) > 100 else input_text
            },
            {
                "option_number": 2,
                "change": "Create comprehensive knowledge entry with enhanced formatting",
                "updated_text": f"# {basic_category}\n\n{input_text}\n\n*Added to knowledge base for future reference.*",
                "category": f"Enhanced {basic_category}",
                "tags": basic_tags[:4] + ["enhanced"],
                "preview": f"Enhanced entry for {basic_category} with structured formatting"
            },
            {
                "option_number": 3,
                "change": "Store with general classification for broad accessibility",
                "updated_text": input_text,
                "category": "General Knowledge",
                "tags": ["general", "uncategorized"] + basic_tags[:3],
                "preview": "General knowledge entry for later categorization and refinement"
            }
        ]
        
        return recommendations