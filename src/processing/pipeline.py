from typing import Optional

from core.similarity import SSC
from core.llm_updater import LLMUpdater
from core.embeddings import get_embedding
from database.supabase_manager import supabase_manager
from processing.file_handler import read_input_from_file
from config.settings import settings

def process_input_text(
    input_text: str, 
    similarity_threshold: float = None,
    llm_type: str = None, 
    database_folder: str = None  # Ignored but kept for compatibility
) -> None:
    """
    Process input text through the similarity checking and update pipeline using Supabase
    
    Args:
        input_text: Text to process
        similarity_threshold: Minimum similarity to consider existing knowledge
        llm_type: Must be "azure_openai" (only supported option)
        database_folder: Ignored (kept for compatibility)
    """
    # Use defaults from settings if not provided
    if similarity_threshold is None:
        similarity_threshold = settings.DEFAULT_SIMILARITY_THRESHOLD
    if llm_type is None:
        llm_type = settings.DEFAULT_LLM_TYPE
    
    # Validate LLM type
    if llm_type != "azure_openai":
        print(f"⚠️  Warning: '{llm_type}' is not supported. Using 'azure_openai' instead.")
        llm_type = "azure_openai"
    
    print(f"Processing input: {input_text[:50]}...")
    
    # Get the most similar existing knowledge category from Supabase
    most_similar = SSC(input_text, similarity_threshold)
    
    if most_similar:
        print(f"Most similar category: '{most_similar['category']}' (similarity: {most_similar['similarity_score']:.3f})")
    else:
        print("No similar category found above threshold.")
    
    # Get 3 recommendations from Azure OpenAI
    print(f"Getting recommendations from {llm_type.upper()}...")
    recommendations = LLMUpdater(input_text, most_similar, llm_type)
    
    # Display options to user
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}: {rec['change']}")
        print(f"   Category: {rec['category']}")
        print(f"   Preview: {rec['updated_text'][:100]}...")
    
    # Get user decision
    decision = input(f"\nPick an option (1-3): ").strip()
    
    # Process decision
    try:
        option_num = int(decision)
        if 1 <= option_num <= 3:
            selected_rec = recommendations[option_num - 1]
            
            # Create complete knowledge item
            knowledge_item = {
                'category': selected_rec['category'],
                'content': selected_rec['updated_text'],
                'tags': ['updated'] if most_similar else ['new'],
                'embedding': get_embedding(selected_rec['updated_text'])
            }
            
            # Add/update the knowledge item in Supabase
            result = supabase_manager.add_knowledge_item(knowledge_item)
            print(f"✅ Applied recommendation {option_num}")
            print(f"📁 Updated category: {selected_rec['category']}")
            print(f"🆔 Record ID: {result.get('id') if result else 'Unknown'}")
        else:
            print("Invalid option number (1-3)")
    except ValueError:
        print("Invalid input, please enter 1, 2, or 3")

def process_file_input(
    file_path: str, 
    similarity_threshold: float = None,
    llm_type: str = None, 
    database_folder: str = None
) -> None:
    """
    Process input text from a file through the similarity checking and update pipeline
    
    Args:
        file_path: Path to the .txt file containing input text
        similarity_threshold: Minimum similarity to consider existing knowledge
        llm_type: Must be "azure_openai" (only supported option)
        database_folder: Ignored (kept for compatibility)
    """
    # Read text from file
    input_text = read_input_from_file(file_path)
    
    if input_text is None:
        return
    
    if not input_text:
        print("❌ File is empty")
        return
    
    # Process the text through the normal pipeline
    process_input_text(input_text, similarity_threshold, llm_type, database_folder)