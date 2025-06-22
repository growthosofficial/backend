import openai
from typing import List
from config.settings import settings

def get_embedding(text: str) -> List[float]:
    """
    Get Azure OpenAI embedding for text
    
    Args:
        text: Input text to embed
        
    Returns:
        List of embedding values
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
    
    clean_text = text.replace('\n', ' ').strip()
    
    # Make sure we're using the EMBEDDING deployment, not the chat deployment
    response = client.embeddings.create(
        input=[clean_text],
        model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT  # This should be 'test-embedding'
    )
    
    return response.data[0].embedding