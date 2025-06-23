import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings from environment variables"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    # Supabase Configuration
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
    
    # Defaults
    DEFAULT_SIMILARITY_THRESHOLD: float = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.8"))
    DEFAULT_LLM_TYPE: str = os.getenv("DEFAULT_LLM_TYPE", "azure_openai")
    
    def validate(self):
        """Validate required settings"""
        if not self.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY must be set")
        if not self.AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be set")
        if not self.SUPABASE_URL:
            raise ValueError("SUPABASE_URL must be set")
        if not self.SUPABASE_KEY:
            raise ValueError("SUPABASE_KEY must be set")

settings = Settings()