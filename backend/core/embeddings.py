import logging
from typing import List
import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "text-embedding-ada-002"

    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for input text using OpenAI."""
        try:
            # Clean and prepare text
            text = text.replace("\n", " ").strip()
            
            if not text:
                raise ValueError("Empty text provided for embedding")
            
            # Generate embedding
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            logger.info(f"Generated embedding for text (length: {len(text)})")
            return embedding
            
        except openai.RateLimitError:
            logger.error("OpenAI rate limit exceeded")
            raise Exception("Rate limit exceeded. Please try again later.")
        except openai.AuthenticationError:
            logger.error("OpenAI authentication failed")
            raise Exception("OpenAI API authentication failed")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise Exception(f"Embedding generation failed: {str(e)}")

    async def check_api_connection(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            await self.get_embedding("test")
            return True
        except Exception as e:
            logger.error(f"OpenAI API connection check failed: {e}")
            return False