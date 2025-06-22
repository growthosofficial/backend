import json
import logging
from typing import Any, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """Safely parse JSON string with fallback."""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse JSON: {json_string}")
        return default if default is not None else []


def safe_json_dumps(data: Any) -> str:
    """Safely convert data to JSON string."""
    try:
        return json.dumps(data)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to convert to JSON: {e}")
        return "[]"


def extract_preview(text: str, max_length: int = 150) -> str:
    """Extract a preview from text content."""
    if not text:
        return ""
    
    # Clean up the text
    clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
    
    # Truncate if too long
    if len(clean_text) <= max_length:
        return clean_text
    
    # Find last complete word within limit
    truncated = clean_text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can keep most of the text
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."


def generate_basic_tags(text: str, max_tags: int = 5) -> List[str]:
    """Generate basic tags from text content."""
    if not text:
        return []
    
    # Common stop words to exclude
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
        'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
        'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 
        'boy', 'did', 'she', 'use', 'your', 'said', 'each', 'make', 'most', 
        'over', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 
        'long', 'many', 'such', 'take', 'than', 'them', 'well', 'were', 'this',
        'that', 'with', 'have', 'will', 'from', 'they', 'know', 'want', 'been',
        'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just'
    }
    
    # Extract words, clean them, and filter
    words = []
    for word in text.lower().split():
        # Remove punctuation and check length
        clean_word = ''.join(char for char in word if char.isalnum())
        if (len(clean_word) >= 3 and 
            clean_word not in stop_words and 
            clean_word.isalpha()):
            words.append(clean_word)
    
    # Get unique words and limit
    unique_words = list(dict.fromkeys(words))  # Preserve order while removing duplicates
    return unique_words[:max_tags]


def validate_embedding(embedding: List[float]) -> bool:
    """Validate that embedding is a proper vector."""
    if not isinstance(embedding, list):
        return False
    
    if len(embedding) == 0:
        return False
    
    # Check if all elements are numbers
    for item in embedding:
        if not isinstance(item, (int, float)):
            return False
    
    return True


def format_datetime_for_api(dt: datetime) -> str:
    """Format datetime for API response."""
    return dt.isoformat()


def create_error_response(error_message: str, detail: str = None) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "error": error_message,
        "detail": detail,
        "timestamp": datetime.now().isoformat()
    }


def log_api_call(endpoint: str, method: str, status_code: int, duration: float = None):
    """Log API call information."""
    log_message = f"{method} {endpoint} - Status: {status_code}"
    if duration is not None:
        log_message += f" - Duration: {duration:.3f}s"
    
    if status_code >= 400:
        logger.warning(log_message)
    else:
        logger.info(log_message)


def clean_text_input(text: str) -> str:
    """Clean and normalize text input."""
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize
    cleaned = ' '.join(text.split())
    
    # Remove or replace problematic characters
    cleaned = cleaned.replace('\x00', '')  # Remove null bytes
    
    return cleaned.strip()