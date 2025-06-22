# Second Brain Knowledge Management Backend

A production-ready FastAPI backend for an AI-powered knowledge management system that processes text input, finds semantic similarities, and generates intelligent recommendations using OpenAI.

## Features

- **Semantic Similarity Core (SSC)**: Finds similar knowledge using OpenAI embeddings and cosine similarity
- **LLM Updater System**: Generates 3 contextual recommendations using GPT-4
- **Database Layer**: SQLite with SQLModel for storing knowledge items and embeddings
- **RESTful API**: Comprehensive CRUD operations for knowledge management
- **Health Checks**: Database and OpenAI API status monitoring
- **CORS Support**: Configured for frontend integration
- **Comprehensive Logging**: Request tracking and error handling

## Quick Start

### 1. Environment Setup

```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `DATABASE_URL`: SQLite database path (default: sqlite:///knowledge.db)

### 3. Run the Server

```bash
python main.py
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## API Endpoints

### Core Processing
- `POST /api/process-text` - Process text and get AI recommendations
- `GET /health` - Health check for database and OpenAI API

### Knowledge Management
- `POST /api/knowledge` - Create new knowledge item
- `GET /api/knowledge` - Get all knowledge items
- `PUT /api/knowledge/{id}` - Update knowledge item
- `DELETE /api/knowledge/{id}` - Delete knowledge item

### Data & Analytics
- `GET /api/categories` - Get all categories with metadata
- `GET /api/stats` - Get database statistics

## Core Functions

### Semantic Similarity Core (SSC)
```python
def SSC(input_embedding: List[float], threshold: float = 0.8) -> Optional[Dict]:
    # Compares input with stored embeddings using cosine similarity
    # Returns most similar item above threshold
```

### LLM Updater
```python
async def LLMUpdater(input_text: str, existing_item: Optional[Dict]) -> List[Dict]:
    # Generates 3 different AI recommendations:
    # 1. Merge/Update approach
    # 2. Replace/Revise approach  
    # 3. New Category approach
```

## Project Structure

```
backend/
├── main.py              # FastAPI app and endpoints
├── models.py            # Pydantic models
├── database.py          # Database operations
├── core/
│   ├── embeddings.py    # OpenAI embedding generation
│   ├── similarity.py    # Semantic similarity search
│   └── llm_updater.py   # AI recommendation system
├── utils/
│   └── helpers.py       # Utility functions
├── requirements.txt     # Dependencies
├── .env.example        # Environment template
└── README.md           # This file
```

## Database Schema

```sql
CREATE TABLE knowledge_items (
    id INTEGER PRIMARY KEY,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT DEFAULT '[]',        -- JSON array
    embedding TEXT DEFAULT '[]',   -- JSON array (vector)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Example Usage

### Process Text
```bash
curl -X POST "http://localhost:8000/api/process-text" \
-H "Content-Type: application/json" \
-d '{
    "text": "Machine learning is a subset of artificial intelligence",
    "threshold": 0.8
}'
```

### Create Knowledge Item
```bash
curl -X POST "http://localhost:8000/api/knowledge" \
-H "Content-Type: application/json" \
-d '{
    "category": "AI/ML",
    "content": "Deep learning uses neural networks with multiple layers",
    "tags": ["deep-learning", "neural-networks", "ai"]
}'
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for embeddings and GPT-4
- `DATABASE_URL`: Database connection string
- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `CORS_ORIGINS`: Allowed origins for CORS
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### CORS Configuration
The API is configured to allow connections from:
- http://localhost:3000 (React dev server)
- http://localhost:5173 (Vite dev server)

## Error Handling

The API includes comprehensive error handling:
- Input validation with detailed error messages
- OpenAI API error handling (rate limits, authentication)
- Database connection and operation errors
- Global exception handler for unexpected errors

## Logging

All API requests and errors are logged with:
- Request method and endpoint
- Response status code
- Request duration
- Error details and stack traces

## Performance Features

- Efficient vector similarity calculations using scikit-learn
- Database connection pooling
- Async/await for non-blocking operations
- Proper resource cleanup and error recovery

## Production Deployment

For production deployment:

1. Set appropriate environment variables
2. Use a production WSGI server like Gunicorn
3. Configure proper logging and monitoring
4. Set up database backups
5. Implement rate limiting if needed

## Dependencies

- FastAPI: Modern web framework
- SQLModel: Database ORM with type safety
- OpenAI: AI embeddings and text generation
- NumPy + scikit-learn: Vector similarity calculations
- Uvicorn: ASGI server

## License

This project is part of the Second Brain Knowledge Management System.