# Second Brain Knowledge Management Backend

A production-ready FastAPI backend for an AI-powered knowledge management system that processes text input, finds semantic similarities, generates intelligent recommendations, and provides comprehensive self-testing capabilities using Azure OpenAI.

## Features

- **Semantic Similarity Core (SSC)**: Finds similar knowledge using Azure OpenAI embeddings and cosine similarity
- **LLM Updater System**: Generates 3 contextual recommendations using GPT-4
- **Self-Test System**: AI-powered question generation and answer evaluation for knowledge assessment
- **Two-Tier Categorization**: Academic subject-based main categories with specific sub-categories
- **Database Layer**: Supabase integration for storing knowledge items, embeddings, and evaluations
- **RESTful API**: Comprehensive endpoints for knowledge management and assessment
- **Health Checks**: Database and Azure OpenAI API status monitoring
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

- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_DEPLOYMENT_NAME`: Your GPT-4 deployment name
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`: Your embedding model deployment name
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase service role key

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

- `POST /api/process-text` - Process text and get AI recommendations with two-tier categorization
- `GET /health` - Health check for database and Azure OpenAI API

### Knowledge Management (Read-Only)

- `GET /api/knowledge` - Get all knowledge items
- `GET /api/knowledge/category/{main_category}` - Get knowledge items by academic subject
- `GET /api/categories` - Get all categories grouped by academic subject
- `GET /api/academic-subjects` - Get list of all available academic subjects
- `GET /api/search?q={term}` - Search knowledge items by content

### Self-Test & Assessment

- `POST /api/self-test/generate` - Generate questions from knowledge base
- `POST /api/self-test/evaluate` - Evaluate answers and get detailed feedback

### Analytics & Statistics

- `GET /api/stats` - Get comprehensive database statistics
- `GET /api/analytics/strength-distribution` - Get knowledge strength score distribution
- `GET /api/analytics/category-strength` - Get strength analysis by academic category
- `GET /api/analytics/items-due` - Get items due for review (spaced repetition)

## Self-Test System

The self-test system provides AI-powered knowledge assessment capabilities:

### Question Generation

Generate thought-provoking questions from your knowledge base:

```bash
curl -X POST "http://localhost:8000/api/self-test/generate?num_questions=5" \
-H "Content-Type: application/json"
```

**Response:**

```json
{
  "questions": [
    {
      "question_text": "How does quantum entanglement challenge our classical understanding of locality and information transfer?",
      "category": "Physics",
      "knowledge_id": 123,
      "answer": ""
    }
  ],
  "categories_covered": ["Physics", "Mathematics", "Computer Science"],
  "total_questions": 5
}
```

### Answer Evaluation

Submit answers for AI-powered evaluation with detailed feedback:

```bash
curl -X POST "http://localhost:8000/api/self-test/evaluate" \
-H "Content-Type: application/json" \
-d '{
  "answers": [
    {
      "knowledge_id": 123,
      "question_text": "How does quantum entanglement work?",
      "answer": "Quantum entanglement occurs when particles become correlated..."
    }
  ]
}'
```

**Response:**

```json
{
  "evaluations": [
    {
      "question_text": "How does quantum entanglement work?",
      "answer": "Quantum entanglement occurs when particles become correlated...",
      "score": 4,
      "feedback": "Good understanding of the basic concept with clear explanation...",
      "correct_points": [
        "Correctly identified particle correlation",
        "Mentioned instantaneous state changes"
      ],
      "incorrect_points": ["Could elaborate more on measurement effects"],
      "evaluation_id": 456
    }
  ],
  "total_evaluated": 1,
  "average_score": 4.0
}
```

## Core Functions

### Semantic Similarity Core (SSC)

```python
def SSC(input_text: str, threshold: float = 0.8) -> Optional[Dict]:
    # Compares input with stored embeddings using cosine similarity
    # Returns most similar item above threshold with main_category and sub_category
```

### LLM Updater

```python
def LLMUpdater(input_text: str, existing_item: Optional[Dict]) -> List[Dict]:
    # Generates 3 different AI recommendations:
    # 1. Merge/Update approach
    # 2. Replace/Revise approach
    # 3. New Category approach
    # Each with proper academic subject categorization
```

### Self-Test Functions

```python
def generate_question(category: str, content: str) -> Optional[Dict]:
    # Generates thought-provoking questions using Azure OpenAI
    # Focuses on deep understanding and critical thinking

def evaluate_answer(question_text: str, answer: str, knowledge_content: str, category: str) -> Dict:
    # Evaluates answers on 1-5 scale with detailed feedback
    # Provides specific improvement suggestions
```

## Two-Tier Categorization System

The system uses academic subject-based categorization:

### Main Categories (Academic Subjects)

- **STEM**: Mathematics, Physics, Chemistry, Biology, Computer Science, Engineering, Medicine, etc.
- **Social Sciences**: Psychology, Sociology, Economics, Political Science, etc.
- **Humanities**: History, Philosophy, Literature, Languages, Religious Studies, etc.
- **Applied Fields**: Business Administration, Law, Education, Architecture, etc.
- **Arts & Creative**: Visual Arts, Music, Creative Writing, Film Studies, etc.

### Sub-Categories

Specific topics within each academic subject (e.g., "quantum mechanics applications", "Renaissance art techniques", "startup financial modeling")

## Project Structure

```
backend/
├── main.py                    # Main FastAPI app with read-only endpoints
├── models.py                  # Pydantic models for requests/responses
├── api/
│   └── self_test.py          # Self-test router with question generation and evaluation
├── src/
│   ├── config/
│   │   └── settings.py       # Azure OpenAI and Supabase configuration
│   ├── core/
│   │   ├── embeddings.py     # Azure OpenAI embedding generation
│   │   ├── similarity.py     # Semantic similarity search (SSC)
│   │   ├── llm_updater.py    # AI recommendation system
│   │   └── self_test.py      # Question generation and answer evaluation
│   ├── database/
│   │   └── supabase_manager.py # Supabase operations (read-only)
│   ├── utils/
│   │   ├── helpers.py        # Utility functions
│   │   └── category_mapping.py # Academic subject categorization
│   └── processing/
│       ├── pipeline.py       # Text processing pipeline
│       └── file_handler.py   # File input handling
├── requirements.txt          # Dependencies
├── .env.example             # Environment template
└── README.md               # This file
```

## Database Schema

### Knowledge Items Table

```sql
CREATE TABLE knowledge_items (
    id SERIAL PRIMARY KEY,
    main_category TEXT NOT NULL,        -- Academic subject (e.g., "Physics")
    sub_category TEXT NOT NULL,         -- Specific topic (e.g., "quantum mechanics")
    content TEXT NOT NULL,
    tags TEXT[] DEFAULT '{}',           -- Array of tags
    embedding VECTOR(1536),             -- OpenAI embedding vector
    source TEXT DEFAULT 'text',
    strength_score FLOAT,               -- For spaced repetition (0.0-1.0)
    last_reviewed TIMESTAMP,
    next_review_due TIMESTAMP,
    review_count INTEGER DEFAULT 0,
    ease_factor FLOAT DEFAULT 2.5,
    interval_days INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);
```

### Evaluations Table

```sql
CREATE TABLE evaluations (
    id SERIAL PRIMARY KEY,
    knowledge_id INTEGER REFERENCES knowledge_items(id),
    question_text TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    score INTEGER CHECK (score >= 1 AND score <= 5),
    feedback TEXT NOT NULL,
    correct_points TEXT[] DEFAULT '{}',
    incorrect_points TEXT[] DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Example Usage

### Process Text with Academic Categorization

```bash
curl -X POST "http://localhost:8000/api/process-text" \
-H "Content-Type: application/json" \
-d '{
    "text": "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning approaches",
    "threshold": 0.8
}'
```

### Generate Self-Test Questions

```bash
curl -X POST "http://localhost:8000/api/self-test/generate?num_questions=3" \
-H "Content-Type: application/json"
```

### Evaluate Learning Progress

```bash
curl -X POST "http://localhost:8000/api/self-test/evaluate" \
-H "Content-Type: application/json" \
-d '{
  "answers": [
    {
      "knowledge_id": 1,
      "question_text": "Explain the difference between supervised and unsupervised learning",
      "answer": "Supervised learning uses labeled training data to learn patterns, while unsupervised learning finds hidden patterns in unlabeled data without predefined outcomes."
    }
  ]
}'
```

### Get Academic Subject Statistics

```bash
curl "http://localhost:8000/api/analytics/category-strength"
```

## Configuration

### Environment Variables

- `AZURE_OPENAI_API_KEY`: Required for embeddings and GPT-4
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI service endpoint
- `AZURE_OPENAI_DEPLOYMENT_NAME`: GPT-4 deployment name
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`: Embedding model deployment name
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_KEY`: Supabase service role key
- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `CORS_ORIGINS`: Allowed origins for CORS
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### CORS Configuration

The API is configured to allow connections from:

- http://localhost:3000 (React dev server)
- http://localhost:5173 (Vite dev server)

## Self-Test Features

### Question Generation

- **AI-Powered**: Uses Azure OpenAI to generate thought-provoking questions
- **Content-Based**: Questions are generated from your actual knowledge base
- **Customizable**: Specify number of questions (1-20)
- **Category Coverage**: Ensures questions span different academic subjects
- **Deep Learning Focus**: Questions test understanding, not just memorization

### Answer Evaluation

- **Intelligent Scoring**: 1-5 scale based on understanding depth
- **Detailed Feedback**: Specific strengths and weaknesses identified
- **Improvement Suggestions**: Actionable advice for better understanding
- **Batch Processing**: Evaluate multiple answers efficiently
- **Progress Tracking**: Evaluations stored for learning analytics

### Learning Analytics

- **Performance Tracking**: Monitor learning progress over time
- **Category Analysis**: Identify strong and weak subject areas
- **Strength Distribution**: Visualize knowledge mastery levels
- **Review Scheduling**: Spaced repetition system for optimal retention

## Error Handling

The API includes comprehensive error handling:

- Input validation with detailed error messages
- Azure OpenAI API error handling (rate limits, authentication)
- Database connection and operation errors
- Self-test specific errors (invalid questions, evaluation failures)
- Global exception handler for unexpected errors

## Logging

All API requests and errors are logged with:

- Request method and endpoint
- Response status code
- Request duration
- Error details and stack traces
- Self-test operation tracking

## Performance Features

- Efficient vector similarity calculations using scikit-learn
- Database connection pooling with Supabase
- Async/await for non-blocking operations
- Batch processing for multiple evaluations
- Proper resource cleanup and error recovery
- Optimized question generation and evaluation

## Production Deployment

For production deployment:

1. Set appropriate environment variables
2. Use a production WSGI server like Gunicorn
3. Configure proper logging and monitoring
4. Set up database backups
5. Implement rate limiting for AI endpoints
6. Monitor Azure OpenAI usage and costs

## Dependencies

- **FastAPI**: Modern web framework with automatic API documentation
- **Supabase**: PostgreSQL database with real-time capabilities
- **Azure OpenAI**: GPT-4 for recommendations and question generation/evaluation
- **NumPy + scikit-learn**: Vector similarity calculations
- **Uvicorn**: ASGI server for development and production

## License

This project is part of the Second Brain Knowledge Management System.

## Self-Test API Reference

### POST /api/self-test/generate

Generate questions from knowledge base.

**Parameters:**

- `num_questions` (query, optional): Number of questions (1-20, default: 5)

**Response:** List of questions with metadata

### POST /api/self-test/evaluate

Evaluate submitted answers with AI feedback.

**Body:** Array of answers with question context

**Response:** Detailed evaluations with scores and improvement suggestions

Both endpoints integrate seamlessly with the existing knowledge management system and provide powerful learning assessment capabilities.
