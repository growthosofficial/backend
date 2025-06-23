"""
Simplified FastAPI application for Second Brain Knowledge Management System
Only handles AI processing and returns recommendations - frontend handles database writes
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from config.settings import settings
    from core.similarity import SSC
    from core.llm_updater import LLMUpdater
    from database.supabase_manager import supabase_manager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    raise

# Create FastAPI app
app = FastAPI(
    title="Second Brain Knowledge Management API",
    description="AI-powered knowledge processing - returns recommendations only",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ProcessTextRequest(BaseModel):
    """Request model for processing text input"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to process")
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold (0.0-1.0)")

class RecommendationResponse(BaseModel):
    """Single recommendation response"""
    option_number: int = Field(..., description="Recommendation option number (1-3)")
    change: str = Field(..., description="Explanation of what changes will be made")
    updated_text: str = Field(..., description="The complete updated/new text content")
    category: str = Field(..., description="Category name for this recommendation")
    tags: list[str] = Field(default_factory=list, description="Content-based tags for this recommendation")
    preview: str = Field(..., description="Short preview of the content")

class ProcessTextResponse(BaseModel):
    """Response model for text processing"""
    recommendations: list[RecommendationResponse] = Field(..., description="List of 3 recommendations")
    similar_category: str | None = Field(None, description="Most similar existing category if found")
    similarity_score: float | None = Field(None, description="Similarity score if match found")
    status: str = Field("success", description="Request status")

class CategoryResponse(BaseModel):
    """Single knowledge category response"""
    id: int | None = Field(None, description="Database record ID")
    category: str = Field(..., description="Category name")
    content_preview: str = Field(..., description="First 100 characters of content")
    tags: list[str] = Field(default_factory=list, description="Category tags")
    last_updated: str | None = Field(None, description="Last update timestamp")

class CategoriesResponse(BaseModel):
    """Response model for listing categories"""
    categories: list[CategoryResponse] = Field(..., description="List of all categories")
    total_count: int = Field(..., description="Total number of categories")
    status: str = Field("success", description="Request status")

class DatabaseStatsResponse(BaseModel):
    """Response model for database statistics"""
    total_knowledge_items: int = Field(..., description="Total number of knowledge items")
    unique_tags: int = Field(..., description="Number of unique tags")
    categories: list[str] = Field(..., description="List of all category names")
    most_common_tags: list[tuple] = Field(..., description="Most frequently used tags")
    database_type: str = Field(..., description="Type of database being used")
    status: str = Field("success", description="Request status")

@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup"""
    try:
        settings.validate()
        print("✅ FastAPI server started successfully!")
        print("📚 Second Brain Knowledge Management System")
        print(f"🔗 API Documentation: http://localhost:8000/docs")
        print(f"🔗 Alternative Docs: http://localhost:8000/redoc")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        raise

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "message": "Second Brain Knowledge Management API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    try:
        # Test database connection
        stats = supabase_manager.get_database_stats()
        
        return {
            "status": "healthy",
            "database": "connected",
            "total_categories": stats['total_knowledge_items'],
            "azure_openai": "configured" if settings.AZURE_OPENAI_API_KEY else "not configured"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/api/process-full", response_model=ProcessTextResponse, tags=["Knowledge Processing"])
async def process_full(request: ProcessTextRequest):
    """
    Main endpoint: Process input text and return AI recommendations
    
    This endpoint:
    1. Finds similar existing knowledge using semantic search
    2. Generates 3 AI-powered recommendations using Azure OpenAI
    3. Returns structured recommendations for frontend to handle
    
    Frontend will then apply selected recommendations directly to Supabase.
    """
    try:
        # Step 1: Check for similar existing knowledge
        most_similar = SSC(request.text, request.threshold)
        
        # Step 2: Get AI recommendations from Azure OpenAI
        recommendations_raw = LLMUpdater(request.text, most_similar, "azure_openai")
        
        # Step 3: Format recommendations for frontend
        formatted_recommendations = []
        for i, rec in enumerate(recommendations_raw, 1):
            formatted_recommendations.append(
                RecommendationResponse(
                    option_number=i,
                    change=rec['change'],
                    updated_text=rec['updated_text'],
                    category=rec['category'],
                    tags=rec.get('tags', []),  # Include tags from LLM response
                    preview=rec['updated_text'][:100] + "..." if len(rec['updated_text']) > 100 else rec['updated_text']
                )
            )
        
        return ProcessTextResponse(
            recommendations=formatted_recommendations,
            similar_category=most_similar['category'] if most_similar else None,
            similarity_score=most_similar['similarity_score'] if most_similar else None
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process text: {str(e)}"
        )

@app.get("/api/categories", response_model=CategoriesResponse, tags=["Knowledge Management"])
async def get_categories():
    """
    Get all knowledge categories from the database
    
    Optional endpoint - frontend can also access Supabase directly.
    Returns a list of all knowledge categories with previews and metadata.
    """
    try:
        knowledge_items = supabase_manager.load_all_knowledge()
        
        categories = []
        for item in knowledge_items:
            categories.append(
                CategoryResponse(
                    id=item.get('id'),
                    category=item.get('category', 'Unknown'),
                    content_preview=item.get('content', '')[:100] + "..." if len(item.get('content', '')) > 100 else item.get('content', ''),
                    tags=item.get('tags', []),
                    last_updated=item.get('last_updated')
                )
            )
        
        return CategoriesResponse(
            categories=categories,
            total_count=len(categories)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving categories: {str(e)}"
        )

@app.get("/api/stats", response_model=DatabaseStatsResponse, tags=["Knowledge Management"])
async def get_database_stats():
    """
    Get database statistics and insights
    
    Optional endpoint - frontend can also calculate stats from Supabase directly.
    Returns statistics about your knowledge database.
    """
    try:
        stats = supabase_manager.get_database_stats()
        
        return DatabaseStatsResponse(
            total_knowledge_items=stats['total_knowledge_items'],
            unique_tags=stats['unique_tags'],
            categories=stats['categories'],
            most_common_tags=stats['most_common_tags'],
            database_type=stats['database_type']
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving database stats: {str(e)}"
        )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status": "error"
    }

if __name__ == "__main__":
    import uvicorn
    
    print(f"🌟 Starting Second Brain API on port 8000")
    print(f"📚 API docs available at: http://localhost:8000/docs")
    print(f"📖 Alternative docs at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )