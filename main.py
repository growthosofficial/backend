"""
Read-Only FastAPI application for Second Brain Knowledge Management System
Focuses on data retrieval and AI-powered recommendation generation
Frontend handles all database updates
"""
import os
import sys
import logging
import json
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Add src directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from models import (
        ProcessTextRequest, ProcessTextResponse, RecommendationResponse,
        KnowledgeItemResponse, CategoryResponse, CategoriesResponse,
        HealthResponse, StatsResponse, StrengthDistributionResponse, 
        CategoryStrengthResponse, ItemsDueResponse, SearchResponse, ErrorResponse
    )
    from config.settings import settings
    from core.similarity import SSC
    from core.llm_updater import LLMUpdater
    from database.supabase_manager import supabase_manager
    from utils.category_mapping import get_all_subject_categories
    from utils.helpers import log_api_call, create_error_response
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory and all dependencies are installed")
    raise

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    try:
        # Validate settings
        settings.validate()
        logger.info("Configuration validated successfully")
        
        # Test Supabase connection
        stats = supabase_manager.get_database_stats()
        logger.info(f"Supabase connected - {stats['total_knowledge_items']} items available for analysis")
        
        logger.info("✅ Read-only knowledge analysis system initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Application shutdown")


# Create FastAPI app
app = FastAPI(
    title="Second Brain Knowledge Analysis API",
    description="Read-only AI-powered knowledge analysis and recommendation system. Frontend handles database updates.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
cors_origins = os.getenv("CORS_ORIGINS", '["http://localhost:3000", "http://localhost:5173"]')
try:
    origins = json.loads(cors_origins)
except json.JSONDecodeError:
    origins = ["http://localhost:3000", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only GET and POST for read-only + recommendations
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            "Internal server error",
            "An unexpected error occurred. Please try again later."
        )
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "Second Brain Knowledge Analysis API",
        "status": "running",
        "version": "2.0.0",
        "mode": "Read-Only Analysis & Recommendations",
        "docs": "/docs",
        "database": "Supabase (Read-Only)"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    start_time = datetime.now()
    
    try:
        # Check Supabase connection
        try:
            stats = supabase_manager.get_database_stats()
            db_status = "healthy"
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            db_status = "unhealthy"
        
        # Check Azure OpenAI API
        try:
            from core.embeddings import get_embedding
            test_embedding = get_embedding("test")
            openai_status = "healthy" if test_embedding else "unhealthy"
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            openai_status = "unhealthy"
        
        overall_status = "healthy" if db_status == "healthy" and openai_status == "healthy" else "unhealthy"
        
        response = HealthResponse(
            status=overall_status,
            database=db_status,
            openai_api=openai_status,
            timestamp=datetime.now()
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/health", "GET", 200, duration)
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/health", "GET", 500, duration)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# MAIN FEATURE: AI-POWERED RECOMMENDATION GENERATION

@app.post("/api/process-text", response_model=ProcessTextResponse, tags=["AI Recommendations"])
async def process_text(request: ProcessTextRequest):
    """
    **MAIN ENDPOINT**: Process input text and generate AI recommendations.
    
    This endpoint:
    1. Analyzes input text using semantic similarity (SSC)
    2. Generates 3 AI-powered recommendations using Azure OpenAI (LLMUpdater)
    3. Returns structured recommendations for frontend to handle
    
    Frontend will apply selected recommendations to database.
    """
    start_time = datetime.now()
    
    try:
        cleaned_text = request.text.strip()
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        logger.info(f"Processing text: {cleaned_text[:100]}...")
        
        # 1. Find most similar existing knowledge using SSC
        similar_item = SSC(cleaned_text, request.threshold)
        
        # 2. Generate 3 AI recommendations using LLMUpdater
        recommendations_data = LLMUpdater(cleaned_text, similar_item, "azure_openai")
        
        # 3. Format recommendations for frontend
        recommendations = []
        for rec_data in recommendations_data:
            recommendation = RecommendationResponse(
                option_number=rec_data.get("option_number", len(recommendations) + 1),
                change=rec_data["change"],
                updated_text=rec_data["updated_text"],
                main_category=rec_data["main_category"],
                sub_category=rec_data["sub_category"],
                tags=rec_data.get("tags", []),
                preview=rec_data.get("preview", rec_data["updated_text"][:100] + "..." if len(rec_data["updated_text"]) > 100 else rec_data["updated_text"])
            )
            recommendations.append(recommendation)
        
        # 4. Create response
        response = ProcessTextResponse(
            recommendations=recommendations,
            similar_main_category=similar_item.get("main_category") if similar_item else None,
            similar_sub_category=similar_item.get("sub_category") if similar_item else None,
            similarity_score=similar_item.get("similarity_score") if similar_item else None,
            status="success"
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/process-text", "POST", 200, duration)
        
        logger.info(f"✅ Generated {len(recommendations)} recommendations for frontend")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text processing failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/process-text", "POST", 500, duration)
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")


# READ-ONLY DATA ENDPOINTS FOR FRONTEND

@app.get("/api/categories", response_model=CategoriesResponse, tags=["Data Retrieval"])
async def get_categories():
    """Get all knowledge categories grouped by academic subject (read-only)."""
    start_time = datetime.now()
    
    try:
        grouped_categories = supabase_manager.get_categories_grouped()
        
        categories = []
        total_sub_categories = 0
        total_items = 0
        
        for main_cat_data in grouped_categories.values():
            category_response = CategoryResponse(
                main_category=main_cat_data["main_category"],
                sub_categories=main_cat_data["sub_categories"],
                total_items=main_cat_data["total_items"],
                last_updated=datetime.fromisoformat(main_cat_data["last_updated"]) if main_cat_data["last_updated"] else None
            )
            categories.append(category_response)
            total_sub_categories += len(main_cat_data["sub_categories"])
            total_items += main_cat_data["total_items"]
        
        response = CategoriesResponse(
            categories=categories,
            total_main_categories=len(categories),
            total_sub_categories=total_sub_categories,
            total_items=total_items
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/categories", "GET", 200, duration)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get categories: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/categories", "GET", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")


@app.get("/api/academic-subjects", tags=["Data Retrieval"])
async def get_academic_subjects():
    """Get list of all available academic subjects."""
    start_time = datetime.now()
    
    try:
        subjects = get_all_subject_categories()
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/academic-subjects", "GET", 200, duration)
        
        return {
            "academic_subjects": subjects,
            "total_count": len(subjects),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get academic subjects: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/academic-subjects", "GET", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to get academic subjects: {str(e)}")


@app.get("/api/stats", response_model=StatsResponse, tags=["Analytics"])
async def get_stats():
    """Get comprehensive database statistics (read-only)."""
    start_time = datetime.now()
    
    try:
        stats = supabase_manager.get_database_stats()
        
        response = StatsResponse(
            total_items=stats['total_knowledge_items'],
            unique_main_categories=stats['unique_main_categories'],
            unique_sub_categories=stats['unique_sub_categories'],
            unique_tags=stats['unique_tags'],
            unique_sources=stats['unique_sources'],
            main_category_distribution=stats['main_category_distribution'],
            source_distribution=stats['source_distribution'],
            most_common_tags=stats['most_common_tags'],
            avg_strength_score=stats['avg_strength_score'],
            items_with_strength_score=stats['items_with_strength_score'],
            strong_items=stats['strong_items'],
            weak_items=stats['weak_items'],
            total_reviews_completed=stats['total_reviews_completed'],
            database_type=stats['database_type']
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/stats", "GET", 200, duration)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/stats", "GET", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/api/knowledge", tags=["Data Retrieval"])
async def get_all_knowledge():
    """Get all knowledge items (read-only)."""
    start_time = datetime.now()
    
    try:
        items = supabase_manager.load_all_knowledge()
        
        result = []
        for item in items:
            item_data = KnowledgeItemResponse(
                id=item.get('id'),
                main_category=item.get('main_category', 'General Studies'),
                sub_category=item.get('sub_category', 'unknown'),
                content=item.get('content', ''),
                tags=item.get('tags', []),
                source=item.get('source', 'text'),
                strength_score=item.get('strength_score'),
                last_reviewed=datetime.fromisoformat(item['last_reviewed']) if item.get('last_reviewed') else None,
                next_review_due=datetime.fromisoformat(item['next_review_due']) if item.get('next_review_due') else None,
                review_count=item.get('review_count', 0),
                ease_factor=item.get('ease_factor', 2.5),
                interval_days=item.get('interval_days', 1),
                created_at=datetime.fromisoformat(item['created_at']),
                last_updated=datetime.fromisoformat(item['last_updated'])
            )
            result.append(item_data)
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/knowledge", "GET", 200, duration)
        
        return {
            "knowledge_items": result, 
            "total_items": len(result),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get knowledge items: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/knowledge", "GET", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge items: {str(e)}")


@app.get("/api/knowledge/category/{main_category}", tags=["Data Retrieval"])
async def get_knowledge_by_category(main_category: str):
    """Get all knowledge items for a specific academic subject (read-only)."""
    start_time = datetime.now()
    
    try:
        items = supabase_manager.get_knowledge_by_main_category(main_category)
        
        result = []
        for item in items:
            item_data = KnowledgeItemResponse(
                id=item.get('id'),
                main_category=item.get('main_category'),
                sub_category=item.get('sub_category'),
                content=item.get('content', ''),
                tags=item.get('tags', []),
                source=item.get('source', 'text'),
                strength_score=item.get('strength_score'),
                last_reviewed=datetime.fromisoformat(item['last_reviewed']) if item.get('last_reviewed') else None,
                next_review_due=datetime.fromisoformat(item['next_review_due']) if item.get('next_review_due') else None,
                review_count=item.get('review_count', 0),
                ease_factor=item.get('ease_factor', 2.5),
                interval_days=item.get('interval_days', 1),
                created_at=datetime.fromisoformat(item['created_at']),
                last_updated=datetime.fromisoformat(item['last_updated'])
            )
            result.append(item_data)
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call(f"/api/knowledge/category/{main_category}", "GET", 200, duration)
        
        return {
            "knowledge_items": result,
            "main_category": main_category,
            "total_items": len(result),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get knowledge for category {main_category}: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call(f"/api/knowledge/category/{main_category}", "GET", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge for category: {str(e)}")


@app.get("/api/search", response_model=SearchResponse, tags=["Data Retrieval"])
async def search_knowledge(
    q: str = Query(..., description="Search term"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results")
):
    """Search knowledge items by content (read-only)."""
    start_time = datetime.now()
    
    try:
        items = supabase_manager.search_knowledge_by_content(q, limit)
        
        result = []
        for item in items:
            item_data = KnowledgeItemResponse(
                id=item.get('id'),
                main_category=item.get('main_category'),
                sub_category=item.get('sub_category'),
                content=item.get('content', ''),
                tags=item.get('tags', []),
                source=item.get('source', 'text'),
                strength_score=item.get('strength_score'),
                last_reviewed=datetime.fromisoformat(item['last_reviewed']) if item.get('last_reviewed') else None,
                next_review_due=datetime.fromisoformat(item['next_review_due']) if item.get('next_review_due') else None,
                review_count=item.get('review_count', 0),
                ease_factor=item.get('ease_factor', 2.5),
                interval_days=item.get('interval_days', 1),
                created_at=datetime.fromisoformat(item['created_at']),
                last_updated=datetime.fromisoformat(item['last_updated'])
            )
            result.append(item_data)
        
        response = SearchResponse(
            results=result,
            total_results=len(result),
            search_term=q,
            status="success"
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/search", "GET", 200, duration)
        
        return response
        
    except Exception as e:
        logger.error(f"Search failed for term '{q}': {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/search", "GET", 500, duration)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ANALYTICS ENDPOINTS

@app.get("/api/analytics/strength-distribution", response_model=StrengthDistributionResponse, tags=["Analytics"])
async def get_strength_distribution():
    """Get knowledge strength score distribution for analytics."""
    start_time = datetime.now()
    
    try:
        distribution = supabase_manager.get_knowledge_strength_distribution()
        
        response = StrengthDistributionResponse(
            very_weak=distribution['very_weak'],
            weak=distribution['weak'],
            medium=distribution['medium'],
            strong=distribution['strong'],
            very_strong=distribution['very_strong'],
            no_score=distribution['no_score']
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/analytics/strength-distribution", "GET", 200, duration)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get strength distribution: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/analytics/strength-distribution", "GET", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to get strength distribution: {str(e)}")


@app.get("/api/analytics/category-strength", tags=["Analytics"])
async def get_category_strength_analysis():
    """Get strength analysis by academic category."""
    start_time = datetime.now()
    
    try:
        analysis = supabase_manager.get_category_strength_analysis()
        
        result = []
        for category, data in analysis.items():
            result.append(CategoryStrengthResponse(
                category=category,
                avg_strength=data['avg_strength'],
                total_items=data['total_items'],
                strong_items=data['strong_items'],
                weak_items=data['weak_items']
            ))
        
        # Sort by average strength descending
        result.sort(key=lambda x: x.avg_strength, reverse=True)
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/analytics/category-strength", "GET", 200, duration)
        
        return {
            "categories": result,
            "total_categories": len(result),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get category strength analysis: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/analytics/category-strength", "GET", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to get category strength analysis: {str(e)}")


@app.get("/api/analytics/items-due", response_model=ItemsDueResponse, tags=["Analytics"])
async def get_items_due_for_review(limit: int = Query(50, ge=1, le=200, description="Maximum number of items")):
    """Get knowledge items due for review (analytics only - no updates)."""
    start_time = datetime.now()
    
    try:
        items_due = supabase_manager.get_items_due_for_review(limit)
        
        formatted_items = []
        for item in items_due:
            formatted_item = KnowledgeItemResponse(
                id=item['id'],
                main_category=item['main_category'],
                sub_category=item['sub_category'],
                content=item['content'],
                tags=item.get('tags', []),
                source=item.get('source', 'text'),
                strength_score=item.get('strength_score'),
                last_reviewed=datetime.fromisoformat(item['last_reviewed']) if item.get('last_reviewed') else None,
                next_review_due=datetime.fromisoformat(item['next_review_due']) if item.get('next_review_due') else None,
                review_count=item.get('review_count', 0),
                ease_factor=item.get('ease_factor', 2.5),
                interval_days=item.get('interval_days', 1),
                created_at=datetime.fromisoformat(item['created_at']),
                last_updated=datetime.fromisoformat(item['last_updated'])
            )
            formatted_items.append(formatted_item)
        
        response = ItemsDueResponse(
            items_due=formatted_items,
            total_due=len(formatted_items)
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/analytics/items-due", "GET", 200, duration)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get items due for review: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/analytics/items-due", "GET", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to get items due for review: {str(e)}")


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    logger.info(f"🌟 Starting Second Brain Analysis API on {host}:{port}")
    logger.info("📚 Read-Only Mode: Database updates handled by frontend")
    logger.info("🤖 AI Recommendations: Powered by Azure OpenAI")
    logger.info("📊 Analytics: Comprehensive knowledge insights")
    logger.info(f"📖 API docs: http://{host}:{port}/docs")
    logger.info(f"📋 Alternative docs: http://{host}:{port}/redoc")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )