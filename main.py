import os
import logging
import json
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from models import (
    ProcessTextRequest, ProcessTextResponse, RecommendationResponse,
    KnowledgeItemCreate, KnowledgeItemUpdate, KnowledgeItemResponse,
    HealthResponse, StatsResponse, ErrorResponse, CategoriesResponse, CategoryResponse
)
from database import DatabaseManager
from core.embeddings import EmbeddingService
from core.similarity import SemanticSimilarityCore
from core.llm_updater import LLMUpdater
from utils.helpers import (
    safe_json_loads, safe_json_dumps, extract_preview, 
    create_error_response, log_api_call, clean_text_input
)
from src.utils.category_mapping import get_main_category, get_all_main_categories

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables
db_manager: DatabaseManager = None
embedding_service: EmbeddingService = None
similarity_core: SemanticSimilarityCore = None
llm_updater: LLMUpdater = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global db_manager, embedding_service, similarity_core, llm_updater
    
    try:
        # Initialize database
        database_url = os.getenv("DATABASE_URL", "sqlite:///knowledge.db")
        db_manager = DatabaseManager(database_url)
        logger.info("Database initialized successfully")
        
        # Initialize OpenAI services
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        embedding_service = EmbeddingService(openai_api_key)
        similarity_core = SemanticSimilarityCore(db_manager)
        llm_updater = LLMUpdater(openai_api_key)
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Cleanup (if needed)
    logger.info("Application shutdown")


# Create FastAPI app
app = FastAPI(
    title="Second Brain Knowledge Management API",
    description="AI-powered knowledge management system with two-tier categorization and semantic similarity",
    version="2.0.0",
    lifespan=lifespan
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
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_manager() -> DatabaseManager:
    """Dependency to get database manager."""
    if db_manager is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db_manager


def get_embedding_service() -> EmbeddingService:
    """Dependency to get embedding service."""
    if embedding_service is None:
        raise HTTPException(status_code=500, detail="Embedding service not initialized")
    return embedding_service


def get_similarity_core() -> SemanticSimilarityCore:
    """Dependency to get similarity core."""
    if similarity_core is None:
        raise HTTPException(status_code=500, detail="Similarity core not initialized")
    return similarity_core


def get_llm_updater() -> LLMUpdater:
    """Dependency to get LLM updater."""
    if llm_updater is None:
        raise HTTPException(status_code=500, detail="LLM updater not initialized")
    return llm_updater


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


@app.get("/health", response_model=HealthResponse)
async def health_check(
    db: DatabaseManager = Depends(get_db_manager),
    embeddings: EmbeddingService = Depends(get_embedding_service)
):
    """Health check endpoint."""
    start_time = datetime.now()
    
    try:
        # Check database
        db_status = "healthy" if db.check_connection() else "unhealthy"
        
        # Check OpenAI API
        try:
            openai_status = "healthy" if await embeddings.check_api_connection() else "unhealthy"
        except Exception:
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


@app.post("/api/process-text", response_model=ProcessTextResponse)
async def process_text(
    request: ProcessTextRequest,
    db: DatabaseManager = Depends(get_db_manager),
    embeddings: EmbeddingService = Depends(get_embedding_service),
    similarity: SemanticSimilarityCore = Depends(get_similarity_core),
    llm: LLMUpdater = Depends(get_llm_updater)
):
    """Main text processing endpoint with two-tier categorization."""
    start_time = datetime.now()
    
    try:
        # Clean input text
        cleaned_text = clean_text_input(request.text)
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        logger.info(f"Processing text: {cleaned_text[:100]}...")
        
        # 1. Generate embedding for input text
        input_embedding = await embeddings.get_embedding(cleaned_text)
        
        # 2. Find most similar existing knowledge (SSC function)
        similar_item = similarity.SSC(input_embedding, request.threshold)
        
        # 3. Generate 3 AI recommendations (LLMUpdater function)
        recommendations_data = await llm.LLMUpdater(cleaned_text, similar_item)
        
        # 4. Format recommendations with two-tier categorization
        recommendations = []
        for rec_data in recommendations_data:
            recommendation = RecommendationResponse(
                option_number=rec_data.get("option_number", len(recommendations) + 1),
                change=rec_data["change"],
                updated_text=rec_data["updated_text"],
                main_category=rec_data["main_category"],
                sub_category=rec_data["sub_category"],
                tags=rec_data["tags"],
                preview=rec_data["preview"]
            )
            recommendations.append(recommendation)
        
        # 5. Create response
        response = ProcessTextResponse(
            recommendations=recommendations,
            similar_main_category=similar_item.get("main_category") if similar_item else None,
            similar_sub_category=similar_item.get("sub_category") if similar_item else None,
            similarity_score=similar_item.get("similarity_score") if similar_item else None,
            status="success"
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/process-text", "POST", 200, duration)
        
        logger.info(f"Successfully processed text with {len(recommendations)} recommendations")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text processing failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/process-text", "POST", 500, duration)
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")


@app.get("/api/categories", response_model=CategoriesResponse)
async def get_categories(db: DatabaseManager = Depends(get_db_manager)):
    """Get all knowledge categories grouped by main category."""
    start_time = datetime.now()
    
    try:
        items = db.get_all_knowledge_items()
        
        # Group by main category
        main_categories = {}
        total_items = 0
        
        for item in items:
            # Handle backward compatibility
            main_cat = getattr(item, 'main_category', None)
            sub_cat = getattr(item, 'sub_category', getattr(item, 'category', 'unknown'))
            
            if not main_cat:
                main_cat = get_main_category(sub_cat)
            
            if main_cat not in main_categories:
                main_categories[main_cat] = {
                    "main_category": main_cat,
                    "sub_categories": [],
                    "total_items": 0,
                    "last_updated": None
                }
            
            # Parse tags
            tags = safe_json_loads(getattr(item, 'tags', '[]'), [])
            
            sub_cat_data = {
                "id": item.id,
                "sub_category": sub_cat,
                "content": item.content,
                "tags": tags,
                "created_at": item.created_at.isoformat(),
                "last_updated": item.last_updated.isoformat()
            }
            
            main_categories[main_cat]["sub_categories"].append(sub_cat_data)
            main_categories[main_cat]["total_items"] += 1
            total_items += 1
            
            # Update last_updated
            if (main_categories[main_cat]["last_updated"] is None or 
                item.last_updated > datetime.fromisoformat(main_categories[main_cat]["last_updated"])):
                main_categories[main_cat]["last_updated"] = item.last_updated.isoformat()
        
        # Convert to response format
        categories = []
        for main_cat_data in main_categories.values():
            category_response = CategoryResponse(
                main_category=main_cat_data["main_category"],
                sub_categories=main_cat_data["sub_categories"],
                total_items=main_cat_data["total_items"],
                last_updated=datetime.fromisoformat(main_cat_data["last_updated"]) if main_cat_data["last_updated"] else None
            )
            categories.append(category_response)
        
        response = CategoriesResponse(
            categories=categories,
            total_main_categories=len(categories),
            total_sub_categories=sum(len(cat.sub_categories) for cat in categories),
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


@app.get("/api/main-categories")
async def get_main_categories():
    """Get list of all available main categories."""
    start_time = datetime.now()
    
    try:
        main_categories = get_all_main_categories()
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/main-categories", "GET", 200, duration)
        
        return {
            "main_categories": main_categories,
            "total_count": len(main_categories)
        }
        
    except Exception as e:
        logger.error(f"Failed to get main categories: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/main-categories", "GET", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to get main categories: {str(e)}")


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats(db: DatabaseManager = Depends(get_db_manager)):
    """Get database statistics with two-tier categorization."""
    start_time = datetime.now()
    
    try:
        items = db.get_all_knowledge_items()
        
        # Collect statistics
        main_categories = set()
        sub_categories = set()
        all_tags = []
        main_category_counts = {}
        latest_update = None
        
        for item in items:
            # Handle backward compatibility
            main_cat = getattr(item, 'main_category', None)
            sub_cat = getattr(item, 'sub_category', getattr(item, 'category', 'unknown'))
            
            if not main_cat:
                main_cat = get_main_category(sub_cat)
            
            main_categories.add(main_cat)
            sub_categories.add(sub_cat)
            
            # Count main categories
            main_category_counts[main_cat] = main_category_counts.get(main_cat, 0) + 1
            
            # Collect tags
            tags = safe_json_loads(getattr(item, 'tags', '[]'), [])
            all_tags.extend(tags)
            
            # Track latest update
            if latest_update is None or item.last_updated > latest_update:
                latest_update = item.last_updated
        
        # Count tag frequencies
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        most_common_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        response = StatsResponse(
            total_items=len(items),
            unique_main_categories=len(main_categories),
            unique_sub_categories=len(sub_categories),
            unique_tags=len(set(all_tags)),
            latest_update=latest_update,
            main_category_distribution=main_category_counts,
            most_common_tags=most_common_tags
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/stats", "GET", 200, duration)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/stats", "GET", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/api/knowledge", response_model=KnowledgeItemResponse)
async def create_knowledge_item(
    item: KnowledgeItemCreate,
    db: DatabaseManager = Depends(get_db_manager),
    embeddings: EmbeddingService = Depends(get_embedding_service)
):
    """Create a new knowledge item with two-tier categorization."""
    start_time = datetime.now()
    
    try:
        # Clean content
        cleaned_content = clean_text_input(item.content)
        if not cleaned_content:
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        # Ensure main_category is set
        main_category = item.main_category
        if not main_category:
            main_category = get_main_category(item.sub_category)
        
        # Generate embedding
        embedding = await embeddings.get_embedding(cleaned_content)
        
        # Create item in database
        db_item = db.create_knowledge_item(
            main_category=main_category,
            sub_category=item.sub_category,
            content=cleaned_content,
            tags=item.tags,
            embedding=embedding
        )
        
        # Format response
        response = KnowledgeItemResponse(
            id=db_item.id,
            main_category=main_category,
            sub_category=item.sub_category,
            content=db_item.content,
            tags=safe_json_loads(db_item.tags, []),
            created_at=db_item.created_at,
            last_updated=db_item.last_updated
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/knowledge", "POST", 201, duration)
        
        logger.info(f"Created knowledge item with ID: {db_item.id} (Main: {main_category}, Sub: {item.sub_category})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create knowledge item: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/knowledge", "POST", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to create knowledge item: {str(e)}")


@app.get("/api/knowledge")
async def get_all_knowledge(db: DatabaseManager = Depends(get_db_manager)):
    """Get all knowledge items with two-tier categorization."""
    start_time = datetime.now()
    
    try:
        items = db.get_all_knowledge_items()
        
        result = []
        for item in items:
            # Handle backward compatibility
            main_cat = getattr(item, 'main_category', None)
            sub_cat = getattr(item, 'sub_category', getattr(item, 'category', 'unknown'))
            
            if not main_cat:
                main_cat = get_main_category(sub_cat)
            
            item_data = {
                "id": item.id,
                "main_category": main_cat,
                "sub_category": sub_cat,
                "content": item.content,
                "tags": safe_json_loads(getattr(item, 'tags', '[]'), []),
                "created_at": item.created_at.isoformat(),
                "last_updated": item.last_updated.isoformat()
            }
            result.append(item_data)
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/knowledge", "GET", 200, duration)
        
        return {"knowledge_items": result, "total_items": len(result)}
        
    except Exception as e:
        logger.error(f"Failed to get knowledge items: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call("/api/knowledge", "GET", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge items: {str(e)}")


@app.put("/api/knowledge/{item_id}", response_model=KnowledgeItemResponse)
async def update_knowledge_item(
    item_id: int,
    item: KnowledgeItemUpdate,
    db: DatabaseManager = Depends(get_db_manager),
    embeddings: EmbeddingService = Depends(get_embedding_service)
):
    """Update existing knowledge item with two-tier categorization."""
    start_time = datetime.now()
    
    try:
        # Check if item exists
        existing_item = db.get_knowledge_item_by_id(item_id)
        if not existing_item:
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        
        # Prepare update data
        update_data = {}
        new_embedding = None
        
        if item.sub_category is not None:
            update_data["sub_category"] = item.sub_category
            # Auto-determine main_category if not explicitly provided
            if item.main_category is None:
                update_data["main_category"] = get_main_category(item.sub_category)
        
        if item.main_category is not None:
            update_data["main_category"] = item.main_category
        
        if item.content is not None:
            cleaned_content = clean_text_input(item.content)
            if not cleaned_content:
                raise HTTPException(status_code=400, detail="Content cannot be empty")
            update_data["content"] = cleaned_content
            
            # Generate new embedding for updated content
            new_embedding = await embeddings.get_embedding(cleaned_content)
        
        if item.tags is not None:
            update_data["tags"] = item.tags
        
        # Update item in database
        updated_item = db.update_knowledge_item(
            item_id=item_id,
            main_category=update_data.get("main_category"),
            sub_category=update_data.get("sub_category"),
            content=update_data.get("content"),
            tags=update_data.get("tags"),
            embedding=new_embedding
        )
        
        if not updated_item:
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        
        # Format response
        response = KnowledgeItemResponse(
            id=updated_item.id,
            main_category=getattr(updated_item, 'main_category', get_main_category(getattr(updated_item, 'sub_category', 'unknown'))),
            sub_category=getattr(updated_item, 'sub_category', getattr(updated_item, 'category', 'unknown')),
            content=updated_item.content,
            tags=safe_json_loads(updated_item.tags, []),
            created_at=updated_item.created_at,
            last_updated=updated_item.last_updated
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call(f"/api/knowledge/{item_id}", "PUT", 200, duration)
        
        logger.info(f"Updated knowledge item with ID: {item_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update knowledge item {item_id}: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call(f"/api/knowledge/{item_id}", "PUT", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to update knowledge item: {str(e)}")


@app.delete("/api/knowledge/{item_id}")
async def delete_knowledge_item(
    item_id: int,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Delete knowledge item."""
    start_time = datetime.now()
    
    try:
        success = db.delete_knowledge_item(item_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call(f"/api/knowledge/{item_id}", "DELETE", 200, duration)
        
        logger.info(f"Deleted knowledge item with ID: {item_id}")
        return {"message": "Knowledge item deleted successfully", "id": item_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete knowledge item {item_id}: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call(f"/api/knowledge/{item_id}", "DELETE", 500, duration)
        raise HTTPException(status_code=500, detail=f"Failed to delete knowledge item: {str(e)}")


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    logger.info(f"Starting FastAPI server on {host}:{port}")
    logger.info("Two-tier categorization system enabled")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )