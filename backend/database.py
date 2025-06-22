import json
import logging
from datetime import datetime
from typing import List, Optional
from sqlmodel import SQLModel, create_engine, Session, select
from models import KnowledgeItem

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, echo=False)
        self.create_tables()

    def create_tables(self):
        """Create database tables if they don't exist."""
        try:
            SQLModel.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

    def get_session(self) -> Session:
        """Get database session."""
        return Session(self.engine)

    def create_knowledge_item(self, category: str, content: str, tags: List[str], embedding: List[float]) -> KnowledgeItem:
        """Create a new knowledge item."""
        with self.get_session() as session:
            try:
                item = KnowledgeItem(
                    category=category,
                    content=content,
                    tags=json.dumps(tags),
                    embedding=json.dumps(embedding),
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
                session.add(item)
                session.commit()
                session.refresh(item)
                logger.info(f"Created knowledge item with ID: {item.id}")
                return item
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to create knowledge item: {e}")
                raise

    def get_all_knowledge_items(self) -> List[KnowledgeItem]:
        """Get all knowledge items."""
        with self.get_session() as session:
            try:
                statement = select(KnowledgeItem)
                items = session.exec(statement).all()
                return list(items)
            except Exception as e:
                logger.error(f"Failed to get knowledge items: {e}")
                raise

    def get_knowledge_item_by_id(self, item_id: int) -> Optional[KnowledgeItem]:
        """Get knowledge item by ID."""
        with self.get_session() as session:
            try:
                statement = select(KnowledgeItem).where(KnowledgeItem.id == item_id)
                item = session.exec(statement).first()
                return item
            except Exception as e:
                logger.error(f"Failed to get knowledge item {item_id}: {e}")
                raise

    def update_knowledge_item(self, item_id: int, category: Optional[str] = None, 
                            content: Optional[str] = None, tags: Optional[List[str]] = None,
                            embedding: Optional[List[float]] = None) -> Optional[KnowledgeItem]:
        """Update knowledge item."""
        with self.get_session() as session:
            try:
                item = session.get(KnowledgeItem, item_id)
                if not item:
                    return None
                
                if category is not None:
                    item.category = category
                if content is not None:
                    item.content = content
                if tags is not None:
                    item.tags = json.dumps(tags)
                if embedding is not None:
                    item.embedding = json.dumps(embedding)
                
                item.last_updated = datetime.now()
                session.add(item)
                session.commit()
                session.refresh(item)
                logger.info(f"Updated knowledge item with ID: {item_id}")
                return item
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to update knowledge item {item_id}: {e}")
                raise

    def delete_knowledge_item(self, item_id: int) -> bool:
        """Delete knowledge item."""
        with self.get_session() as session:
            try:
                item = session.get(KnowledgeItem, item_id)
                if not item:
                    return False
                
                session.delete(item)
                session.commit()
                logger.info(f"Deleted knowledge item with ID: {item_id}")
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to delete knowledge item {item_id}: {e}")
                raise

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self.get_session() as session:
            try:
                items = self.get_all_knowledge_items()
                
                categories = set()
                all_tags = set()
                latest_update = None
                
                for item in items:
                    categories.add(item.category)
                    try:
                        tags = json.loads(item.tags)
                        all_tags.update(tags)
                    except json.JSONDecodeError:
                        pass
                    
                    if latest_update is None or item.last_updated > latest_update:
                        latest_update = item.last_updated
                
                return {
                    "total_items": len(items),
                    "unique_categories": len(categories),
                    "unique_tags": len(all_tags),
                    "latest_update": latest_update
                }
            except Exception as e:
                logger.error(f"Failed to get database stats: {e}")
                raise

    def check_connection(self) -> bool:
        """Check database connection."""
        try:
            with self.get_session() as session:
                session.exec(select(KnowledgeItem).limit(1))
                return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False