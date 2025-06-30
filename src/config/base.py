"""
Base configuration settings for the application
"""
import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from functools import lru_cache


class BaseConfig(BaseSettings):
    """Base configuration class with common settings"""
    
    # Application Settings
    APP_NAME: str = "Second Brain Knowledge Management API"
    APP_VERSION: str = "2.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    
    # Security Settings
    SECRET_KEY: str
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    BCRYPT_ROUNDS: int = 12
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-ada-002"
    
    # Supabase Configuration
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None
    
    # Database Configuration (if using direct PostgreSQL)
    DATABASE_URL: Optional[str] = None
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "knowledge_db"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    
    # Redis Configuration
    REDIS_URL: Optional[str] = None
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # CORS Settings
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 10485760  # 10MB
    UPLOAD_DIR: str = "uploads"
    ALLOWED_EXTENSIONS: str = "txt,pdf,docx,md"
    
    # External API Settings
    OPENAI_TIMEOUT: int = 30
    HTTP_TIMEOUT: int = 10
    MAX_RETRIES: int = 3
    
    # Monitoring & Logging
    SENTRY_DSN: Optional[str] = None
    ENABLE_METRICS: bool = True
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "logs/app.log"
    
    # Background Tasks
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    
    # Development Settings
    RELOAD: bool = False
    WORKERS: int = 1
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v: str) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("ALLOWED_EXTENSIONS", pre=True)
    def parse_allowed_extensions(cls, v: str) -> List[str]:
        """Parse allowed extensions from comma-separated string"""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v
    
    @validator("JWT_SECRET_KEY", pre=True, always=True)
    def set_jwt_secret_key(cls, v: Optional[str], values: dict) -> str:
        """Set JWT secret key to SECRET_KEY if not provided"""
        if v is None:
            return values.get("SECRET_KEY", "")
        return v
    
    def validate_required_settings(self) -> None:
        """Validate that all required settings are present"""
        required_fields = [
            "SECRET_KEY",
            "AZURE_OPENAI_API_KEY", 
            "AZURE_OPENAI_ENDPOINT",
            "SUPABASE_URL",
            "SUPABASE_KEY"
        ]
        
        missing_fields = []
        for field in required_fields:
            if not getattr(self, field):
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


class DevelopmentConfig(BaseConfig):
    """Development environment configuration"""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    ENVIRONMENT: str = "development"
    RELOAD: bool = True
    WORKERS: int = 1


class ProductionConfig(BaseConfig):
    """Production environment configuration"""
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    ENVIRONMENT: str = "production"
    RELOAD: bool = False
    WORKERS: int = 4
    RATE_LIMIT_REQUESTS: int = 50  # Stricter rate limiting
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15  # Shorter token expiry
    BCRYPT_ROUNDS: int = 14  # More secure hashing


class TestingConfig(BaseConfig):
    """Testing environment configuration"""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    ENVIRONMENT: str = "testing"
    DB_NAME: str = "test_knowledge_db"
    REDIS_DB: int = 15  # Use different Redis DB for tests


@lru_cache()
def get_config() -> BaseConfig:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig
    }
    
    config_class = config_map.get(env, DevelopmentConfig)
    config = config_class()
    config.validate_required_settings()
    
    return config


# Global config instance
config = get_config()