"""Configuration management"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    
    # App
    APP_NAME: str = "AI Portfolio Rebalancing Agent"
    VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/portfolio_agent"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # API Keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None
    POLYGON_API_KEY: Optional[str] = None
    
    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000", "http://localhost:8501"]
    
    # Market Data Settings
    MARKET_DATA_UPDATE_FREQUENCY: str = "daily"  # daily or weekly
    DEFAULT_MARKET_SYMBOLS: list[str] = ["SPY", "QQQ", "IEF", "GLD", "BTC-USD"]
    
    # ML Model Settings
    MODEL_PATH: str = "./ml-training/models"
    USE_PRETRAINED_MODELS: bool = True
    
    # Rebalancing Settings
    DEFAULT_REBALANCING_THRESHOLD: float = 0.05  # 5% drift
    TRANSACTION_COST_PERCENTAGE: float = 0.001  # 0.1%
    
    # Polygon.io Settings
    POLYGON_BASE_URL: str = "https://api.polygon.io"
    POLYGON_RATE_LIMIT: int = 5  # calls per minute
    
    # Caching Settings
    CACHE_EXPIRY_HOURS: int = 1
    CACHE_MIN_DATA_POINTS: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

