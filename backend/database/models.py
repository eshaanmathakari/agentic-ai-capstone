"""
Database models for the AI Portfolio Rebalancing Agent
"""
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, 
    Text, JSON, Enum as SQLEnum, Boolean, Index
)
from sqlalchemy.orm import relationship, DeclarativeBase
import enum


class Base(DeclarativeBase):
    """Base class for all models"""
    pass


class RiskLevel(str, enum.Enum):
    """Risk level enumeration"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class RebalancingStatus(str, enum.Enum):
    """Status of rebalancing suggestions"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    risk_profile = relationship("RiskProfile", back_populates="user", uselist=False)
    portfolios = relationship("Portfolio", back_populates="user")


class RiskProfile(Base):
    """Risk profile model"""
    __tablename__ = "risk_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    risk_level = Column(SQLEnum(RiskLevel), nullable=False)
    risk_score = Column(Float, nullable=False)  # 0-100 scale
    
    # Investor characteristics
    age = Column(Integer)
    investment_horizon = Column(Integer)  # Years
    annual_income = Column(Float)
    net_worth = Column(Float)
    
    # Questionnaire responses
    questionnaire_data = Column(JSON)  # Store all questionnaire answers
    
    # Behavioral traits
    behavioral_traits = Column(JSON)  # loss_aversion, overconfidence, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="risk_profile")


class Portfolio(Base):
    """Portfolio model"""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Portfolio characteristics
    total_value = Column(Float, default=0.0)
    cash_balance = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")
    rebalancing_suggestions = relationship("RebalancingSuggestion", back_populates="portfolio")
    performance_history = relationship("PortfolioPerformance", back_populates="portfolio")


class Holding(Base):
    """Holding model - represents assets in a portfolio"""
    __tablename__ = "holdings"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    
    # Asset information
    asset_symbol = Column(String(50), nullable=False)
    asset_name = Column(String(255))
    asset_type = Column(String(50))  # stock, bond, etf, crypto, etc.
    
    # Position details
    quantity = Column(Float, nullable=False)
    purchase_price = Column(Float, nullable=False)
    current_price = Column(Float)
    purchase_date = Column(DateTime, nullable=False)
    
    # Calculated fields
    cost_basis = Column(Float)  # quantity * purchase_price
    current_value = Column(Float)  # quantity * current_price
    unrealized_gain_loss = Column(Float)
    unrealized_gain_loss_pct = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="holdings")


class RebalancingSuggestion(Base):
    """Rebalancing suggestion model"""
    __tablename__ = "rebalancing_suggestions"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    
    # Current and suggested allocations
    current_allocation = Column(JSON, nullable=False)  # {symbol: {quantity, value, pct}}
    suggested_allocation = Column(JSON, nullable=False)  # {symbol: {quantity, value, pct}}
    
    # Reasoning and analysis
    reasoning = Column(JSON, nullable=False)  # Detailed explanation
    trigger_reason = Column(String(255))  # What triggered this suggestion
    
    # Metrics
    confidence_score = Column(Float)  # 0-1 scale
    expected_improvement = Column(JSON)  # Expected metrics improvement
    estimated_transaction_cost = Column(Float)
    
    # Market conditions at time of suggestion
    market_regime = Column(String(50))  # bull, bear, volatile, stable
    market_indicators = Column(JSON)
    
    # Status
    status = Column(SQLEnum(RebalancingStatus), default=RebalancingStatus.PENDING)
    
    # User feedback
    user_notes = Column(Text)
    executed_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="rebalancing_suggestions")


class MarketData(Base):
    """Market data model for storing historical and current market data"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    
    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float)
    adjusted_close = Column(Float)
    
    # Technical indicators (stored as JSON for flexibility)
    features = Column(JSON)  # RSI, MACD, Bollinger Bands, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Composite index for efficient queries
    __table_args__ = (
        {'extend_existing': True}
    )


class SentimentScore(Base):
    """Sentiment score model for news and social media sentiment"""
    __tablename__ = "sentiment_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Target (can be symbol or sector)
    target_type = Column(String(50), nullable=False)  # symbol, sector
    target_value = Column(String(100), nullable=False, index=True)
    
    # Sentiment data
    date = Column(DateTime, nullable=False, index=True)
    sentiment_score = Column(Float, nullable=False)  # -1 to 1 scale
    sentiment_magnitude = Column(Float)  # Confidence/strength of sentiment
    
    # Source information
    source = Column(String(100), nullable=False)  # news, twitter, reddit, etc.
    article_count = Column(Integer)
    
    # Raw data (optional)
    sample_headlines = Column(JSON)  # Store sample headlines/posts
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class PortfolioPerformance(Base):
    """Portfolio performance tracking"""
    __tablename__ = "portfolio_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    
    # Performance metrics
    total_value = Column(Float, nullable=False)
    daily_return = Column(Float)
    cumulative_return = Column(Float)
    
    # Risk metrics
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Additional metrics stored as JSON
    metrics = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="performance_history")


class UserAction(Base):
    """Track user actions for behavioral analysis"""
    __tablename__ = "user_actions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Action details
    action_type = Column(String(100), nullable=False)  # view, approve, reject, modify, etc.
    action_target = Column(String(100))  # portfolio_id, suggestion_id, etc.
    action_data = Column(JSON)  # Additional context
    
    # Market context at time of action
    market_conditions = Column(JSON)
    portfolio_state = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class CachedMarketData(Base):
    """Cached market data model for 1-hour TTL caching"""
    __tablename__ = "cached_market_data"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False, index=True)
    timeframe = Column(String(20), default="1h")  # 1h, 1d, etc.
    
    # OHLCV data
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    
    # Technical indicators (computed once, cached)
    indicators = Column(JSON)  # {sma_20, sma_50, rsi, macd, etc.}
    
    # Cache metadata
    fetched_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)  # fetched_at + 1 hour
    data_source = Column(String(50), default="polygon")
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_expires', 'expires_at'),
    )

