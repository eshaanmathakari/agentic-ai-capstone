"""Market data API routes"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from backend.database import get_db, User
from backend.api.routes.auth import get_current_user
# Simplified imports - using yfinance directly in tools

router = APIRouter()


# Pydantic models
class PriceUpdate(BaseModel):
    """Price update response"""
    symbol: str
    price: float
    timestamp: datetime


class AssetInfo(BaseModel):
    """Asset information"""
    symbol: str
    name: str
    sector: Optional[str]
    industry: Optional[str]
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    beta: Optional[float]


# Routes
@router.get("/prices")
async def get_latest_prices(
    symbols: List[str] = Query(..., description="List of symbols to fetch"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get latest prices for multiple symbols"""
    collector = MarketDataCollector(db)
    
    prices = collector.get_multiple_latest_prices(symbols)
    
    if not prices:
        raise HTTPException(status_code=404, detail="No prices found for the specified symbols")
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "prices": prices
    }


@router.get("/asset/{symbol}")
async def get_asset_info(
    symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed asset information"""
    collector = MarketDataCollector(db)
    
    info = collector.get_asset_info(symbol)
    
    return info


@router.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    days: int = Query(default=365, ge=1, le=3650),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get historical data for a symbol"""
    collector = MarketDataCollector(db)
    feature_engineer = FeatureEngineer()
    
    # Get data from database
    df = collector.get_historical_data(symbol, days=days)
    
    if df.empty:
        # If no data in database, fetch from Yahoo Finance
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        data_dict = collector.fetch_yahoo_finance_data(
            [symbol],
            start_date=start_date,
            end_date=end_date
        )
        
        if symbol not in data_dict:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        df = data_dict[symbol]
        
        # Save to database
        collector.save_market_data(symbol, df)
    
    # Calculate features
    df_with_features = feature_engineer.calculate_all_features(df)
    
    # Convert to JSON-serializable format
    df_reset = df_with_features.reset_index()
    df_reset['date'] = df_reset['date'].dt.strftime('%Y-%m-%d')
    
    data = df_reset.to_dict(orient='records')
    
    return {
        "symbol": symbol,
        "records": len(data),
        "data": data
    }


@router.post("/update")
async def trigger_market_data_update(
    symbols: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Manually trigger market data update"""
    collector = MarketDataCollector(db)
    
    results = collector.update_all_symbols(symbols=symbols, period="5d")
    
    return {
        "message": "Market data update completed",
        "results": results
    }

