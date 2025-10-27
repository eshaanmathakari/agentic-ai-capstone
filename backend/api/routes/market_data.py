"""Market data API routes"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
import logging

from backend.database import get_db, User
from backend.api.routes.auth import get_current_user
from backend.agents.tools import fetch_market_data_tool, get_live_prices_tool

router = APIRouter()
logger = logging.getLogger(__name__)


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
    try:
        # Use the tools module to fetch live prices
        result = get_live_prices_tool(symbols)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=f"Failed to fetch prices: {result.get('error', 'Unknown error')}")
        
        live_prices = result.get("live_prices", {})
        
        if not live_prices:
            raise HTTPException(status_code=404, detail="No prices found for the specified symbols")
        
        return {
            "timestamp": result.get("timestamp", datetime.utcnow().isoformat()),
            "prices": live_prices
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching latest prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/asset/{symbol}")
async def get_asset_info(
    symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed asset information"""
    try:
        # Fetch market data to get asset info
        result = fetch_market_data_tool([symbol], days=5, db_session=db)
        
        if not result.get("success") or symbol not in result.get("data", {}):
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        asset_data = result["data"][symbol]
        
        # Return basic asset info
        return {
            "symbol": symbol,
            "name": symbol,  # Could be enhanced with company name lookup
            "current_price": asset_data.get("current_price", 0),
            "volume": asset_data.get("volume", 0),
            "data_source": asset_data.get("data_source", "unknown"),
            "data_points": asset_data.get("data_points", 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching asset info for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    days: int = Query(default=365, ge=1, le=3650),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get historical data for a symbol"""
    try:
        # Fetch historical data using tools
        result = fetch_market_data_tool([symbol], days=days, db_session=db)
        
        if not result.get("success") or symbol not in result.get("data", {}):
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        symbol_data = result["data"][symbol]
        
        if "error" in symbol_data:
            raise HTTPException(status_code=404, detail=symbol_data["error"])
        
        historical_data = symbol_data.get("historical_data", [])
        
        return {
            "symbol": symbol,
            "records": len(historical_data),
            "data": historical_data,
            "current_price": symbol_data.get("current_price", 0),
            "data_source": symbol_data.get("data_source", "unknown")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update")
async def trigger_market_data_update(
    symbols: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Manually trigger market data update"""
    try:
        # If no symbols provided, use default ones
        if not symbols:
            symbols = ["SPY", "QQQ", "IEF", "GLD", "BTC-USD"]
        
        # Fetch fresh data for all symbols
        result = fetch_market_data_tool(symbols, days=30, db_session=db)
        
        return {
            "message": "Market data update completed",
            "results": result,
            "successful_fetches": result.get("successful_fetches", 0),
            "total_symbols": len(symbols)
        }
    except Exception as e:
        logger.error(f"Error updating market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

