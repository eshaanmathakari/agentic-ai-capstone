"""Market Assets API - Top 500 Assets Management"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta

from backend.database import get_db, User
from backend.api.routes.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

# Top 500 Assets by Market Cap (simplified lists for demo)
TOP_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH",
    "JNJ", "XOM", "JPM", "V", "PG", "MA", "HD", "CVX", "PFE", "ABBV",
    "BAC", "KO", "AVGO", "PEP", "TMO", "COST", "WMT", "DHR", "ABT", "VZ",
    "ADBE", "ACN", "NFLX", "NKE", "TXN", "CRM", "LIN", "AMD", "QCOM", "T",
    "NEE", "PM", "SPGI", "UNP", "RTX", "HON", "LOW", "AMGN", "IBM", "SBUX",
    "CAT", "GE", "GS", "AXP", "BKNG", "BLK", "MMM", "BA", "CVS", "GILD",
    "UPS", "DE", "PYPL", "NOW", "ADP", "ISRG", "AMAT", "INTU", "MDT", "CMCSA",
    "MO", "TGT", "LMT", "CI", "SPG", "SO", "DUK", "PLD", "EOG", "SLB",
    "PGR", "FISV", "ICE", "EW", "AON", "ITW", "SYK", "CL", "BDX", "PNC",
    "USB", "EMR", "NOC", "BSX", "APD", "AEP", "EXC", "ETN", "WM", "ECL",
    "SHW", "FDX", "COP", "VRTX", "MSI", "A", "GIS", "HCA", "NSC", "LHX"
]

TOP_CRYPTO = [
    "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD", "XRP-USD", "USDC-USD", 
    "ADA-USD", "AVAX-USD", "DOGE-USD", "TRX-USD", "LINK-USD", "DOT-USD", "MATIC-USD",
    "SHIB-USD", "LTC-USD", "UNI-USD", "ATOM-USD", "ETC-USD", "XLM-USD"
]

PRECIOUS_METALS = [
    "GC=F",  # Gold
    "SI=F",  # Silver
    "PL=F",  # Platinum
    "PA=F",  # Palladium
    "HG=F",  # Copper
    "ZC=F",  # Corn
    "ZS=F",  # Soybeans
    "ZW=F",  # Wheat
    "NG=F",  # Natural Gas
    "CL=F"   # Crude Oil
]


@router.get("/top500")
async def get_top500_assets(
    category: Optional[str] = Query(None, description="Filter by category: stocks, crypto, metals"),
    limit: int = Query(500, description="Maximum number of assets to return"),
    db: Session = Depends(get_db)
):
    """Get top 500 assets by market cap across different categories"""
    try:
        all_assets = []
        
        if not category or category == "stocks":
            # Get stock data
            stock_data = await _fetch_asset_data(TOP_STOCKS[:100], "stocks")
            all_assets.extend(stock_data)
        
        if not category or category == "crypto":
            # Get crypto data
            crypto_data = await _fetch_asset_data(TOP_CRYPTO, "crypto")
            all_assets.extend(crypto_data)
        
        if not category or category == "metals":
            # Get precious metals data
            metals_data = await _fetch_asset_data(PRECIOUS_METALS, "metals")
            all_assets.extend(metals_data)
        
        # Sort by market cap (descending) and limit results
        all_assets.sort(key=lambda x: x.get("market_cap", 0), reverse=True)
        all_assets = all_assets[:limit]
        
        return {
            "success": True,
            "count": len(all_assets),
            "category": category or "all",
            "assets": all_assets,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching top500 assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_assets(
    q: str = Query(..., description="Search query for asset name or symbol"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(20, description="Maximum number of results"),
    db: Session = Depends(get_db)
):
    """Search assets by name or symbol"""
    try:
        query = q.upper().strip()
        if len(query) < 2:
            raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
        
        # Get all assets
        all_assets = []
        if not category or category == "stocks":
            stock_data = await _fetch_asset_data(TOP_STOCKS, "stocks")
            all_assets.extend(stock_data)
        
        if not category or category == "crypto":
            crypto_data = await _fetch_asset_data(TOP_CRYPTO, "crypto")
            all_assets.extend(crypto_data)
        
        if not category or category == "metals":
            metals_data = await _fetch_asset_data(PRECIOUS_METALS, "metals")
            all_assets.extend(metals_data)
        
        # Filter by search query
        results = []
        for asset in all_assets:
            symbol = asset.get("symbol", "").upper()
            name = asset.get("name", "").upper()
            
            if query in symbol or query in name:
                results.append(asset)
        
        # Sort by relevance (exact symbol match first, then name match)
        results.sort(key=lambda x: (
            0 if query == x.get("symbol", "").upper() else 1,
            -x.get("market_cap", 0)
        ))
        
        results = results[:limit]
        
        return {
            "success": True,
            "query": q,
            "count": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/price/{symbol}")
async def get_live_price(
    symbol: str,
    db: Session = Depends(get_db)
):
    """Get live price for a specific asset"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or "currentPrice" not in info and "regularMarketPrice" not in info:
            raise HTTPException(status_code=404, detail=f"Asset {symbol} not found")
        
        current_price = info.get("currentPrice", info.get("regularMarketPrice"))
        
        return {
            "success": True,
            "symbol": symbol,
            "price": current_price,
            "change": info.get("regularMarketChange"),
            "change_percent": info.get("regularMarketChangePercent"),
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track")
async def add_to_watchlist(
    symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add asset to user's watchlist"""
    try:
        # Get asset data
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or "currentPrice" not in info and "regularMarketPrice" not in info:
            raise HTTPException(status_code=404, detail=f"Asset {symbol} not found")
        
        # TODO: Implement watchlist functionality
        # For now, just return success
        return {
            "success": True,
            "message": f"Asset {symbol} added to watchlist",
            "user_id": current_user.id,
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding {symbol} to watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _fetch_asset_data(symbols: List[str], category: str) -> List[Dict[str, Any]]:
    """Fetch data for a list of symbols"""
    assets = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                continue
            
            current_price = info.get("currentPrice", info.get("regularMarketPrice"))
            if not current_price:
                continue
            
            asset_data = {
                "symbol": symbol,
                "name": info.get("longName", info.get("shortName", symbol)),
                "price": current_price,
                "change": info.get("regularMarketChange", 0),
                "change_percent": info.get("regularMarketChangePercent", 0),
                "volume": info.get("volume", 0),
                "market_cap": info.get("marketCap", 0),
                "category": category,
                "sector": info.get("sector", "Unknown"),
                "currency": info.get("currency", "USD")
            }
            
            assets.append(asset_data)
            
        except Exception as e:
            logger.warning(f"Error fetching data for {symbol}: {e}")
            continue
    
    return assets
