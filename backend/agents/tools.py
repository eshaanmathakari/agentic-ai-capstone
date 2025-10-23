"""CrewAI tools wrapper for existing ML models"""

# Note: Using function-based tools instead of @tool decorator for now
# from crewai_tools import tool
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Import yfinance for market data
import yfinance as yf


def fetch_market_data_tool(symbols: List[str], days: int = 365) -> Dict[str, Any]:
    """
    Fetches historical market data for given symbols using yfinance
    
    Args:
        symbols: List of asset symbols (e.g., ['AAPL', 'GOOGL'])
        days: Number of days of historical data
    
    Returns:
        Dict containing market data for each symbol
    """
    try:
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=f"{days}d")
                
                if not df.empty:
                    # Calculate basic technical indicators
                    df['SMA_20'] = df['Close'].rolling(window=20).mean()
                    df['SMA_50'] = df['Close'].rolling(window=50).mean()
                    df['RSI'] = calculate_rsi(df['Close'])
                    
                    data[symbol] = {
                        'historical_data': df.tail(30).to_dict('records'),  # Last 30 days
                        'current_price': df['Close'].iloc[-1],
                        'sma_20': df['SMA_20'].iloc[-1],
                        'sma_50': df['SMA_50'].iloc[-1],
                        'rsi': df['RSI'].iloc[-1],
                        'volume': df['Volume'].iloc[-1]
                    }
                else:
                    data[symbol] = {'error': 'No data available'}
                    
            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {e}")
                data[symbol] = {'error': str(e)}
        
        return {
            "success": True,
            "data": data,
            "symbols": symbols,
            "days": days
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": {}
        }

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_indicators_tool(symbol: str, data: List[Dict]) -> Dict[str, Any]:
    """
    Calculate technical indicators for a given symbol's data
    
    Args:
        symbol: Asset symbol
        data: List of OHLCV data records
    
    Returns:
        Dict containing calculated indicators
    """
    try:
        if not data:
            return {"success": False, "error": "No data provided"}
        
        df = pd.DataFrame(data)
        engineer = FeatureEngineer()
        
        # Calculate all features
        df_with_features = engineer.calculate_all_features(df)
        
        # Get latest feature vector
        latest_features = engineer.get_feature_vector(df_with_features)
        
        return {
            "success": True,
            "symbol": symbol,
            "features": latest_features,
            "total_features": len(latest_features)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "features": {}
        }


def portfolio_optimizer_tool(current_weights: Dict[str, float], risk_level: str = "moderate") -> Dict[str, Any]:
    """
    Simple portfolio optimization based on risk level
    
    Args:
        current_weights: Dict with symbol -> current weight
        risk_level: Risk level ('conservative', 'moderate', 'aggressive')
    
    Returns:
        Dict containing optimized weights
    """
    try:
        symbols = list(current_weights.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {"success": False, "error": "No assets to optimize"}
        
        # Simple optimization based on risk level
        if risk_level == "conservative":
            # Equal weight with slight bias to first asset
            weights = {symbol: 1.0 / n_assets for symbol in symbols}
            # Add 10% bias to first asset
            first_symbol = symbols[0]
            weights[first_symbol] += 0.1
            # Normalize
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
        elif risk_level == "aggressive":
            # Concentrated portfolio - 60% to top 2 assets
            weights = {symbol: 0.1 for symbol in symbols}
            if n_assets >= 2:
                weights[symbols[0]] = 0.4
                weights[symbols[1]] = 0.2
            else:
                weights[symbols[0]] = 0.7
            # Normalize
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
        else:  # moderate
            # Equal weight portfolio
            weights = {symbol: 1.0 / n_assets for symbol in symbols}
        
        return {
            "success": True,
            "method": f"risk_based_{risk_level}",
            "weights": weights,
            "total_weight": sum(weights.values()),
            "risk_level": risk_level
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "weights": {}
        }


def market_regime_detector_tool(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple market regime detection based on technical indicators
    
    Args:
        market_data: Dict with symbol -> market data
    
    Returns:
        Dict containing regime information
    """
    try:
        # Simple regime detection based on RSI and price trends
        regimes = []
        confidences = []
        
        for symbol, data in market_data.items():
            if 'error' in data:
                continue
                
            rsi = data.get('rsi', 50)
            current_price = data.get('current_price', 0)
            sma_20 = data.get('sma_20', current_price)
            sma_50 = data.get('sma_50', current_price)
            
            # Simple regime logic
            if rsi > 70 and current_price > sma_20:
                regime = "bull"
                confidence = 0.8
            elif rsi < 30 and current_price < sma_20:
                regime = "bear"
                confidence = 0.8
            elif sma_20 > sma_50:
                regime = "stable"
                confidence = 0.6
            else:
                regime = "volatile"
                confidence = 0.7
                
            regimes.append(regime)
            confidences.append(confidence)
        
        # Overall regime (most common)
        if regimes:
            overall_regime = max(set(regimes), key=regimes.count)
            avg_confidence = sum(confidences) / len(confidences)
        else:
            overall_regime = "unknown"
            avg_confidence = 0.0
        
        return {
            "success": True,
            "regime": overall_regime,
            "regime_name": overall_regime.title(),
            "confidence": avg_confidence,
            "individual_regimes": dict(zip(market_data.keys(), regimes))
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "regime": "unknown"
        }


def risk_calculator_tool(portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple portfolio risk calculation
    
    Args:
        portfolio_data: Dict with portfolio information
    
    Returns:
        Dict containing risk metrics
    """
    try:
        # Simple risk metrics based on portfolio composition
        holdings = portfolio_data.get('holdings', [])
        
        if not holdings:
            return {"success": False, "error": "No holdings to analyze"}
        
        # Calculate basic metrics
        total_value = sum(h.get('current_value', 0) for h in holdings)
        total_gain_loss = sum(h.get('unrealized_gain_loss', 0) for h in holdings)
        
        # Simple diversification score (1 = perfectly diversified)
        n_assets = len(holdings)
        diversification_score = min(1.0, n_assets / 10.0)  # Max score at 10+ assets
        
        # Simple risk score based on concentration
        max_weight = max((h.get('current_value', 0) / total_value) for h in holdings) if total_value > 0 else 0
        concentration_risk = max_weight  # Higher = more concentrated
        
        # Simple volatility estimate
        volatility = 0.15  # Default 15% annual volatility
        if concentration_risk > 0.5:
            volatility *= 1.5  # Higher volatility for concentrated portfolios
        
        return {
            "success": True,
            "metrics": {
                "total_value": total_value,
                "total_gain_loss": total_gain_loss,
                "diversification_score": diversification_score,
                "concentration_risk": concentration_risk,
                "estimated_volatility": volatility,
                "number_of_assets": n_assets,
                "risk_level": "high" if concentration_risk > 0.4 else "medium" if concentration_risk > 0.2 else "low"
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "metrics": {}
        }


def rebalancing_engine_tool(portfolio_id: int, current_weights: Dict[str, float], 
                          target_weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Simple rebalancing recommendations
    
    Args:
        portfolio_id: Portfolio ID
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
    
    Returns:
        Dict containing rebalancing recommendations
    """
    try:
        actions = []
        total_trades = 0
        
        # Calculate rebalancing actions
        for symbol in set(current_weights.keys()) | set(target_weights.keys()):
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            
            # Only include significant changes (>1%)
            if abs(weight_diff) > 0.01:
                action_type = "buy" if weight_diff > 0 else "sell"
                action_amount = abs(weight_diff)
                
                actions.append({
                    "symbol": symbol,
                    "action": action_type,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "weight_change": weight_diff,
                    "action_amount": action_amount,
                    "reason": f"Rebalance to {target_weight:.1%} target allocation"
                })
                total_trades += 1
        
        # Generate simple explanation
        explanation = {
            "summary": f"Rebalancing requires {total_trades} trades to align with target allocation",
            "key_changes": [f"{a['symbol']}: {a['action']} {a['action_amount']:.1%}" for a in actions[:3]],
            "rationale": "Portfolio rebalancing helps maintain target risk-return profile"
        }
        
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "actions": actions,
            "explanation": explanation,
            "total_trades": total_trades
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "actions": []
        }


def get_live_prices_tool(symbols: List[str]) -> Dict[str, Any]:
    """
    Get live prices for given symbols
    
    Args:
        symbols: List of asset symbols
    
    Returns:
        Dict containing live price data
    """
    try:
        import yfinance as yf
        
        live_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                live_data[symbol] = {
                    "price": info.get("currentPrice", info.get("regularMarketPrice")),
                    "change": info.get("regularMarketChange"),
                    "change_percent": info.get("regularMarketChangePercent"),
                    "volume": info.get("volume"),
                    "market_cap": info.get("marketCap")
                }
            except Exception as e:
                logging.error(f"Error getting live price for {symbol}: {e}")
                live_data[symbol] = {"error": str(e)}
        
        return {
            "success": True,
            "live_prices": live_data,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "live_prices": {}
        }
