"""CrewAI tools wrapper for existing ML models"""

# Note: Using function-based tools instead of @tool decorator for now
# from crewai_tools import tool
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from scipy.optimize import minimize
from datetime import datetime, timedelta

# Import yfinance for market data
import yfinance as yf

# OpenAI integration
try:
    from openai import OpenAI
    from backend.config import get_settings
    settings = get_settings()
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
except ImportError:
    openai_client = None
    logging.warning("OpenAI not available - falling back to rule-based logic")


def call_openai_for_analysis(context: str, analysis_type: str = "portfolio") -> Dict[str, Any]:
    """
    Call OpenAI API for portfolio analysis and recommendations
    
    Args:
        context: Portfolio and market context
        analysis_type: Type of analysis (portfolio, diversification, risk)
    
    Returns:
        Dict containing OpenAI response or fallback analysis
    """
    if not openai_client:
        return {
            "success": False,
            "error": "OpenAI not available",
            "fallback": True
        }
    
    try:
        system_prompts = {
            "portfolio": "You are a portfolio strategy advisor. Analyze the given portfolio data and provide intelligent rebalancing recommendations based on Modern Portfolio Theory, risk-return optimization, and market conditions. Be specific about allocation changes and reasoning.",
            "diversification": "You are a diversification specialist. Analyze the portfolio's sector exposure and suggest specific ETFs or assets to improve diversification. Explain why each suggestion improves the portfolio.",
            "risk": "You are a risk management expert. Analyze the portfolio's risk characteristics and provide insights on risk reduction strategies, volatility management, and risk-adjusted returns."
        }
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompts.get(analysis_type, system_prompts["portfolio"])},
                {"role": "user", "content": context}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        return {
            "success": True,
            "analysis": response.choices[0].message.content,
            "model": "gpt-4"
        }
        
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback": True
        }


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


def portfolio_optimizer_tool(current_weights: Dict[str, float], risk_level: str = "moderate", 
                            market_data: Dict[str, Any] = None, user_risk_profile: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    MPT-based portfolio optimization with OpenAI reasoning
    
    Args:
        current_weights: Dict with symbol -> current weight
        risk_level: Risk level ('conservative', 'moderate', 'aggressive')
        market_data: Historical market data for optimization
        user_risk_profile: User's risk profile for personalization
    
    Returns:
        Dict containing optimized weights and reasoning
    """
    try:
        symbols = list(current_weights.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {"success": False, "error": "No assets to optimize"}
        
        # If no market data, fall back to simple optimization
        if not market_data:
            return _simple_portfolio_optimization(current_weights, risk_level)
        
        # Calculate returns and covariance matrix from market data
        returns_data = {}
        for symbol in symbols:
            if symbol in market_data and 'historical_data' in market_data[symbol]:
                try:
                    df = pd.DataFrame(market_data[symbol]['historical_data'])
                    if 'Close' in df.columns:
                        returns_data[symbol] = df['Close'].pct_change().dropna()
                except Exception as e:
                    logging.warning(f"Could not process data for {symbol}: {e}")
        
        if len(returns_data) < 2:
            return _simple_portfolio_optimization(current_weights, risk_level)
        
        # Align returns data
        min_length = min(len(returns) for returns in returns_data.values())
        aligned_returns = {}
        for symbol, returns in returns_data.items():
            aligned_returns[symbol] = returns.tail(min_length)
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(aligned_returns)
        
        # Calculate expected returns (mean annualized)
        expected_returns = returns_df.mean() * 252
        
        # Calculate covariance matrix (annualized)
        cov_matrix = returns_df.cov() * 252
        
        # Risk-free rate
        risk_free_rate = 0.02
        
        # Optimization based on risk level
        if risk_level == "conservative":
            # Minimize volatility
            weights = _minimize_volatility(expected_returns, cov_matrix, symbols)
        elif risk_level == "aggressive":
            # Maximize expected return with volatility constraint
            weights = _maximize_return_with_constraint(expected_returns, cov_matrix, symbols)
        else:  # moderate
            # Maximize Sharpe ratio
            weights = _maximize_sharpe_ratio(expected_returns, cov_matrix, symbols, risk_free_rate)
        
        # Generate OpenAI analysis
        context = f"""
        Portfolio Optimization Analysis:
        - Current weights: {current_weights}
        - Optimized weights: {weights}
        - Risk level: {risk_level}
        - Expected returns: {expected_returns.to_dict()}
        - Portfolio volatility: {np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))):.4f}
        - User risk profile: {user_risk_profile}
        """
        
        openai_analysis = call_openai_for_analysis(context, "portfolio")
        
        return {
            "success": True,
            "method": f"mpt_{risk_level}",
            "weights": weights,
            "total_weight": sum(weights.values()),
            "risk_level": risk_level,
            "expected_return": np.dot(weights, expected_returns),
            "expected_volatility": np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))),
            "sharpe_ratio": (np.dot(weights, expected_returns) - risk_free_rate) / np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))),
            "openai_analysis": openai_analysis.get("analysis", "Analysis not available"),
            "optimization_details": {
                "covariance_matrix": cov_matrix.to_dict(),
                "expected_returns": expected_returns.to_dict()
            }
        }
        
    except Exception as e:
        logging.error(f"Portfolio optimization error: {e}")
        return _simple_portfolio_optimization(current_weights, risk_level)


def _simple_portfolio_optimization(current_weights: Dict[str, float], risk_level: str) -> Dict[str, Any]:
    """Fallback simple optimization when market data is not available"""
    symbols = list(current_weights.keys())
    n_assets = len(symbols)
    
    if risk_level == "conservative":
        # Equal weight with slight bias to first asset
        weights = {symbol: 1.0 / n_assets for symbol in symbols}
        first_symbol = symbols[0]
        weights[first_symbol] += 0.1
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
    elif risk_level == "aggressive":
        # Concentrated portfolio
        weights = {symbol: 0.1 for symbol in symbols}
        if n_assets >= 2:
            weights[symbols[0]] = 0.4
            weights[symbols[1]] = 0.2
        else:
            weights[symbols[0]] = 0.7
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
    else:  # moderate
        weights = {symbol: 1.0 / n_assets for symbol in symbols}
    
    return {
        "success": True,
        "method": f"simple_{risk_level}",
        "weights": weights,
        "total_weight": sum(weights.values()),
        "risk_level": risk_level
    }


def _minimize_volatility(expected_returns: pd.Series, cov_matrix: pd.DataFrame, symbols: List[str]) -> Dict[str, float]:
    """Minimize portfolio volatility"""
    n = len(symbols)
    
    def objective(weights):
        return np.dot(weights, np.dot(cov_matrix, weights))
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n)]
    
    result = minimize(objective, [1/n] * n, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return dict(zip(symbols, result.x))


def _maximize_return_with_constraint(expected_returns: pd.Series, cov_matrix: pd.DataFrame, symbols: List[str]) -> Dict[str, float]:
    """Maximize expected return with volatility constraint"""
    n = len(symbols)
    
    def objective(weights):
        return -np.dot(weights, expected_returns)  # Negative for maximization
    
    def volatility_constraint(weights):
        return 0.25 - np.dot(weights, np.dot(cov_matrix, weights))  # Max 25% volatility
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': volatility_constraint}
    ]
    bounds = [(0, 1) for _ in range(n)]
    
    result = minimize(objective, [1/n] * n, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return dict(zip(symbols, result.x))


def _maximize_sharpe_ratio(expected_returns: pd.Series, cov_matrix: pd.DataFrame, symbols: List[str], risk_free_rate: float) -> Dict[str, float]:
    """Maximize Sharpe ratio"""
    n = len(symbols)
    
    def negative_sharpe(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n)]
    
    result = minimize(negative_sharpe, [1/n] * n, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return dict(zip(symbols, result.x))


def suggest_diversification_assets_tool(current_portfolio: Dict[str, float], market_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Suggest diversification assets based on sector analysis
    
    Args:
        current_portfolio: Current portfolio weights
        market_data: Market data for analysis
    
    Returns:
        Dict containing diversification suggestions
    """
    try:
        # Define sector asset universe
        sector_assets = {
            "Technology": ["QQQ", "XLK", "VGT"],
            "Financial": ["XLF", "KRE", "VFH"],
            "Healthcare": ["XLV", "IBB", "VHT"],
            "Consumer": ["XLY", "XLP", "VCR", "VDC"],
            "Industrial": ["XLI", "VIS"],
            "Energy": ["XLE", "VDE"],
            "Materials": ["XLB", "VAW"],
            "Real Estate": ["XLRE", "VNQ"],
            "Utilities": ["XLU", "VPU"],
            "Bonds": ["AGG", "BND", "TLT", "IEF"],
            "International": ["EFA", "VWO", "VEA", "VXUS"],
            "Commodities": ["GLD", "SLV", "DBA"]
        }
        
        # Analyze current portfolio sector exposure
        current_sectors = _analyze_portfolio_sectors(current_portfolio)
        
        # Identify under-represented sectors
        target_sector_weights = {
            "Technology": 0.20,
            "Financial": 0.15,
            "Healthcare": 0.15,
            "Consumer": 0.15,
            "Industrial": 0.10,
            "Energy": 0.05,
            "Materials": 0.05,
            "Real Estate": 0.05,
            "Utilities": 0.05,
            "Bonds": 0.20,
            "International": 0.15,
            "Commodities": 0.05
        }
        
        # Find sectors that need more exposure
        diversification_needs = []
        for sector, target_weight in target_sector_weights.items():
            current_weight = current_sectors.get(sector, 0)
            if current_weight < target_weight * 0.5:  # Less than half of target
                diversification_needs.append({
                    "sector": sector,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "suggested_assets": sector_assets[sector][:2]  # Top 2 assets per sector
                })
        
        # Sort by diversification need
        diversification_needs.sort(key=lambda x: x["target_weight"] - x["current_weight"], reverse=True)
        
        # Generate suggestions (top 3 sectors)
        suggestions = []
        for need in diversification_needs[:3]:
            for asset in need["suggested_assets"]:
                if asset not in current_portfolio:
                    suggestions.append({
                        "symbol": asset,
                        "sector": need["sector"],
                        "reason": f"Improve {need['sector']} diversification",
                        "suggested_weight": min(0.05, need["target_weight"] - need["current_weight"]),
                        "current_sector_exposure": need["current_weight"],
                        "target_sector_exposure": need["target_weight"]
                    })
        
        # Generate OpenAI analysis
        context = f"""
        Portfolio Diversification Analysis:
        - Current portfolio: {current_portfolio}
        - Current sector exposure: {current_sectors}
        - Diversification needs: {diversification_needs[:3]}
        - Suggested assets: {[s['symbol'] for s in suggestions]}
        """
        
        openai_analysis = call_openai_for_analysis(context, "diversification")
        
        return {
            "success": True,
            "current_sector_exposure": current_sectors,
            "diversification_needs": diversification_needs,
            "suggested_assets": suggestions,
            "openai_analysis": openai_analysis.get("analysis", "Diversification analysis not available"),
            "recommendations": {
                "add_assets": suggestions[:3],  # Top 3 suggestions
                "sector_improvements": diversification_needs[:3]
            }
        }
        
    except Exception as e:
        logging.error(f"Diversification analysis error: {e}")
        return {
            "success": False,
            "error": str(e),
            "suggested_assets": []
        }


def _analyze_portfolio_sectors(portfolio_weights: Dict[str, float]) -> Dict[str, float]:
    """Analyze current portfolio sector exposure"""
    # Simplified sector mapping (in production, use a comprehensive mapping)
    sector_mapping = {
        # Technology
        "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "AMZN": "Technology",
        "META": "Technology", "NFLX": "Technology", "NVDA": "Technology", "TSLA": "Technology",
        "QQQ": "Technology", "XLK": "Technology", "VGT": "Technology",
        
        # Financial
        "JPM": "Financial", "BAC": "Financial", "WFC": "Financial", "GS": "Financial",
        "XLF": "Financial", "KRE": "Financial", "VFH": "Financial",
        
        # Healthcare
        "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare", "ABBV": "Healthcare",
        "XLV": "Healthcare", "IBB": "Healthcare", "VHT": "Healthcare",
        
        # Consumer
        "KO": "Consumer", "PEP": "Consumer", "WMT": "Consumer", "PG": "Consumer",
        "XLY": "Consumer", "XLP": "Consumer", "VCR": "Consumer", "VDC": "Consumer",
        
        # Bonds
        "AGG": "Bonds", "BND": "Bonds", "TLT": "Bonds", "IEF": "Bonds",
        
        # International
        "EFA": "International", "VWO": "International", "VEA": "International", "VXUS": "International",
        
        # Commodities
        "GLD": "Commodities", "SLV": "Commodities", "DBA": "Commodities"
    }
    
    sector_weights = {}
    for symbol, weight in portfolio_weights.items():
        sector = sector_mapping.get(symbol, "Other")
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    return sector_weights


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


def risk_calculator_tool(portfolio_data: Dict[str, Any], user_risk_profile: Dict[str, Any] = None, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Enhanced portfolio risk calculation using MPT and user profile integration
    
    Args:
        portfolio_data: Dict with portfolio information
        user_risk_profile: User's risk profile from database
        market_data: Historical market data for volatility calculations
    
    Returns:
        Dict containing comprehensive risk metrics
    """
    try:
        holdings = portfolio_data.get('holdings', [])
        
        if not holdings:
            return {"success": False, "error": "No holdings to analyze"}
        
        # Calculate basic portfolio metrics
        total_value = sum(h.get('current_value', 0) for h in holdings)
        total_gain_loss = sum(h.get('unrealized_gain_loss', 0) for h in holdings)
        n_assets = len(holdings)
        
        # Calculate weights
        weights = {}
        for holding in holdings:
            symbol = holding.get('asset_symbol', '')
            value = holding.get('current_value', 0)
            weights[symbol] = value / total_value if total_value > 0 else 0
        
        # Calculate actual portfolio volatility from market data if available
        portfolio_volatility = 0.15  # Default fallback
        if market_data:
            try:
                # Calculate returns for each asset
                returns_data = {}
                for symbol, data in market_data.items():
                    if 'historical_data' in data and data['historical_data']:
                        df = pd.DataFrame(data['historical_data'])
                        if 'Close' in df.columns:
                            returns_data[symbol] = df['Close'].pct_change().dropna()
                
                if returns_data:
                    # Calculate portfolio returns
                    portfolio_returns = pd.Series(0, index=list(returns_data.values())[0].index)
                    for symbol, returns in returns_data.items():
                        if symbol in weights:
                            portfolio_returns += returns * weights[symbol]
                    
                    # Calculate annualized volatility
                    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            except Exception as e:
                logging.warning(f"Could not calculate volatility from market data: {e}")
        
        # Calculate concentration risk (Herfindahl index)
        concentration_risk = sum(w**2 for w in weights.values())
        
        # Calculate diversification score (1 - concentration_risk)
        diversification_score = 1 - concentration_risk
        
        # User risk profile integration
        user_risk_score = 0.5  # Default moderate
        if user_risk_profile:
            # Extract risk tolerance from user profile
            risk_level = user_risk_profile.get('risk_level', 'moderate')
            risk_score = user_risk_profile.get('risk_score', 50)  # 0-100 scale
            
            # Convert to 0-1 scale
            user_risk_score = risk_score / 100.0
            
            # Adjust based on investment horizon
            horizon = user_risk_profile.get('investment_horizon', 5)
            if horizon < 3:
                user_risk_score *= 0.8  # Reduce risk for short horizon
            elif horizon > 10:
                user_risk_score *= 1.2  # Increase risk tolerance for long horizon
        
        # Calculate Sharpe ratio (simplified)
        expected_return = 0.08  # Assume 8% annual return
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Calculate beta (simplified - would need market data)
        portfolio_beta = 1.0  # Default to market beta
        
        # Risk-adjusted score combining portfolio metrics (60%) and user tolerance (40%)
        portfolio_risk_score = (concentration_risk * 0.4 + (1 - diversification_score) * 0.3 + 
                               min(portfolio_volatility / 0.2, 1) * 0.3)  # Normalize volatility
        
        combined_risk_score = portfolio_risk_score * 0.6 + (1 - user_risk_score) * 0.4
        
        # Risk level classification
        if combined_risk_score < 0.3:
            risk_level = "low"
        elif combined_risk_score < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            "success": True,
            "metrics": {
                "total_value": total_value,
                "total_gain_loss": total_gain_loss,
                "diversification_score": diversification_score,
                "concentration_risk": concentration_risk,
                "portfolio_volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "portfolio_beta": portfolio_beta,
                "number_of_assets": n_assets,
                "risk_level": risk_level,
                "combined_risk_score": combined_risk_score,
                "user_risk_tolerance": user_risk_score,
                "weights": weights
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
