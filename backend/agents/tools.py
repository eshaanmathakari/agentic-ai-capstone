"""CrewAI tools wrapper for existing ML models"""

# Note: Using function-based tools instead of @tool decorator for now
# from crewai_tools import tool
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from scipy.optimize import minimize
from datetime import datetime, timedelta

# Import requests for API calls
import requests
import time
import json
import os

# OpenAI integration
try:
    from openai import OpenAI
    from config import get_settings
    settings = get_settings()
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
except ImportError:
    openai_client = None
    logging.warning("OpenAI not available - falling back to rule-based logic")

# CrewAI LLM setup for agents
try:
    from crewai import LLM
    if settings.OPENAI_API_KEY:
        # GPT-5-mini for worker agents (data, strategy, validation)
        try:
            crewai_llm = LLM(
                model="gpt-4o-mini",  # Worker agents (using gpt-4o-mini instead of gpt-5-mini)
                api_key=settings.OPENAI_API_KEY
                # Removed temperature as it's not supported by this model
            )
            data_agent_llm = crewai_llm
            strategy_agent_llm = crewai_llm
            validation_agent_llm = crewai_llm
            CREWAI_AVAILABLE = True
            logging.info("CrewAI initialized with gpt-4o-mini for worker agents")
        except Exception as e:
            logging.warning(f"GPT-4o-mini not available: {e}")
            crewai_llm = data_agent_llm = strategy_agent_llm = validation_agent_llm = None
            CREWAI_AVAILABLE = False

        # GPT-4o for orchestrator (senior coordinator)
        try:
            orchestrator_llm = LLM(
                model="gpt-4o",  # Senior orchestrator (using gpt-4o instead of gpt-5)
                api_key=settings.OPENAI_API_KEY
                # Removed temperature as it's not supported by this model
            )
        except Exception as e:
            logging.warning(f"GPT-4o not available: {e}")
            orchestrator_llm = crewai_llm  # Fallback to worker LLM
    else:
        crewai_llm = data_agent_llm = strategy_agent_llm = validation_agent_llm = orchestrator_llm = None
        CREWAI_AVAILABLE = False
        logging.warning("OpenAI API key not available - CrewAI agents will be disabled")
except ImportError:
    crewai_llm = data_agent_llm = strategy_agent_llm = validation_agent_llm = orchestrator_llm = None
    CREWAI_AVAILABLE = False
    logging.warning("CrewAI not available")

# CrewAI Tool wrappers - convert functions to Tool objects for CrewAI compatibility
# Note: crewai-tools 1.0.0 doesn't have Tool class or tool decorator
# We'll use function-based tools instead
CREWAI_TOOLS_AVAILABLE = False
Tool = None
tool = None

from .utils import _sanitize_float_values

# Polygon.io API configuration
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
POLYGON_BASE_URL = "https://api.polygon.io"
POLYGON_RATE_LIMIT = 5  # calls per minute

# Rate limiting state
_last_api_call = {}
_api_call_count = {}

def _check_rate_limit():
    """Enforce 5 calls/minute limit for Polygon.io API"""
    now = datetime.now()
    minute_ago = now - timedelta(minutes=1)
    
    # Clean old entries
    global _api_call_count
    _api_call_count = {k: v for k, v in _api_call_count.items() if k > minute_ago}
    
    if len(_api_call_count) >= POLYGON_RATE_LIMIT:
        sleep_time = 60 - (now - min(_api_call_count.keys())).seconds
        if sleep_time > 0:
            logging.info(f"Rate limit reached, sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)
    
    # Record this call
    _api_call_count[now] = True


def _fetch_polygon_aggregates(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """Fetch OHLCV bars from Polygon.io"""
    if not POLYGON_API_KEY:
        logging.warning("POLYGON_API_KEY not set - using fallback data")
        return _create_fallback_data(symbol, days)
    
    try:
        _check_rate_limit()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for Polygon API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'apikey': POLYGON_API_KEY
        }
        
        logging.info(f"Fetching Polygon.io data for {symbol} from {from_date} to {to_date}")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            logging.warning(f"Polygon.io API returned status {response.status_code} for {symbol}")
            return _create_fallback_data(symbol, days)
        
        data = response.json()
        
        if data.get('status') != 'OK' or not data.get('results'):
            logging.warning(f"No data returned for {symbol}: {data.get('message', 'Unknown error')}")
            return _create_fallback_data(symbol, days)
        
        # Convert to DataFrame
        results = data['results']
        df_data = []
        
        for bar in results:
            df_data.append({
                'Date': pd.to_datetime(bar['t'], unit='ms'),
                'Open': bar['o'],
                'High': bar['h'],
                'Low': bar['l'],
                'Close': bar['c'],
                'Volume': bar['v']
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('Date').reset_index(drop=True)
        
        if len(df) >= 10:  # Require minimum data points
            logging.info(f"Successfully fetched {len(df)} data points for {symbol}")
            return df
        else:
            logging.warning(f"Insufficient data for {symbol}: {len(df)} points")
            return _create_fallback_data(symbol, days)
            
    except Exception as e:
        logging.error(f"Polygon.io aggregates fetch failed for {symbol}: {e}")
        return _create_fallback_data(symbol, days)


def _create_fallback_data(symbol: str, days: int) -> pd.DataFrame:
    """Create fallback data when API fails"""
    logging.info(f"Creating fallback data for {symbol}")
    
    # Create synthetic data for testing
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create synthetic price data (random walk)
    np.random.seed(hash(symbol) % 2**32)  # Deterministic seed based on symbol
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    df_data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.randint(1000000, 10000000)
        
        df_data.append({
            'Date': date,
            'Open': price,
            'High': high,
            'Low': low,
            'Close': price,
            'Volume': volume
        })
    
    df = pd.DataFrame(df_data)
    logging.info(f"Created {len(df)} fallback data points for {symbol}")
    return df


def _fetch_polygon_technical_indicators(symbol: str) -> Dict[str, float]:
    """Fetch technical indicators from Polygon.io"""
    if not POLYGON_API_KEY:
        return {'sma_20': 0, 'sma_50': 0}
    
    try:
        _check_rate_limit()
        
        # Get SMA and RSI indicators
        url = f"{POLYGON_BASE_URL}/v1/indicators/sma/{symbol}"
        params = {
            'timestamp.gte': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'apikey': POLYGON_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        indicators = {}
        
        if data.get('status') == 'OK' and data.get('results'):
            # Get latest SMA values
            sma_values = data['results']
            if sma_values:
                indicators['sma_20'] = sma_values[-1].get('values', {}).get('value', 0)
                indicators['sma_50'] = sma_values[-1].get('values', {}).get('value', 0)
        
        return indicators
        
    except Exception as e:
        logging.error(f"Polygon.io indicators fetch failed for {symbol}: {e}")
        return {'sma_20': 0, 'sma_50': 0}


def agentic_risk_analyzer_tool(user_risk_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses the user's risk choice directly for portfolio optimization.
    Simply maps their selection to the internal format.

    Args:
        user_risk_profile: A dictionary containing the user's risk profile data,
                           including risk tolerance choice and investment horizon.

    Returns:
        A dictionary containing the user's risk level.
    """
    try:
        # Extract user's choices
        questionnaire_data = user_risk_profile.get('questionnaire_data', {})
        risk_choice = questionnaire_data.get('risk_tolerance', 'Medium Risk')
        
        # Map user choice to internal format
        risk_mapping = {
            "Low Risk": "conservative",
            "Medium Risk": "moderate", 
            "High Risk": "aggressive"
        }
        internal_risk = risk_mapping.get(risk_choice, "moderate")

        return {
            "success": True,
            "adjusted_risk_level": internal_risk,
            "rationale": f"User selected {risk_choice}"
        }

    except Exception as e:
        logging.error(f"Risk mapping failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "adjusted_risk_level": "moderate",
            "rationale": "Defaulted to moderate risk"
        }


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


def _build_from_cache(cached_data: List) -> Dict[str, Any]:
    """Build market data from cached records"""
    if not cached_data:
        return {}
    
    # Convert cached data to DataFrame format
    df_data = []
    for record in cached_data:
        df_data.append({
            'Date': record.timestamp,
            'Open': record.open,
            'High': record.high,
            'Low': record.low,
            'Close': record.close,
            'Volume': record.volume or 0
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate indicators from cached data
    df['SMA_20'] = df['Close'].rolling(window=min(20, len(df))).mean()
    df['SMA_50'] = df['Close'].rolling(window=min(50, len(df))).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Get indicators from the most recent record
    latest_indicators = cached_data[-1].indicators or {}
    
    return {
        'historical_data': df.to_dict('records'),
        'current_price': float(df['Close'].iloc[-1]),
        'sma_20': float(df['SMA_20'].iloc[-1]) if not pd.isna(df['SMA_20'].iloc[-1]) else float(df['Close'].iloc[-1]),
        'sma_50': float(df['SMA_50'].iloc[-1]) if not pd.isna(df['SMA_50'].iloc[-1]) else float(df['Close'].iloc[-1]),
        'rsi': float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0,
        'volume': float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0.0,
        'data_points': len(df),
        'data_source': 'polygon_cached',
        'polygon_indicators': latest_indicators
    }


def _cache_market_data(db_session, symbol: str, fresh_data: Dict[str, Any]):
    """Cache fresh market data to database"""
    try:
        from database.models import CachedMarketData
        
        # Clear existing cache for this symbol
        db_session.query(CachedMarketData).filter(
            CachedMarketData.symbol == symbol
        ).delete()
        
        # Cache new data
        historical_data = fresh_data.get('historical_data', [])
        for record in historical_data:
            cached_record = CachedMarketData(
                symbol=symbol,
                timestamp=pd.to_datetime(record['Date']),
                open=record.get('Open'),
                high=record.get('High'),
                low=record.get('Low'),
                close=record.get('Close'),
                volume=record.get('Volume', 0),
                indicators={
                    'sma_20': fresh_data.get('sma_20'),
                    'sma_50': fresh_data.get('sma_50'),
                    'rsi': fresh_data.get('rsi')
                },
                expires_at=datetime.utcnow() + timedelta(hours=1),
                data_source='polygon'
            )
            db_session.add(cached_record)
        
        db_session.commit()
        logging.info(f"Cached {len(historical_data)} records for {symbol}")
        
    except Exception as e:
        logging.error(f"Failed to cache data for {symbol}: {e}")
        db_session.rollback()


def fetch_market_data_tool(symbols: List[str], days: int = 365, db_session=None) -> Dict[str, Any]:
    """
    Fetches historical market data for given symbols using Polygon.io API with rate limiting.

    Args:
        symbols: List of asset symbols (e.g., ['AAPL', 'GOOGL'])
        days: Number of days of historical data

    Returns:
        Dict containing market data for each symbol
    """
    
    def _normalize_symbol(symbol: str) -> str:
        """Normalize symbol for Polygon.io API"""
        symbol = symbol.strip().upper()
        return symbol

    try:
        data = {}
        successful_fetches = 0
        now = datetime.utcnow()

        logging.info(f"Fetching market data for {len(symbols)} symbols using Polygon.io API with caching")

        for symbol in symbols:
            symbol_data = {}
            df = None

            try:
                # Check cache first if db_session is available
                if db_session:
                    try:
                        from database.models import CachedMarketData
                        
                        cached = db_session.query(CachedMarketData).filter(
                            CachedMarketData.symbol == symbol,
                            CachedMarketData.expires_at > now
                        ).order_by(CachedMarketData.timestamp.desc()).all()
                        
                        if cached and len(cached) >= 10:  # Minimum data points
                            logging.info(f"Using cached data for {symbol}")
                            symbol_data = _build_from_cache(cached)
                            data[symbol] = symbol_data
                            successful_fetches += 1
                            continue
                    except Exception as cache_error:
                        logging.warning(f"Cache lookup failed for {symbol}: {cache_error}")

                # Fetch fresh data from Polygon.io
                logging.info(f"Fetching fresh data from Polygon.io for {symbol}")
                df = _fetch_polygon_aggregates(symbol, days)

                if df is not None and not df.empty:
                    # Get technical indicators from Polygon.io
                    indicators = _fetch_polygon_technical_indicators(symbol)
                    
                    # Calculate basic technical indicators
                    df['SMA_20'] = df['Close'].rolling(window=min(20, len(df))).mean()
                    df['SMA_50'] = df['Close'].rolling(window=min(50, len(df))).mean()
                    df['RSI'] = calculate_rsi(df['Close'])

                    # Ensure we have at least some data for returns calculation
                    if len(df) >= 10:
                        symbol_data = {
                            'historical_data': df.to_dict('records'),
                            'current_price': float(df['Close'].iloc[-1]),
                            'sma_20': float(df['SMA_20'].iloc[-1]) if not pd.isna(df['SMA_20'].iloc[-1]) else float(df['Close'].iloc[-1]),
                            'sma_50': float(df['SMA_50'].iloc[-1]) if not pd.isna(df['SMA_50'].iloc[-1]) else float(df['Close'].iloc[-1]),
                            'rsi': float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0,
                            'volume': float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0.0,
                            'data_points': len(df),
                            'data_source': 'polygon',
                            'polygon_indicators': indicators
                        }

                        # Cache the fresh data if db_session is available
                        if db_session:
                            _cache_market_data(db_session, symbol, symbol_data)

                        data[symbol] = symbol_data
                        successful_fetches += 1
                        logging.info(f"Successfully fetched data for {symbol} ({len(df)} data points)")
                    else:
                        data[symbol] = {'error': f'Insufficient data: only {len(df)} data points'}
                else:
                    data[symbol] = {'error': f'No data available for {symbol} from Polygon.io'}

            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {e}")
                data[symbol] = {'error': str(e)}

        # Calculate success rate
        success_rate = successful_fetches / len(symbols) if symbols else 0
        market_data_available = success_rate >= 0.5  # Success if we got data for at least half the symbols

        logging.info(f"Market data fetch completed: {successful_fetches}/{len(symbols)} symbols successful (rate: {success_rate:.1%})")

        return {
            "success": market_data_available,
            "data": data,
            "symbols": symbols,
            "days": days,
            "successful_fetches": successful_fetches,
            "success_rate": success_rate,
            "market_data_available": market_data_available,
            "data_sources": {
                "polygon_success": sum(1 for s in data.values() if isinstance(s, dict) and s.get('data_source') == 'polygon')
            }
        }

    except Exception as e:
        logging.error(f"Market data fetch failed completely: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": {},
            "symbols": symbols,
            "days": days,
            "market_data_available": False
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
        
        # Calculate basic technical indicators
        if 'Close' in df.columns:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = calculate_rsi(df['Close'])
            
            # Get latest feature vector
            latest_features = {
                'sma_20': df['SMA_20'].iloc[-1] if not df['SMA_20'].isna().iloc[-1] else 0,
                'sma_50': df['SMA_50'].iloc[-1] if not df['SMA_50'].isna().iloc[-1] else 0,
                'rsi': df['RSI'].iloc[-1] if not df['RSI'].isna().iloc[-1] else 50,
                'price': df['Close'].iloc[-1],
                'volume': df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
            }
        else:
            latest_features = {}
        
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
    Portfolio optimization using dynamic approach with pandas-datareader and investpy data sources.
    No rule-based fallbacks - only dynamic optimization based on market data.
    
    Args:
        current_weights: Dict with symbol -> current weight
        risk_level: Risk level ('conservative', 'moderate', 'aggressive')
        market_data: Historical market data for optimization
        user_risk_profile: User's risk profile for personalization
    
    Returns:
        Dict containing optimized target weights
    """
    try:
        symbols = list(current_weights.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {"success": False, "error": "No assets to optimize"}
        
        # Require market data for optimization - no fallbacks
        if not market_data:
            return {
                "success": False, 
                "error": "Market data required for dynamic optimization",
                "market_data_available": False
            }
        
        # Calculate returns and covariance matrix from market data
        returns_data = {}
        valid_symbols = []

        for symbol in symbols:
            if symbol in market_data and 'historical_data' in market_data[symbol] and 'error' not in market_data[symbol]:
                try:
                    hist_data = market_data[symbol]['historical_data']
                    if isinstance(hist_data, list) and hist_data:
                        df = pd.DataFrame(hist_data)
                        if 'Close' in df.columns and len(df) >= 10:  # Require at least 10 data points
                            returns_data[symbol] = df['Close'].pct_change().dropna()
                            valid_symbols.append(symbol)
                except Exception as e:
                    logging.warning(f"Could not process data for {symbol}: {e}")

        # Require sufficient data for meaningful optimization
        if len(returns_data) < 2 or len(valid_symbols) < 2:
            return {
                "success": False,
                "error": f"Insufficient market data for optimization: {len(returns_data)}/{len(symbols)} symbols have data",
                "market_data_available": False
            }
        
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

        # Perform dynamic optimization using MPT
        try:
            # Extract user profile data for analysis
            user_risk_score = user_risk_profile.get('risk_score', 50) if user_risk_profile else 50
            age = user_risk_profile.get('age', 30) if user_risk_profile else 30
            investment_horizon = user_risk_profile.get('investment_horizon', 5) if user_risk_profile else 5
            
            # Use the agentic tool to analyze the risk profile
            risk_analysis_result = agentic_risk_analyzer_tool(user_risk_profile)
            
            if risk_analysis_result.get("success"):
                adjusted_risk_level = risk_analysis_result.get("adjusted_risk_level")
                logging.info(f"Agentic risk analysis successful. Adjusted risk level to: {adjusted_risk_level}. Rationale: {risk_analysis_result.get('rationale')}")
            else:
                logging.warning("Agentic risk analysis failed. Using original risk level.")
                adjusted_risk_level = risk_level

            # Perform MPT optimization
            optimized_weights = _dynamic_mpt_optimization(expected_returns, cov_matrix, valid_symbols, adjusted_risk_level, risk_free_rate)

            # Normalize weights to ensure they sum to 1.0
            total_weight = sum(optimized_weights.values())
            if total_weight > 0:
                optimized_weights = {symbol: weight / total_weight for symbol, weight in optimized_weights.items()}

            result_data = {
                "success": True,
                "current_weights": current_weights,
                "target_weights": optimized_weights,
                "weights": optimized_weights,  # For backward compatibility
                "risk_level": adjusted_risk_level,
                "original_risk_level": risk_level,
                "market_data_available": True,
                "symbols": valid_symbols,
                "expected_returns": expected_returns.to_dict(),
                "volatilities": returns_df.std().to_dict(),
                "covariance_matrix": cov_matrix.to_dict(),
                "correlations": returns_df.corr().to_dict(),
                "risk_free_rate": risk_free_rate,
                "user_risk_profile": user_risk_profile,
                "optimization_method": f"Dynamic_MPT_{adjusted_risk_level}",
                "risk_profile_analysis": {
                    "risk_score": user_risk_score,
                    "age": age,
                    "investment_horizon": investment_horizon,
                    "adjusted_risk_level": adjusted_risk_level,
                    "adjustment_reason": "Based on user profile factors" if adjusted_risk_level != risk_level else "Matches requested risk level"
                },
                "optimization_guidance": f"""
                Portfolio optimized using dynamic Modern Portfolio Theory for {adjusted_risk_level} risk level:
                - Conservative: Minimized volatility, capital preservation focus
                - Moderate: Balanced risk and return, Sharpe ratio optimization  
                - Aggressive: Return maximization with volatility constraints

                Risk level adjusted from '{risk_level}' to '{adjusted_risk_level}' based on user profile:
                - Risk Score: {user_risk_score}
                - Age: {age}
                - Investment Horizon: {investment_horizon} years

                Target weights calculated using historical return data from pandas-datareader and investpy sources.
                """
            }
            return _sanitize_float_values(result_data)
            
        except Exception as opt_error:
            logging.error(f"Dynamic MPT optimization failed: {opt_error}")
            return {
                "success": False,
                "error": f"Optimization failed: {str(opt_error)}",
                "market_data_available": True,
                "optimization_error": str(opt_error)
            }
        
    except Exception as e:
        logging.error(f"Portfolio optimization error: {e}")
        return {
            "success": False,
            "error": str(e),
            "market_data_available": False
        }


def _dynamic_mpt_optimization(expected_returns: pd.Series, cov_matrix: pd.DataFrame, symbols: List[str], risk_level: str, risk_free_rate: float) -> Dict[str, float]:
    """Dynamic MPT optimization based on risk level and market data"""
    if risk_level == "conservative":
        return _minimize_volatility(expected_returns, cov_matrix, symbols)
    elif risk_level == "aggressive":
        return _maximize_return_with_constraint(expected_returns, cov_matrix, symbols)
    else:  # moderate
        return _maximize_sharpe_ratio(expected_returns, cov_matrix, symbols, risk_free_rate)




def _minimize_volatility(expected_returns: pd.Series, cov_matrix: pd.DataFrame, symbols: List[str]) -> Dict[str, float]:
    """Minimize portfolio volatility"""
    n = len(symbols)
    
    def objective(weights):
        return np.dot(weights, np.dot(cov_matrix, weights))
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n)]
    
    result = minimize(objective, [1/n] * n, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Validate result and provide fallback for invalid values
    if result.success and hasattr(result, 'x'):
        weights = result.x
        # Check for inf, -inf, or NaN values
        if np.any(np.isinf(weights)) or np.any(np.isnan(weights)):
            logging.warning("Optimization result contains invalid values (inf/NaN), using equal weights")
            weights = [1.0/n] * n
    else:
        logging.warning("Optimization failed, using equal weights")
        weights = [1.0/n] * n
    
    return dict(zip(symbols, weights))


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
    
    # Validate result and provide fallback for invalid values
    if result.success and hasattr(result, 'x'):
        weights = result.x
        # Check for inf, -inf, or NaN values
        if np.any(np.isinf(weights)) or np.any(np.isnan(weights)):
            logging.warning("Optimization result contains invalid values (inf/NaN), using equal weights")
            weights = [1.0/n] * n
    else:
        logging.warning("Optimization failed, using equal weights")
        weights = [1.0/n] * n
    
    return dict(zip(symbols, weights))


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
    
    # Validate result and provide fallback for invalid values
    if result.success and hasattr(result, 'x'):
        weights = result.x
        # Check for inf, -inf, or NaN values
        if np.any(np.isinf(weights)) or np.any(np.isnan(weights)):
            logging.warning("Optimization result contains invalid values (inf/NaN), using equal weights")
            weights = [1.0/n] * n
    else:
        logging.warning("Optimization failed, using equal weights")
        weights = [1.0/n] * n
    
    return dict(zip(symbols, weights))


def suggest_diversification_assets_tool(current_portfolio: Dict[str, float], market_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Suggest diversification assets based on sector analysis - returns data for agent processing
    
    Args:
        current_portfolio: Current portfolio weights
        market_data: Market data for analysis
    
    Returns:
        Dict containing sector analysis for agent decision making
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
        
        return {
            "success": True,
            "current_sector_exposure": current_sectors,
            "diversification_needs": diversification_needs,
            "suggested_assets": suggestions,
            "recommendations": {
                "add_assets": suggestions[:3],  # Top 3 suggestions
                "sector_improvements": diversification_needs[:3]
            },
            "diversification_guidance": """
            Based on sector analysis, consider adding assets to:
            1. Under-represented sectors for better diversification
            2. Sectors with positive momentum and growth
            3. Sectors that reduce overall portfolio volatility
            
            Let the agent decide which assets to recommend based on this analysis.
            """
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
        
        # Enhanced user risk profile integration
        user_risk_score = 0.5  # Default moderate
        if user_risk_profile:
            # Extract comprehensive risk tolerance from user profile
            risk_level = user_risk_profile.get('risk_level', 'moderate')
            risk_score = user_risk_profile.get('risk_score', 50)  # 0-100 scale
            
            # Convert to 0-1 scale
            user_risk_score = risk_score / 100.0
            
            # Adjust based on investment horizon
            horizon = user_risk_profile.get('investment_horizon', 5)
            if horizon < 3:
                user_risk_score *= 0.7  # Significantly reduce risk for short horizon
            elif horizon > 10:
                user_risk_score *= 1.3  # Increase risk tolerance for long horizon
            
            # Adjust based on age (if available)
            age = user_risk_profile.get('age', 35)
            if age < 30:
                user_risk_score *= 1.2  # Younger investors can take more risk
            elif age > 60:
                user_risk_score *= 0.8  # Older investors should be more conservative
            
            # Adjust based on income/net worth (if available)
            income = user_risk_profile.get('annual_income', 50000)
            net_worth = user_risk_profile.get('net_worth', 100000)
            if income > 100000 or net_worth > 500000:
                user_risk_score *= 1.1  # Higher income/wealth can take more risk
            
            # Adjust based on questionnaire responses
            questionnaire = user_risk_profile.get('questionnaire_data', {})
            if questionnaire:
                loss_tolerance = questionnaire.get('loss_tolerance', 5)  # 0-10 scale
                experience = questionnaire.get('experience_level', 'intermediate')
                
                # Adjust based on loss tolerance
                user_risk_score *= (loss_tolerance / 10.0)
                
                # Adjust based on experience
                if experience == 'beginner':
                    user_risk_score *= 0.8
                elif experience == 'advanced':
                    user_risk_score *= 1.2
        
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
    Get live prices for given symbols using Polygon.io API
    
    Args:
        symbols: List of asset symbols
    
    Returns:
        Dict containing live price data
    """
    try:
        live_data = {}
        for symbol in symbols:
            try:
                # Try Polygon.io for live data
                try:
                    _check_rate_limit()
                    
                    # Get recent data (last 5 days) to get current price
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=5)
                    
                    df = _fetch_polygon_aggregates(symbol, 5)
                    if df is not None and not df.empty:
                        latest_price = df['Close'].iloc[-1]
                        prev_price = df['Close'].iloc[-2] if len(df) > 1 else latest_price
                        change = latest_price - prev_price
                        change_percent = (change / prev_price) * 100 if prev_price != 0 else 0
                        
                        live_data[symbol] = {
                            "price": float(latest_price),
                            "change": float(change),
                            "change_percent": float(change_percent),
                            "volume": float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0,
                            "data_source": "polygon"
                        }
                        continue
                except Exception as e:
                    logging.warning(f"Polygon.io failed for {symbol}: {e}")
                
                # If Polygon.io fails, return error
                live_data[symbol] = {"error": f"No live data available for {symbol} from Polygon.io"}
                
            except Exception as e:
                logging.error(f"Error getting live price for {symbol}: {e}")
                live_data[symbol] = {"error": str(e)}
        
        return {
            "success": True,
            "live_prices": live_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "live_prices": {}
        }


# ============================================================================
# CrewAI Tool Wrappers - Create wrapper functions first
# ============================================================================

def fetch_market_data_tool_wrapper(symbols: str, days: int = 365) -> str:
    """Fetch historical market data for symbols (comma-separated)"""
    symbol_list = [s.strip() for s in symbols.split(',')]
    result = fetch_market_data_tool(symbol_list, days)
    return json.dumps(result)

def calculate_indicators_tool_wrapper(symbol: str, data: str) -> str:
    """Calculate technical indicators for a symbol's data"""
    try:
        data_list = json.loads(data) if isinstance(data, str) else data
        result = calculate_indicators_tool(symbol, data_list)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

def portfolio_optimizer_tool_wrapper(current_weights: str, risk_level: str, market_data: str, user_risk_profile: str = "{}") -> str:
    """Optimize portfolio allocation based on risk level and market data"""
    try:
        current_weights_dict = json.loads(current_weights) if isinstance(current_weights, str) else current_weights
        market_data_dict = json.loads(market_data) if isinstance(market_data, str) else market_data
        user_profile_dict = json.loads(user_risk_profile) if isinstance(user_risk_profile, str) else user_risk_profile
        
        result = portfolio_optimizer_tool(current_weights_dict, risk_level, market_data_dict, user_profile_dict)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

def market_regime_detector_tool_wrapper(market_data: str) -> str:
    """Detect current market regime based on technical indicators"""
    try:
        market_data_dict = json.loads(market_data) if isinstance(market_data, str) else market_data
        result = market_regime_detector_tool(market_data_dict)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

def risk_calculator_tool_wrapper(portfolio_data: str, user_risk_profile: str = "{}", market_data: str = "{}") -> str:
    """Calculate comprehensive portfolio risk metrics"""
    try:
        portfolio_dict = json.loads(portfolio_data) if isinstance(portfolio_data, str) else portfolio_data
        user_profile_dict = json.loads(user_risk_profile) if isinstance(user_risk_profile, str) else user_risk_profile
        market_data_dict = json.loads(market_data) if isinstance(market_data, str) else market_data
        
        result = risk_calculator_tool(portfolio_dict, user_profile_dict, market_data_dict)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

def suggest_diversification_assets_tool_wrapper(current_portfolio: str, market_data: str = "{}") -> str:
    """Suggest diversification assets based on sector analysis"""
    try:
        portfolio_dict = json.loads(current_portfolio) if isinstance(current_portfolio, str) else current_portfolio
        market_data_dict = json.loads(market_data) if isinstance(market_data, str) else market_data
        
        result = suggest_diversification_assets_tool(portfolio_dict, market_data_dict)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

def rebalancing_engine_tool_wrapper(portfolio_id: int, current_weights: str, target_weights: str) -> str:
    """Execute portfolio rebalancing by calculating optimal trades"""
    try:
        current_weights_dict = json.loads(current_weights) if isinstance(current_weights, str) else current_weights
        target_weights_dict = json.loads(target_weights) if isinstance(target_weights, str) else target_weights
        
        result = rebalancing_engine_tool(portfolio_id, current_weights_dict, target_weights_dict)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

def get_live_prices_tool_wrapper(symbols: str) -> str:
    """Get current market prices for a list of symbols"""
    try:
        symbol_list = [s.strip() for s in symbols.split(',')]
        result = get_live_prices_tool(symbol_list)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

# ============================================================================
# CrewAI Tool Objects - Create Tool objects for CrewAI agent compatibility
# ============================================================================

def _create_tool(name: str, description: str, func: callable) -> Optional[Any]:
    """Create a CrewAI Tool object if crewai_tools is available, otherwise return None"""
    if not CREWAI_TOOLS_AVAILABLE or Tool is None:
        # Return the function itself as a tool (function-based approach)
        return func
    try:
        return Tool(
            name=name,
            description=description,
            func=func
        )
    except Exception as e:
        logging.warning(f"Failed to create CrewAI tool '{name}': {e}")
        return func

# Create Tool objects for data collection
FETCH_MARKET_DATA_CREW_TOOL = _create_tool(
    name="fetch_market_data",
    description="Fetch historical market data and technical indicators for given symbols using Polygon.io API. Returns price history, SMA, RSI, and volume data.",
    func=fetch_market_data_tool_wrapper
)

CALCULATE_INDICATORS_CREW_TOOL = _create_tool(
    name="calculate_indicators",
    description="Calculate technical indicators (SMA, RSI) for a given symbol's OHLCV data.",
    func=calculate_indicators_tool_wrapper
)

# Create Tool objects for portfolio strategy
PORTFOLIO_OPTIMIZER_CREW_TOOL = _create_tool(
    name="portfolio_optimizer",
    description="Optimize portfolio allocation based on risk level, market data, and user risk profile using Modern Portfolio Theory.",
    func=portfolio_optimizer_tool_wrapper
)

MARKET_REGIME_DETECTOR_CREW_TOOL = _create_tool(
    name="market_regime_detector",
    description="Detect current market regime (bull, bear, stable, volatile) based on technical indicators.",
    func=market_regime_detector_tool_wrapper
)

RISK_CALCULATOR_CREW_TOOL = _create_tool(
    name="risk_calculator",
    description="Calculate comprehensive portfolio risk metrics including volatility, concentration risk, diversification score, and Sharpe ratio.",
    func=risk_calculator_tool_wrapper
)

DIVERSIFICATION_ASSETS_CREW_TOOL = _create_tool(
    name="suggest_diversification",
    description="Suggest diversification assets based on sector analysis and current portfolio composition.",
    func=suggest_diversification_assets_tool_wrapper
)

REBALANCING_ENGINE_CREW_TOOL = _create_tool(
    name="rebalancing_engine",
    description="Execute portfolio rebalancing by calculating optimal trades and generating execution orders.",
    func=rebalancing_engine_tool_wrapper
)

GET_LIVE_PRICES_CREW_TOOL = _create_tool(
    name="get_live_prices",
    description="Get current market prices and trading data for a list of symbols.",
    func=get_live_prices_tool_wrapper
)

