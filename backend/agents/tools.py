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

# Import requests for API calls
import requests
import time

# OpenAI integration
try:
    from openai import OpenAI
    from backend.config import get_settings
    settings = get_settings()
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
except ImportError:
    openai_client = None
    logging.warning("OpenAI not available - falling back to rule-based logic")

# CrewAI LLM setup for agents
try:
    from crewai import LLM
    if settings.OPENAI_API_KEY:
        # Use a single LLM configuration for simplicity
        crewai_llm = LLM(
            model="gpt-5-nano",  
            api_key=settings.OPENAI_API_KEY,
            temperature=0.1
        )
        data_agent_llm = crewai_llm  # gpt-5-nano for data analysis tasks
        strategy_agent_llm = crewai_llm  # gpt-5-nano for strategy optimization
        validation_agent_llm = crewai_llm  # gpt-5-nano for risk validation

        orchestrator_llm = LLM(
            model="gpt-5",  # gpt-5 for orchestrator coordination
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2
        )

        CREWAI_AVAILABLE = True
        logging.info("CrewAI initialized with gpt-5-nano for agents and gpt-5 for orchestrator")
    else:
        crewai_llm = data_agent_llm = strategy_agent_llm = validation_agent_llm = orchestrator_llm = None
        CREWAI_AVAILABLE = False
        logging.warning("OpenAI API key not available - CrewAI agents will be disabled")
except ImportError:
    crewai_llm = data_agent_llm = strategy_agent_llm = validation_agent_llm = orchestrator_llm = None
    CREWAI_AVAILABLE = False
    logging.warning("CrewAI not available")

# CrewAI Tool wrappers - convert functions to Tool objects for CrewAI compatibility
try:
    from crewai_tools import Tool
    CREWAI_TOOLS_AVAILABLE = True
except ImportError:
    CREWAI_TOOLS_AVAILABLE = False
    logging.warning("crewai_tools not available - tools will be passed as functions")
    Tool = None

from .utils import _sanitize_float_values


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


def fetch_market_data_tool(symbols: List[str], days: int = 365) -> Dict[str, Any]:
    """
    Fetches historical market data for given symbols using multiple sources:
    1. yfinance (primary - free and reliable)
    2. Alpha Vantage API (fallback if configured)

    Args:
        symbols: List of asset symbols (e.g., ['AAPL', 'GOOGL'])
        days: Number of days of historical data

    Returns:
        Dict containing market data for each symbol
    """
    
    def _normalize_symbol(symbol: str) -> str:
        """Normalize symbol for different data sources"""
        # Handle special cases like BRK.B -> BRK-B for some APIs
        symbol = symbol.strip().upper()
        return symbol
    def _try_alpha_vantage_first(symbol: str, api_key: str) -> Optional[pd.DataFrame]:
        """Try to fetch data from Alpha Vantage first (more reliable)"""
        try:
            import requests
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': api_key,
                'outputsize': 'full'  # Get more data points
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            if 'Time Series (Daily)' in data:
                # Convert to DataFrame
                time_series = data['Time Series (Daily)']
                df = pd.DataFrame([
                    {
                        'Date': date,
                        'Close': float(values['4. close']),
                        'Volume': float(values['5. volume'])
                    }
                    for date, values in time_series.items()
                ])
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                return df
            else:
                logging.warning(f"Alpha Vantage error for {symbol}: {data.get('Error Message', 'Unknown error')}")
                return None

        except ImportError:
            logging.warning("requests module not available for Alpha Vantage API")
            return None
        except Exception as e:
            logging.error(f"Alpha Vantage failed for {symbol}: {e}")
            return None

    def _try_yahoo_finance_direct(symbol: str) -> Optional[pd.DataFrame]:
        """Try to fetch data directly from Yahoo Finance API (free)"""
        try:
            import requests
            import json
            
            # Yahoo Finance API endpoint
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'range': '2y',
                'interval': '1d',
                'includePrePost': 'true',
                'events': 'div,split'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]
                
                df = pd.DataFrame({
                    'Date': pd.to_datetime(timestamps, unit='s'),
                    'Close': quotes['close'],
                    'Volume': quotes['volume']
                })
                
                # Remove NaN values
                df = df.dropna()
                
                if len(df) > 10:
                    return df
                    
        except Exception as e:
            logging.warning(f"Yahoo Finance direct API failed for {symbol}: {e}")
            
        return None

    def _try_yfinance_fallback(symbol: str, max_retries: int = 2) -> Optional[pd.DataFrame]:
        """Try to fetch data from yfinance as fallback"""
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                # Try multiple period formats
                for period in ["1y", "2y", "5y"]:
                    df = ticker.history(period=period)
                    if not df.empty and len(df) > 10:
                        return df
                time.sleep(1)  # Wait between attempts
            except Exception as e:
                logging.warning(f"yfinance attempt {attempt + 1} failed for {symbol}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        return None

    def _try_alpha_vantage(symbol: str, api_key: str) -> Optional[pd.DataFrame]:
        """Try to fetch data from Alpha Vantage"""
        try:
            import requests
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': api_key,
                'outputsize': 'compact'  # Last 100 data points
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'Time Series (Daily)' in data:
                # Convert to DataFrame
                time_series = data['Time Series (Daily)']
                df = pd.DataFrame([
                    {
                        'Date': date,
                        'Close': float(values['4. close']),
                        'Volume': float(values['5. volume'])
                    }
                    for date, values in time_series.items()
                ])
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                return df
            else:
                logging.warning(f"Alpha Vantage error for {symbol}: {data.get('Error Message', 'Unknown error')}")
                return None

        except ImportError:
            logging.warning("requests module not available for Alpha Vantage API")
            return None
        except Exception as e:
            logging.error(f"Alpha Vantage failed for {symbol}: {e}")
            return None

    def _get_news_sentiment(symbol: str, api_key: str) -> float:
        """Get news sentiment for a symbol"""
        try:
            import requests
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{symbol} stock',
                'apiKey': api_key,
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 5
            }

            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()

            data = response.json()

            if data.get('status') == 'ok' and data.get('articles'):
                # Simple sentiment analysis based on article titles
                positive_words = ['up', 'rise', 'gain', 'profit', 'bullish', 'strong', 'growth', 'beat']
                negative_words = ['down', 'fall', 'loss', 'bearish', 'weak', 'decline', 'drop', 'sell']

                sentiment_score = 0
                for article in data['articles'][:3]:  # Check first 3 articles
                    title = article.get('title', '').lower()
                    pos_count = sum(1 for word in positive_words if word in title)
                    neg_count = sum(1 for word in negative_words if word in title)
                    sentiment_score += (pos_count - neg_count)

                return max(-1.0, min(1.0, sentiment_score / 3))  # Normalize to [-1, 1]
            return 0.0

        except ImportError:
            logging.warning("requests module not available for News API")
            return 0.0
        except Exception as e:
            logging.warning(f"News API failed for {symbol}: {e}")
            return 0.0

    try:
        data = {}
        successful_fetches = 0
        alpha_vantage_key = getattr(settings, 'ALPHA_VANTAGE_API_KEY', None)
        news_api_key = getattr(settings, 'NEWS_API_KEY', None)

        logging.info(f"Fetching market data for {len(symbols)} symbols using multiple sources")

        for symbol in symbols:
            symbol_data = {}
            df = None

            try:
                # 1. Try yfinance library first (free and reliable)
                logging.info(f"Trying yfinance library for {symbol}")
                df = _try_yfinance_fallback(symbol)

                # 2. If yfinance fails, try Yahoo Finance direct API
                if df is None:
                    logging.info(f"Trying Yahoo Finance direct API for {symbol}")
                    df = _try_yahoo_finance_direct(symbol)

                # 3. If all fail, try alternative symbol formats (e.g., BRK.B <-> BRK-B)
                if df is None:
                    alt_symbols = []
                    if '.' in symbol:
                        alt_symbols.append(symbol.replace('.', '-'))  # e.g., BRK.B -> BRK-B
                    elif '-' in symbol:
                        alt_symbols.append(symbol.replace('-', '.'))  # e.g., BRK-B -> BRK.B

                    for alt_symbol in alt_symbols:
                        logging.info(f"Trying alternative symbol {alt_symbol} for {symbol}")
                        df = _try_yfinance_fallback(alt_symbol)
                        if df is None:
                            df = _try_yahoo_finance_direct(alt_symbol)
                        if df is not None:
                            logging.info(f"Success with alternative symbol {alt_symbol}")
                            break
                
                # 4. Only try Alpha Vantage as last resort if API key is available
                if df is None and alpha_vantage_key:
                    logging.info(f"Trying Alpha Vantage as last resort for {symbol}")
                    df = _try_alpha_vantage_first(symbol, alpha_vantage_key)

                # 5. If still no data and we have News API key, try sentiment analysis
                if df is None and news_api_key:
                    logging.info(f"Trying News API sentiment for {symbol}")
                    sentiment = _get_news_sentiment(symbol, news_api_key)
                    if sentiment != 0.0:
                        # Create synthetic data point based on sentiment
                        symbol_data = {
                            'historical_data': [{'Close': 100.0, 'Date': datetime.now().strftime('%Y-%m-%d')}],
                            'current_price': 100.0,
                            'sma_20': 100.0,
                            'sma_50': 100.0,
                            'rsi': 50.0 + (sentiment * 20),  # Sentiment affects RSI
                            'volume': 1000000,
                            'data_points': 1,
                            'data_source': 'news_sentiment',
                            'news_sentiment': sentiment
                        }
                        data[symbol] = symbol_data
                        successful_fetches += 1
                        logging.info(f"Successfully created synthetic data for {symbol} based on sentiment")
                        continue

                if df is not None and not df.empty:
                    # Calculate basic technical indicators
                    df['SMA_20'] = df['Close'].rolling(window=min(20, len(df))).mean()
                    df['SMA_50'] = df['Close'].rolling(window=min(50, len(df))).mean()
                    df['RSI'] = calculate_rsi(df['Close'])

                    # Ensure we have at least some data for returns calculation
                    if len(df) >= 10:  # Reduced requirement for Alpha Vantage data
                        symbol_data = {
                            'historical_data': df.to_dict('records'),
                            'current_price': float(df['Close'].iloc[-1]),
                            'sma_20': float(df['SMA_20'].iloc[-1]) if not pd.isna(df['SMA_20'].iloc[-1]) else float(df['Close'].iloc[-1]),
                            'sma_50': float(df['SMA_50'].iloc[-1]) if not pd.isna(df['SMA_50'].iloc[-1]) else float(df['Close'].iloc[-1]),
                            'rsi': float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0,
                            'volume': float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0.0,
                            'data_points': len(df),
                            'data_source': 'yfinance' if df is not None and 'Date' in df.columns else 'alpha_vantage'
                        }

                        # Add news sentiment if available
                        if news_api_key:
                            sentiment = _get_news_sentiment(symbol, news_api_key)
                            symbol_data['news_sentiment'] = sentiment

                        data[symbol] = symbol_data
                        successful_fetches += 1
                        logging.info(f"Successfully fetched data for {symbol} ({len(df)} data points)")
                    else:
                        data[symbol] = {'error': f'Insufficient data: only {len(df)} data points'}
                else:
                    data[symbol] = {'error': f'No data available for {symbol} after trying multiple sources'}

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
                "yfinance_success": sum(1 for s in data.values() if isinstance(s, dict) and s.get('data_source') == 'yfinance'),
                "alpha_vantage_success": sum(1 for s in data.values() if isinstance(s, dict) and s.get('data_source') == 'alpha_vantage'),
                "news_api_success": sum(1 for s in data.values() if isinstance(s, dict) and 'news_sentiment' in s)
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
    Portfolio optimization data for CrewAI agents to process
    Returns market metrics and optimization context for LLM-driven decision making
    
    Args:
        current_weights: Dict with symbol -> current weight
        risk_level: Risk level ('conservative', 'moderate', 'aggressive')
        market_data: Historical market data for optimization
        user_risk_profile: User's risk profile for personalization
    
    Returns:
        Dict containing market metrics for agent decision making
    """
    try:
        symbols = list(current_weights.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {"success": False, "error": "No assets to optimize"}
        
        # If no market data, return current weights with simple guidance
        if not market_data:
            # Use agentic tool to adjust risk level even when no market data
            adjusted_risk_level = risk_level  # Default
            risk_rationale = "No adjustment made"
            
            if user_risk_profile:
                try:
                    risk_analysis_result = agentic_risk_analyzer_tool(user_risk_profile)
                    if risk_analysis_result.get("success"):
                        adjusted_risk_level = risk_analysis_result.get("adjusted_risk_level")
                        risk_rationale = risk_analysis_result.get("rationale", "Adjusted by agentic analysis")
                        logging.info(f"Agentic risk analysis successful (no market data). Adjusted risk level to: {adjusted_risk_level}. Rationale: {risk_rationale}")
                except Exception as e:
                    logging.warning(f"Agentic risk analysis failed: {e}")
            
            # Simple risk-based allocation when no market data available
            if adjusted_risk_level == "conservative":
                # Favor first asset (assumed more stable) and equal weight others
                adjusted_weights = {symbol: 0.1 for symbol in symbols}
                if symbols:
                    adjusted_weights[symbols[0]] = 0.6  # 60% in first asset
                total = sum(adjusted_weights.values())
                adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
            elif adjusted_risk_level == "aggressive":
                # More equal distribution for diversification
                adjusted_weights = {symbol: 1.0/n_assets for symbol in symbols}
            else:  # moderate
                adjusted_weights = {symbol: 1.0/n_assets for symbol in symbols}

            return _sanitize_float_values({
                "success": True,
                "current_weights": current_weights,
                "target_weights": adjusted_weights,
                "weights": adjusted_weights,  # For backward compatibility
                "risk_level": adjusted_risk_level,
                "original_risk_level": risk_level,
                "market_data_available": False,
                "optimization_method": f"simple_{adjusted_risk_level}",
                "risk_profile_analysis": {
                    "adjusted_risk_level": adjusted_risk_level,
                    "adjustment_reason": risk_rationale
                },
                "guidance": f"Insufficient market data. Used simple risk-based allocation for {adjusted_risk_level} profile."
            })
        
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

        # Check if we have enough data for meaningful optimization
        if len(returns_data) >= 2 and len(valid_symbols) >= 2:
            # We have sufficient data for optimization
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

            # Perform the actual optimization and return target weights
            # Use fallback MPT optimization functions with enhanced risk profile integration
            try:
                # Extract user profile data for logging/display
                user_risk_score = user_risk_profile.get('risk_score', 50) if user_risk_profile else 50
                age = user_risk_profile.get('age', 30) if user_risk_profile else 30
                investment_horizon = user_risk_profile.get('investment_horizon', 5) if user_risk_profile else 5
                
                # Use the agentic tool to analyze the risk profile
                risk_analysis_result = agentic_risk_analyzer_tool(user_risk_profile)
                
                if risk_analysis_result.get("success"):
                    adjusted_risk_level = risk_analysis_result.get("adjusted_risk_level")
                    logging.info(f"Agentic risk analysis successful. Adjusted risk level to: {adjusted_risk_level}. Rationale: {risk_analysis_result.get('rationale')}")
                else:
                    logging.warning("Agentic risk analysis failed. Falling back to original risk level.")
                    adjusted_risk_level = risk_level

                optimized_weights = _fallback_mpt_optimization(expected_returns, cov_matrix, valid_symbols, adjusted_risk_level, risk_free_rate)

                # Normalize weights to ensure they sum to 1.0
                total_weight = sum(optimized_weights.values())
                if total_weight > 0:
                    optimized_weights = {symbol: weight / total_weight for symbol, weight in optimized_weights.items()}

                result_data = {
                    "success": True,
                    "current_weights": current_weights,
                    "target_weights": optimized_weights,
                    "weights": optimized_weights,  # For backward compatibility with strategy agent
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
                    "optimization_method": f"MPT_{adjusted_risk_level}",
                    "risk_profile_analysis": {
                        "risk_score": user_risk_score,
                        "age": age,
                        "investment_horizon": investment_horizon,
                        "adjusted_risk_level": adjusted_risk_level,
                        "adjustment_reason": "Based on user profile factors" if adjusted_risk_level != risk_level else "Matches requested risk level"
                    },
                    "optimization_guidance": f"""
                    Portfolio optimized using Modern Portfolio Theory for {adjusted_risk_level} risk level:
                    - Conservative: Minimized volatility, capital preservation focus
                    - Moderate: Balanced risk and return, Sharpe ratio optimization
                    - Aggressive: Return maximization with volatility constraints

                    Risk level adjusted from '{risk_level}' to '{adjusted_risk_level}' based on user profile:
                    - Risk Score: {user_risk_score}
                    - Age: {age}
                    - Investment Horizon: {investment_horizon} years

                    Target weights calculated using historical return data and user-specific risk factors.
                    """
                }
                return _sanitize_float_values(result_data)
            except Exception as opt_error:
                logging.warning(f"MPT optimization failed, using simple allocation: {opt_error}")
                # Fallback to simple equal-weight allocation
                equal_weights = {symbol: 1.0 / n_assets for symbol in symbols}
                fallback_data = {
                    "success": True,
                    "current_weights": current_weights,
                    "target_weights": equal_weights,
                    "weights": equal_weights,
                    "risk_level": adjusted_risk_level,
                    "original_risk_level": risk_level,
                    "market_data_available": False,
                    "optimization_method": "equal_weight_fallback",
                    "optimization_error": str(opt_error)
                }
                return _sanitize_float_values(fallback_data)
        else:
            # Insufficient data for optimization - use fallback allocation
            logging.warning(f"Insufficient market data for optimization: {len(returns_data)}/{len(symbols)} symbols have data")

            # Use agentic tool to adjust risk level even for simple allocation
            risk_analysis_result = agentic_risk_analyzer_tool(user_risk_profile)
            if risk_analysis_result.get("success"):
                adjusted_risk_level = risk_analysis_result.get("adjusted_risk_level")
                logging.info(f"Agentic risk analysis successful for fallback. Adjusted risk level to: {adjusted_risk_level}. Rationale: {risk_analysis_result.get('rationale')}")
            else:
                logging.warning("Agentic risk analysis failed for fallback. Falling back to original risk level.")
                adjusted_risk_level = risk_level
            
            if adjusted_risk_level == "conservative":
                # Favor first asset (assumed more stable) and equal weight others
                adjusted_weights = {symbol: 0.1 for symbol in symbols}
                if symbols:
                    adjusted_weights[symbols[0]] = 0.6  # 60% in first asset
                total = sum(adjusted_weights.values())
                adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
            elif adjusted_risk_level == "aggressive":
                # More equal distribution for diversification
                adjusted_weights = {symbol: 1.0/n_assets for symbol in symbols}
            else:  # moderate
                adjusted_weights = {symbol: 1.0/n_assets for symbol in symbols}

            return {
                "success": True,
                "current_weights": current_weights,
                "target_weights": adjusted_weights,
                "weights": adjusted_weights,  # For backward compatibility
                "risk_level": adjusted_risk_level,
                "original_risk_level": risk_level,
                "market_data_available": False,
                "optimization_method": f"simple_{adjusted_risk_level}_insufficient_data",
                "risk_profile_analysis": {
                    "adjusted_risk_level": adjusted_risk_level,
                    "adjustment_reason": "Based on user profile factors" if adjusted_risk_level != risk_level else "Matches requested risk level"
                },
                "guidance": f"Insufficient return data for optimization. Used simple allocation for {adjusted_risk_level} profile."
            }
        
    except Exception as e:
        logging.error(f"Portfolio optimization error: {e}")
        # Fallback to simple equal weight allocation on error
        equal_weights = {symbol: 1.0/n_assets for symbol in symbols} if n_assets > 0 else {}

        # Adjust optimization based on user risk profile for error fallback too
        adjusted_risk_level = risk_level
        if user_risk_profile:
            # Extract comprehensive risk factors from user profile
            user_risk_score = user_risk_profile.get('risk_score', 50)  # 0-100 scale
            age = user_risk_profile.get('age', 35)
            investment_horizon = user_risk_profile.get('investment_horizon', 5)

            # Adjust risk level based on user profile
            if user_risk_score < 30 or age > 65 or investment_horizon < 3:
                adjusted_risk_level = "conservative"
            elif user_risk_score > 70 and age < 40 and investment_horizon > 10:
                adjusted_risk_level = "aggressive"

        error_fallback_data = {
            "success": True,  # Return success with fallback weights
            "current_weights": current_weights,
            "target_weights": equal_weights,
            "weights": equal_weights,  # For backward compatibility
            "risk_level": adjusted_risk_level,
            "original_risk_level": risk_level,
            "market_data_available": False,
            "optimization_method": "equal_weight_error_fallback",
            "risk_profile_analysis": {
                "adjusted_risk_level": adjusted_risk_level,
                "adjustment_reason": "Error fallback - using original risk level"
            },
            "error": str(e),
            "guidance": f"Optimization failed. Using equal weight allocation as fallback for {adjusted_risk_level} profile."
        }
        return _sanitize_float_values(error_fallback_data)


def _fallback_mpt_optimization(expected_returns: pd.Series, cov_matrix: pd.DataFrame, symbols: List[str], risk_level: str, risk_free_rate: float) -> Dict[str, float]:
    """Fallback MPT optimization when LLM is unavailable"""
    if risk_level == "conservative":
        return _minimize_volatility(expected_returns, cov_matrix, symbols)
    elif risk_level == "aggressive":
        return _maximize_return_with_constraint(expected_returns, cov_matrix, symbols)
    else:  # moderate
        return _maximize_sharpe_ratio(expected_returns, cov_matrix, symbols, risk_free_rate)


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


# ============================================================================
# CrewAI Tool Wrappers - Create Tool objects for CrewAI agent compatibility
# ============================================================================

def _create_tool(name: str, description: str, func: callable) -> Optional[Any]:
    """Create a CrewAI Tool object if crewai_tools is available, otherwise return None"""
    if not CREWAI_TOOLS_AVAILABLE or Tool is None:
        return None
    try:
        return Tool(
            name=name,
            description=description,
            func=func
        )
    except Exception as e:
        logging.warning(f"Failed to create CrewAI tool '{name}': {e}")
        return None


# Create Tool objects for data collection (with error handling)
try:
    FETCH_MARKET_DATA_CREW_TOOL = _create_tool(
        name="fetch_market_data",
        description="Fetch historical market data and technical indicators for given symbols using yfinance. Returns price history, SMA, RSI, and volume data.",
        func=fetch_market_data_tool
    )
except Exception as e:
    logging.warning(f"Failed to create FETCH_MARKET_DATA_CREW_TOOL: {e}")
    FETCH_MARKET_DATA_CREW_TOOL = None

try:
    CALCULATE_INDICATORS_CREW_TOOL = _create_tool(
        name="calculate_indicators",
        description="Calculate technical indicators (SMA, RSI) for a given symbol's OHLCV data.",
        func=calculate_indicators_tool
    )
except Exception as e:
    logging.warning(f"Failed to create CALCULATE_INDICATORS_CREW_TOOL: {e}")
    CALCULATE_INDICATORS_CREW_TOOL = None

# Create Tool objects for portfolio strategy (with error handling)
try:
    PORTFOLIO_OPTIMIZER_CREW_TOOL = _create_tool(
        name="portfolio_optimizer",
        description="Optimize portfolio allocation based on risk level, market data, and user risk profile using Modern Portfolio Theory.",
        func=portfolio_optimizer_tool
    )
except Exception as e:
    logging.warning(f"Failed to create PORTFOLIO_OPTIMIZER_CREW_TOOL: {e}")
    PORTFOLIO_OPTIMIZER_CREW_TOOL = None

try:
    MARKET_REGIME_DETECTOR_CREW_TOOL = _create_tool(
        name="market_regime_detector",
        description="Detect current market regime (bull, bear, stable, volatile) based on technical indicators.",
        func=market_regime_detector_tool
    )
except Exception as e:
    logging.warning(f"Failed to create MARKET_REGIME_DETECTOR_CREW_TOOL: {e}")
    MARKET_REGIME_DETECTOR_CREW_TOOL = None

try:
    RISK_CALCULATOR_CREW_TOOL = _create_tool(
        name="risk_calculator",
        description="Calculate comprehensive portfolio risk metrics including volatility, concentration risk, diversification score, and Sharpe ratio.",
        func=risk_calculator_tool
    )
except Exception as e:
    logging.warning(f"Failed to create RISK_CALCULATOR_CREW_TOOL: {e}")
    RISK_CALCULATOR_CREW_TOOL = None

try:
    DIVERSIFICATION_ASSETS_CREW_TOOL = _create_tool(
        name="suggest_diversification",
        description="Suggest diversification assets based on sector analysis and current portfolio composition.",
        func=suggest_diversification_assets_tool
    )
except Exception as e:
    logging.warning(f"Failed to create DIVERSIFICATION_ASSETS_CREW_TOOL: {e}")
    DIVERSIFICATION_ASSETS_CREW_TOOL = None

try:
    REBALANCING_ENGINE_CREW_TOOL = _create_tool(
        name="rebalancing_engine",
        description="Execute portfolio rebalancing by calculating optimal trades and generating execution orders.",
        func=rebalancing_engine_tool
    )
except Exception as e:
    logging.warning(f"Failed to create REBALANCING_ENGINE_CREW_TOOL: {e}")
    REBALANCING_ENGINE_CREW_TOOL = None

try:
    GET_LIVE_PRICES_CREW_TOOL = _create_tool(
        name="get_live_prices",
        description="Get current market prices and trading data for a list of symbols.",
        func=get_live_prices_tool
    )
except Exception as e:
    logging.warning(f"Failed to create GET_LIVE_PRICES_CREW_TOOL: {e}")
    GET_LIVE_PRICES_CREW_TOOL = None
