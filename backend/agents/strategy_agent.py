"""Strategy Agent - Portfolio Strategy Advisor
Implements CrewAI Agent for intelligent strategy generation"""

from typing import Dict, Any, List
import logging
import pandas as pd
import numpy as np
from crewai import Agent, Task
from .base_agent import BaseAgent
from .tools import (
    portfolio_optimizer_tool,
    market_regime_detector_tool,
    risk_calculator_tool,
    suggest_diversification_assets_tool,
    PORTFOLIO_OPTIMIZER_CREW_TOOL,
    MARKET_REGIME_DETECTOR_CREW_TOOL,
    RISK_CALCULATOR_CREW_TOOL,
    DIVERSIFICATION_ASSETS_CREW_TOOL
)

logger = logging.getLogger(__name__)


# CrewAI Agent Definition
def create_strategy_agent(llm=None) -> Agent:
    """Create Strategy Agent using CrewAI framework"""
    from .tools import (
        PORTFOLIO_OPTIMIZER_CREW_TOOL,
        MARKET_REGIME_DETECTOR_CREW_TOOL,
        portfolio_optimizer_tool,
        market_regime_detector_tool,
        crewai_llm
    )

    # Use crew LLM if available, otherwise use provided llm
    if crewai_llm and llm is None:
        llm = crewai_llm

    # Check if we have valid CrewAI tools (not functions)
    tools_list = []
    if PORTFOLIO_OPTIMIZER_CREW_TOOL is not None and not callable(PORTFOLIO_OPTIMIZER_CREW_TOOL):
        tools_list.append(PORTFOLIO_OPTIMIZER_CREW_TOOL)
    if MARKET_REGIME_DETECTOR_CREW_TOOL is not None and not callable(MARKET_REGIME_DETECTOR_CREW_TOOL):
        tools_list.append(MARKET_REGIME_DETECTOR_CREW_TOOL)
    if RISK_CALCULATOR_CREW_TOOL is not None and not callable(RISK_CALCULATOR_CREW_TOOL):
        tools_list.append(RISK_CALCULATOR_CREW_TOOL)
    if DIVERSIFICATION_ASSETS_CREW_TOOL is not None and not callable(DIVERSIFICATION_ASSETS_CREW_TOOL):
        tools_list.append(DIVERSIFICATION_ASSETS_CREW_TOOL)

    return Agent(
        role='Chief Portfolio Strategist',
        goal='Develop optimal asset allocation strategies using advanced portfolio optimization techniques',
        backstory="""You are a Chief Portfolio Strategist with deep expertise in portfolio optimization and asset allocation strategy.

        Your expertise is portfolio optimization and asset allocation strategy. You excel at:
        - Modern Portfolio Theory (MPT) and mean-variance optimization
        - Risk-adjusted return maximization and volatility minimization
        - Market regime detection and adaptive strategy adjustments
        - Multi-asset class allocation and correlation analysis

        You are strategic, analytical, and always optimize for the best risk-adjusted returns within given constraints.
        - Creating optimized portfolios that balance risk and return using advanced quantitative methods
        - Tailoring strategies to individual investor profiles, considering age, risk tolerance, and investment horizon
        - Applying behavioral finance principles to improve long-term outcomes

        Your recommendations are always grounded in sound financial principles:
        - You prioritize diversification across asset classes and sectors
        - You consider transaction costs and tax implications
        - You provide clear rationale for each recommendation
        - You adapt strategies based on changing market conditions
        - You focus on risk-adjusted returns rather than absolute performance

        You have a proven track record of helping clients achieve their financial goals while managing downside risk effectively.""",
        tools=tools_list,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


class StrategyAgent(BaseAgent):
    """Portfolio Strategy Advisor Agent"""
    
    def __init__(self, db_session=None):
        super().__init__("StrategyAgent", db_session)
        self.role = "Portfolio Strategy Advisor"
        self.goal = "Develop optimal rebalancing strategies based on market conditions and risk profiles"
    
    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategy analysis and recommendation tasks"""
        try:
            self.log_action("execute_strategy_task", task_input)
            
            # Validate required fields
            required_fields = ["portfolio_id", "market_data", "risk_profile"]
            if not self.validate_input(required_fields, task_input):
                return {"success": False, "error": "Missing required fields"}
            
            portfolio_id = task_input["portfolio_id"]
            market_data = task_input["market_data"]
            risk_profile = task_input["risk_profile"]
            current_weights = task_input.get("current_weights", {})
            
            # Detect market regime
            regime_analysis = market_regime_detector_tool(market_data)
            market_regime = regime_analysis.get("regime", "unknown")
            
            # Calculate risk metrics with enhanced user profile integration
            holdings_list = []
            for symbol, weight in current_weights.items():
                holdings_list.append({
                    "symbol": symbol,
                    "weight": weight,
                    "current_value": weight * 100000,  # Estimate based on weight
                    "unrealized_gain_loss": 0  # Default for now
                })
            
            # Enhanced risk calculation with user profile and market data
            risk_metrics = risk_calculator_tool(
                portfolio_data={"holdings": holdings_list},
                user_risk_profile=risk_profile,
                market_data=market_data
            )
            
            # Optimize portfolio with MPT and market data
            risk_level = risk_profile.get("level", "moderate")
            if not isinstance(risk_level, str):
                risk_level = str(risk_level) if risk_level is not None else "moderate"
            
            optimization_results = portfolio_optimizer_tool(
                current_weights=current_weights,
                risk_level=risk_level,
                market_data=market_data,
                user_risk_profile=risk_profile
            )
            
            # Get diversification suggestions
            diversification_results = suggest_diversification_assets_tool(
                current_portfolio=current_weights,
                market_data=market_data
            )
            
            # Get target weights from optimization results (note: key can be "weights" or "target_weights")
            target_weights = optimization_results.get("target_weights", optimization_results.get("weights", {}))
            
            # Log optimization results for debugging
            logging.info(f"Optimization results: success={optimization_results.get('success')}, "
                        f"market_data_available={optimization_results.get('market_data_available')}, "
                        f"target_weights={list(target_weights.keys()) if target_weights else 'empty'}, "
                        f"error={optimization_results.get('error', 'none')}")
            
            # If optimization failed, return error with details (DO NOT use fallback)
            if not target_weights or not optimization_results.get("success"):
                error_msg = optimization_results.get("error", "Unknown optimization error")
                processing_errors = optimization_results.get("processing_errors", {})
                logging.error(f"Portfolio optimization failed: {error_msg}")
                logging.error(f"Processing errors: {processing_errors}")
                
                return {
                    "success": False,
                    "error": f"Portfolio optimization failed: {error_msg}",
                    "optimization_results": optimization_results,
                    "processing_errors": processing_errors,
                    "symbols_requested": list(current_weights.keys()),
                    "market_data_status": "Market data fetch may have failed or data format invalid"
                }
            
            # Generate rebalancing actions based on current vs target weights
            actions = []
            if target_weights:
                for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
                    current_weight = current_weights.get(symbol, 0)
                    target_weight = target_weights.get(symbol, 0)
                    weight_diff = target_weight - current_weight
                    
                    if abs(weight_diff) > 0.01:  # Only if difference is > 1%
                        action = "buy" if weight_diff > 0 else "sell"
                        actions.append({
                            "symbol": symbol,
                            "action": action,
                            "weight_change": weight_diff,
                            "current_weight": current_weight,
                            "target_weight": target_weight
                        })
            
            # Calculate expected improvements based on current vs target allocation
            expected_improvements = {}
            if current_weights and target_weights:
                try:
                    # Calculate portfolio metrics for current allocation
                    current_total = sum(current_weights.values())
                    if current_total > 0:
                        current_normalized = {k: v/current_total for k, v in current_weights.items()}
                    else:
                        current_normalized = current_weights

                    # Get market data for expected return calculations
                    market_data = task_input.get("market_data", {})
                    if market_data:
                        # Calculate expected returns based on historical data
                        portfolio_returns = []
                        portfolio_volatility = 0

                        for symbol, weight in target_weights.items():
                            if symbol in market_data and 'historical_data' in market_data[symbol]:
                                hist_data = market_data[symbol]['historical_data']
                                if hist_data:
                                    df = pd.DataFrame(hist_data)
                                    if 'Close' in df.columns:
                                        returns = df['Close'].pct_change().dropna()
                                        if not returns.empty:
                                            expected_return = returns.mean() * 252  # Annualized
                                            expected_vol = returns.std() * np.sqrt(252)  # Annualized

                                            portfolio_returns.append(expected_return * weight)
                                            portfolio_volatility += (expected_vol ** 2) * (weight ** 2)

                                            # Add cross-asset correlations for diversification effect
                                            for other_symbol, other_weight in target_weights.items():
                                                if symbol != other_symbol and other_symbol in market_data:
                                                    # Simplified correlation impact
                                                    correlation = 0.3  # Assume moderate correlation
                                                    portfolio_volatility += 2 * weight * other_weight * expected_vol * correlation * (returns.std() * np.sqrt(252) if other_symbol in market_data and 'historical_data' in market_data[other_symbol] else expected_vol)

                        if portfolio_returns:
                            expected_portfolio_return = sum(portfolio_returns)
                            expected_portfolio_volatility = np.sqrt(portfolio_volatility)

                            # Calculate Sharpe ratio (assuming 2% risk-free rate)
                            risk_free_rate = 0.02
                            sharpe_ratio = (expected_portfolio_return - risk_free_rate) / expected_portfolio_volatility if expected_portfolio_volatility > 0 else 0

                            # Estimate current portfolio metrics for comparison
                            current_portfolio_return = sum(returns.mean() * 252 * current_normalized.get(symbol, 0)
                                                         for symbol in current_normalized
                                                         if symbol in market_data and 'historical_data' in market_data.get(symbol, {}))

                            current_portfolio_volatility = 0
                            for symbol, weight in current_normalized.items():
                                if symbol in market_data and 'historical_data' in market_data[symbol]:
                                    returns = pd.DataFrame(market_data[symbol]['historical_data'])['Close'].pct_change().dropna()
                                    if not returns.empty:
                                        vol = returns.std() * np.sqrt(252)
                                        current_portfolio_volatility += (vol ** 2) * (weight ** 2)

                            current_portfolio_volatility = np.sqrt(current_portfolio_volatility)

                            # Calculate improvements
                            return_improvement = expected_portfolio_return - current_portfolio_return
                            volatility_improvement = current_portfolio_volatility - expected_portfolio_volatility
                            sharpe_improvement = sharpe_ratio - (current_portfolio_return - risk_free_rate) / current_portfolio_volatility if current_portfolio_volatility > 0 else sharpe_ratio

                            expected_improvements = {
                                "expected_return": expected_portfolio_return,
                                "expected_volatility": expected_portfolio_volatility,
                                "sharpe_ratio": sharpe_ratio,
                                "return_improvement": return_improvement,
                                "volatility_improvement": volatility_improvement,
                                "sharpe_improvement": sharpe_improvement,
                                "risk_reduction": abs(volatility_improvement) if volatility_improvement < 0 else 0,
                                "expected_return_display": f"{expected_portfolio_return:.1%}",
                                "expected_volatility_display": f"{expected_portfolio_volatility:.1%}",
                                "sharpe_ratio_display": f"{sharpe_ratio:.2f}",
                                "return_improvement_display": f"{return_improvement:+.1%}",
                                "volatility_improvement_display": f"{volatility_improvement:+.1%}",
                                "sharpe_improvement_display": f"{sharpe_improvement:+.2f}"
                            }
                        else:
                            # No market data available for accurate projections - use estimates
                            logging.warning("No market data available for accurate projections, using estimates")
                            expected_improvements = {
                                "expected_return": 0.08,  # Assume 8% return
                                "expected_volatility": 0.15,  # Assume 15% volatility
                                "sharpe_ratio": 0.4,  # (0.08 - 0.02) / 0.15
                                "return_improvement": 0,
                                "volatility_improvement": 0,
                                "sharpe_improvement": 0,
                                "risk_reduction": 0,
                                "expected_return_display": "8.0%",
                                "expected_volatility_display": "15.0%",
                                "sharpe_ratio_display": "0.40",
                                "return_improvement_display": "+0.0%",
                                "volatility_improvement_display": "+0.0%",
                                "sharpe_improvement_display": "+0.00",
                                "note": "Estimates based on historical market averages (market data unavailable)"
                            }
                except Exception as e:
                    logging.error(f"Error calculating expected improvements: {e}")
                    # Use default estimates instead of failing completely
                    expected_improvements = {
                        "expected_return": 0.08,
                        "expected_volatility": 0.15,
                        "sharpe_ratio": 0.4,
                        "return_improvement": 0,
                        "volatility_improvement": 0,
                        "sharpe_improvement": 0,
                        "risk_reduction": 0,
                        "expected_return_display": "8.0%",
                        "expected_volatility_display": "15.0%",
                        "sharpe_ratio_display": "0.40",
                        "return_improvement_display": "+0.0%",
                        "volatility_improvement_display": "+0.0%",
                        "sharpe_improvement_display": "+0.00",
                        "error": str(e),
                        "note": "Using default estimates due to calculation error"
                    }

            # Create structured recommendations object with enhanced data
            recommendations = {
                "target_weights": target_weights,
                "actions": actions,
                "market_regime": market_regime,
                "risk_level": optimization_results.get("risk_level", risk_level),
                "original_risk_level": optimization_results.get("original_risk_level", risk_level),
                "market_analysis": regime_analysis,
                "diversification_suggestions": diversification_results.get("suggested_assets", []),
                "sector_analysis": diversification_results.get("current_sector_exposure", {}),
                "optimization_method": optimization_results.get("optimization_method", "unknown"),
                "risk_profile_analysis": optimization_results.get("risk_profile_analysis", {}),
                "market_data_available": optimization_results.get("market_data_available", False),
                "llm_reasoning": optimization_results.get("llm_reasoning", "Strategy reasoning not available"),
                "diversification_analysis": diversification_results.get("openai_analysis", ""),
                "optimization_guidance": optimization_results.get("optimization_guidance", ""),
                "expected_improvements": expected_improvements,
                "expected_metrics": {
                    "expected_return": expected_improvements.get("expected_return", 0),
                    "expected_volatility": expected_improvements.get("expected_volatility", 0),
                    "sharpe_ratio": expected_improvements.get("sharpe_ratio", 0)
                }
            }
            
            self.log_action("strategy_task_completed", {
                "portfolio_id": portfolio_id,
                "risk_profile": str(risk_profile.get("level", "unknown")),
                "success": True,
                "actions_generated": len(actions)
            })
            
            return {
                "success": True,
                "agent": "StrategyAgent",
                "portfolio_id": portfolio_id,
                "recommendations": recommendations,
                "risk_metrics": risk_metrics,
                "optimization": optimization_results,
                "market_regime": market_regime,
                "risk_profile": risk_profile
            }
            
        except Exception as e:
            self.log_action("strategy_task_failed", {"error": str(e)})
            return {
                "success": False,
                "agent": "StrategyAgent",
                "error": str(e)
            }