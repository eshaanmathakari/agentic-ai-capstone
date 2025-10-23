"""Strategy Agent - Portfolio Strategy Advisor"""

from typing import Dict, Any, List
import logging
from .base_agent import BaseAgent
from .tools import (
    portfolio_optimizer_tool,
    market_regime_detector_tool,
    risk_calculator_tool
)

logger = logging.getLogger(__name__)


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
            
            # Calculate risk metrics - convert current_weights dict to holdings format
            holdings_list = []
            for symbol, weight in current_weights.items():
                holdings_list.append({
                    "symbol": symbol,
                    "weight": weight,
                    "current_value": weight * 100000,  # Estimate based on weight
                    "unrealized_gain_loss": 0  # Default for now
                })
            
            risk_metrics = risk_calculator_tool({
                "holdings": holdings_list,
                "market_data": market_data
            })
            
            # Optimize portfolio - ensure risk_level is a string
            risk_level = risk_profile.get("level", "moderate")
            if not isinstance(risk_level, str):
                # Handle case where risk_level might be an enum or other type
                risk_level = str(risk_level) if risk_level is not None else "moderate"
            
            optimization_results = portfolio_optimizer_tool(
                current_weights, 
                risk_level
            )
            
            # Get target weights from optimization results (note: key is "weights" not "target_weights")
            target_weights = optimization_results.get("weights", {})
            
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
            
            # Create structured recommendations object
            recommendations = {
                "target_weights": target_weights,
                "actions": actions,
                "market_regime": market_regime,
                "risk_level": risk_level,
                "market_analysis": regime_analysis
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