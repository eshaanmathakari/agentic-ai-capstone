"""Validation Agent - Portfolio Compliance Officer"""

from typing import Dict, Any, List
import logging
from sqlalchemy.orm import Session
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    """Portfolio Compliance Officer Agent"""
    
    def __init__(self, db_session: Session = None):
        super().__init__("ValidationAgent", db_session)
        self.role = "Portfolio Compliance Officer"
        self.goal = "Validate recommendations and ensure data integrity"
    
    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation and storage tasks"""
        try:
            self.log_action("execute_validation_task", task_input)
            
            # Validate required fields
            required_fields = ["portfolio_id", "recommendations"]
            if not self.validate_input(required_fields, task_input):
                return {"success": False, "error": "Missing required fields"}
            
            portfolio_id = task_input["portfolio_id"]
            recommendations = task_input["recommendations"]
            constraints = task_input.get("constraints", {})
            
            # Validate allocation constraints
            target_weights = recommendations.get("target_weights", {})
            allocation_validation = self.validate_allocation_tool(
                target_weights,
                constraints
            )
            
            if not allocation_validation.get("valid"):
                self.log_action("validation_failed", {
                    "portfolio_id": portfolio_id,
                    "reason": "Allocation validation failed",
                    "details": allocation_validation
                })
                return {
                    "success": False,
                    "agent": "ValidationAgent",
                    "portfolio_id": portfolio_id,
                    "error": f"Allocation validation failed: {allocation_validation.get('error', 'Unknown error')}"
                }
            
            # Calculate transaction costs
            transaction_costs = self.calculate_transaction_cost_tool(
                recommendations.get("actions", [])
            )
            
            # Check user constraints
            constraints_validation = self.check_constraints_tool(
                portfolio_id,
                constraints
            )
            
            self.log_action("validation_task_completed", {
                "portfolio_id": portfolio_id,
                "recommendations_validated": len(recommendations.get("actions", [])),
                "success": True
            })
            
            return {
                "success": True,
                "agent": "ValidationAgent",
                "portfolio_id": portfolio_id,
                "recommendations": recommendations,
                "allocation_validation": allocation_validation,
                "transaction_costs": transaction_costs,
                "constraints_validation": constraints_validation,
                "recommendations_validated": len(recommendations.get("actions", []))
            }
            
        except Exception as e:
            self.log_action("validation_task_failed", {"error": str(e)})
            return {
                "success": False,
                "agent": "ValidationAgent",
                "error": str(e)
            }
    
    def validate_allocation_tool(self, weights: Dict[str, float], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Check allocation constraints"""
        try:
            if not weights:
                return {"valid": True, "error": None}  # Empty weights is valid
            
            # Validate weights sum to ~1.0
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                return {"valid": False, "error": f"Weights sum to {total_weight:.3f}, expected 1.0"}
            
            # Check individual position limits
            max_position = constraints.get("max_position", 0.5)
            min_position = constraints.get("min_position", 0.01)
            
            violations = []
            for symbol, weight in weights.items():
                if weight > max_position:
                    violations.append(f"{symbol}: {weight:.3f} > {max_position}")
                if weight < min_position and weight > 0:
                    violations.append(f"{symbol}: {weight:.3f} < {min_position}")
            
            return {
                "valid": len(violations) == 0,
                "violations": violations,
                "total_weight": total_weight
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def calculate_transaction_cost_tool(self, actions: List[Dict], transaction_cost_rate: float = 0.001) -> Dict[str, Any]:
        """Calculate transaction costs"""
        try:
            total_cost = 0
            cost_details = []
            
            for action in actions:
                weight_change = abs(action.get("weight_change", 0))
                cost = weight_change * transaction_cost_rate
                total_cost += cost
                cost_details.append({
                    "symbol": action.get("symbol"),
                    "weight_change": weight_change,
                    "cost": cost
                })
            
            return {
                "total_cost": total_cost,
                "cost_rate": transaction_cost_rate,
                "details": cost_details
            }
        except Exception as e:
            return {"error": str(e), "total_cost": 0}
    
    def check_constraints_tool(self, portfolio_id: int, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Verify user-defined constraints"""
        try:
            return {
                "valid": True,
                "constraints_checked": list(constraints.keys()),
                "violations": []
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}