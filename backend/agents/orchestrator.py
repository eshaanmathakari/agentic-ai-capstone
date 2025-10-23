"""Orchestrator Agent - Portfolio Management Coordinator"""

# Simplified orchestrator without CrewAI
from typing import Dict, Any, List
import logging
from datetime import datetime
from sqlalchemy.orm import Session

from .base_agent import BaseAgent
from .data_agent import DataAgent
from .strategy_agent import StrategyAgent
from .validation_agent import ValidationAgent

logger = logging.getLogger(__name__)


class PortfolioOrchestrator(BaseAgent):
    """Portfolio Management Orchestrator Agent"""
    
    def __init__(self, portfolio_id: int, db_session: Session = None):
        super().__init__("PortfolioOrchestrator", db_session)
        self.portfolio_id = portfolio_id
        
        # Initialize specialized agents
        self.data_agent = DataAgent(db_session)
        self.strategy_agent = StrategyAgent(db_session)
        self.validation_agent = ValidationAgent(db_session)
        
        # Simplified workflow without CrewAI
        self.workflow_steps = [
            "data_collection",
            "strategy_analysis", 
            "validation"
        ]
    
    def execute_rebalancing_workflow(self, **kwargs) -> Dict[str, Any]:
        """Execute the complete portfolio rebalancing workflow"""
        try:
            self.log_action("start_rebalancing_workflow", {
                "portfolio_id": self.portfolio_id,
                "kwargs": kwargs
            })
            
            # Step 1: Data Collection
            data_result = self._execute_data_collection(**kwargs)
            if not data_result["success"]:
                return data_result
            
            # Step 2: Strategy Analysis
            strategy_result = self._execute_strategy_analysis(data_result, **kwargs)
            if not strategy_result["success"]:
                return strategy_result
            
            # Step 3: Validation and Storage
            validation_result = self._execute_validation(strategy_result, **kwargs)
            if not validation_result["success"]:
                return validation_result
            
            # Compile final result
            final_result = {
                "success": True,
                "portfolio_id": self.portfolio_id,
                "workflow_completed_at": datetime.utcnow().isoformat(),
                "data_collection": data_result,
                "strategy_analysis": strategy_result,
                "validation": validation_result,
                "recommendation_id": validation_result.get("recommendation_id"),
                "summary": self._generate_workflow_summary(data_result, strategy_result, validation_result)
            }
            
            self.log_action("rebalancing_workflow_completed", {
                "portfolio_id": self.portfolio_id,
                "success": True,
                "recommendation_id": final_result.get("recommendation_id")
            })
            
            return final_result
            
        except Exception as e:
            self.log_action("rebalancing_workflow_failed", {
                "portfolio_id": self.portfolio_id,
                "error": str(e)
            })
            return {
                "success": False,
                "portfolio_id": self.portfolio_id,
                "error": str(e),
                "workflow_failed_at": datetime.utcnow().isoformat()
            }
    
    def _execute_data_collection(self, **kwargs) -> Dict[str, Any]:
        """Execute data collection phase"""
        try:
            # Get portfolio symbols from database or kwargs
            symbols = kwargs.get("symbols", ["AAPL", "GOOGL", "MSFT"])  # Default for testing
            days = kwargs.get("days", 365)
            
            task_input = {
                "portfolio_id": self.portfolio_id,
                "symbols": symbols,
                "days": days
            }
            
            data_result = self.data_agent.execute(task_input)
            
            # Ensure we're returning the correct structure expected downstream
            if data_result.get("success"):
                market_data = data_result.get("market_data", {})
                return {
                    **data_result,
                    "market_data": market_data
                }
            
            return data_result
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_strategy_analysis(self, data_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute strategy analysis phase"""
        try:
            # Get risk profile from database or kwargs
            risk_profile = kwargs.get("risk_profile", {"level": "moderate"})
            current_weights = kwargs.get("current_weights", {})
            
            # Debug: Log the risk_profile structure
            self.log_action("strategy_analysis_debug", {
                "risk_profile": risk_profile,
                "risk_profile_type": type(risk_profile),
                "risk_level": risk_profile.get("level") if isinstance(risk_profile, dict) else None,
                "risk_level_type": type(risk_profile.get("level")) if isinstance(risk_profile, dict) else None
            })
            
            task_input = {
                "portfolio_id": self.portfolio_id,
                "market_data": data_result.get("market_data", {}),
                "risk_profile": risk_profile,
                "current_weights": current_weights
            }
            
            return self.strategy_agent.execute(task_input)
            
        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_validation(self, strategy_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute validation and storage phase"""
        try:
            constraints = kwargs.get("constraints", {})
            
            task_input = {
                "portfolio_id": self.portfolio_id,
                "recommendations": strategy_result.get("recommendations", {}),
                "constraints": constraints
            }
            
            return self.validation_agent.execute(task_input)
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_workflow_summary(self, data_result: Dict, strategy_result: Dict, validation_result: Dict) -> Dict[str, Any]:
        """Generate a summary of the complete workflow"""
        # Extract data from strategy recommendations
        strategy_recommendations = strategy_result.get("recommendations", {})
        
        return {
            "data_collected": {
                "symbols_processed": len(data_result.get("symbols_processed", [])),
                "data_quality": "high" if data_result.get("success") else "low"
            },
            "strategy_analysis": {
                "market_regime": strategy_result.get("market_regime", "unknown"),
                "recommendations_generated": len(strategy_recommendations.get("actions", [])),
                "risk_assessment": "completed" if strategy_result.get("success") else "failed"
            },
            "validation": {
                "recommendations_validated": validation_result.get("recommendations_validated", 0),
                "compliance_status": "passed" if validation_result.get("success") else "failed",
                "stored": validation_result.get("recommendation_id") is not None
            },
            "next_steps": [
                "Review recommendations in dashboard",
                "Approve or modify rebalancing actions",
                "Execute approved trades",
                "Monitor portfolio performance"
            ]
        }
    
    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute orchestrator task (implements BaseAgent interface)"""
        return self.execute_rebalancing_workflow(**task_input)
