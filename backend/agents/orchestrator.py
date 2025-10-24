"""Orchestrator Agent - Portfolio Management Coordinator
Implements CrewAI Crew for multi-agent collaboration"""

# Simplified orchestrator using CrewAI framework
from typing import Dict, Any, List
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from crewai import Crew, Task, Process

from .base_agent import BaseAgent
from .data_agent import create_data_agent, DataAgent
from .strategy_agent import create_strategy_agent, StrategyAgent
from .validation_agent import create_validation_agent, ValidationAgent
from .tools import orchestrator_llm, CREWAI_AVAILABLE
from .utils import _sanitize_float_values

logger = logging.getLogger(__name__)


class PortfolioOrchestrator(BaseAgent):
    """Portfolio Management Orchestrator - Coordinates CrewAI Crew
    
    The Chief Investment Officer coordinating a team of specialists:
    - Data Agent (gpt-5-nano): Market data collection and analysis
    - Strategy Agent (gpt-5-nano): Portfolio optimization and asset allocation
    - Validation Agent (gpt-5-nano): Risk assessment and compliance validation
    - Orchestrator (gpt-5): High-level coordination and decision making
    """
    
    def __init__(self, portfolio_id: int, db_session: Session = None):
        super().__init__("PortfolioOrchestrator", db_session)
        self.portfolio_id = portfolio_id
        
        # Create CrewAI Agents - wrapped in try/except for safe degradation
        try:
            # Agents use gpt-5-nano for efficiency, orchestrator uses gpt-5 for coordination
            self.data_agent_crew = create_data_agent()  # Uses gpt-5-nano
            self.strategy_agent_crew = create_strategy_agent()  # Uses gpt-5-nano
            self.validation_agent_crew = create_validation_agent()  # Uses gpt-5-nano
        except Exception as e:
            logger.warning(f"Failed to initialize CrewAI agents, will use legacy agents only: {e}")
            self.data_agent_crew = None
            self.strategy_agent_crew = None
            self.validation_agent_crew = None
        
        # Keep legacy agents for backward compatibility
        self.data_agent = DataAgent(db_session)
        self.strategy_agent = StrategyAgent(db_session)
        self.validation_agent = ValidationAgent(db_session)
    
    def execute_rebalancing_workflow(self, **kwargs) -> Dict[str, Any]:
        """Execute the complete portfolio rebalancing workflow using CrewAI Crew"""
        try:
            self.log_action("start_rebalancing_workflow", {
                "portfolio_id": self.portfolio_id,
                "kwargs": kwargs
            })
            
            # Get input parameters
            symbols = kwargs.get("symbols", ["AAPL", "GOOGL", "MSFT"])
            risk_profile = kwargs.get("risk_profile", {"level": "moderate"})
            current_weights = kwargs.get("current_weights", {})
            
            # Try to execute CrewAI workflow if agents are available
            crew_output = None
            if self.data_agent_crew and self.strategy_agent_crew and self.validation_agent_crew:
                try:
                    # Create CrewAI Tasks
                    data_task = Task(
                        description=f"""Fetch and analyze market data for symbols: {symbols}.
                        Analyze price trends, volatility, and technical indicators.
                        Return structured market analysis data.""",
                        agent=self.data_agent_crew,
                        expected_output="Comprehensive market data with technical indicators"
                    )
                    
                    strategy_task = Task(
                        description=f"""Based on the market data and risk profile {risk_profile['level']},
                        generate optimal portfolio allocation recommendations.
                        Consider user's risk tolerance, investment horizon, and diversification needs.
                        Provide specific target weights with detailed reasoning.""",
                        agent=self.strategy_agent_crew,
                        expected_output="Target weights with comprehensive analysis and reasoning"
                    )
                    
                    validation_task = Task(
                        description="""Validate the recommended allocation for:
                        1. Regulatory compliance (concentration limits)
                        2. Risk management suitability
                        3. Transaction cost feasibility
                        4. User risk profile alignment
                        Provide compliance report and approval recommendation.""",
                        agent=self.validation_agent_crew,
                        expected_output="Compliance validation report with detailed analysis"
                    )
                    
                    # Create and execute Crew
                    crew_kwargs = {
                        "agents": [self.data_agent_crew, self.strategy_agent_crew, self.validation_agent_crew],
                        "tasks": [data_task, strategy_task, validation_task],
                        "verbose": True,
                        "process": Process.sequential
                    }

                    # Add orchestrator LLM as manager if available
                    if orchestrator_llm and CREWAI_AVAILABLE:
                        crew_kwargs["manager_llm"] = orchestrator_llm
                        logging.info("Using gpt-5 orchestrator for crew coordination")

                    crew = Crew(**crew_kwargs)
                    
                    # Execute crew workflow
                    crew_output = crew.kickoff(inputs={
                        "symbols": symbols,
                        "risk_profile": risk_profile,
                        "current_weights": current_weights
                    })
                except Exception as e:
                    logger.warning(f"CrewAI workflow failed, falling back to legacy agents: {e}")
                    crew_output = None
            
            # Always run legacy workflow for data collection, strategy analysis, and validation
            data_result = self._execute_data_collection(**kwargs)
            strategy_result = self._execute_strategy_analysis(data_result, **kwargs)
            validation_result = self._execute_validation(strategy_result, **kwargs)
            
            # Compile final result
            final_result = {
                "success": True,
                "portfolio_id": self.portfolio_id,
                "workflow_completed_at": datetime.utcnow().isoformat(),
                "crew_output": crew_output,  # CrewAI output (may be None)
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
            
            return _sanitize_float_values(final_result)
            
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
