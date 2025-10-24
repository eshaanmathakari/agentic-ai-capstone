"""Data Agent - Market Data Specialist
Implements CrewAI Agent for data collection and processing"""

from typing import Dict, Any, List
import logging
import pandas as pd
from crewai import Agent, Task
from .base_agent import BaseAgent
from .tools import (
    fetch_market_data_tool,
    calculate_indicators_tool,
    FETCH_MARKET_DATA_CREW_TOOL,
    CALCULATE_INDICATORS_CREW_TOOL
)

logger = logging.getLogger(__name__)


# CrewAI Agent Definition
def create_data_agent(llm=None) -> Agent:
    """Create Data Agent using CrewAI framework"""
    from .tools import (
        FETCH_MARKET_DATA_CREW_TOOL,
        CALCULATE_INDICATORS_CREW_TOOL,
        fetch_market_data_tool,
        calculate_indicators_tool,
        crewai_llm
    )

    # Use crew LLM if available, otherwise use provided llm
    if crewai_llm and llm is None:
        llm = crewai_llm

    # Use CrewAI Tool objects if available, otherwise use empty list
    tools_list = [FETCH_MARKET_DATA_CREW_TOOL, CALCULATE_INDICATORS_CREW_TOOL]
    tools_list = [t for t in tools_list if t is not None]

    # If no valid tools available, use empty list (agents will work without tools)
    if not tools_list:
        tools_list = []

    return Agent(
        role='Market Data Specialist',
        goal='Gather and validate financial market data with precision and accuracy',
        backstory="""You are a specialized Market Data Analyst with deep expertise in financial data collection and validation.

        Your sole focus is gathering and validating financial market data. You excel at:
        - Fetching real-time and historical market data from multiple sources
        - Calculating technical indicators (RSI, MACD, moving averages, volatility)
        - Validating data quality and identifying anomalies
        - Processing large datasets efficiently and accurately

        You are methodical, precise, and always ensure data integrity. Your analysis provides the foundation for all portfolio decisions.
        - Market data validation and quality assurance
        - Understanding of market correlations and sector rotations
        - News sentiment analysis and market impact assessment

        You always prioritize data accuracy, handle API rate limits gracefully, and provide comprehensive market intelligence that forms the foundation for sound investment decisions.""",
        tools=tools_list,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


class DataAgent(BaseAgent):
    """Market Data Specialist Agent"""
    
    def __init__(self, db_session=None):
        super().__init__("DataAgent", db_session)
        self.role = "Market Data Specialist"
        self.goal = "Fetch, process, and curate high-quality market data for portfolio analysis"
    
    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data collection and processing tasks"""
        try:
            self.log_action("execute_data_task", task_input)
            
            # Validate required fields
            required_fields = ["portfolio_id", "symbols"]
            if not self.validate_input(required_fields, task_input):
                return {"success": False, "error": "Missing required fields"}
            
            portfolio_id = task_input["portfolio_id"]
            symbols = task_input["symbols"]
            days = task_input.get("days", 365)
            
            # Fetch market data (tool returns metadata wrapper)
            fetch_result = fetch_market_data_tool(symbols, days, self.db_session)
            if not fetch_result.get("success"):
                error_msg = fetch_result.get("error", "Failed to fetch market data")
                self.log_action("data_task_failed", {"error": error_msg})
                return {
                    "success": False,
                    "agent": "DataAgent",
                    "error": error_msg,
                    "market_data": fetch_result.get("data", {}),
                    "metadata": {
                        "symbols": symbols,
                        "days": days
                    }
                }
            
            market_data = fetch_result.get("data", {})
            
            # Calculate indicators for each symbol
            indicators = {}
            for symbol, data in market_data.items():
                if isinstance(data, dict) and 'historical_data' in data:
                    indicators[symbol] = calculate_indicators_tool(symbol, data['historical_data'])
                else:
                    indicators[symbol] = {"success": False, "error": "No historical data available"}
            
            self.log_action("data_task_completed", {
                "portfolio_id": portfolio_id,
                "symbols_count": len(symbols),
                "success": True
            })
            
            return {
                "success": True,
                "agent": "DataAgent",
                "portfolio_id": portfolio_id,
                "market_data": market_data,
                "indicators": indicators,
                "symbols_processed": list(market_data.keys()),
                "metadata": {
                    "days": fetch_result.get("days", days),
                    "requested_symbols": symbols
                }
            }
            
        except Exception as e:
            self.log_action("data_task_failed", {"error": str(e)})
            return {
                "success": False,
                "agent": "DataAgent",
                "error": str(e)
            }