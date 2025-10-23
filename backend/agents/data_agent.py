"""Data Agent - Market Data Specialist"""

from typing import Dict, Any, List
import logging
import pandas as pd
from .base_agent import BaseAgent
from .tools import (
    fetch_market_data_tool,
    calculate_indicators_tool
)

logger = logging.getLogger(__name__)


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
            fetch_result = fetch_market_data_tool(symbols, days)
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