"""Base agent class with common functionality"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
from datetime import datetime
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.utils import setup_logger

logger = setup_logger(__name__)


class BaseAgent(ABC):
    """Base class for all portfolio agents"""
    
    def __init__(self, name: str, db_session: Session = None):
        self.name = name
        self.logger = setup_logger(f"agent.{name}")
        self.db_session = db_session
        self.created_at = datetime.utcnow()
        
    def log_action(self, action: str, details: Dict[str, Any] = None):
        """Log agent action for audit trail"""
        log_entry = {
            "agent": self.name,
            "action": action,
            "timestamp": datetime.utcnow(),
            "details": details or {}
        }
        self.logger.info(f"Agent {self.name}: {action}", extra=log_entry)
        
        # TODO: Store in AgentLog model when created
        # if self.db_session:
        #     agent_log = AgentLog(
        #         agent_name=self.name,
        #         action=action,
        #         details=details,
        #         timestamp=datetime.utcnow()
        #     )
        #     self.db_session.add(agent_log)
        #     self.db_session.commit()
    
    @abstractmethod
    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent task"""
        pass
    
    def validate_input(self, required_fields: List[str], input_data: Dict[str, Any]) -> bool:
        """Validate input data has required fields"""
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            return False
        return True
