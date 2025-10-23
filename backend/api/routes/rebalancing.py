"""Rebalancing routes"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from backend.database import get_db, User, Portfolio, RebalancingSuggestion, RebalancingStatus, Holding, RiskProfile
from backend.api.routes.auth import get_current_user
from backend.agents.orchestrator import PortfolioOrchestrator

router = APIRouter()


# Pydantic models
class RebalancingSuggestionResponse(BaseModel):
    """Rebalancing suggestion response"""
    id: int
    portfolio_id: int
    current_allocation: Dict[str, Any]
    suggested_allocation: Dict[str, Any]
    reasoning: Dict[str, Any]
    trigger_reason: Optional[str]
    confidence_score: Optional[float]
    expected_improvement: Optional[Dict[str, Any]]
    estimated_transaction_cost: Optional[float]
    market_regime: Optional[str]
    status: RebalancingStatus
    created_at: datetime
    
    class Config:
        from_attributes = True


# Routes
@router.get("/{portfolio_id}/suggestions")
async def get_rebalancing_suggestions(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get rebalancing suggestions for a portfolio"""
    # Verify portfolio ownership
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    suggestions = db.query(RebalancingSuggestion).filter(
        RebalancingSuggestion.portfolio_id == portfolio_id
    ).order_by(RebalancingSuggestion.created_at.desc()).limit(10).all()
    
    return {"success": True, "data": suggestions}


@router.post("/{portfolio_id}/generate")
async def generate_rebalancing_suggestion_agentic(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate a new rebalancing suggestion using agentic AI workflow"""
    # Verify portfolio ownership
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    # Get user risk profile
    risk_profile = db.query(RiskProfile).filter(
        RiskProfile.user_id == current_user.id
    ).first()
    
    if not risk_profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please complete risk profile questionnaire first"
        )
    
    # Get holdings
    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
    
    if not holdings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Portfolio has no holdings"
        )
    
    # Prepare orchestrator inputs
    symbols = [holding.asset_symbol for holding in holdings]
    current_weights = {}
    total_value = sum(holding.current_value or holding.cost_basis for holding in holdings)
    
    for holding in holdings:
        weight = (holding.current_value or holding.cost_basis) / total_value
        current_weights[holding.asset_symbol] = weight
    
    # Create orchestrator and execute workflow
    orchestrator = PortfolioOrchestrator(portfolio_id, db)
    
    workflow_input = {
        "symbols": symbols,
        "risk_profile": {
            "level": risk_profile.risk_level.value,  # Convert enum to string value
            "age": risk_profile.age,
            "investment_horizon": risk_profile.investment_horizon,
            "questionnaire_data": risk_profile.questionnaire_data
        },
        "current_weights": current_weights,
        "constraints": {
            "max_position": 0.5,
            "min_position": 0.01,
            "transaction_cost_rate": 0.001
        }
    }
    
    # Execute agentic workflow
    result = orchestrator.execute_rebalancing_workflow(**workflow_input)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agentic workflow failed: {result.get('error', 'Unknown error')}"
        )
    
    # Extract recommendations from workflow result
    strategy_result = result.get("strategy_analysis", {})
    validation_result = result.get("validation", {})
    
    if not strategy_result.get("success") or not validation_result.get("success"):
        return {
            "success": True,
            "data": {
                "message": "Agentic analysis completed but no actionable recommendations generated",
                "portfolio_id": portfolio_id,
                "workflow_summary": result.get("summary", {}),
                "status": "analysis_complete"
            }
        }
    
    # Create suggestion record (simplified for now)
    new_suggestion = RebalancingSuggestion(
        portfolio_id=portfolio_id,
        current_allocation=current_weights,
        suggested_allocation=validation_result.get("recommendations", {}).get("target_weights", {}),  # Use validated target weights
        reasoning=result.get("summary", {}),
        trigger_reason="Agentic AI Analysis",
        confidence_score=0.9,  # High confidence from multi-agent analysis
        expected_improvement={},
        estimated_transaction_cost=validation_result.get("transaction_costs", {}).get("total_cost", 0.0),  # Use calculated cost
        market_regime=strategy_result.get("market_regime", "unknown"),
        market_indicators={},
        status=RebalancingStatus.PENDING
    )
    
    db.add(new_suggestion)
    db.commit()
    db.refresh(new_suggestion)
    
    return {
        "success": True,
        "data": {
            "message": "Agentic rebalancing analysis completed",
            "suggestion_id": new_suggestion.id,
            "workflow_result": result,
            "agentic_summary": result.get("summary", {}),
            "recommendation_id": result.get("recommendation_id"),
            "status": "agentic_analysis_complete"
        }
    }


@router.patch("/suggestions/{suggestion_id}/approve")
async def approve_suggestion(
    suggestion_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Approve a rebalancing suggestion"""
    # Get suggestion
    suggestion = db.query(RebalancingSuggestion).filter(
        RebalancingSuggestion.id == suggestion_id
    ).first()
    
    if not suggestion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Suggestion not found"
        )
    
    # Verify portfolio ownership
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == suggestion.portfolio_id,
        Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized"
        )
    
    # Update suggestion status
    suggestion.status = RebalancingStatus.APPROVED
    suggestion.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {"success": True, "data": {"message": "Suggestion approved", "suggestion_id": suggestion_id}}


@router.patch("/suggestions/{suggestion_id}/reject")
async def reject_suggestion(
    suggestion_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Reject a rebalancing suggestion"""
    # Get suggestion
    suggestion = db.query(RebalancingSuggestion).filter(
        RebalancingSuggestion.id == suggestion_id
    ).first()
    
    if not suggestion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Suggestion not found"
        )
    
    # Verify portfolio ownership
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == suggestion.portfolio_id,
        Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized"
        )
    
    # Update suggestion status
    suggestion.status = RebalancingStatus.REJECTED
    suggestion.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {"success": True, "data": {"message": "Suggestion rejected", "suggestion_id": suggestion_id}}

