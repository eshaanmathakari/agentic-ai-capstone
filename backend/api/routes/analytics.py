"""Analytics routes"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

from backend.database import get_db, User, Portfolio, Holding
from backend.api.routes.auth import get_current_user


# Removed fallback risk metrics - system now requires real market data

router = APIRouter()


# Routes
@router.get("/{portfolio_id}/performance")
async def get_portfolio_performance(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio performance metrics"""
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
    
    # Get holdings
    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
    
    if not holdings:
        return {
            "portfolio_id": portfolio_id,
            "message": "No holdings in portfolio",
            "metrics": {}
        }
    
    # Get historical prices for each holding
    total_value = sum((h.current_value or h.cost_basis or 0) for h in holdings)
    weights = {}
    if total_value > 0:
        for holding in holdings:
            value = holding.current_value or holding.cost_basis or 0
            if value > 0:
                weights[holding.asset_symbol] = value / total_value
    
    metrics = {
        "total_value": total_value,
        "top_holdings": sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5],
        "weights": weights,
        "holdings_count": len(holdings)
    }
    
    return {
        "portfolio_id": portfolio_id,
        "metrics": metrics,
        "data_source": "portfolio_only",
        "note": "Advanced analytics require market data integration"
    }


@router.get("/{portfolio_id}/risk")
async def get_portfolio_risk_metrics(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio risk metrics"""
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
    
    # Get holdings
    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
    
    if not holdings:
        return {
            "portfolio_id": portfolio_id,
            "message": "No holdings in portfolio",
            "risk_metrics": {}
        }
    
    return {
        "portfolio_id": portfolio_id,
        "risk_metrics": {
            "holdings_count": len(holdings),
            "total_value": sum(h.current_value or h.cost_basis or 0 for h in holdings),
            "data_source": "portfolio_only",
            "note": "Advanced risk metrics require market data integration"
        }
    }


@router.get("/{portfolio_id}/stress-test")
async def run_stress_test(
    portfolio_id: int,
    scenario: Optional[str] = Query(None, description="Specific scenario to run"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Run stress test on portfolio"""
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
    
    # Get holdings
    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
    holdings_dict = {
        h.asset_symbol: h.current_value or h.cost_basis
        for h in holdings
    }
    
    portfolio_value = sum(holdings_dict.values())
    
    # Run stress test
    stress_tester = StressTester()
    
    if scenario:
        results = [stress_tester.run_scenario_by_key(portfolio_value, holdings_dict, scenario)]
    else:
        results = stress_tester.run_all_scenarios(portfolio_value, holdings_dict)
    
    summary = stress_tester.get_scenario_summary(results)
    
    return {
        "portfolio_id": portfolio_id,
        "portfolio_value": portfolio_value,
        "stress_test_results": results,
        "summary": summary
    }


@router.get("/{portfolio_id}/monte-carlo")
async def run_monte_carlo_simulation(
    portfolio_id: int,
    n_simulations: int = Query(1000, ge=100, le=10000),
    n_periods: int = Query(252, ge=30, le=1260, description="Number of periods (days)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Run Monte Carlo simulation for portfolio"""
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
    
    # Get holdings and historical data
    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
    
    if not holdings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Portfolio has no holdings"
        )
    
    # Get historical prices
    collector = MarketDataCollector(db)
    prices_dict = {}
    holdings_dict = {}
    
    for holding in holdings:
        df = collector.get_historical_data(holding.asset_symbol, days=365)
        if not df.empty:
            prices_dict[holding.asset_symbol] = df['close']
            holdings_dict[holding.asset_symbol] = holding.current_value or holding.cost_basis
    
    # Calculate portfolio value series
    prices_df = pd.DataFrame(prices_dict).dropna()
    total_value = sum(holdings_dict.values())
    weights = {s: v / total_value for s, v in holdings_dict.items()}
    
    portfolio_value = pd.Series(0, index=prices_df.index)
    for symbol, weight in weights.items():
        if symbol in prices_df.columns:
            portfolio_value += prices_df[symbol] * weight
    
    # Calculate returns
    returns = portfolio_value.pct_change().dropna()
    
    # Run Monte Carlo simulation
    mc_sim = MonteCarloSimulator(n_simulations=n_simulations)
    simulated_paths, analysis = mc_sim.run_simulation(
        returns,
        total_value,
        n_periods=n_periods,
        method="parametric"
    )
    
    # Get confidence bands for visualization
    bands = mc_sim.get_confidence_bands(simulated_paths)
    
    return {
        "portfolio_id": portfolio_id,
        "initial_value": total_value,
        "analysis": analysis,
        "confidence_bands": bands.to_dict(orient='list')
    }

