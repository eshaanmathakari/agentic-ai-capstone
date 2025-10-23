#!/usr/bin/env python3
"""
Test data initialization script for development and testing
"""

import sys
import os
from pathlib import Path

# Add backend to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy.orm import Session
from datetime import datetime
from passlib.context import CryptContext

from database.connection import get_db, init_db
from database.models import User, Portfolio, Holding, RiskProfile, RiskLevel

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

def create_test_user(db: Session):
    """Create test user with specified credentials"""
    email = "test@example.com"
    password = "testpass"
    
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        print(f"✅ Test user {email} already exists")
        return existing_user
    
    # Create new test user
    hashed_password = get_password_hash(password)
    test_user = User(
        email=email,
        password_hash=hashed_password,
        full_name="Test User",
        is_active=True
    )
    
    db.add(test_user)
    db.commit()
    db.refresh(test_user)
    
    print(f"✅ Created test user: {email}")
    return test_user

def create_test_risk_profile(db: Session, user_id: int):
    """Create test risk profile"""
    existing_profile = db.query(RiskProfile).filter(RiskProfile.user_id == user_id).first()
    if existing_profile:
        print("✅ Test risk profile already exists")
        return existing_profile
    
    risk_profile = RiskProfile(
        user_id=user_id,
        risk_level=RiskLevel.MODERATE,
        risk_score=65.0,
        age=35,
        investment_horizon=10,
        annual_income=75000.0,
        net_worth=150000.0,
        questionnaire_data={
            "loss_tolerance": 4,
            "return_expectations": 4,
            "investment_experience": 3,
            "volatility_comfort": 3,
            "emergency_fund": 4
        },
        behavioral_traits={
            "loss_aversion": 0.3,
            "overconfidence": 0.2
        }
    )
    
    db.add(risk_profile)
    db.commit()
    db.refresh(risk_profile)
    
    print("✅ Created test risk profile")
    return risk_profile

def create_test_portfolio(db: Session, user_id: int):
    """Create test portfolio with sample holdings"""
    # Check if portfolio already exists
    existing_portfolio = db.query(Portfolio).filter(
        Portfolio.user_id == user_id,
        Portfolio.name == "Test Portfolio"
    ).first()
    
    if existing_portfolio:
        print("✅ Test portfolio already exists")
        return existing_portfolio
    
    # Create test portfolio
    portfolio = Portfolio(
        user_id=user_id,
        name="Test Portfolio",
        description="Sample portfolio for testing purposes",
        total_value=100000.0,
        cash_balance=10000.0
    )
    
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    
    # Add sample holdings
    sample_holdings = [
        {
            "asset_symbol": "AAPL",
            "asset_name": "Apple Inc.",
            "asset_type": "stock",
            "quantity": 50,
            "purchase_price": 150.00,
            "current_price": 175.00,
            "purchase_date": datetime(2024, 1, 15)
        },
        {
            "asset_symbol": "GOOGL",
            "asset_name": "Alphabet Inc.",
            "asset_type": "stock",
            "quantity": 25,
            "purchase_price": 120.00,
            "current_price": 135.00,
            "purchase_date": datetime(2024, 2, 1)
        },
        {
            "asset_symbol": "MSFT",
            "asset_name": "Microsoft Corporation",
            "asset_type": "stock",
            "quantity": 30,
            "purchase_price": 300.00,
            "current_price": 320.00,
            "purchase_date": datetime(2024, 1, 20)
        },
        {
            "asset_symbol": "TSLA",
            "asset_name": "Tesla Inc.",
            "asset_type": "stock",
            "quantity": 10,
            "purchase_price": 200.00,
            "current_price": 180.00,
            "purchase_date": datetime(2024, 2, 10)
        }
    ]
    
    total_value = portfolio.cash_balance
    
    for holding_data in sample_holdings:
        quantity = holding_data["quantity"]
        purchase_price = holding_data["purchase_price"]
        current_price = holding_data["current_price"]
        
        cost_basis = quantity * purchase_price
        current_value = quantity * current_price
        unrealized_gain_loss = current_value - cost_basis
        unrealized_gain_loss_pct = (unrealized_gain_loss / cost_basis) * 100 if cost_basis > 0 else 0
        
        holding = Holding(
            portfolio_id=portfolio.id,
            asset_symbol=holding_data["asset_symbol"],
            asset_name=holding_data["asset_name"],
            asset_type=holding_data["asset_type"],
            quantity=quantity,
            purchase_price=purchase_price,
            current_price=current_price,
            purchase_date=holding_data["purchase_date"],
            cost_basis=cost_basis,
            current_value=current_value,
            unrealized_gain_loss=unrealized_gain_loss,
            unrealized_gain_loss_pct=unrealized_gain_loss_pct
        )
        
        db.add(holding)
        total_value += current_value
    
    # Update portfolio total value
    portfolio.total_value = total_value
    
    db.commit()
    db.refresh(portfolio)
    
    print("✅ Created test portfolio with sample holdings")
    return portfolio

def clear_test_data(db: Session):
    """Clear all test data"""
    print("🧹 Clearing existing test data...")
    
    # Delete test user and related data (cascade should handle related records)
    test_user = db.query(User).filter(User.email == "test@example.com").first()
    if test_user:
        db.delete(test_user)
        db.commit()
        print("✅ Cleared test data")
    else:
        print("ℹ️ No test data found to clear")

def main():
    """Initialize test data"""
    print("🚀 Initializing test data...")
    
    # Initialize database
    init_db()
    
    # Get database session
    db = next(get_db())
    
    try:
        # Clear existing test data first
        clear_test_data(db)
        
        # Create test user
        test_user = create_test_user(db)
        
        # Create risk profile
        create_test_risk_profile(db, test_user.id)
        
        # Create portfolio with holdings
        create_test_portfolio(db, test_user.id)
        
        print("\n🎉 Test data initialization complete!")
        print("\nTest credentials:")
        print("Email: test@example.com")
        print("Password: testpass")
        
    except Exception as e:
        print(f"❌ Error initializing test data: {e}")
        db.rollback()
        return 1
    finally:
        db.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
