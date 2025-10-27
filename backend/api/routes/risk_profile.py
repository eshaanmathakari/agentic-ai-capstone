"""Risk profile routes"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

from backend.database import get_db, User, RiskProfile, RiskLevel
from backend.api.routes.auth import get_current_user

router = APIRouter()


# Pydantic models
class RiskProfileCreate(BaseModel):
    """Risk profile creation model"""
    age: Optional[int] = None
    investment_horizon: Optional[int] = None
    annual_income: Optional[float] = None
    net_worth: Optional[float] = None
    questionnaire_data: Dict[str, Any]


class RiskProfileResponse(BaseModel):
    """Risk profile response model"""
    id: int
    user_id: int
    risk_level: RiskLevel
    risk_score: float
    age: Optional[int]
    investment_horizon: Optional[int]
    annual_income: Optional[float]
    net_worth: Optional[float]
    questionnaire_data: Optional[Dict[str, Any]]
    behavioral_traits: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


def calculate_risk_score(profile_data: RiskProfileCreate) -> tuple[float, RiskLevel]:
    """
    Map user's risk choice directly to risk level.
    Returns (score, risk_level) - score is just for database compatibility
    """
    questionnaire = profile_data.questionnaire_data or {}
    
    # Get user's risk choice
    risk_tolerance = questionnaire.get('risk_tolerance', 'Medium Risk')
    
    # Map to risk level and assign a simple score for database storage
    risk_mapping = {
        'High Risk': (RiskLevel.AGGRESSIVE, 80.0),
        'Medium Risk': (RiskLevel.MODERATE, 50.0),
        'Low Risk': (RiskLevel.CONSERVATIVE, 20.0)
    }
    
    risk_level, score = risk_mapping.get(risk_tolerance, (RiskLevel.MODERATE, 50.0))
    
    return score, risk_level


# Routes
@router.post("/", response_model=RiskProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_risk_profile(
    profile_data: RiskProfileCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create or update risk profile"""
    # Check if profile already exists
    existing_profile = db.query(RiskProfile).filter(
        RiskProfile.user_id == current_user.id
    ).first()
    
    # Calculate risk score and level
    risk_score, risk_level = calculate_risk_score(profile_data)
    
    if existing_profile:
        # Update existing profile
        existing_profile.age = profile_data.age
        existing_profile.investment_horizon = profile_data.investment_horizon
        existing_profile.annual_income = profile_data.annual_income
        existing_profile.net_worth = profile_data.net_worth
        existing_profile.questionnaire_data = profile_data.questionnaire_data
        existing_profile.risk_score = risk_score
        existing_profile.risk_level = risk_level
        existing_profile.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(existing_profile)
        
        return existing_profile
    else:
        # Create new profile
        new_profile = RiskProfile(
            user_id=current_user.id,
            age=profile_data.age,
            investment_horizon=profile_data.investment_horizon,
            annual_income=profile_data.annual_income,
            net_worth=profile_data.net_worth,
            questionnaire_data=profile_data.questionnaire_data,
            risk_score=risk_score,
            risk_level=risk_level,
            behavioral_traits={}
        )
        
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
        
        return new_profile


@router.put("/", response_model=RiskProfileResponse)
async def update_risk_profile(
    profile_data: RiskProfileCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update existing risk profile"""
    profile = db.query(RiskProfile).filter(
        RiskProfile.user_id == current_user.id
    ).first()
    
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    # Calculate risk score from questionnaire
    risk_score, risk_level = calculate_risk_score(profile_data)
    
    # Update fields
    profile.age = profile_data.age
    profile.investment_horizon = profile_data.investment_horizon
    profile.annual_income = profile_data.annual_income
    profile.net_worth = profile_data.net_worth
    profile.questionnaire_data = profile_data.questionnaire_data
    profile.risk_score = risk_score
    profile.risk_level = risk_level
    profile.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(profile)
    
    return profile


@router.get("/", response_model=Optional[RiskProfileResponse])
async def get_risk_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get risk profile for current user"""
    profile = db.query(RiskProfile).filter(
        RiskProfile.user_id == current_user.id
    ).first()
    
    if not profile:
        return None  # Return None instead of 404 error
    
    return profile


@router.get("/questionnaire")
async def get_questionnaire():
    """Get simplified 2-question risk assessment questionnaire"""
    return {
        "questions": [
            {
                "id": "risk_tolerance",
                "question": "What is your risk tolerance?",
                "type": "select",
                "options": [
                    {"value": "Low Risk", "label": "Low Risk"},
                    {"value": "Medium Risk", "label": "Medium Risk"},
                    {"value": "High Risk", "label": "High Risk"}
                ],
                "default": "Medium Risk",
                "help": "Choose your risk comfort level",
                "required": True
            },
            {
                "id": "investment_horizon",
                "question": "What is your investment time horizon?",
                "type": "number",
                "min": 1,
                "max": 50,
                "default": 5,
                "unit": "years",
                "help": "How many years can you invest before needing the money?",
                "required": True
            }
        ],
        "description": "This 2-question assessment helps determine your optimal risk profile for portfolio management."
    }

