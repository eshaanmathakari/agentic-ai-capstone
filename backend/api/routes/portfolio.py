"""Portfolio management routes"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd
import yfinance as yf
import io
import re

from backend.database import get_db, User, Portfolio, Holding
from backend.api.routes.auth import get_current_user

router = APIRouter()


# Utility functions
def map_csv_columns(df: pd.DataFrame) -> dict:
    """
    Intelligently map CSV columns to required fields using semantic matching
    
    Args:
        df: DataFrame with CSV data
        
    Returns:
        dict: Column mapping {required_field: actual_column_name}
    """
    # Define column mapping patterns with priorities
    symbol_patterns = {
        'symbol': ['symbol', 'ticker', 'stock', 'asset', 'ticker_symbol', 'stock_symbol'],
        'capitalized': ['Symbol', 'Ticker', 'Stock', 'Asset']
    }
    
    quantity_patterns = {
        'exact': ['quantity', 'shares', 'units', 'amount', 'qty'],
        'capitalized': ['Quantity', 'Shares', 'Units', 'Amount', 'Qty']
    }
    
    price_patterns = {
        'exact': ['purchase_price', 'buy_price', 'price', 'cost', 'purchase', 'avg_price', 'average_price'],
        'capitalized': ['Purchase_Price', 'Buy_Price', 'Price', 'Cost', 'Average_Buy_Price', 'Average_Price'],
        'variations': ['BUY_PRICE', 'PURCHASE_PRICE', 'purchasePrice', 'buyPrice', 'averageBuyPrice']
    }
    
    columns = df.columns.tolist()
    mapping = {}
    
    def semantic_match(col_name: str, patterns_dict: dict) -> bool:
        """Check if column matches any pattern using semantic matching"""
        col_lower = col_name.lower()
        col_normalized = col_lower.replace('_', '').replace(' ', '')
        
        # Check all pattern categories
        for category, patterns in patterns_dict.items():
            for pattern in patterns:
                pattern_lower = pattern.lower()
                pattern_normalized = pattern_lower.replace('_', '').replace(' ', '')
                
                # Exact match (highest priority)
                if col_lower == pattern_lower:
                    return True
                
                # Normalized match (handles underscores)
                if col_normalized == pattern_normalized:
                    return True
        
        return False
    
    # Find symbol column
    for col in columns:
        if semantic_match(col, symbol_patterns):
            mapping['symbol'] = col
            break
    
    # Find quantity column
    for col in columns:
        if semantic_match(col, quantity_patterns):
            mapping['quantity'] = col
            break
    
    # Find price column with enhanced semantic matching
    # This needs special handling for compound names like "Average_Buy_Price"
    best_price_match = None
    best_match_score = 0
    
    for col in columns:
        col_lower = col.lower()
        col_normalized = col_lower.replace('_', '').replace(' ', '')
        
        # Score based on how specific the match is
        score = 0
        
        # Check for price-related keywords
        if 'price' in col_normalized:
            score += 2
        if 'purchase' in col_normalized or 'buy' in col_normalized or 'avg' in col_normalized:
            score += 3
        if 'current' in col_normalized:
            score -= 1  # Deprioritize current price
        if 'cost' in col_normalized:
            score += 2
        
        # Check exact matches from patterns
        for pattern in price_patterns.get('exact', []) + price_patterns.get('capitalized', []) + price_patterns.get('variations', []):
            if col_lower == pattern.lower() or col_normalized == pattern.lower().replace('_', '').replace(' ', ''):
                score += 5  # Boost score for exact pattern match
        
        if score > best_match_score:
            best_match_score = score
            best_price_match = col
    
    if best_price_match and best_match_score > 0:
        mapping['purchase_price'] = best_price_match
    
    return mapping


def validate_and_map_csv(df: pd.DataFrame) -> tuple:
    """
    Validate CSV data and return mapped DataFrame
    
    Returns:
        tuple: (success, mapped_df_or_error_message, column_mapping)
    """
    # Map columns
    mapping = map_csv_columns(df)
    
    # Check for required columns
    missing_columns = []
    if 'symbol' not in mapping:
        missing_columns.append('symbol (or ticker/stock/asset/Symbol)')
    if 'quantity' not in mapping:
        missing_columns.append('quantity (or shares/units/amount/Shares)')
    if 'purchase_price' not in mapping:
        missing_columns.append('purchase_price (or price/cost/buy_price/Average_Buy_Price)')
    
    if missing_columns:
        error_msg = f"Missing required columns: {', '.join(missing_columns)}. "
        error_msg += f"Found columns: {', '.join(df.columns.tolist())}"
        return False, error_msg, None
    
    # Create mapped DataFrame
    mapped_df = df.copy()
    mapped_df = mapped_df.rename(columns={
        mapping['symbol']: 'symbol',
        mapping['quantity']: 'quantity',
        mapping['purchase_price']: 'purchase_price'
    })
    
    return True, mapped_df, mapping


# Pydantic models
class PortfolioCreate(BaseModel):
    """Portfolio creation model"""
    name: str
    description: Optional[str] = None
    cash_balance: float = 0.0


class PortfolioResponse(BaseModel):
    """Portfolio response model"""
    id: int
    name: str
    description: Optional[str]
    total_value: float
    cash_balance: float
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class HoldingCreate(BaseModel):
    """Holding creation model"""
    asset_symbol: str
    asset_name: Optional[str] = None
    asset_type: Optional[str] = "stock"
    quantity: float
    purchase_price: float
    purchase_date: datetime


class HoldingResponse(BaseModel):
    """Holding response model"""
    id: int
    asset_symbol: str
    asset_name: Optional[str]
    asset_type: Optional[str]
    quantity: float
    purchase_price: float
    current_price: Optional[float]
    cost_basis: Optional[float]
    current_value: Optional[float]
    unrealized_gain_loss: Optional[float]
    unrealized_gain_loss_pct: Optional[float]
    purchase_date: datetime
    
    class Config:
        from_attributes = True


class PortfolioDetail(PortfolioResponse):
    """Portfolio with holdings"""
    holdings: List[HoldingResponse]


# Routes
@router.post("/", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    portfolio_data: PortfolioCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new portfolio"""
    new_portfolio = Portfolio(
        user_id=current_user.id,
        name=portfolio_data.name,
        description=portfolio_data.description,
        cash_balance=portfolio_data.cash_balance,
        total_value=portfolio_data.cash_balance
    )
    
    db.add(new_portfolio)
    db.commit()
    db.refresh(new_portfolio)
    
    return new_portfolio


@router.get("/")
async def get_portfolios(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all portfolios for current user"""
    portfolios = db.query(Portfolio).filter(
        Portfolio.user_id == current_user.id
    ).all()
    
    return {"success": True, "data": portfolios}


@router.get("/{portfolio_id}")
async def get_portfolio(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio by ID with holdings"""
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    # Get holdings for this portfolio
    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
    
    # Serialize holdings to dictionaries for JSON response
    holdings_data = []
    for holding in holdings:
        holdings_data.append({
            "id": holding.id,
            "asset_symbol": holding.asset_symbol,
            "asset_name": holding.asset_name,
            "asset_type": holding.asset_type,
            "quantity": holding.quantity,
            "purchase_price": holding.purchase_price,
            "current_price": holding.current_price,
            "purchase_date": holding.purchase_date,
            "cost_basis": holding.cost_basis,
            "current_value": holding.current_value,
            "unrealized_gain_loss": holding.unrealized_gain_loss,
            "unrealized_gain_loss_pct": holding.unrealized_gain_loss_pct
        })
    
    # Create response dict with holdings
    portfolio_dict = {
        "id": portfolio.id,
        "name": portfolio.name,
        "description": portfolio.description,
        "total_value": portfolio.total_value,
        "cash_balance": portfolio.cash_balance,
        "created_at": portfolio.created_at,
        "updated_at": portfolio.updated_at,
        "holdings": holdings_data
    }
    
    return {"success": True, "data": portfolio_dict}


@router.delete("/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_portfolio(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete portfolio"""
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    db.delete(portfolio)
    db.commit()
    
    return None


@router.post("/{portfolio_id}/holdings", response_model=HoldingResponse, status_code=status.HTTP_201_CREATED)
async def add_holding(
    portfolio_id: int,
    holding_data: HoldingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add holding to portfolio"""
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
    
    # Calculate cost basis
    cost_basis = holding_data.quantity * holding_data.purchase_price
    
    # Create holding
    new_holding = Holding(
        portfolio_id=portfolio_id,
        asset_symbol=holding_data.asset_symbol,
        asset_name=holding_data.asset_name,
        asset_type=holding_data.asset_type,
        quantity=holding_data.quantity,
        purchase_price=holding_data.purchase_price,
        purchase_date=holding_data.purchase_date,
        cost_basis=cost_basis,
        current_price=holding_data.purchase_price,  # Initially same as purchase price
        current_value=cost_basis
    )
    
    db.add(new_holding)
    
    # Update portfolio total value
    portfolio.total_value = portfolio.total_value + cost_basis
    
    db.commit()
    db.refresh(new_holding)
    
    return new_holding


@router.get("/{portfolio_id}/holdings")
async def get_holdings(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all holdings for a portfolio"""
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
    
    holdings = db.query(Holding).filter(
        Holding.portfolio_id == portfolio_id
    ).all()
    
    return {"success": True, "data": holdings}


@router.delete("/{portfolio_id}/holdings/{holding_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_holding(
    portfolio_id: int,
    holding_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete holding from portfolio"""
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
    
    # Get holding
    holding = db.query(Holding).filter(
        Holding.id == holding_id,
        Holding.portfolio_id == portfolio_id
    ).first()
    
    if not holding:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Holding not found"
        )
    
    # Update portfolio total value
    portfolio.total_value = portfolio.total_value - (holding.current_value or holding.cost_basis or 0)
    
    db.delete(holding)
    db.commit()
    
    return None


@router.post("/upload-csv")
async def upload_portfolio_csv(
    file: UploadFile = File(...),
    portfolio_name: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload portfolio via CSV file"""
    try:
        # Read CSV content
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Validate and map CSV columns intelligently
        success, result, mapping = validate_and_map_csv(df)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result
            )
        
        df = result  # Use the mapped DataFrame
        
        # Create portfolio
        new_portfolio = Portfolio(
            user_id=current_user.id,
            name=portfolio_name or f"Portfolio from {file.filename}",
            description=f"Imported from CSV: {file.filename}",
            total_value=0.0,
            cash_balance=0.0
        )
        
        db.add(new_portfolio)
        db.commit()
        db.refresh(new_portfolio)
        
        # Process each row
        holdings_created = []
        total_value = 0.0
        
        for _, row in df.iterrows():
            symbol = str(row['symbol']).upper().strip()
            quantity = float(row['quantity'])
            purchase_price = float(row['purchase_price'])
            
            # Get current price
            info = {}  # Initialize info to prevent variable scope issues
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice') or purchase_price
            except:
                current_price = purchase_price  # Fallback to purchase price
            
            # Calculate values
            cost_basis = quantity * purchase_price
            current_value = quantity * current_price
            unrealized_gain_loss = current_value - cost_basis
            unrealized_gain_loss_pct = (unrealized_gain_loss / cost_basis) * 100 if cost_basis > 0 else 0
            
            # Create holding
            holding = Holding(
                portfolio_id=new_portfolio.id,
                asset_symbol=symbol,
                asset_name=info.get('longName', symbol),
                asset_type='stock',
                quantity=quantity,
                purchase_price=purchase_price,
                current_price=current_price,
                purchase_date=datetime.utcnow(),  # Default to current date
                cost_basis=cost_basis,
                current_value=current_value,
                unrealized_gain_loss=unrealized_gain_loss,
                unrealized_gain_loss_pct=unrealized_gain_loss_pct
            )
            
            db.add(holding)
            holdings_created.append(holding)
            total_value += current_value
        
        # Update portfolio total value
        new_portfolio.total_value = total_value
        
        db.commit()
        
        return {
            "success": True,
            "message": f"Portfolio '{new_portfolio.name}' created successfully",
            "portfolio_id": new_portfolio.id,
            "holdings_count": len(holdings_created),
            "total_value": total_value,
            "holdings": [
                {
                    "symbol": h.asset_symbol,
                    "quantity": h.quantity,
                    "current_value": h.current_value,
                    "gain_loss_pct": h.unrealized_gain_loss_pct
                }
                for h in holdings_created
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing CSV: {str(e)}"
        )

