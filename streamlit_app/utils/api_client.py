"""API client for communicating with the backend"""
import requests
import json
from typing import Dict, Any, Optional
import streamlit as st

class APIClient:
    """Client for API communication"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Check if token exists in session state and restore it
        if 'user_token' in st.session_state and st.session_state.user_token:
            self.session.headers.update({"Authorization": f"Bearer {st.session_state.user_token}"})
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None, 
                     files: Dict = None, headers: Dict = None) -> Dict[str, Any]:
        """Make HTTP request to API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, headers=headers)
            elif method.upper() == "POST":
                if files:
                    response = self.session.post(url, data=data, files=files, headers=headers)
                elif headers and headers.get("Content-Type") == "application/x-www-form-urlencoded":
                    response = self.session.post(url, data=data, headers=headers)
                else:
                    response = self.session.post(url, json=data, headers=headers)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, headers=headers)
            elif method.upper() == "PATCH":
                response = self.session.patch(url, json=data, headers=headers)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, headers=headers)
            else:
                return {"success": False, "error": f"Unsupported method: {method}"}
            
            # Handle response
            if response.status_code in [200, 201]:
                try:
                    response_data = response.json()
                    # Check if response already has success/data structure
                    if "success" in response_data:
                        return response_data
                    else:
                        # Wrap response in standard format
                        return {"success": True, "data": response_data}
                except json.JSONDecodeError:
                    return {"success": True, "data": response.text}
            else:
                try:
                    error_data = response.json()
                    return {"success": False, "error": error_data.get("detail", "Unknown error")}
                except json.JSONDecodeError:
                    return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                    
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Cannot connect to backend server. Please ensure it's running."}
        except Exception as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}
    
    def login(self, email: str, password: str) -> Dict[str, Any]:
        """Login user"""
        data = {
            "username": email,
            "password": password
        }
        # Send as form data, not JSON
        response = self._make_request("POST", "/api/auth/login", data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
        
        if response.get("success"):
            # Store token for future requests
            token_data = response.get("data", {})
            token = token_data.get("access_token")
            if token:
                self.session.headers.update({"Authorization": f"Bearer {token}"})
                # Store token in session state for persistence
                st.session_state.user_token = token
                # Get user info
                user_info = self.get_current_user()
                if user_info.get("success"):
                    response["user_info"] = user_info.get("data")
        
        return response
    
    def register(self, email: str, password: str, full_name: str) -> Dict[str, Any]:
        """Register new user"""
        data = {
            "email": email,
            "password": password,
            "full_name": full_name
        }
        return self._make_request("POST", "/api/auth/register", data=data)
    
    def get_current_user(self) -> Dict[str, Any]:
        """Get current user information"""
        return self._make_request("GET", "/api/auth/me")
    
    def get_portfolios(self) -> Dict[str, Any]:
        """Get all portfolios for current user"""
        return self._make_request("GET", "/api/portfolio/")
    
    def create_portfolio(self, name: str, description: str = None, cash_balance: float = 0.0) -> Dict[str, Any]:
        """Create new portfolio"""
        data = {
            "name": name,
            "description": description,
            "cash_balance": cash_balance
        }
        return self._make_request("POST", "/api/portfolio/", data=data)
    
    def upload_portfolio_csv(self, file_content: bytes, filename: str, portfolio_name: str) -> Dict[str, Any]:
        """Upload portfolio via CSV"""
        files = {"file": (filename, file_content, "text/csv")}
        data = {"portfolio_name": portfolio_name}
        return self._make_request("POST", "/api/portfolio/upload-csv", data=data, files=files)
    
    def get_portfolio(self, portfolio_id: int) -> Dict[str, Any]:
        """Get portfolio details"""
        return self._make_request("GET", f"/api/portfolio/{portfolio_id}")
    
    def get_holdings(self, portfolio_id: int) -> Dict[str, Any]:
        """Get portfolio holdings"""
        return self._make_request("GET", f"/api/portfolio/{portfolio_id}/holdings")
    
    def get_risk_profile(self) -> Dict[str, Any]:
        """Get user risk profile"""
        return self._make_request("GET", "/api/risk-profile/")
    
    def create_risk_profile(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update risk profile"""
        return self._make_request("POST", "/api/risk-profile/", data=risk_data)
    
    def get_rebalancing_suggestions(self, portfolio_id: int) -> Dict[str, Any]:
        """Get rebalancing suggestions for portfolio"""
        return self._make_request("GET", f"/api/rebalancing/{portfolio_id}/suggestions")
    
    def generate_rebalancing_suggestion(self, portfolio_id: int) -> Dict[str, Any]:
        """Generate new rebalancing suggestion"""
        return self._make_request("POST", f"/api/rebalancing/{portfolio_id}/generate")
    
    def approve_suggestion(self, suggestion_id: int) -> Dict[str, Any]:
        """Approve rebalancing suggestion"""
        return self._make_request("PATCH", f"/api/rebalancing/suggestions/{suggestion_id}/approve")
    
    def reject_suggestion(self, suggestion_id: int) -> Dict[str, Any]:
        """Reject a rebalancing suggestion"""
        return self._make_request(
            "PATCH",
            f"/api/rebalancing/suggestions/{suggestion_id}/reject"
        )

    def get_analytics_summary(self, portfolio_id: int) -> Dict[str, Any]:
        """Get analytics summary for a portfolio"""
        return self._make_request(
            "GET",
            f"/api/analytics/{portfolio_id}/risk"
        )

    def get_portfolio_performance(self, portfolio_id: int) -> Dict[str, Any]:
        """Get portfolio performance metrics"""
        return self._make_request(
            "GET",
            f"/api/analytics/{portfolio_id}/performance"
        )
    
    def create_or_update_risk_profile(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update risk profile with questionnaire data"""
        return self._make_request("POST", "/api/risk-profile/", data=risk_data)
    
    def get_analytics(self, portfolio_id: int) -> Dict[str, Any]:
        """Get portfolio analytics"""
        return self._make_request("GET", f"/api/analytics/{portfolio_id}/performance")
    
    def get_risk_metrics(self, portfolio_id: int) -> Dict[str, Any]:
        """Get portfolio risk metrics"""
        return self._make_request("GET", f"/api/analytics/{portfolio_id}/risk")
