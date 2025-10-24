import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import numpy as np
from utils.api_client import APIClient

# Page configuration
st.set_page_config(
    page_title="AI Portfolio Rebalancer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disable Streamlit's default sidebar search and clean up UI
st.markdown("""
<style>
.stApp > div:first-child {
    padding-top: 0rem;
}
/* Hide any search elements */
[data-testid="stSidebar"] input[type="search"],
[data-testid="stSidebar"] input[placeholder*="search"],
[data-testid="stSidebar"] input[placeholder*="Search"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient()

def main():
    st.title("🤖 AI Portfolio Rebalancer")
    st.markdown("**Intelligent Portfolio Management with Agentic AI**")
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_token' not in st.session_state:
        st.session_state.user_token = None
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/4CAF50/FFFFFF?text=AI+Portfolio", width=200)
        
        if st.session_state.authenticated:
            user_info = st.session_state.user_info or {}
            st.success(f"Welcome, {user_info.get('email', 'User')}!")
            
            if st.button("🚪 Logout"):
                st.session_state.authenticated = False
                st.session_state.user_token = None
                st.session_state.user_info = None
                st.rerun()
        else:
            st.info("Please login to access the application")
    
    # Main content
    if not st.session_state.authenticated:
        # Quick test mode - add a bypass button for testing
        st.warning("🔧 Test Mode: Use the login form below, or click 'Skip Login for Testing'")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Skip Login for Testing"):
                st.session_state.authenticated = True
                st.session_state.user_token = "test_token"
                st.session_state.user_info = {"email": "test@example.com", "full_name": "Test User"}
                st.rerun()
        
        login_page()
    else:
        # Use tabs for navigation
        tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📊 Portfolio", "⚖️ Rebalancing", "📈 Analytics"])
        
        with tab1:
            dashboard_page()
        
        with tab2:
            portfolio_page()
        
        with tab3:
            rebalancing_page()
        
        with tab4:
            analytics_page()

def login_page():
    """Login and registration page"""
    st.header("🔐 Authentication")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            st.subheader("Login")
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if email and password:
                    api_client = get_api_client()
                    response = api_client.login(email, password)
                    
                    if response.get('success'):
                        st.session_state.authenticated = True
                        # Extract token from nested data structure
                        token_data = response.get('data', {})
                        st.session_state.user_token = token_data.get('access_token')
                        st.session_state.user_info = response.get('user_info', {})
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(f"Login failed: {response.get('error', 'Unknown error')}")
                else:
                    st.error("Please fill in all fields")
    
    with tab2:
        with st.form("register_form"):
            st.subheader("Register")
            email = st.text_input("Email", placeholder="your@email.com", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_password")
            full_name = st.text_input("Full Name", placeholder="John Doe", key="reg_name")
            submit = st.form_submit_button("Register")
            
            if submit:
                if email and password and full_name:
                    api_client = get_api_client()
                    response = api_client.register(email, password, full_name)
                    
                    if response.get('success'):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error(f"Registration failed: {response.get('error', 'Unknown error')}")
                else:
                    st.error("Please fill in all fields")

def dashboard_page():
    """Main dashboard page"""
    st.header("📊 Dashboard")
    
    api_client = APIClient()
    
    # Fetch user's portfolios
    portfolios_response = api_client.get_portfolios()
    portfolios = portfolios_response.get("data", []) if portfolios_response.get("success") else []
    
    # Calculate metrics
    total_portfolios = len(portfolios)
    total_value = sum(p.get('total_value', 0) for p in portfolios)
    
    # Fetch all suggestions to count active ones
    all_suggestions = []
    for portfolio in portfolios:
        suggestions_response = api_client.get_rebalancing_suggestions(portfolio.get('id', 0))
        if suggestions_response.get("success"):
            all_suggestions.extend(suggestions_response.get("data", []))
    
    active_suggestions = len([s for s in all_suggestions if s.get('status') == 'pending'])
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Portfolios", total_portfolios)
    with col2:
        st.metric("Total Value", f"${total_value:,.2f}")
    with col3:
        st.metric("Active Suggestions", active_suggestions)
    
    st.markdown("---")
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Allocation Analysis")
        if portfolios:
            # Use first portfolio for allocation analysis
            primary_portfolio = portfolios[0]
            portfolio_id = primary_portfolio.get('id')
            
            # Get rebalancing suggestions for this portfolio
            suggestions_response = api_client.get_rebalancing_suggestions(portfolio_id)
            
            if suggestions_response.get("success") and suggestions_response.get("data"):
                suggestions = suggestions_response.get("data", [])
                if suggestions:
                    latest_suggestion = suggestions[0]  # Most recent suggestion

                    # Get current allocation from actual portfolio holdings (not suggestion)
                    portfolio_response = api_client.get_portfolio(portfolio_id)
                    current_allocation = {}
                    suggested_allocation = latest_suggestion.get('suggested_allocation', {})

                    if portfolio_response.get("success"):
                        portfolio_data = portfolio_response.get("data", {})
                        holdings = portfolio_data.get("holdings", [])
                        total_value = portfolio_data.get("total_value", 1)

                        # Calculate current allocation from actual holdings
                        for holding in holdings:
                            symbol = holding.get('asset_symbol', '')
                            value = holding.get('current_value', 0)
                            if total_value > 0:
                                current_allocation[symbol] = value / total_value
                            else:
                                current_allocation[symbol] = 0

                    if current_allocation and suggested_allocation:
                        # Create comparison chart
                        symbols = list(set(current_allocation.keys()) | set(suggested_allocation.keys()))
                        current_values = [current_allocation.get(symbol, 0) for symbol in symbols]
                        suggested_values = [suggested_allocation.get(symbol, 0) for symbol in symbols]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            name='Current Allocation',
                            x=symbols,
                            y=current_values,
                            marker_color='lightblue'
                        ))
                        fig.add_trace(go.Bar(
                            name='Recommended Allocation',
                            x=symbols,
                            y=suggested_values,
                            marker_color='darkblue'
                        ))
                        
                        fig.update_layout(
                            title="Current vs Recommended Allocation",
                            xaxis_title="Assets",
                            yaxis_title="Weight (%)",
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"allocation_comparison_{portfolio_id}")
                        
                        # Show improvement metrics
                        expected_improvements = latest_suggestion.get('expected_improvements', {})
                        reasoning = latest_suggestion.get('reasoning', {})

                        if expected_improvements or reasoning:
                            st.subheader("Expected Improvements")
                            col_metric1, col_metric2, col_metric3 = st.columns(3)

                            # Use expected_improvements if available, otherwise fall back to reasoning
                            risk_reduction = expected_improvements.get('risk_reduction_display') or expected_improvements.get('risk_reduction', reasoning.get('risk_reduction', 'N/A'))
                            expected_return = expected_improvements.get('expected_return_display') or expected_improvements.get('expected_return', reasoning.get('expected_return', 'N/A'))
                            sharpe_ratio = expected_improvements.get('sharpe_ratio_display') or expected_improvements.get('sharpe_ratio', reasoning.get('sharpe_ratio', 'N/A'))

                            with col_metric1:
                                st.metric("Risk Reduction", risk_reduction if risk_reduction != 'N/A' else "0.0%")
                            with col_metric2:
                                st.metric("Expected Return", expected_return if expected_return != 'N/A' else "8.0%")
                            with col_metric3:
                                st.metric("Sharpe Ratio", sharpe_ratio if sharpe_ratio != 'N/A' else "0.40")
                    else:
                        st.info("No allocation data available for comparison.")
                else:
                    st.info("No rebalancing suggestions available. Generate recommendations in the Rebalancing tab.")
            else:
                st.info("No rebalancing suggestions available. Generate recommendations in the Rebalancing tab.")
        else:
            st.info("No portfolios available. Create one in the Portfolio tab.")
    
    with col2:
        st.subheader("Asset Allocation")
        if portfolios:
            # Aggregate all holdings across portfolios
            all_holdings = {}
            for portfolio in portfolios:
                portfolio_id = portfolio.get('id')
                portfolio_response = api_client.get_portfolio(portfolio_id)
                if portfolio_response.get("success"):
                    holdings = portfolio_response.get("data", {}).get("holdings", [])
                    for holding in holdings:
                        symbol = holding.get('asset_symbol', '')
                        value = holding.get('current_value', 0)
                        all_holdings[symbol] = all_holdings.get(symbol, 0) + value
            
            if all_holdings:
                symbols = list(all_holdings.keys())
                values = list(all_holdings.values())
                
                fig = px.pie(
                    values=values, 
                    names=symbols, 
                    title="Aggregated Asset Allocation"
                )
                st.plotly_chart(fig, use_container_width=True, key="dashboard_alloc_total")
            else:
                st.info("No holdings available.")
        else:
            st.info("No portfolios available.")


def analytics_page():
    """Analytics page"""
    st.header("📈 Analytics")
    
    api_client = APIClient()
    
    # Get portfolios for selection
    portfolios_response = api_client.get_portfolios()
    portfolios = portfolios_response.get("data", []) if portfolios_response.get("success") else []
    
    if not portfolios:
        st.info("No portfolios available. Create one in the Portfolio tab to view analytics.")
        return
    
    # Portfolio selector
    portfolio_options = {f"{p.get('name', 'Unnamed')} (${p.get('total_value', 0):,.2f})": p.get('id') 
                        for p in portfolios}
    
    selected_portfolio_name = st.selectbox(
        "Select Portfolio",
        list(portfolio_options.keys()),
        help="Choose a portfolio to analyze"
    )
    
    if not selected_portfolio_name:
        return
    
    portfolio_id = portfolio_options[selected_portfolio_name]
    
    # Get portfolio details
    portfolio_response = api_client.get_portfolio(portfolio_id)
    if not portfolio_response.get("success"):
        st.error("Failed to load portfolio details")
        return
    
    portfolio_data = portfolio_response.get("data", {})
    holdings = portfolio_data.get("holdings", [])
    
    if not holdings:
        st.info("No holdings in this portfolio.")
        return
    
    # Risk metrics
    st.subheader("Risk Analysis")
    
    # Fetch analytics data
    analytics_response = api_client.get_analytics_summary(portfolio_id)
    analytics_data = analytics_response.get("data", {}) if analytics_response.get("success") else {}
    risk_metrics = analytics_data.get("risk_metrics") or analytics_data.get("metrics", {})
    data_source = analytics_data.get("data_source")
    message = analytics_data.get("message")
    
    if message:
        st.caption(message)
    if data_source == "fallback":
        st.caption("Using estimated metrics based on current holdings (market data unavailable).")

    col1, col2, col3, col4 = st.columns(4)
    
    sharpe_ratio = risk_metrics.get('sharpe_ratio', risk_metrics.get('var_95'))
    max_drawdown = risk_metrics.get('max_drawdown', risk_metrics.get('var_99'))
    volatility = risk_metrics.get('volatility')
    beta = risk_metrics.get('beta')

    with col1:
        sharpe_ratio_display = "N/A" if sharpe_ratio is None else f"{sharpe_ratio:.2f}"
        st.metric("Sharpe Ratio", sharpe_ratio_display)
    with col2:
        max_drawdown_display = "N/A" if max_drawdown is None else f"{max_drawdown:.2f}%"
        st.metric("Max Drawdown", max_drawdown_display)
    with col3:
        volatility_display = "N/A" if volatility is None else f"{volatility:.2f}%"
        st.metric("Volatility", volatility_display)
    with col4:
        beta_display = "N/A" if beta is None else f"{beta:.2f}"
        st.metric("Beta", beta_display)
    
    st.markdown("---")
    
    # Performance comparison
    st.subheader("Portfolio Composition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Holdings Summary")
        
        holdings_data = []
        for holding in holdings:
            holdings_data.append({
                'Symbol': holding.get('asset_symbol', ''),
                'Quantity': f"{holding.get('quantity', 0):.0f}",
                'Current Price': f"${holding.get('current_price', 0):.2f}",
                'Current Value': f"${holding.get('current_value', 0):,.2f}",
                'Gain/Loss %': f"{holding.get('unrealized_gain_loss_pct', 0):.2f}%"
            })
        
        if holdings_data:
            df_holdings = pd.DataFrame(holdings_data)
            st.dataframe(df_holdings, use_container_width=True)
        else:
            st.info("No holdings data available.")
    
    with col2:
        st.subheader("Asset Allocation")
        
        symbols = [h.get('asset_symbol', '') for h in holdings]
        values = [h.get('current_value', 0) for h in holdings]
        
        if symbols and values:
            fig = px.pie(
                values=values,
                names=symbols,
                title="Current Allocation"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"analytics_pie_{portfolio_id}")
        else:
            st.info("No allocation data available.")

    with col2:
        st.subheader("Performance by Asset")
        
        symbols = [h.get('asset_symbol', '') for h in holdings]
        gains = [h.get('unrealized_gain_loss_pct', 0) for h in holdings]
        
        fig = px.bar(
            x=symbols,
            y=gains,
            title="Gain/Loss by Asset (%)",
            color=gains,
            color_continuous_scale=['red', 'white', 'green']
        )
        st.plotly_chart(fig, use_container_width=True, key=f"analytics_bar_{portfolio_id}")


def portfolio_page():
    """Portfolio management page"""
    st.header("📊 Portfolio Management")
    
    api_client = APIClient()
    
    # Tabs for different portfolio actions
    tab1, tab2, tab3 = st.tabs(["📈 View Portfolios", "📤 Upload CSV", "➕ Create Portfolio"])
    
    with tab1:
        view_portfolios(api_client)
    
    with tab2:
        upload_csv_portfolio(api_client)
    
    with tab3:
        create_manual_portfolio(api_client)


def view_portfolios(api_client: APIClient):
    """View existing portfolios"""
    st.subheader("Your Portfolios")
    
    # Get portfolios
    response = api_client.get_portfolios()
    
    if not response.get("success"):
        st.error(f"Failed to load portfolios: {response.get('error', 'Unknown error')}")
        return
    
    portfolios = response.get("data", [])
    
    if not portfolios:
        st.info("No portfolios found. Create your first portfolio using the tabs above.")
        return
    
    # Display portfolios
    for portfolio in portfolios:
        with st.expander(f"📊 {portfolio.get('name', 'Unnamed Portfolio')} - ${portfolio.get('total_value', 0):,.2f}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Value", f"${portfolio.get('total_value', 0):,.2f}")
            with col2:
                st.metric("Cash Balance", f"${portfolio.get('cash_balance', 0):,.2f}")
            with col3:
                st.metric("Holdings", len(portfolio.get('holdings', [])))
            
            # Get detailed portfolio info
            portfolio_id = portfolio.get('id')
            if portfolio_id:
                detailed_response = api_client.get_portfolio(portfolio_id)
                if detailed_response.get("success"):
                    detailed_portfolio = detailed_response.get("data", {})
                else:
                    st.error(f" Failed to load portfolio details: {detailed_response.get('error', 'Unknown error')}")
                    detailed_portfolio = {}
                
                holdings = detailed_portfolio.get('holdings', [])
                
                if holdings:
                    st.subheader("Holdings")
                    
                    # Create holdings DataFrame
                    holdings_data = []
                    for holding in holdings:
                        holdings_data.append({
                            'Symbol': holding.get('asset_symbol', ''),
                            'Quantity': holding.get('quantity', 0),
                            'Purchase Price': f"${holding.get('purchase_price', 0):.2f}",
                            'Current Price': f"${holding.get('current_price', 0):.2f}",
                            'Current Value': f"${holding.get('current_value', 0):,.2f}",
                            'Gain/Loss': f"${holding.get('unrealized_gain_loss', 0):,.2f}",
                            'Gain/Loss %': f"{holding.get('unrealized_gain_loss_pct', 0):.2f}%"
                        })
                    
                    df = pd.DataFrame(holdings_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Portfolio allocation chart
                    if len(holdings) > 0:
                        st.subheader("Portfolio Allocation")
                        
                        symbols = [h.get('asset_symbol', '') for h in holdings]
                        values = [h.get('current_value', 0) for h in holdings]
                        
                        fig = px.pie(
                            values=values,
                            names=symbols,
                            title="Asset Allocation"
                        )
                        st.plotly_chart(fig, use_container_width=True)


def upload_csv_portfolio(api_client: APIClient):
    """Upload portfolio via CSV"""
    st.subheader("📤 Upload Portfolio from CSV")
    
    st.markdown("""
    **CSV Format Required:**
    Your CSV file should have the following columns:
    - `symbol`: Stock symbol (e.g., AAPL, GOOGL)
    - `quantity`: Number of shares
    - `purchase_price`: Price per share when purchased
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with your portfolio holdings"
    )
    
    if uploaded_file is not None:
        # Portfolio name input
        portfolio_name = st.text_input(
            "Portfolio Name",
            placeholder="My Investment Portfolio",
            help="Give your portfolio a name"
        )
        
        if st.button("📤 Upload Portfolio", disabled=not portfolio_name):
            if portfolio_name:
                # Read and validate CSV
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Show preview
                    st.subheader("CSV Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Upload to backend
                    file_content = uploaded_file.getvalue()
                    response = api_client.upload_portfolio_csv(
                        file_content, 
                        uploaded_file.name, 
                        portfolio_name
                    )
                    
                    if response.get("success"):
                        st.success(f"Portfolio '{portfolio_name}' created successfully!")
                        st.json(response.get("data", {}))
                    else:
                        st.error(f"Upload failed: {response.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Error processing CSV: {str(e)}")
            else:
                st.warning("Please enter a portfolio name")


def create_manual_portfolio(api_client: APIClient):
    """Create portfolio manually"""
    st.subheader("➕ Create New Portfolio")
    
    with st.form("create_portfolio_form"):
        portfolio_name = st.text_input("Portfolio Name", placeholder="My Investment Portfolio")
        description = st.text_area("Description (Optional)", placeholder="Describe your investment strategy...")
        cash_balance = st.number_input("Initial Cash Balance", min_value=0.0, value=0.0, step=100.0)
        
        submitted = st.form_submit_button("Create Portfolio")
        
        if submitted:
            if portfolio_name:
                response = api_client.create_portfolio(portfolio_name, description, cash_balance)
                
                if response.get("success"):
                    st.success(f"Portfolio '{portfolio_name}' created successfully!")
                    st.json(response.get("data", {}))
                else:
                    st.error(f"Creation failed: {response.get('error', 'Unknown error')}")
            else:
                st.warning("Please enter a portfolio name")


def rebalancing_page():
    """Rebalancing recommendations page"""
    st.header("⚖️ Portfolio Rebalancing")
    
    api_client = APIClient()
    
    # Get portfolios first
    portfolios_response = api_client.get_portfolios()
    
    if not portfolios_response.get("success"):
        st.error(f"Failed to load portfolios: {portfolios_response.get('error', 'Unknown error')}")
        return
    
    portfolios = portfolios_response.get("data", [])
    
    if not portfolios:
        st.info("No portfolios found. Please create a portfolio first.")
        return
    
    # Portfolio selection
    portfolio_options = {f"{p.get('name', 'Unnamed')} (${p.get('total_value', 0):,.2f})": p.get('id') 
                        for p in portfolios}
    
    selected_portfolio_name = st.selectbox(
        "Select Portfolio",
        list(portfolio_options.keys()),
        help="Choose a portfolio to analyze for rebalancing"
    )
    
    if not selected_portfolio_name:
        return
    
    portfolio_id = portfolio_options[selected_portfolio_name]
    
    # Tabs for different rebalancing actions
    tab1, tab2, tab3 = st.tabs(["🤖 AI Analysis", "📊 Current Allocation", "📈 Suggestions"])
    
    with tab1:
        ai_analysis_tab(api_client, portfolio_id)
    
    with tab2:
        current_allocation_tab(api_client, portfolio_id)
    
    with tab3:
        suggestions_tab(api_client, portfolio_id)


def ai_analysis_tab(api_client: APIClient, portfolio_id: int):
    """AI-powered analysis tab"""
    st.subheader("🤖 AI-Powered Portfolio Analysis")
    
    if 'risk_profile' not in st.session_state:
        st.session_state.risk_profile = None

    # Function to fetch risk profile and cache it in session state
    def fetch_risk_profile():
        try:
            response = api_client.get_risk_profile()
            if response.get("success"):
                profile_data = response.get("data")
                if profile_data is not None:
                    st.session_state.risk_profile = profile_data
                else:
                    st.session_state.risk_profile = None
            else:
                st.session_state.risk_profile = None
        except Exception as e:
            st.session_state.risk_profile = None
    
    # Fetch profile if not in session state
    if st.session_state.risk_profile is None:
        fetch_risk_profile()

    risk_profile = st.session_state.risk_profile
    
    if not risk_profile:
        st.warning("⚠️ No risk profile found. Please complete your risk assessment to get personalized AI analysis.")
        
        if st.button("📝 Create Risk Profile"):
            with st.expander("Risk Assessment Questionnaire", expanded=True):
                st.write("Answer a few questions about your investment profile to enable AI-powered recommendations.")
                
                with st.form("risk_profile_form"):
                    risk_tolerance = st.selectbox(
                        "Risk Tolerance",
                        ["Low Risk", "Medium Risk", "High Risk"],
                        help="Choose your risk comfort level"
                    )
                    
                    investment_horizon = st.number_input(
                        "Investment Horizon (years)",
                        min_value=1,
                        max_value=50,
                        value=5,
                        help="How many years can you invest before needing the money?"
                    )

                    if st.form_submit_button("Create Risk Profile"):
                        profile_data = {
                            "age": 30,  # Default age
                            "investment_horizon": investment_horizon,
                            "annual_income": 75000,  # Default income
                            "net_worth": 150000,  # Default net worth
                            "questionnaire_data": {
                                "risk_tolerance": risk_tolerance,
                                "investment_horizon": investment_horizon
                            }
                        }
                        
                        response = api_client.create_or_update_risk_profile(profile_data)
                        
                        if response.get("success"):
                            st.success("✅ Risk profile created successfully!")
                            fetch_risk_profile() # Refresh data in session state
                            st.rerun()
                        else:
                            st.error(f"Failed to create risk profile: {response.get('error', 'Unknown error')}")
        return
    
    # Get fresh risk profile data after potential updates
    risk_profile = st.session_state.risk_profile
    
    # Display risk profile
    questionnaire_data = risk_profile.get('questionnaire_data', {})
    user_risk_choice = questionnaire_data.get('risk_tolerance', 'Medium Risk')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Risk Tolerance", user_risk_choice)
    with col2:
        st.metric("Investment Horizon", f"{risk_profile.get('investment_horizon', 0)} years")
    
    if st.button("🔄 Update Risk Profile"):
        with st.expander("Update Risk Assessment Questionnaire", expanded=True):
            st.write("Update your risk profile information.")
            
            with st.form("update_risk_profile_form"):
                # Extract current values from questionnaire_data
                questionnaire_data = risk_profile.get('questionnaire_data', {})
                current_risk_tolerance = questionnaire_data.get('risk_tolerance', 'Medium Risk')
                
                # Map internal values back to user-friendly labels for display
                risk_level_mapping = {
                    "conservative": "Low Risk",
                    "moderate": "Medium Risk", 
                    "aggressive": "High Risk"
                }
                
                # Get current risk level from database and map to display value
                current_risk_level = risk_profile.get('risk_level', 'moderate')
                if current_risk_level in risk_level_mapping:
                    current_display_risk = risk_level_mapping[current_risk_level]
                else:
                    current_display_risk = current_risk_tolerance if current_risk_tolerance in ["Low Risk", "Medium Risk", "High Risk"] else "Medium Risk"
                
                risk_options = ["Low Risk", "Medium Risk", "High Risk"]
                current_index = risk_options.index(current_display_risk) if current_display_risk in risk_options else 1
                
                risk_tolerance = st.selectbox(
                    "Risk Tolerance",
                    risk_options,
                    index=current_index,
                    help="Choose your risk comfort level"
                )
                
                investment_horizon = st.number_input(
                    "Investment Horizon (years)",
                    min_value=1,
                    max_value=50,
                    value=risk_profile.get('investment_horizon', 5),
                    help="How many years can you invest before needing the money?"
                )

                if st.form_submit_button("Update Risk Profile"):
                    profile_data = {
                        "age": risk_profile.get('age', 30),
                        "investment_horizon": investment_horizon,
                        "annual_income": risk_profile.get('annual_income', 75000),
                        "net_worth": risk_profile.get('net_worth', 150000),
                        "questionnaire_data": {
                            "risk_tolerance": risk_tolerance,
                            "investment_horizon": investment_horizon
                        }
                    }
                    
                    response = api_client.create_or_update_risk_profile(profile_data)
                    
                    if response.get("success"):
                        st.success("✅ Risk profile updated successfully!")
                        fetch_risk_profile() # Refresh data in session state
                        st.rerun()
                    else:
                        st.error(f"Failed to update risk profile: {response.get('error', 'Unknown error')}")
    
    st.markdown("---")
    
    # Generate AI analysis
    if st.button("🚀 Generate AI Rebalancing Analysis", type="primary"):
        with st.spinner("🤖 AI agents are analyzing your portfolio..."):
            try:
                response = api_client.generate_rebalancing_suggestion(portfolio_id)
                
                # Debug: Show raw response in expander
                with st.expander("🔍 Debug: Raw API Response", expanded=False):
                    st.json(response)
                
                if response.get("success"):
                    st.success("✅ AI analysis completed!")
                    
                    # Extract data from response
                    response_data = response.get("data", {})
                    workflow_result = response_data.get("workflow_result", {})
                    
                    if workflow_result:
                        st.subheader("🔍 Analysis Results")
                        
                        # Data collection results
                        data_collection = workflow_result.get("data_collection", {})
                        if data_collection.get("success"):
                            st.success("✅ Market data collected successfully")
                            with st.expander("📊 Market Data Details"):
                                st.json(data_collection.get("data", {}))
                        else:
                            st.error(f"❌ Data collection failed: {data_collection.get('error', 'Unknown error')}")
                        
                        # Strategy analysis results
                        strategy_analysis = workflow_result.get("strategy_analysis", {})
                        if strategy_analysis.get("success"):
                            st.success("✅ Strategy analysis completed")
                            with st.expander("📈 Strategy Recommendations"):
                                st.json(strategy_analysis.get("recommendations", {}))
                        else:
                            st.error(f"❌ Strategy analysis failed: {strategy_analysis.get('error', 'Unknown error')}")
                        
                        # Validation results
                        validation = workflow_result.get("validation", {})
                        if validation.get("success"):
                            st.success("✅ Recommendations validated")
                            with st.expander("✅ Validated Recommendations"):
                                st.json(validation.get("validated_recommendations", {}))
                        else:
                            st.error(f"❌ Validation failed: {validation.get('error', 'Unknown error')}")
                    
                    # Agentic summary
                    agentic_summary = response_data.get("agentic_summary", {})
                    if agentic_summary:
                        st.subheader("📋 AI Summary")
                        
                        for key, value in agentic_summary.items():
                            if isinstance(value, list):
                                st.write(f"**{key.replace('_', ' ').title()}:**")
                                for item in value:
                                    st.write(f"• {item}")
                            else:
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    
                    # Show suggestion ID if created
                    suggestion_id = response_data.get("suggestion_id")
                    if suggestion_id:
                        st.info(f"💡 Suggestion created with ID: {suggestion_id}")
                        st.rerun()  # Refresh to show new suggestion
                
                else:
                    error_msg = response.get('error', 'Unknown error')
                    st.error(f"❌ AI analysis failed: {error_msg}")
                    
                    # Provide helpful suggestions based on error
                    if "risk profile" in error_msg.lower():
                        st.info("💡 **Tip:** Make sure you've completed your risk profile questionnaire first.")
                    elif "holdings" in error_msg.lower():
                        st.info("💡 **Tip:** Make sure your portfolio has holdings before running analysis.")
                    elif "connection" in error_msg.lower():
                        st.error("🔌 **Connection Error:** Please ensure the backend server is running on localhost:8000")
                        
            except Exception as e:
                st.error(f"❌ Unexpected error during analysis: {str(e)}")
                st.info("💡 **Tip:** Check if the backend server is running and accessible.")


def current_allocation_tab(api_client: APIClient, portfolio_id: int):
    """Current portfolio allocation tab"""
    st.subheader("📊 Current Portfolio Allocation")
    
    # Get portfolio details
    portfolio_response = api_client.get_portfolio(portfolio_id)
    
    if not portfolio_response.get("success"):
        st.error(f"Failed to load portfolio: {portfolio_response.get('error', 'Unknown error')}")
        return
    
    portfolio = portfolio_response.get("data", {})
    holdings = portfolio.get('holdings', [])
    
    if not holdings:
        st.info("No holdings found in this portfolio.")
        return
    
    # Portfolio summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Value", f"${portfolio.get('total_value', 0):,.2f}")
    with col2:
        st.metric("Cash Balance", f"${portfolio.get('cash_balance', 0):,.2f}")
    with col3:
        st.metric("Number of Holdings", len(holdings))
    with col4:
        total_gain_loss = sum(h.get('unrealized_gain_loss', 0) for h in holdings)
        st.metric("Total Gain/Loss", f"${total_gain_loss:,.2f}")
    
    st.markdown("---")
    
    # Holdings table
    st.subheader("Holdings Details")
    
    holdings_data = []
    for holding in holdings:
        holdings_data.append({
            'Symbol': holding.get('asset_symbol', ''),
            'Quantity': holding.get('quantity', 0),
            'Purchase Price': f"${holding.get('purchase_price', 0):.2f}",
            'Current Price': f"${holding.get('current_price', 0):.2f}",
            'Current Value': f"${holding.get('current_value', 0):,.2f}",
            'Weight': f"{(holding.get('current_value', 0) / portfolio.get('total_value', 1)) * 100:.1f}%",
            'Gain/Loss': f"${holding.get('unrealized_gain_loss', 0):,.2f}",
            'Gain/Loss %': f"{holding.get('unrealized_gain_loss_pct', 0):.2f}%"
        })
    
    df = pd.DataFrame(holdings_data)
    st.dataframe(df, use_container_width=True)
    
    # Allocation charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Asset Allocation")
        
        symbols = [h.get('asset_symbol', '') for h in holdings]
        values = [h.get('current_value', 0) for h in holdings]
        
        fig = px.pie(
            values=values,
            names=symbols,
            title="Current Allocation"
        )
        st.plotly_chart(fig, use_container_width=True, key=f"alloc_pie_{portfolio_id}")

    with col2:
        st.subheader("Performance by Asset")
        
        symbols = [h.get('asset_symbol', '') for h in holdings]
        gains = [h.get('unrealized_gain_loss_pct', 0) for h in holdings]
        
        fig = px.bar(
            x=symbols,
            y=gains,
            title="Gain/Loss by Asset (%)",
            color=gains,
            color_continuous_scale=['red', 'white', 'green']
        )
        st.plotly_chart(fig, use_container_width=True, key=f"alloc_bar_{portfolio_id}")


def format_ai_reasoning(reasoning: dict) -> str:
    """Format AI reasoning into user-friendly display"""
    if not reasoning:
        return "No AI reasoning available."

    formatted_output = []

    # Target Weights Section
    target_weights = reasoning.get('target_weights', {})
    if target_weights:
        formatted_output.append("### 📊 Recommended Portfolio Allocation")
        formatted_output.append("")
        for symbol, weight in target_weights.items():
            # Get full company name if possible
            company_names = {
                'AAPL': 'Apple Inc.',
                'GOOGL': 'Alphabet Inc. (Google)',
                'MSFT': 'Microsoft Corporation',
                'TSLA': 'Tesla, Inc.',
                'NVDA': 'NVIDIA Corporation',
                'AMZN': 'Amazon.com, Inc.',
                'META': 'Meta Platforms, Inc.',
                'NFLX': 'Netflix, Inc.',
                'JPM': 'JPMorgan Chase & Co.',
                'BRK.B': 'Berkshire Hathaway Inc.'
            }
            company_name = company_names.get(symbol, symbol)
            formatted_output.append(f"- **{company_name} ({symbol})**: {weight:.1%} of portfolio")

    # Actions Section
    actions = reasoning.get('actions', [])
    if actions:
        formatted_output.append("")
        formatted_output.append("### 🎯 Suggested Rebalancing Actions")
        formatted_output.append("")

        buy_actions = [a for a in actions if a.get('action') == 'buy']
        sell_actions = [a for a in actions if a.get('action') == 'sell']

        if buy_actions:
            formatted_output.append("**💰 Buy Orders:**")
            for action in buy_actions:
                symbol = action.get('symbol', 'Unknown')
                weight_change = abs(action.get('weight_change', 0))
                current_weight = action.get('current_weight', 0)
                target_weight = action.get('target_weight', 0)
                company_names = {
                    'AAPL': 'Apple', 'GOOGL': 'Google', 'MSFT': 'Microsoft',
                    'TSLA': 'Tesla', 'NVDA': 'NVIDIA', 'AMZN': 'Amazon',
                    'META': 'Meta', 'NFLX': 'Netflix', 'JPM': 'JPMorgan',
                    'BRK.B': 'Berkshire Hathaway'
                }
                company_name = company_names.get(symbol, symbol)
                formatted_output.append(f"  - Increase **{company_name}** position by {weight_change:.1%} (from {current_weight:.1%} to {target_weight:.1%})")

        if sell_actions:
            formatted_output.append("")
            formatted_output.append("**💸 Sell Orders:**")
            for action in sell_actions:
                symbol = action.get('symbol', 'Unknown')
                weight_change = abs(action.get('weight_change', 0))
                current_weight = action.get('current_weight', 0)
                target_weight = action.get('target_weight', 0)
                company_names = {
                    'AAPL': 'Apple', 'GOOGL': 'Google', 'MSFT': 'Microsoft',
                    'TSLA': 'Tesla', 'NVDA': 'NVIDIA', 'AMZN': 'Amazon',
                    'META': 'Meta', 'NFLX': 'Netflix', 'JPM': 'JPMorgan',
                    'BRK.B': 'Berkshire Hathaway'
                }
                company_name = company_names.get(symbol, symbol)
                formatted_output.append(f"  - Decrease **{company_name}** position by {weight_change:.1%} (from {current_weight:.1%} to {target_weight:.1%})")

    # Expected Improvements Section
    expected_improvements = reasoning.get('expected_improvements', {})
    if expected_improvements:
        formatted_output.append("")
        formatted_output.append("### 📈 Expected Portfolio Performance")
        formatted_output.append("")

        # Return and Risk Metrics
        expected_return = expected_improvements.get('expected_return_display', expected_improvements.get('expected_return', 'N/A'))
        expected_volatility = expected_improvements.get('expected_volatility_display', expected_improvements.get('expected_volatility', 'N/A'))
        sharpe_ratio = expected_improvements.get('sharpe_ratio_display', expected_improvements.get('sharpe_ratio', 'N/A'))

        if expected_return != 'N/A':
            formatted_output.append(f"- **Projected Annual Return**: {expected_return}")
        if expected_volatility != 'N/A':
            formatted_output.append(f"- **Expected Volatility**: {expected_volatility}")
        if sharpe_ratio != 'N/A':
            formatted_output.append(f"- **Risk-Adjusted Return (Sharpe Ratio)**: {sharpe_ratio}")

        # Improvement metrics
        return_improvement = expected_improvements.get('return_improvement_display', expected_improvements.get('return_improvement', 0))
        volatility_improvement = expected_improvements.get('volatility_improvement_display', expected_improvements.get('volatility_improvement', 0))
        risk_reduction = expected_improvements.get('risk_reduction_display', expected_improvements.get('risk_reduction', 0))

        if return_improvement and return_improvement != '0.0%':
            formatted_output.append(f"- **Expected Return Improvement**: {return_improvement}")
        if volatility_improvement and volatility_improvement != '0.0%':
            formatted_output.append(f"- **Volatility Change**: {volatility_improvement}")
        if risk_reduction and risk_reduction != '0.0%':
            formatted_output.append(f"- **Risk Reduction**: {risk_reduction}")

    # Diversification Suggestions
    diversification_suggestions = reasoning.get('diversification_suggestions', [])
    if diversification_suggestions:
        formatted_output.append("")
        formatted_output.append("### 🌱 Diversification Opportunities")
        formatted_output.append("")

        # Group by sector
        sectors = {}
        for suggestion in diversification_suggestions:
            sector = suggestion.get('sector', 'Other')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(suggestion)

        for sector, suggestions in sectors.items():
            formatted_output.append(f"**{sector} Sector:**")
            for suggestion in suggestions[:2]:  # Show max 2 per sector
                symbol = suggestion.get('symbol', 'Unknown')
                reason = suggestion.get('reason', 'Improve diversification')
                suggested_weight = suggestion.get('suggested_weight', 0)
                formatted_output.append(f"  - Consider adding **{symbol}** ({suggested_weight:.1%} allocation) - {reason}")
            if len(suggestions) > 2:
                formatted_output.append(f"  - ... and {len(suggestions) - 2} more opportunities in {sector}")

    # Market Analysis
    market_regime = reasoning.get('market_regime', 'unknown')
    if market_regime and market_regime != 'unknown':
        formatted_output.append("")
        formatted_output.append("### 📊 Market Analysis")
        formatted_output.append("")

        regime_descriptions = {
            'bull': '🟢 **Bull Market**: Strong upward trends with positive investor sentiment',
            'bear': '🔴 **Bear Market**: Declining prices with negative investor sentiment',
            'stable': '🟡 **Stable Market**: Sideways movement with balanced risk and return',
            'volatile': '🟠 **Volatile Market**: High uncertainty with large price swings'
        }

        regime_desc = regime_descriptions.get(market_regime.lower(), f'📈 **{market_regime.title()} Market**')
        formatted_output.append(regime_desc)

        risk_level = reasoning.get('risk_level', 'moderate')
        original_risk_level = reasoning.get('original_risk_level', 'moderate')
        if risk_level != original_risk_level:
            formatted_output.append(f"- **Risk Level**: Adjusted from {original_risk_level} to {risk_level} based on your profile")

    # Optimization Method and Guidance
    optimization_method = reasoning.get('optimization_method', '')
    optimization_guidance = reasoning.get('optimization_guidance', '')

    if optimization_method and 'simple' in optimization_method.lower():
        formatted_output.append("")
        formatted_output.append("### 💡 Optimization Note")
        formatted_output.append("")
        formatted_output.append("Portfolio optimized using simplified allocation due to limited market data availability. For more precise optimization, consider waiting for full market data to be available.")

    if optimization_guidance:
        formatted_output.append("")
        formatted_output.append("### 🎯 Investment Strategy")
        formatted_output.append("")
        # Clean up the guidance text for display
        guidance = optimization_guidance.replace('\n', ' ').replace('  ', ' ').strip()
        formatted_output.append(guidance)

    return '\n'.join(formatted_output)


def suggestions_tab(api_client: APIClient, portfolio_id: int):
    """Rebalancing suggestions tab"""
    st.subheader("📈 Rebalancing Suggestions")
    
    # Get existing suggestions
    suggestions_response = api_client.get_rebalancing_suggestions(portfolio_id)
    
    if not suggestions_response.get("success"):
        st.error(f"Failed to load suggestions: {suggestions_response.get('error', 'Unknown error')}")
        return
    
    suggestions = suggestions_response.get("data", [])
    
    if not suggestions:
        st.info("No rebalancing suggestions found. Generate AI analysis first.")
        return
    
    # Display suggestions
    for i, suggestion in enumerate(suggestions):
        with st.expander(f"💡 Suggestion #{i+1} - {suggestion.get('status', 'Unknown').title()}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Confidence", f"{suggestion.get('confidence_score', 0):.1%}")
            with col2:
                st.metric("Transaction Cost", f"${suggestion.get('estimated_transaction_cost', 0):.2f}")
            with col3:
                st.metric("Market Regime", suggestion.get('market_regime', 'Unknown').title())
            
            # Current vs Suggested allocation - get current from actual portfolio
            suggested_allocation = suggestion.get('suggested_allocation', {})

            # Get current allocation from actual portfolio holdings
            portfolio_response = api_client.get_portfolio(portfolio_id)
            current_allocation = {}

            if portfolio_response.get("success"):
                portfolio_data = portfolio_response.get("data", {})
                holdings = portfolio_data.get("holdings", [])
                total_value = portfolio_data.get("total_value", 1)

                # Calculate current allocation from actual holdings
                for holding in holdings:
                    symbol = holding.get('asset_symbol', '')
                    value = holding.get('current_value', 0)
                    if total_value > 0:
                        current_allocation[symbol] = value / total_value
                    else:
                        current_allocation[symbol] = 0

            if current_allocation and suggested_allocation:
                st.subheader("Allocation Comparison")
                
                # Create comparison DataFrame
                comparison_data = []
                all_symbols = set(current_allocation.keys()) | set(suggested_allocation.keys())
                
                for symbol in all_symbols:
                    current_entry = current_allocation.get(symbol, {})
                    suggested_entry = suggested_allocation.get(symbol, {})
                    
                    # Handle both dict and numeric formats
                    current_weight = current_entry
                    if isinstance(current_entry, dict):
                        current_weight = current_entry.get('pct') or current_entry.get('weight') or 0
                    
                    suggested_weight = suggested_entry
                    if isinstance(suggested_entry, dict):
                        suggested_weight = suggested_entry.get('pct') or suggested_entry.get('weight') or 0
                    
                    # Ensure numeric values
                    try:
                        current_weight = float(current_weight)
                    except (TypeError, ValueError):
                        current_weight = 0.0
                    
                    try:
                        suggested_weight = float(suggested_weight)
                    except (TypeError, ValueError):
                        suggested_weight = 0.0
                    
                    comparison_data.append({
                        'Symbol': symbol,
                        'Current Weight': f"{current_weight:.1%}",
                        'Suggested Weight': f"{suggested_weight:.1%}",
                        'Change': f"{(suggested_weight - current_weight):.1%}"
                    })
                
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
            
            # AI Reasoning - User Friendly Display
            reasoning = suggestion.get('reasoning', {})
            if reasoning:
                st.subheader("🤖 AI Portfolio Analysis")

                # Use formatted display instead of raw JSON
                formatted_reasoning = format_ai_reasoning(reasoning)
                st.markdown(formatted_reasoning)

                # Technical Details (collapsible)
                with st.expander("🔧 Technical Details"):
                    st.json(reasoning)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"✅ Approve", key=f"approve_{i}"):
                    try:
                        approve_response = api_client.approve_suggestion(suggestion.get('id'))
                        if approve_response.get("success"):
                            st.success("Suggestion approved!")
                            st.rerun()
                        else:
                            st.error(f"Failed to approve: {approve_response.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error approving suggestion: {str(e)}")
            
            with col2:
                if st.button(f"❌ Reject", key=f"reject_{i}"):
                    try:
                        reject_response = api_client.reject_suggestion(suggestion.get('id'))
                        if reject_response.get("success"):
                            st.success("Suggestion rejected!")
                            st.rerun()
                        else:
                            st.error(f"Failed to reject: {reject_response.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error rejecting suggestion: {str(e)}")
            
            with col3:
                if st.button(f"📝 View Details", key=f"details_{i}"):
                    st.json(suggestion)


if __name__ == "__main__":
    main()
