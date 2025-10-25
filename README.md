# 🤖 AI Portfolio Rebalancing Agent

A sophisticated multi-agent AI system for intelligent portfolio management and rebalancing using CrewAI framework, modern portfolio theory, and real-time market data.

## ✨ Features

### 🧠 Multi-Agent AI System
- **Data Agent**: Real-time market data collection and technical analysis
- **Strategy Agent**: Advanced portfolio optimization using Modern Portfolio Theory (MPT)
- **Validation Agent**: Risk assessment and compliance validation
- **Orchestrator**: Intelligent coordination of all agents

### 📊 Portfolio Management
- **Real-time Market Data**: Yahoo Finance integration for reliable data
- **Portfolio Optimization**: MPT-based asset allocation with risk constraints
- **Risk Profiling**: User-specific risk assessment and management
- **Performance Analytics**: Comprehensive portfolio metrics and analysis
- **Rebalancing Suggestions**: AI-powered portfolio rebalancing recommendations

### 🎯 Key Capabilities
- **Dynamic Risk Assessment**: Personalized risk profiling based on user preferences
- **Market Regime Detection**: Intelligent market condition analysis
- **Transaction Cost Optimization**: Cost-aware rebalancing strategies
- **Diversification Analysis**: Sector and asset class diversification insights
- **Stress Testing**: Portfolio resilience under various market scenarios

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (for AI agents)
- Yahoo Finance access (no API key required)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-portfolio-rebalancing-agent.git
cd ai-portfolio-rebalancing-agent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r backend/requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file in project root
OPENAI_API_KEY=your_openai_api_key_here
```

5. **Initialize database**
```bash
cd backend
python -c "from database.connection import init_db; init_db()"
```

### Running the Application

**Terminal 1 - Backend API:**
```bash
cd backend
source ../venv/bin/activate
python -c "
import uvicorn
from api.main import app
print('🚀 Starting AI Portfolio Rebalancing Agent Backend...')
print('🌐 Server: http://localhost:8000')
print('📚 API Docs: http://localhost:8000/docs')
uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
"
```

**Terminal 2 - Frontend:**
```bash
source venv/bin/activate
streamlit run streamlit_app/app.py
```

## 🌐 Access Points

- **Frontend Application**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🏗️ Architecture

### Backend (FastAPI)
```
backend/
├── agents/           # CrewAI multi-agent system
│   ├── data_agent.py      # Market data specialist
│   ├── strategy_agent.py  # Portfolio optimization
│   ├── validation_agent.py # Risk & compliance
│   └── orchestrator.py    # Agent coordination
├── api/              # REST API endpoints
│   └── routes/       # API route handlers
├── database/         # Data models and connections
└── utils/           # Shared utilities
```

### Frontend (Streamlit)
```
streamlit_app/
├── app.py           # Main application
└── utils/
    └── api_client.py # Backend API integration
```

## 🔧 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user

### Portfolio Management
- `GET /api/portfolio/` - List user portfolios
- `POST /api/portfolio/` - Create new portfolio
- `GET /api/portfolio/{id}` - Get portfolio details
- `PUT /api/portfolio/{id}` - Update portfolio

### Risk Profiling
- `GET /api/risk-profile/` - Get user risk profile
- `POST /api/risk-profile/` - Create risk profile
- `PUT /api/risk-profile/` - Update risk profile

### AI Analysis
- `POST /api/rebalancing/{portfolio_id}/generate` - Generate AI rebalancing suggestions
- `GET /api/rebalancing/{portfolio_id}/suggestions` - Get rebalancing suggestions

### Analytics
- `GET /api/analytics/{portfolio_id}` - Portfolio analytics
- `GET /api/analytics/{portfolio_id}/risk` - Risk metrics
- `GET /api/analytics/{portfolio_id}/stress-test` - Stress testing

## 🧪 Usage Example

1. **Register and Login**: Create your account
2. **Create Portfolio**: Add your holdings and initial allocation
3. **Complete Risk Profile**: Answer risk assessment questions
4. **Generate AI Analysis**: Get intelligent rebalancing recommendations
5. **Review Suggestions**: Analyze AI-powered portfolio optimization

## 🔒 Security Features

- **JWT Authentication**: Secure user sessions
- **Password Hashing**: bcrypt password encryption
- **Input Validation**: Pydantic data validation
- **SQL Injection Protection**: SQLAlchemy ORM protection

## 📈 Performance Features

- **Real-time Data**: Live market data integration
- **Caching**: Intelligent data caching for performance
- **Rate Limiting**: API rate limiting and optimization
- **Error Handling**: Comprehensive error management

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **CrewAI**: Multi-agent AI framework
- **SQLAlchemy**: Database ORM
- **Pydantic**: Data validation
- **yfinance**: Market data integration

### Frontend
- **Streamlit**: Interactive web application
- **Plotly**: Data visualization
- **Pandas**: Data manipulation

### AI/ML
- **OpenAI GPT-4**: Language models for agents
- **Modern Portfolio Theory**: Mathematical optimization
- **Technical Analysis**: Market indicators and signals

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For questions and support, please open an issue on GitHub.

## 🙏 Acknowledgments

- CrewAI framework for multi-agent AI capabilities
- Yahoo Finance for reliable market data
- OpenAI for advanced language models
- Modern Portfolio Theory for mathematical optimization

---

**Built with ❤️ for intelligent portfolio management**