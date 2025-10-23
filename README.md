# 🤖 AI Portfolio Rebalancer

An intelligent portfolio management system powered by agentic AI that provides automated rebalancing recommendations using multi-agent workflows.

## ✨ Features

### 🧠 Agentic AI Architecture
- **Data Agent**: Fetches and processes market data using yfinance
- **Strategy Agent**: Generates optimization recommendations using OpenAI
- **Validation Agent**: Ensures compliance and validates recommendations
- **Orchestrator**: Coordinates the multi-agent workflow

### 📊 Portfolio Management
- **CSV Import**: Upload portfolios via CSV files
- **Real-time Data**: Live market prices and technical indicators
- **Risk Profiling**: 3-question simplified risk assessment
- **Portfolio Analytics**: Comprehensive performance metrics

### ⚖️ Rebalancing Engine
- **Smart Triggers**: Threshold-based and regime-aware rebalancing
- **Multiple Strategies**: Conservative, Moderate, and Aggressive approaches
- **Explainable AI**: Clear reasoning for every recommendation
- **Transaction Cost Analysis**: Cost-benefit evaluation

### 🎨 Modern UI
- **Streamlit Frontend**: Clean, responsive interface
- **Interactive Dashboards**: Real-time portfolio visualization
- **CSV Upload**: Drag-and-drop portfolio import
- **Risk Assessment**: Simple 3-question questionnaire

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (recommended)
- OpenAI API key

### 1. Clone and Setup
```bash
git clone <repository-url>
cd agentic-ai-capstone

# Copy environment file
cp .env.example .env
# Edit .env with your API keys
```

### 2. Environment Configuration
Create `.env` file:
```bash
DATABASE_URL=sqlite:///./portfolio_agent.db
SECRET_KEY=dev-secret-key-change-in-prod
OPENAI_API_KEY=your-openai-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=1440
DEBUG=True
```

### 3. Install Dependencies
```bash
# Backend
cd backend
pip install -r requirements.txt

# Streamlit (if running separately)
pip install streamlit plotly
```

### 4. Run the Application

#### Option A: Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# Backend API: http://localhost:8000
# Streamlit UI: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

#### Option B: Manual Setup
```bash
# Terminal 1: Backend
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit
cd streamlit_app
streamlit run app.py --server.port 8501
```

## 📁 Project Structure

```
agentic-ai-capstone/
├── backend/                    # FastAPI backend
│   ├── agents/                # AI agents
│   │   ├── base_agent.py      # Base agent class
│   │   ├── data_agent.py      # Market data agent
│   │   ├── strategy_agent.py  # Strategy agent
│   │   ├── validation_agent.py # Validation agent
│   │   ├── orchestrator.py    # Workflow orchestrator
│   │   └── tools.py          # Agent tools
│   ├── api/                   # API routes
│   │   ├── main.py           # FastAPI app
│   │   └── routes/           # API endpoints
│   ├── database/             # Database models
│   ├── ml_models/            # ML models
│   ├── risk_engine/          # Risk calculations
│   └── strategy_engine/      # Rebalancing logic
├── streamlit_app/            # Streamlit frontend
│   ├── app.py               # Main Streamlit app
│   ├── pages/               # App pages
│   └── utils/               # Utilities
├── deployment/              # AWS deployment
├── sample_portfolio.csv    # Sample data
├── docker-compose.yml       # Local development
├── Dockerfile              # Container definition
└── README.md               # This file
```

## 🔧 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login and get JWT token
- `GET /api/auth/me` - Get current user info

### Portfolio Management
- `GET /api/portfolio/` - List portfolios
- `POST /api/portfolio/` - Create portfolio
- `POST /api/portfolio/upload-csv` - Upload CSV portfolio
- `GET /api/portfolio/{id}` - Get portfolio details
- `GET /api/portfolio/{id}/holdings` - Get holdings

### Risk Profile
- `GET /api/risk-profile/` - Get risk profile
- `POST /api/risk-profile/` - Create/update profile
- `GET /api/risk-profile/questionnaire` - Get 3-question form

### Rebalancing
- `GET /api/rebalancing/{portfolio_id}/suggestions` - Get suggestions
- `POST /api/rebalancing/{portfolio_id}/generate` - Generate AI analysis
- `PATCH /api/rebalancing/suggestions/{id}/approve` - Approve suggestion

### Analytics
- `GET /api/analytics/{portfolio_id}/performance` - Performance metrics
- `GET /api/analytics/{portfolio_id}/risk` - Risk metrics

## 📊 CSV Format

Upload portfolios using CSV files with the following format:

```csv
symbol,quantity,purchase_price
AAPL,10,150.00
GOOGL,5,120.00
MSFT,15,300.00
TSLA,8,200.00
NVDA,12,400.00
```

**Required columns:**
- `symbol`: Stock symbol (e.g., AAPL, GOOGL)
- `quantity`: Number of shares
- `purchase_price`: Price per share when purchased

## 🧠 AI Agent Workflow

### 1. Data Collection
- Fetches historical market data (365 days)
- Calculates technical indicators (RSI, SMA)
- Gets live prices and market info
- Validates data quality

### 2. Strategy Analysis
- Analyzes current portfolio allocation
- Applies risk-based optimization
- Detects market regime (bull/bear/volatile/stable)
- Generates target allocation

### 3. Validation
- Checks portfolio constraints
- Calculates transaction costs
- Validates compliance rules
- Stores recommendations

### 4. Orchestration
- Coordinates agent workflow
- Handles error recovery
- Compiles final results
- Provides explanations

## 🎯 Risk Assessment

Simple 3-question risk profile:

1. **Investment Horizon**: 1-3 years, 3-7 years, 7+ years
2. **Loss Tolerance**: 0-10 scale (0 = no loss, 10 = high risk)
3. **Experience Level**: Beginner, Intermediate, Advanced

**Risk Levels:**
- **Conservative**: Low risk, stable returns
- **Moderate**: Balanced risk-return
- **Aggressive**: Higher risk, growth focus

## 🚀 AWS Deployment

### Prerequisites
- AWS CLI configured
- Docker installed
- Domain name (optional)

### Deploy to AWS
```bash
# Make deployment script executable
chmod +x deployment/deploy.sh

# Deploy to AWS ECS
./deployment/deploy.sh
```

### Infrastructure
- **ECS Fargate**: Container orchestration
- **Application Load Balancer**: Traffic routing
- **RDS PostgreSQL**: Database
- **ElastiCache Redis**: Caching
- **ECR**: Container registry

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed setup instructions.

## 🧪 Testing

### Sample Data
Use the provided `sample_portfolio.csv` for testing:

```bash
# Upload sample portfolio
curl -X POST "http://localhost:8000/api/portfolio/upload-csv" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@sample_portfolio.csv" \
  -F "portfolio_name=Test Portfolio"
```

### Test Workflow
1. Register/Login user
2. Complete risk profile questionnaire
3. Upload CSV portfolio
4. Generate AI rebalancing analysis
5. Review recommendations
6. Approve/reject suggestions

## 🔍 Monitoring

### Health Checks
- Backend: `GET /health`
- Database connectivity
- Redis connectivity
- Agent workflow status

### Logging
- Application logs via CloudWatch
- Agent action logging
- Error tracking and alerts

## 🛠️ Development

### Local Development
```bash
# Start backend
cd backend
uvicorn api.main:app --reload

# Start Streamlit
cd streamlit_app
streamlit run app.py
```

### Code Structure
- **Agents**: Modular AI components
- **API**: RESTful endpoints
- **Database**: SQLAlchemy models
- **Frontend**: Streamlit pages

### Adding New Features
1. Create agent tools in `backend/agents/tools.py`
2. Add API endpoints in `backend/api/routes/`
3. Update Streamlit pages in `streamlit_app/pages/`
4. Test with sample data

## 📈 Performance

### Benchmarks
- **Portfolio Optimization**: <1s for <20 assets
- **Risk Calculations**: 1-2s for typical portfolio
- **AI Analysis**: 3-5s end-to-end
- **CSV Upload**: 2-10s depending on symbols

### Optimization
- Caching with Redis
- Database connection pooling
- Async processing
- Efficient data structures

## 🔒 Security

### Authentication
- JWT-based authentication
- Password hashing with bcrypt
- Token expiration
- Secure session management

### Data Protection
- Environment variable configuration
- Database encryption
- HTTPS enforcement
- Input validation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **CrewAI**: Agent orchestration framework
- **FastAPI**: Modern web framework
- **Streamlit**: Rapid UI development
- **yfinance**: Market data integration
- **OpenAI**: LLM capabilities

## 📞 Support

For issues or questions:
1. Check the documentation
2. Review CloudWatch logs
3. Create GitHub issue
4. Contact development team

---

**Built with ❤️ using Python, FastAPI, Streamlit, and CrewAI**