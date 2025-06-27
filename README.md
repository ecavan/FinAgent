# 🤖 FinAgent Pro - AI Financial Analysis Platform

**Multi-agent financial analysis powered by MLX DeepSeek with smart dual data strategy**

### 🧠 Multi-Agent AI System
- **Fundamental Agent**: 50+ financial ratios, company health analysis
- **Technical Agent**: Price trends, momentum indicators, volume analysis  
- **Sentiment Agent**: Real-time news sentiment and market psychology
- **Economic Agent**: Macroeconomic climate assessment
- **Supervisor Agent**: Investment synthesis and grade recommendations

### 📊 Smart Data Strategy
- **Primary Source**: Yahoo Finance (unlimited, reliable)
  - ✅ Financial statements, ratios, market data
  - ✅ Technical indicators, price history
  - ✅ Real-time quotes, no rate limits
- **Enhanced Source**: Alpha Vantage (optional, cached)
  - ⚡ News sentiment analysis  
  - ⚡ Economic indicators (CPI, GDP, unemployment)
  - ⚡ Advanced technical indicators


### Download MLX Model
```bash
# Download DeepSeek model (4-bit quantized for Apple Silicon)
huggingface-cli download mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit --local-dir ~/mlx-models
```

### Alpha Vantage API Key
```bash
# Get free API key from: https://www.alphavantage.co/support/#api-key
# Add to enhanced_agents.py line 1234 or set as environment variable
export ALPHA_VANTAGE_API_KEY="your_key_here"
```

### Run Application
```bash
streamlit run app.py
```

## 📊 Sample Output

```
📈 AAPL - Investment Analysis
═══════════════════════════════════════

💰 FUNDAMENTAL ANALYSIS: 8.5/10
- Revenue: $394.3B | ROE: 26.4% | D/E: 1.73
- Strong profitability, moderate leverage

📈 TECHNICAL ANALYSIS: 7.2/10  
- RSI: 45.2 | Trend: Bullish above 20-day SMA
- Volume confirmation, healthy momentum

📰 SENTIMENT ANALYSIS: 6.8/10
- Overall sentiment: Neutral-Positive
- 247 articles analyzed, 78% relevance

🌍 ECONOMIC CLIMATE: 6.5/10
- GDP: 2.3% growth | Inflation: 3.1%
- Mixed signals, Fed policy uncertainty

🧠 INVESTMENT GRADE: B+ (7.8/10)
RECOMMENDATION: BUY with 12-month target
```

## ⚠️ Alpha Vantage Free Tier Limitations

### Rate Limits
- **25 requests per day**
- **5 requests per minute**
- Requests reset at midnight EST

### Available Data
- ✅ Company overview, financial statements
- ✅ Technical indicators (RSI, MACD, SMA)  
- ✅ News sentiment analysis
- ✅ Economic indicators (monthly updates)
- ❌ Real-time intraday data
- ❌ Options data, crypto data

### Smart Caching Strategy
- Economic data: **4-hour cache** (reduces daily API usage)
- Sentiment data: **1-hour cache** (for active trading)
- Company data: **Session cache** (until app restart)
- Fallback analysis when rate limited

## 📋 File Structure

```
FinAgent/
├── app.py          # Main Streamlit application
├── agents.py       # Multi-agent system core
├── alpha_vantage_client.py  # API client with caching
├── requirements.txt         # Python dependencies
└── README.md               # This file
```
