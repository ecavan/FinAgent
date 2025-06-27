# 🤖 FinAgent - Multi-Agent Financial Analysis System

A free, local multi-agent financial analysis system powered by MLX (optimized for Apple Silicon) and yfinance for real-time financial data.

## 🚀 Features

### Multi-Agent Architecture

- 🔍 Fundamental Agent: Extracts financial ratios using free data sources
- 💰 Profitability Agent: Analyzes ROA, ROE, profit margins with scoring
- 💧 Liquidity Agent: Evaluates current ratio, quick ratio, debt-to-equity
- 👔 Supervisor Agent: Generates comprehensive executive reports

## Key Benefits

- ✅ 100% Free: No API costs (uses yfinance + local MLX)
- 🔒 Private: All analysis runs locally on your machine
- ⚡ Fast: MLX-optimized for Apple Silicon
- 📊 Visual: Interactive charts and professional reports
- 📱 User-Friendly: Streamlit web interface



Ensure your MLX model is in ~/mlx-models (or update the path in main.py)
If you need to download a model:
```
huggingface-cli download mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit --local-dir ~/mlx-models 
```


## 📈 Sample Analysis Output

EXECUTIVE SUMMARY
Apple (AAPL) demonstrates exceptional profitability with industry-leading margins 
and strong asset utilization. However, liquidity metrics suggest potential 
short-term constraints requiring management attention.

OVERALL FINANCIAL HEALTH RATING: A-
PROFITABILITY SCORE: 9.2/10
LIQUIDITY SCORE: 6.8/10

KEY RECOMMENDATIONS:
1. Monitor current ratio improvement opportunities
2. Leverage strong profitability for growth investments  
3. Consider debt structure optimization

## To run

```
streamlit run main.py
```