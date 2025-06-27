import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import warnings
import os
import mlx.core as mx
from mlx_lm import load, generate
import streamlit as st

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION AND DATA CLASSES
# ============================================================================

class AgentRole(Enum):
    FUNDAMENTAL = "fundamental"
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    TRENDS = "trends"
    NEWS = "news"
    SUPERVISOR = "supervisor"

@dataclass
class FinancialRatios:
    ticker: str
    year: str
    revenue: float
    market_cap: float
    ROA: float
    ROE: float
    net_profit_margin: float
    gross_margin: float
    current_ratio: float
    quick_ratio: float
    debt_to_equity: float
    interest_coverage: float = float('inf')
    
@dataclass
class TrendData:
    ticker: str
    current_price: float
    price_change_1d: float
    price_change_5d: float
    price_change_30d: float
    volume_avg_30d: float
    volume_current: float
    rsi: float
    moving_avg_20: float
    moving_avg_50: float
    volatility: float

@dataclass
class NewsData:
    ticker: str
    sentiment_score: float
    news_count: int
    recent_headlines: List[str]
    summary: str

@dataclass
class AgentAnalysis:
    agent_role: AgentRole
    analysis: str
    score: float
    recommendations: List[str]
    timestamp: datetime
    data: Optional[Any] = None

# ============================================================================
# MLX MODEL INTERFACE
# ============================================================================

class MLXClient:
    def __init__(self, model_path: str = "~/mlx-models"):
        """Initialize MLX client with your downloaded model"""
        self.model_path = os.path.expanduser(model_path)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    def load_model(self):
        """Load the MLX model if not already loaded"""
        if self.model is None:
            try:
                with st.spinner("Loading MLX DeepSeek model (this may take a moment)..."):
                    self.model, self.tokenizer = load(self.model_path)
                    self.is_loaded = True
                    st.success("✅ MLX Model loaded successfully!")
                return True
            except Exception as e:
                st.error(f"❌ Error loading MLX model: {str(e)}")
                self.is_loaded = False
                return False
        return True
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response using MLX"""
        if not self.is_loaded:
            if not self.load_model():
                return "Error: MLX model not available. Please check model path."
        
        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

# ============================================================================
# BASE AGENT CLASS
# ============================================================================

class BaseFinancialAgent:
    def __init__(self, mlx_client: MLXClient):
        """Initialize base agent with MLX client"""
        self.mlx = mlx_client
        self.agent_role = None
    
    def generate_analysis(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate analysis using MLX"""
        return self.mlx.generate(prompt, max_tokens)

# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class FundamentalAgent(BaseFinancialAgent):
    def __init__(self, mlx_client: MLXClient):
        super().__init__(mlx_client)
        self.agent_role = AgentRole.FUNDAMENTAL
    
    def get_financial_data(self, ticker: str) -> Optional[FinancialRatios]:
        """Extract financial data using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial statements
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            info = stock.info
            
            if financials.empty or balance_sheet.empty:
                return None
            
            # Get latest year data
            latest_year = financials.columns[0]
            
            # Extract financial data
            revenue = self._safe_get(financials, 'Total Revenue', latest_year, 0)
            net_income = self._safe_get(financials, 'Net Income', latest_year, 0)
            gross_profit = self._safe_get(financials, 'Gross Profit', latest_year, 0)
            
            total_assets = self._safe_get(balance_sheet, 'Total Assets', latest_year, 1)
            total_equity = self._safe_get(balance_sheet, 'Total Stockholder Equity', latest_year, 1)
            current_assets = self._safe_get(balance_sheet, 'Current Assets', latest_year, 0)
            current_liabilities = self._safe_get(balance_sheet, 'Current Liabilities', latest_year, 1)
            total_debt = self._safe_get(balance_sheet, 'Total Debt', latest_year, 0)
            
            # Get market cap from info
            market_cap = info.get('marketCap', 0) / 1e9 if info.get('marketCap') else 0
            
            # Calculate ratios
            ratios = FinancialRatios(
                ticker=ticker.upper(),
                year=str(latest_year)[:4],
                revenue=revenue / 1e9,  # In billions
                market_cap=market_cap,
                ROA=(net_income / total_assets) * 100 if total_assets != 0 else 0,
                ROE=(net_income / total_equity) * 100 if total_equity != 0 else 0,
                net_profit_margin=(net_income / revenue) * 100 if revenue != 0 else 0,
                gross_margin=(gross_profit / revenue) * 100 if revenue != 0 else 0,
                current_ratio=current_assets / current_liabilities if current_liabilities != 0 else 0,
                quick_ratio=(current_assets * 0.8) / current_liabilities if current_liabilities != 0 else 0,
                debt_to_equity=total_debt / total_equity if total_equity != 0 else 0,
            )
            
            return ratios
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def _safe_get(self, df, index_name, column_name, default):
        """Safely extract value from DataFrame"""
        try:
            if index_name in df.index:
                return df.loc[index_name, column_name]
            return default
        except:
            return default

class ProfitabilityAgent(BaseFinancialAgent):
    def __init__(self, mlx_client: MLXClient):
        super().__init__(mlx_client)
        self.agent_role = AgentRole.PROFITABILITY
    
    def analyze(self, ratios: FinancialRatios) -> AgentAnalysis:
        """Analyze profitability metrics"""
        prompt = f"""
You are a Senior Financial Analyst specializing in profitability analysis. Analyze {ratios.ticker} ({ratios.year}):

PROFITABILITY METRICS:
- Return on Assets (ROA): {ratios.ROA:.2f}%
- Return on Equity (ROE): {ratios.ROE:.2f}%
- Net Profit Margin: {ratios.net_profit_margin:.2f}%
- Gross Margin: {ratios.gross_margin:.2f}%
- Revenue: ${ratios.revenue:.1f} billion
- Market Cap: ${ratios.market_cap:.1f} billion

THRESHOLDS:
- ROA: Excellent >15%, Good >5%, Fair 2-5%, Poor <2%
- ROE: Excellent >20%, Good >15%, Fair 8-15%, Poor <8%
- Net Margin: Excellent >15%, Good >10%, Fair 5-10%, Poor <5%
- Gross Margin: Excellent >50%, Good >40%, Fair 20-40%, Poor <20%

Provide:
1. Score each metric (1-10) with reasoning
2. Overall profitability assessment
3. Competitive positioning analysis
4. 3 specific recommendations

Format clearly with sections and bullet points.
"""
        
        analysis_text = self.generate_analysis(prompt)
        
        # Calculate scores
        scores = []
        if ratios.ROA >= 15: scores.append(10)
        elif ratios.ROA >= 5: scores.append(8)
        elif ratios.ROA >= 2: scores.append(5)
        else: scores.append(2)
        
        if ratios.ROE >= 20: scores.append(10)
        elif ratios.ROE >= 15: scores.append(8)
        elif ratios.ROE >= 8: scores.append(5)
        else: scores.append(2)
        
        if ratios.net_profit_margin >= 15: scores.append(10)
        elif ratios.net_profit_margin >= 10: scores.append(8)
        elif ratios.net_profit_margin >= 5: scores.append(5)
        else: scores.append(2)
        
        if ratios.gross_margin >= 50: scores.append(10)
        elif ratios.gross_margin >= 40: scores.append(8)
        elif ratios.gross_margin >= 20: scores.append(5)
        else: scores.append(2)
        
        overall_score = sum(scores) / len(scores)
        
        return AgentAnalysis(
            agent_role=self.agent_role,
            analysis=analysis_text,
            score=overall_score,
            recommendations=[],
            timestamp=datetime.now(),
            data=ratios
        )

class LiquidityAgent(BaseFinancialAgent):
    def __init__(self, mlx_client: MLXClient):
        super().__init__(mlx_client)
        self.agent_role = AgentRole.LIQUIDITY
    
    def analyze(self, ratios: FinancialRatios) -> AgentAnalysis:
        """Analyze liquidity and solvency metrics"""
        prompt = f"""
You are a Senior Financial Analyst specializing in liquidity analysis. Analyze {ratios.ticker} ({ratios.year}):

LIQUIDITY METRICS:
- Current Ratio: {ratios.current_ratio:.2f}
- Quick Ratio: {ratios.quick_ratio:.2f}
- Debt-to-Equity: {ratios.debt_to_equity:.2f}
- Interest Coverage: {ratios.interest_coverage}

THRESHOLDS:
- Current Ratio: Excellent >2.5, Good 1.5-2.5, Warning 1.0-1.5, Poor <1.0
- Quick Ratio: Excellent >1.5, Good >1.0, Warning 0.5-1.0, Poor <0.5
- Debt-to-Equity: Conservative <0.5, Moderate 0.5-1.5, Aggressive 1.5-3.0, Risky >3.0
- Interest Coverage: Excellent >10, Good >5, Adequate >2.5, Poor <2.5

Provide:
1. Score each metric (1-10) with reasoning
2. Short-term financial health assessment
3. Debt management evaluation
4. Risk factors and recommendations

Format clearly with sections.
"""
        
        analysis_text = self.generate_analysis(prompt)
        
        # Calculate scores
        scores = []
        if ratios.current_ratio >= 2.5: scores.append(10)
        elif ratios.current_ratio >= 1.5: scores.append(8)
        elif ratios.current_ratio >= 1.0: scores.append(5)
        else: scores.append(2)
        
        if ratios.quick_ratio >= 1.5: scores.append(10)
        elif ratios.quick_ratio >= 1.0: scores.append(8)
        elif ratios.quick_ratio >= 0.5: scores.append(5)
        else: scores.append(2)
        
        if ratios.debt_to_equity <= 0.5: scores.append(10)
        elif ratios.debt_to_equity <= 1.5: scores.append(8)
        elif ratios.debt_to_equity <= 3.0: scores.append(5)
        else: scores.append(2)
        
        overall_score = sum(scores) / len(scores)
        
        return AgentAnalysis(
            agent_role=self.agent_role,
            analysis=analysis_text,
            score=overall_score,
            recommendations=[],
            timestamp=datetime.now(),
            data=ratios
        )

class TrendsAgent(BaseFinancialAgent):
    def __init__(self, mlx_client: MLXClient):
        super().__init__(mlx_client)
        self.agent_role = AgentRole.TRENDS
    
    def get_trend_data(self, ticker: str) -> Optional[TrendData]:
        """Get price and volume trend data"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get historical data (90 days for trend analysis)
            hist = stock.history(period="90d")
            info = stock.info
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            # Calculate price changes
            price_1d = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0
            price_5d = ((current_price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]) * 100 if len(hist) > 5 else 0
            price_30d = ((current_price - hist['Close'].iloc[-31]) / hist['Close'].iloc[-31]) * 100 if len(hist) > 30 else 0
            
            # Volume analysis
            volume_current = hist['Volume'].iloc[-1]
            volume_avg_30d = hist['Volume'].tail(30).mean()
            
            # Technical indicators
            rsi = self._calculate_rsi(hist['Close'])
            ma_20 = hist['Close'].tail(20).mean()
            ma_50 = hist['Close'].tail(50).mean()
            volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
            
            return TrendData(
                ticker=ticker.upper(),
                current_price=current_price,
                price_change_1d=price_1d,
                price_change_5d=price_5d,
                price_change_30d=price_30d,
                volume_avg_30d=volume_avg_30d,
                volume_current=volume_current,
                rsi=rsi,
                moving_avg_20=ma_20,
                moving_avg_50=ma_50,
                volatility=volatility
            )
            
        except Exception as e:
            print(f"Error fetching trend data for {ticker}: {str(e)}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50.0
        except:
            return 50.0
    
    def analyze(self, trend_data: TrendData) -> AgentAnalysis:
        """Analyze price and volume trends"""
        prompt = f"""
You are a Technical Analysis Specialist. Analyze {trend_data.ticker} trends:

PRICE MOVEMENTS:
- Current Price: ${trend_data.current_price:.2f}
- 1-Day Change: {trend_data.price_change_1d:+.2f}%
- 5-Day Change: {trend_data.price_change_5d:+.2f}%
- 30-Day Change: {trend_data.price_change_30d:+.2f}%

TECHNICAL INDICATORS:
- RSI: {trend_data.rsi:.1f} (70+ overbought, 30- oversold)
- 20-Day MA: ${trend_data.moving_avg_20:.2f}
- 50-Day MA: ${trend_data.moving_avg_50:.2f}
- Volatility: {trend_data.volatility:.1f}% (annualized)

VOLUME ANALYSIS:
- Current Volume: {trend_data.volume_current:,.0f}
- 30-Day Avg Volume: {trend_data.volume_avg_30d:,.0f}
- Volume Ratio: {trend_data.volume_current/trend_data.volume_avg_30d:.2f}x

Provide:
1. Technical trend assessment (bullish/bearish/neutral)
2. Support/resistance analysis based on moving averages
3. Volume confirmation analysis
4. Short-term outlook and key levels to watch

Format with clear sections.
"""
        
        analysis_text = self.generate_analysis(prompt)
        
        # Calculate trend score
        score = 5.0  # Base neutral score
        
        # Price momentum scoring
        if trend_data.price_change_30d > 10: score += 2
        elif trend_data.price_change_30d > 0: score += 1
        elif trend_data.price_change_30d < -10: score -= 2
        elif trend_data.price_change_30d < 0: score -= 1
        
        # RSI scoring
        if 30 <= trend_data.rsi <= 70: score += 1  # Healthy range
        elif trend_data.rsi > 80 or trend_data.rsi < 20: score -= 1  # Extreme levels
        
        # Moving average trend
        if trend_data.current_price > trend_data.moving_avg_20 > trend_data.moving_avg_50:
            score += 1  # Bullish alignment
        elif trend_data.current_price < trend_data.moving_avg_20 < trend_data.moving_avg_50:
            score -= 1  # Bearish alignment
        
        # Volatility penalty for high risk
        if trend_data.volatility > 50: score -= 0.5
        
        score = max(1, min(10, score))  # Clamp between 1-10
        
        return AgentAnalysis(
            agent_role=self.agent_role,
            analysis=analysis_text,
            score=score,
            recommendations=[],
            timestamp=datetime.now(),
            data=trend_data
        )

class NewsAgent(BaseFinancialAgent):
    def __init__(self, mlx_client: MLXClient):
        super().__init__(mlx_client)
        self.agent_role = AgentRole.NEWS
    
    def get_news_data(self, ticker: str) -> NewsData:
        """Get news sentiment data (placeholder for future Finnhub integration)"""
        # TODO: Integrate with Finnhub or other news APIs
        return NewsData(
            ticker=ticker.upper(),
            sentiment_score=0.0,  # Neutral for now
            news_count=0,
            recent_headlines=[],
            summary="News analysis will be available when API integration is added."
        )
    
    def analyze(self, news_data: NewsData) -> AgentAnalysis:
        """Analyze news sentiment and market impact"""
        analysis_text = f"""
## News Sentiment Analysis for {news_data.ticker}

**Current Status**: News analysis module is ready for integration.

**Planned Features**:
- Real-time news sentiment scoring
- Headline impact analysis  
- Social media sentiment tracking
- Earnings call transcript analysis
- SEC filing change detection

**Integration Ready For**:
- Finnhub API (free tier: 60 calls/min)
- Alpha Vantage News API
- NewsAPI.org
- Reddit/Twitter sentiment analysis

{news_data.summary}

**Next Steps**: Add API credentials to enable live news analysis.
"""
        
        return AgentAnalysis(
            agent_role=self.agent_role,
            analysis=analysis_text,
            score=5.0,  # Neutral score until implemented
            recommendations=["Add news API integration", "Configure sentiment analysis"],
            timestamp=datetime.now(),
            data=news_data
        )

class SupervisorAgent(BaseFinancialAgent):
    def __init__(self, mlx_client: MLXClient):
        super().__init__(mlx_client)
        self.agent_role = AgentRole.SUPERVISOR
    
    def synthesize_analysis(self, all_analyses: List[AgentAnalysis], 
                          ratios: FinancialRatios) -> AgentAnalysis:
        """Generate comprehensive executive summary"""
        
        # Extract individual scores
        prof_score = next((a.score for a in all_analyses if a.agent_role == AgentRole.PROFITABILITY), 0)
        liq_score = next((a.score for a in all_analyses if a.agent_role == AgentRole.LIQUIDITY), 0)
        trend_score = next((a.score for a in all_analyses if a.agent_role == AgentRole.TRENDS), 0)
        news_score = next((a.score for a in all_analyses if a.agent_role == AgentRole.NEWS), 0)
        
        prompt = f"""
You are the Chief Investment Officer providing an executive summary for {ratios.ticker} ({ratios.year}):

COMPANY OVERVIEW:
- Market Cap: ${ratios.market_cap:.1f}B
- Revenue: ${ratios.revenue:.1f}B
- Analysis Date: {datetime.now().strftime('%Y-%m-%d')}

SPECIALIST TEAM SCORES:
- Profitability Analyst: {prof_score:.1f}/10
- Liquidity Analyst: {liq_score:.1f}/10  
- Technical Analyst: {trend_score:.1f}/10
- News Analyst: {news_score:.1f}/10

KEY FINANCIAL METRICS:
- ROA: {ratios.ROA:.2f}% | ROE: {ratios.ROE:.2f}%
- Net Margin: {ratios.net_profit_margin:.2f}% | Gross Margin: {ratios.gross_margin:.2f}%
- Current Ratio: {ratios.current_ratio:.2f} | D/E Ratio: {ratios.debt_to_equity:.2f}

As Chief Investment Officer, provide:

## EXECUTIVE SUMMARY
(2-3 sentences on overall investment thesis)

## INVESTMENT GRADE
(A+ to F with explanation)

## KEY STRENGTHS
(Top 3 competitive advantages)

## KEY RISKS  
(Top 3 concerns to monitor)

## STRATEGIC RECOMMENDATIONS
(3 specific actionable recommendations)

## OUTLOOK
(12-month outlook: Bullish/Neutral/Bearish with rationale)

## INVESTMENT SUITABILITY
(Conservative/Moderate/Aggressive investor profile match)

Format as a professional investment memo.
"""
        
        analysis_text = self.generate_analysis(prompt, max_tokens=3000)
        
        # Calculate weighted overall score
        weights = {
            'profitability': 0.35,
            'liquidity': 0.25, 
            'trends': 0.25,
            'news': 0.15
        }
        
        overall_score = (
            prof_score * weights['profitability'] +
            liq_score * weights['liquidity'] + 
            trend_score * weights['trends'] +
            news_score * weights['news']
        )
        
        return AgentAnalysis(
            agent_role=self.agent_role,
            analysis=analysis_text,
            score=overall_score,
            recommendations=[],
            timestamp=datetime.now(),
            data={'individual_scores': {
                'profitability': prof_score,
                'liquidity': liq_score,
                'trends': trend_score,
                'news': news_score
            }}
        )

# ============================================================================
# MAIN AGENT ORCHESTRATOR
# ============================================================================

class FinancialAnalysisOrchestrator:
    def __init__(self, model_path: str = "~/mlx-models"):
        """Initialize the multi-agent system with MLX"""
        self.mlx = MLXClient(model_path)
        
        # Initialize all agents
        self.fundamental_agent = FundamentalAgent(self.mlx)
        self.profitability_agent = ProfitabilityAgent(self.mlx)
        self.liquidity_agent = LiquidityAgent(self.mlx)
        self.trends_agent = TrendsAgent(self.mlx)
        self.news_agent = NewsAgent(self.mlx)
        self.supervisor_agent = SupervisorAgent(self.mlx)
        
        self.analysis_history = []
    
    def run_full_analysis(self, ticker: str) -> Tuple[FinancialRatios, List[AgentAnalysis]]:
        """Run complete multi-agent analysis"""
        
        # Step 1: Get fundamental data
        ratios = self.fundamental_agent.get_financial_data(ticker)
        if not ratios:
            raise ValueError(f"Could not fetch financial data for {ticker}")
        
        # Step 2: Get trend data  
        trend_data = self.trends_agent.get_trend_data(ticker)
        if not trend_data:
            raise ValueError(f"Could not fetch trend data for {ticker}")
        
        # Step 3: Get news data
        news_data = self.news_agent.get_news_data(ticker)
        
        # Step 4: Run specialist analyses
        analyses = []
        
        # Profitability analysis
        prof_analysis = self.profitability_agent.analyze(ratios)
        analyses.append(prof_analysis)
        
        # Liquidity analysis
        liq_analysis = self.liquidity_agent.analyze(ratios)
        analyses.append(liq_analysis)
        
        # Trends analysis
        trend_analysis = self.trends_agent.analyze(trend_data)
        analyses.append(trend_analysis)
        
        # News analysis
        news_analysis = self.news_agent.analyze(news_data)
        analyses.append(news_analysis)
        
        # Step 5: Supervisor synthesis
        supervisor_analysis = self.supervisor_agent.synthesize_analysis(analyses, ratios)
        analyses.append(supervisor_analysis)
        
        # Store in history
        self.analysis_history.append({
            'ticker': ticker,
            'timestamp': datetime.now(),
            'ratios': ratios,
            'analyses': analyses
        })
        
        return ratios, analyses
    
    def is_available(self) -> bool:
        """Check if MLX is available"""
        return self.mlx.is_loaded or self.mlx.load_model()
    
    def get_model_status(self) -> str:
        """Get current model status"""
        if self.mlx.is_loaded:
            return "✅ MLX DeepSeek model ready"
        else:
            return "❌ MLX model not loaded. Loading will begin on first analysis."