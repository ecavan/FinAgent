"""
Enhanced Multi-Agent Financial Analysis System - FIXED API STRATEGY
PRIMARY: Yahoo Finance (unlimited) | SECONDARY: Alpha Vantage (cached, optional)
"""

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

# Import our Alpha Vantage client (optional use only)
try:
    from alpha_vantage_client import AlphaVantageClient, NewsAnalysis, EconomicIndicators
    ALPHA_VANTAGE_AVAILABLE = True
except:
    ALPHA_VANTAGE_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================================
# ENHANCED CONFIGURATION AND DATA CLASSES
# ============================================================================

class AgentRole(Enum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    NEWS_SENTIMENT = "news_sentiment"
    ECONOMIC_CLIMATE = "economic_climate"
    SUPERVISOR = "supervisor"

@dataclass
class EnhancedFinancialRatios:
    """Comprehensive financial data primarily from Yahoo Finance"""
    ticker: str
    fiscal_year: str
    
    # Revenue and Profitability
    total_revenue: float
    gross_profit: float
    operating_income: float
    net_income: float
    ebitda: float
    eps: float
    
    # Profitability Ratios
    gross_margin: float
    operating_margin: float
    net_profit_margin: float
    return_on_assets: float
    return_on_equity: float
    return_on_invested_capital: float
    
    # Liquidity Ratios
    current_ratio: float
    quick_ratio: float
    cash_ratio: float
    
    # Leverage Ratios
    debt_to_equity: float
    debt_to_assets: float
    interest_coverage_ratio: float
    debt_service_coverage: float
    
    # Efficiency Ratios
    asset_turnover: float
    inventory_turnover: float
    receivables_turnover: float
    
    # Valuation Ratios
    price_to_earnings: float
    price_to_book: float
    price_to_sales: float
    price_to_cash_flow: float
    enterprise_value_to_ebitda: float
    
    # Cash Flow
    operating_cash_flow: float
    free_cash_flow: float
    free_cash_flow_yield: float
    
    # Market Data
    market_cap: float
    enterprise_value: float
    shares_outstanding: float
    
    # Additional metrics
    working_capital: float
    book_value_per_share: float
    tangible_book_value: float

@dataclass
class TechnicalData:
    """Technical analysis data from Yahoo Finance"""
    ticker: str
    current_price: float
    high_52_week: float
    low_52_week: float
    
    # Price changes
    price_change_1d: float
    price_change_5d: float
    price_change_30d: float
    
    # Moving averages
    sma_20: float
    sma_50: float
    sma_200: float
    
    # Technical indicators
    rsi: float
    bollinger_upper: float
    bollinger_lower: float
    
    # Volume
    volume_current: float
    volume_avg_30d: float
    volume_ratio: float
    
    # Volatility
    volatility: float

@dataclass
class AgentAnalysis:
    agent_role: AgentRole
    analysis: str
    score: float
    confidence: float
    recommendations: List[str]
    key_metrics: Dict[str, Any]
    timestamp: datetime
    data: Optional[Any] = None

# ============================================================================
# HELPER FUNCTIONS FOR SAFE CALCULATIONS
# ============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        result = numerator / denominator
        return result if not (pd.isna(result) or np.isinf(result)) else default
    except:
        return default

def safe_percentage(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely calculate percentage, returning default if denominator is zero"""
    return safe_divide(numerator, denominator, default) * 100

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        if value is None or pd.isna(value):
            return default
        result = float(value)
        return result if not (pd.isna(result) or np.isinf(result)) else default
    except:
        return default

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI using price series"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return safe_float(rsi.iloc[-1], 50.0)
    except:
        return 50.0

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
        self.generation_cache = {}
    
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
    
    def generate(self, prompt: str, max_tokens: int = 2500) -> str:
        """Generate response using MLX without temperature parameter"""
        if not self.is_loaded:
            if not self.load_model():
                return "Error: MLX model not available. Please check model path."
        
        # Simple caching based on prompt hash
        prompt_hash = hash(prompt[:200])
        if prompt_hash in self.generation_cache:
            return self.generation_cache[prompt_hash]
        
        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens
            )
            
            # Cache the response
            self.generation_cache[prompt_hash] = response
            
            # Limit cache size
            if len(self.generation_cache) > 50:
                oldest_key = next(iter(self.generation_cache))
                del self.generation_cache[oldest_key]
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

# ============================================================================
# BASE AGENT CLASS
# ============================================================================

class BaseFinancialAgent:
    def __init__(self, mlx_client: MLXClient, alpha_vantage_client = None):
        """Initialize base agent with MLX and optional Alpha Vantage clients"""
        self.mlx = mlx_client
        self.av = alpha_vantage_client
        self.agent_role = None
        self.confidence_weights = {
            'data_quality': 0.3,
            'analysis_depth': 0.25,
            'market_conditions': 0.2,
            'historical_accuracy': 0.25
        }
    
    def generate_analysis(self, prompt: str, max_tokens: int = 2500) -> str:
        """Generate analysis using MLX"""
        return self.mlx.generate(prompt, max_tokens)
    
    def calculate_confidence(self, **factors) -> float:
        """Calculate confidence score based on various factors"""
        confidence = 0.0
        for factor, weight in self.confidence_weights.items():
            confidence += factors.get(factor, 0.5) * weight
        return min(max(confidence, 0.0), 1.0)

# ============================================================================
# ENHANCED SPECIALIZED AGENTS
# ============================================================================

class FundamentalAnalysisAgent(BaseFinancialAgent):
    def __init__(self, mlx_client: MLXClient, alpha_vantage_client = None):
        super().__init__(mlx_client, alpha_vantage_client)
        self.agent_role = AgentRole.FUNDAMENTAL
    
    def get_comprehensive_data(self, ticker: str) -> Optional[EnhancedFinancialRatios]:
        """Get comprehensive financial data primarily from Yahoo Finance"""
        try:
            # Get Yahoo Finance data (primary source)
            stock = yf.Ticker(ticker)
            info = stock.info
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            
            if financials.empty or balance_sheet.empty:
                st.warning(f"Could not fetch complete financial data for {ticker}")
                return None
            
            # Get latest year data
            latest_year = financials.columns[0]
            
            # Safely extract financial statement data
            total_revenue = safe_float(self._safe_get(financials, 'Total Revenue', latest_year))
            gross_profit = safe_float(self._safe_get(financials, 'Gross Profit', latest_year))
            operating_income = safe_float(self._safe_get(financials, 'Operating Income', latest_year))
            net_income = safe_float(self._safe_get(financials, 'Net Income', latest_year))
            ebitda = safe_float(self._safe_get(financials, 'EBITDA', latest_year))
            
            total_assets = safe_float(self._safe_get(balance_sheet, 'Total Assets', latest_year))
            shareholders_equity = safe_float(self._safe_get(balance_sheet, 'Total Stockholder Equity', latest_year))
            current_assets = safe_float(self._safe_get(balance_sheet, 'Current Assets', latest_year))
            current_liabilities = safe_float(self._safe_get(balance_sheet, 'Current Liabilities', latest_year), 1)
            long_term_debt = safe_float(self._safe_get(balance_sheet, 'Long Term Debt', latest_year))
            cash_and_equivalents = safe_float(self._safe_get(balance_sheet, 'Cash And Cash Equivalents', latest_year))
            
            operating_cash_flow = safe_float(self._safe_get(cashflow, 'Operating Cash Flow', latest_year))
            capital_expenditures = safe_float(self._safe_get(cashflow, 'Capital Expenditures', latest_year))
            
            # Get market data from info
            market_cap = safe_float(info.get('marketCap', 0))
            shares_outstanding = safe_float(info.get('sharesOutstanding', 1))
            enterprise_value = safe_float(info.get('enterpriseValue', 0))
            pe_ratio = safe_float(info.get('trailingPE', 0))
            pb_ratio = safe_float(info.get('priceToBook', 0))
            eps = safe_float(info.get('trailingEps', 0))
            
            # Calculate ratios with safe division
            gross_margin = safe_percentage(gross_profit, total_revenue)
            operating_margin = safe_percentage(operating_income, total_revenue)
            net_profit_margin = safe_percentage(net_income, total_revenue)
            return_on_assets = safe_percentage(net_income, total_assets)
            return_on_equity = safe_percentage(net_income, shareholders_equity)
            
            # ROIC calculation
            invested_capital = shareholders_equity + long_term_debt
            roic = safe_percentage(operating_income, invested_capital)
            
            # Liquidity ratios
            current_ratio = safe_divide(current_assets, current_liabilities, 0)
            quick_ratio = safe_divide(current_assets * 0.8, current_liabilities, 0)  # Approximation
            cash_ratio = safe_divide(cash_and_equivalents, current_liabilities, 0)
            
            # Leverage ratios
            debt_to_equity = safe_divide(long_term_debt, shareholders_equity, 0)
            debt_to_assets = safe_percentage(long_term_debt, total_assets)
            
            # Interest coverage (estimate)
            estimated_interest = long_term_debt * 0.05  # Assume 5% rate
            interest_coverage_ratio = safe_divide(operating_income, estimated_interest, float('inf'))
            debt_service_coverage = safe_divide(operating_cash_flow, long_term_debt, float('inf'))
            
            # Efficiency ratios
            asset_turnover = safe_divide(total_revenue, total_assets, 0)
            
            # Valuation ratios
            price_to_sales = safe_divide(market_cap, total_revenue, 0)
            price_to_cash_flow = safe_divide(market_cap, operating_cash_flow, 0)
            ev_to_ebitda = safe_divide(enterprise_value, ebitda, 0)
            
            # Cash flow calculations
            free_cash_flow = operating_cash_flow - abs(capital_expenditures)
            free_cash_flow_yield = safe_percentage(free_cash_flow, market_cap)
            
            # Additional metrics
            working_capital = current_assets - current_liabilities
            book_value_per_share = safe_divide(shareholders_equity, shares_outstanding, 0)
            
            return EnhancedFinancialRatios(
                ticker=ticker.upper(),
                fiscal_year=str(latest_year)[:4],
                
                # Revenue and Profitability (in billions)
                total_revenue=total_revenue / 1e9,
                gross_profit=gross_profit / 1e9,
                operating_income=operating_income / 1e9,
                net_income=net_income / 1e9,
                ebitda=ebitda / 1e9,
                eps=eps,
                
                # Profitability Ratios
                gross_margin=gross_margin,
                operating_margin=operating_margin,
                net_profit_margin=net_profit_margin,
                return_on_assets=return_on_assets,
                return_on_equity=return_on_equity,
                return_on_invested_capital=roic,
                
                # Liquidity Ratios
                current_ratio=current_ratio,
                quick_ratio=quick_ratio,
                cash_ratio=cash_ratio,
                
                # Leverage Ratios
                debt_to_equity=debt_to_equity,
                debt_to_assets=debt_to_assets,
                interest_coverage_ratio=min(interest_coverage_ratio, 999),
                debt_service_coverage=min(debt_service_coverage, 999),
                
                # Efficiency Ratios
                asset_turnover=asset_turnover,
                inventory_turnover=0,  # Would need more detailed data
                receivables_turnover=0,
                
                # Valuation Ratios
                price_to_earnings=pe_ratio,
                price_to_book=pb_ratio,
                price_to_sales=price_to_sales,
                price_to_cash_flow=price_to_cash_flow,
                enterprise_value_to_ebitda=ev_to_ebitda,
                
                # Cash Flow (in billions)
                operating_cash_flow=operating_cash_flow / 1e9,
                free_cash_flow=free_cash_flow / 1e9,
                free_cash_flow_yield=free_cash_flow_yield,
                
                # Market Data (in billions)
                market_cap=market_cap / 1e9,
                enterprise_value=enterprise_value / 1e9,
                shares_outstanding=shares_outstanding / 1e6,  # In millions
                
                # Additional metrics
                working_capital=working_capital / 1e9,
                book_value_per_share=book_value_per_share,
                tangible_book_value=shareholders_equity / 1e9
            )
            
        except Exception as e:
            st.error(f"Error fetching fundamental data for {ticker}: {str(e)}")
            return None
    
    def _safe_get(self, df, index_name, column_name, default=0):
        """Safely extract value from DataFrame"""
        try:
            if index_name in df.index:
                return df.loc[index_name, column_name]
            return default
        except:
            return default
    
    def analyze(self, ratios: EnhancedFinancialRatios) -> AgentAnalysis:
        """Comprehensive fundamental analysis"""
        
        prompt = f"""
You are a Senior Fundamental Analyst with CFA designation. Perform comprehensive analysis for {ratios.ticker} (FY {ratios.fiscal_year}):

REVENUE & PROFITABILITY ANALYSIS:
- Total Revenue: ${ratios.total_revenue:.1f}B
- Gross Profit: ${ratios.gross_profit:.1f}B (Margin: {ratios.gross_margin:.1f}%)
- Operating Income: ${ratios.operating_income:.1f}B (Margin: {ratios.operating_margin:.1f}%)
- Net Income: ${ratios.net_income:.1f}B (Margin: {ratios.net_profit_margin:.1f}%)
- EBITDA: ${ratios.ebitda:.1f}B
- EPS: ${ratios.eps:.2f}

RETURNS ANALYSIS:
- Return on Assets (ROA): {ratios.return_on_assets:.2f}%
- Return on Equity (ROE): {ratios.return_on_equity:.2f}%
- Return on Invested Capital (ROIC): {ratios.return_on_invested_capital:.2f}%

FINANCIAL HEALTH:
- Current Ratio: {ratios.current_ratio:.2f}
- Quick Ratio: {ratios.quick_ratio:.2f}
- Debt-to-Equity: {ratios.debt_to_equity:.2f}
- Debt-to-Assets: {ratios.debt_to_assets:.1f}%

CASH FLOW ANALYSIS:
- Operating Cash Flow: ${ratios.operating_cash_flow:.1f}B
- Free Cash Flow: ${ratios.free_cash_flow:.1f}B
- FCF Yield: {ratios.free_cash_flow_yield:.1f}%

VALUATION METRICS:
- P/E Ratio: {ratios.price_to_earnings:.1f}x
- P/B Ratio: {ratios.price_to_book:.1f}x
- P/S Ratio: {ratios.price_to_sales:.1f}x

## FUNDAMENTAL STRENGTH ASSESSMENT
Rate the overall fundamental strength as Excellent/Good/Fair/Poor with detailed reasoning.

## PROFITABILITY ANALYSIS
Analyze revenue quality, margin sustainability, and return metrics vs industry standards.

## FINANCIAL STABILITY
Evaluate liquidity position, debt management, and cash generation capability.

## VALUATION ASSESSMENT
Compare current valuation multiples to historical averages and fair value estimates.

## KEY RISKS & OPPORTUNITIES
Identify the top 3 fundamental risks and top 3 growth opportunities.

## INVESTMENT RECOMMENDATION
Provide Buy/Hold/Sell recommendation with target timeframe.

Format as professional equity research report.
"""
        
        analysis_text = self.generate_analysis(prompt, max_tokens=3000)
        
        # Calculate fundamental score (1-10)
        try:
            score_components = []
            
            # Profitability scoring (40% weight)
            prof_score = 0
            if ratios.return_on_equity >= 20: prof_score += 3
            elif ratios.return_on_equity >= 15: prof_score += 2.5
            elif ratios.return_on_equity >= 8: prof_score += 1.5
            
            if ratios.return_on_assets >= 15: prof_score += 2
            elif ratios.return_on_assets >= 5: prof_score += 1.5
            elif ratios.return_on_assets >= 2: prof_score += 1
            
            if ratios.net_profit_margin >= 15: prof_score += 2
            elif ratios.net_profit_margin >= 10: prof_score += 1.5
            elif ratios.net_profit_margin >= 5: prof_score += 1
            
            score_components.append(min(prof_score, 4.0) * 2.5)
            
            # Financial Health scoring (30% weight)
            health_score = 0
            if 1.5 <= ratios.current_ratio <= 3.0: health_score += 2
            elif ratios.current_ratio >= 1.0: health_score += 1
            
            if ratios.quick_ratio >= 1.0: health_score += 1.5
            elif ratios.quick_ratio >= 0.5: health_score += 1
            
            if ratios.debt_to_equity <= 0.5: health_score += 2
            elif ratios.debt_to_equity <= 1.5: health_score += 1.5
            elif ratios.debt_to_equity <= 3.0: health_score += 1
            
            if ratios.free_cash_flow > 0: health_score += 1.5
            
            score_components.append(min(health_score, 3.0) * 3.33)
            
            # Valuation scoring (20% weight)
            val_score = 5.0  # Base neutral
            score_components.append(val_score * 2.0)
            
            # Growth Quality scoring (10% weight)
            growth_score = 5.0  # Base neutral
            score_components.append(growth_score * 1.0)
            
            overall_score = sum(score_components) / 10
            overall_score = max(1.0, min(10.0, overall_score))
            
        except Exception as e:
            st.warning(f"Error calculating fundamental score: {str(e)}")
            overall_score = 5.0
        
        confidence = self.calculate_confidence(
            data_quality=0.9 if ratios.total_revenue > 0 else 0.3,
            analysis_depth=0.8,
            market_conditions=0.7,
            historical_accuracy=0.7
        )
        
        key_metrics = {
            'roe': ratios.return_on_equity,
            'roa': ratios.return_on_assets,
            'current_ratio': ratios.current_ratio,
            'debt_to_equity': ratios.debt_to_equity,
            'free_cash_flow': ratios.free_cash_flow,
            'pe_ratio': ratios.price_to_earnings
        }
        
        recommendations = [
            f"Monitor ROE trend (Current: {ratios.return_on_equity:.1f}%)",
            f"Assess debt levels (D/E: {ratios.debt_to_equity:.2f})",
            f"Evaluate cash generation (FCF: ${ratios.free_cash_flow:.1f}B)"
        ]
        
        return AgentAnalysis(
            agent_role=self.agent_role,
            analysis=analysis_text,
            score=overall_score,
            confidence=confidence,
            recommendations=recommendations,
            key_metrics=key_metrics,
            timestamp=datetime.now(),
            data=ratios
        )

class TechnicalAnalysisAgent(BaseFinancialAgent):
    def __init__(self, mlx_client: MLXClient, alpha_vantage_client = None):
        super().__init__(mlx_client, alpha_vantage_client)
        self.agent_role = AgentRole.TECHNICAL
    
    def get_technical_data(self, ticker: str) -> Optional[TechnicalData]:
        """Get technical data from Yahoo Finance (unlimited)"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get historical data (90 days for indicators)
            hist = stock.history(period="90d")
            info = stock.info
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            # Price changes
            price_1d = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0
            price_5d = ((current_price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]) * 100 if len(hist) > 5 else 0
            price_30d = ((current_price - hist['Close'].iloc[-31]) / hist['Close'].iloc[-31]) * 100 if len(hist) > 30 else 0
            
            # 52-week high/low
            high_52_week = safe_float(info.get('fiftyTwoWeekHigh', current_price * 1.2))
            low_52_week = safe_float(info.get('fiftyTwoWeekLow', current_price * 0.8))
            
            # Moving averages
            sma_20 = hist['Close'].tail(20).mean()
            sma_50 = hist['Close'].tail(50).mean() if len(hist) >= 50 else sma_20
            sma_200 = hist['Close'].tail(200).mean() if len(hist) >= 200 else sma_50
            
            # RSI calculation
            rsi = calculate_rsi(hist['Close'])
            
            # Bollinger Bands (20-day SMA ± 2 std dev)
            rolling_mean = hist['Close'].rolling(window=20).mean()
            rolling_std = hist['Close'].rolling(window=20).std()
            bollinger_upper = (rolling_mean + (rolling_std * 2)).iloc[-1] if len(rolling_mean) > 0 else current_price * 1.02
            bollinger_lower = (rolling_mean - (rolling_std * 2)).iloc[-1] if len(rolling_mean) > 0 else current_price * 0.98
            
            # Volume analysis
            volume_current = hist['Volume'].iloc[-1]
            volume_avg_30d = hist['Volume'].tail(30).mean()
            volume_ratio = safe_divide(volume_current, volume_avg_30d, 1.0)
            
            # Volatility (30-day)
            volatility = hist['Close'].pct_change().tail(30).std() * np.sqrt(252) * 100  # Annualized
            
            return TechnicalData(
                ticker=ticker.upper(),
                current_price=current_price,
                high_52_week=high_52_week,
                low_52_week=low_52_week,
                price_change_1d=price_1d,
                price_change_5d=price_5d,
                price_change_30d=price_30d,
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                rsi=rsi,
                bollinger_upper=bollinger_upper,
                bollinger_lower=bollinger_lower,
                volume_current=volume_current,
                volume_avg_30d=volume_avg_30d,
                volume_ratio=volume_ratio,
                volatility=volatility
            )
            
        except Exception as e:
            st.error(f"Error fetching technical data for {ticker}: {str(e)}")
            return None
    
    def analyze(self, ticker: str) -> AgentAnalysis:
        """Technical analysis using Yahoo Finance data"""
        
        tech_data = self.get_technical_data(ticker)
        
        if not tech_data:
            return AgentAnalysis(
                agent_role=self.agent_role,
                analysis="Technical analysis data unavailable - please check ticker symbol",
                score=5.0,
                confidence=0.1,
                recommendations=["Verify ticker symbol and retry"],
                key_metrics={},
                timestamp=datetime.now()
            )
        
        prompt = f"""
You are a Professional Technical Analyst (CMT) analyzing {ticker}:

PRICE ACTION:
- Current Price: ${tech_data.current_price:.2f}
- 52-Week High: ${tech_data.high_52_week:.2f}
- 52-Week Low: ${tech_data.low_52_week:.2f}
- 1-Day Change: {tech_data.price_change_1d:+.2f}%
- 5-Day Change: {tech_data.price_change_5d:+.2f}%
- 30-Day Change: {tech_data.price_change_30d:+.2f}%

MOVING AVERAGES:
- SMA 20: ${tech_data.sma_20:.2f}
- SMA 50: ${tech_data.sma_50:.2f}
- SMA 200: ${tech_data.sma_200:.2f}

TECHNICAL INDICATORS:
- RSI (14): {tech_data.rsi:.1f}
- Bollinger Upper: ${tech_data.bollinger_upper:.2f}
- Bollinger Lower: ${tech_data.bollinger_lower:.2f}
- Volatility (30d): {tech_data.volatility:.1f}%

VOLUME ANALYSIS:
- Current Volume: {tech_data.volume_current:,.0f}
- 30-Day Avg Volume: {tech_data.volume_avg_30d:,.0f}
- Volume Ratio: {tech_data.volume_ratio:.2f}x

## TREND ANALYSIS
Analyze primary trend direction using moving average alignment and price action.

## MOMENTUM INDICATORS
Evaluate RSI levels, potential overbought/oversold conditions, and momentum shifts.

## SUPPORT & RESISTANCE
Identify key support/resistance levels using moving averages and Bollinger Bands.

## VOLUME CONFIRMATION
Assess volume patterns and their confirmation of price movements.

## TECHNICAL OUTLOOK
Provide short-term trading signals, key levels to watch, and risk management guidance.

## PROBABILITY ASSESSMENT
Estimate probability of bullish vs bearish outcomes over next 1-3 months.

Format as professional technical analysis report.
"""
        
        analysis_text = self.generate_analysis(prompt, max_tokens=2500)
        
        # Calculate technical score
        try:
            score = 5.0  # Base neutral
            
            # RSI scoring
            if 30 <= tech_data.rsi <= 70:
                score += 1  # Healthy range
            elif tech_data.rsi > 80 or tech_data.rsi < 20:
                score -= 1  # Extreme levels
            
            # Moving average alignment
            if tech_data.current_price > tech_data.sma_20 > tech_data.sma_50:
                score += 1.5  # Bullish alignment
            elif tech_data.current_price < tech_data.sma_20 < tech_data.sma_50:
                score -= 1.5  # Bearish alignment
            
            # Momentum scoring
            if tech_data.price_change_30d > 10:
                score += 1  # Strong uptrend
            elif tech_data.price_change_30d < -10:
                score -= 1  # Strong downtrend
            
            # Volume confirmation
            if tech_data.volume_ratio > 1.2:
                score += 0.5  # Above average volume
            elif tech_data.volume_ratio < 0.8:
                score -= 0.3  # Below average volume
            
            # Position within 52-week range
            week_52_position = (tech_data.current_price - tech_data.low_52_week) / (tech_data.high_52_week - tech_data.low_52_week)
            if week_52_position > 0.8:
                score += 0.5  # Near highs
            elif week_52_position < 0.2:
                score -= 0.5  # Near lows
            
            score = max(1.0, min(10.0, score))
            
        except Exception as e:
            st.warning(f"Error calculating technical score: {str(e)}")
            score = 5.0
        
        confidence = self.calculate_confidence(
            data_quality=0.9 if tech_data.current_price > 0 else 0.2,
            analysis_depth=0.9,
            market_conditions=0.8,
            historical_accuracy=0.8
        )
        
        key_metrics = {
            'rsi': tech_data.rsi,
            'trend': 'Bullish' if tech_data.current_price > tech_data.sma_20 else 'Bearish',
            'volume_ratio': tech_data.volume_ratio,
            'current_price': tech_data.current_price,
            'price_change_30d': tech_data.price_change_30d
        }
        
        recommendations = [
            f"Monitor RSI levels (Current: {tech_data.rsi:.1f})",
            f"Key support: ${tech_data.sma_20:.2f} (20-day SMA)",
            f"Volume trend: {tech_data.volume_ratio:.2f}x average"
        ]
        
        return AgentAnalysis(
            agent_role=self.agent_role,
            analysis=analysis_text,
            score=score,
            confidence=confidence,
            recommendations=recommendations,
            key_metrics=key_metrics,
            timestamp=datetime.now(),
            data=tech_data
        )

class NewsAndSentimentAgent(BaseFinancialAgent):
    def __init__(self, mlx_client: MLXClient, alpha_vantage_client = None):
        super().__init__(mlx_client, alpha_vantage_client)
        self.agent_role = AgentRole.NEWS_SENTIMENT
    
    def analyze(self, ticker: str) -> AgentAnalysis:
        """News sentiment analysis with Alpha Vantage fallback"""
        
        # Try Alpha Vantage if available and cached
        sentiment_data = None
        if self.av and ALPHA_VANTAGE_AVAILABLE:
            try:
                sentiment_data = self.av.get_news_sentiment(ticker)
            except:
                pass  # Fallback to basic analysis
        
        if sentiment_data:
            # Use Alpha Vantage data
            overall_sentiment = safe_float(sentiment_data.overall_sentiment)
            news_count = safe_float(sentiment_data.news_count)
            
            prompt = f"""
You are a Market Sentiment Analyst for {ticker}:

SENTIMENT DATA:
- Overall Sentiment: {overall_sentiment:.3f} (-1.0 to +1.0)
- Sentiment Label: {sentiment_data.sentiment_label}
- News Articles: {news_count}
- Relevance Score: {sentiment_data.relevance_score:.3f}

RECENT HEADLINES:
{chr(10).join(f"• {headline}" for headline in sentiment_data.recent_headlines[:5])}

Analyze market sentiment impact and provide trading implications.
"""
            
            # Convert sentiment (-1 to +1) to score (1-10)
            if overall_sentiment > 0.3:
                score = 8.5 + (overall_sentiment - 0.3) * 2.14
            elif overall_sentiment > 0.15:
                score = 7.0 + (overall_sentiment - 0.15) * 10
            elif overall_sentiment > -0.15:
                score = 4.0 + (overall_sentiment + 0.15) * 10
            else:
                score = max(1.0, 4.0 + overall_sentiment * 5)
            
            confidence = 0.7
            key_metrics = {
                'sentiment_score': overall_sentiment,
                'sentiment_label': sentiment_data.sentiment_label,
                'news_count': news_count
            }
            
        else:
            # Fallback to basic sentiment analysis
            prompt = f"""
You are a Market Sentiment Analyst analyzing {ticker}:

ANALYSIS FRAMEWORK:
- Recent market trends and sector performance
- General market sentiment indicators
- Sector-specific news themes
- Social media and retail investor sentiment

Without access to real-time news data, provide general sentiment analysis based on:
1. Current market environment assessment
2. Sector trends affecting {ticker}
3. Typical sentiment drivers for this type of stock
4. Risk factors to monitor

Provide professional sentiment assessment with caveats about data limitations.
"""
            
            score = 5.0  # Neutral without data
            confidence = 0.3  # Low confidence without real data
            key_metrics = {
                'sentiment_score': 0.0,
                'sentiment_label': 'Neutral (No Data)',
                'news_count': 0
            }
        
        analysis_text = self.generate_analysis(prompt, max_tokens=2000)
        
        recommendations = [
            "Monitor news developments closely",
            "Watch for sentiment extreme reversals",
            "Consider contrarian opportunities"
        ]
        
        return AgentAnalysis(
            agent_role=self.agent_role,
            analysis=analysis_text,
            score=max(1.0, min(10.0, score)),
            confidence=confidence,
            recommendations=recommendations,
            key_metrics=key_metrics,
            timestamp=datetime.now(),
            data=sentiment_data
        )

class EconomicClimateAgent(BaseFinancialAgent):
    def __init__(self, mlx_client: MLXClient, alpha_vantage_client = None):
        super().__init__(mlx_client, alpha_vantage_client)
        self.agent_role = AgentRole.ECONOMIC_CLIMATE
    
    def analyze(self) -> AgentAnalysis:
        """Economic climate analysis with Alpha Vantage fallback"""
        
        # Try Alpha Vantage if available and cached
        econ_data = None
        if self.av and ALPHA_VANTAGE_AVAILABLE:
            try:
                econ_data = self.av.get_economic_indicators()
            except:
                pass  # Fallback to general analysis
        
        if econ_data:
            # Use Alpha Vantage economic data
            prompt = f"""
You are a Senior Macroeconomic Analyst:

CURRENT ECONOMIC INDICATORS:
- GDP Growth: {econ_data.gdp_growth_rate:.2f}%
- Unemployment: {econ_data.unemployment_rate:.1f}%
- CPI Inflation: {econ_data.cpi_yoy_change:.2f}%
- Fed Funds Rate: {econ_data.fed_funds_rate:.2f}%

Provide comprehensive economic climate assessment and market implications.
"""
            
            # Score based on economic indicators
            score = 5.0
            if econ_data.gdp_growth_rate > 3.0: score += 1.5
            elif econ_data.gdp_growth_rate < 0: score -= 2.0
            
            if econ_data.unemployment_rate < 4.0: score += 1.0
            elif econ_data.unemployment_rate > 8.0: score -= 2.0
            
            inflation_deviation = abs(econ_data.cpi_yoy_change - 2.0)
            if inflation_deviation < 1.0: score += 1.0
            elif inflation_deviation > 4.0: score -= 2.0
            
            confidence = 0.8
            key_metrics = {
                'gdp_growth': econ_data.gdp_growth_rate,
                'unemployment': econ_data.unemployment_rate,
                'inflation': econ_data.cpi_yoy_change,
                'fed_funds_rate': econ_data.fed_funds_rate
            }
            
        else:
            # Fallback to general economic analysis
            prompt = """
You are a Senior Macroeconomic Analyst providing general market climate assessment:

ANALYSIS FRAMEWORK:
- Current economic cycle phase assessment
- Federal Reserve policy stance and trajectory
- Inflation environment and trends
- Employment market conditions
- Global economic factors

Without access to real-time economic data, provide general assessment based on:
1. Typical economic cycle patterns
2. Historical precedents
3. General market environment
4. Key risks and opportunities

Provide professional economic outlook with caveats about data limitations.
"""
            
            score = 5.0  # Neutral without data
            confidence = 0.4  # Low confidence without real data
            key_metrics = {
                'gdp_growth': 0.0,
                'unemployment': 0.0,
                'inflation': 0.0,
                'fed_funds_rate': 0.0,
                'economic_phase': 'Unknown (No Data)'
            }
        
        analysis_text = self.generate_analysis(prompt, max_tokens=2500)
        
        recommendations = [
            "Monitor Fed policy announcements",
            "Watch inflation trajectory",
            "Track employment data releases"
        ]
        
        return AgentAnalysis(
            agent_role=self.agent_role,
            analysis=analysis_text,
            score=max(1.0, min(10.0, score)),
            confidence=confidence,
            recommendations=recommendations,
            key_metrics=key_metrics,
            timestamp=datetime.now(),
            data=econ_data
        )

class SupervisorAgent(BaseFinancialAgent):
    def __init__(self, mlx_client: MLXClient, alpha_vantage_client = None):
        super().__init__(mlx_client, alpha_vantage_client)
        self.agent_role = AgentRole.SUPERVISOR
    
    def synthesize_analysis(self, all_analyses: List[AgentAnalysis], 
                          fundamental_data: EnhancedFinancialRatios) -> AgentAnalysis:
        """Generate comprehensive executive synthesis"""
        
        try:
            # Extract individual analyses with safe defaults
            fundamental_analysis = next((a for a in all_analyses if a.agent_role == AgentRole.FUNDAMENTAL), None)
            technical_analysis = next((a for a in all_analyses if a.agent_role == AgentRole.TECHNICAL), None)
            sentiment_analysis = next((a for a in all_analyses if a.agent_role == AgentRole.NEWS_SENTIMENT), None)
            economic_analysis = next((a for a in all_analyses if a.agent_role == AgentRole.ECONOMIC_CLIMATE), None)
            
            # Extract scores and confidence levels with safe defaults
            fund_score = fundamental_analysis.score if fundamental_analysis else 5.0
            tech_score = technical_analysis.score if technical_analysis else 5.0
            sent_score = sentiment_analysis.score if sentiment_analysis else 5.0
            econ_score = economic_analysis.score if economic_analysis else 5.0
            
            fund_conf = fundamental_analysis.confidence if fundamental_analysis else 0.5
            tech_conf = technical_analysis.confidence if technical_analysis else 0.5
            sent_conf = sentiment_analysis.confidence if sentiment_analysis else 0.5
            econ_conf = economic_analysis.confidence if economic_analysis else 0.5
            
            prompt = f"""
You are the Chief Investment Officer synthesizing analysis for {fundamental_data.ticker}:

COMPANY PROFILE:
- Ticker: {fundamental_data.ticker}
- Market Cap: ${fundamental_data.market_cap:.1f}B
- Revenue: ${fundamental_data.total_revenue:.1f}B
- Fiscal Year: {fundamental_data.fiscal_year}

ANALYSIS TEAM SCORES:
1. FUNDAMENTAL: {fund_score:.1f}/10 (Confidence: {fund_conf:.0%})
   Key: ROE {fundamental_data.return_on_equity:.1f}%, D/E {fundamental_data.debt_to_equity:.2f}

2. TECHNICAL: {tech_score:.1f}/10 (Confidence: {tech_conf:.0%})
   Trend: {"Bullish" if tech_score > 6 else "Bearish" if tech_score < 4 else "Neutral"}

3. SENTIMENT: {sent_score:.1f}/10 (Confidence: {sent_conf:.0%})
   Market: {"Positive" if sent_score > 6 else "Negative" if sent_score < 4 else "Neutral"}

4. ECONOMIC: {econ_score:.1f}/10 (Confidence: {econ_conf:.0%})
   Climate: {"Favorable" if econ_score > 6 else "Challenging" if econ_score < 4 else "Mixed"}

## EXECUTIVE INVESTMENT SUMMARY
Provide 3-4 sentence overall investment thesis.

## INVESTMENT GRADE & RATING
Assign overall grade (A+ to F) and investment rating (Strong Buy/Buy/Hold/Sell/Strong Sell).

## RISK-RETURN PROFILE
Assess expected returns, risk level, and key factors to monitor.

## KEY INVESTMENT DRIVERS
- Top 3 bullish factors
- Top 3 bearish factors

## INVESTMENT RECOMMENDATION
Provide specific action, timeline, and position sizing guidance.

Format as institutional investment committee memorandum.
"""
            
            analysis_text = self.generate_analysis(prompt, max_tokens=3500)
            
            # Calculate weighted overall score
            try:
                weights = {'fundamental': 0.40, 'technical': 0.25, 'sentiment': 0.15, 'economic': 0.20}
                
                conf_weighted_scores = [
                    fund_score * fund_conf * weights['fundamental'],
                    tech_score * tech_conf * weights['technical'],
                    sent_score * sent_conf * weights['sentiment'],
                    econ_score * econ_conf * weights['economic']
                ]
                
                total_weighted_conf = sum([
                    fund_conf * weights['fundamental'],
                    tech_conf * weights['technical'],
                    sent_conf * weights['sentiment'],
                    econ_conf * weights['economic']
                ])
                
                overall_score = safe_divide(sum(conf_weighted_scores), total_weighted_conf, 5.0)
                overall_score = max(1.0, min(10.0, overall_score))
                
            except:
                overall_score = (fund_score + tech_score + sent_score + econ_score) / 4
            
            overall_confidence = (fund_conf + tech_conf + sent_conf + econ_conf) / 4
            
            # Generate investment grade
            if overall_score >= 9: grade = "A+"
            elif overall_score >= 8: grade = "A"
            elif overall_score >= 7: grade = "B+"
            elif overall_score >= 6: grade = "B"
            elif overall_score >= 5: grade = "C"
            elif overall_score >= 4: grade = "D"
            else: grade = "F"
            
            key_metrics = {
                'overall_score': overall_score,
                'fundamental_score': fund_score,
                'technical_score': tech_score,
                'sentiment_score': sent_score,
                'economic_score': econ_score,
                'market_cap': fundamental_data.market_cap,
                'pe_ratio': fundamental_data.price_to_earnings
            }
            
            recommendations = [
                f"Investment Grade: {grade} (Score: {overall_score:.1f}/10)",
                f"Fundamental: {fund_score:.1f}/10",
                f"Technical: {tech_score:.1f}/10",
                f"Confidence: {overall_confidence:.0%}"
            ]
            
            return AgentAnalysis(
                agent_role=self.agent_role,
                analysis=analysis_text,
                score=overall_score,
                confidence=overall_confidence,
                recommendations=recommendations,
                key_metrics=key_metrics,
                timestamp=datetime.now(),
                data={
                    'investment_grade': grade,
                    'component_scores': {
                        'fundamental': fund_score,
                        'technical': tech_score,
                        'sentiment': sent_score,
                        'economic': econ_score
                    }
                }
            )
            
        except Exception as e:
            st.error(f"Error in supervisor analysis: {str(e)}")
            return AgentAnalysis(
                agent_role=self.agent_role,
                analysis=f"Supervisor analysis failed: {str(e)}",
                score=5.0,
                confidence=0.1,
                recommendations=["Review individual analyses"],
                key_metrics={},
                timestamp=datetime.now()
            )

# ============================================================================
# ENHANCED ORCHESTRATOR WITH SMART API USAGE
# ============================================================================

class EnhancedFinancialAnalysisOrchestrator:
    def __init__(self, model_path: str = "~/mlx-models", alpha_vantage_api_key: str = "FPTDFNNAMTMH6091"):
        """Initialize system with smart API usage strategy"""
        try:
            self.mlx = MLXClient(model_path)
            
            # Initialize Alpha Vantage client if available (optional)
            self.av = None
            if ALPHA_VANTAGE_AVAILABLE and alpha_vantage_api_key:
                try:
                    self.av = AlphaVantageClient(alpha_vantage_api_key)
                except:
                    st.warning("Alpha Vantage client initialization failed - using Yahoo Finance only")
            
            # Initialize all agents
            self.fundamental_agent = FundamentalAnalysisAgent(self.mlx, self.av)
            self.technical_agent = TechnicalAnalysisAgent(self.mlx, self.av)
            self.sentiment_agent = NewsAndSentimentAgent(self.mlx, self.av)
            self.economic_agent = EconomicClimateAgent(self.mlx, self.av)
            self.supervisor_agent = SupervisorAgent(self.mlx, self.av)
            
            self.analysis_history = []
            self.economic_cache = None
            self.economic_cache_time = None
            
        except Exception as e:
            st.error(f"Error initializing orchestrator: {str(e)}")
            raise
    
    def run_comprehensive_analysis(self, ticker: str) -> Tuple[EnhancedFinancialRatios, List[AgentAnalysis]]:
        """Run analysis with smart API usage"""
        
        analyses = []
        
        try:
            # Step 1: Fundamental Analysis (Yahoo Finance - reliable)
            with st.status(f"📊 Running fundamental analysis for {ticker}..."):
                fundamental_data = self.fundamental_agent.get_comprehensive_data(ticker)
                if not fundamental_data:
                    raise ValueError(f"Could not fetch fundamental data for {ticker}")
                
                fundamental_analysis = self.fundamental_agent.analyze(fundamental_data)
                analyses.append(fundamental_analysis)
        except Exception as e:
            st.error(f"Fundamental analysis failed: {str(e)}")
            raise
        
        # Step 2: Technical Analysis (Yahoo Finance - unlimited)
        try:
            with st.status(f"📈 Running technical analysis for {ticker}..."):
                technical_analysis = self.technical_agent.analyze(ticker)
                analyses.append(technical_analysis)
        except Exception as e:
            st.warning(f"Technical analysis failed: {str(e)}")
            analyses.append(AgentAnalysis(
                agent_role=AgentRole.TECHNICAL,
                analysis=f"Technical analysis unavailable: {str(e)}",
                score=5.0, confidence=0.1, recommendations=["Retry later"],
                key_metrics={}, timestamp=datetime.now()
            ))
        
        # Step 3: Sentiment Analysis (Alpha Vantage - optional, cached)
        try:
            with st.status(f"📰 Analyzing sentiment for {ticker}..."):
                sentiment_analysis = self.sentiment_agent.analyze(ticker)
                analyses.append(sentiment_analysis)
        except Exception as e:
            st.warning(f"Sentiment analysis limited: {str(e)}")
            analyses.append(AgentAnalysis(
                agent_role=AgentRole.NEWS_SENTIMENT,
                analysis="Sentiment analysis using fallback method due to API limits",
                score=5.0, confidence=0.3, recommendations=["Monitor news manually"],
                key_metrics={}, timestamp=datetime.now()
            ))
        
        # Step 4: Economic Analysis (Alpha Vantage - cached for 4 hours)
        try:
            with st.status("🌍 Analyzing economic climate..."):
                current_time = datetime.now()
                if (self.economic_cache is None or 
                    self.economic_cache_time is None or 
                    current_time - self.economic_cache_time > timedelta(hours=4)):
                    
                    economic_analysis = self.economic_agent.analyze()
                    self.economic_cache = economic_analysis
                    self.economic_cache_time = current_time
                    analyses.append(economic_analysis)
                else:
                    analyses.append(self.economic_cache)
        except Exception as e:
            st.warning(f"Economic analysis limited: {str(e)}")
            analyses.append(AgentAnalysis(
                agent_role=AgentRole.ECONOMIC_CLIMATE,
                analysis="Economic analysis using general framework due to API limits",
                score=5.0, confidence=0.4, recommendations=["Monitor Fed announcements"],
                key_metrics={}, timestamp=datetime.now()
            ))
        
        # Step 5: Supervisor Synthesis (always works)
        try:
            with st.status("🧠 Synthesizing analysis..."):
                supervisor_analysis = self.supervisor_agent.synthesize_analysis(analyses, fundamental_data)
                analyses.append(supervisor_analysis)
        except Exception as e:
            st.warning(f"Supervisor synthesis failed: {str(e)}")
            analyses.append(AgentAnalysis(
                agent_role=AgentRole.SUPERVISOR,
                analysis=f"Synthesis failed: {str(e)}",
                score=5.0, confidence=0.1, recommendations=["Review individual analyses"],
                key_metrics={}, timestamp=datetime.now()
            ))
        
        # Store in history
        try:
            self.analysis_history.append({
                'ticker': ticker,
                'timestamp': datetime.now(),
                'fundamental_data': fundamental_data,
                'analyses': analyses
            })
        except:
            pass
        
        return fundamental_data, analyses
    
    def is_available(self) -> bool:
        """Check if system is available"""
        return True  # Always available with Yahoo Finance
    
    def get_model_status(self) -> str:
        """Get MLX model status"""
        try:
            if self.mlx.is_loaded:
                return "✅ MLX DeepSeek model ready"
            else:
                return "❌ MLX model loading on first analysis"
        except:
            return "❌ MLX model error"
    
    def get_api_status(self) -> str:
        """Get API status - focused on Yahoo Finance"""
        try:
            # Test Yahoo Finance
            test_stock = yf.Ticker("AAPL")
            test_info = test_stock.info
            if test_info.get('marketCap'):
                yf_status = "✅ Yahoo Finance connected"
            else:
                yf_status = "⚠️ Yahoo Finance issues"
            
            # Test Alpha Vantage if available
            av_status = ""
            if self.av:
                try:
                    test_data = self.av.get_company_overview("AAPL")
                    if test_data and 'Symbol' in test_data:
                        av_status = " | ✅ Alpha Vantage available"
                    else:
                        av_status = " | ⚠️ Alpha Vantage limited"
                except:
                    av_status = " | ❌ Alpha Vantage rate limited"
            else:
                av_status = " | ℹ️ Alpha Vantage not configured"
            
            return yf_status + av_status
            
        except Exception as e:
            return f"❌ API connection failed: {str(e)}"
