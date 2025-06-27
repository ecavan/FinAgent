"""
Alpha Vantage API Client for FinAgent
Provides access to comprehensive financial data, economic indicators, and news sentiment
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import time
import streamlit as st

# ============================================================================
# ALPHA VANTAGE DATA CLASSES
# ============================================================================

@dataclass
class ComprehensiveFinancials:
    """Enhanced financial data from Alpha Vantage"""
    ticker: str
    
    # Income Statement
    total_revenue: float
    gross_profit: float
    operating_income: float
    net_income: float
    ebitda: float
    eps: float
    
    # Balance Sheet
    total_assets: float
    total_liabilities: float
    shareholders_equity: float
    current_assets: float
    current_liabilities: float
    long_term_debt: float
    cash_and_equivalents: float
    
    # Cash Flow
    operating_cash_flow: float
    capital_expenditures: float
    free_cash_flow: float
    dividend_payments: float
    
    # Calculated Ratios
    debt_to_assets: float
    return_on_assets: float
    return_on_equity: float
    current_ratio: float
    quick_ratio: float
    debt_to_equity: float
    price_to_earnings: float
    price_to_book: float
    
    fiscal_year: str

@dataclass 
class TechnicalIndicators:
    """Technical analysis data from Alpha Vantage"""
    ticker: str
    
    # Price data
    current_price: float
    high_52_week: float
    low_52_week: float
    
    # Moving averages
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    
    # Technical indicators
    rsi: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bollinger_upper: float
    bollinger_lower: float
    atr: float  # Average True Range
    adx: float  # Average Directional Index
    
    # Volume indicators
    volume_20_avg: float
    volume_current: float
    obv: float  # On Balance Volume

@dataclass
class EconomicIndicators:
    """Economic climate data from Alpha Vantage"""
    
    # Inflation & Consumer Prices
    cpi_current: float
    cpi_yoy_change: float
    core_cpi: float
    
    # Economic Growth
    gdp_current: float
    gdp_growth_rate: float
    unemployment_rate: float
    
    # Market indicators
    fed_funds_rate: float
    treasury_10y: float
    dollar_index: float
    
    # Commodity prices
    crude_oil_price: float
    gold_price: float
    
    last_updated: datetime

@dataclass
class NewsAnalysis:
    """News sentiment analysis from Alpha Vantage"""
    ticker: str
    
    # Sentiment scores
    overall_sentiment: float  # -1 to 1
    relevance_score: float
    sentiment_label: str  # Bearish, Neutral, Bullish
    
    # News details
    news_count: int
    recent_headlines: List[str]
    top_topics: List[str]
    
    # Time-based sentiment
    sentiment_24h: float
    sentiment_7d: float
    sentiment_30d: float
    
    last_updated: datetime

# ============================================================================
# ALPHA VANTAGE API CLIENT
# ============================================================================

class AlphaVantageClient:
    """Enhanced Alpha Vantage API client with rate limiting and caching"""
    
    def __init__(self, api_key: str = "FPTDFNNAMTMH6091"):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.cache = {}  # Simple in-memory cache
        self.last_request_time = 0
        self.rate_limit_delay = 12  # 5 requests per minute for free tier
        
    def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make rate-limited API request with caching"""
        
        # Create cache key
        cache_key = f"{params.get('function')}_{params.get('symbol', '')}"
        
        # Check cache (5 minute expiry)
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < 300:  # 5 minutes
                return cached_data
        
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        # Make request
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
            if "Note" in data:
                raise ValueError(f"Alpha Vantage Rate Limit: {data['Note']}")
            
            # Cache the result
            self.cache[cache_key] = (time.time(), data)
            self.last_request_time = time.time()
            
            return data
            
        except Exception as e:
            st.error(f"Alpha Vantage API error: {str(e)}")
            return {}

    def get_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive company overview"""
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol
        }
        return self._make_request(params)

    def get_income_statement(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get annual income statements"""
        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': symbol
        }
        return self._make_request(params)

    def get_balance_sheet(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get annual balance sheets"""
        params = {
            'function': 'BALANCE_SHEET',
            'symbol': symbol
        }
        return self._make_request(params)

    def get_cash_flow(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get annual cash flow statements"""
        params = {
            'function': 'CASH_FLOW',
            'symbol': symbol
        }
        return self._make_request(params)

    def get_technical_indicators(self, symbol: str) -> Optional[TechnicalIndicators]:
        """Get comprehensive technical indicators"""
        try:
            # Get multiple technical indicators
            indicators = {}
            
            # RSI
            rsi_params = {
                'function': 'RSI',
                'symbol': symbol,
                'interval': 'daily',
                'time_period': '14',
                'series_type': 'close'
            }
            rsi_data = self._make_request(rsi_params)
            
            # MACD
            macd_params = {
                'function': 'MACD',
                'symbol': symbol,
                'interval': 'daily',
                'series_type': 'close'
            }
            macd_data = self._make_request(macd_params)
            
            # SMA
            sma20_params = {
                'function': 'SMA',
                'symbol': symbol,
                'interval': 'daily',
                'time_period': '20',
                'series_type': 'close'
            }
            sma20_data = self._make_request(sma20_params)
            
            # Get current quote
            quote_params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol
            }
            quote_data = self._make_request(quote_params)
            
            # Extract latest values
            current_price = 0
            if quote_data and 'Global Quote' in quote_data:
                current_price = float(quote_data['Global Quote'].get('05. price', 0))
            
            # Create technical indicators object
            return TechnicalIndicators(
                ticker=symbol.upper(),
                current_price=current_price,
                high_52_week=0,  # Would need additional API call
                low_52_week=0,   # Would need additional API call
                sma_20=self._get_latest_indicator_value(sma20_data, 'Technical Analysis: SMA'),
                sma_50=0,  # Would need additional API call
                sma_200=0, # Would need additional API call
                ema_12=0,  # Would need additional API call
                ema_26=0,  # Would need additional API call
                rsi=self._get_latest_indicator_value(rsi_data, 'Technical Analysis: RSI'),
                macd_line=self._get_latest_macd_value(macd_data, 'MACD'),
                macd_signal=self._get_latest_macd_value(macd_data, 'MACD_Signal'),
                macd_histogram=self._get_latest_macd_value(macd_data, 'MACD_Hist'),
                bollinger_upper=0,  # Would need additional API call
                bollinger_lower=0,  # Would need additional API call
                atr=0,  # Would need additional API call
                adx=0,  # Would need additional API call
                volume_20_avg=0,
                volume_current=float(quote_data.get('Global Quote', {}).get('06. volume', 0)),
                obv=0  # Would need additional API call
            )
            
        except Exception as e:
            st.warning(f"Could not fetch technical indicators for {symbol}: {str(e)}")
            return None

    def _get_latest_indicator_value(self, data: Dict, key: str) -> float:
        """Extract latest indicator value from API response"""
        try:
            if key in data and data[key]:
                latest_date = max(data[key].keys())
                return float(list(data[key][latest_date].values())[0])
            return 0.0
        except:
            return 0.0

    def _get_latest_macd_value(self, data: Dict, indicator: str) -> float:
        """Extract latest MACD value from API response"""
        try:
            if 'Technical Analysis: MACD' in data and data['Technical Analysis: MACD']:
                latest_date = max(data['Technical Analysis: MACD'].keys())
                return float(data['Technical Analysis: MACD'][latest_date].get(indicator, 0))
            return 0.0
        except:
            return 0.0

    def get_economic_indicators(self) -> Optional[EconomicIndicators]:
        """Get current economic indicators"""
        try:
            indicators = {}
            
            # CPI
            cpi_params = {
                'function': 'CPI',
                'interval': 'monthly'
            }
            cpi_data = self._make_request(cpi_params)
            
            # GDP
            gdp_params = {
                'function': 'REAL_GDP',
                'interval': 'quarterly'
            }
            gdp_data = self._make_request(gdp_params)
            
            # Unemployment
            unemployment_params = {
                'function': 'UNEMPLOYMENT',
                'interval': 'monthly'
            }
            unemployment_data = self._make_request(unemployment_params)
            
            # Federal funds rate
            fed_params = {
                'function': 'FEDERAL_FUNDS_RATE',
                'interval': 'monthly'
            }
            fed_data = self._make_request(fed_params)
            
            # Parse the data
            cpi_current = self._get_latest_economic_value(cpi_data, 'CPI')
            gdp_current = self._get_latest_economic_value(gdp_data, 'REAL_GDP')
            unemployment = self._get_latest_economic_value(unemployment_data, 'UNEMPLOYMENT')
            fed_rate = self._get_latest_economic_value(fed_data, 'FEDERAL_FUNDS_RATE')
            
            return EconomicIndicators(
                cpi_current=cpi_current,
                cpi_yoy_change=0,  # Would need calculation
                core_cpi=0,  # Would need additional API call
                gdp_current=gdp_current,
                gdp_growth_rate=0,  # Would need calculation
                unemployment_rate=unemployment,
                fed_funds_rate=fed_rate,
                treasury_10y=0,  # Would need additional API call
                dollar_index=0,  # Would need additional API call
                crude_oil_price=0,  # Would need additional API call
                gold_price=0,  # Would need additional API call
                last_updated=datetime.now()
            )
            
        except Exception as e:
            st.warning(f"Could not fetch economic indicators: {str(e)}")
            return None

    def _get_latest_economic_value(self, data: Dict, key: str) -> float:
        """Extract latest economic indicator value"""
        try:
            if 'data' in data and data['data']:
                latest_entry = data['data'][0]  # Most recent is first
                return float(latest_entry.get('value', 0))
            return 0.0
        except:
            return 0.0

    def get_news_sentiment(self, symbol: str) -> Optional[NewsAnalysis]:
        """Get news sentiment analysis for a stock"""
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'time_from': (datetime.now() - timedelta(days=30)).strftime('%Y%m%dT%H%M'),
                'limit': '200'
            }
            data = self._make_request(params)
            
            if not data or 'feed' not in data:
                return None
            
            articles = data['feed']
            if not articles:
                return None
            
            # Analyze sentiment
            sentiments = []
            headlines = []
            topics = []
            
            for article in articles[:20]:  # Top 20 articles
                # Extract ticker-specific sentiment
                ticker_sentiments = article.get('ticker_sentiment', [])
                for ts in ticker_sentiments:
                    if ts.get('ticker') == symbol.upper():
                        sentiments.append(float(ts.get('ticker_sentiment_score', 0)))
                        break
                
                headlines.append(article.get('title', ''))
                
                # Extract topics (simplified)
                if 'topics' in article:
                    for topic in article['topics']:
                        topics.append(topic.get('topic', ''))
            
            # Calculate overall sentiment
            overall_sentiment = np.mean(sentiments) if sentiments else 0.0
            
            # Determine sentiment label
            if overall_sentiment > 0.15:
                sentiment_label = "Bullish"
            elif overall_sentiment < -0.15:
                sentiment_label = "Bearish"
            else:
                sentiment_label = "Neutral"
            
            return NewsAnalysis(
                ticker=symbol.upper(),
                overall_sentiment=overall_sentiment,
                relevance_score=np.mean([float(a.get('relevance_score', 0)) for a in articles[:10]]),
                sentiment_label=sentiment_label,
                news_count=len(articles),
                recent_headlines=headlines[:10],
                top_topics=list(set(topics))[:10],
                sentiment_24h=overall_sentiment,  # Simplified
                sentiment_7d=overall_sentiment,   # Simplified
                sentiment_30d=overall_sentiment,  # Simplified
                last_updated=datetime.now()
            )
            
        except Exception as e:
            st.warning(f"Could not fetch news sentiment for {symbol}: {str(e)}")
            return None

    def get_comprehensive_financials(self, symbol: str) -> Optional[ComprehensiveFinancials]:
        """Get comprehensive financial data from multiple endpoints"""
        try:
            # Get all financial statements
            overview = self.get_company_overview(symbol)
            income_stmt = self.get_income_statement(symbol)
            balance_sheet = self.get_balance_sheet(symbol)
            cash_flow = self.get_cash_flow(symbol)
            
            if not all([overview, income_stmt, balance_sheet, cash_flow]):
                return None
            
            # Extract latest annual data
            latest_income = income_stmt.get('annualReports', [{}])[0] if income_stmt.get('annualReports') else {}
            latest_balance = balance_sheet.get('annualReports', [{}])[0] if balance_sheet.get('annualReports') else {}
            latest_cash = cash_flow.get('annualReports', [{}])[0] if cash_flow.get('annualReports') else {}
            
            # Helper function to safely convert to float
            def safe_float(value, default=0.0):
                try:
                    return float(value) if value and value != 'None' else default
                except:
                    return default
            
            # Extract financial data
            total_revenue = safe_float(latest_income.get('totalRevenue'))
            gross_profit = safe_float(latest_income.get('grossProfit'))
            operating_income = safe_float(latest_income.get('operatingIncome'))
            net_income = safe_float(latest_income.get('netIncome'))
            ebitda = safe_float(latest_income.get('ebitda'))
            
            total_assets = safe_float(latest_balance.get('totalAssets'))
            total_liabilities = safe_float(latest_balance.get('totalLiabilities'))
            shareholders_equity = safe_float(latest_balance.get('totalShareholderEquity'))
            current_assets = safe_float(latest_balance.get('currentAssets'))
            current_liabilities = safe_float(latest_balance.get('currentLiabilities'))
            long_term_debt = safe_float(latest_balance.get('longTermDebt'))
            cash_and_equivalents = safe_float(latest_balance.get('cashAndCashEquivalentsAtCarryingValue'))
            
            operating_cash_flow = safe_float(latest_cash.get('operatingCashflow'))
            capital_expenditures = safe_float(latest_cash.get('capitalExpenditures'))
            dividend_payments = safe_float(latest_cash.get('dividendPayout'))
            
            # Calculate ratios
            current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
            debt_to_equity = total_liabilities / shareholders_equity if shareholders_equity > 0 else 0
            debt_to_assets = total_liabilities / total_assets if total_assets > 0 else 0
            return_on_assets = net_income / total_assets if total_assets > 0 else 0
            return_on_equity = net_income / shareholders_equity if shareholders_equity > 0 else 0
            quick_ratio = (current_assets - safe_float(latest_balance.get('inventory'))) / current_liabilities if current_liabilities > 0 else 0
            free_cash_flow = operating_cash_flow - abs(capital_expenditures)
            
            # Get valuation ratios from overview
            pe_ratio = safe_float(overview.get('PERatio'))
            pb_ratio = safe_float(overview.get('PriceToBookRatio'))
            eps = safe_float(overview.get('EPS'))
            
            return ComprehensiveFinancials(
                ticker=symbol.upper(),
                total_revenue=total_revenue,
                gross_profit=gross_profit,
                operating_income=operating_income,
                net_income=net_income,
                ebitda=ebitda,
                eps=eps,
                total_assets=total_assets,
                total_liabilities=total_liabilities,
                shareholders_equity=shareholders_equity,
                current_assets=current_assets,
                current_liabilities=current_liabilities,
                long_term_debt=long_term_debt,
                cash_and_equivalents=cash_and_equivalents,
                operating_cash_flow=operating_cash_flow,
                capital_expenditures=capital_expenditures,
                free_cash_flow=free_cash_flow,
                dividend_payments=dividend_payments,
                debt_to_assets=debt_to_assets * 100,
                return_on_assets=return_on_assets * 100,
                return_on_equity=return_on_equity * 100,
                current_ratio=current_ratio,
                quick_ratio=quick_ratio,
                debt_to_equity=debt_to_equity,
                price_to_earnings=pe_ratio,
                price_to_book=pb_ratio,
                fiscal_year=latest_income.get('fiscalDateEnding', 'Unknown')
            )
            
        except Exception as e:
            st.warning(f"Could not fetch comprehensive financials for {symbol}: {str(e)}")
            return None
