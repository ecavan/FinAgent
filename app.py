"""
Enhanced FinAgent - FIXED Multi-Agent Financial Analysis Platform
SMART API STRATEGY: Yahoo Finance Primary + Alpha Vantage Secondary
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Any

# Import our API-fixed agents
from agents import (
    EnhancedFinancialAnalysisOrchestrator, 
    AgentRole, 
    EnhancedFinancialRatios, 
    AgentAnalysis
)

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="FinAgent Pro - AI Financial Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_currency(value, decimals=1):
    """Format currency values with appropriate scaling"""
    try:
        if value >= 1e12:
            return f"${value/1e12:.{decimals}f}T"
        elif value >= 1e9:
            return f"${value/1e9:.{decimals}f}B"
        elif value >= 1e6:
            return f"${value/1e6:.{decimals}f}M"
        else:
            return f"${value:,.{decimals}f}"
    except:
        return "$0.0"

def format_percentage(value, decimals=2):
    """Format percentage values"""
    try:
        return f"{value:+.{decimals}f}%"
    except:
        return "0.00%"

def get_score_color(score):
    """Get color based on score"""
    try:
        if score >= 8:
            return "#28a745"  # Green
        elif score >= 6:
            return "#ffc107"  # Yellow
        elif score >= 4:
            return "#fd7e14"  # Orange
        else:
            return "#dc3545"  # Red
    except:
        return "#6c757d"  # Gray

def get_grade_from_score(score):
    """Convert numerical score to letter grade"""
    try:
        if score >= 9:
            return "A+"
        elif score >= 8:
            return "A"
        elif score >= 7:
            return "B+"
        elif score >= 6:
            return "B"
        elif score >= 5:
            return "C"
        elif score >= 4:
            return "D"
        else:
            return "F"
    except:
        return "N/A"

# ============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# ============================================================================

def create_agent_performance_radar(analyses: List[AgentAnalysis]):
    """Create radar chart showing all agent scores"""
    try:
        categories = []
        scores = []
        confidences = []
        
        for analysis in analyses:
            if analysis.agent_role == AgentRole.FUNDAMENTAL:
                categories.append("Fundamental")
                scores.append(analysis.score)
                confidences.append(analysis.confidence)
            elif analysis.agent_role == AgentRole.TECHNICAL:
                categories.append("Technical")
                scores.append(analysis.score)
                confidences.append(analysis.confidence)
            elif analysis.agent_role == AgentRole.NEWS_SENTIMENT:
                categories.append("Sentiment")
                scores.append(analysis.score)
                confidences.append(analysis.confidence)
            elif analysis.agent_role == AgentRole.ECONOMIC_CLIMATE:
                categories.append("Economic")
                scores.append(analysis.score)
                confidences.append(analysis.confidence)
        
        if not categories:
            return None
        
        fig = go.Figure()
        
        # Add scores
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Analysis Score',
            line_color='rgb(31, 119, 180)',
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
        
        # Add confidence as a separate trace
        fig.add_trace(go.Scatterpolar(
            r=[c * 10 for c in confidences],  # Scale confidence to 10
            theta=categories,
            fill='toself',
            name='Confidence Level',
            line_color='rgb(255, 127, 14)',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line_dash='dash'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickmode='linear',
                    tick0=0,
                    dtick=2
                )),
            showlegend=True,
            title="Multi-Agent Analysis Dashboard",
            height=500,
            font=dict(size=12)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating radar chart: {str(e)}")
        return None

def create_financial_metrics_dashboard(ratios: EnhancedFinancialRatios):
    """Create comprehensive financial metrics visualization"""
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Profitability (%)', 'Liquidity & Leverage', 'Valuation Ratios',
                'Cash Flow ($B)', 'Returns (%)', 'Financial Health'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Profitability Metrics
        fig.add_trace(
            go.Bar(
                x=['Gross', 'Operating', 'Net'],
                y=[ratios.gross_margin, ratios.operating_margin, ratios.net_profit_margin],
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
            ),
            row=1, col=1
        )
        
        # Liquidity & Leverage
        fig.add_trace(
            go.Bar(
                x=['Current', 'Quick', 'D/E'],
                y=[ratios.current_ratio, ratios.quick_ratio, ratios.debt_to_equity],
                marker_color=['#d62728', '#9467bd', '#8c564b']
            ),
            row=1, col=2
        )
        
        # Valuation Ratios
        fig.add_trace(
            go.Bar(
                x=['P/E', 'P/B', 'P/S'],
                y=[ratios.price_to_earnings, ratios.price_to_book, ratios.price_to_sales],
                marker_color=['#e377c2', '#7f7f7f', '#bcbd22']
            ),
            row=1, col=3
        )
        
        # Cash Flow Analysis
        fig.add_trace(
            go.Bar(
                x=['Operating', 'Free CF', 'FCF Yield'],
                y=[ratios.operating_cash_flow, ratios.free_cash_flow, ratios.free_cash_flow_yield],
                marker_color=['#17becf', '#1f77b4', '#ff7f0e']
            ),
            row=2, col=1
        )
        
        # Returns Analysis
        fig.add_trace(
            go.Bar(
                x=['ROA', 'ROE', 'ROIC'],
                y=[ratios.return_on_assets, ratios.return_on_equity, ratios.return_on_invested_capital],
                marker_color=['#2ca02c', '#d62728', '#9467bd']
            ),
            row=2, col=2
        )
        
        # Financial Health
        fig.add_trace(
            go.Bar(
                x=['Revenue', 'Market Cap', 'Working Cap'],
                y=[ratios.total_revenue, ratios.market_cap, ratios.working_capital],
                marker_color=['#8c564b', '#e377c2', '#7f7f7f']
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Financial Metrics Dashboard"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating financial dashboard: {str(e)}")
        return None

def create_comparison_analysis(companies_data: Dict):
    """Create advanced comparison analysis with proper scaling"""
    
    if len(companies_data) < 2:
        return None
    
    try:
        # Extract data for comparison
        tickers = list(companies_data.keys())
        
        # Create comprehensive comparison metrics
        metrics = {
            'Revenue ($B)': [],
            'Net Margin (%)': [],
            'ROE (%)': [],
            'ROA (%)': [],
            'Current Ratio': [],
            'D/E Ratio': [],
            'P/E Ratio': [],
            'FCF Yield (%)': []
        }
        
        for ticker in tickers:
            ratios = companies_data[ticker]['fundamental_data']
            metrics['Revenue ($B)'].append(ratios.total_revenue)
            metrics['Net Margin (%)'].append(ratios.net_profit_margin)
            metrics['ROE (%)'].append(ratios.return_on_equity)
            metrics['ROA (%)'].append(ratios.return_on_assets)
            metrics['Current Ratio'].append(ratios.current_ratio)
            metrics['D/E Ratio'].append(ratios.debt_to_equity)
            metrics['P/E Ratio'].append(ratios.price_to_earnings)
            metrics['FCF Yield (%)'].append(ratios.free_cash_flow_yield)
        
        # Create normalized radar chart for each company
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, ticker in enumerate(tickers):
            # Normalize metrics for radar chart (scale 0-10) with proper scaling
            try:
                normalized_values = [
                    min(max(metrics['Net Margin (%)'][i] / 5, 0), 10),  # Scale net margin
                    min(max(metrics['ROE (%)'][i] / 3, 0), 10),         # Scale ROE  
                    min(max(metrics['ROA (%)'][i] / 2, 0), 10),         # Scale ROA
                    min(max(metrics['Current Ratio'][i] / 0.3, 0), 10), # Scale current ratio
                    max(min(10 - metrics['D/E Ratio'][i] * 2, 10), 0),  # Inverse D/E (lower is better)
                    max(min(10 - metrics['P/E Ratio'][i] / 3, 10), 0),  # Inverse P/E (lower is better)
                    min(max(metrics['FCF Yield (%)'][i] * 2, 0), 10)    # Scale FCF yield
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=['Net Margin', 'ROE', 'ROA', 'Liquidity', 'Leverage', 'Valuation', 'FCF Yield'],
                    fill='toself',
                    name=ticker,
                    line_color=colors[i % len(colors)]
                ))
            except Exception as e:
                st.warning(f"Error normalizing data for {ticker}: {str(e)}")
                continue
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Multi-Company Financial Comparison (Normalized)",
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating comparison analysis: {str(e)}")
        return None

def create_risk_return_scatter(companies_data: Dict):
    """Create risk-return scatter plot with market cap sizing"""
    
    if len(companies_data) < 2:
        return None
    
    try:
        tickers = []
        returns = []
        risks = []
        market_caps = []
        
        for ticker, data in companies_data.items():
            try:
                analyses = data['analyses']
                ratios = data['fundamental_data']
                
                # Get overall score as return proxy
                supervisor_analysis = next((a for a in analyses if a.agent_role == AgentRole.SUPERVISOR), None)
                if supervisor_analysis:
                    returns.append(supervisor_analysis.score)
                    
                    # Calculate risk score (inverse of financial stability)
                    stability_score = 0
                    if ratios.current_ratio > 1.0:
                        stability_score += 2
                    if ratios.debt_to_equity < 1.0:
                        stability_score += 3
                    if ratios.free_cash_flow > 0:
                        stability_score += 2
                    if ratios.return_on_equity > 10:
                        stability_score += 3
                    
                    risk_score = max(1, 10 - stability_score)
                    risks.append(risk_score)
                    
                    tickers.append(ticker)
                    market_caps.append(max(ratios.market_cap, 0.1))  # Avoid zero market cap
            except Exception as e:
                st.warning(f"Error processing {ticker} for risk-return: {str(e)}")
                continue
        
        if len(tickers) < 2:
            return None
        
        fig = go.Figure()
        
        # Calculate bubble sizes (scale market cap for visibility)
        min_cap = min(market_caps)
        max_cap = max(market_caps)
        if max_cap > min_cap:
            normalized_caps = [(mc - min_cap) / (max_cap - min_cap) * 40 + 15 for mc in market_caps]
        else:
            normalized_caps = [25] * len(market_caps)
        
        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='markers+text',
            text=tickers,
            textposition="top center",
            marker=dict(
                size=normalized_caps,
                color=returns,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Overall Score"),
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            name='Companies'
        ))
        
        fig.update_layout(
            title="Risk-Return Analysis (Bubble Size = Market Cap)",
            xaxis_title="Risk Score (1-10, Higher = More Risk)",
            yaxis_title="Return Potential Score (1-10)",
            height=500,
            xaxis=dict(range=[0, 11]),
            yaxis=dict(range=[0, 11])
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating risk-return scatter: {str(e)}")
        return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.title("🤖 FinAgent Pro - AI Financial Analysis")
    st.markdown("*Smart multi-agent analysis with Yahoo Finance + Alpha Vantage*")
    
    # Initialize session state
    if 'orchestrator' not in st.session_state:
        try:
            st.session_state.orchestrator = EnhancedFinancialAnalysisOrchestrator()
        except Exception as e:
            st.error(f"Failed to initialize system: {str(e)}")
            st.stop()
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    orchestrator = st.session_state.orchestrator
    
    # ============================================================================
    # SIDEBAR - Enhanced Status & Controls
    # ============================================================================
    
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Model status
        model_status = orchestrator.get_model_status()
        if "✅" in model_status:
            st.success(model_status)
        else:
            st.info(model_status)
        
        # API status with detailed breakdown
        api_status = orchestrator.get_api_status()
        st.markdown("**Data Sources:**")
        if "Yahoo Finance connected" in api_status:
            st.success("✅ Yahoo Finance (Primary)")
        else:
            st.error("❌ Yahoo Finance (Primary)")
        
        if "Alpha Vantage available" in api_status:
            st.success("✅ Alpha Vantage (Enhanced)")
        elif "Alpha Vantage limited" in api_status:
            st.warning("⚠️ Alpha Vantage (Limited)")
        elif "Alpha Vantage rate limited" in api_status:
            st.warning("⚠️ Alpha Vantage (Rate Limited)")
        else:
            st.info("ℹ️ Alpha Vantage (Not Available)")
        
        st.markdown("""
        **Data Strategy:**
        - 📊 Fundamentals: Yahoo Finance  
        - 📈 Technical: Yahoo Finance
        - 📰 Sentiment: Alpha Vantage (optional)
        - 🌍 Economic: Alpha Vantage (cached)
        """)
        
        st.divider()
        
        # Analysis History
        st.header("📝 Analysis History")
        if st.session_state.analysis_results:
            for ticker in list(st.session_state.analysis_results.keys())[-5:]:
                timestamp = st.session_state.analysis_results[ticker]['timestamp']
                if st.button(f"{ticker} - {timestamp.strftime('%H:%M')}", key=f"hist_{ticker}"):
                    st.session_state.selected_ticker = ticker
                    st.rerun()
        else:
            st.info("No analysis history")
        
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_results = {}
            st.rerun()
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Minimum confidence for recommendations"
        )
        
        show_debug = st.checkbox("Show debug info", value=False)
    
    # ============================================================================
    # MAIN INTERFACE
    # ============================================================================
    
    # Input Section
    st.header("🎯 Stock Analysis")
    
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        ticker = st.text_input(
            "Enter Stock Ticker(s)",
            value=st.session_state.get('selected_ticker', ''),
            placeholder="e.g., AAPL or AAPL,MSFT,GOOGL (comma-separated for comparison)"
        ).upper()
    
    with col2:
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    with col3:
        compare_mode = st.toggle("Compare Mode", value=False)
    
    with col4:
        export_mode = st.toggle("Export Report", value=False)
    
    # Quick Select Categories
    st.markdown("**Quick Select:**")
    quick_categories = {
        "Tech Giants": ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
        "Financial": ['JPM', 'BAC', 'WFC', 'GS'],
        "Healthcare": ['JNJ', 'PFE', 'UNH', 'ABBV'],
        "Industrial": ['BA', 'CAT', 'GE', 'MMM'],
        "Consumer": ['KO', 'PG', 'WMT', 'HD']
    }
    
    selected_category = st.selectbox("Category:", list(quick_categories.keys()))
    
    cols = st.columns(len(quick_categories[selected_category]))
    for i, quick_ticker in enumerate(quick_categories[selected_category]):
        with cols[i]:
            if st.button(quick_ticker, key=f"quick_{quick_ticker}"):
                st.session_state.selected_ticker = quick_ticker
                st.rerun()
    
    # Analysis Execution
    if analyze_button and ticker:
        tickers = [t.strip() for t in ticker.split(',') if t.strip()]
        
        if len(tickers) > 5:
            st.error("❌ Maximum 5 companies for optimal performance")
            return
        
        # Progress tracking
        progress_container = st.container()
        
        with progress_container:
            overall_progress = st.progress(0)
            status_text = st.empty()
            
            for i, single_ticker in enumerate(tickers):
                try:
                    status_text.text(f"🤖 Analyzing {single_ticker}... ({i+1}/{len(tickers)})")
                    overall_progress.progress((i) / len(tickers))
                    
                    # Run comprehensive analysis
                    fundamental_data, analyses = orchestrator.run_comprehensive_analysis(single_ticker)
                    
                    # Store results
                    st.session_state.analysis_results[single_ticker] = {
                        'fundamental_data': fundamental_data,
                        'analyses': analyses,
                        'timestamp': datetime.now()
                    }
                    
                    overall_progress.progress((i + 1) / len(tickers))
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing {single_ticker}: {str(e)}")
                    if show_debug:
                        st.exception(e)
                    continue
            
            status_text.text("✅ Analysis Complete!")
            overall_progress.progress(1.0)
            
            # Clear progress after 2 seconds
            import time
            time.sleep(1)
            progress_container.empty()
    
    # ============================================================================
    # RESULTS DISPLAY
    # ============================================================================
    
    if st.session_state.analysis_results:
        
        if compare_mode and len(st.session_state.analysis_results) > 1:
            # ========================================================================
            # COMPARISON MODE - RESTORED
            # ========================================================================
            st.header("📊 Multi-Company Investment Analysis")
            
            # Comparison Overview Cards
            st.subheader("Portfolio Overview")
            cols = st.columns(min(len(st.session_state.analysis_results), 5))
            
            for i, (ticker, data) in enumerate(st.session_state.analysis_results.items()):
                if i >= 5:  # Limit to 5 columns
                    break
                with cols[i]:
                    ratios = data['fundamental_data']
                    analyses = data['analyses']
                    supervisor_analysis = next((a for a in analyses if a.agent_role == AgentRole.SUPERVISOR), None)
                    
                    overall_score = supervisor_analysis.score if supervisor_analysis else 5.0
                    grade = get_grade_from_score(overall_score)
                    
                    st.metric(
                        label=f"**{ticker}**",
                        value=f"Grade: {grade}",
                        delta=f"{overall_score:.1f}/10"
                    )
                    st.caption(f"Market Cap: {format_currency(ratios.market_cap * 1e9)}")
                    st.caption(f"P/E: {ratios.price_to_earnings:.1f}x")
            
            # Advanced Comparison Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                comparison_fig = create_comparison_analysis(st.session_state.analysis_results)
                if comparison_fig:
                    st.plotly_chart(comparison_fig, use_container_width=True)
                else:
                    st.info("Comparison chart unavailable")
            
            with col2:
                risk_return_fig = create_risk_return_scatter(st.session_state.analysis_results)
                if risk_return_fig:
                    st.plotly_chart(risk_return_fig, use_container_width=True)
                else:
                    st.info("Risk-return chart unavailable")
            
            # Detailed Comparison Table
            st.subheader("📋 Comprehensive Metrics Comparison")
            
            comparison_data = []
            for ticker, data in st.session_state.analysis_results.items():
                try:
                    ratios = data['fundamental_data']
                    analyses = data['analyses']
                    
                    supervisor_analysis = next((a for a in analyses if a.agent_role == AgentRole.SUPERVISOR), None)
                    overall_score = supervisor_analysis.score if supervisor_analysis else 0
                    
                    comparison_data.append({
                        'Company': ticker,
                        'Grade': get_grade_from_score(overall_score),
                        'Score': f"{overall_score:.1f}/10",
                        'Revenue ($B)': f"{ratios.total_revenue:.1f}",
                        'Market Cap ($B)': f"{ratios.market_cap:.1f}",
                        'ROE (%)': f"{ratios.return_on_equity:.1f}",
                        'ROA (%)': f"{ratios.return_on_assets:.1f}",
                        'Net Margin (%)': f"{ratios.net_profit_margin:.1f}",
                        'Current Ratio': f"{ratios.current_ratio:.2f}",
                        'D/E Ratio': f"{ratios.debt_to_equity:.2f}",
                        'P/E Ratio': f"{ratios.price_to_earnings:.1f}",
                        'FCF Yield (%)': f"{ratios.free_cash_flow_yield:.1f}"
                    })
                except Exception as e:
                    st.warning(f"Error processing {ticker} for comparison: {str(e)}")
                    continue
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
            
            # Investment Recommendations Summary
            st.subheader("💡 Portfolio Construction Recommendations")
            
            for ticker, data in st.session_state.analysis_results.items():
                try:
                    analyses = data['analyses']
                    supervisor_analysis = next((a for a in analyses if a.agent_role == AgentRole.SUPERVISOR), None)
                    
                    if supervisor_analysis and supervisor_analysis.confidence >= confidence_threshold:
                        with st.expander(f"📈 {ticker} Investment Thesis"):
                            st.markdown(supervisor_analysis.analysis)
                except Exception as e:
                    if show_debug:
                        st.warning(f"Error displaying {ticker} recommendations: {str(e)}")
                    continue
        
        else:
            # ========================================================================
            # SINGLE COMPANY DETAILED ANALYSIS
            # ========================================================================
            
            if len(st.session_state.analysis_results) == 1:
                ticker = list(st.session_state.analysis_results.keys())[0]
            else:
                ticker = st.selectbox(
                    "Select company for detailed analysis:", 
                    list(st.session_state.analysis_results.keys())
                )
            
            if ticker in st.session_state.analysis_results:
                data = st.session_state.analysis_results[ticker]
                ratios = data['fundamental_data']
                analyses = data['analyses']
                
                # Header with Key Metrics
                st.header(f"📈 {ticker} - Investment Analysis")
                
                # Key Performance Indicators
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Revenue", format_currency(ratios.total_revenue * 1e9))
                
                with col2:
                    st.metric("Market Cap", format_currency(ratios.market_cap * 1e9))
                
                with col3:
                    st.metric("P/E Ratio", f"{ratios.price_to_earnings:.1f}x")
                
                with col4:
                    st.metric("ROE", f"{ratios.return_on_equity:.1f}%")
                
                with col5:
                    supervisor_analysis = next((a for a in analyses if a.agent_role == AgentRole.SUPERVISOR), None)
                    if supervisor_analysis:
                        grade = get_grade_from_score(supervisor_analysis.score)
                        st.metric("Overall Grade", grade, f"{supervisor_analysis.score:.1f}/10")
                
                # Agent Performance Dashboard
                st.subheader("🤖 AI Agent Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    radar_fig = create_agent_performance_radar(analyses)
                    if radar_fig:
                        st.plotly_chart(radar_fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Agent Scores:**")
                    for analysis in analyses:
                        if analysis.agent_role != AgentRole.SUPERVISOR:
                            role_name = analysis.agent_role.value.replace('_', ' ').title()
                            confidence_emoji = "🟢" if analysis.confidence > 0.7 else "🟡" if analysis.confidence > 0.4 else "🔴"
                            st.markdown(f"{confidence_emoji} **{role_name}**: {analysis.score:.1f}/10 ({analysis.confidence:.0%})")
                
                # Financial Metrics Dashboard
                st.subheader("📊 Financial Dashboard")
                financial_dashboard = create_financial_metrics_dashboard(ratios)
                if financial_dashboard:
                    st.plotly_chart(financial_dashboard, use_container_width=True)
                
                # Analysis Tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Executive Summary", "💰 Fundamental", "📈 Technical", 
                    "📰 Sentiment", "🌍 Economic"
                ])
                
                with tab1:
                    if supervisor_analysis:
                        st.header("🧠 Investment Committee Summary")
                        
                        # Investment Grade Badge
                        grade = get_grade_from_score(supervisor_analysis.score)
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        border-radius: 15px; color: white; margin: 1rem 0;">
                                <h3 style="margin: 0; color: white;">Investment Grade: {grade}</h3>
                                <p style="margin: 0; font-size: 1.1rem;">Score: {supervisor_analysis.score:.1f}/10</p>
                                <p style="margin: 0;">Confidence: {supervisor_analysis.confidence:.0%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown(supervisor_analysis.analysis)
                
                with tab2:
                    fundamental_analysis = next((a for a in analyses if a.agent_role == AgentRole.FUNDAMENTAL), None)
                    if fundamental_analysis:
                        st.header("💰 Fundamental Analysis")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(fundamental_analysis.analysis)
                        with col2:
                            st.metric("Score", f"{fundamental_analysis.score:.1f}/10")
                            st.metric("Confidence", f"{fundamental_analysis.confidence:.0%}")
                            
                            # Key metrics display
                            if fundamental_analysis.key_metrics:
                                st.markdown("**Key Metrics:**")
                                for metric, value in fundamental_analysis.key_metrics.items():
                                    if isinstance(value, (int, float)):
                                        st.markdown(f"• {metric}: {value:.2f}")
                                    else:
                                        st.markdown(f"• {metric}: {value}")
                
                with tab3:
                    technical_analysis = next((a for a in analyses if a.agent_role == AgentRole.TECHNICAL), None)
                    if technical_analysis:
                        st.header("📈 Technical Analysis")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(technical_analysis.analysis)
                        with col2:
                            st.metric("Score", f"{technical_analysis.score:.1f}/10")
                            st.metric("Confidence", f"{technical_analysis.confidence:.0%}")
                            
                            # Technical signals
                            if technical_analysis.key_metrics:
                                st.markdown("**Current Signals:**")
                                for metric, value in technical_analysis.key_metrics.items():
                                    if isinstance(value, (int, float)):
                                        st.markdown(f"• {metric}: {value:.2f}")
                                    else:
                                        st.markdown(f"• {metric}: {value}")
                
                with tab4:
                    sentiment_analysis = next((a for a in analyses if a.agent_role == AgentRole.NEWS_SENTIMENT), None)
                    if sentiment_analysis:
                        st.header("📰 News Sentiment")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(sentiment_analysis.analysis)
                        with col2:
                            st.metric("Score", f"{sentiment_analysis.score:.1f}/10")
                            st.metric("Confidence", f"{sentiment_analysis.confidence:.0%}")
                            
                            # Sentiment metrics
                            if sentiment_analysis.key_metrics:
                                st.markdown("**Sentiment Data:**")
                                for metric, value in sentiment_analysis.key_metrics.items():
                                    st.markdown(f"• {metric}: {value}")
                
                with tab5:
                    economic_analysis = next((a for a in analyses if a.agent_role == AgentRole.ECONOMIC_CLIMATE), None)
                    if economic_analysis:
                        st.header("🌍 Economic Climate")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(economic_analysis.analysis)
                        with col2:
                            st.metric("Score", f"{economic_analysis.score:.1f}/10")
                            st.metric("Confidence", f"{economic_analysis.confidence:.0%}")
                            
                            # Economic indicators
                            if economic_analysis.key_metrics:
                                st.markdown("**Economic Data:**")
                                for metric, value in economic_analysis.key_metrics.items():
                                    if isinstance(value, (int, float)):
                                        st.markdown(f"• {metric}: {value:.2f}")
                                    else:
                                        st.markdown(f"• {metric}: {value}")
                
                # Export functionality
                if export_mode:
                    st.subheader("📤 Export Report")
                    
                    # Create text report
                    full_report = f"""
# FINAGENT PRO ANALYSIS REPORT
## {ticker} - {datetime.now().strftime('%Y-%m-%d %H:%M')}

### EXECUTIVE SUMMARY
{supervisor_analysis.analysis if supervisor_analysis else 'Not available'}

### FUNDAMENTAL ANALYSIS
{fundamental_analysis.analysis if fundamental_analysis else 'Not available'}

### TECHNICAL ANALYSIS  
{technical_analysis.analysis if technical_analysis else 'Not available'}

### SENTIMENT ANALYSIS
{sentiment_analysis.analysis if sentiment_analysis else 'Not available'}

### ECONOMIC ANALYSIS
{economic_analysis.analysis if economic_analysis else 'Not available'}

---
Generated by FinAgent Pro - Smart Multi-Agent Financial Analysis
Data Sources: Yahoo Finance (Primary) + Alpha Vantage (Enhanced)
"""
                    
                    st.download_button(
                        label="📥 Download Full Report",
                        data=full_report,
                        file_name=f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )
    
    else:
        # Welcome screen
        st.info("👆 Enter a stock ticker above to begin AI-powered financial analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🤖 Smart Multi-Agent Analysis:
            
            **🧠 AI-Powered Agents:**
            - 📊 **Fundamental Agent**: Financial ratios & company health
            - 📈 **Technical Agent**: Price trends & momentum
            - 📰 **Sentiment Agent**: News & market psychology  
            - 🌍 **Economic Agent**: Macro environment assessment
            - 👔 **Supervisor Agent**: Investment synthesis
            
            **🔍 Analysis Features:**
            - Multi-stock comparison with scaling
            - Risk-return portfolio analysis
            - Professional-grade reports
            - Export capabilities
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Smart Data Strategy:
            
            **🎯 Primary Sources (Unlimited):**
            - Yahoo Finance: Fundamentals & Technical
            - Reliable, fast, no rate limits
            
            **⚡ Enhanced Sources (Cached):**
            - Alpha Vantage: News sentiment
            - Economic indicators (4hr cache)
            - Graceful fallbacks when limited
            
            **✅ Always Works:**
            - Core analysis from Yahoo Finance
            - Smart caching prevents rate limits
            - Fallback strategies for all components
            """)
    
    # Footer
    st.divider()
    st.markdown("**FinAgent Pro** - Smart AI Financial Analysis | Primary: Yahoo Finance | Enhanced: Alpha Vantage | For educational purposes only")

if __name__ == "__main__":
    main()
