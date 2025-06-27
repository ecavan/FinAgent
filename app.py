import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from agent import FinancialAnalysisOrchestrator, AgentRole, FinancialRatios, AgentAnalysis
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="FinAgent - Multi-Agent Financial Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_currency(value, decimals=1):
    """Format currency values with appropriate scaling"""
    if value >= 1e12:
        return f"${value/1e12:.{decimals}f}T"
    elif value >= 1e9:
        return f"${value/1e9:.{decimals}f}B"
    elif value >= 1e6:
        return f"${value/1e6:.{decimals}f}M"
    else:
        return f"${value:,.{decimals}f}"

def format_percentage(value, decimals=2):
    """Format percentage values"""
    return f"{value:+.{decimals}f}%"

def get_score_color(score):
    """Get color based on score"""
    if score >= 8:
        return "green"
    elif score >= 6:
        return "orange"
    elif score >= 4:
        return "yellow"
    else:
        return "red"

def get_grade_from_score(score):
    """Convert numerical score to letter grade"""
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

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_normalized_comparison_chart(companies_data):
    """Create normalized charts for comparing companies of different sizes"""
    
    if not companies_data:
        return None, None
    
    # Extract data for comparison
    tickers = []
    revenues = []
    market_caps = []
    roas = []
    roes = []
    profit_margins = []
    gross_margins = []
    current_ratios = []
    debt_ratios = []
    
    for ticker, data in companies_data.items():
        ratios = data['ratios']
        tickers.append(ticker)
        revenues.append(ratios.revenue)
        market_caps.append(ratios.market_cap)
        roas.append(ratios.ROA)
        roes.append(ratios.ROE)
        profit_margins.append(ratios.net_profit_margin)
        gross_margins.append(ratios.gross_margin)
        current_ratios.append(ratios.current_ratio)
        debt_ratios.append(ratios.debt_to_equity)
    
    # Create profitability comparison (percentage-based, naturally comparable)
    fig_profit = go.Figure()
    
    x_pos = np.arange(len(tickers))
    
    fig_profit.add_trace(go.Bar(
        name='ROA (%)',
        x=tickers,
        y=roas,
        marker_color='lightblue',
        yaxis='y',
        offsetgroup=1
    ))
    
    fig_profit.add_trace(go.Bar(
        name='ROE (%)',
        x=tickers,
        y=roes,
        marker_color='lightgreen',
        yaxis='y',
        offsetgroup=2
    ))
    
    fig_profit.add_trace(go.Bar(
        name='Net Profit Margin (%)',
        x=tickers,
        y=profit_margins,
        marker_color='lightcoral',
        yaxis='y',
        offsetgroup=3
    ))
    
    fig_profit.add_trace(go.Bar(
        name='Gross Margin (%)',
        x=tickers,
        y=gross_margins,
        marker_color='lightyellow',
        yaxis='y',
        offsetgroup=4
    ))
    
    fig_profit.update_layout(
        title="Profitability Metrics Comparison",
        xaxis_title="Companies",
        yaxis_title="Percentage (%)",
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    # Create size comparison with dual axis (log scale for better comparison)
    fig_size = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Revenue Comparison", "Market Cap Comparison"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Revenue chart (log scale to handle different sizes)
    fig_size.add_trace(
        go.Bar(
            name='Revenue ($B)',
            x=tickers,
            y=revenues,
            marker_color='steelblue',
            text=[f"${r:.1f}B" for r in revenues],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Market cap chart (log scale)
    fig_size.add_trace(
        go.Bar(
            name='Market Cap ($B)',
            x=tickers,
            y=market_caps,
            marker_color='darkgreen',
            text=[f"${mc:.1f}B" for mc in market_caps],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # Use log scale for better comparison
    fig_size.update_yaxes(type="log", row=1, col=1, title_text="Revenue ($B, log scale)")
    fig_size.update_yaxes(type="log", row=1, col=2, title_text="Market Cap ($B, log scale)")
    
    fig_size.update_layout(
        title="Company Size Comparison (Log Scale)",
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig_profit, fig_size

def create_score_radar_chart(analyses):
    """Create radar chart showing scores from different agents"""
    
    categories = []
    scores = []
    
    for analysis in analyses:
        if analysis.agent_role == AgentRole.PROFITABILITY:
            categories.append("Profitability")
            scores.append(analysis.score)
        elif analysis.agent_role == AgentRole.LIQUIDITY:
            categories.append("Liquidity")
            scores.append(analysis.score)
        elif analysis.agent_role == AgentRole.TRENDS:
            categories.append("Technical")
            scores.append(analysis.score)
        elif analysis.agent_role == AgentRole.NEWS:
            categories.append("Sentiment")
            scores.append(analysis.score)
    
    if not categories:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Agent Scores',
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=False,
        title="Multi-Agent Analysis Scores",
        height=400
    )
    
    return fig

def create_trend_chart(trend_data):
    """Create price trend visualization"""
    if not trend_data:
        return None
    
    # Create sample price chart (would need historical data for full implementation)
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    # Simulate price movement based on trend data
    base_price = trend_data.current_price
    price_change_30d = trend_data.price_change_30d / 100
    
    # Simple simulation for visualization
    prices = []
    current_price = base_price / (1 + price_change_30d)  # Price 30 days ago
    
    for i in range(30):
        daily_change = (price_change_30d / 30) + np.random.normal(0, 0.01)
        current_price *= (1 + daily_change)
        prices.append(current_price)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add moving averages
    fig.add_hline(
        y=trend_data.moving_avg_20,
        line_dash="dash",
        line_color="orange",
        annotation_text="20-day MA"
    )
    
    fig.add_hline(
        y=trend_data.moving_avg_50,
        line_dash="dot",
        line_color="red",
        annotation_text="50-day MA"
    )
    
    fig.update_layout(
        title=f"{trend_data.ticker} Price Trend (30 days)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template='plotly_white',
        height=400
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🤖 FinAgent - Multi-Agent Financial Analysis")
    st.markdown("*Powered by MLX DeepSeek and free financial data sources*")
    
    # Initialize session state
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = FinancialAnalysisOrchestrator()
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    orchestrator = st.session_state.orchestrator
    
    # ============================================================================
    # SIDEBAR
    # ============================================================================
    
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Model status
        status = orchestrator.get_model_status()
        if "✅" in status:
            st.success(status)
        else:
            st.info(status)
            st.markdown("""
            **MLX Model Info:**
            - Model: DeepSeek-R1-Distill-Qwen-14B-4bit
            - Path: ~/mlx-models
            - Will auto-load on first analysis
            """)
        
        st.divider()
        
        # Analysis history
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
        show_detailed_analysis = st.checkbox("Show detailed analysis", value=True)
        auto_refresh = st.checkbox("Auto-refresh data", value=False)
        
        if auto_refresh:
            st.info("Auto-refresh enabled (30s)")
    
    # ============================================================================
    # MAIN INTERFACE
    # ============================================================================
    
    # Input section
    st.header("🎯 Stock Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ticker = st.text_input(
            "Enter Stock Ticker(s)",
            value=st.session_state.get('selected_ticker', ''),
            placeholder="e.g., AAPL or AAPL,TSLA,MSFT (comma-separated for comparison)"
        ).upper()
    
    with col2:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    with col3:
        compare_mode = st.toggle("Compare Mode", value=False)
    
    # Quick select buttons
    st.markdown("**Quick Select:**")
    quick_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
    
    cols = st.columns(len(quick_tickers))
    for i, quick_ticker in enumerate(quick_tickers):
        with cols[i]:
            if st.button(quick_ticker, key=f"quick_{quick_ticker}"):
                st.session_state.selected_ticker = quick_ticker
                st.rerun()
    
    # Analysis execution
    if analyze_button and ticker:
        tickers = [t.strip() for t in ticker.split(',') if t.strip()]
        
        if len(tickers) > 5:
            st.error("Maximum 5 companies can be analyzed at once")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, single_ticker in enumerate(tickers):
            try:
                status_text.text(f"🤖 Analyzing {single_ticker}... ({i+1}/{len(tickers)})")
                progress_bar.progress((i + 0.5) / len(tickers))
                
                # Run analysis
                ratios, analyses = orchestrator.run_full_analysis(single_ticker)
                
                # Store results
                st.session_state.analysis_results[single_ticker] = {
                    'ratios': ratios,
                    'analyses': analyses,
                    'timestamp': datetime.now()
                }
                
                progress_bar.progress((i + 1) / len(tickers))
                
            except Exception as e:
                st.error(f"Error analyzing {single_ticker}: {str(e)}")
                continue
        
        status_text.text("✅ Analysis Complete!")
        progress_bar.empty()
        status_text.empty()
    
    # ============================================================================
    # RESULTS DISPLAY
    # ============================================================================
    
    if st.session_state.analysis_results:
        
        if compare_mode and len(st.session_state.analysis_results) > 1:
            # COMPARISON MODE
            st.header("📊 Multi-Company Comparison")
            
            # Create comparison charts
            fig_profit, fig_size = create_normalized_comparison_chart(st.session_state.analysis_results)
            
            if fig_profit and fig_size:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_profit, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_size, use_container_width=True)
            
            # Comparison table
            st.subheader("📋 Key Metrics Comparison")
            
            comparison_data = []
            for ticker, data in st.session_state.analysis_results.items():
                ratios = data['ratios']
                analyses = data['analyses']
                
                # Get overall score
                supervisor_analysis = next((a for a in analyses if a.agent_role == AgentRole.SUPERVISOR), None)
                overall_score = supervisor_analysis.score if supervisor_analysis else 0
                
                comparison_data.append({
                    'Company': ticker,
                    'Revenue ($B)': f"{ratios.revenue:.1f}",
                    'Market Cap ($B)': f"{ratios.market_cap:.1f}",
                    'ROA (%)': f"{ratios.ROA:.2f}",
                    'ROE (%)': f"{ratios.ROE:.2f}",
                    'Net Margin (%)': f"{ratios.net_profit_margin:.2f}",
                    'Current Ratio': f"{ratios.current_ratio:.2f}",
                    'Overall Score': f"{overall_score:.1f}/10",
                    'Grade': get_grade_from_score(overall_score)
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
        
        else:
            # SINGLE COMPANY DETAILED VIEW
            if len(st.session_state.analysis_results) == 1:
                ticker = list(st.session_state.analysis_results.keys())[0]
            else:
                ticker = st.selectbox("Select company for detailed view:", 
                                    list(st.session_state.analysis_results.keys()))
            
            if ticker in st.session_state.analysis_results:
                data = st.session_state.analysis_results[ticker]
                ratios = data['ratios']
                analyses = data['analyses']
                
                # Header with key metrics
                st.header(f"📈 {ticker} - Financial Analysis")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Revenue", format_currency(ratios.revenue * 1e9), f"Year: {ratios.year}")
                
                with col2:
                    st.metric("Market Cap", format_currency(ratios.market_cap * 1e9))
                
                # Get individual scores
                prof_analysis = next((a for a in analyses if a.agent_role == AgentRole.PROFITABILITY), None)
                liq_analysis = next((a for a in analyses if a.agent_role == AgentRole.LIQUIDITY), None)
                trend_analysis = next((a for a in analyses if a.agent_role == AgentRole.TRENDS), None)
                supervisor_analysis = next((a for a in analyses if a.agent_role == AgentRole.SUPERVISOR), None)
                
                with col3:
                    if prof_analysis:
                        st.metric("Profitability", f"{prof_analysis.score:.1f}/10", 
                                help="ROA, ROE, Profit Margins")
                
                with col4:
                    if liq_analysis:
                        st.metric("Liquidity", f"{liq_analysis.score:.1f}/10",
                                help="Current Ratio, Quick Ratio, Debt/Equity")
                
                with col5:
                    if supervisor_analysis:
                        grade = get_grade_from_score(supervisor_analysis.score)
                        st.metric("Overall Grade", grade, f"{supervisor_analysis.score:.1f}/10")
                
                # Detailed tabs
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 Overview", "💰 Profitability", "💧 Liquidity", 
                    "📈 Technical", "📰 News", "📋 Executive Summary"
                ])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Radar chart
                        radar_fig = create_score_radar_chart(analyses)
                        if radar_fig:
                            st.plotly_chart(radar_fig, use_container_width=True)
                    
                    with col2:
                        # Key ratios table
                        st.subheader("Key Financial Ratios")
                        ratios_df = pd.DataFrame({
                            'Metric': ['ROA', 'ROE', 'Net Profit Margin', 'Gross Margin', 
                                     'Current Ratio', 'Quick Ratio', 'Debt-to-Equity'],
                            'Value': [f"{ratios.ROA:.2f}%", f"{ratios.ROE:.2f}%", 
                                    f"{ratios.net_profit_margin:.2f}%", f"{ratios.gross_margin:.2f}%",
                                    f"{ratios.current_ratio:.2f}", f"{ratios.quick_ratio:.2f}",
                                    f"{ratios.debt_to_equity:.2f}"],
                            'Status': ['Excellent' if ratios.ROA > 15 else 'Good' if ratios.ROA > 5 else 'Fair',
                                     'Excellent' if ratios.ROE > 20 else 'Good' if ratios.ROE > 15 else 'Fair',
                                     'Excellent' if ratios.net_profit_margin > 15 else 'Good' if ratios.net_profit_margin > 10 else 'Fair',
                                     'Excellent' if ratios.gross_margin > 50 else 'Good' if ratios.gross_margin > 40 else 'Fair',
                                     'Good' if 1.5 <= ratios.current_ratio <= 3.0 else 'Warning',
                                     'Good' if ratios.quick_ratio >= 1.0 else 'Warning',
                                     'Good' if 0.3 <= ratios.debt_to_equity <= 1.5 else 'Warning']
                        })
                        st.dataframe(ratios_df, use_container_width=True)
                
                with tab2:
                    if prof_analysis:
                        st.header("💰 Profitability Analysis")
                        st.markdown(f"**Score: {prof_analysis.score:.1f}/10**")
                        st.markdown("---")
                        st.markdown(prof_analysis.analysis)
                    else:
                        st.warning("Profitability analysis not available")
                
                with tab3:
                    if liq_analysis:
                        st.header("💧 Liquidity Analysis")
                        st.markdown(f"**Score: {liq_analysis.score:.1f}/10**")
                        st.markdown("---")
                        st.markdown(liq_analysis.analysis)
                    else:
                        st.warning("Liquidity analysis not available")
                
                with tab4:
                    if trend_analysis:
                        st.header("📈 Technical Analysis")
                        st.markdown(f"**Score: {trend_analysis.score:.1f}/10**")
                        st.markdown("---")
                        
                        # Show trend chart if data available
                        if hasattr(trend_analysis, 'data') and trend_analysis.data:
                            trend_fig = create_trend_chart(trend_analysis.data)
                            if trend_fig:
                                st.plotly_chart(trend_fig, use_container_width=True)
                        
                        st.markdown(trend_analysis.analysis)
                    else:
                        st.warning("Technical analysis not available")
                
                with tab5:
                    news_analysis = next((a for a in analyses if a.agent_role == AgentRole.NEWS), None)
                    if news_analysis:
                        st.header("📰 News & Sentiment Analysis")
                        st.markdown(f"**Score: {news_analysis.score:.1f}/10**")
                        st.markdown("---")
                        st.markdown(news_analysis.analysis)
                    else:
                        st.warning("News analysis not available")
                
                with tab6:
                    if supervisor_analysis:
                        st.header("📋 Executive Investment Summary")
                        st.markdown(f"**Overall Score: {supervisor_analysis.score:.1f}/10 (Grade: {get_grade_from_score(supervisor_analysis.score)})**")
                        st.markdown("---")
                        st.markdown(supervisor_analysis.analysis)
                        
                        # Download report
                        if st.button("📥 Download Full Report"):
                            report_content = f"""
FINAGENT COMPREHENSIVE ANALYSIS REPORT
=====================================
Company: {ticker}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Overall Grade: {get_grade_from_score(supervisor_analysis.score)} ({supervisor_analysis.score:.1f}/10)

EXECUTIVE SUMMARY
{supervisor_analysis.analysis}

DETAILED ANALYSES
{'-'*50}

PROFITABILITY ANALYSIS
{prof_analysis.analysis if prof_analysis else 'Not available'}

LIQUIDITY ANALYSIS  
{liq_analysis.analysis if liq_analysis else 'Not available'}

TECHNICAL ANALYSIS
{trend_analysis.analysis if trend_analysis else 'Not available'}

Generated by FinAgent Multi-Agent Analysis System
                            """
                            
                            st.download_button(
                                label="Download Report",
                                data=report_content,
                                file_name=f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain"
                            )
                    else:
                        st.warning("Executive summary not available")
    
    else:
        # Welcome message
        st.info("👆 Enter a stock ticker above to begin multi-agent financial analysis")
        
        st.markdown("""
        ### 🤖 How FinAgent Works:
        
        1. **🔍 Fundamental Agent**: Extracts financial ratios from free data sources
        2. **💰 Profitability Agent**: Analyzes ROA, ROE, and profit margins  
        3. **💧 Liquidity Agent**: Evaluates financial stability and debt levels
        4. **📈 Technical Agent**: Analyzes price trends and technical indicators
        5. **📰 News Agent**: Sentiment analysis (coming soon with API integration)
        6. **👔 Supervisor Agent**: Synthesizes all analyses into investment recommendations
        
        **Features:**
        - ✅ 100% Free (MLX + yfinance)
        - 🔒 Private (runs locally)
        - 📊 Professional visualizations
        - 📋 Downloadable reports
        - 🆚 Multi-company comparisons
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    **FinAgent v2.0** - Multi-Agent Financial Analysis System  
    🔧 *Built with MLX DeepSeek, Streamlit, and yfinance*  
    📊 *Professional financial analysis, completely free*
    """)

if __name__ == "__main__":
    main()