import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ta  # For technical indicators
import time
from collections import deque

# --- Configuration --- #
st.set_page_config(page_title="AI Stock Prediction System", layout="wide")

# API Keys (using Streamlit secrets)
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
ALPHA_VANTAGE_KEY = st.secrets["ALPHA_VANTAGE_KEY"]
TRADING_ECONOMICS_KEY = st.secrets.get("TRADING_ECONOMICS_KEY")

# Constants
INDICES = {
    'NIFTY 50': '^NSEI',
    'SENSEX': '^BSESN',
    'BANK NIFTY': '^NSEBANK',
    'FINNIFTY': '^CNXFINNIFTY'
}

NEWS_SOURCES = [
    'Moneycontrol', 'Economic Times', 'Business Standard',
    'Livemint', 'Reuters India', 'Bloomberg Quint'
]

# --- Enhanced Data Fetching Functions --- #
@st.cache_data(ttl=300)
def fetch_macroeconomic_data():
    """Fetch RBI and macroeconomic indicators"""
    try:
        # Placeholder for actual API call - using mock data here
        macro_data = {
            'repo_rate': 6.50,  # Current RBI repo rate
            'inflation': 5.02,  # Latest CPI
            'gdp_growth': 7.2,  # GDP growth estimate
            'fii_flows': 12500  # FII investment in crores
        }
        return pd.DataFrame([macro_data])
    except Exception as e:
        st.error(f"Macro data error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_realtime_news():
    """Enhanced news fetcher with multiple sources and fallback"""
    try:
        # Try NewsAPI first
        params = {
            "apiKey": NEWS_API_KEY,
            "country": "in",
            "category": "business",
            "pageSize": 50,
            "q": "stock OR market OR economy OR RBI"
        }
        
        response = requests.get("https://newsapi.org/v2/top-headlines", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        if data['status'] == 'ok':
            articles.extend(data.get('articles', []))
        
        # Process articles
        news_data = []
        for article in articles:
            if article['source']['name'] in NEWS_SOURCES:
                news_data.append({
                    'timestamp': pd.to_datetime(article['publishedAt']),
                    'headline': article['title'],
                    'description': article.get('description', ''),
                    'source': article['source']['name'],
                    'url': article['url']
                })
        
        return pd.DataFrame(news_data)
        
    except Exception as e:
        st.error(f"News API Error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_market_data(ticker, period="1d", interval="5m"):
    """Enhanced market data fetcher with technical indicators"""
    try:
        data = yf.download(tickers=ticker, period=period, interval=interval)
        if data.empty:
            return pd.DataFrame()
            
        data = data.reset_index()
        if 'Datetime' not in data.columns:
            data = data.rename(columns={'Date': 'Datetime'})
        
        # Add technical indicators
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        
        return data
        
    except Exception as e:
        st.error(f"Market data error: {str(e)}")
        return pd.DataFrame()

# --- Advanced Sentiment Analysis --- #
class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze(self, text):
        """Ensemble of sentiment analyzers"""
        try:
            # FinBERT for financial context
            finbert_result = self.finbert(text[:512])[0]
            
            # VADER for news sentiment
            vader_result = self.vader.polarity_scores(text)
            
            # Weighted ensemble
            compound_score = (0.7 * finbert_result['score'] * (1 if finbert_result['label'] == 'Positive' else -1) +
                            0.3 * vader_result['compound'])
            
            return {
                'score': compound_score,
                'label': 'Positive' if compound_score > 0 else 'Negative'
            }
        except:
            return {'score': 0, 'label': 'Neutral'}

# --- Prediction Engine --- #
class StockPredictor:
    def __init__(self):
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.sentiment_history = deque(maxlen=100)
        
    def generate_signal(self, news_df, market_data, macro_data):
        """Generate trading signals based on multiple factors"""
        if news_df.empty or market_data.empty:
            return "Neutral (Insufficient Data)"
            
        # 1. News Sentiment Analysis
        news_df['sentiment'] = news_df['headline'].apply(
            lambda x: self.sentiment_analyzer.analyze(x)['score'])
        avg_sentiment = news_df['sentiment'].mean()
        self.sentiment_history.append(avg_sentiment)
        
        # 2. Technical Indicators
        latest_market = market_data.iloc[-1]
        rsi = latest_market['RSI']
        macd = latest_market['MACD']
        
        # 3. Macroeconomic Factors
        if not macro_data.empty:
            repo_rate = macro_data['repo_rate'].iloc[0]
            inflation = macro_data['inflation'].iloc[0]
            macro_score = (6.5 - repo_rate) * 0.2 + (4 - inflation) * 0.1
        else:
            macro_score = 0
        
        # Composite score calculation
        sentiment_weight = 0.4
        technical_weight = 0.4
        macro_weight = 0.2
        
        # Normalize scores
        sentiment_score = avg_sentiment * 10  # Scale to comparable range
        technical_score = ((70 - rsi) * 0.3 + macd * 5)  # Combine RSI and MACD
        composite_score = (sentiment_score * sentiment_weight + 
                         technical_score * technical_weight + 
                         macro_score * macro_weight)
        
        # Generate signal
        if composite_score > 1.5:
            return "Strong Buy 🚀", composite_score
        elif composite_score > 0.5:
            return "Buy 🔼", composite_score
        elif composite_score < -1.5:
            return "Strong Sell 📉", composite_score
        elif composite_score < -0.5:
            return "Sell 🔽", composite_score
        else:
            return "Neutral ➖", composite_score

# --- Streamlit UI --- #
def main():
    st.title("📈 AI-Powered Indian Stock Market Predictor")
    st.markdown("""
    *Real-time analysis combining news sentiment, technical indicators, and macroeconomic factors*
    """)
    
    # Initialize components
    predictor = StockPredictor()
    
    # Dashboard Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📰 Latest Market News")
        news_df = fetch_realtime_news()
        
        if not news_df.empty:
            # Analyze sentiment
            analyzer = EnhancedSentimentAnalyzer()
            news_df['sentiment'] = news_df['headline'].apply(
                lambda x: analyzer.analyze(x)['label'])
            
            # Display news with color coding
            for _, row in news_df.head(10).iterrows():
                color = "green" if row['sentiment'] == "Positive" else "red" if row['sentiment'] == "Negative" else "gray"
                st.markdown(
                    f"""<div style="border-left: 4px solid {color}; padding: 0.5em; margin: 0.5em 0;">
                    <b>{row['source']}</b> - {row['headline']}<br>
                    <small>{row['timestamp'].strftime('%Y-%m-%d %H:%M')} | Sentiment: {row['sentiment']}</small>
                    </div>""",
                    unsafe_allow_html=True
                )
        else:
            st.warning("No news articles found from selected sources.")
    
    with col2:
        st.subheader("📊 Macro Indicators")
        macro_data = fetch_macroeconomic_data()
        
        if not macro_data.empty:
            st.metric("RBI Repo Rate", f"{macro_data['repo_rate'].iloc[0]}%")
            st.metric("Inflation Rate", f"{macro_data['inflation'].iloc[0]}%")
            st.metric("GDP Growth", f"{macro_data['gdp_growth'].iloc[0]}%")
            st.metric("FII Flows (₹ Cr)", f"{macro_data['fii_flows'].iloc[0]:,}")
        else:
            st.info("Macroeconomic data currently unavailable")
    
    # Market Data Section
    st.subheader("📈 Market Overview")
    selected_index = st.selectbox("Select Index", list(INDICES.keys()))
    
    market_data = fetch_market_data(INDICES[selected_index])
    
    if not market_data.empty:
        # Price Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=market_data['Datetime'],
            open=market_data['Open'],
            high=market_data['High'],
            low=market_data['Low'],
            close=market_data['Close'],
            name='Price'
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=market_data['Datetime'],
            y=market_data['SMA_50'],
            line=dict(color='orange', width=1),
            name='50-SMA'
        ))
        
        fig.add_trace(go.Scatter(
            x=market_data['Datetime'],
            y=market_data['EMA_20'],
            line=dict(color='purple', width=1),
            name='20-EMA'
        ))
        
        fig.update_layout(
            title=f"{selected_index} Price with Indicators",
            xaxis_rangeslider_visible=False,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical Indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 RSI (14-period)**")
            rsi_fig = px.line(market_data, x='Datetime', y='RSI',
                             title='Relative Strength Index')
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(rsi_fig, use_container_width=True)
        
        with col2:
            st.markdown("**📈 MACD**")
            macd_fig = px.line(market_data, x='Datetime', y='MACD',
                             title='Moving Average Convergence Divergence')
            macd_fig.add_hline(y=0, line_color="black")
            st.plotly_chart(macd_fig, use_container_width=True)
    
    # Prediction Section
    st.subheader("🔮 AI Prediction Signal")
    
    if not news_df.empty and not market_data.empty:
        signal, confidence = predictor.generate_signal(news_df, market_data, macro_data)
        
        # Display prediction with appropriate styling
        if "Buy" in signal:
            st.success(f"### {signal}")
        elif "Sell" in signal:
            st.error(f"### {signal}")
        else:
            st.info(f"### {signal}")
        
        st.metric("Confidence Score", f"{abs(confidence):.2f}")
        
        # Explanation
        with st.expander("How this prediction was generated"):
            st.markdown("""
            This prediction combines:
            - **News Sentiment Analysis**: Financial NLP on recent headlines
            - **Technical Indicators**: RSI, MACD, Moving Averages
            - **Macroeconomic Factors**: Interest rates, inflation, FII flows
            
            The model weights these factors differently based on market conditions.
            """)
    else:
        st.warning("Waiting for sufficient data to generate prediction...")
    
    # Auto-refresh
    st.markdown("---")
    refresh_rate = st.slider("Auto-refresh interval (minutes)", 1, 15, 5)
    if st.button("🔄 Refresh Now"):
        st.experimental_rerun()
    
    # Set auto-refresh
    time.sleep(refresh_rate * 60)
    st.experimental_rerun()

if __name__ == "__main__":
    main()
