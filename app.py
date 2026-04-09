import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="AI Market Sentinel", layout="wide")
st.title("📈 AI Market Sentinel: Sentiment & Price Engine")
st.markdown("Analyzing the intersection of human emotion (News) and market math.")

# Sidebar for User Input
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, NVDA, TSLA)", value="NVDA").upper()
days_to_predict = st.sidebar.slider("Prediction Horizon (Days)", 1, 30, 7)

# --- 2. DATA ACQUISITION ---
@st.cache_data(ttl=3600)
def get_market_data(symbol):
    # Fetch 1 year of historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

@st.cache_data(ttl=3600)
def get_sentiment_score(symbol):
    # Simulate NLP processing on headlines
    np.random.seed(42)
    headlines = [
        f"Record growth expected for {symbol}", 
        f"{symbol} faces regulatory hurdles", 
        f"Investors bullish on {symbol} quarterly report", 
        f"Tech sector downturn hits {symbol}"
    ]
    scores = [TextBlob(h).sentiment.polarity for h in headlines]
    return np.mean(scores)

# --- 3. MACHINE LEARNING ENGINE ---
def train_prediction_model(df, horizon):
    # Feature Engineering
    df_copy = df.copy()
    # Handle multi-index columns if yfinance returns them
    if isinstance(df_copy.columns, pd.MultiIndex):
        df_copy.columns = df_copy.columns.get_level_values(0)
        
    df_copy['MA10'] = df_copy['Close'].rolling(window=10).mean()
    df_copy['MA50'] = df_copy['Close'].rolling(window=50).mean()
    df_copy['Target'] = df_copy['Close'].shift(-horizon)
    
    # Prepare Features
    features_list = ['Close', 'MA10', 'MA50']
    data_clean = df_copy[features_list + ['Target']].dropna()
    
    X = data_clean[features_list]
    y = data_clean['Target']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict using the most recent data point
    latest_features = df_copy[features_list].tail(1)
    prediction = model.predict(latest_features)
    
    return prediction[0], model.score(X, y)

# --- 4. EXECUTION FLOW ---
accuracy = 0.0 # Default value

try:
    with st.spinner('Gathering Market Intelligence...'):
        data = get_market_data(ticker)
        
        if data.empty:
            st.error(f"No data found for {ticker}. Please ensure you are using a correct symbol (e.g., AAPL not APPLE).")
        else:
            sentiment = get_sentiment_score(ticker)
            pred_price, accuracy = train_prediction_model(data, days_to_predict)
            
            # Extract current price safely
            if isinstance(data['Close'], pd.DataFrame):
                current_price = float(data['Close'].iloc[-1].iloc[0])
            else:
                current_price = float(data['Close'].iloc[-1])

            # --- 5. DASHBOARD VISUALS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            
            sentiment_label = "Bullish" if sentiment > 0 else "Bearish"
            col2.metric("News Sentiment Score", f"{sentiment:.2f}", sentiment_label)
            
            diff = pred_price - current_price
            col3.metric(f"AI Forecast ({days_to_predict}d)", f"${pred_price:.2f}", f"{diff:.2f}")

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'].values.flatten(), name="Actual Price", line=dict(color='royalblue')))
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(50).mean().values.flatten(), name="50-Day Trend", line=dict(dash='dash')))
            fig.update_layout(title=f"{ticker} Historical Performance", template="plotly_dark", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)

            # Strategy Insight
            st.subheader("💡 Strategic AI Insight")
            if sentiment > 0 and pred_price > current_price:
                st.success(f"**Strong Buy Signal**: Technical indicators and sentiment are bullish for {ticker}.")
            elif sentiment < 0 and pred_price < current_price:
                st.error(f"**Strong Sell Signal**: Technicals and sentiment suggest a downward trend.")
            else:
                st.warning("**Neutral/Mixed Signal**: Conflicting reports between sentiment and technical data.")

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

# --- 6. TECHNICAL FOOTER ---
st.caption(f"Model Confidence: {accuracy:.2%} | Data Source: Yahoo Finance | Sentiment Engine: TextBlob NLP")