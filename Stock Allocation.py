import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Technical Indicators
def add_indicators(df):
    df = df.copy()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    
    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ✅ Bollinger Bands - Clean and safe
    middle_band = df["Close"].rolling(window=20).mean()
    std_dev = df["Close"].rolling(window=20).std()
    df["BB_Middle"] = middle_band
    df["BB_Upper"] = middle_band + (2 * std_dev)
    df["BB_Lower"] = middle_band - (2 * std_dev)

    df.dropna(inplace=True)
    return df

# ML Model
def train_ml_model(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['MA50', 'MA200', 'MACD', 'Signal', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower']
    df = df.dropna()
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    prediction = model.predict_proba([X.iloc[-1]])[0][1]  # probability of going up
    
    return acc, prediction

# Allocation logic
def allocate(capital, predictions):
    sorted_pred = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    total_weight = sum([p for _, p in sorted_pred])
    allocation = {}
    for stock, pred in sorted_pred:
        weight = pred / total_weight if total_weight > 0 else 1 / len(predictions)
        allocation[stock] = round(capital * weight, 2)
    return allocation

# Main Streamlit App
st.title("📈 Enhanced Smart Stock Allocation")

tickers_input = st.text_input("Enter stock tickers (comma-separated):", "AAPL, MSFT, TSLA")
capital = st.number_input("Investment Capital (USD):", min_value=100.0, value=1000.0)

if st.button("Analyze & Allocate"):
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    predictions = {}
    accuracies = {}
    data_cache = {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="6mo", interval="1d")
            df = add_indicators(df)
            acc, pred = train_ml_model(df)
            predictions[ticker] = pred
            accuracies[ticker] = acc
            data_cache[ticker] = df
        except Exception as e:
            st.error(f"{ticker} failed: {e}")
    
    if predictions:
        allocation = allocate(capital, predictions)
        st.subheader("📊 Allocation Recommendation")
        st.write(allocation)

        st.subheader("📉 ML Confidence & Accuracy")
        st.dataframe(pd.DataFrame({
            "ML Prediction Confidence (prob of ↑)": predictions,
            "Model Accuracy (last 6mo)": accuracies
        }).round(2))

        for ticker, df in data_cache.items():
            st.markdown(f"### {ticker} Chart & Indicators")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(, label="Close")
            ax.plot(df["MA50"], label="MA50")
            ax.plot(df["MA200"], label="MA200")
            ax.plot(df["BB_Upper"], label="BB Upper", linestyle='--', color='grey')
            ax.plot(df["BB_Lower"], label="BB Lower", linestyle='--', color='grey')
            ax.set_title(f"{ticker} Price + Indicators")
            ax.legend()
            st.pyplot(fig)

            fig_rsi, ax_rsi = plt.subplots(figsize=(10, 2))
            ax_rsi.plot(df["RSI"], label="RSI", color="purple")
            ax_rsi.axhline(70, linestyle='--', color='red')
            ax_rsi.axhline(30, linestyle='--', color='green')
            ax_rsi.set_title(f"{ticker} RSI")
            ax_rsi.legend()
            st.pyplot(fig_rsi)
    else:
        st.warning("No valid data or predictions.")
