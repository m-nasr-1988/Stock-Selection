import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Stock Allocator", layout="wide")

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
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    mean = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["BB_Middle"] = mean
    df["BB_Upper"] = mean + 2 * std
    df["BB_Lower"] = mean - 2 * std

    df.dropna(inplace=True)
    return df

def get_stock_score(df):
    df = add_indicators(df)

    if df.shape[0] < 50:
        raise ValueError("Not enough data to make a reliable prediction.")

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    features = ["MA50", "MA200", "MACD", "Signal", "RSI", "BB_Middle", "BB_Upper", "BB_Lower"]
    X = df[features]
    y = df["Target"]

    if len(X) < 10:
        raise ValueError("Not enough samples for training.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    prediction = model.predict(X.tail(1))[0]
    score = model.predict_proba(X.tail(1))[0][1]

    return prediction, score

# Streamlit UI
st.title("📈 Smart Stock Investment Advisor")

tickers = st.text_input("Enter stock tickers separated by commas (e.g. AAPL, MSFT, TSLA):", "AAPL, MSFT, TSLA")
investment = st.number_input("Enter amount to invest (USD):", min_value=100.0, value=1000.0)

if st.button("Analyze & Allocate"):
    tickers = [t.strip().upper() for t in tickers.split(",")]
    scores = {}
    failed = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, period="6mo")
            if data.empty:
                failed.append(ticker)
                continue
            pred, score = get_stock_score(data)
            if pred == 1:
                scores[ticker] = score
        except Exception as e:
            failed.append(f"{ticker} ({str(e)})")

    if scores:
        total_score = sum(scores.values())
        allocations = {ticker: round((score / total_score) * investment, 2) for ticker, score in scores.items()}

        st.subheader("📊 Investment Allocation Recommendation")
        alloc_df = pd.DataFrame(list(allocations.items()), columns=["Ticker", "Allocated ($)"])
        st.dataframe(alloc_df)

        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie(allocations.values(), labels=allocations.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.warning("No strong buy signals found based on indicators and model predictions.")

    if failed:
        st.error(f"Some tickers failed to process: {', '.join(failed)}")
