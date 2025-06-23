import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def get_stock_data(ticker):
    data = yf.download(ticker, period="6mo", interval="1d")
    data["MA50"] = data["Close"].rolling(window=50).mean()
    data["MA200"] = data["Close"].rolling(window=200).mean()
    
    # MACD calculation
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # Buy/Sell signal
    data["Buy_Signal"] = (data["MA50"] > data["MA200"]) & (data["MACD"] > data["Signal"])

    return data

def score_stock(data):
    if data["Buy_Signal"].iloc[-1]:
        return 1  # Strong signal
    else:
        return 0  # Weak

def allocate_funds(scores, capital):
    scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_stock = scores_sorted[0][0]
    return {best_stock: capital}

st.title("📈 Smart Stock Allocation Tool")

tickers_input = st.text_input("Enter stock tickers (comma-separated):", "AAPL, MSFT, TSLA")
capital = st.number_input("Amount to invest (USD):", min_value=100.0, value=1000.0)

if st.button("Analyze and Allocate"):
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    scores = {}
    stock_data_dict = {}

    for ticker in tickers:
        try:
            data = get_stock_data(ticker)
            score = score_stock(data)
            scores[ticker] = score
            stock_data_dict[ticker] = data
        except Exception as e:
            st.error(f"Error loading {ticker}: {e}")
    
    if scores:
        allocation = allocate_funds(scores, capital)
        st.subheader("📊 Allocation Recommendation")
        st.write(allocation)

        for ticker, data in stock_data_dict.items():
            st.subheader(f"📉 {ticker} - Chart & Indicators")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data.index, data["Close"], label="Close", color="blue")
            ax.plot(data.index, data["MA50"], label="MA50", color="green")
            ax.plot(data.index, data["MA200"], label="MA200", color="red")
            ax.set_title(f"{ticker} Price Chart")
            ax.legend()
            st.pyplot(fig)

            fig_macd, ax_macd = plt.subplots(figsize=(10, 2))
            ax_macd.plot(data.index, data["MACD"], label="MACD", color="black")
            ax_macd.plot(data.index, data["Signal"], label="Signal", color="orange")
            ax_macd.set_title(f"{ticker} MACD")
            ax_macd.legend()
            st.pyplot(fig_macd)

            st.markdown(f"**Buy Signal (last date):** {data['Buy_Signal'].iloc[-1]}")
    else:
        st.warning("No valid stock data found.")
