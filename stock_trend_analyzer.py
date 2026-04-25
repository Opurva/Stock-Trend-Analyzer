import streamlit as st
import yfinance as yf
import pandas as pd

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Stock Analyzer", layout="wide")

# ------------------ TITLE ------------------
st.title("📊 Stock Trend Analyzer")

# ------------------ SIDEBAR ------------------
st.sidebar.header("User Input")

stock_options = ["AAPL", "MSFT", "GOOGL", "TSLA", "RELIANCE.NS", "TCS.NS"]

selected_stocks = st.sidebar.multiselect(
    "Select Stocks",
    stock_options,
    default=["AAPL"]
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# ------------------ FETCH DATA ------------------
if selected_stocks:
    data = yf.download(selected_stocks, start=start_date, end=end_date)['Close']

    # Handle empty data
    if data.empty:
        st.error("❌ No data found. Try different stocks or date range.")
    else:
        # ------------------ TABS ------------------
        tab1, tab2, tab3 = st.tabs(["📊 Charts", "📉 Indicators", "🧠 Analysis"])

        # ================== TAB 1: CHARTS ==================
        with tab1:
            st.subheader("📈 Price Comparison")
            st.line_chart(data)

            # Normalize
            normalized = data / data.iloc[0] * 100

            st.subheader("⚖️ Performance Comparison")
            st.line_chart(normalized)

            # Metrics
            st.subheader("📊 Latest Prices")
            cols = st.columns(len(selected_stocks))

            for i, stock in enumerate(selected_stocks):
                latest_price = data[stock].iloc[-1]
                cols[i].metric(stock, round(latest_price, 2))

        # ================== TAB 2: INDICATORS ==================
        with tab2:
            stock = selected_stocks[0]

            st.subheader(f"📉 RSI Indicator ({stock})")

            delta = data[stock].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            st.line_chart(rsi)

        # ================== TAB 3: ANALYSIS ==================
        with tab3:
            st.subheader("🧠 Trend Analysis")

            for stock in selected_stocks:
                ma5 = data[stock].rolling(5).mean().iloc[-1]
                ma20 = data[stock].rolling(20).mean().iloc[-1]

                if ma5 > ma20:
                    st.write(f"{stock}: 📈 Uptrend")
                elif ma5 < ma20:
                    st.write(f"{stock}: 📉 Downtrend")
                else:
                    st.write(f"{stock}: 😐 Sideways")

            # Best performer
            normalized = data / data.iloc[0] * 100
            returns = normalized.iloc[-1]
            best_stock = returns.idxmax()

            st.success(f"🏆 Best Performer: {best_stock}")
