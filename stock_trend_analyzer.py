import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Stock Analyzer", layout="wide")

# ------------------ TITLE ------------------
st.title("📊 Stock Trend Analyzer")

# ------------------ SESSION STATE ------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

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

    if data.empty or len(data) < 2:
        st.error("❌ Not enough data.")
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["📊 Charts", "📉 Indicators", "🧠 Analysis", "💰 Portfolio", "📰 News", "🤖 Prediction"]
        )

        # ================== TAB 1: CHARTS ==================
        with tab1:
            st.subheader("📊 Charts")

            stock = st.selectbox("Select stock for chart", selected_stocks)
            df = yf.download(stock, start=start_date, end=end_date)

            # Candlestick Chart
            st.subheader("🕯️ Candlestick Chart")
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )])

            fig.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # Line Chart
            st.subheader("📈 Price Comparison")
            st.line_chart(data)

            # Normalized
            normalized = data / data.iloc[0] * 100
            st.subheader("⚖️ Performance Comparison")
            st.line_chart(normalized)

            # Metrics
            st.subheader("📊 Latest Prices")
            cols = st.columns(len(selected_stocks))

            for i, s in enumerate(selected_stocks):
                latest = data[s].iloc[-1]
                prev = data[s].iloc[-2]
                change = latest - prev

                if change > 0:
                    cols[i].metric(s, round(latest, 2), f"+{round(change,2)} 🟢")
                else:
                    cols[i].metric(s, round(latest, 2), f"{round(change,2)} 🔴")

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

            for s in selected_stocks:
                ma5 = data[s].rolling(5).mean().iloc[-1]
                ma20 = data[s].rolling(20).mean().iloc[-1]

                if ma5 > ma20:
                    st.write(f"{s}: 📈 Uptrend")
                elif ma5 < ma20:
                    st.write(f"{s}: 📉 Downtrend")
                else:
                    st.write(f"{s}: 😐 Sideways")

            st.subheader("📊 Buy/Sell Signals")

            for s in selected_stocks:
                ma5 = data[s].rolling(5).mean().iloc[-1]
                ma20 = data[s].rolling(20).mean().iloc[-1]

                if ma5 > ma20:
                    st.success(f"{s}: 🟢 BUY Signal")
                else:
                    st.error(f"{s}: 🔴 SELL Signal")

        # ================== TAB 4: PORTFOLIO ==================
        with tab4:
            st.subheader("💰 Portfolio Manager")

            col1, col2 = st.columns(2)

            with col1:
                money = st.number_input("Investment Amount", min_value=1000, value=10000)

            with col2:
                buy_stock = st.selectbox("Select Stock", selected_stocks)

            if st.button("Buy Stock"):
                st.session_state.portfolio.append({
                    "stock": buy_stock,
                    "amount": money
                })
                st.success(f"Added {buy_stock}")

            if st.button("Clear Portfolio"):
                st.session_state.portfolio = []

            st.subheader("📊 Your Portfolio")

            if len(st.session_state.portfolio) == 0:
                st.write("No investments yet.")
            else:
                total_value = 0

                for item in st.session_state.portfolio:
                    stock = item["stock"]
                    invested = item["amount"]

                    if stock in data.columns:
                        current_price = data[stock].iloc[-1]
                        initial_price = data[stock].iloc[0]

                        shares = invested / initial_price
                        current_value = shares * current_price
                        profit = current_value - invested

                        total_value += current_value

                        if profit > 0:
                            st.success(f"{stock}: ₹{round(current_value,2)} (+{round(profit,2)}) 🟢")
                        else:
                            st.error(f"{stock}: ₹{round(current_value,2)} ({round(profit,2)}) 🔴")

                st.info(f"💼 Total Value: ₹{round(total_value,2)}")

        # ================== TAB 5: NEWS ==================
        with tab5:
            st.subheader("📰 Latest News")

            api_key = "YOUR_API_KEY_HERE"

            news_stock = st.selectbox("Select stock for news", selected_stocks)

            url = f"https://newsapi.org/v2/everything?q={news_stock} stock&apiKey={api_key}"

            try:
                response = requests.get(url)
                news_data = response.json()

                articles = news_data.get("articles", [])

                if len(articles) == 0:
                    st.write("No news found.")
                else:
                    for article in articles[:5]:
                        st.markdown(f"### {article['title']}")
                        st.write(f"📰 {article['source']['name']}")
                        st.markdown(f"[Read more]({article['url']})")
                        st.write("---")
            except:
                st.error("Error fetching news")

        # ================== TAB 6: ML PREDICTION ==================
        with tab6:
            st.subheader("🤖 Price Prediction")

            stock = selected_stocks[0]

            df = data[stock].reset_index()
            df['Days'] = np.arange(len(df))

            X = df[['Days']]
            y = df[stock]

            model = LinearRegression()
            model.fit(X, y)

            future_days = 30
            future_X = np.arange(len(df), len(df) + future_days).reshape(-1, 1)

            predictions = model.predict(future_X)

            st.line_chart(predictions)
