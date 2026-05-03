import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from textblob import TextBlob
import plotly.graph_objects as go

# ---------------- PAGE ----------------
st.set_page_config(page_title="Stock Intelligence Platform", layout="wide")
st.title("📊 AI Stock Intelligence Platform")

# ---------------- SESSION ----------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

# ---------------- LOGOS ----------------
logo_urls = {
    "AAPL": "https://logo.clearbit.com/apple.com",
    "MSFT": "https://logo.clearbit.com/microsoft.com",
    "GOOGL": "https://logo.clearbit.com/google.com",
    "TSLA": "https://logo.clearbit.com/tesla.com",
    "RELIANCE.NS": "https://logo.clearbit.com/ril.com",
    "TCS.NS": "https://logo.clearbit.com/tcs.com"
}

# ---------------- SIDEBAR ----------------
st.sidebar.header("User Input")

stock_options = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "TSLA",
    "RELIANCE.NS",
    "TCS.NS"
]

selected_stocks = st.sidebar.multiselect(
    "Select Stocks",
    stock_options,
    default=["AAPL"]
)

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2022-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    pd.to_datetime("2024-12-31")
)

# ---------------- FETCH DATA ----------------
if selected_stocks:

    data = yf.download(
        selected_stocks,
        start=start_date,
        end=end_date
    )["Close"]

    if data.empty:
        st.error("No stock data found.")
        st.stop()

    if len(selected_stocks) == 1:
        data = pd.DataFrame(data)

    tabs = st.tabs([
        "📈 Charts",
        "📉 Indicators",
        "🧠 Analysis",
        "💰 Portfolio",
        "📰 News",
        "🤖 Prediction"
    ])

    # ==========================================================
    # TAB 1
    # ==========================================================
    with tabs[0]:

        st.subheader("Charts")

        chart_stock = st.selectbox(
            "Select stock",
            selected_stocks
        )

        if chart_stock in logo_urls:
            st.image(
                logo_urls[chart_stock],
                width=80
            )

        stock_df = yf.download(
            chart_stock,
            start=start_date,
            end=end_date
        )

        # Candlestick
        st.subheader("🕯 Candlestick Chart")

        fig = go.Figure(data=[
            go.Candlestick(
                x=stock_df.index,
                open=stock_df["Open"],
                high=stock_df["High"],
                low=stock_df["Low"],
                close=stock_df["Close"]
            )
        ])

        fig.update_layout(
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Price comparison
        st.subheader("📈 Price Comparison")
        st.line_chart(data)

        # Normalized comparison
        st.subheader("⚖ Performance Comparison")

        normalized = data / data.iloc[0] * 100
        st.line_chart(normalized)

    # ==========================================================
    # TAB 2
    # ==========================================================
    with tabs[1]:

        indicator_stock = st.selectbox(
            "Choose stock for RSI",
            selected_stocks,
            key="rsi"
        )

        st.subheader(f"RSI - {indicator_stock}")

        delta = data[indicator_stock].diff()

        gain = delta.clip(
            lower=0
        ).rolling(14).mean()

        loss = -delta.clip(
            upper=0
        ).rolling(14).mean()

        rs = gain / loss

        rsi = 100 - (
            100 / (1 + rs)
        )

        st.line_chart(rsi)

    # ==========================================================
    # TAB 3
    # ==========================================================
    with tabs[2]:

        st.subheader("Trend + Risk Analysis")

        for stock in selected_stocks:

            prices = data[stock]

            ma5 = prices.rolling(5).mean().iloc[-1]
            ma20 = prices.rolling(20).mean().iloc[-1]

            returns = prices.pct_change()

            volatility = returns.std()

            # Trend
            if ma5 > ma20:
                trend = "📈 Uptrend"
                signal = "BUY"
            else:
                trend = "📉 Downtrend"
                signal = "SELL"

            # Risk
            if volatility < 0.015:
                risk = "🟢 Low Risk"
            elif volatility < 0.03:
                risk = "🟡 Medium Risk"
            else:
                risk = "🔴 High Risk"

            st.write(
                f"{stock} | {trend} | {risk} | Recommendation: {signal}"
            )

    # ==========================================================
    # TAB 4
    # ==========================================================
    with tabs[3]:

        st.subheader("Portfolio Manager")

        invest_stock = st.selectbox(
            "Select stock",
            selected_stocks,
            key="portfolio"
        )

        invest_amount = st.number_input(
            "Investment Amount",
            min_value=1000,
            value=10000
        )

        if st.button("Add to Portfolio"):

            st.session_state.portfolio.append({
                "stock": invest_stock,
                "amount": invest_amount
            })

        if st.button("Clear Portfolio"):

            st.session_state.portfolio = []

        total_value = 0

        for item in st.session_state.portfolio:

            stock = item["stock"]
            amount = item["amount"]

            buy_price = data[stock].iloc[0]
            current_price = data[stock].iloc[-1]

            shares = amount / buy_price

            current_value = shares * current_price

            profit = current_value - amount

            total_value += current_value

            if profit >= 0:
                st.success(
                    f"{stock}: ₹{round(current_value,2)} (+₹{round(profit,2)})"
                )
            else:
                st.error(
                    f"{stock}: ₹{round(current_value,2)} (-₹{abs(round(profit,2))})"
                )

        st.info(
            f"Total Portfolio Value: ₹{round(total_value,2)}"
        )

    # ==========================================================
    # TAB 5
    # ==========================================================
    with tabs[4]:

        st.subheader("News + Sentiment")

        news_stock = st.selectbox(
            "Choose stock",
            selected_stocks,
            key="news"
        )

        api_key = "4ae345ea76394297b12b5cfdc8f6fd9e"

        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={news_stock}&apiKey={api_key}"
        )

        try:

            response = requests.get(url)
            articles = response.json()["articles"][:5]

            for article in articles:

                title = article["title"]

                polarity = TextBlob(
                    title
                ).sentiment.polarity

                if polarity > 0:
                    sentiment = "🟢 Positive"
                elif polarity < 0:
                    sentiment = "🔴 Negative"
                else:
                    sentiment = "⚪ Neutral"

                st.markdown(f"### {title}")
                st.write(sentiment)
                st.markdown(
                    f"[Read More]({article['url']})"
                )
                st.write("---")

        except:
            st.warning(
                "News unavailable."
            )

    # ==========================================================
    # TAB 6
    # ==========================================================
    with tabs[5]:

        st.subheader("ML Stock Prediction")

        ml_stock = st.selectbox(
            "Choose stock",
            selected_stocks,
            key="ml"
        )

        df = data[ml_stock].reset_index()

        df["Day"] = np.arange(
            len(df)
        )

        X = df[["Day"]]
        y = df[ml_stock]

        split = int(
            len(df) * 0.8
        )

        X_train = X[:split]
        X_test = X[split:]

        y_train = y[:split]
        y_test = y[split:]

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        st.write(
            f"MAE: {round(mae,2)}"
        )

        st.write(
            f"R² Score: {round(r2,2)}"
        )

        # Future prediction
        future_days = 30

        future_X = pd.DataFrame({
            "Day": np.arange(
                len(df),
                len(df) + future_days
            )
        })

        future_predictions = model.predict(
            future_X
        )

        st.subheader(
            "Next 30 Days Prediction"
        )

        pred_df = pd.DataFrame({
            "Predicted Price": future_predictions
        })

        st.line_chart(pred_df)
