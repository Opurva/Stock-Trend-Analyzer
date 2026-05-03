import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from textblob import TextBlob


# =====================================================
# PAGE
# =====================================================
st.set_page_config(
    page_title="Stock Intelligence Platform",
    layout="wide"
)

st.title("📊 AI Stock Intelligence Platform")


# =====================================================
# SESSION
# =====================================================
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []


# =====================================================
# LOGOS
# =====================================================
logo_urls = {
    "AAPL": "https://companieslogo.com/img/orig/AAPL-bf1a4314.png",
    "MSFT": "https://companieslogo.com/img/orig/MSFT-a203b22d.png",
    "GOOGL": "https://companieslogo.com/img/orig/GOOGL-0ed88f7c.png",
    "TSLA": "https://companieslogo.com/img/orig/TSLA-6da550e8.png",
    "RELIANCE.NS": "https://upload.wikimedia.org/wikipedia/commons/8/8e/Reliance_Industries_Logo.svg",
    "TCS.NS": "https://upload.wikimedia.org/wikipedia/commons/b/b1/Tata_Consultancy_Services_Logo.svg"
}


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Stock Filters")

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

today = pd.Timestamp.today()

start_date = st.sidebar.date_input(
    "Start Date",
    value=today - pd.DateOffset(years=1),
    max_value=today
)

end_date = st.sidebar.date_input(
    "End Date",
    value=today,
    max_value=today
)


# =====================================================
# MAIN
# =====================================================
if selected_stocks:

    try:

        data = yf.download(
            selected_stocks,
            start=start_date,
            end=end_date,
            progress=False
        )["Close"]

    except:

        st.error("Unable to fetch stock data.")
        st.stop()

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


    # =====================================================
    # TAB 1 : CHARTS
    # =====================================================
    with tabs[0]:

        st.subheader("Stock Charts")

        chart_stock = st.selectbox(
            "Select Stock",
            selected_stocks,
            key="chart"
        )

        # Logo
        if chart_stock in logo_urls:

            try:
                st.image(
                    logo_urls[chart_stock],
                    width=80
                )
            except:
                pass

        stock_df = yf.download(
            chart_stock,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )

        stock_df = stock_df.dropna()

        # Candlestick
        st.subheader("🕯 Candlestick Chart")

        if not stock_df.empty:

            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=stock_df.index,
                        open=stock_df["Open"],
                        high=stock_df["High"],
                        low=stock_df["Low"],
                        close=stock_df["Close"]
                    )
                ]
            )

            fig.update_layout(
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning("Candlestick data unavailable.")

        # Comparison
        st.subheader("📈 Price Comparison")
        st.line_chart(data)

        # Normalized
        st.subheader("⚖ Performance Comparison")

        normalized = data / data.iloc[0] * 100

        st.line_chart(normalized)


    # =====================================================
    # TAB 2 : RSI
    # =====================================================
    with tabs[1]:

        indicator_stock = st.selectbox(
            "Select Stock",
            selected_stocks,
            key="rsi"
        )

        st.subheader(
            f"RSI Indicator - {indicator_stock}"
        )

        prices = data[indicator_stock]

        delta = prices.diff()

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


    # =====================================================
    # TAB 3 : ANALYSIS
    # =====================================================
    with tabs[2]:

        st.subheader("Trend + Risk Analysis")

        for stock in selected_stocks:

            prices = data[stock]

            ma5 = prices.rolling(5).mean().iloc[-1]
            ma20 = prices.rolling(20).mean().iloc[-1]

            returns = prices.pct_change()

            volatility = returns.std()

            if ma5 > ma20:

                trend = "📈 Uptrend"
                signal = "BUY"

            else:

                trend = "📉 Downtrend"
                signal = "SELL"

            if volatility < 0.015:

                risk = "🟢 Low Risk"

            elif volatility < 0.03:

                risk = "🟡 Medium Risk"

            else:

                risk = "🔴 High Risk"

            st.write(
                f"{stock} | {trend} | {risk} | Recommendation: {signal}"
            )


    # =====================================================
    # TAB 4 : PORTFOLIO
    # =====================================================
    with tabs[3]:

        st.subheader("Portfolio Manager")

        invest_stock = st.selectbox(
            "Select Stock",
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

            stock = item.get("stock")
            amount = item.get("amount")

            if stock not in data.columns:
                continue

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
            f"Total Value: ₹{round(total_value,2)}"
        )


    # =====================================================
    # TAB 5 : NEWS
    # =====================================================
    with tabs[4]:

        st.subheader("News + Sentiment")

        news_stock = st.selectbox(
            "Select Stock",
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

            articles = response.json().get(
                "articles",
                []
            )[:5]

            for article in articles:

                title = article.get(
                    "title",
                    "No Title"
                )

                polarity = TextBlob(
                    title
                ).sentiment.polarity

                if polarity > 0:

                    sentiment = "🟢 Positive"

                elif polarity < 0:

                    sentiment = "🔴 Negative"

                else:

                    sentiment = "⚪ Neutral"

                st.markdown(
                    f"### {title}"
                )

                st.write(
                    sentiment
                )

                st.markdown(
                    f"[Read More]({article.get('url','#')})"
                )

                st.write("---")

        except:

            st.warning(
                "News unavailable."
            )


    # =====================================================
    # TAB 6 : BETTER ML
    # =====================================================
    with tabs[5]:

        st.subheader(
            "ML Price Prediction"
        )

        ml_stock = st.selectbox(
            "Select Stock",
            selected_stocks,
            key="ml"
        )

        prices = data[ml_stock].values

        X = []
        y = []

        # Use previous 5 days
        for i in range(5, len(prices)):

            X.append(
                prices[i-5:i]
            )

            y.append(
                prices[i]
            )

        X = np.array(X)
        y = np.array(y)

        split = int(
            len(X) * 0.8
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

        predictions = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        st.write(
            f"MAE: {round(mae,2)}"
        )

        st.write(
            f"R² Score: {round(r2,2)}"
        )

        # Future prediction
        last_window = list(
            prices[-5:]
        )

        future_preds = []

        for _ in range(30):

            pred = model.predict(
                [last_window]
            )[0]

            future_preds.append(
                pred
            )

            last_window.pop(0)

            last_window.append(
                pred
            )

        pred_df = pd.DataFrame({
            "Predicted Price": future_preds
        })

        st.subheader(
            "Next 30 Days Forecast"
        )

        st.line_chart(
            pred_df
        )

else:

    st.warning(
        "Please select at least one stock."
    )
