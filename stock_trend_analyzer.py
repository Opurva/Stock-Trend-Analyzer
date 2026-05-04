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
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Stock Intelligence Platform",
    layout="wide"
)


# =====================================================
# LOGIN SYSTEM
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []


# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:

    st.title("🔐 Stock Platform Login")

    username = st.text_input("Username")
    password = st.text_input(
        "Password",
        type="password"
    )

    valid_users = {
        "opurva": "12345",
        "demo": "demo123",
        "vasu": "Glitterydihh69"
    }

    if st.button("Login"):

        if (
            username in valid_users
            and valid_users[username] == password
        ):

            st.session_state.logged_in = True
            st.session_state.current_user = username

            st.success("Login Successful")
            st.rerun()

        else:

            st.error("Invalid username or password")

    st.stop()


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.success(
        f"Welcome {st.session_state.current_user}"
    )

    if st.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.current_user = None

        st.rerun()


# =====================================================
# MAIN APP
# =====================================================
st.title("📊 AI Stock Intelligence Platform")


# =====================================================
# LOGOS
# =====================================================
logo_urls = {
    "AAPL": "https://companieslogo.com/img/orig/AAPL-bf1a4314.png",
    "MSFT": "https://companieslogo.com/img/orig/MSFT-a203b22d.png",
    "GOOGL": "https://companieslogo.com/img/orig/GOOGL-0ed88f7c.png",
    "TSLA": "https://companieslogo.com/img/orig/TSLA-6da550e8.png"
}


# =====================================================
# STOCK FILTERS
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
# DATA FETCH
# =====================================================
if len(selected_stocks) == 0:

    st.warning(
        "Please select at least one stock."
    )

    st.stop()

try:

    data = yf.download(
        selected_stocks,
        start=start_date,
        end=end_date,
        progress=False
    )["Close"]

except:

    st.error(
        "Unable to fetch stock data."
    )

    st.stop()

if data.empty:

    st.error(
        "No stock data found."
    )

    st.stop()

if len(selected_stocks) == 1:

    data = pd.DataFrame(data)


# =====================================================
# TABS
# =====================================================
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

    st.subheader("Price Comparison")

    st.line_chart(data)


# =====================================================
# TAB 2 : RSI
# =====================================================
with tabs[1]:

    stock = st.selectbox(
        "Select Stock",
        selected_stocks,
        key="rsi"
    )

    prices = data[stock]

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

    st.subheader(
        f"RSI - {stock}"
    )

    st.line_chart(rsi)


# =====================================================
# TAB 3 : ANALYSIS
# =====================================================
with tabs[2]:

    st.subheader("Trend Analysis")

    for stock in selected_stocks:

        prices = data[stock]

        ma5 = prices.rolling(5).mean().iloc[-1]

        ma20 = prices.rolling(20).mean().iloc[-1]

        if ma5 > ma20:

            st.success(
                f"{stock}: BUY Signal"
            )

        else:

            st.error(
                f"{stock}: SELL Signal"
            )


# =====================================================
# TAB 4 : PORTFOLIO
# =====================================================
with tabs[3]:

    st.subheader("Portfolio")

    stock = st.selectbox(
        "Select Stock",
        selected_stocks,
        key="portfolio"
    )

    amount = st.number_input(
        "Investment Amount",
        min_value=1000,
        value=10000
    )

    if st.button("Add to Portfolio"):

        st.session_state.portfolio.append({
            "stock": stock,
            "amount": amount
        })

        st.success(
            "Added Successfully"
        )

    if st.button("Clear Portfolio"):

        st.session_state.portfolio = []

    for item in st.session_state.portfolio:

        st.write(item)


# =====================================================
# TAB 5 : NEWS
# =====================================================
with tabs[4]:

    st.subheader("Stock News")

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

    except:

        st.warning(
            "News unavailable"
        )


# =====================================================
# TAB 6 : ML PREDICTION
# =====================================================
with tabs[5]:

    st.subheader(
        "ML Prediction"
    )

    stock = st.selectbox(
        "Select Stock",
        selected_stocks,
        key="ml"
    )

    prices = data[
        stock
    ].dropna().values

    if len(prices) < 30:

        st.warning(
            "Not enough data."
        )

    else:

        X = []
        y = []

        for i in range(
            5,
            len(prices)
        ):

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
