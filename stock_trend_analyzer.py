import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go

import json
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from textblob import TextBlob


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Stock Intelligence Platform",
    layout="wide"
)


# =====================================================
# USER AUTH
# =====================================================
USER_FILE = "users.json"


def load_users():

    if not os.path.exists(USER_FILE):

        with open(USER_FILE, "w") as f:

            json.dump({}, f)

    with open(USER_FILE, "r") as f:

        return json.load(f)


def save_users(users):

    with open(USER_FILE, "w") as f:

        json.dump(users, f)


# Session Auth
if "logged_in" not in st.session_state:

    st.session_state.logged_in = False

if "username" not in st.session_state:

    st.session_state.username = None

if "portfolio" not in st.session_state:

    st.session_state.portfolio = []


# =====================================================
# LOGIN / SIGNUP
# =====================================================
if not st.session_state.logged_in:

    st.title("🔐 Stock Platform Authentication")

    auth_tabs = st.tabs([
        "Login",
        "Sign Up"
    ])

    # LOGIN
    with auth_tabs[0]:

        username = st.text_input(
            "Username",
            key="login_user"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_pass"
        )

        if st.button("Login"):

            users = load_users()

            if (
                username in users
                and users[username] == password
            ):

                st.session_state.logged_in = True
                st.session_state.username = username

                st.rerun()

            else:

                st.error(
                    "Invalid credentials"
                )

    # SIGNUP
    with auth_tabs[1]:

        new_user = st.text_input(
            "Create Username",
            key="signup_user"
        )

        new_pass = st.text_input(
            "Create Password",
            type="password",
            key="signup_pass"
        )

        if st.button("Create Account"):

            users = load_users()

            if new_user in users:

                st.error(
                    "Username already exists"
                )

            elif len(new_user) < 3:

                st.error(
                    "Username too short"
                )

            elif len(new_pass) < 4:

                st.error(
                    "Password too short"
                )

            else:

                users[new_user] = new_pass

                save_users(users)

                st.success(
                    "Account created successfully!"
                )

    st.stop()


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.success(
    f"Welcome {st.session_state.username}"
)

if st.sidebar.button("Logout"):

    st.session_state.logged_in = False
    st.session_state.username = None

    st.rerun()


# =====================================================
# TITLE
# =====================================================
st.title("📊 AI Stock Intelligence Platform")


# =====================================================
# LOGOS
# =====================================================
logo_urls = {
    "AAPL": "https://companieslogo.com/img/orig/AAPL-bf1a4314.png",
    "MSFT": "https://companieslogo.com/img/orig/MSFT-a203b22d.png"
}


# =====================================================
# SIDEBAR FILTERS
# =====================================================
stock_options = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "TSLA",
    "RELIANCE.NS",
    "TCS.NS"
]

st.sidebar.header(
    "Stock Filters"
)

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


if len(selected_stocks) == 0:

    st.warning(
        "Please select at least one stock."
    )

    st.stop()


# =====================================================
# FETCH DATA
# =====================================================
try:

    data = yf.download(
        selected_stocks,
        start=start_date,
        end=end_date,
        progress=False
    )["Close"]

except:

    st.error(
        "Unable to fetch data."
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
    ).dropna()

    st.subheader(
        "Candlestick Chart"
    )

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

    st.subheader(
        "Price Comparison"
    )

    st.line_chart(
        data
    )

    normalized = data / data.iloc[0] * 100

    st.subheader(
        "Performance Comparison"
    )

    st.line_chart(
        normalized
    )


# =====================================================
# TAB 2 : RSI
# =====================================================
with tabs[1]:

    rsi_stock = st.selectbox(
        "Select Stock",
        selected_stocks,
        key="rsi"
    )

    prices = data[
        rsi_stock
    ]

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
        f"RSI : {rsi_stock}"
    )

    st.line_chart(
        rsi
    )


# =====================================================
# TAB 3 : ANALYSIS
# =====================================================
with tabs[2]:

    st.subheader(
        "AI Trend Analysis"
    )

    for stock in selected_stocks:

        prices = data[
            stock
        ]

        ma5 = prices.rolling(
            5
        ).mean().iloc[-1]

        ma20 = prices.rolling(
            20
        ).mean().iloc[-1]

        returns = prices.pct_change()

        volatility = returns.std()

        if ma5 > ma20:

            trend = "📈 Uptrend"
            signal = "🟢 BUY"

        else:

            trend = "📉 Downtrend"
            signal = "🔴 SELL"

        if volatility < 0.015:

            risk = "Low Risk"

        elif volatility < 0.03:

            risk = "Medium Risk"

        else:

            risk = "High Risk"

        st.info(
            f"{stock} | {trend} | {risk} | {signal}"
        )


# =====================================================
# TAB 4 : PORTFOLIO
# =====================================================
with tabs[3]:

    stock = st.selectbox(
        "Select Stock",
        selected_stocks,
        key="portfolio_stock"
    )

    amount = st.number_input(
        "Investment Amount",
        min_value=1000,
        value=10000
    )

    if st.button(
        "Add to Portfolio"
    ):

        st.session_state.portfolio.append({
            "stock": stock,
            "amount": amount
        })

    total_value = 0

    for item in st.session_state.portfolio:

        if not isinstance(
            item,
            dict
        ):
            continue

        stock = item.get(
            "stock"
        )

        amount = item.get(
            "amount"
        )

        if stock not in data.columns:
            continue

        buy_price = data[
            stock
        ].iloc[0]

        current_price = data[
            stock
        ].iloc[-1]

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
        f"Portfolio Value: ₹{round(total_value,2)}"
    )


# =====================================================
# TAB 5 : NEWS
# =====================================================
with tabs[4]:

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

        response = requests.get(
            url
        )

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

            st.write("---")

    except:

        st.warning(
            "News unavailable."
        )


# =====================================================
# TAB 6 : ML
# =====================================================
with tabs[5]:

    ml_stock = st.selectbox(
        "Select Stock",
        selected_stocks,
        key="ml"
    )

    prices = data[
        ml_stock
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

        X = np.array(
            X
        )

        y = np.array(
            y
        )

        split = int(
            len(X) * 0.8
        )

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(
            X[:split],
            y[:split]
        )

        predictions = model.predict(
            X[split:]
        )

        mae = mean_absolute_error(
            y[split:],
            predictions
        )

        r2 = r2_score(
            y[split:],
            predictions
        )

        st.write(
            f"MAE: {round(mae,2)}"
        )

        st.write(
            f"R²: {round(r2,2)}"
        )

        # Future Prediction
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
            last_window.append(pred)

        st.subheader(
            "30 Days Forecast"
        )

        forecast_df = pd.DataFrame({
            "Predicted Price": future_preds
        })

        st.line_chart(
            forecast_df
        )
