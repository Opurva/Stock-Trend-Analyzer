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
    page_title="Stock Intelligence Platform",
    layout="wide"
)


# =====================================================
# USER FILE
# =====================================================
USER_FILE = "users.json"


def load_users():

    if not os.path.exists(
        USER_FILE
    ):

        with open(
            USER_FILE,
            "w"
        ) as f:

            json.dump(
                {},
                f
            )

    with open(
        USER_FILE,
        "r"
    ) as f:

        return json.load(
            f
        )


def save_users(
    users
):

    with open(
        USER_FILE,
        "w"
    ) as f:

        json.dump(
            users,
            f
        )


# =====================================================
# SESSION STATE
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []


# =====================================================
# LOGIN + SIGNUP
# =====================================================
if not st.session_state.logged_in:

    st.title(
        "🔐 AI Stock Platform"
    )

    auth_tabs = st.tabs([
        "Login",
        "Sign Up"
    ])


    # ================= LOGIN =================
    with auth_tabs[0]:

        login_user = st.text_input(
            "Username",
            key="login_user"
        )

        login_pass = st.text_input(
            "Password",
            type="password",
            key="login_pass"
        )

        if st.button(
            "Login"
        ):

            users = load_users()

            if (
                login_user in users
                and users[
                    login_user
                ] == login_pass
            ):

                st.session_state.logged_in = True

                st.session_state.username = login_user

                st.rerun()

            else:

                st.error(
                    "Invalid credentials"
                )


    # ================= SIGNUP =================
    with auth_tabs[1]:

        signup_user = st.text_input(
            "Choose Username",
            key="signup_user"
        )

        signup_pass = st.text_input(
            "Choose Password",
            type="password",
            key="signup_pass"
        )

        if st.button(
            "Create Account"
        ):

            users = load_users()

            if signup_user in users:

                st.error(
                    "Username already exists"
                )

            elif len(
                signup_user
            ) < 3:

                st.error(
                    "Username too short"
                )

            elif len(
                signup_pass
            ) < 4:

                st.error(
                    "Password too short"
                )

            else:

                users[
                    signup_user
                ] = signup_pass

                save_users(
                    users
                )

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

if st.sidebar.button(
    "Logout"
):

    st.session_state.logged_in = False
    st.session_state.username = None

    st.rerun()


# =====================================================
# TITLE
# =====================================================
st.title(
    "📊 AI Stock Intelligence Platform"
)


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
data = yf.download(
    selected_stocks,
    start=start_date,
    end=end_date,
    progress=False
)["Close"]

if len(selected_stocks) == 1:

    data = pd.DataFrame(
        data
    )


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
# CHARTS
# =====================================================
with tabs[0]:

    stock = st.selectbox(
        "Select Stock",
        selected_stocks,
        key="chart"
    )

    if stock in logo_urls:

        try:

            st.image(
                logo_urls[stock],
                width=80
            )

        except:
            pass

    stock_df = yf.download(
        stock,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False
    ).dropna()

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

    st.line_chart(
        data
    )


# =====================================================
# RSI
# =====================================================
with tabs[1]:

    stock = st.selectbox(
        "RSI Stock",
        selected_stocks,
        key="rsi"
    )

    prices = data[
        stock
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

    st.line_chart(
        rsi
    )


# =====================================================
# ANALYSIS
# =====================================================
with tabs[2]:

    st.subheader(
        "AI Analysis"
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

        if ma5 > ma20:

            st.success(
                f"{stock}: BUY"
            )

        else:

            st.error(
                f"{stock}: SELL"
            )


# =====================================================
# PORTFOLIO
# =====================================================
with tabs[3]:

    stock = st.selectbox(
        "Portfolio Stock",
        selected_stocks,
        key="portfolio"
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

    for item in st.session_state.portfolio:

        if not isinstance(
            item,
            dict
        ):
            continue

        st.write(
            item
        )


# =====================================================
# NEWS
# =====================================================
with tabs[4]:

    stock = st.selectbox(
        "News Stock",
        selected_stocks,
        key="news"
    )

    api_key = "4ae345ea76394297b12b5cfdc8f6fd9e"

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={stock}&apiKey={api_key}"
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

            sentiment = "⚪ Neutral"

            if polarity > 0:
                sentiment = "🟢 Positive"

            elif polarity < 0:
                sentiment = "🔴 Negative"

            st.markdown(
                f"### {title}"
            )

            st.write(
                sentiment
            )

    except:

        st.warning(
            "News unavailable."
        )


# =====================================================
# ML
# =====================================================
with tabs[5]:

    stock = st.selectbox(
        "ML Stock",
        selected_stocks,
        key="ml"
    )

    prices = data[
        stock
    ].dropna().values

    if len(
        prices
    ) > 30:

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
            random_state=42
        )

        model.fit(
            X[:split],
            y[:split]
        )

        predictions = model.predict(
            X[split:]
        )

        st.write(
            "MAE:",
            round(
                mean_absolute_error(
                    y[split:],
                    predictions
                ),
                2
            )
        )

        st.write(
            "R²:",
            round(
                r2_score(
                    y[split:],
                    predictions
                ),
                2
            )
        )
