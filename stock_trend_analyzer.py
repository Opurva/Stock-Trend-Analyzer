import streamlit as st
import streamlit_authenticator as stauth
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
names = ["Opurva", "Demo User"]

usernames = ["Opurva", "demo"]

passwords = ["12345", "demo123"]

hashed_passwords = stauth.Hasher(passwords).generate()

credentials = {
    "usernames": {
        usernames[0]: {
            "name": names[0],
            "password": hashed_passwords[0]
        },
        usernames[1]: {
            "name": names[1],
            "password": hashed_passwords[1]
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "stock_cookie",
    "abcdef",
    cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login(
    "Login",
    "main"
)


# =====================================================
# AUTH CHECK
# =====================================================
if authentication_status == False:

    st.error("Invalid username or password")


elif authentication_status == None:

    st.warning("Please login")


elif authentication_status:

    authenticator.logout("Logout", "sidebar")

    st.sidebar.success(
        f"Welcome {name}"
    )

    st.title("📊 AI Stock Intelligence Platform")


    # =====================================================
    # SESSION STATE
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
    if len(selected_stocks) > 0:

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

            st.line_chart(data)


        # =====================================================
        # TAB 2 : RSI
        # =====================================================
        with tabs[1]:

            indicator_stock = st.selectbox(
                "Select Stock",
                selected_stocks,
                key="rsi"
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

            for stock in selected_stocks:

                prices = data[stock]

                ma5 = prices.rolling(5).mean().iloc[-1]
                ma20 = prices.rolling(20).mean().iloc[-1]

                if ma5 > ma20:

                    st.success(
                        f"{stock}: BUY"
                    )

                else:

                    st.error(
                        f"{stock}: SELL"
                    )


        # =====================================================
        # TAB 4 : PORTFOLIO
        # =====================================================
        with tabs[3]:

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

            for item in st.session_state.portfolio:

                if not isinstance(item, dict):
                    continue

                st.write(item)


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

                    st.write(sentiment)

            except:

                st.warning("News unavailable.")


        # =====================================================
        # TAB 6 : ML
        # =====================================================
        with tabs[5]:

            ml_stock = st.selectbox(
                "Select Stock",
                selected_stocks,
                key="ml"
            )

            prices = data[ml_stock].dropna().values

            if len(prices) > 30:

                X = []
                y = []

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

                model = RandomForestRegressor()

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

    else:

        st.warning(
            "Please select at least one stock."
        )
