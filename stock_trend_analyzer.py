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
credentials = {
    "usernames": {
        "opurva": {
            "name": "Opurva",
            "password": "$2b$12$yQnM0P4l6oJ3XWw4eBq6DeFf0J1xD0W1k7Jg7p8NwV8z8wM1rQ4xW"
        },
        "demo": {
            "name": "Demo User",
            "password": "$2b$12$Hf7JmN5kL2pQxR9sT4uV7eA8bC1dE3fG5hI7jK9lM2nP4qR6sT8uW"
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
    location="main"
)


# =====================================================
# LOGIN CHECK
# =====================================================
if authentication_status == False:

    st.error("Invalid username or password")

elif authentication_status == None:

    st.warning("Please login")

elif authentication_status:

    authenticator.logout(
    location="sidebar"
    )

    st.sidebar.success(
        f"Welcome {name}"
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
    if len(selected_stocks) > 0:

        data = yf.download(
            selected_stocks,
            start=start_date,
            end=end_date,
            progress=False
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


        # =====================================================
        # TAB 1
        # =====================================================
        with tabs[0]:

            chart_stock = st.selectbox(
                "Select Stock",
                selected_stocks
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
        # TAB 2
        # =====================================================
        with tabs[1]:

            rsi_stock = st.selectbox(
                "RSI Stock",
                selected_stocks
            )

            prices = data[rsi_stock]

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
        # TAB 3
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
        # TAB 4
        # =====================================================
        with tabs[3]:

            invest_stock = st.selectbox(
                "Portfolio Stock",
                selected_stocks
            )

            amount = st.number_input(
                "Investment Amount",
                min_value=1000
            )

            if st.button("Add"):

                st.session_state.portfolio.append({
                    "stock": invest_stock,
                    "amount": amount
                })

            st.write(
                st.session_state.portfolio
            )


        # =====================================================
        # TAB 5
        # =====================================================
        with tabs[4]:

            news_stock = st.selectbox(
                "News Stock",
                selected_stocks
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

                    st.write(title)
                    st.write(sentiment)

            except:

                st.warning(
                    "News unavailable"
                )


        # =====================================================
        # TAB 6
        # =====================================================
        with tabs[5]:

            ml_stock = st.selectbox(
                "ML Stock",
                selected_stocks
            )

            prices = data[
                ml_stock
            ].dropna().values

            if len(prices) > 30:

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

                model = RandomForestRegressor()

                model.fit(
                    X[:split],
                    y[:split]
                )

                preds = model.predict(
                    X[split:]
                )

                st.write(
                    "MAE:",
                    round(
                        mean_absolute_error(
                            y[split:],
                            preds
                        ),
                        2
                    )
                )

                st.write(
                    "R²:",
                    round(
                        r2_score(
                            y[split:],
                            preds
                        ),
                        2
                    )
                )

    else:

        st.warning(
            "Select at least one stock"
        )
