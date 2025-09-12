# helios_app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
import io, zipfile, requests

plt.rcParams["figure.dpi"] = 120

# -----------------------
# --- Page config
# -----------------------
st.set_page_config(page_title="Helios — Dashboard", layout="wide")
st.title("🌞 Helios — Dashboard")

# -----------------------
# --- Example portfolios
# -----------------------
portfolios = {
    "US": {"AAPL": 10, "MSFT": 5, "GOOGL": 2},
    "Kenya": {"Equity Bank": 10, "Safaricom": 5, "KCB Group": 2},
    "Energy": {"XOM": 15, "CVX": 8, "BP": 12}
}

# -----------------------
# --- Helper functions
# -----------------------
def fetch_prices_for_tickers(tickers, start=None, end=None):
    """Fetch price series from Yahoo Finance."""
    if not tickers:
        return pd.DataFrame()
    df = yf.download(tickers, start=start, end=end, progress=False, threads=False)
    if df.empty:
        return pd.DataFrame()
    prices = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    return prices.ffill().dropna()

def build_portfolio_series(holdings, market="US"):
    """Return a portfolio dataframe (Date, Value) and prices DF for holdings."""
    start = "2024-01-01"
    tickers = list(holdings.keys())
    if market != "Kenya":
        prices = fetch_prices_for_tickers(
            tickers, start=start, end=pd.Timestamp.today().strftime("%Y-%m-%d")
        )
        if prices.empty:
            return pd.DataFrame({"Date": [], "Value": []}), pd.DataFrame()
        values = (prices * pd.Series(holdings)).sum(axis=1)
    else:
        # Kenya: Mock API-like prices
        dates = pd.date_range(start=start, periods=200)
        base = np.linspace(100, 130, len(dates)) + np.random.randn(len(dates)) * 2
        prices = pd.DataFrame({t: base * (i+1) for i, t in enumerate(holdings)}, index=dates)
        values = (prices * pd.Series(holdings)).sum(axis=1)

    portfolio = pd.DataFrame({"Date": values.index, "Value": values})
    portfolio["Return"] = portfolio["Value"].pct_change().fillna(0)
    return portfolio, prices

def risk_metrics(portfolio):
    """Compute risk stats in plain terms."""
    returns = portfolio["Return"]
    vol = returns.std() * np.sqrt(252) if not returns.empty else np.nan
    sharpe = (returns.mean() * 252) / vol if vol and vol > 0 else np.nan
    cum = (1 + returns).cumprod() if not returns.empty else pd.Series()
    mdd = (cum - cum.cummax()).min() if not returns.empty else np.nan
    return vol, sharpe, mdd

# -----------------------
# --- Market News (No TextBlob)
# -----------------------
def fetch_market_news():
    """Fetch top market news headlines (no sentiment)."""
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "category": "business",
        "language": "en",
        "apiKey": "20ID6FmuDor5AEfsMq9BwpBtcOjbkyzZ"  # replace with your key from newsapi.org
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        news_items = []
        for a in articles[:5]:
            news_items.append({
                "title": a.get("title", "No Title"),
                "source": a.get("source", {}).get("name", "Unknown"),
                "url": a.get("url", "#")
            })
        return news_items
    except Exception as e:
        return [{"title": f"Error fetching news: {e}", "source": "System", "url": "#"}]

# -----------------------
# --- Sidebar
# -----------------------
st.sidebar.header("Controls")
selected_portfolios = st.sidebar.multiselect("Select Portfolios", list(portfolios.keys()), default=["US"])
drawdown_alert_pct = st.sidebar.slider("Drawdown alert threshold (%)", 1, 100, 20)
watchlist_input = st.sidebar.text_input("Watchlist (tickers)", value="TSLA, NVDA")
watch_alert_pct = st.sidebar.slider("Watchlist intraday alert (%)", 1, 50, 5)
export = st.sidebar.checkbox("Export Results (ZIP)")

# -----------------------
# --- Tabs
# -----------------------
tab1, tab2, tab3 = st.tabs(["📊 Monitoring", "🔮 Forecasting", "🎯 Optimization"])

# -----------------------
# --- Monitoring
# -----------------------
with tab1:
    st.header("Monitoring")

    if not selected_portfolios:
        st.warning("Pick at least one portfolio.")
        st.stop()

    all_results, heat_rows = {}, []

    for pname in selected_portfolios:
        holdings = portfolios[pname]
        market = "Kenya" if pname.lower() == "kenya" else "US"
        pf, prices = build_portfolio_series(holdings, market)
        all_results[pname] = {"portfolio": pf, "prices": prices, "holdings": holdings}

        if not prices.empty and len(prices) >= 2:
            for t in prices.columns:
                val = prices[t].iloc[-1]
                pct = (prices[t].iloc[-1] / prices[t].iloc[-2] - 1)
                heat_rows.append({"Portfolio": pname, "Ticker": t, "Last": val, "PctChange": pct})

    # Alerts
    st.subheader("Live Alerts")
    for pname, data in all_results.items():
        pf = data["portfolio"]
        if pf.empty:
            continue
        peak = pf["Value"].cummax()
        drawdown = (pf["Value"].iloc[-1] / peak.max() - 1) * 100
        if drawdown <= -abs(drawdown_alert_pct):
            st.error(f"🚨 {pname} drawdown = {drawdown:.2f}% (limit {drawdown_alert_pct}%)")
        else:
            st.info(f"{pname} drawdown = {drawdown:.2f}% (safe)")

    # Watchlist
    st.subheader("Watchlist Alerts")
    wl = [w.strip().upper() for w in watchlist_input.split(",") if w.strip()]
    if wl:
        wl_prices = fetch_prices_for_tickers(wl, start=None, end=pd.Timestamp.today().strftime("%Y-%m-%d"))
        if not wl_prices.empty and len(wl_prices) >= 2:
            for t in wl:
                if t in wl_prices.columns:
                    pct = (wl_prices[t].iloc[-1] / wl_prices[t].iloc[-2] - 1) * 100
                    if abs(pct) >= watch_alert_pct:
                        st.warning(f"Watchlist alert: {t} moved {pct:.2f}%")
        else:
            st.write("No watchlist data available.")

    # Heatmap
    st.subheader("Heatmap Overview")
    heat_df = pd.DataFrame(heat_rows)
    if not heat_df.empty:
        heat_display = heat_df.pivot(index="Ticker", columns="Portfolio", values="PctChange")
        st.dataframe(
            heat_display.style.format("{:.2%}").applymap(
                lambda v: "color: green" if v >= 0 else "color: red" if pd.notna(v) else ""
            )
        )

    # Portfolios
    st.subheader("Portfolios (Details)")
    for pname, data in all_results.items():
        pf, prices = data["portfolio"], data["prices"]
        st.markdown(f"### {pname}")
        if pf.empty:
            st.write("No data.")
            continue
        vol, sharpe, mdd = risk_metrics(pf)
        st.write(f"📊 Volatility (risk): {vol:.2%}")
        st.write(f"📈 Sharpe ratio (risk-adjusted return): {sharpe:.2f}")
        st.write(f"📉 Maximum drawdown (worst fall): {mdd:.2%}")

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(pf["Date"], pf["Value"], label=pname)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=45)
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

    # Market news
    st.subheader("📰 Market News")
    news_items = fetch_market_news()
    for item in news_items:
        st.markdown(f"- [{item['title']}]({item['url']}) — *{item['source']}*")

# -----------------------
# --- Forecasting
# -----------------------
with tab2:
    st.header("Forecasting (Monte Carlo Simulation)")
    pname = st.selectbox("Pick a portfolio to forecast", selected_portfolios)
    pf = all_results[pname]["portfolio"]

    if not pf.empty:
        returns = pf["Return"].dropna()
        mean, vol = returns.mean(), returns.std()
        sims, T = 200, 252
        results = []

        for _ in range(sims):
            series = [pf["Value"].iloc[-1]]
            for _ in range(T):
                shock = np.random.normal(mean, vol)
                series.append(series[-1] * (1 + shock))
            results.append(series)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(pd.DataFrame(results).T, color="gray", alpha=0.2)
        ax.set_title(f"Monte Carlo Forecast — {pname}")
        st.pyplot(fig)
    else:
        st.write("No data available for forecast.")

# -----------------------
# --- Optimization
# -----------------------
with tab3:
    st.header("Optimization (Efficient Frontier)")

    pname = st.selectbox("Pick a portfolio to optimize", selected_portfolios, key="opt")
    holdings = portfolios[pname]
    tickers = list(holdings.keys())
    prices = all_results[pname]["prices"]

    if not prices.empty:
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        def portfolio_performance(weights):
            ret = np.dot(weights, mean_returns) * 252
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe = ret / vol if vol > 0 else 0
            return np.array([ret, vol, sharpe])

        def min_func_sharpe(weights):
            return -portfolio_performance(weights)[2]

        bounds = tuple((0, 1) for _ in range(len(tickers)))
        init_guess = [1. / len(tickers)] * len(tickers)
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        opt_result = minimize(min_func_sharpe, init_guess, bounds=bounds, constraints=constraints)

        if opt_result.success:
            weights = opt_result.x
            perf = portfolio_performance(weights)
            st.write("📌 Optimal Weights:")
            for t, w in zip(tickers, weights):
                st.write(f"{t}: {w:.2%}")
            st.write(f"📊 Expected Return: {perf[0]:.2%}")
            st.write(f"📈 Volatility: {perf[1]:.2%}")
            st.write(f"⚖️ Sharpe Ratio: {perf[2]:.2f}")
        else:
            st.error("Optimization failed.")
    else:
        st.write("Not enough data to optimize.")

# -----------------------
# --- Export
# -----------------------
if export:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for pname, data in all_results.items():
            pf, prices, holdings = data["portfolio"], data["prices"], data["holdings"]
            if not pf.empty:
                zf.writestr(f"{pname}_Portfolio.csv", pf.to_csv(index=False))
            if not prices.empty:
                zf.writestr(f"{pname}_Prices.csv", prices.to_csv())
            zf.writestr(f"{pname}_Holdings.txt", "\n".join([f"{k},{v}" for k, v in holdings.items()]))
    buffer.seek(0)
    st.download_button("⬇️ Download Portfolio Book (ZIP)", buffer, "Helios_PortfolioBook.zip", "application/zip")
