# helios_app.py  -- Helios (Monitoring + Forecasting + Optimization + News)
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
import io
import zipfile
import requests
from bs4 import BeautifulSoup

plt.rcParams["figure.dpi"] = 120

# -----------------------
# --- Config
# -----------------------
st.set_page_config(page_title="Helios — Dashboard", layout="wide")
st.title("🌞 Helios — Dashboard")

# Example portfolios (editable)
PORTFOLIOS = {
    "US": {"AAPL": 10, "MSFT": 5, "GOOGL": 2},
    "Kenya": {"Equity Bank": 10, "Safaricom": 5, "KCB Group": 2},
    "Energy": {"XOM": 15, "CVX": 8, "BP": 12}
}

# -----------------------
# --- Helpers
# -----------------------
def fetch_prices(tickers, start=None, end=None):
    if not tickers:
        return pd.DataFrame()
    try:
        df = yf.download(tickers, start=start, end=end, threads=False, progress=False)
    except Exception as e:
        st.warning(f"yfinance error for {tickers}: {e}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    prices = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    return prices.ffill().dropna()

def build_portfolio(holdings, market="US", start="2024-01-01"):
    tickers = list(holdings.keys())
    if market == "Kenya":
        dates = pd.date_range(start=start, periods=200)
        mock = np.linspace(1000, 1200, len(dates)) + np.random.randn(len(dates)) * 5
        prices = pd.DataFrame({t: mock * (i + 1) for i, t in enumerate(holdings)}, index=dates)
    else:
        prices = fetch_prices(tickers, start=start, end=pd.Timestamp.today().strftime("%Y-%m-%d"))

    if prices.empty:
        return pd.DataFrame({"Date": [], "Value": []}), pd.DataFrame()
    values = (prices * pd.Series(holdings)).sum(axis=1)
    df = pd.DataFrame({"Date": values.index, "Value": values})
    df["Return"] = df["Value"].pct_change().fillna(0)
    return df, prices

def risk_metrics_from_series(series):
    if series.empty or "Return" not in series:
        return np.nan, np.nan, np.nan
    returns = series["Return"].dropna()
    vol = returns.std() * np.sqrt(252) if not returns.empty else np.nan
    sharpe = (returns.mean() * 252) / vol if vol and vol > 0 else np.nan
    cum = (1 + returns).cumprod() if not returns.empty else pd.Series(dtype=float)
    mdd = (cum - cum.cummax()).min() if not returns.empty else np.nan
    return vol, sharpe, mdd

def monte_carlo_sim(portfolio_df, horizon=252, sims=1000, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    returns = portfolio_df["Return"].dropna() if not portfolio_df.empty else pd.Series(dtype=float)
    if returns.empty or portfolio_df["Value"].empty:
        return np.empty((0, 0)), {}
    last = portfolio_df["Value"].iloc[-1]
    mean, std = returns.mean(), returns.std()
    sim_matrix = np.empty((sims, horizon + 1))
    for i in range(sims):
        path = [last]
        for _ in range(horizon):
            path.append(path[-1] * (1 + np.random.normal(mean, std)))
        sim_matrix[i, :] = path
    final = sim_matrix[:, -1]
    p5, p50, p95 = np.percentile(final, [5, 50, 95])
    summary = {"p5": p5, "p50": p50, "p95": p95, "final_array": final, "sim_matrix": sim_matrix}
    return sim_matrix, summary

def optimize_weights(prices_df):
    if prices_df.empty:
        return pd.Series([], dtype=float), np.nan, np.nan, np.nan
    returns = prices_df.pct_change().dropna()
    if returns.empty:
        idx = prices_df.columns if hasattr(prices_df, "columns") else [prices_df.name]
        return pd.Series([1.0 / len(idx)] * len(idx), index=idx), np.nan, np.nan, np.nan
    mean_returns = returns.mean()
    cov = returns.cov()
    n = len(prices_df.columns)

    def neg_sharpe(w):
        port_ret = np.dot(mean_returns, w) * 252
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov * 252, w)))
        return -(port_ret / port_vol) if port_vol > 0 else 1e6

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init = np.array([1.0 / n] * n)
    try:
        res = minimize(neg_sharpe, init, bounds=bounds, constraints=cons)
        weights = pd.Series(res.x, index=prices_df.columns)
        port_return = np.dot(mean_returns, weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov * 252, weights)))
        sharpe = port_return / port_vol if port_vol > 0 else np.nan
        return weights, port_return, port_vol, sharpe
    except Exception as e:
        st.warning(f"Optimization failed: {e}")
        idx = prices_df.columns if hasattr(prices_df, "columns") else [prices_df.name]
        return pd.Series([1.0 / len(idx)] * len(idx), index=idx), np.nan, np.nan, np.nan

# -----------------------
# --- News Scraper (Yahoo RSS)
# -----------------------
@st.cache_data(ttl=3600)
def fetch_market_news():
    url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL,MSFT,GOOG&region=US&lang=en-US"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "xml")
        items = soup.find_all("item")
        news = []
        for item in items[:20]:
            title = item.title.text if item.title else "No title"
            link = item.link.text if item.link else ""
            pubdate = item.pubDate.text if item.pubDate else ""
            news.append({"title": title, "link": link, "pubdate": pubdate})
        return news
    except Exception as e:
        st.warning(f"News fetch error: {e}")
        return []

# -----------------------
# --- Sidebar controls
# -----------------------
st.sidebar.header("Controls")
selected = st.sidebar.multiselect("Select portfolios", list(PORTFOLIOS.keys()), default=["US"])
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 30, 756, 252, step=30)
forecast_sims = st.sidebar.slider("Monte Carlo sims", 100, 3000, 1000, step=100)
drawdown_threshold = st.sidebar.slider("Drawdown alert threshold (%)", 1, 100, 20)
export_all = st.sidebar.checkbox("Export results (ZIP)")

# News refresh
news_refresh = st.sidebar.button("Refresh News")

# -----------------------
# --- Tabs
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["Monitoring", "Forecasting", "Optimization", "News"])

# Pre-fetch portfolios
fetched = {}
for name in selected:
    holdings = PORTFOLIOS[name]
    market = "Kenya" if name.lower() == "kenya" else "US"
    df, prices = build_portfolio(holdings, market=market)
    fetched[name] = {"portfolio": df, "prices": prices, "holdings": holdings}

# -------- Monitoring --------
with tab1:
    st.header("Monitoring")
    for name, obj in fetched.items():
        pf = obj["portfolio"]
        if pf.empty:
            st.warning(f"{name}: no data")
            continue
        vol, sharpe, mdd = risk_metrics_from_series(pf)
        last_value = pf["Value"].iloc[-1]
        peak = pf["Value"].cummax().max()
        drawdown_pct = (last_value / peak - 1) * 100
        if drawdown_pct <= -drawdown_threshold:
            st.error(f"🚨 {name} drawdown = {drawdown_pct:.2f}%")
        else:
            st.info(f"{name} drawdown = {drawdown_pct:.2f}% | Vol (ann): {vol:.2%} | Sharpe: {sharpe:.2f}")

# -------- Forecasting --------
with tab2:
    st.header("Forecasting")
    for name, obj in fetched.items():
        pf = obj["portfolio"]
        if pf.empty:
            st.warning(f"{name}: no data for forecasting")
            continue
        sims, summary = monte_carlo_sim(pf, horizon=forecast_horizon, sims=forecast_sims)
        if not summary:
            st.warning("Monte Carlo could not run")
            continue
        p5, p50, p95 = summary["p5"], summary["p50"], summary["p95"]
        st.write(f"{name} → p5: {p5:,.0f}, median: {p50:,.0f}, p95: {p95:,.0f}")

# -------- Optimization --------
with tab3:
    st.header("Optimization")
    for name, obj in fetched.items():
        st.subheader(name)
        prices = obj["prices"]
        holdings = obj["holdings"]
        if prices.empty:
            st.warning("No price data")
            continue
        opt_weights, opt_ret, opt_vol, opt_sharpe = optimize_weights(prices)
        st.write("Optimized weights:")
        st.dataframe(opt_weights.apply(lambda x: f"{x:.2%}"))

# -------- News --------
with tab4:
    st.header("📢 Market News")
    if news_refresh:
        fetch_market_news.clear()
    news_items = fetch_market_news()
    if not news_items:
        st.info("No news available right now.")
    else:
        for n in news_items:
            st.markdown(f"- [{n['title']}]({n['link']}) — {n['pubdate']}")

