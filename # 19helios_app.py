# helios_app.py  -- Helios  (Monitoring + Forecasting + Optimization + Sentiment/Alerts)
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob
import io, zipfile

plt.rcParams["figure.dpi"] = 120

# -----------------------
# --- Config / portfolios
# -----------------------
st.set_page_config(page_title="Helios — Dashboard", layout="wide")
st.title("🌞 Helios — Dashboard")

PORTFOLIOS = {
    "US": {"AAPL": 10, "MSFT": 5, "GOOGL": 2},
    "Kenya": {"Equity Bank": 10, "Safaricom": 5, "KCB Group": 2},
    "Energy": {"XOM": 15, "CVX": 8, "BP": 12}
}

HEADERS = {"User-Agent": "Mozilla/5.0 (HeliosBot/1.0)"}

# -----------------------
# --- Data helpers
# -----------------------
@st.cache_data(ttl=3600)
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
        # Mock data for demo
        dates = pd.date_range(start=start, periods=200)
        mock = np.linspace(1000, 1200, len(dates)) + np.random.randn(len(dates)) * 5
        prices = pd.DataFrame({t: mock * (i+1) for i,t in enumerate(holdings)}, index=dates)
    else:
        prices = fetch_prices(tickers, start=start, end=pd.Timestamp.today().strftime("%Y-%m-%d"))
    if prices.empty:
        return pd.DataFrame({"Date": [], "Value": []}), pd.DataFrame()
    values = (prices * pd.Series(holdings)).sum(axis=1)
    df = pd.DataFrame({"Date": values.index, "Value": values})
    df["Return"] = df["Value"].pct_change().fillna(0)
    return df, prices

def risk_metrics_from_series(df):
    if df.empty or "Return" not in df:
        return np.nan, np.nan, np.nan
    returns = df["Return"].dropna()
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else np.nan
    cum = (1+returns).cumprod()
    mdd = (cum - cum.cummax()).min()
    return vol, sharpe, mdd

def monte_carlo_sim(portfolio_df, horizon=252, sims=1000, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    returns = portfolio_df["Return"].dropna()
    if returns.empty:
        return np.empty((0,0)), {}
    last = portfolio_df["Value"].iloc[-1]
    mean, std = returns.mean(), returns.std()
    sim_matrix = np.empty((sims, horizon+1))
    for i in range(sims):
        path = [last]
        for _ in range(horizon):
            path.append(path[-1] * (1 + np.random.normal(mean, std)))
        sim_matrix[i,:] = path
    final = sim_matrix[:,-1]
    p5,p50,p95 = np.percentile(final,[5,50,95])
    return sim_matrix, {"p5":p5,"p50":p50,"p95":p95,"final_array":final,"sim_matrix":sim_matrix}

def optimize_weights(prices_df, robust=False):
    if prices_df.empty: return pd.Series([],dtype=float),np.nan,np.nan,np.nan
    returns = prices_df.pct_change().dropna()
    mean_returns = returns.mean()
    cov = returns.cov()
    n = len(prices_df.columns)

    def neg_sharpe(w):
        port_ret = np.dot(mean_returns, w)*252
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov*252, w)))
        return -(port_ret/port_vol) if port_vol>0 else 1e6

    cons = ({'type':'eq','fun': lambda w: np.sum(w)-1})
    bounds = tuple((0,1) for _ in range(n))
    init = np.array([1.0/n]*n)
    res = minimize(neg_sharpe, init, bounds=bounds, constraints=cons)
    weights = pd.Series(res.x, index=prices_df.columns)

    port_return = np.dot(mean_returns, weights)*252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov*252, weights)))
    sharpe = port_return/port_vol if port_vol>0 else np.nan

    if robust:
        port_return *= 0.9  # penalize return
        sharpe = port_return/port_vol if port_vol>0 else np.nan

    return weights, port_return, port_vol, sharpe

# -----------------------
# --- News + Sentiment
# -----------------------
@st.cache_data(ttl=3600)
def fetch_news_yahoo(ticker, count=5):
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
        r = requests.get(url, timeout=10).json()
        headlines = []
        for item in r.get("news", [])[:count]:
            title = item.get("title","")
            sentiment = TextBlob(title).sentiment.polarity
            headlines.append((title, sentiment))
        return headlines
    except:
        return [("No news available",0.0)]

# -----------------------
# --- Sidebar controls
# -----------------------
st.sidebar.header("Controls")
selected = st.sidebar.multiselect("Select portfolios", list(PORTFOLIOS.keys()), default=["US"])
monitor_refresh = st.sidebar.button("Refresh data")
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 30, 756, 252, step=30)
forecast_sims = st.sidebar.slider("Monte Carlo sims", 100, 3000, 1000, step=100)

# Alerts
st.sidebar.subheader("Custom Alerts")
alert_price = st.sidebar.slider("Price change alert (%)", 1, 20, 5)
alert_vol = st.sidebar.slider("Volatility alert (%)", 5, 50, 20)
alert_sentiment = st.sidebar.slider("Sentiment alert threshold", -1.0, 1.0, -0.3)

# Optimization
robust_opt = st.sidebar.checkbox("Use robust optimizer (conservative)", value=False)

# -----------------------
# --- Tabs
# -----------------------
tab1, tab2, tab3 = st.tabs(["Monitoring", "Forecasting", "Optimization"])

# Pre-fetch
fetched = {}
for name in selected:
    holdings = PORTFOLIOS[name]
    market = "Kenya" if name.lower()=="kenya" else "US"
    df, prices = build_portfolio(holdings, market=market)
    fetched[name] = {"portfolio": df, "prices": prices, "holdings": holdings}

# ---------------- Monitoring ----------------
with tab1:
    st.header("Monitoring")
    for name,obj in fetched.items():
        pf,prices = obj["portfolio"], obj["prices"]
        if pf.empty: continue
        st.subheader(name)

        vol,sharpe,mdd = risk_metrics_from_series(pf)
        last_val = pf["Value"].iloc[-1]
        st.metric("Value", f"${last_val:,.0f}")
        st.metric("Volatility (ann.)", f"{vol:.2%}" if not np.isnan(vol) else "n/a")
        st.metric("Sharpe", f"{sharpe:.2f}" if not np.isnan(sharpe) else "n/a")

        # Alerts
        if len(pf)>=2:
            pct_change = (pf["Value"].iloc[-1]/pf["Value"].iloc[-2]-1)*100
            if abs(pct_change)>=alert_price:
                st.warning(f"⚠ Price moved {pct_change:.2f}% (>{alert_price}%)")
        if vol*100>alert_vol:
            st.error(f"🚨 Volatility {vol:.2%} above {alert_vol}%")

        # News & Sentiment
        st.markdown("**News Headlines & Sentiment**")
        for ticker in prices.columns[:3]:
            news = fetch_news_yahoo(ticker)
            for title,senti in news:
                color = "🟢" if senti>0.1 else "🔴" if senti<-0.1 else "⚪"
                st.write(f"{color} {ticker}: {title} ({senti:.2f})")
                if senti<=alert_sentiment:
                    st.error(f"Negative sentiment alert for {ticker}")

        # Chart
        fig,ax = plt.subplots(figsize=(7,3))
        ax.plot(pf["Date"], pf["Value"], label=name)
        ax.set_title(f"{name} Portfolio Value")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=45)
        st.pyplot(fig)

# ---------------- Forecasting ----------------
with tab2:
    st.header("Forecasting (Monte Carlo)")
    for name,obj in fetched.items():
        pf=obj["portfolio"]
        if pf.empty: continue
        sims,summary = monte_carlo_sim(pf, horizon=forecast_horizon, sims=forecast_sims)
        if not summary: continue
        st.subheader(name)
        p5,p50,p95 = summary["p5"],summary["p50"],summary["p95"]
        st.write(f"{forecast_horizon}d forecast → p5: ${p5:,.0f}, median: ${p50:,.0f}, p95: ${p95:,.0f}")

        # Chart with subset of paths
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot(pf["Date"], pf["Value"], color="black", lw=2)
        for i in range(min(50,sims.shape[0])):
            future=pd.date_range(start=pf["Date"].iloc[-1]+pd.Timedelta(days=1), periods=forecast_horizon+1)
            ax.plot(future, sims[i,:], alpha=0.1)
        ax.axhline(p5,color="red",ls="--"); ax.axhline(p50,color="gray",ls="--"); ax.axhline(p95,color="green",ls="--")
        st.pyplot(fig)

# ---------------- Optimization ----------------
with tab3:
    st.header("Optimization (Max Sharpe)")
    for name,obj in fetched.items():
        prices,holdings=obj["prices"],obj["holdings"]
        if prices.empty: continue
        st.subheader(name)

        latest=prices.iloc[-1]
        current_w = (latest*pd.Series(holdings))/ (latest*pd.Series(holdings)).sum()

        opt_w,opt_ret,opt_vol,opt_sharpe = optimize_weights(prices, robust=robust_opt)

        st.write("Current weights:"); st.dataframe(current_w.apply(lambda x:f"{x:.2%}"))
        st.write("Optimized weights:"); st.dataframe(opt_w.apply(lambda x:f"{x:.2%}"))

        st.write(f"Optimized Sharpe: {opt_sharpe:.2f}, Volatility: {opt_vol:.2%}")

        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,4))
        ax1.pie(current_w, labels=current_w.index, autopct="%1.1f%%"); ax1.set_title("Current")
        ax2.pie(opt_w, labels=opt_w.index, autopct="%1.1f%%"); ax2.set_title("Optimized")
        st.pyplot(fig)

# ---------------- Refresh ----------------
if monitor_refresh:
    st.experimental_rerun()

st.success("✅ Helios updated with Monitoring + Forecasting + Optimization + Alerts + Sentiment")
