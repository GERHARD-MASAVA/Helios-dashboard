# helios_app.py 
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import io, zipfile
import time

plt.rcParams["figure.dpi"] = 120

st.set_page_config(page_title="Helios Day 28 — Live Monitor", layout="wide")
st.title("🌞 Helios — Dashboard")

# -----------------------
# --- Mini-Aladdin portfolios (example)
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
    """Fetch latest price series for tickers using yfinance."""
    if not tickers:
        return pd.DataFrame()
    df = yf.download(tickers, start=start, end=end, threads=False, progress=False)
    if df.empty:
        return pd.DataFrame()
    if "Adj Close" in df.columns:
        prices = df["Adj Close"]
    else:
        prices = df["Close"]
    return prices.ffill().dropna()

def build_portfolio_series(holdings, market="US"):
    """Return a portfolio dataframe (Date, Value) and prices DF for holdings."""
    start = "2024-01-01"
    tickers = list(holdings.keys())
    if market != "Kenya":
        prices = fetch_prices_for_tickers(tickers, start=start, end=pd.Timestamp.today().strftime("%Y-%m-%d"))
        if prices.empty:
            return pd.DataFrame({"Date": [], "Value": []}), pd.DataFrame()
        values = (prices * pd.Series(holdings)).sum(axis=1)
    else:
        # Mock Kenya series
        dates = pd.date_range(start=start, periods=100)
        mock = np.linspace(1000, 1200, len(dates)) + np.random.randn(len(dates)) * 5
        prices = pd.DataFrame({t: mock * (v / 10) for t, v in holdings.items()}, index=dates)
        values = (prices * pd.Series(holdings)).sum(axis=1)

    portfolio = pd.DataFrame({"Date": values.index, "Value": values})
    portfolio["Return"] = portfolio["Value"].pct_change().fillna(0)
    return portfolio, prices

def risk_metrics(portfolio):
    returns = portfolio["Return"]
    vol = returns.std() * np.sqrt(252) if not returns.empty else np.nan
    sharpe = (returns.mean() * 252) / vol if vol and vol > 0 else np.nan
    cum = (1 + returns).cumprod() if not returns.empty else pd.Series()
    mdd = (cum - cum.cummax()).min() if not returns.empty else np.nan
    return vol, sharpe, mdd

def monte_carlo(portfolio, forecast_days=252, num_sim=500):
    returns = portfolio["Return"].dropna()
    if returns.empty:
        return [], "No returns to simulate."
    last_price = portfolio["Value"].iloc[-1]
    mean, std = returns.mean(), returns.std()
    sims = []
    for _ in range(num_sim):
        path = [last_price]
        for _ in range(forecast_days):
            path.append(path[-1] * (1 + np.random.normal(mean, std)))
        sims.append(path)
    final_prices = [s[-1] for s in sims]
    p5, p95 = np.percentile(final_prices, 5), np.percentile(final_prices, 95)
    alert_msg = f"Monte Carlo worst-case (5%): ${p5:,.0f} | best-case (95%): ${p95:,.0f}"
    return sims, alert_msg

def optimize_portfolio(prices):
    returns = prices.pct_change().dropna()
    if returns.empty:
        idx = prices.columns if hasattr(prices, "columns") else [prices.name]
        return pd.Series(1.0/len(idx), index=idx), np.nan, np.nan, np.nan
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(prices.columns)

    def neg_sharpe(weights):
        port_return = np.sum(mean_returns * weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        return -(port_return / port_vol)

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(neg_sharpe, num_assets * [1.0 / num_assets], bounds=bounds, constraints=constraints)
    optimized_weights = pd.Series(result.x, index=prices.columns)
    port_return = np.sum(mean_returns * optimized_weights) * 252
    port_vol = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix * 252, optimized_weights)))
    sharpe = port_return / port_vol
    return optimized_weights, port_return, port_vol, sharpe

# -----------------------
# --- Sidebar controls
# -----------------------
st.sidebar.header("Live Monitor Controls")
selected_portfolios = st.sidebar.multiselect("Select Portfolios to Monitor", list(portfolios.keys()), default=["US"])
refresh_interval = st.sidebar.slider("Suggested refresh (seconds)", min_value=10, max_value=600, value=60, step=10)
drawdown_alert_pct = st.sidebar.slider("Drawdown alert threshold (%)", min_value=1, max_value=100, value=20, step=1)
watchlist_input = st.sidebar.text_input("Watchlist (comma-separated tickers)", value="TSLA, NVDA")
watch_alert_pct = st.sidebar.slider("Watchlist intraday alert (%)", min_value=1, max_value=50, value=5, step=1)
refresh_now = st.sidebar.button("🔄 Refresh Now")

export = st.sidebar.checkbox("Export Results (ZIP)")

st.sidebar.markdown(f"**Tip:** Click **Refresh Now** to fetch latest prices.")

# -----------------------
# --- Main monitoring logic
# -----------------------
if not selected_portfolios:
    st.warning("Pick at least one portfolio to monitor on the left.")
    st.stop()

all_results = {}
portfolio_heat_rows = []

for pname in selected_portfolios:
    holdings = portfolios[pname]
    market = "Kenya" if pname.lower() == "kenya" else "US"
    if market == "US":
        tickers = list(holdings.keys())
        prices_df = fetch_prices_for_tickers(tickers, start=None, end=pd.Timestamp.today().strftime("%Y-%m-%d"))
        if prices_df.empty:
            st.error(f"No data returned for portfolio {pname}. Check tickers or Yahoo availability.")
            portfolio_df = pd.DataFrame({"Date": [], "Value": []})
        else:
            values = (prices_df * pd.Series(holdings)).sum(axis=1)
            portfolio_df = pd.DataFrame({"Date": values.index, "Value": values})
            portfolio_df["Return"] = portfolio_df["Value"].pct_change().fillna(0)
    else:
        dates = pd.date_range(start="2024-01-01", periods=100)
        mock = np.linspace(1000, 1200, len(dates)) + np.random.randn(len(dates)) * 5
        prices_df = pd.DataFrame({t: mock * (v / 10) for t, v in holdings.items()}, index=dates)
        values = (prices_df * pd.Series(holdings)).sum(axis=1)
        portfolio_df = pd.DataFrame({"Date": values.index, "Value": values})
        portfolio_df["Return"] = portfolio_df["Value"].pct_change().fillna(0)

    all_results[pname] = {"portfolio": portfolio_df, "prices": prices_df, "holdings": holdings}

    if not prices_df.empty and len(prices_df) >= 2:
        for t in prices_df.columns:
            val = prices_df[t].iloc[-1]
            pct = (prices_df[t].iloc[-1] / prices_df[t].iloc[-2] - 1)
            portfolio_heat_rows.append({"Portfolio": pname, "Ticker": t, "Last": val, "PctChange": pct})
    else:
        for t in holdings.keys():
            portfolio_heat_rows.append({"Portfolio": pname, "Ticker": t, "Last": np.nan, "PctChange": np.nan})

# -----------------------
# --- Alerts
# -----------------------
st.header("Live Alerts")
for pname, data in all_results.items():
    pf = data["portfolio"]
    if pf.empty or pf["Value"].empty:
        continue
    cumulative = pf["Value"].cummax()
    drawdown = (pf["Value"].iloc[-1] / cumulative.max() - 1) * 100
    if drawdown <= -abs(drawdown_alert_pct):
        st.error(f"🚨 {pname} drawdown = {drawdown:.2f}% (trigger: -{drawdown_alert_pct}%)")
    else:
        st.info(f"{pname} drawdown = {drawdown:.2f}%")

# -----------------------
# --- Watchlist alerts
# -----------------------
st.header("Watchlist Alerts")
wl = [w.strip().upper() for w in watchlist_input.split(",") if w.strip()]
if wl:
    try:
        wl_prices = fetch_prices_for_tickers(wl, start=None, end=pd.Timestamp.today().strftime("%Y-%m-%d"))
        if wl_prices.empty:
            st.write("No watchlist price data available.")
        else:
            watch_alerts = []
            for t in wl:
                if t in wl_prices.columns and len(wl_prices) >= 2:
                    pct = (wl_prices[t].iloc[-1] / wl_prices[t].iloc[-2] - 1) * 100
                    if abs(pct) >= watch_alert_pct:
                        watch_alerts.append((t, pct))
            if watch_alerts:
                for t, pct in watch_alerts:
                    st.warning(f⚠️ Watchlist alert: {t} moved {pct:.2f}% vs previous close (threshold {watch_alert_pct}%)")
            else:
                st.write("No watchlist alerts currently.")
    except Exception as e:
        st.write(f"Watchlist fetch failed: {e}")

# -----------------------
# --- Heatmap
# -----------------------
st.header("Heatmap Overview (latest pct change per ticker)")
heat_df = pd.DataFrame(portfolio_heat_rows)
if not heat_df.empty:
    heat_display = heat_df.pivot(index="Ticker", columns="Portfolio", values="PctChange")
    st.dataframe(
        heat_display.style.format("{:.2%}", na_rep="n/a").applymap(
            lambda v: "color: green" if (not pd.isna(v) and v >= 0) else ("color: red" if not pd.isna(v) else "")
        )
    )
else:
    st.write("No tickers/prices to show in heatmap.")

# -----------------------
# --- Portfolio panels
# -----------------------
st.header("Portfolios (detailed)")
for pname, data in all_results.items():
    pf = data["portfolio"]
    prices = data["prices"]
    st.subheader(pname)
    if pf.empty:
        st.write("No data for this portfolio.")
        continue
    last_time = pf["Date"].iloc[-1]
    st.write(f"Last data point: {last_time.date() if hasattr(last_time, 'date') else last_time}")
    vol, sharpe, mdd = risk_metrics(pf)
    st.write(f"Volatility (ann): {vol:.2%} | Sharpe: {sharpe:.2f} | Max Drawdown: {mdd:.2%}")

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(pf["Date"], pf["Value"], label=pname)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45)
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

# -----------------------
# --- Export
# -----------------------
if export:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for pname, data in all_results.items():
            pf = data["portfolio"]
            prices = data["prices"]
            holdings = data["holdings"]
            if not pf.empty:
                zf.writestr(f"{pname}_Portfolio.csv", pf.to_csv(index=False))
            if prices is not None and not prices.empty:
                zf.writestr(f"{pname}_Prices.csv", prices.to_csv())
            zf.writestr(f"{pname}_Holdings.txt", ("\n".join([f"{k},{v}" for k, v in holdings.items()])))
        zf.writestr("Helios_Metadata.txt", f"Exported: {pd.Timestamp.now()}\nRefresh interval: {refresh_interval}s\n")
    buffer.seek(0)
    st.success("💾 Export ready — click to download ZIP")
    st.download_button("⬇️ Download Portfolio Book (ZIP)", data=buffer.getvalue(), file_name="Helios_PortfolioBook.zip", mime="application/zip")

# -----------------------
# --- Manual refresh
# -----------------------
if refresh_now:
    st.experimental_rerun()

st.info("Manual refresh available. Change any sidebar control or click 'Refresh Now'.")
