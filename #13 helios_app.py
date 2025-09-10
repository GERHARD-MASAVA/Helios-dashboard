# helios_app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
import io, zipfile
import requests
from bs4 import BeautifulSoup

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
        prices = fetch_prices_for_tickers(tickers, start=start, end=pd.Timestamp.today().strftime("%Y-%m-%d"))
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

def fetch_kenya_tbill_yields():
    """
    Scrape latest Kenya Treasury Bill yields from CBK website.
    Returns dict {tenor_days: yield_percent}.
    """
    url = "https://www.centralbank.go.ke/securities/treasury-bills/"
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        yields = {}

        rows = soup.find_all("tr")
        for r in rows:
            cols = [c.get_text(strip=True) for c in r.find_all("td")]
            if len(cols) >= 2 and "Days" in cols[0]:
                try:
                    tenor = int(cols[0].split()[0])  # e.g. "91 Days" → 91
                    rate = float(cols[1].replace("%", ""))  # e.g. "8.05%" → 8.05
                    yields[tenor] = rate
                except:
                    continue
        return yields
    except Exception as e:
        st.write(f"⚠️ Could not fetch CBK yields: {e}")
        return {}

# -----------------------
# --- Sidebar
# -----------------------
st.sidebar.header("Controls")
selected_portfolios = st.sidebar.multiselect("Select Portfolios", list(portfolios.keys()), default=["US"])
drawdown_alert_pct = st.sidebar.slider("Drawdown alert threshold (%)", 1, 100, 20)
watchlist_input = st.sidebar.text_input("Watchlist (tickers)", value="TSLA, NVDA")
watch_alert_pct = st.sidebar.slider("Watchlist intraday alert (%)", 1, 50, 5)
export = st.sidebar.checkbox("Export Results (ZIP)")
show_yield_curve = st.sidebar.checkbox("Show Kenya Yield Curve", value=True)
refresh_now = st.sidebar.button("🔄 Refresh Now")

# -----------------------
# --- Main analysis
# -----------------------
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

# -----------------------
# --- Alerts
# -----------------------
st.header("Live Alerts")
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

# -----------------------
# --- Watchlist
# -----------------------
st.header("Watchlist Alerts")
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

# -----------------------
# --- Heatmap
# -----------------------
st.header("Heatmap Overview")
heat_df = pd.DataFrame(heat_rows)
if not heat_df.empty:
    heat_display = heat_df.pivot(index="Ticker", columns="Portfolio", values="PctChange")
    st.dataframe(
        heat_display.style.format("{:.2%}").applymap(
            lambda v: "color: green" if v >= 0 else "color: red" if pd.notna(v) else ""
        )
    )

# -----------------------
# --- Portfolio panels
# -----------------------
st.header("Portfolios (Details)")
for pname, data in all_results.items():
    pf, prices = data["portfolio"], data["prices"]
    st.subheader(pname)
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

# -----------------------
# --- Kenya T-Bill Yield Curve
# -----------------------
if show_yield_curve:
    st.header("Kenya Treasury Bill Yield Curve")
    yields = fetch_kenya_tbill_yields()
    if yields:
        tenors = sorted(yields.keys())
        rates = [yields[t] for t in tenors]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tenors, rates, marker="o", linewidth=2, color="navy")
        ax.set_title("Kenya T-Bill Yield Curve", fontsize=14, weight="bold")
        ax.set_xlabel("Tenor (Days)")
        ax.set_ylabel("Yield (%)")
        ax.grid(True, linestyle="--", alpha=0.6)

        for x, y in zip(tenors, rates):
            ax.text(x, y + 0.05, f"{y:.2f}%", ha="center", fontsize=9)

        st.pyplot(fig)
    else:
        st.write("No CBK T-Bill yield data available at the moment.")

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
