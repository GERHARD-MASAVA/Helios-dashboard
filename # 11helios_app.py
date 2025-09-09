# helios_app.py  -- Helios Day 29 (Monitoring + Forecasting + Optimization)
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
import io
import zipfile

plt.rcParams["figure.dpi"] = 120

# -----------------------
# --- Config / portfolios
# -----------------------
st.set_page_config(page_title="Helios Day 29", layout="wide")
st.title("Helios: Monitor · Forecast · Optimize")

# Example portfolios (you can edit)
PORTFOLIOS = {
    "US": {"AAPL": 10, "MSFT": 5, "GOOGL": 2},
    "Kenya": {"Equity Bank": 10, "Safaricom": 5, "KCB Group": 2},
    "Energy": {"XOM": 15, "CVX": 8, "BP": 12}
}

# -----------------------
# --- Helpers
# -----------------------
def fetch_prices(tickers, start=None, end=None):
    """Return price DataFrame indexed by date (Adj Close if available)."""
    if not tickers:
        return pd.DataFrame()
    df = yf.download(tickers, start=start, end=end, threads=False, progress=False)
    if df.empty:
        return pd.DataFrame()
    prices = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    return prices.ffill().dropna()

def build_portfolio(holdings, market="US", start="2024-01-01"):
    """Return (portfolio_df, prices_df) where portfolio_df has Date, Value, Return."""
    tickers = list(holdings.keys())
    if market == "Kenya":
        # Mock Kenya data for demo
        dates = pd.date_range(start=start, periods=100)
        mock = np.linspace(1000, 1200, len(dates)) + np.random.randn(len(dates)) * 5
        prices = pd.DataFrame({t: mock * (v / 10) for t, v in holdings.items()}, index=dates)
    else:
        prices = fetch_prices(tickers, start=start, end=pd.Timestamp.today().strftime("%Y-%m-%d"))
    if prices.empty:
        return pd.DataFrame({"Date": [], "Value": []}), pd.DataFrame()
    values = (prices * pd.Series(holdings)).sum(axis=1)
    df = pd.DataFrame({"Date": values.index, "Value": values})
    df["Return"] = df["Value"].pct_change().fillna(0)
    return df, prices

def risk_metrics_from_series(series):
    """Given a portfolio df with 'Return', compute vol (ann), sharpe, max drawdown."""
    if series.empty:
        return np.nan, np.nan, np.nan
    returns = series["Return"].dropna()
    vol = returns.std() * np.sqrt(252) if not returns.empty else np.nan
    sharpe = (returns.mean() * 252) / vol if vol and vol > 0 else np.nan
    cum = (1 + returns).cumprod() if not returns.empty else pd.Series()
    mdd = (cum - cum.cummax()).min() if not returns.empty else np.nan
    return vol, sharpe, mdd

def monte_carlo_sim(portfolio_df, horizon=252, sims=1000):
    """Return sims (list of price paths) and percentile summary (p5, p50, p95)."""
    returns = portfolio_df["Return"].dropna()
    if returns.empty or portfolio_df["Value"].empty:
        return [], {}
    last = portfolio_df["Value"].iloc[-1]
    mean, std = returns.mean(), returns.std()
    sim_matrix = np.empty((sims, horizon+1))
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
    """Max Sharpe optimizer (long-only, fully invested). Returns weights Series and metrics."""
    returns = prices_df.pct_change().dropna()
    if returns.empty:
        idx = prices_df.columns if hasattr(prices_df, "columns") else [prices_df.name]
        return pd.Series([1.0/len(idx)]*len(idx), index=idx), np.nan, np.nan, np.nan
    mean_returns = returns.mean()
    cov = returns.cov()
    n = len(prices_df.columns)

    def neg_sharpe(w):
        port_ret = np.dot(mean_returns, w) * 252
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov * 252, w)))
        return -(port_ret / port_vol) if port_vol > 0 else 1e6

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init = np.array([1.0/n]*n)
    res = minimize(neg_sharpe, init, bounds=bounds, constraints=cons)
    weights = pd.Series(res.x, index=prices_df.columns)
    port_return = np.dot(mean_returns, weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov * 252, weights)))
    sharpe = port_return / port_vol if port_vol > 0 else np.nan
    return weights, port_return, port_vol, sharpe

# -----------------------
# --- Sidebar controls (global)
# -----------------------
st.sidebar.header("Controls")
selected = st.sidebar.multiselect("Select portfolios to work with", list(PORTFOLIOS.keys()), default=["US"])
monitor_refresh = st.sidebar.button("Refresh data (manual)")
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 30, 756, 252, step=30)
forecast_sims = st.sidebar.slider("Monte Carlo sims", 100, 3000, 1000, step=100)
export_all = st.sidebar.checkbox("Export selected results (ZIP)")

# VaR/alert controls (used in monitoring tab)
drawdown_threshold = st.sidebar.slider("Drawdown alert threshold (%)", 1, 100, 20)

# -----------------------
# --- Tabs: Monitoring | Forecasting | Optimization
# -----------------------
tab1, tab2, tab3 = st.tabs(["Monitoring", "Forecasting", "Optimization"])

# Pre-fetch all selected portfolios data (shared)
fetched = {}
for name in selected:
    holdings = PORTFOLIOS[name]
    market = "Kenya" if name.lower() == "kenya" else "US"
    df, prices = build_portfolio(holdings, market=market)
    fetched[name] = {"portfolio": df, "prices": prices, "holdings": holdings}

# --------- Monitoring Tab ----------
with tab1:
    st.header("Monitoring")
    st.write("Live portfolio snapshots, alerts, and heatmap.")
    # Alerts area
    st.subheader("Alerts")
    for name, obj in fetched.items():
        pf = obj["portfolio"]
        if pf.empty or pf["Value"].empty:
            st.warning(f"{name}: no data available")
            continue
        vol, sharpe, mdd = risk_metrics_from_series(pf)
        last_value = pf["Value"].iloc[-1]
        peak = pf["Value"].cummax().max()
        drawdown_pct = (last_value / peak - 1) * 100  # percent
        if drawdown_pct <= -drawdown_threshold:
            st.error(f"🚨 {name} drawdown = {drawdown_pct:.2f}% (threshold = -{drawdown_threshold}%)")
        else:
            st.info(f"{name} drawdown = {drawdown_pct:.2f}% | Vol (ann): {vol:.2%} | Sharpe: {sharpe:.2f}")

    # Heatmap overview
    st.subheader("Heatmap: latest pct change per ticker")
    rows = []
    for name, obj in fetched.items():
        prices = obj["prices"]
        if prices.empty:
            continue
        if len(prices) >= 2:
            latest = prices.iloc[-1]
            prev = prices.iloc[-2]
            for t in prices.columns:
                pct = (latest[t] / prev[t] - 1)
                rows.append({"Portfolio": name, "Ticker": t, "Last": latest[t], "PctChange": pct})
    if rows:
        heat_df = pd.DataFrame(rows)
        heat_pivot = heat_df.pivot(index="Ticker", columns="Portfolio", values="PctChange")
        st.dataframe(heat_pivot.style.format("{:.2%}", na_rep="n/a"))
    else:
        st.write("No tickers/prices available for heatmap.")

    # Detail panels
    st.subheader("Portfolio details")
    for name, obj in fetched.items():
        pf = obj["portfolio"]
        st.markdown(f"**{name}**")
        if pf.empty:
            st.write("No data")
            continue
        col1, col2 = st.columns([2, 3])
        with col1:
            vol, sharpe, mdd = risk_metrics_from_series(pf)
            st.metric("Volatility (ann.)", f"{vol:.2%}" if not np.isnan(vol) else "n/a")
            st.metric("Sharpe", f"{sharpe:.2f}" if not np.isnan(sharpe) else "n/a")
            st.metric("Max Drawdown", f"{mdd:.2%}" if not np.isnan(mdd) else "n/a")
        with col2:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(pf["Date"], pf["Value"], label=name)
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            fig.autofmt_xdate(rotation=45)
            ax.legend()
            st.pyplot(fig)

# --------- Forecasting Tab ----------
with tab2:
    st.header("Forecasting (Monte Carlo)")
    st.write("Run Monte Carlo simulations per selected portfolio and inspect percentiles.")
    for name, obj in fetched.items():
        pf = obj["portfolio"]
        if pf.empty:
            st.write(f"{name}: no data for forecasting")
            continue
        st.subheader(name)
        sims, summary = monte_carlo_sim(pf, horizon=forecast_horizon, sims=forecast_sims)
        if sims.size == 0 or not summary:
            st.write("Monte Carlo could not run (insufficient returns).")
            continue
        p5, p50, p95 = summary["p5"], summary["p50"], summary["p95"]
        st.write(f"Forecast horizon: {forecast_horizon} days — p5: ${p5:,.0f}, median: ${p50:,.0f}, p95: ${p95:,.0f}")

        # Overlay a subsample of simulation paths on historical chart
        fig, ax = plt.subplots(figsize=(9, 4))
        # historical
        ax.plot(pf["Date"], pf["Value"], color="black", linewidth=2, label="Historical")
        # sample 50 paths (or fewer)
        nplot = min(50, sims.shape[0])
        for i in range(nplot):
            path = sims[i, :]
            # create future dates
            last_date = pf["Date"].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon+1)
            ax.plot(future_dates, path, alpha=0.15)
        # percentiles at horizon
        ax.axhline(p5, color="red", linestyle="--", label="p5")
        ax.axhline(p50, color="gray", linestyle="--", label="median")
        ax.axhline(p95, color="green", linestyle="--", label="p95")
        ax.set_title(f"{name} Monte Carlo (subset of {forecast_sims} sims)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=45)
        ax.legend()
        st.pyplot(fig)

        # show distribution of final values
        final_vals = summary["final_array"]
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.hist(final_vals, bins=40)
        ax2.set_title("Distribution of final portfolio values (simulations)")
        st.pyplot(fig2)

# --------- Optimization Tab ----------
with tab3:
    st.header("Optimization (Max Sharpe)")
    st.write("Compare current allocation vs optimized allocation (max Sharpe, long-only).")
    for name, obj in fetched.items():
        st.subheader(name)
        prices = obj["prices"]
        holdings = obj["holdings"]
        if prices.empty:
            st.write("No price data to optimize.")
            continue

        # current weights by market value
        latest_prices = prices.iloc[-1]
        values = latest_prices * pd.Series(holdings)
        current_weights = values / values.sum()

        # optimized
        opt_weights, opt_ret, opt_vol, opt_sharpe = optimize_weights(prices)

        # current metrics
        portfolio_df = obj["portfolio"]
        cur_vol, cur_sharpe, cur_mdd = risk_metrics_from_series(portfolio_df)

        st.write("Current weights (by market value):")
        st.dataframe(current_weights.apply(lambda x: f"{x:.2%}"))

        st.write("Optimized weights (max Sharpe):")
        st.dataframe(opt_weights.apply(lambda x: f"{x:.2%}"))

        # metrics comparison
        st.write("Metrics comparison:")
        comparison = pd.DataFrame({
            "Metric": ["Annualized Volatility", "Sharpe"],
            "Current": [f"{cur_vol:.2%}" if not np.isnan(cur_vol) else "n/a", f"{cur_sharpe:.2f}" if not np.isnan(cur_sharpe) else "n/a"],
            "Optimized": [f"{opt_vol:.2%}" if not np.isnan(opt_vol) else "n/a", f"{opt_sharpe:.2f}" if not np.isnan(opt_sharpe) else "n/a"]
        })
        st.table(comparison)

        # pie charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.pie(current_weights, labels=current_weights.index, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Current Allocation")
        ax2.pie(opt_weights, labels=opt_weights.index, autopct="%1.1f%%", startangle=90)
        ax2.set_title("Optimized Allocation")
        st.pyplot(fig)

# -----------------------
# --- Export selected results (ZIP)
# -----------------------
if export_all:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, obj in fetched.items():
            pf = obj["portfolio"]
            prices = obj["prices"]
            holdings = obj["holdings"]
            if not pf.empty:
                zf.writestr(f"{name}_Portfolio.csv", pf.to_csv(index=False))
            if prices is not None and not prices.empty:
                zf.writestr(f"{name}_Prices.csv", prices.to_csv())
            zf.writestr(f"{name}_Holdings.txt", ("\n".join([f"{k},{v}" for k, v in holdings.items()])))
        zf.writestr("Helios_Metadata.txt", f"Exported: {pd.Timestamp.now()}\nForecast horizon: {forecast_horizon}\nMonte Carlo sims: {forecast_sims}\n")
    buffer.seek(0)
    st.success("Export ready")
    st.download_button("Download Portfolio Book (ZIP)", data=buffer.getvalue(), file_name="Helios_Day29_PortfolioBook.zip", mime="application/zip")

# -----------------------
# --- End
# -----------------------
st.write("Day 29 complete — Monitoring, Forecasting, and Optimization panels are available.")