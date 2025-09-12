# helios_app.py  — Helios (Refactored: Monitoring | Forecasting | Optimization)
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
import time

plt.rcParams["figure.dpi"] = 120

# -----------------------
# --- Page config
# -----------------------
st.set_page_config(page_title="Helios — Dashboard", layout="wide")
st.title("Helios — Dashboard")

# -----------------------
# --- Example portfolios (editable)
# -----------------------
PORTFOLIOS = {
    "US": {"AAPL": 10, "MSFT": 5, "GOOGL": 2},
    "Kenya": {"EQTY.NBO": 10, "SCOM.NBO": 5, "KCB.NBO": 2},  # placeholder ticker names for NSE (replace if you have local symbols)
    "Energy": {"XOM": 15, "CVX": 8, "BP": 12}
}

# -----------------------
# --- Helpers (prices, portfolio construction, risk, MC, optimization)
# -----------------------
def fetch_prices(tickers, start=None, end=None):
    """Return price DataFrame indexed by date (Adj Close if available)."""
    if not tickers:
        return pd.DataFrame()
    try:
        df = yf.download(tickers, start=start, end=end, progress=False, threads=False)
    except Exception as e:
        st.warning(f"yfinance error for {tickers}: {e}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    prices = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    return prices.ffill().dropna()

def build_portfolio(holdings, market="US", start="2024-01-01"):
    """Return (portfolio_df, prices_df) where portfolio_df has Date, Value, Return."""
    tickers = list(holdings.keys())
    if market == "Kenya":
        # Mock Kenya data for demo (unless you provide NSE tickers that Yahoo supports)
        dates = pd.date_range(start=start, periods=200)
        mock = np.linspace(1000, 1200, len(dates)) + np.random.randn(len(dates)) * 5
        # scale mock series by holdings to create per-ticker series
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
    """Given a portfolio df with 'Return', compute vol (ann), sharpe, max drawdown."""
    if series.empty or "Return" not in series:
        return np.nan, np.nan, np.nan
    returns = series["Return"].dropna()
    vol = returns.std() * np.sqrt(252) if not returns.empty else np.nan
    sharpe = (returns.mean() * 252) / vol if vol and vol > 0 else np.nan
    cum = (1 + returns).cumprod() if not returns.empty else pd.Series(dtype=float)
    mdd = (cum - cum.cummax()).min() if not returns.empty else np.nan
    return vol, sharpe, mdd

def monte_carlo_sim(portfolio_df, horizon=252, sims=1000, random_seed=None):
    """Return sims (array of shape sims x (horizon+1)) and percentile summary (p5, p50, p95)."""
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
    """Max Sharpe optimizer (long-only, fully invested). Returns (weights, ret, vol, sharpe)."""
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
# --- Fixed Income scrapers (cached)
# -----------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (HeliosBot/1.0)"}

@st.cache_data(ttl=3600)
def fetch_cbk_tbills_cached():
    """Scrape CBK T-bills — returns dict with dataframe and fetched_at timestamp."""
    url = "https://www.centralbank.go.ke/securities/treasury-bills/"
    try:
        resp = requests.get(url, timeout=12, headers=HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            tenors, yields = [], []
            for r in rows:
                cols = [c.get_text(strip=True) for c in r.find_all(["td", "th"])]
                if not cols:
                    continue
                if len(cols) >= 2 and "Days" in cols[0]:
                    try:
                        tenor = int(cols[0].split()[0])
                        rate_text = cols[1].replace("%", "").replace(",", "").strip()
                        rate = float(rate_text)
                        tenors.append(tenor)
                        yields.append(rate)
                    except Exception:
                        continue
            if tenors and yields:
                df = pd.DataFrame({"Tenor (Days)": tenors, "Yield (%)": yields})
                df = df.sort_values("Tenor (Days)").reset_index(drop=True)
                return {"df": df, "fetched_at": pd.Timestamp.now()}
        return {"df": pd.DataFrame({"Tenor (Days)": [], "Yield (%)": []}), "fetched_at": pd.Timestamp.now()}
    except Exception as e:
        st.warning(f"CBK T-bill fetch error: {e}")
        return {"df": pd.DataFrame({"Tenor (Days)": [], "Yield (%)": []}), "fetched_at": pd.Timestamp.now()}

@st.cache_data(ttl=3600)
def fetch_kenya_bonds_cached():
    """Scrape Investing.com for Kenya bond/eurobond yields — returns dict with dataframe and timestamp."""
    url = "https://www.investing.com/rates-bonds/kenya-government-bonds"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")
        maturities, yields = [], []
        for table in tables:
            rows = table.find_all("tr")
            for r in rows:
                cols = [c.get_text(strip=True) for c in r.find_all("td")]
                if len(cols) >= 2:
                    try:
                        mat_text = cols[0]
                        mat_num = None
                        for tok in mat_text.split():
                            tok_clean = tok.replace("Y", "").replace("Yr", "").replace("Year", "").replace("year", "")
                            if tok_clean.replace('.', '', 1).isdigit():
                                mat_num = float(tok_clean)
                                break
                            if tok.endswith("Y") and tok[:-1].isdigit():
                                mat_num = float(tok[:-1])
                                break
                        if mat_num is None:
                            continue
                        y_val = float(cols[1].replace("%", "").replace(",", "").strip())
                        maturities.append(mat_num)
                        yields.append(y_val)
                    except Exception:
                        continue
            if maturities and yields:
                df = pd.DataFrame({"Maturity (Years)": maturities, "Yield (%)": yields})
                df = df.drop_duplicates().sort_values("Maturity (Years)").reset_index(drop=True)
                return {"df": df, "fetched_at": pd.Timestamp.now()}
        return {"df": pd.DataFrame({"Maturity (Years)": [], "Yield (%)": []}), "fetched_at": pd.Timestamp.now()}
    except Exception as e:
        st.warning(f"Investing.com fetch error: {e}")
        return {"df": pd.DataFrame({"Maturity (Years)": [], "Yield (%)": []}), "fetched_at": pd.Timestamp.now()}

# Helper to clear cached FI data (manual refresh)
def refresh_fixed_income_cache():
    try:
        st.cache_data.clear()
    except Exception:
        # older/newer versions may differ; best-effort
        pass

# -----------------------
# --- Sidebar controls
# -----------------------
st.sidebar.header("Controls")
selected = st.sidebar.multiselect("Select portfolios", list(PORTFOLIOS.keys()), default=["US"])
monitor_refresh = st.sidebar.button("Refresh data (manual)")
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 30, 756, 252, step=30)
forecast_sims = st.sidebar.slider("Monte Carlo sims", 100, 3000, 1000, step=100)
export_all = st.sidebar.checkbox("Export selected results (ZIP)")

drawdown_threshold = st.sidebar.slider("Drawdown alert threshold (%)", 1, 100, 20)
watchlist_input = st.sidebar.text_input("Watchlist (comma-separated tickers)", value="TSLA, NVDA")
watch_alert_pct = st.sidebar.slider("Watchlist alert threshold (%)", 1, 50, 5)

show_fixed_income = st.sidebar.checkbox("Show Kenya Fixed Income Dashboard", value=True)
force_refresh_fi = st.sidebar.button("Refresh FI data (clear FI cache)")

# Clear FI cache when requested
if force_refresh_fi:
    refresh_fixed_income_cache()
    st.sidebar.success("Fixed income cache cleared. Data will refresh on next load.")

# -----------------------
# --- Tabs
# -----------------------
tab_monitor, tab_forecast, tab_opt = st.tabs(["Monitoring", "Forecasting", "Optimization"])

# Pre-fetch selected portfolios
fetched = {}
for name in selected:
    holdings = PORTFOLIOS[name]
    market = "Kenya" if name.lower() == "kenya" else "US"
    df, prices = build_portfolio(holdings, market=market)
    fetched[name] = {"portfolio": df, "prices": prices, "holdings": holdings}

# -----------------------
# --- Monitoring Tab
# -----------------------
with tab_monitor:
    st.header("Monitoring")
    st.write("Snapshots, alerts, heatmap, and fixed-income market view.")

    # Alerts: drawdown + risk metrics
    st.subheader("Portfolio Alerts & Risk")
    for name, obj in fetched.items():
        pf = obj["portfolio"]
        if pf.empty or pf["Value"].empty:
            st.warning(f"{name}: no data available")
            continue
        vol, sharpe, mdd = risk_metrics_from_series(pf)
        last_value = pf["Value"].iloc[-1]
        peak = pf["Value"].cummax().max()
        drawdown_pct = (last_value / peak - 1) * 100
        if drawdown_pct <= -drawdown_threshold:
            st.error(f"{name}: drawdown = {drawdown_pct:.2f}% (threshold = -{drawdown_threshold}%)")
        else:
            st.info(f"{name}: drawdown = {drawdown_pct:.2f}% | Vol (ann): {vol:.2%} | Sharpe: {sharpe:.2f}")

    # Watchlist alerts
    st.subheader("Watchlist Alerts")
    wl = [w.strip().upper() for w in watchlist_input.split(",") if w.strip()]
    if wl:
        wl_prices = fetch_prices(wl, start=None, end=pd.Timestamp.today().strftime("%Y-%m-%d"))
        if not wl_prices.empty and len(wl_prices) >= 2:
            alerts = []
            for t in wl:
                if t in wl_prices.columns:
                    pct = (wl_prices[t].iloc[-1] / wl_prices[t].iloc[-2] - 1) * 100
                    if abs(pct) >= watch_alert_pct:
                        alerts.append((t, pct))
            if alerts:
                for t, pct in alerts:
                    st.warning(f"Watchlist alert: {t} moved {pct:.2f}% vs previous close (threshold {watch_alert_pct}%)")
            else:
                st.info("No watchlist alerts right now.")
        else:
            st.info("No watchlist price data available.")
    else:
        st.write("No watchlist tickers configured.")

    # Heatmap overview
    st.subheader("Heatmap: latest pct change per ticker")
    rows = []
    for name, obj in fetched.items():
        prices = obj["prices"]
        if prices.empty or len(prices) < 2:
            continue
        latest = prices.iloc[-1]
        prev = prices.iloc[-2]
        for t in prices.columns:
            try:
                pct = (latest[t] / prev[t] - 1)
            except Exception:
                pct = np.nan
            rows.append({"Portfolio": name, "Ticker": t, "Last": latest[t], "PctChange": pct})
    if rows:
        heat_df = pd.DataFrame(rows)
        heat_pivot = heat_df.pivot(index="Ticker", columns="Portfolio", values="PctChange")
        st.dataframe(heat_pivot.style.format("{:.2%}", na_rep="n/a"))
    else:
        st.write("No tickers/prices available for heatmap.")

    # Portfolio detail panels
    st.subheader("Portfolio Details")
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

    # Fixed Income summary (placed under Monitoring)
    if show_fixed_income:
        st.subheader("Kenya Fixed Income (CBK T-Bills + Bonds)")
        cbk = fetch_cbk_tbills_cached()
        bonds = fetch_kenya_bonds_cached()

        # Show CBK T-bills and timestamp
        st.markdown("**T-Bills (CBK)**")
        t_df = cbk.get("df", pd.DataFrame())
        t_ts = cbk.get("fetched_at", None)
        if not t_df.empty:
            st.write(f"Last updated: {t_ts}")
            st.dataframe(t_df)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(t_df["Tenor (Days)"], t_df["Yield (%)"], marker="o", linestyle="-")
            ax.set_xlabel("Tenor (Days)")
            ax.set_ylabel("Yield (%)")
            ax.set_title("Kenya T-Bill Yield Curve (CBK)")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("CBK T-bill data not available. Showing mock example.")
            mock_tb = pd.DataFrame({"Tenor (Days)": [91, 182, 364], "Yield (%)": [8.0, 8.3, 9.1]})
            st.dataframe(mock_tb)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(mock_tb["Tenor (Days)"], mock_tb["Yield (%)"], marker="o")
            st.pyplot(fig)

        # Investing.com bonds and timestamp
        st.markdown("**Government / Eurobond Yields (Investing.com)**")
        e_df = bonds.get("df", pd.DataFrame())
        e_ts = bonds.get("fetched_at", None)
        if not e_df.empty:
            st.write(f"Last updated: {e_ts}")
            st.dataframe(e_df)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(e_df["Maturity (Years)"], e_df["Yield (%)"], marker="s", color="orange")
            ax.set_xlabel("Maturity (Years)")
            ax.set_ylabel("Yield (%)")
            ax.set_title("Kenya Bond / Eurobond Yield Curve (scraped)")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("No live bond data available. Showing mock example.")
            mock_e = pd.DataFrame({"Maturity (Years)": [2, 5, 10, 30], "Yield (%)": [8.5, 9.2, 10.1, 11.3]})
            st.dataframe(mock_e)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(mock_e["Maturity (Years)"], mock_e["Yield (%)"], marker="o", color="orange")
            st.pyplot(fig)

        # Simple spread analysis if both present
        if (not t_df.empty) and (not e_df.empty):
            try:
                spread = e_df["Yield (%)"].mean() - t_df["Yield (%)"].mean()
                st.write(f"Average bond yield − average T-bill yield = {spread:.2f}%")
            except Exception:
                pass

# -----------------------
# --- Forecasting Tab
# -----------------------
with tab_forecast:
    st.header("Forecasting (Monte Carlo)")
    st.write("Monte Carlo simulations and forecast percentiles.")

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

        # overlay sample simulation paths on historical chart
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(pf["Date"], pf["Value"], color="black", linewidth=2, label="Historical")
        nplot = min(50, sims.shape[0])
        for i in range(nplot):
            path = sims[i, :]
            last_date = pf["Date"].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon + 1)
            ax.plot(future_dates, path, alpha=0.12)
        ax.axhline(p5, color="red", linestyle="--", label="p5")
        ax.axhline(p50, color="gray", linestyle="--", label="median")
        ax.axhline(p95, color="green", linestyle="--", label="p95")
        ax.set_title(f"{name} Monte Carlo (subset of {forecast_sims} sims)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=45)
        ax.legend()
        st.pyplot(fig)

        final_vals = summary["final_array"]
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.hist(final_vals, bins=40)
        ax2.set_title("Distribution of final portfolio values (simulations)")
        st.pyplot(fig2)

# -----------------------
# --- Optimization Tab
# -----------------------
with tab_opt:
    st.header("Optimization (Max Sharpe)")
    st.write("Compare current allocation vs optimized allocation (long-only, fully invested).")

    for name, obj in fetched.items():
        st.subheader(name)
        prices = obj["prices"]
        holdings = obj["holdings"]
        if prices.empty:
            st.write("No price data to optimize.")
            continue

        latest_prices = prices.iloc[-1]
        values = latest_prices * pd.Series(holdings)
        current_weights = values / values.sum()

        opt_weights, opt_ret, opt_vol, opt_sharpe = optimize_weights(prices)

        portfolio_df = obj["portfolio"]
        cur_vol, cur_sharpe, cur_mdd = risk_metrics_from_series(portfolio_df)

        st.write("Current weights (by market value):")
        st.dataframe(current_weights.apply(lambda x: f"{x:.2%}"))

        st.write("Optimized weights (max Sharpe):")
        st.dataframe(opt_weights.apply(lambda x: f"{x:.2%}"))

        comparison = pd.DataFrame({
            "Metric": ["Annualized Volatility", "Sharpe"],
            "Current": [f"{cur_vol:.2%}" if not np.isnan(cur_vol) else "n/a", f"{cur_sharpe:.2f}" if not np.isnan(cur_sharpe) else "n/a"],
            "Optimized": [f"{opt_vol:.2%}" if not np.isnan(opt_vol) else "n/a", f"{opt_sharpe:.2f}" if not np.isnan(opt_sharpe) else "n/a"]
        })
        st.table(comparison)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        try:
            ax1.pie(current_weights, labels=current_weights.index, autopct="%1.1f%%", startangle=90)
        except Exception:
            ax1.text(0.5, 0.5, "Cannot plot current allocation", ha="center")
        try:
            ax2.pie(opt_weights, labels=opt_weights.index, autopct="%1.1f%%", startangle=90)
        except Exception:
            ax2.text(0.5, 0.5, "Cannot plot optimized allocation", ha="center")
        ax1.set_title("Current Allocation")
        ax2.set_title("Optimized Allocation")
        st.pyplot(fig)

    # Export button (ZIP)
    st.markdown("---")
    st.write("Export selected results (portfolios + fixed income) as a ZIP file.")
    if st.button("Prepare Export ZIP"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for name, obj in fetched.items():
                pf, prices, holdings = obj["portfolio"], obj["prices"], obj["holdings"]
                if not pf.empty:
                    zf.writestr(f"{name}_Portfolio.csv", pf.to_csv(index=False))
                if prices is not None and not prices.empty:
                    zf.writestr(f"{name}_Prices.csv", prices.to_csv())
                zf.writestr(f"{name}_Holdings.txt", "\n".join([f"{k},{v}" for k, v in holdings.items()]))
            # add FI if available
            cbk = fetch_cbk_tbills_cached()
            bonds = fetch_kenya_bonds_cached()
            t_df = cbk.get("df", pd.DataFrame())
            e_df = bonds.get("df", pd.DataFrame())
            if not t_df.empty:
                zf.writestr("Kenya_TBills.csv", t_df.to_csv(index=False))
            if not e_df.empty:
                zf.writestr("Kenya_Bonds.csv", e_df.to_csv(index=False))
            zf.writestr("Helios_Metadata.txt", f"Exported: {pd.Timestamp.now()}\nForecast horizon: {forecast_horizon}\nMonte Carlo sims: {forecast_sims}\n")
        buffer.seek(0)
        st.success("Export ready — click to download")
        st.download_button("⬇️ Download Helios Book (ZIP)", data=buffer.getvalue(), file_name="Helios_PortfolioBook.zip", mime="application/zip")

# -----------------------
# --- Manual refresh
# -----------------------
if monitor_refresh:
    st.experimental_rerun()

st.info("Helios loaded. Use the tabs above to work with Monitoring, Forecasting or Optimization.")
