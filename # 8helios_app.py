# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import matplotlib.dates as mdates
from scipy.optimize import minimize
import io, zipfile

plt.rcParams["figure.dpi"] = 120

# --- Portfolio Definitions (Mini-Aladdin Book) ---
portfolios = {
    "US": {"AAPL": 10, "MSFT": 5, "GOOGL": 2},
    "Kenya": {"Equity Bank": 10, "Safaricom": 5, "KCB Group": 2},
    "Energy": {"XOM": 15, "CVX": 8, "BP": 12}
}

# --- Fetch Portfolio ---
def fetch_portfolio(holdings, market="US"):
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    start = '2024-01-01'
    tickers = list(holdings.keys())

    if market != "Kenya":
        data = yf.download(tickers, start=start, end=today, threads=False)
        if len(tickers) == 1:
            prices = data["Adj Close"].to_frame() if "Adj Close" in data.columns else data["Close"].to_frame()
            prices.columns = tickers
        else:
            prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
        prices = prices.ffill().dropna()
        portfolio_values = (prices * pd.Series(holdings)).sum(axis=1)
    else:
        # Kenya market = mock data
        dates = pd.date_range(start=start, periods=100)
        mock_values = np.linspace(1000, 1200, 100) + np.random.randn(100) * 5
        portfolio_values = pd.Series(mock_values, index=dates)
        prices = pd.DataFrame({t: mock_values * (v / 10) for t, v in holdings.items()}, index=dates)

    portfolio = pd.DataFrame({"Date": portfolio_values.index, "Value": portfolio_values})
    portfolio["Return"] = portfolio["Value"].pct_change().fillna(0)
    return portfolio, prices

# --- Risk Metrics ---
def risk_metrics(portfolio):
    returns = portfolio["Return"]
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0
    cum = (1 + returns).cumprod()
    mdd = (cum - cum.cummax()).min()
    return vol, sharpe, mdd

# --- Monte Carlo Simulation ---
def monte_carlo(portfolio, forecast_days=252, num_sim=500):
    returns = portfolio["Return"].dropna()
    last_price = portfolio["Value"].iloc[-1]
    mean, std = returns.mean(), returns.std()
    sims = []
    for _ in range(num_sim):
        prices = [last_price]
        for _ in range(forecast_days):
            prices.append(prices[-1] * (1 + np.random.normal(mean, std)))
        sims.append(prices)
    final_prices = [s[-1] for s in sims]
    p5, p95 = np.percentile(final_prices, 5), np.percentile(final_prices, 95)
    alert_msg = f"Monte Carlo worst-case (5%): ${p5:,.0f} | best-case (95%): ${p95:,.0f}"
    return sims, alert_msg

# --- Portfolio Optimization ---
def optimize_portfolio(prices):
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(prices.columns)

    def neg_sharpe(weights):
        port_return = np.sum(mean_returns * weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        return -port_return / port_vol

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(neg_sharpe, num_assets * [1. / num_assets, ], bounds=bounds, constraints=constraints)

    optimized_weights = pd.Series(result.x, index=prices.columns)
    port_return = np.sum(mean_returns * optimized_weights) * 252
    port_vol = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix * 252, optimized_weights)))
    sharpe = port_return / port_vol
    return optimized_weights, port_return, port_vol, sharpe

# --- Streamlit UI ---
st.set_page_config(page_title="Helios Mini-Aladdin", layout="wide")
st.title("🌞 Helios Day 27 – Mini-Aladdin Portfolio Manager")

# Sidebar
st.sidebar.header("Controls")
selected_portfolios = st.sidebar.multiselect("Select Portfolios to Analyze", list(portfolios.keys()), default=["US"])
forecast_days = st.sidebar.slider("Forecast Days", min_value=30, max_value=756, step=30, value=252)
export = st.sidebar.checkbox("Export Results")

# --- Analysis ---
if selected_portfolios:
    all_results = {}
    consolidated_values = None

    for name in selected_portfolios:
        st.subheader(f"📂 Portfolio: {name}")
        portfolio, prices = fetch_portfolio(portfolios[name], market="Kenya" if name=="Kenya" else "US")
        sims, mc_alert = monte_carlo(portfolio, forecast_days)
        vol, sharpe, mdd = risk_metrics(portfolio)
        opt_weights, exp_return, opt_vol, opt_sharpe = optimize_portfolio(prices)

        # Save results for later consolidation/export
        all_results[name] = {
            "portfolio": portfolio,
            "prices": prices,
            "opt_weights": opt_weights
        }

        # Metrics
        st.write(f"Volatility: {vol:.2%}, Sharpe: {sharpe:.2f}, Max Drawdown: {mdd:.2%}")
        st.warning(mc_alert)

        # Plot
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(portfolio["Date"], portfolio["Value"], label=f"{name} Portfolio")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend()
        st.pyplot(fig)

        # Optimal Allocation
        st.write("Optimal Allocation:")
        st.dataframe(opt_weights.apply(lambda x: f"{x:.2%}"))

        # Allocation Pie Chart
        fig2, ax2 = plt.subplots()
        ax2.pie(opt_weights, labels=opt_weights.index, autopct='%1.1f%%', startangle=90)
        st.pyplot(fig2)

        # Consolidation
        if consolidated_values is None:
            consolidated_values = portfolio.set_index("Date")["Value"]
        else:
            consolidated_values = consolidated_values.add(portfolio.set_index("Date")["Value"], fill_value=0)

    # --- Consolidated Master Portfolio ---
    master_df = None
    if len(selected_portfolios) > 1:
        st.subheader("🌍 Consolidated Master Portfolio")
        master_df = pd.DataFrame({"Date": consolidated_values.index, "Value": consolidated_values.values})
        master_df["Return"] = master_df["Value"].pct_change().fillna(0)
        vol, sharpe, mdd = risk_metrics(master_df)
        st.write(f"Master Portfolio → Volatility: {vol:.2%}, Sharpe: {sharpe:.2f}, Max Drawdown: {mdd:.2%}")

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(master_df["Date"], master_df["Value"], label="Master Portfolio", color="black", linewidth=2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend()
        st.pyplot(fig)

    # --- Export as ZIP ---
    if export:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for name, data in all_results.items():
                data["portfolio"].to_csv(f"{name}_Portfolio.csv", index=False)
                data["prices"].to_csv(f"{name}_Prices.csv")
                data["opt_weights"].to_csv(f"{name}_OptimalWeights.csv")
                zf.write(f"{name}_Portfolio.csv")
                zf.write(f"{name}_Prices.csv")
                zf.write(f"{name}_OptimalWeights.csv")
            if master_df is not None:
                master_df.to_csv("MasterPortfolio.csv", index=False)
                zf.write("MasterPortfolio.csv")

        st.success("💾 All results exported successfully!")
        st.download_button(
            label="⬇️ Download All Results (ZIP)",
            data=buffer.getvalue(),
            file_name="Helios_PortfolioBook.zip",
            mime="application/zip"
        )
