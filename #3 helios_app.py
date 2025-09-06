# helios_app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

plt.rcParams["figure.dpi"] = 120

st.set_page_config(page_title="Helios Portfolio Dashboard", layout="wide")

st.title("🌞 Helios Portfolio Dashboard")

# --- Portfolio Holdings ---
us_portfolio_holdings = {"AAPL": 10, "MSFT": 5, "GOOGL": 2}
kenya_portfolio_holdings = {"Equity Bank": 10, "Safaricom": 5, "KCB Group": 2}

# --- Functions ---
def fetch_portfolio(market):
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    start = '2024-01-01'
    if market == "US":
        tickers = list(us_portfolio_holdings.keys())
        data = yf.download(tickers, start=start, end=today, threads=False)
        if len(tickers) == 1:
            prices = data["Adj Close"].to_frame() if "Adj Close" in data.columns else data["Close"].to_frame()
            prices.columns = tickers
        else:
            prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
        prices = prices.ffill().dropna()
        portfolio_values = (prices * pd.Series(us_portfolio_holdings)).sum(axis=1)
    else:
        dates = pd.date_range(start=start, periods=100)
        mock_values = np.linspace(1000, 1200, 100) + np.random.randn(100) * 5
        portfolio_values = pd.Series(mock_values, index=dates)
        tickers = list(kenya_portfolio_holdings.keys())
        prices = pd.DataFrame({t: mock_values * (v / 10) for t, v in kenya_portfolio_holdings.items()}, index=dates)
    portfolio = pd.DataFrame({"Date": portfolio_values.index, "Value": portfolio_values})
    portfolio["Return"] = portfolio["Value"].pct_change().fillna(0)
    return portfolio, prices

def risk_metrics(portfolio):
    returns = portfolio["Return"]
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0
    cum = (1 + returns).cumprod()
    mdd = (cum - cum.cummax()).min()
    return vol, sharpe, mdd

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

def optimize_portfolio(prices):
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(prices.columns)

    def neg_sharpe(weights):
        port_return = np.sum(mean_returns * weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe = port_return / port_vol
        return -sharpe

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(neg_sharpe, num_assets * [1. / num_assets, ], bounds=bounds, constraints=constraints)
    optimized_weights = pd.Series(result.x, index=prices.columns)
    port_return = np.sum(mean_returns * optimized_weights) * 252
    port_vol = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix * 252, optimized_weights)))
    sharpe = port_return / port_vol
    return optimized_weights, port_return, port_vol, sharpe

def apply_scenario(prices, scenario_type, market):
    stressed_prices = prices.copy()
    if scenario_type == "Market Crash -10%":
        stressed_prices = stressed_prices * 0.90
    elif scenario_type == "Tech Sector Shock -15%":
        for col in stressed_prices.columns:
            if col in ["AAPL", "MSFT", "GOOGL"]:
                stressed_prices[col] = stressed_prices[col] * 0.85
    elif scenario_type == "Interest Rate Spike -5%":
        stressed_prices = stressed_prices * 0.95
    return stressed_prices

def portfolio_from_prices(prices, market):
    if market == "US":
        weights = pd.Series(us_portfolio_holdings)
    else:
        weights = pd.Series(kenya_portfolio_holdings)
    portfolio_values = (prices * weights).sum(axis=1)
    portfolio = pd.DataFrame({"Date": portfolio_values.index, "Value": portfolio_values})
    portfolio["Return"] = portfolio["Value"].pct_change().fillna(0)
    return portfolio

# --- Sidebar Inputs ---
st.sidebar.header("Settings")
market = st.sidebar.selectbox("Select Market", ["US", "Kenya"])
forecast_days = st.sidebar.slider("Forecast Days", 30, 756, 252, step=30)
scenario = st.sidebar.selectbox("Scenario", ["Market Crash -10%", "Tech Sector Shock -15%", "Interest Rate Spike -5%"])
export = st.sidebar.checkbox("Export Results")

# --- Run Dashboard ---
portfolio, prices = fetch_portfolio(market)
sims, mc_alert = monte_carlo(portfolio, forecast_days)
vol, sharpe, mdd = risk_metrics(portfolio)
opt_weights, exp_return, opt_vol, opt_sharpe = optimize_portfolio(prices)
stressed_prices = apply_scenario(prices, scenario, market)
stressed_portfolio = portfolio_from_prices(stressed_prices, market)
stressed_vol, stressed_sharpe, stressed_mdd = risk_metrics(stressed_portfolio)

# --- Display Metrics ---
st.subheader("📊 Portfolio Risk Metrics")
st.write(f"Normal: Volatility={vol:.2%}, Sharpe={sharpe:.2f}, Max Drawdown={mdd:.2%}")
st.write(f"Scenario ({scenario}): Volatility={stressed_vol:.2%}, Sharpe={stressed_sharpe:.2f}, Max Drawdown={stressed_mdd:.2%}")
st.write("⚠️ Monte Carlo Alerts:", mc_alert)

st.subheader("💡 Suggested Optimal Allocation")
st.dataframe(opt_weights.apply(lambda x: f"{x:.2%}"))

# --- Portfolio Value & Scenario Plot ---
st.subheader("📈 Portfolio Value & Scenario")
fig, ax = plt.subplots()
ax.plot(portfolio["Date"], portfolio["Value"], label="Portfolio")
ax.plot(stressed_portfolio["Date"], stressed_portfolio["Value"], label=f"Scenario: {scenario}")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.legend()

# ✅ Fix x-axis dates
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# --- Allocation Pie Chart ---
st.subheader("🥧 Allocation Pie Chart")
fig2, ax2 = plt.subplots()
ax2.pie(opt_weights, labels=opt_weights.index, autopct='%1.1f%%', startangle=90)
plt.tight_layout()
st.pyplot(fig2)

# --- Export ---
if export:
    pd.DataFrame(sims).to_csv(f"Helios_Simulations_{market}.csv", index=False)
    opt_weights.to_csv(f"Helios_OptimalAllocation_{market}.csv")
    stressed_portfolio.to_csv(f"Helios_{market}_Scenario_{scenario.replace(' ','_')}.csv")
    st.success("💾 All results exported successfully!")
