# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import matplotlib.dates as mdates
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

plt.rcParams["figure.dpi"] = 120

# --- Portfolio Holdings ---
us_portfolio_holdings = {"AAPL":10, "MSFT":5, "GOOGL":2}
kenya_portfolio_holdings = {"Equity Bank":10, "Safaricom":5, "KCB Group":2}

# --- Fetch Portfolio ---
def fetch_portfolio(market):
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    start = '2024-01-01'
    if market=="US":
        tickers = list(us_portfolio_holdings.keys())
        data = yf.download(tickers, start=start, end=today, threads=False)
        if len(tickers)==1:
            prices = data["Adj Close"].to_frame() if "Adj Close" in data.columns else data["Close"].to_frame()
            prices.columns = tickers
        else:
            prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
        prices = prices.ffill().dropna()
        portfolio_values = (prices * pd.Series(us_portfolio_holdings)).sum(axis=1)
    else:
        dates = pd.date_range(start=start, periods=100)
        mock_values = np.linspace(1000,1200,100)+np.random.randn(100)*5
        portfolio_values = pd.Series(mock_values, index=dates)
        tickers = list(kenya_portfolio_holdings.keys())
        prices = pd.DataFrame({t: mock_values*(v/10) for t,v in kenya_portfolio_holdings.items()}, index=dates)
    
    portfolio = pd.DataFrame({"Date": portfolio_values.index, "Value": portfolio_values})
    portfolio["Return"] = portfolio["Value"].pct_change().fillna(0)
    return portfolio, prices

# --- Risk Metrics ---
def risk_metrics(portfolio):
    returns = portfolio["Return"]
    vol = returns.std()*np.sqrt(252)
    sharpe = (returns.mean()*252)/vol if vol>0 else 0
    cum = (1+returns).cumprod()
    mdd = (cum - cum.cummax()).min()
    return vol, sharpe, mdd

# --- Monte Carlo ---
def monte_carlo(portfolio, forecast_days=252, num_sim=500):
    returns = portfolio["Return"].dropna()
    last_price = portfolio["Value"].iloc[-1]
    mean, std = returns.mean(), returns.std()
    sims=[]
    for _ in range(num_sim):
        prices=[last_price]
        for _ in range(forecast_days):
            prices.append(prices[-1]*(1+np.random.normal(mean,std)))
        sims.append(prices)
    final_prices=[s[-1] for s in sims]
    p5, p95 = np.percentile(final_prices,5), np.percentile(final_prices,95)
    alert_msg = f"Monte Carlo worst-case (5%): ${p5:,.0f} | best-case (95%): ${p95:,.0f}"
    return sims, alert_msg

# --- Portfolio Optimization ---
def optimize_portfolio(prices):
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(prices.columns)
    
    def neg_sharpe(weights):
        port_return = np.sum(mean_returns*weights)*252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix*252, weights)))
        sharpe = port_return/port_vol
        return -sharpe
    
    constraints = ({'type':'eq','fun':lambda x: np.sum(x)-1})
    bounds = tuple((0,1) for _ in range(num_assets))
    result = minimize(neg_sharpe, num_assets*[1./num_assets,], bounds=bounds, constraints=constraints)
    
    optimized_weights = pd.Series(result.x, index=prices.columns)
    port_return = np.sum(mean_returns*optimized_weights)*252
    port_vol = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix*252, optimized_weights)))
    sharpe = port_return/port_vol
    return optimized_weights, port_return, port_vol, sharpe

# --- Multi-Benchmark Fetch ---
def fetch_benchmarks(benchmarks, start, end):
    bench_data = {}
    for b in benchmarks:
        try:
            data = yf.download(b, start=start, end=end, threads=False)
            if not data.empty:
                prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
                bench_data[b] = prices.ffill().dropna()
        except:
            continue
    return bench_data

# --- Rolling Beta & Correlation ---
def rolling_beta_corr(portfolio, bench_data, window=21):
    results = {}
    df_port = portfolio.set_index("Date")["Value"]
    returns_port = df_port.pct_change().dropna()
    for b, series in bench_data.items():
        bench_ret = series.pct_change().dropna()
        min_len = min(len(returns_port), len(bench_ret))
        rolling_beta=[]
        rolling_corr=[]
        for i in range(window, min_len):
            X = bench_ret.values[i-window:i].reshape(-1,1)
            y = returns_port.values[i-window:i]
            reg = LinearRegression().fit(X,y)
            rolling_beta.append(reg.coef_[0])
            rolling_corr.append(np.corrcoef(y,X.flatten())[0,1])
        dates = returns_port.index[window:window+len(rolling_beta)]
        results[b] = {"dates":dates, "beta":rolling_beta, "corr":rolling_corr}
    return results

# --- Scenario Stress Test ---
def apply_scenario(prices, scenario_type, market):
    stressed_prices = prices.copy()
    if scenario_type=="Market Crash -10%":
        stressed_prices = stressed_prices * 0.90
    elif scenario_type=="Tech Sector Shock -15%":
        for col in stressed_prices.columns:
            if col in ["AAPL","MSFT","GOOGL"]:
                stressed_prices[col] = stressed_prices[col]*0.85
    elif scenario_type=="Interest Rate Spike -5%":
        stressed_prices = stressed_prices * 0.95
    return stressed_prices

def portfolio_from_prices(prices, market):
    if market=="US":
        weights = pd.Series(us_portfolio_holdings)
    else:
        weights = pd.Series(kenya_portfolio_holdings)
    portfolio_values = (prices * weights).sum(axis=1)
    portfolio = pd.DataFrame({"Date":portfolio_values.index,"Value":portfolio_values})
    portfolio["Return"]=portfolio["Value"].pct_change().fillna(0)
    return portfolio

# --- Master Dashboard ---
def helios_master_dashboard(forecast_days=252, market="US", scenario="Market Crash -10%", export=False):
    portfolio, prices = fetch_portfolio(market)
    sims, mc_alert = monte_carlo(portfolio, forecast_days)
    vol, sharpe, mdd = risk_metrics(portfolio)
    
    # --- Optimization ---
    opt_weights, exp_return, opt_vol, opt_sharpe = optimize_portfolio(prices)
    
    # --- Benchmarks ---
    if market=="US":
        benchmark_symbols=["^GSPC","^IXIC","^DJI"]
    else:
        benchmark_symbols=["^NSE20"]
    bench_data=fetch_benchmarks(benchmark_symbols, portfolio["Date"].iloc[0], portfolio["Date"].iloc[-1])
    
    # --- Rolling Beta & Correlation ---
    beta_corr_results = rolling_beta_corr(portfolio, bench_data)
    
    # --- Scenario Stress Test ---
    stressed_prices = apply_scenario(prices, scenario, market)
    stressed_portfolio = portfolio_from_prices(stressed_prices, market)
    stressed_vol, stressed_sharpe, stressed_mdd = risk_metrics(stressed_portfolio)
    
    # --- Plot Portfolio & Scenario ---
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(portfolio["Date"], portfolio["Value"], label="Portfolio")
    ax.plot(stressed_portfolio["Date"], stressed_portfolio["Value"], label=f"Scenario: {scenario}", linestyle="--")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45)
    ax.set_title(f"{market} Portfolio Value & Scenario")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    st.pyplot(fig)
    
    # --- Rolling Beta ---
    fig, ax = plt.subplots(figsize=(10,4))
    for b in beta_corr_results:
        ax.plot(beta_corr_results[b]["dates"], beta_corr_results[b]["beta"], label=f"{b} Beta")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45)
    ax.set_title("Rolling Beta vs Benchmarks")
    ax.legend()
    st.pyplot(fig)
    
    # --- Rolling Correlation ---
    fig, ax = plt.subplots(figsize=(10,4))
    for b in beta_corr_results:
        ax.plot(beta_corr_results[b]["dates"], beta_corr_results[b]["corr"], label=f"{b} Correlation")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45)
    ax.set_title("Rolling Correlation vs Benchmarks")
    ax.legend()
    st.pyplot(fig)
    
    # --- Risk Metrics ---
    st.subheader(f"📊 {market} Portfolio Risk Snapshot")
    st.write(f"Volatility: {vol:.2%}, Sharpe: {sharpe:.2f}, Max Drawdown: {mdd:.2%}")
    st.write(f"Scenario ({scenario}): Volatility: {stressed_vol:.2%}, Sharpe: {stressed_sharpe:.2f}, Max Drawdown: {stressed_mdd:.2%}")
    st.warning(mc_alert)
    
    # --- Optimal Allocation ---
    st.subheader("💡 Suggested Optimal Allocation")
    st.write(opt_weights.apply(lambda x: f"{x:.2%}"))
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(opt_weights, labels=opt_weights.index, autopct='%1.1f%%', startangle=90)
    ax.set_title(f"{market} Portfolio Suggested Allocation")
    st.pyplot(fig)

    # --- Export ---
    if export:
        sims_df = pd.DataFrame(sims)
        opt_weights_df = opt_weights.to_frame(name="Weight")
        stressed_df = stressed_portfolio.copy()

        st.success("💾 Results exported successfully! Download below:")

        st.download_button(
            "⬇️ Monte Carlo Simulations",
            sims_df.to_csv(index=False),
            file_name=f"Helios_Simulations_{market}.csv",
            mime="text/csv"
        )

        st.download_button(
            "⬇️ Optimal Allocation",
            opt_weights_df.to_csv(),
            file_name=f"Helios_OptimalAllocation_{market}.csv",
            mime="text/csv"
        )

        st.download_button(
            "⬇️ Scenario Portfolio",
            stressed_df.to_csv(),
            file_name=f"Helios_{market}_Scenario_{scenario.replace(' ','_')}.csv",
            mime="text/csv"
        )

# --- Streamlit Sidebar ---
st.sidebar.header("Helios Dashboard Controls")
forecast_days = st.sidebar.slider("Forecast Days", min_value=30, max_value=756, step=30, value=252)
market = st.sidebar.selectbox("Market", ["US","Kenya"])
scenario = st.sidebar.selectbox("Scenario", ["Market Crash -10%","Tech Sector Shock -15%","Interest Rate Spike -5%"])
export = st.sidebar.checkbox("Export Results")

# --- Run Dashboard ---
helios_master_dashboard(forecast_days, market, scenario, export)
