# Import necessary libraries
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize

# Step 1: Data Preparation

# Function to load data from CSV files
def load_data(tickers, file_path):
    """Load data from CSV files."""
    dfs = []
    for ticker in tickers:
        df = pd.read_csv(f"{file_path}/{ticker}.csv", parse_dates=['Date'], index_col='Date')
        dfs.append(df['Adj Close'])
    return pd.concat(dfs, axis=1, keys=tickers)

# Define tickers and file path
tickers = ['AAPL', 'ORCL', 'MSFT', 'GOOG']
file_path = 'C:/Users/akmik/OneDrive/Desktop/BFIN 491/HW/HW5'

# Load data from CSV files
stock_prices = load_data(tickers, file_path)

# Step 2: Calculate Returns and Covariance Matrix

# Function to calculate monthly returns from daily adjusted closing prices
def calculate_returns(prices):
    """Calculates monthly returns from daily adjusted closing prices."""
    returns = prices.pct_change().dropna()
    return returns

# Calculate returns
returns = calculate_returns(stock_prices)

# Calculate expected returns and covariance matrix
expected_returns = returns.mean()
cov_matrix = returns.cov()

# Print expected returns and covariance matrix
print("Expected Returns:")
print(expected_returns)
print("\nCovariance Matrix:")
print(cov_matrix)

# Step 3: Efficient Frontier

# a. Monte Carlo Simulations

# Function to generate random portfolios
def generate_random_portfolios(expected_returns, cov_matrix, num_portfolios):
    """Generates random portfolios."""
    num_assets = len(expected_returns)
    portfolio_returns = []
    portfolio_volatility = []
    portfolio_weights = []

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Ensure weights sum up to 1
        portfolio_weights.append(weights)
        portfolio_returns.append(np.sum(expected_returns * weights))
        portfolio_volatility.append(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

    return portfolio_returns, portfolio_volatility, portfolio_weights

# Define parameters for Monte Carlo simulations
num_portfolios = 10000

# Generate random portfolios
portfolio_returns_mc, portfolio_volatility_mc, portfolio_weights_mc = generate_random_portfolios(expected_returns, cov_matrix, num_portfolios)

# b. Markowitz Optimization

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(weights, expected_returns, cov_matrix):
    """Calculates portfolio metrics (return, volatility)."""
    portfolio_return = np.sum(expected_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Function to optimize portfolio weights using Markowitz optimization
def optimize_portfolio(expected_returns, cov_matrix):
    """Optimizes portfolio weights using Markowitz optimization."""
    num_assets = len(expected_returns)
    initial_weights = np.ones(num_assets) / num_assets  # Equal weights as initial guess
    bounds = ((0, 1),) * num_assets  # Bounds for weights (0 <= weight <= 1)

    # Define objective function (minimize negative Sharpe ratio)
    def objective_function(weights, expected_returns, cov_matrix):
        portfolio_return, portfolio_volatility = calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
        return -portfolio_return / portfolio_volatility

    # Perform optimization
    result = minimize(objective_function, initial_weights, args=(expected_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},))

    return result.x

# Optimize portfolio weights using Markowitz optimization
optimal_weights_markowitz = optimize_portfolio(expected_returns, cov_matrix)

# Print optimal weights
print("Optimal Weights (Markowitz Optimization):")
print(optimal_weights_markowitz)

# Step 4: Backtesting

# a. Training

# Function to rebalance portfolio on the first Friday of each month
def rebalance_portfolio(expected_returns, cov_matrix, start_date, end_date):
    """Rebalances portfolio on the first Friday of each month."""
    portfolio_weights = []
    for date_str in expected_returns.index:
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')  # Convert date string to datetime object
            if date.weekday() == 4 and start_date <= date <= end_date:  # Check if it's Friday and within the specified range
                # Perform optimization for this month
                optimal_weights = optimize_portfolio(expected_returns.loc[date_str], cov_matrix)
                portfolio_weights.append((date, optimal_weights))
        except ValueError:
            continue  # Skip non-date entries

    return portfolio_weights

# Rebalance portfolio on the first Friday of each month from January 2020 to December 2023
portfolio_weights_training = rebalance_portfolio(expected_returns, cov_matrix, datetime(2020, 1, 1), datetime(2023, 12, 31))

# b. Testing

# Function to backtest the portfolio
def backtest_portfolio(portfolio_weights, returns, tickers, start_date):
    """Backtests the portfolio."""
    portfolio_values = []
    for date, weights in portfolio_weights:
        portfolio_value = np.sum(returns.loc[date:, tickers] * weights, axis=1).cumsum()
        portfolio_values.append((date, portfolio_value))

    return portfolio_values

# Define start date for testing
start_date_testing = datetime(2024, 1, 1)

# Backtest the portfolio from January 2024 to the present
portfolio_values_testing = backtest_portfolio(portfolio_weights_training, returns, tickers, start_date_testing)

# Calculate cumulative returns for the equal-weighted portfolio
equal_weights = np.ones(len(tickers)) / len(tickers)
portfolio_equal_weighted = np.sum(returns.loc[start_date_testing:] * equal_weights, axis=1).cumsum()

# Plot the performance of the portfolios
plt.figure(figsize=(12, 6))
for date, portfolio_value in portfolio_values_testing:
    plt.plot(portfolio_value, label=date.strftime('%Y-%m-%d'))
plt.plot(portfolio_equal_weighted, color='black', linestyle='--', label='Equal-weighted Portfolio')
plt.title('Portfolio Performance (Testing)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.show()