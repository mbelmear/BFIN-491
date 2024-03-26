# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
from datetime import datetime, timedelta
import yfinance as yf

# Function to fetch stock data using yfinance
def fetch_stock_data(symbol, start_date, end_date):
    """
    This function downloads historical stock data for a specified symbol, start date, and end date using the yfinance library.

    Args:
        symbol (str): The stock ticker symbol (e.g., AAPL)
        start_date (str): The start date for data retrieval in YYYY-MM-DD format
        end_date (str): The end date for data retrieval in YYYY-MM-DD format

    Returns:
        pandas.DataFrame: The downloaded stock data as a Pandas DataFrame
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

# Function to calculate technical indicators for the stock data
def calculate_technical_indicators(df):
    """
    This function calculates and adds several technical indicators to the provided DataFrame:

    - SMA_50: 50-day Simple Moving Average
    - SMA_200: 200-day Simple Moving Average
    - RSI: Relative Strength Index (14-day)
    - MACD: Moving Average Convergence Divergence
    - MACD_Signal: Signal line for MACD

    Args:
        df (pandas.DataFrame): The DataFrame containing stock data (Close prices)

    Returns:
        pandas.DataFrame: The DataFrame with additional technical indicator columns
    """
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    macd, signal, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    return df

# Function to create and display plots for the technical indicators
def plot_technical_indicators(df):
    """
    This function generates three separate plots for the following:

    1. Close Price vs. Moving Averages (50-day and 200-day)
    2. Relative Strength Index (RSI)
    3. Moving Average Convergence Divergence (MACD) with signal line

    Args:
        df (pandas.DataFrame): The DataFrame containing stock data and technical indicators
    """

    # Plot 1: Close Price vs. Moving Averages
    plt.figure(figsize=(14, 7))  # Set figure size
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['SMA_50'], label='50-day SMA')
    plt.plot(df['SMA_200'], label='200-day SMA')
    plt.title('Oracle Stock Price and Moving Averages')
    plt.legend()
    plt.show()

    # Plot 2: Relative Strength Index (RSI)
    plt.figure(figsize=(14, 7))
    plt.plot(df['RSI'], label='RSI')
    plt.axhline(70, linestyle='--', color='r', label='Overbought Threshold')  # Add horizontal line for overbought threshold
    plt.axhline(30, linestyle='--', color='r', label='Oversold Threshold')  # Add horizontal line for oversold threshold
    plt.title('Relative Strength Index (RSI)')
    plt.legend()
    plt.show()

    # Plot 3: Moving Average Convergence Divergence (MACD)
    plt.figure(figsize=(14, 7))
    plt.plot(df['MACD'], label='MACD')
    plt.plot(df['MACD_Signal'], label='MACD Signal')
    plt.title('MACD')
    plt.legend()
    plt.show()

def predict_trend(df):
    """
    This function uses a simple rule to predict the price trend based on the most recent closing positions of the 50-day and 200-day SMAs.

    - Upward trend: If the 50-day SMA is above the 200-day SMA at the most recent data point.
    - Downward trend: If the 50-day SMA is below the 200-day SMA at the most recent data point.

    **Note:** This is a basic strategy and should be used with caution. 

    Args:
        df (pandas.DataFrame): The DataFrame containing stock data and technical indicators

    Returns:
        str: The predicted trend ("Upward" or "Downward")
    """
    if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
        return "Predicted trend: Upward"
    else:
        return "Predicted trend: Downward"

def main():
    """
    This function serves as the main entry point for the script.

    1. Prompts the user to enter a stock symbol.
    2. Defines the start and end dates for data retrieval (one year).
    3. Fetches stock data using the fetch_stock_data function.
    4. Calculates technical indicators using the calculate_technical_indicators function.
    5. Creates plots for the technical indicators using the plot_technical_indicators function.
    6. Predicts the trend using the predict_trend function and prints the result.
    """
    symbol = "ORCL"
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    df = fetch_stock_data(symbol, start_date, end_date)
    df = calculate_technical_indicators(df)
    plot_technical_indicators(df)
    trend = predict_trend(df)
    print(trend)

if __name__ == "__main__":
    main()
