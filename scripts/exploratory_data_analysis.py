# src/exploratory_data_analysis.py
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import mplfinance as mpf

# Configure logging
if not os.path.exists('../logs'):
    os.makedirs('../logs')

logging.basicConfig(
    filename='../logs/exploratory_data_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ExploratoryDataAnalysis:
    def __init__(self, data):
        self.data = data
        sns.set_palette('viridis')  # Set the color palette to viridis
        sns.set_style("whitegrid")  # Set the background style

    def visualize_closing_price(self):
        """Visualize the closing price over time."""
        for ticker, df in self.data.items():
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=df.index, y=df['Close'].squeeze(), label='Close Price', linewidth=2.5)
            plt.title(f'{ticker} Closing Price Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def visualize_closing_price_with_fluctuations(self):
        """Visualize the closing price over time with markers for fluctuations."""
        for ticker, df in self.data.items():
            df['Daily Return'] = df['Close'].pct_change()  # Ensure daily returns are calculated
            df = df.dropna()  # Drop NaN values to avoid indexing issues
            
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=df.index, y=df['Close'].squeeze(), label='Close Price', linewidth=2.5)
            
            # Mark significant fluctuations
            high_fluctuation_days = df['Daily Return'].abs() > 0.05  # Example threshold of 5%
            plt.scatter(df.index[high_fluctuation_days], df['Close'][high_fluctuation_days],
                        color='red', s=50, label='Fluctuation', edgecolor='black', zorder=5)

            plt.title(f'{ticker} Closing Price with Fluctuations', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def calculate_daily_returns(self):
        """Calculate and plot daily percentage change."""
        for ticker, df in self.data.items():
            df['Daily Return'] = df['Close'].pct_change()
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=df.index, y=df['Daily Return'].squeeze(), label='Daily Returns', color='orange', linewidth=2.5)
            plt.title(f'{ticker} Daily Returns', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Daily Return', fontsize=14)
            plt.axhline(0, linestyle='--', color='red', lw=2)
            plt.xticks(rotation=45)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def analyze_volatility(self):
        """Analyze volatility using rolling means and standard deviations."""
        for ticker, df in self.data.items():
            df['Rolling Mean'] = df['Close'].rolling(window=20).mean()
            df['Rolling Std'] = df['Close'].rolling(window=20).std()
            
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=df.index, y=df['Close'].squeeze(), label='Close Price', color='blue', linewidth=2.5)
            sns.lineplot(x=df.index, y=df['Rolling Mean'].squeeze(), label='20-Day Rolling Mean', color='orange', linewidth=2.5)
            plt.fill_between(df.index, df['Rolling Mean'] - df['Rolling Std'], 
                             df['Rolling Mean'] + df['Rolling Std'], color='gray', alpha=0.3)
            plt.title(f'{ticker} Price with Rolling Mean and Std Dev', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def visualize_bollinger_bands(self):    
        """Visualize Bollinger Bands to show price fluctuations."""
        for ticker, df in self.data.items():
            # Calculate the rolling mean and standard deviation
            df['Rolling Mean'] = df['Close'].rolling(window=20).mean()
            df['Rolling Std'] = df['Close'].rolling(window=20).std()

            # Ensure that we are getting Series for Upper Band and Lower Band
            df['Upper Band'] = df['Rolling Mean'] + (df['Rolling Std'] * 2)
            df['Lower Band'] = df['Rolling Mean'] - (df['Rolling Std'] * 2)

            plt.figure(figsize=(12, 6))
            sns.lineplot(x=df.index, y=df['Close'].squeeze(), label='Close Price', color='blue', linewidth=2.5)
            sns.lineplot(x=df.index, y=df['Rolling Mean'], label='20-Day Rolling Mean', color='orange', linewidth=2.5)
            plt.fill_between(df.index, df['Upper Band'], df['Lower Band'], color='gray', alpha=0.3, label='Bollinger Bands')

            plt.title(f'{ticker} Closing Price with Bollinger Bands', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def visualize_candlestick(self):
        """Visualize candlestick chart to show price fluctuations."""
        for ticker, df in self.data.items():
            df = df[['Open', 'High', 'Low', 'Close']]
            df.index.name = 'Date'
            
            mpf.plot(df, type='candle', volume=False, title=f'{ticker} Candlestick Chart',
                     style='charles', ylabel='Price', figsize=(12, 6))

    def detect_outliers(self):
        """Detect outliers based on daily returns."""
        for ticker, df in self.data.items():
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=df['Daily Return'].squeeze(), color='lightblue')
            plt.title(f'{ticker} Daily Returns Outlier Detection', fontsize=16, fontweight='bold')
            plt.xlabel('Daily Return', fontsize=14)
            plt.grid(True)
            plt.tight_layout()  # Adjust layout
            plt.show()
            
    def seasonal_decomposition(self):
        """Perform seasonal decomposition of the time series."""
        for ticker, df in self.data.items():
            result = seasonal_decompose(df['Close'], model='additive', period=30)
            result.plot()
            plt.title(f'Seasonal Decomposition of {ticker}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()

    def calculate_var(self, confidence_level=0.95):
        """Calculate Value at Risk (VaR) for each stock."""
        for ticker, df in self.data.items():
            var = df['Daily Return'].quantile(1 - confidence_level)
            logging.info(f'The Value at Risk (VaR) for {ticker} at {confidence_level * 100}% confidence level is {var:.2%}')
            print(f'The Value at Risk (VaR) for {ticker} at {confidence_level * 100}% confidence level is {var:.2%}')

    def calculate_sharpe_ratio(self, risk_free_rate=0.01):
        """Calculate the Sharpe Ratio for each stock."""
        for ticker, df in self.data.items():
            mean_return = df['Daily Return'].mean()
            std_dev = df['Daily Return'].std()
            sharpe_ratio = (mean_return - risk_free_rate) / std_dev
            logging.info(f'The Sharpe Ratio for {ticker} is {sharpe_ratio:.2f}')
            print(f'The Sharpe Ratio for {ticker} is {sharpe_ratio:.2f}')

    def summarize_findings(self):
        """Summarize findings from the analysis."""
        for ticker, df in self.data.items():
            mean_return = df['Daily Return'].mean()
            var = df['Daily Return'].quantile(0.05)
            sharpe_ratio = (mean_return - 0.01) / df['Daily Return'].std()
            print(f'Summary for {ticker}:')
            print(f' - Mean Daily Return: {mean_return:.2%}')
            print(f' - Value at Risk (VaR) at 95% confidence: {var:.2%}')
            print(f' - Sharpe Ratio: {sharpe_ratio:.2f}')
            print('---')