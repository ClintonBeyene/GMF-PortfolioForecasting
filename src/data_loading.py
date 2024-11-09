import yfinance as yf
import logging
import os
import pandas as pd

# Configure logging
if not os.path.exists('../logs'):
    os.makedirs('../logs')

logging.basicConfig(
    filename='../logs/data_loading.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StockDataLoader:
    def __init__(self, tickers, start_date, end_date, data_folder='../data'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data_folder = data_folder
        self.data = {}

    def load_data(self):
        """Load historical stock data for given tickers."""
        for ticker in self.tickers:
            try:
                logging.info(f'Loading data for {ticker} from {self.start_date} to {self.end_date}')
                df = yf.download(ticker, start=self.start_date, end=self.end_date)
                
                # Ensure DataFrame is not empty
                if df.empty:
                    logging.warning(f'No data found for {ticker}.')
                    continue

                # Store the DataFrame directly without resetting the index
                self.data[ticker] = df
                logging.info(f'Data for {ticker} loaded successfully.')
            except Exception as e:
                logging.error(f'Failed to load data for {ticker}. Error: {str(e)}')

    def flatten_columns(self, df):
        """Flatten the multi-index columns of the DataFrame."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[1] for col in df.columns]  # Use only the second level (the ticker symbol)
        return df

    def save_data_to_csv(self):
        """Save loaded stock data to CSV files in the specified folder."""
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        
        for ticker, df in self.data.items():
            # Flatten the columns
            df = self.flatten_columns(df)

            # Ensure the DataFrame has the correct column names
            df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

            file_path = os.path.join(self.data_folder, f"{ticker}.csv")
            
            # Log the DataFrame structure before saving
            logging.info(f'DataFrame for {ticker} before saving:')
            logging.info(df.head())

            # Save DataFrame to CSV with correct headers and index
            df.to_csv(file_path, index_label='Date')  # Save DataFrame to CSV with index labeled as 'Date'
            logging.info(f'Data for {ticker} saved to {file_path}')
            
            # Show DataFrame head, info, and shape
            print(f"\nData for {ticker}:")
            print("Head:")
            print(df.head())
            print("Info:")
            print(df.info())
            print("Shape:")
            print(df.shape)

    def clean_data(self):
        """Clean the loaded stock data by handling missing values and converting types."""
        for ticker in self.data:
            df = self.data[ticker]
            logging.info(f'Checking missing values for {ticker}')
            missing_values = df.isnull().sum()
            logging.info(f'Missing values for {ticker}: {missing_values}')
            
            # Ensure the DataFrame is not empty and contains the expected columns
            if not df.empty:
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else:
                        logging.warning(f'Column {col} not found in {ticker} DataFrame.')
            
                # Fill missing values
                df.fillna(method='ffill', inplace=True)  # Forward fill
                df.index.name = 'Date'
                self.data[ticker] = df
            else:
                logging.warning(f'DataFrame for {ticker} is empty.')
        
        logging.info('Data cleaning completed.')
        return self.data

    def save_cleaned_data_to_csv(self):
        """Save cleaned stock data to CSV files in the specified folder."""
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        
        for ticker, df in self.data.items():
            file_path = os.path.join(self.data_folder, f"cleaned_{ticker}.csv")
            
            # Log the cleaned DataFrame structure before saving
            logging.info(f'DataFrame for cleaned {ticker} before saving:')
            logging.info(df.head())

            # Save cleaned DataFrame to CSV with correct headers and index
            df.to_csv(file_path, index_label='Date')  # Save DataFrame to CSV with index labeled as 'Date'
            logging.info(f'Cleaned data for {ticker} saved to {file_path}')
            
            # Show cleaned DataFrame head, info, and shape
            print(f"\nCleaned Data for {ticker}:")
            print("Head:")
            print(df.head())
            print("Info:")
            print(df.info())
            print("Shape:")
            print(df.shape)