import unittest
from unittest.mock import patch
import pandas as pd
from src.data_loading import StockDataLoader  

class TestStockDataLoader(unittest.TestCase):

    @patch('yfinance.download')
    def test_load_data(self, mock_download):
        # Create a mock DataFrame to simulate downloaded data
        mock_data = {
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [104, 103, 102],
            'Adj Close': [104, 103, 102],
            'Volume': [1000, 1100, 1200]
        }
        mock_df = pd.DataFrame(mock_data, index=pd.date_range(start='2023-01-01', periods=3))
        mock_download.return_value = mock_df

        # Initialize the StockDataLoader
        tickers = ['AAPL']
        loader = StockDataLoader(tickers, '2023-01-01', '2023-01-03')
        
        # Load the data
        loader.load_data()

        # Check that the data is loaded correctly
        self.assertIn('AAPL', loader.data)
        self.assertEqual(loader.data['AAPL'].shape, (3, 6))  # Ensure it has 3 rows and 6 columns
        self.assertTrue((loader.data['AAPL'].index == mock_df.index).all())  # Check index matches
        self.assertTrue((loader.data['AAPL'] == mock_df).all().all())  # Check DataFrame contents match

if __name__ == '__main__':
    unittest.main()