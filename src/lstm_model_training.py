import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockForecaster:
    """
    A class to handle stock price forecasting using LSTM models.

    Attributes:
        data_path (str): Path to the CSV file containing the stock data.
        ticker (str): Ticker symbol of the stock.
        train_size (float): Proportion of the data to be used for training.
        n_input (int): Number of time steps to look back for making predictions.
        n_features (int): Number of features in the input data.
        model (Sequential): Keras Sequential model for LSTM.
        scaler (MinMaxScaler): Scaler for normalizing the data.
        best_model_path (str): Path to save the best model.
    """

    def __init__(self, data_path, ticker, train_size=0.8, n_input=12, n_features=1):
        """
        Initializes the StockForecaster with the given parameters.

        Args:
            data_path (str): Path to the CSV file containing the stock data.
            ticker (str): Ticker symbol of the stock.
            train_size (float, optional): Proportion of the data to be used for training. Defaults to 0.8.
            n_input (int, optional): Number of time steps to look back for making predictions. Defaults to 12.
            n_features (int, optional): Number of features in the input data. Defaults to 1.
        """
        self.data_path = data_path
        self.ticker = ticker
        self.train_size = train_size
        self.n_input = n_input
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()
        self.best_model_path = f'../../models/best_models/best_model_{ticker}.h5'
        self.load_data()

    def load_data(self):
        """
        Loads the stock data from the CSV file and preprocesses the 'Date' column.
        """
        try:
            logging.info(f'Loading data for {self.ticker} from {self.data_path}')
            self.df = pd.read_csv(self.data_path)
            self.df = self.df[['Date', 'Adj Close']]
            self.df['Date'] = pd.to_datetime(self.df['Date'].apply(lambda x: x.split()[0]))
            self.df = self.df.set_index('Date')
            logging.info(f'Data loaded successfully for {self.ticker}')
        except Exception as e:
            logging.error(f'Error loading data for {self.ticker}: {e}')

    def preprocess_data(self):
        """
        Preprocesses the data by splitting it into training and testing sets and scaling the data.
        """
        try:
            logging.info(f'Preprocessing data for {self.ticker}')
            train_size = round(len(self.df) * self.train_size)
            self.train_data = self.df[:train_size]
            self.test_data = self.df[train_size:]
            
            self.scaler.fit(self.train_data)
            self.scaled_train = self.scaler.transform(self.train_data)
            self.scaled_test = self.scaler.transform(self.test_data)
            logging.info(f'Data preprocessing completed for {self.ticker}')
        except Exception as e:
            logging.error(f'Error preprocessing data for {self.ticker}: {e}')

    def create_timeseries_generator(self):
        """
        Creates timeseries generators for training and testing data.
        """
        try:
            logging.info(f'Creating timeseries generator for {self.ticker}')
            self.train_generator = TimeseriesGenerator(self.scaled_train, self.scaled_train, length=self.n_input, batch_size=1)
            self.test_generator = TimeseriesGenerator(self.scaled_test, self.scaled_test, length=self.n_input, batch_size=1)
            logging.info(f'Timeseries generator created for {self.ticker}')
        except Exception as e:
            logging.error(f'Error creating timeseries generator for {self.ticker}: {e}')

    def build_model(self):
        """
        Builds the LSTM model for stock price forecasting.
        """
        try:
            logging.info(f'Building model for {self.ticker}')
            self.model = Sequential()
            self.model.add(LSTM(100, activation='relu', input_shape=(self.n_input, self.n_features)))
            self.model.add(Dense(1))
            self.model.compile(optimizer='adam', loss='mse')
            logging.info(f'Model built for {self.ticker}')
        except Exception as e:
            logging.error(f'Error building model for {self.ticker}: {e}')

    def train_model(self, epochs=50):
        """
        Trains the LSTM model for the specified number of epochs.

        Args:
            epochs (int, optional): Number of epochs to train the model. Defaults to 50.
        """
        try:
            logging.info(f'Training model for {self.ticker}')
            self.predictions_over_time = []
            for epoch in range(epochs):
                logging.info(f'Epoch {epoch + 1}/{epochs}')
                self.model.fit(self.train_generator, epochs=1, verbose=0)
                predictions = self.model.predict(self.test_generator)
                self.predictions_over_time.append(predictions)
            logging.info(f'Model training completed for {self.ticker}')
            self.save_best_model()
        except Exception as e:
            logging.error(f'Error training model for {self.ticker}: {e}')

    def save_best_model(self):
        """
        Saves the best model to a file.
        """
        try:
            logging.info(f'Saving best model for {self.ticker}')
            self.model.save(self.best_model_path)
            logging.info(f'Best model saved for {self.ticker}')
        except Exception as e:
            logging.error(f'Error saving best model for {self.ticker}: {e}')

    def evaluate_model(self):
        """
        Evaluates the model on the test data and prints the metrics.

        Returns:
            tuple: RMSE, MAE, and MAPE of the model on the test data.
        """
        try:
            logging.info(f'Evaluating model for {self.ticker}')
            predictions = self.model.predict(self.test_generator)
            true_values = self.scaled_test[self.n_input:]
            
            rmse = np.sqrt(mean_squared_error(true_values, predictions))
            mae = mean_absolute_error(true_values, predictions)
            mape = mean_absolute_percentage_error(true_values, predictions)
            
            print('Test RMSE: %.3f' % rmse)
            print('MAE: %.3f' % mae) 
            print('MSE: %.3f' % mape)
    
            logging.info(f'RMSE: {rmse}, MAE: {mae}, MAPE: {mape}')
            return rmse, mae, mape
        except Exception as e:
            logging.error(f'Error evaluating model for {self.ticker}: {e}')
            

    def plot_predictions(self):
        """
        Plots the predictions and true values for the test data.
        """
        try:
            logging.info(f'Plotting predictions for {self.ticker}')
            predictions = self.model.predict(self.test_generator)
            true_values = self.scaled_test[self.n_input:]
            
            plt.figure(figsize=(12, 6))
            plt.plot(true_values, label='True Values')
            plt.plot(predictions, label='Predictions')
            plt.title(f'Test Predictions Over Time for {self.ticker}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.show()
        except Exception as e:
            logging.error(f'Error plotting predictions for {self.ticker}: {e}')

    def plot_evolution_test_predictions(self):
        """
        Plots Evolution of Test Predictions Over Time.
        """
        try:
            logging.info(f'Plotting Evolution of Test Predictions Over Time for {self.ticker}')
            # Flatten the predictions and true values for plotting
            predictions_flat = np.concatenate(self.predictions_over_time).flatten()
            true_values_flat = self.scaled_test[self.n_input:].flatten()

            # Create a time index for the test data
            test_dates = self.df.index[round(len(self.df) * self.train_size) + self.n_input:]

            # Ensure predictions_flat matches the length of true_values_flat
            predictions_flat = predictions_flat[:len(true_values_flat)]  # Adjust predictions if necessary

            # Plotting
            plt.figure(figsize=(14, 7))
            plt.plot(test_dates, true_values_flat, label='True Values', color='blue')
            plt.plot(test_dates[:len(predictions_flat)], predictions_flat, label='Predictions', color='orange')  # Adjusting to match lengths
            plt.title(f'Evolution of Test Predictions Over Time for {self.ticker}')
            plt.xlabel('Date')
            plt.ylabel('Scaled Adjusted Close Price')
            plt.legend()
            plt.show()
        except Exception as e:
            logging.error(f'Error plotting Evolution of Test Predictions Over Time for {self.ticker}: {e}')

    def plot_best_test_prediction(self):
        """
        Plot best test prediction
        """
        try:
            logging.info(f'Finding Best Test Prediction for {self.ticker}')
            # Flatten the predictions and true values for plotting
            predictions_flat = np.concatenate(self.predictions_over_time).flatten()
            true_values_flat = self.scaled_test[self.n_input:].flatten()

            # Ensure predictions_flat matches the length of true_values_flat
            predictions_flat = predictions_flat[:len(true_values_flat)]

            # Calculate the absolute errors
            errors = np.abs(true_values_flat - predictions_flat)

            # Find the indices of the smallest errors
            best_indices = np.argsort(errors)[:50]  # Change 50 to however many best predictions you want to visualize

            # Get the corresponding dates and values
            test_dates = self.df.index[round(len(self.df) * self.train_size) + self.n_input:]
            best_dates = test_dates[best_indices]
            best_true_values = true_values_flat[best_indices]
            best_predictions = predictions_flat[best_indices]

            # Plotting
            plt.figure(figsize=(14, 7))
            plt.scatter(best_dates, best_true_values, label='Best True Values', color='blue', alpha=0.6)
            plt.scatter(best_dates, best_predictions, label='Best Predictions', color='orange', alpha=0.6)
            plt.title(f'Best Test Predictions Over Time for {self.ticker}')
            plt.xlabel('Date')
            plt.ylabel('Scaled Adjusted Close Price')
            plt.legend()
            plt.show()
        except Exception as e:
            logging.error(f'Error plotting Best Test Predictions Over Time for {self.ticker}: {e}')

    def actual_vs_predicted_data(self):
        """
        Check actual vs predicted data  
        """
        try:
            logging.info(f'Actual vs Predicted Data for {self.ticker}')
            # Flatten the predictions and true values for plotting
            true_values = self.scaled_test[self.n_input:]
            predictions_flat = np.concatenate(self.predictions_over_time).flatten()
            true_values_flat = true_values.flatten()

            # Create a time index for the test data
            test_dates = self.df.index[round(len(self.df) * self.train_size) + self.n_input:]

            # Ensure predictions_flat matches the length of true_values_flat
            predictions_flat = predictions_flat[:len(true_values_flat)]  # Adjust predictions if necessary

            # Reverse the scaling for predictions
            predictions_scaled = predictions_flat.reshape(-1, 1)  # Reshape for the scaler
            predictions_actual = self.scaler.inverse_transform(predictions_scaled).flatten()  # Inverse transform

            # Reverse the scaling for true values if needed
            true_values_scaled = true_values_flat.reshape(-1, 1)  # Reshape for the scaler
            true_values_actual = self.scaler.inverse_transform(true_values_scaled).flatten()  # Inverse transform

            # Create a DataFrame to display actual and predicted values
            results_df = pd.DataFrame({
                'Date': test_dates[:len(predictions_actual)],  # Ensure the dates match the length of predictions
                'Actual': true_values_actual,
                'Predicted': predictions_actual
            })

            # Set the Date as the index
            results_df.set_index('Date', inplace=True)

            # Print the DataFrame
            print(results_df)
        except Exception as e:
            logging.error(f'Error creating predicted test data dataframe for {self.ticker}: {e}')

    def generate_forecasts(self, future_months=12):
        """
        Generates forecasts for the specified number of future months and saves the results to a CSV file.

        Args:
            future_months (int, optional): Number of future months to forecast. Defaults to 12.

        Returns:
            DataFrame: DataFrame containing the forecasted dates and values.
        """
        try:
            logging.info(f'Generating forecasts for {self.ticker}')
            last_train_date = self.df.index[-1]
            forecast_dates = pd.date_range(start=last_train_date, periods=future_months * 21 + 1, freq='B')[1:]  # 21 trading days per month
            
            last_train_data = self.scaled_train[-self.n_input:]
            forecasts = []
            
            for _ in range(future_months * 21):  # Forecasting for 21 days per month
                X = last_train_data.reshape((1, self.n_input, self.n_features))
                forecast = self.model.predict(X, verbose=0)
                forecasts.append(forecast[0, 0])
                last_train_data = np.append(last_train_data[1:], forecast, axis=0)
            
            forecasts = np.array(forecasts).reshape(-1, 1)
            forecasts_actual = self.scaler.inverse_transform(forecasts).flatten()
            
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecasts_actual
            })
            
            forecast_df.to_csv(f'../../models/best_models/forecasts_{self.ticker}.csv', index=False)
            logging.info(f'Forecasts generated and saved for {self.ticker}')
            return forecast_df
        except Exception as e:
            logging.error(f'Error generating forecasts for {self.ticker}: {e}')