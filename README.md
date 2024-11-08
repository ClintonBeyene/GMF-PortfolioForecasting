# GMF Investments Portfolio Forecasting

## Overview
This project uses advanced time series forecasting models to enhance portfolio management for Guide Me in Finance (GMF) Investments. By analyzing historical data for Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY), we aim to forecast market trends, optimize asset allocation, and manage risk. The project employs ARIMA, SARIMA, and LSTM models to provide data-driven investment recommendations.

## Project Goals
1. **Market Trend Forecasting**: Predict future market trends for TSLA, BND, and SPY.
2. **Portfolio Optimization**: Optimize portfolio allocation for balanced returns and risk management.
3. **Risk Management**: Adjust portfolio strategy based on predicted volatility.

## Data Sources
- **Tesla (TSLA)**: High-growth, high-volatility stock.
- **Vanguard Total Bond Market ETF (BND)**: Bond ETF for stability and income.
- **S&P 500 ETF (SPY)**: ETF representing U.S. market exposure.

Historical data includes Open, High, Low, Close, Volume, and Adjusted Close prices from January 1, 2015, to December 31, 2024.

## Technologies Used
- **Programming Language**: Python
- **Data Collection**: YFinance API
- **Time Series Models**: ARIMA, SARIMA, LSTM
- **Libraries**: pandas, numpy, statsmodels, tensorflow, scikit-learn

## Project Structure
- **/docs**: Documentation on methodology and models.
- **/notebooks**: Jupyter notebooks for data exploration, modeling, and optimization.
- **requirements.txt**: List of dependencies.
- **LICENSE.md**: Licensing information.

## Getting Started

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/ClintonBeyene/GMF-PortfolioForecasting.git
   cd GMF-PortfolioForecasting
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Data Collection
Use the YFinance library to collect data. Example for Tesla (TSLA):
```python
import yfinance as yf
tsla_data = yf.download("TSLA", start="2015-01-01", end="2024-12-31")
```

### Usage
1. **Data Exploration**: Analyze and preprocess data.
2. **Modeling**: Develop and test ARIMA, SARIMA, and LSTM models.
3. **Portfolio Optimization**: Adjust portfolio allocation based on forecasts.

## Methodology
- **Data Preprocessing**: Clean and analyze data for volatility and trends.
- **Model Development**: Use ARIMA, SARIMA, and LSTM models.
- **Evaluation Metrics**: RMSE and MAE.

## Contributing
Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Steps to Contribute:
1. Fork the repository.
2. Create a feature branch:
   ```sh
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```sh
   git commit -m 'Add new feature'
   ```
4. Push to the branch:
   ```sh
   git push origin feature-name
   ```
5. Create a Pull Request.


For any questions or feedback, contact us at [contact@clintonbeyene.com](mailto:clintonbeyene@gmail.com).
