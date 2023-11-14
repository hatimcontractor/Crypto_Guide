import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime

def crypto_recommendation(symbol='BTC-USD', interval='1d', short_period=12, long_period=26, signal_period=9):
    # Function to fetch historical data from Yahoo Finance
    def fetch_historical_data(symbol, start_date, end_date, interval):
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        data.index.freq = 'D'  # Set the frequency explicitly
        return data

    # Function to calculate MACD and MACD histogram
    def calculate_macd(data, short_period, long_period, signal_period):
        data['ema_short'] = data['Close'].ewm(span=short_period, adjust=False).mean()
        data['ema_long'] = data['Close'].ewm(span=long_period, adjust=False).mean()
        data['macd'] = data['ema_short'] - data['ema_long']
        data['signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['signal']

    # Function to load and predict with SARIMAX model
    def load_and_predict_sarimax(data, exog, start_date, end_date):
        model = SARIMAX(data['Close'], exog=exog, order=(1, 1, 3), seasonal_order=(1, 1, 1, 12), freq='D')
        
        # Increase the number of iterations and try a different optimization algorithm
        model_fit = model.fit(disp=False, maxiter=1000, method='powell')
        
        # Use the mean of exog for forecasting
        exog_mean = exog.mean()
        prediction = model_fit.get_forecast(steps=len(pd.date_range(start=start_date, end=end_date)), exog=[exog_mean])
        return prediction.predicted_mean

    # Function for the trading strategy
    def trading_strategy(data, exog, short_period, long_period, signal_period):
        current_date = data.index[-1]
        current_data = data.iloc[-1]

        calculate_macd(data, short_period, long_period, signal_period)

        # Use MACD histogram as an exogenous variable
        exog = data['macd_histogram'].values.reshape(-1, 1)

        sarimax_prediction = load_and_predict_sarimax(data, exog, current_date, current_date)

        if sarimax_prediction.iloc[0] > current_data['Close']:
            return 'Buy'
        elif sarimax_prediction.iloc[0] < current_data['Close']:
            return 'Sell'
        else:
            return 'Hold'

    # Define start and end dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(days=365)).strftime('%Y-%m-%d')  # One year of historical data

    # Fetch historical data
    historical_data = fetch_historical_data(symbol, start_date, end_date, interval)

    # Apply trading strategy with SARIMAX and MACD histogram as exogenous variable
    signal = trading_strategy(historical_data, exog=None, short_period=short_period, long_period=long_period, signal_period=signal_period)

    return signal
