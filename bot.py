import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import numpy as np

def run_trading_strategy(symbol='BTC-USD', short_period=12, long_period=26, rsi_period=14):
    def fetch_historical_data(symbol, start_date, end_date, interval):
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        return data

    def calculate_macd(data, short_period, long_period):
        data['ema_short'] = data['Close'].ewm(span=short_period, adjust=False).mean()
        data['ema_long'] = data['Close'].ewm(span=long_period, adjust=False).mean()
        data['macd'] = data['ema_short'] - data['ema_long']
        data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()

    def calculate_rsi(data, rsi_period):
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))

    def load_and_predict_sarimax(data, start_date, end_date):
        model = SARIMAX(data['Close'], order=(1, 1, 3), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        prediction = model_fit.get_forecast(start=start_date, end=end_date)
        return prediction.predicted_mean

    def trading_strategy(data, short_period, long_period, rsi_period):
        current_date = data.index[-1]
        current_data = data.iloc[-1]

        calculate_macd(data, short_period, long_period)
        calculate_rsi(data, rsi_period)

        if current_data['macd'] > 0 and current_data['rsi'] < 70:
            sarimax_prediction = load_and_predict_sarimax(data, current_date, current_date)
            if sarimax_prediction > current_data['Close']:
                return 'Buy'
        elif current_data['macd'] < 0 and current_data['rsi'] > 30:
            sarimax_prediction = load_and_predict_sarimax(data, current_date, current_date)
            if sarimax_prediction < current_data['Close']:
                return 'Sell'
        return 'Hold'

    # Define start and end dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(days=30)).strftime('%Y-%m-%d')

    # Fetch historical data
    historical_data = fetch_historical_data(symbol, start_date, end_date, '1d')

    # Apply trading strategy with custom RSI and MACD functions
    signal = trading_strategy(historical_data, short_period, long_period, rsi_period)

    return signal

