# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xQLjlblrwsSYzT3asmfjcNxqb7wcOpb_
"""

import pandas as pd
import yfinance as yf
from datetime import datetime

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    data['ema12'] = data['Close'].ewm(span=short_period, adjust=False).mean()
    data['ema26'] = data['Close'].ewm(span=long_period, adjust=False).mean()
    data['macd'] = data['ema12'] - data['ema26']
    data['signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()

def calculate_rsi(data, rsi_period=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

def get_current_signal(data, rsi_oversold=30, rsi_overbought=70):
    latest_data = data.iloc[-1]
    if latest_data['macd'] > latest_data['signal'] and latest_data['rsi'] > rsi_oversold:
        return 'Buy', latest_data['Close']
    elif latest_data['macd'] < latest_data['signal'] or latest_data['rsi'] > rsi_overbought:
        return 'Sell', latest_data['Close']
    else:
        return 'Hold', latest_data['Close']

def fetch_historical_data(symbol, start_date, end_date, interval):
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    return data

def get_latest_signal_and_price(symbol='BTC-USD', short_period=12, long_period=26, signal_period=9, rsi_period=14,
                                 rsi_oversold=30, rsi_overbought=70):
    # Calculate the end date as the current date
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Calculate the start date based on your desired timeframe
    start_date = (datetime.now() - pd.DateOffset(days=30)).strftime('%Y-%m-%d')

    interval = '1d'  # Timeframe for historical data (e.g., 1 day)

    data = fetch_historical_data(symbol, start_date, end_date, interval)
    calculate_macd(data, short_period, long_period, signal_period)
    calculate_rsi(data, rsi_period)
    current_signal, current_price = get_current_signal(data, rsi_oversold, rsi_overbought)

    return current_signal, current_price

if __name__ == '__main__':
    signal, price = get_latest_signal_and_price()
    print(f"Latest Preferred Option: {signal} at Price: {price:.2f}")
