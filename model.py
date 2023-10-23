# Importing necessary libraries
from datetime import datetime
import time
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from ratelimit import rate_limited,limits

FIFTEEN_MINUTES = 900


# Define a function to construct the download URL for cryptocurrency data

def construct_download_url(
	ticker,
	period1,
	period2,
	interval='monthly'
):
	"""
	:period1 & period2: 'yyyy-mm-dd'
	:interval: {daily; weekly, monthly}
	"""
	# Define a helper function to convert date to seconds since epoch
	def convert_to_seconds(period):
		datetime_value = datetime.strptime(period, '%Y-%m-%d')
		total_seconds = int(time.mktime(datetime_value.timetuple())) + 86400
		return total_seconds

	try:
		# Define interval options and their corresponding codes
		interval_reference = {'daily': '1d', 'weekly': '1wk', 'monthly': '1mo'}
		_interval = interval_reference.get(interval)

		# Check if the interval is valid
		if _interval is None:
			print('interval code is incorrect')
			return

		# Convert date periods to seconds
		p1 = convert_to_seconds(period1)
		p2 = convert_to_seconds(period2)

		# Construct the URL for data download
		url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={p1}&period2={p2}&interval={_interval}&filter=history'
		return url
	except Exception as e:
		print(e)
		return

#To get current date
current_date = datetime.now()
end_date = current_date.strftime('%Y-%m-%d')



# print(btc_df)

#Get preprocessed dataframe

# @rate_limited(1, 1)
#@limits(calls=1, period=5)
def get_preprocessed_df():
	
	df_btc = pd.read_csv(construct_download_url('BTC-USD', '2018-11-09', end_date, 'daily'))
	# df_btc.set_index('Date', inplace=True)

	df_eth = pd.read_csv(construct_download_url('ETH-USD', '2018-11-09', end_date, 'daily'))
	# df_eth.set_index('Date', inplace=True)

	df_ltc = pd.read_csv(construct_download_url('LTC-USD', '2018-11-09', end_date, 'daily'))
	# df_ltc.set_index('Date', inplace=True)

	df_date = df_btc[['Date']]

	df_btc.rename(columns={'Close': 'Bitcoin'}, inplace=True)
	df_eth.rename(columns={'Close': 'Ethereum'}, inplace=True)
	df_ltc.rename(columns={'Close': 'Litecoin'}, inplace=True)

	df_btc = df_btc[['Bitcoin']]
	df_eth = df_eth[['Ethereum']]
	df_ltc = df_ltc[['Litecoin']]

	merged_df = pd.concat([df_date, df_btc, df_eth, df_ltc], axis=1)

	return merged_df


#Price prediction model for BTC, ETH, LTC
def get_price_prediction_model(specific_date):

	merged_df = get_preprocessed_df()
	
	# Convert 'Date' column to datetime
	merged_df['Date'] = pd.to_datetime(merged_df['Date'])

	# Set 'Date' as the index
	merged_df.set_index('Date', inplace=True)

	# Train SARIMAX model for each coin
	coins = ['Bitcoin', 'Ethereum', 'Litecoin']
	models = {}

	for coin in coins:
		model = SARIMAX(merged_df[coin], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
		models[coin] = model.fit(disp=False)

	# Specific date for prediction
	# specific_date = '2023-08-16'

	# Create a DataFrame for the specific date
	specific_date_df = pd.DataFrame(index=[pd.to_datetime(specific_date)])

	# Make predictions for the specific date
	predictions = {}
	for coin in coins:
		model = models[coin]
		prediction = model.get_prediction(start=specific_date_df.index[0], end=specific_date_df.index[0])
		predicted_value = prediction.predicted_mean.iloc[0]
		predictions[coin] = predicted_value

	return predictions



# dictt = get_price_prediction_model('2023-08-16')

# print(dictt)