# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u9-6bqCNY_Imx0GwwxCULza872V4VlSx
"""

import streamlit as st

# from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
import os

import pandas as pd
from model import get_price_prediction_model
from bot import get_latest_signal_and_price

# load the Environment Variables.
load_dotenv()
st.set_page_config(page_title="Crypto Guide")

# Access environment variables
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

#api_key = os.environ.get(api_key)
#api_secret = os.environ.get(api_secret)

#  Sidebar contents
with st.sidebar:
    st.title('Cryptocurrency Price Prediction 📈')
    st.markdown('''
    ## About
    This web-app is an cryptocurrrency price prediction and trading bot.
    ''')

    st.write('Made by [Hatim Contractor](https://github.com/hatimcontractor)')

st.header("Cryptocurrency Price Prediction 📈")
st.divider()

def main():

    st.subheader("Predict future date price: ")
    with st.form(key='my_form'):
        date=st.date_input('Enter the date')
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        st.info("Predicted Price (USD):")
        # answer = get_price_prediction_model('2023-08-16')
        answer = get_price_prediction_model(date)
        st.write(answer)

        if answer:
            # Bot crypto trading
            st.subheader("Trading bot 🤖: ")
            with st.form(key='my_form_bot'):
                current_signal, current_price = get_latest_signal_and_price(symbol='BTC-USD', short_period=12, long_period=26, signal_period=9, rsi_period=14, rsi_oversold=30, rsi_overbought=70)
                st.write(f"Latest Preferred Option: {current_signal} at Price: {current_price:.2f}")
                okay_button = st.form_submit_button(label='Okay')
    # st.divider()

if __name__ == '__main__':
    main()