# Import libraries
import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
# import prediction 
from statsmodels.tsa.arima.model import ARIMA


# Title
app_name = "Stock Market Forecasting"
st.title(app_name)

# Subheader
st.subheader("Stock Market Forecasting App")
# Add an image
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

# Take user input
st.sidebar.header("Select the parameter below to get stock market forecasting")
start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

# Add ticker input
ticker_list = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'META']
ticker = st.sidebar.selectbox("Ticker", ticker_list)

# Fetch data from user input using yfinance library
data = yf.download(ticker, start_date, end_date)


# add date as a column to the dataframe
data.insert(0, 'Date', data.index, True)
data.reset_index(drop=True, inplace=True)
st.write('Data from', start_date, 'to', end_date)

# Display the fetched data
st.dataframe(data)

# plot the data
st.header('Data visulaization')
st.subheader('Closing Price vs Time')

fig = px.line(data, x='Date', y=data.columns , title='Closing Price vs Time' , width=1000, height=500)
st.plotly_chart(fig)


column = st.selectbox('Select the column to be used for forcasting',data.columns[1:])

# subseting the data
data = data[['Date', column]]
st.write('Selected data')
st.write(data)



# ADF test check stationarity
st.header("Is data stationary")
st.write('**Note:** If p-values is less than 0.05, data is stationary')
st.write(adfuller(data[column])[1] < 0.5)


# decompose data
st.header("Decomposing data")
decomposition = seasonal_decompose(data[column] , model='additive', period=12)
st.write(decomposition.plot())


# same plots in plotly

st.write('## Ploting the decompision in plotly')
st.plotly_chart(px.line(x=data['Date'], y=decomposition.observed, title='Observed',width=1000, height=400,labels={'x':'Date','y':'Price'}))
st.plotly_chart(px.line(x=data['Date'], y=decomposition.trend, title='Trend',width=1000, height=400,labels={'x':'Date','y':'Price'}))
st.plotly_chart(px.line(x=data['Date'], y=decomposition.seasonal, title='Seasonal',width=1000, height=400,labels={'x':'Date','y':'Price'}))
st.plotly_chart(px.line(x=data['Date'], y=decomposition.resid, title='Residual',width=1000, height=400,labels={'x':'Date','y':'Price'}))

# user input parameters 
p = st.slider('p', 0, 5, 0)
d = st.slider('d', 0, 5, 1)
q = st.slider('q', 0, 5, 2)
seasonal_order = st.number_input("select the value of seasonal p  ",0,24,12)

model = sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

# train the model
st.header("Model summary")
st.write(model.summary())
st.write("---")


# predict the values
forcast_period = st.number_input("select the value of forcast period ",value = 10)

# predict the futute values
predictions = model.get_prediction(start = len(data), end = len(data) + forcast_period-1)
predictions = predictions.predicted_mean
st.write(predictions)



# predictions.index = pd.date_range(start = end_date, periods = len(predictions), freq = 'D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index , True)
predictions.reset_index(drop=True, inplace=True)
st.write('## Predicted')
st.write(predictions)
st.write("## Actual")
st.write(data)
# Adjusted forecast period input

# lest plot the data
fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual' , line = dict(color='blue')))
fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['predicted_mean'], mode='lines', name='Predicted' , line = dict(color='red')))
fig.update_layout( title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1200, height=400)
st.plotly_chart(fig)

show_plots = False


import streamlit as st
import plotly.express as px

# Assuming 'data' and 'predictions' DataFrames are defined somewhere above this code

# Initialize session state for toggle
if 'show_plots' not in st.session_state:
    st.session_state.show_plots = False

# Button to toggle plots
if st.button('Show Separate Plots'):
    st.session_state.show_plots = not st.session_state.show_plots

# Display plots if toggled to True
if st.session_state.show_plots:
    st.write(px.line(x=data['Date'], y=data[column], title='Actual', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}))
    st.write(px.line(x=predictions['Date'], y=predictions['predicted_mean'], title='Predicted', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}))
