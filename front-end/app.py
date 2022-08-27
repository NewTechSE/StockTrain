import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from utils import get_period_from_interval
import requests
from services.stock_service import schedule_download_stock
from configs import HOST_API

from datetime import datetime

pd.options.mode.chained_assignment = None

st.title("Stock Prediction!")
with st.sidebar:
    modelType = st.selectbox(
        "Model",  ["LSTM", "RNN", "SGBoost"]
    )

    stockSymbol = st.selectbox(
        "Stock Symbol",  ["ADBE", "GOOGL", "MSFT"]
    )

    timeFrame = st.selectbox(
        "Time Frame",  ["1m", "60m", "1d"]
    )
    nCandles = st.select_slider(
        "Number Candles", options=list(range(60, 1000)))

df = yf.download(stockSymbol, period=get_period_from_interval(
    timeFrame), interval=timeFrame)
df = df.reset_index()
# df['Datetime'] = df['Datetime'].apply(lambda e: datetime.fromisoformat((e).replace('-4:00','')))
# scheduler = schedule_download_stock(stockSymbol, timeFrame, df)

placeholder = st.empty()

chart = st.line_chart(df.Close)

try:
    while True:
        df = df[-nCandles:]
        
        with placeholder.container():
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.1, row_width=[0.2, 0.7])

            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))

            WINDOW = 30
            df['sma'] = df['Close'].rolling(WINDOW).mean()
            df['std'] = df['Close'].rolling(WINDOW).std(ddof=0)

            fig.add_trace(go.Scatter(x=df.index,
                                     y=df['sma'] + (df['std'] * 2),
                                     line_color='gray',
                                     line={'dash': 'dash'},
                                     name='upper band',
                                     opacity=0.5),
                          row=1, col=1)

            fig.update_layout(title=stockSymbol,
                              yaxis_title="Price (USD)",
                              width=1200,
                              height=600)

            # chart = st.plotly_chart(fig)
            st.write(fig)

        response = requests.post(
            f"{HOST_API}/stock-train?method=lstm&symbol=adbe&interval=1m")
        predictions = response.json()['data']
        
        df.reset_index(inplace=True, drop=True)
        
        for p in predictions[0:1]:
            dfTemp = pd.DataFrame({
                'Datetime': [p['time']+'-4:00'],
                'Close': [p['value']],
                'Open': [df['Close'][-1:]],
                'High': [p['value']],
                'Low': [p['value']],
            })
            df = pd.concat([df, dfTemp])
            chart.add_rows(dfTemp)
       
        print(len(df))
        
        time.sleep(60)
except (KeyboardInterrupt, SystemExit):
    # Not strictly necessary if daemonic mode is enabled but should be done if possible
    # scheduler.shutdown()
    pass
