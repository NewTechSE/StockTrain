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
from apscheduler.schedulers.background import BackgroundScheduler
from configs import HOST_API

from datetime import datetime

pd.options.mode.chained_assignment = None

scheduler = BackgroundScheduler()

st.title("Stock Prediction!")
with st.sidebar:
    modelType = st.selectbox(
        "Model",  ["LSTM", "RNN", "XGBoost"],
    )

    stockSymbol = st.selectbox(
        "Stock Symbol",  ["ADBE", "GOOGL", "MSFT"]
    )

    timeFrame = st.selectbox(
        "Time Frame",  ["1m", "60m", "1d"]
    )
    nCandles = st.select_slider(
        "Number Candles", options=list(range(60, 1000)))



placeholder = st.empty()

response = requests.get(
        f"{HOST_API}/stock-train?method={modelType}&symbol={stockSymbol}&interval={timeFrame}")
data = pd.DataFrame(response.json())[-nCandles:]
data['Date'] = data['Date'].apply(lambda d: datetime.fromtimestamp(float(d/1000)))
data.set_index(data['Date'], inplace=True)
    
df = data
response = requests.post(
                f"{HOST_API}/stock-train?method={modelType}&symbol={stockSymbol}&interval={timeFrame}")
pre_df = pd.DataFrame(response.json()['data'])

def fetch_data():
    global df
    global pre_df
    response = requests.get(
        f"{HOST_API}/stock-train?method={modelType}&symbol={stockSymbol}&interval={timeFrame}")
    data = pd.DataFrame(response.json())[-nCandles:]
    data['Date'] = data['Date'].apply(lambda d: datetime.fromtimestamp(float(d/1000)))
    data.set_index(data['Date'], inplace=True)
    df = data
    
    response = requests.post(
                f"{HOST_API}/stock-train?method={modelType}&symbol={stockSymbol}&interval={timeFrame}")
    pre_df = pd.DataFrame(response.json()['data'])
    print(pre_df)

try:
    scheduler.add_job(fetch_data, 'interval', seconds=60)
    scheduler.start()
    
    if nCandles:
        while True:
            # df = fetch_data()
            
            # print(df.tail())
            with placeholder.container():
                chart = st.line_chart(df.Close)

                
                fig_col1, _ = st.columns(2)
                with fig_col1:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.1, row_width=[0.2, 0.7])

                    fig.add_trace(go.Candlestick(
                        x=df["Date"], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))

                    WINDOW = 30
                    df['sma'] = df['Close'].rolling(WINDOW).mean()
                    df['std'] = df['Close'].rolling(WINDOW).std(ddof=0)

                    fig.add_trace(go.Scatter(x=df["Date"],
                                            y=df['sma'] + (df['std'] * 2),
                                            line_color='gray',
                                            line={'dash': 'dash'},
                                            name='upper band',
                                            opacity=0.5),
                                row=1, col=1)
                    fig.add_trace(go.Scatter(x=pre_df["time"],
                                            y=pre_df["value"],
                                            line_color='red',
                                            line={'dash': 'dash'},
                                            name='predict',
                                            opacity=0.5),
                                row=1, col=1)
                    fig.update_layout(title=stockSymbol,
                                    yaxis_title="Price (USD)",
                                    width=1200,
                                    height=600)

                    # chart = st.plotly_chart(fig)
                    st.write(fig)

            # scheduler.add_job(fetch_Data, 'interval', seconds=2)

            

            # df.reset_index(inplace=True, drop=True)

            # for p in predictions[0:1]:
            #     dfTemp = pd.DataFrame({
            #         'Datetime': [p['time']+'-4:00'],
            #         'Close': [p['value']],
            #         'Open': [df['Close'][-1:]],
            #         'High': [p['value']],
            #         'Low': [p['value']],
            #     })
            #     df = pd.concat([df, dfTemp])
            #     chart.add_rows(dfTemp)

            # scheduler.start()
            # print('load')
            td = 60
            match timeFrame:
                case '60m':
                    td = 3600
                case '1d':
                    td = 3600*24

            time.sleep(1)

except (KeyboardInterrupt, SystemExit):
    # Not strictly necessary if daemonic mode is enabled but should be done if possible
    scheduler.shutdown()
    pass
