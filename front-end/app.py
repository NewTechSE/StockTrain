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
import sklearn.metrics  
from datetime import datetime
import pytz
import math

pd.options.mode.chained_assignment = None

scheduler = BackgroundScheduler()

st.title("Trading Dashboard Pro")
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
    field = st.selectbox(
        "Prediction", [ "Price of change (n=15)", "Price of change (n=30)", "Close",]
    ) 
    nCandles = st.select_slider(
        "Number Candles", options=list(range(60, 1001)))

# print(datetime.fromtimestamp(1661503620))

placeholder = st.empty()

df = None
pre_df = None

def fetch_data():
    global df, pre_df
    response = requests.get(
        f"{HOST_API}/stock-train?method={modelType}&symbol={stockSymbol}&interval={timeFrame}")
    data = pd.DataFrame(response.json())
    data = data[-nCandles:]
    data['Date'] = data['Date'].apply(lambda d: datetime.fromtimestamp(float(d), tz=pytz.timezone('Asia/Bangkok')).isoformat())
    # data.set_index(data['Date'], inplace=True)
    df = data

    response = requests.post(
                    f"{HOST_API}/stock-train?method={modelType}&symbol={stockSymbol}&interval={timeFrame}")
    data = pd.DataFrame(response.json())
    data = data[-nCandles-10:]
    data['Date'] = data['Date'].apply(lambda d: datetime.fromtimestamp(float(d), tz=pytz.timezone('Asia/Bangkok')).isoformat())
    # data.set_index(data['Date'], inplace=True)
    pre_df = data 

fetch_data()
try:
    scheduler.add_job(fetch_data, 'interval', seconds=60)
    scheduler.start()
    
    if nCandles:
        while True:
            with placeholder.container():
                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

                kpi1.metric(
                    label="Time ⏳",
                    value=datetime.now(tz=pytz.timezone('Asia/Bangkok')).strftime("%H:%M:%S"),
                )
                

                cur = df['Open'].loc[df.index[-1]]
                prev = df['Open'].loc[df.index[-2]]
                kpi2.metric(
                    label="Open ＄",
                    value=f"$ {round(cur, 4)} ",
                    delta=round(-prev + cur,9)
                )

                cur = df['Close'].loc[df.index[-1]]
                prev = df['Close'].loc[df.index[-2]]
                kpi3.metric(
                    label="Close ＄",
                    value=f"$ {round(cur, 4)} ",
                    delta=round(-prev + cur,9)
                )

                cur = df['High'].loc[df.index[-1]]
                prev = df['High'].loc[df.index[-2]]
                kpi4.metric(
                    label="High ＄",
                    value=f"$ {round(cur, 4)} ",
                    delta=round(-prev + cur,9)
                )

                cur = df['Low'].loc[df.index[-1]]
                prev = df['Low'].loc[df.index[-2]]
                kpi5.metric(
                    label="Low ＄",
                    value=f"$ {round(cur, 4)} ",
                    delta=round(-prev + cur,9)
                )



                fig_col1, _ = st.columns(2)
                with fig_col1:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.1, row_width=[0.2, 0.7])

                    fig.add_trace(go.Candlestick(
                        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], text=df['Date']))

                    WINDOW = 30
                    df['sma'] = df['Close'].rolling(WINDOW).mean()
                    df['std'] = df['Close'].rolling(WINDOW).std(ddof=0)

                    fig.add_trace(go.Scatter(x=df.index,
                                            y=df['sma'] + (df['std'] * 2),
                                            line_color='gray',
                                            line={'dash': 'dash'},
                                            name='Upper band',
                                            opacity=0.5),
                                row=1, col=1)

                    fig.add_trace(go.Scatter(x=df.index,
                                            y=df['sma'] - (df['std'] * 2),
                                            line_color='gray',
                                            line={'dash': 'dash'},
                                            name='Lower band',
                                            opacity=0.5),
                                row=1, col=1)

                    fig.add_trace(go.Scatter(x=pre_df.index,
                                            y=pre_df["Prediction"],
                                            line_color='red',
                                            line={'dash': 'dash'},
                                            name='predict',
                                            opacity=0.5),
                                row=1, col=1)
                    fig.update_layout(title=stockSymbol,
                                    yaxis_title="Price (USD)",
                                    width=1200,
                                    height=600)

                    st.write(fig)

                rmse = 0
                st.title(field)
                kpi1, kpi2 = st.columns(2) 
                if field == "Close":
                    chart = st.line_chart(pd.concat([df.Close, pre_df.Prediction[:len(df.Close)]], join='outer', axis=1))

                    mse = sklearn.metrics.mean_squared_error(df.Close, pre_df.Prediction[:len(df)])  
                    rmse = math.sqrt(mse)  


                    kpi1.metric(
                        label="RSME",
                        value=round(rmse, 9)
                    )

                else:
                    n = 30
                    if field == "Price of change (n=15)":
                        n = 15
                    poc = df["Close"]- df["Close"].shift(n) 
                    pre_poc = pre_df[:len(df)]["Prediction"] - pre_df[:len(df)]["Prediction"].shift(n)

                    diff = pd.concat([poc, pre_poc], join='outer', axis=1)
                    diff['Date'] = df['Date']
                    diff.set_index('Date', inplace=True)
                    
                    

                    rsme_df = diff
                    rsme_df.dropna(inplace=True)

                    mse = sklearn.metrics.mean_squared_error(rsme_df.Close, rsme_df.Prediction)  
                    rmse = math.sqrt(mse)  

                    kpi1.metric(
                        label="RSME",
                        value=round(rmse, 9)
                    )

                    chart = st.line_chart(diff)

                st.title("Data Table")        
                st.dataframe(df.dropna())
            time.sleep(1)
       
except (KeyboardInterrupt, SystemExit):
    # Not strictly necessary if daemonic mode is enabled but should be done if possible
    scheduler.shutdown()
    pass
