from threading import Timer

import pandas as pd
import streamlit as st
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from utils import get_period_from_interval


@st.cache
def schedule_download_stock(stockSymbol: str, interval: str, df: pd.DataFrame):
    seconds = 60
    if interval == '60m':
        seconds = 60 * 60
    elif interval == '1d':
        seconds = 60 * 60 * 24

    scheduler = BackgroundScheduler()

    scheduler.add_job(
        lambda: _download_stock(stockSymbol, interval, df), 'interval', seconds=seconds)

    scheduler.start()

    return scheduler


def _download_stock(stockSymbol: str, interval: str, df: pd.DataFrame):
    print(f"Downloading {stockSymbol} - {interval}")
    df = yf.download(stockSymbol, period=get_period_from_interval(
        interval), interval=interval)
    print(df.tail())
