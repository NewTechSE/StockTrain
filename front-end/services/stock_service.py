from threading import Timer

import pandas as pd
import streamlit as st
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from utils import get_period_from_interval


def schedule_download_stock(stockSymbol: str, interval: str, df: pd.DataFrame):
    seconds = 2
    if interval == '60m':
        seconds = 60 * 60
    elif interval == '1d':
        seconds = 60 * 60 * 24

    scheduler = BackgroundScheduler()

    scheduler.add_job(
        lambda: download_stock(stockSymbol, interval, df), 'interval', seconds=seconds)

    scheduler.start()

    return scheduler


def download_stock(stockSymbol: str, interval: str, df: pd.DataFrame):
    print(f"Downloading {stockSymbol} - {interval}")
    df = yf.download(stockSymbol, period=get_period_from_interval(
        interval), interval=interval)
    print(df.tail())
