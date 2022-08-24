import asyncio
import logging
import os

import numpy as np

from trainers.long_short_term_model import LongShortTermModel
from trainers.simple_rnn_model import SimpleRNNModel
from trainers.xg_boost_model import XGBoostModel

import yfinance as yf

logging.basicConfig(level=logging.INFO)


async def main():
    lstm = LongShortTermModel()
    rnn = SimpleRNNModel()
    xg = XGBoostModel()

    data_dir = "../data/stocks"
    tasks = []
    for filename in os.listdir(data_dir):
        tasks.append(asyncio.create_task(lstm.train(os.path.join(data_dir, filename))))
        tasks.append(asyncio.create_task(rnn.train(os.path.join(data_dir, filename))))
        tasks.append(asyncio.create_task(xg.train(os.path.join(data_dir, filename))))

    logging.info(f"Tasks: {len(tasks)}")

    await asyncio.gather(*tasks)


# asyncio.run(main())

stock_data = yf.download('GOOGL', interval='60m', period='1mo')
close_prices = stock_data[['Close', 'High', 'Low']].values[-1:]

model = XGBoostModel()
predict = model.predict("../models/xgboost/GOOGL_1y_60m.csv_close.json", close_prices)

logging.info(f"Predict: {predict}")
