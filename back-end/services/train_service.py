import asyncio
import logging
import os
from datetime import timedelta

from joblib import Parallel, delayed
import pandas as pd
import numpy as np

from trainers.model import Model
from trainers.long_short_term_model import LongShortTermModel
from trainers.simple_rnn_model import SimpleRNNModel
from trainers.xg_boost_model import XGBoostModel

logging.basicConfig(level=logging.INFO)


def train_parallel(trained_model: Model, n_threads: int):
    def train_sync(csv_filename):
        asyncio.run(trained_model.train(csv_filename))

    data_dir = "../data/stocks"
    csv_files = []
    for filename in os.listdir(data_dir):
        csv_files.append(os.path.join(data_dir, filename))

    Parallel(n_jobs=n_threads)(delayed(train_sync)(csv_file)
                               for csv_file in csv_files)


def predict_parallel(trained_model: Model, n_threads: int):
    data_dir = "../data/stocks"
    csv_files = os.listdir(data_dir)

    def predict_job(csv_file: str):
        # Prediction for old data
        model_ext = 'h5' if trained_model.name == 'lstm' or trained_model.name == 'rnn' else 'json'
        model_file = f'../models/{trained_model.name}/{csv_file.replace(".csv", "")}_close.{model_ext}'

        logging.info(f'Predicting {csv_file} with {trained_model.name} model')

        stock_data = pd.read_csv(os.path.join(data_dir, csv_file))
        inputs = None
        if type(trained_model) is LongShortTermModel or type(trained_model) is SimpleRNNModel:
            inputs = stock_data['Close'].values
            inputs = np.reshape(inputs, (-1, 1))
        elif type(trained_model) is XGBoostModel:
            inputs = stock_data[['Close', 'High', 'Low']][1:]

        predictions = trained_model.predict(model_file, inputs)

        prediction_df = stock_data[["Date", "Open", "High", "Low", "Close"]][-len(predictions):].reset_index(
            drop=True).join(
            pd.DataFrame(predictions, columns=['Prediction']))

        predictions_csv = f'../data/predictions/{trained_model.name}/{csv_file.replace(".csv", "")}_close.csv'
        logging.info(f'Writing predictions to {predictions_csv}')

        prediction_df.to_csv(predictions_csv, index=False)

    Parallel(n_jobs=n_threads)(delayed(predict_job)(csv_file)
                               for csv_file in csv_files)


def prepare_input(method: str, stock_data: pd.DataFrame):
    if method == 'lstm' or method == 'rnn':
        inputs = stock_data['Close'].values[-70:]
        inputs = np.reshape(inputs, (-1, 1))
        return inputs
    else:
        inputs = stock_data[['Close', 'High', 'Low']][-10:]
        return inputs


def get_timedelta_by_interval(interval: str):
    if interval == '1m':
        return timedelta(minutes=1)
    elif interval == '60m':
        return timedelta(minutes=60)
    else:
        return timedelta(days=1)
