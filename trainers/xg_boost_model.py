import logging
import os
import time

import pandas as pd
from fastai.tabular.core import add_datepart
from overrides import overrides
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from ta.momentum import RSIIndicator, StochasticOscillator
from xgboost import XGBRegressor

from trainers.model import Model

logging.basicConfig(level=logging.INFO)

pd.options.mode.chained_assignment = None


def label_encode(share: pd.DataFrame):
    le = LabelEncoder()
    share['Is_month_end'] = le.fit_transform(share['Is_month_end'])
    share['Is_month_start'] = le.fit_transform(share['Is_month_start'])
    share['Is_quarter_end'] = le.fit_transform(share['Is_quarter_end'])
    share['Is_quarter_start'] = le.fit_transform(share['Is_quarter_start'])
    share['Is_year_end'] = le.fit_transform(share['Is_year_end'])
    share['Is_year_start'] = le.fit_transform(share['Is_year_start'])
    pass


def feature_calculator(share: pd.DataFrame):
    share['EMA_9'] = share['Close'].ewm(9).mean()  # exponential moving average of window 9
    share['SMA_5'] = share['Close'].rolling(5).mean()  # moving average of window 5
    share['SMA_10'] = share['Close'].rolling(10).mean()  # moving average of window 10
    share['SMA_15'] = share['Close'].rolling(15).mean()  # moving average of window 15
    share['SMA_20'] = share['Close'].rolling(20).mean()  # moving average of window 20
    share['SMA_25'] = share['Close'].rolling(25).mean()  # moving average of window 25
    share['SMA_30'] = share['Close'].rolling(30).mean()  # moving average of window 30
    ema_12 = pd.Series(share['Close'].ewm(span=12, min_periods=12).mean())
    ema_26 = pd.Series(share['Close'].ewm(span=26, min_periods=26).mean())
    share['MACD'] = pd.Series(ema_12 - ema_26)  # calculates Moving Average Convergence Divergence
    share['RSI'] = RSIIndicator(share['Close']).rsi()  # calculates Relative Strength Index
    share['Stochastic'] = StochasticOscillator(share['High'], share['Low'],
                                               share['Close']).stoch()  # Calculates Stochastic Oscillator
    pass


class XGBoostModel(Model):

    @overrides()
    async def train(self, csv_file_path: str) -> str:
        try:
            data = pd.read_csv(csv_file_path)

            logging.info(f'Reading from CSV')
            logging.info(data.head())

            data_new = data.drop(['Volume'], axis=1)
            data_new.reset_index(inplace=True)

            logging.info(f'Add date part, calculate feature, label encode')

            # add_datepart(data_new, 'Date', drop=False)
            # label_encode(data_new)
            feature_calculator(data_new)

            logging.info('Dropping rows with NaN values')

            data_new = data_new.iloc[33:]
            data_new.reset_index(drop=True, inplace=True)

            logging.info('Dropping unused columns')
            # data_new.drop(['Year', 'High', 'Low', 'Open', 'Date', 'index'], inplace=True, axis=1)
            data_new.drop(['High', 'Low', 'Open', 'Date', 'index'], inplace=True, axis=1)
            logging.info(data_new.head())

            # Shifting the features a row up
            data_new[
                [
                    'EMA_9',
                    'SMA_5',
                    'SMA_10',
                    'SMA_15',
                    'SMA_20',
                    'SMA_25',
                    'SMA_30',
                    'MACD',
                    'RSI',
                    'Stochastic']
            ] = data_new[
                [
                    'EMA_9',
                    'SMA_5',
                    'SMA_10',
                    'SMA_15',
                    'SMA_20',
                    'SMA_25',
                    'SMA_30',
                    'MACD',
                    'RSI',
                    'Stochastic']
            ].shift(-1)

            # Splitting the dataset into 70% training, 15% validation and 15% test
            # train test split indexes
            test_size = 0.15
            valid_size = 0.15

            test_split_idx = int(data_new.shape[0] * (1 - test_size))
            valid_split_idx = int(data_new.shape[0] * (1 - (valid_size + test_size)))

            # train test split tcs

            train = data_new.loc[:valid_split_idx]
            valid = data_new.loc[valid_split_idx + 1:test_split_idx]
            test = data_new.loc[test_split_idx + 1:]

            y_train = train['Close']
            x_train = train.drop(['Close'], 1)

            logging.info(f'x_train:\n{x_train}')
            logging.info(f'y_train:\n{y_train}')

            y_valid = valid['Close']
            x_valid = valid.drop(['Close'], 1)

            y_test = test['Close']
            x_test = test.drop(['Close'], 1)

            parameters = {
                'n_estimators': [500, 600],
                'learning_rate': [0.1],
                'max_depth': [8, 12, 15],
                'gamma': [0.005, 0.01, ],
                'random_state': [42],
                'min_child_weight': [4, 3],
                'subsample': [0.8, 1],
                'colsample_bytree': [1],
                'colsample_bylevel': [1]
            }
            kfold = KFold(3)
            eval_set = [(x_train, y_train), (x_valid, y_valid)]
            model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, tree_method="hist")
            clf = GridSearchCV(model, parameters, cv=kfold, scoring='neg_mean_absolute_error', verbose=0)

            start = time.time()
            clf.fit(x_train, y_train)
            end = time.time()

            logging.info(f'Time taken to run clf.fit(x_train, y_train): {end - start} seconds')

            logging.info(f'Best params: {clf.best_params_}')
            logging.info(f'Best validation score = {clf.best_score_}')

            model = XGBRegressor(**clf.best_params_, objective='reg:squarederror', n_jobs=-1)

            start = time.time()
            model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
            end = time.time()

            logging.info(f'Time taken to run model.fit(x_train, y_train, eval_set=eval_set): {end - start} seconds')

            y_pred = model.predict(x_test)
            mae = mean_absolute_error(y_test, y_pred)

            logging.info(f'Mean square abs error: {mae}')

            # Hand-tuning the hyper-parameters
            params = {'colsample_bylevel': 1,
                      'colsample_bytree': 0.6,
                      'gamma': 0.005,
                      'learning_rate': 0.07,
                      'max_depth': 10,
                      'min_child_weight': 1,
                      'n_estimators': 170,
                      'random_state': 42,
                      'subsample': 0.6}
            eval_set = [(x_train, y_train), (x_valid, y_valid)]
            xgb = XGBRegressor(**params, objective='reg:squarederror', n_jobs=-1)
            xgb.fit(x_train, y_train, eval_set=eval_set, verbose=False)
            y_pred = xgb.predict(x_test)
            mae = mean_absolute_error(y_test, y_pred)

            logging.info(f"After tuning, mean square abs: {mae}")

            model_file_name = f'../models/xgboost/{os.path.basename(csv_file_path).split(".")[0]}_close.json'
            xgb.save_model(model_file_name)

            logging.info(f"Saved model to {model_file_name}")

            return model_file_name
        except Exception as e:
            logging.error(e)
            raise e

    @overrides()
    def predict(self, model_file_name: str, previous_data: list) -> list:
        pass
