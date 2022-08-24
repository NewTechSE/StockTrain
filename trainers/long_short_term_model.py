import logging
import os

import keras.models
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from overrides import overrides
from sklearn.preprocessing import MinMaxScaler

from trainers.model import Model

logging.basicConfig(level=logging.INFO)


class LongShortTermModel(Model):
    scaler = MinMaxScaler(feature_range=(0, 1))
    steps_size = 60

    @overrides()
    async def predict(self, model_file_name: str, previous_data: list) -> list:
        try:
            if len(previous_data) < self.steps_size:
                raise Exception(f"Not enough data to predict. Required {self.steps_size} but got {len(previous_data)}")

            x_test = []
            for i in range(self.steps_size, len(previous_data)):
                x_test.append(previous_data[i - self.steps_size: i, 0])

            x_test = np.array(x_test)
            # make numpy array as 3D , adding num of indicator
            x_test = np.reshape(x_test, newshape=(x_test.shape[0], x_test.shape[1], 1))

            # Load the model
            model = keras.models.load_model(model_file_name)
            prediction = model.predict(x_test)

            return self.scaler.inverse_transform(prediction)
        except Exception as e:
            logging.error(e)
            raise e

    @overrides
    async def train(self, csv_file_path: str) -> str:
        try:
            df = pd.read_csv(csv_file_path)
            df.replace(',', '', regex=True, inplace=True)

            logging.info(f"Read from {csv_file_path}, {len(df)} rows")
            logging.info(df.head())

            training_set = df.iloc[:, 4:5].values
            logging.info(f"Training set shape: {training_set.shape}")

            # Feature scaling
            logging.info(f"Scaling data")

            training_set_scaled = self.scaler.fit_transform(training_set)

            x_train = []
            y_train = []

            logging.info(f"Creating x y train data")

            for i in range(self.steps_size, len(training_set_scaled)):
                x_train.append(training_set_scaled[i - self.steps_size: i, 0])
                y_train.append(training_set_scaled[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)

            x_train = np.reshape(x_train, newshape=(x_train.shape[0], x_train.shape[1], 1))

            logging.info(f"Building model")

            model = Sequential()

            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(rate=0.2))
            # add 2nd lstm layer
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(rate=0.2))
            # add 3rd lstm layer
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(rate=0.2))
            # add 4th lstm layer
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(rate=0.2))

            model.add(Dense(units=1))

            model.compile(optimizer='adam', loss='mean_squared_error')

            logging.info(f"Fitting model")

            model.fit(x=x_train, y=y_train, batch_size=32, epochs=20)

            model_file_name = f'../models/lstm/{os.path.basename(csv_file_path).split(".")[0]}_close.h5'

            model.save(model_file_name)

            logging.info(f"Saved model to {model_file_name}")

            return model_file_name
        except Exception as e:
            logging.error(e)
            raise e
