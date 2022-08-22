import logging
import os

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import SimpleRNN, Dropout, Dense
from overrides import overrides
from sklearn.preprocessing import MinMaxScaler

from trainers.model import Model

logging.basicConfig(level=logging.INFO)


class SimpleRNNModel(Model):
    scaler = MinMaxScaler(feature_range=(0, 1))
    steps_size = 60

    @overrides()
    async def train(self, csv_file_path: str) -> str:
        try:
            data = pd.read_csv(csv_file_path)

            logging.info(f'Reading data from {csv_file_path}')
            logging.info(data.head())
            logging.info(data.info())

            # Splitting Data as Train and Validation
            length_data = len(data)  # rows that data has
            split_ratio = 0.7  # %70 train + %30 validation
            length_train = round(length_data * split_ratio)
            length_validation = length_data - length_train

            logging.info(f"Data length: {length_data}")
            logging.info(f"Train data length: {length_train}")
            logging.info(f"Validation data length: {length_validation}")

            train_data = data[:length_train][["Date", "Close"]]
            train_data['Date'] = pd.to_datetime(train_data['Date'])  # converting to date time object

            validation_data = data[length_train:][["Date", "Close"]]
            validation_data['Date'] = pd.to_datetime(validation_data['Date'])  # converting to date time object

            # Creating Train Dataset from Train split
            dataset_train = train_data['Close'].values
            dataset_train = np.reshape(dataset_train, (-1, 1))

            logging.info(f'Train data shape: {dataset_train.shape}')

            dataset_train_scaled = self.scaler.fit_transform(dataset_train)

            # Creating X_train and y_train from Train data
            x_train = []
            y_train = []

            for i in range(self.steps_size, length_train):
                x_train.append(dataset_train_scaled[i - self.steps_size:i, 0])
                y_train.append(dataset_train_scaled[i, 0])

            # convert list to array
            x_train, y_train = np.array(x_train), np.array(y_train)

            logging.info(f"Shape of X_train before reshape: {x_train.shape}")
            logging.info(f"Shape of y_train before reshape: {y_train.shape}")

            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            y_train = np.reshape(y_train, (y_train.shape[0], 1))

            logging.info(f"Shape of X_train after reshape: {x_train.shape}")
            logging.info(f"Shape of y_train after reshape: {y_train.shape}")

            # Creating RNN model
            # initializing the RNN
            regressor = Sequential()

            # adding first RNN layer and dropout regularization
            regressor.add(
                SimpleRNN(units=50,
                          activation="tanh",
                          return_sequences=True,
                          input_shape=(x_train.shape[1], 1))
            )

            regressor.add(Dropout(0.2))

            # adding second RNN layer and dropout regularization

            regressor.add(
                SimpleRNN(units=50,
                          activation="tanh",
                          return_sequences=True)
            )

            regressor.add(Dropout(0.2))

            # adding third RNN layer and dropout regularization

            regressor.add(
                SimpleRNN(units=50,
                          activation="tanh",
                          return_sequences=True)
            )

            regressor.add(Dropout(0.2))

            # adding fourth RNN layer and dropout regularization

            regressor.add(SimpleRNN(units=50))

            regressor.add(Dropout(0.2))

            # adding the output layer
            regressor.add(Dense(units=1))

            # compiling RNN
            regressor.compile(
                optimizer="adam",
                loss="mean_squared_error",
                metrics=["accuracy"]
            )

            logging.info("Training RNN model")
            logging.info(regressor.summary())

            # fitting the RNN
            history = regressor.fit(x_train, y_train, epochs=50, batch_size=32)

            logging.info(f"Loss: {history.history['loss']}")

            model_file_name = f'../models/rnn/{os.path.basename(csv_file_path)}_close.h5'

            regressor.save(model_file_name)

            logging.info(f"Saved model to {model_file_name}")

            return model_file_name
        except Exception as e:
            logging.error(e)
            raise e

    @overrides()
    async def predict(self, model_file_name: str, previous_data: list) -> list:
        pass
