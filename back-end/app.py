import asyncio
import os.path
import subprocess

import numpy as np
import yfinance
from flask import Flask, jsonify
from flask_restx import Resource, Api, reqparse, abort
from flask_restx._http import HTTPStatus
from flask_restx.reqparse import ParseResult

from services.train_service import train_parallel
from trainers.long_short_term_model import LongShortTermModel
from trainers.simple_rnn_model import SimpleRNNModel
from trainers.xg_boost_model import XGBoostModel
from services.stock_service import download_stock_data_by_interval, download_parallel

app = Flask(__name__)
api = Api(app, prefix='/api')

parser = reqparse.RequestParser()

parser.add_argument('method', type=str, location='args', required=True, trim=True, help='lstm | rnn | xgboost')
parser.add_argument('symbol', type=str, location='args', required=True, trim=True, help='adbe | googl | msft')
parser.add_argument('interval', type=str, location='args', required=True, trim=True, help='1m | 60m | 1d')


# noinspection PyMethodMayBeStatic
@api.route('/stock-train')
class StockTrainResource(Resource):
    lstm = LongShortTermModel()
    rnn = SimpleRNNModel()
    xgb = XGBoostModel()

    @api.expect(parser)
    def post(self):
        params = parser.parse_args()

        model_file = self.get_model_file(params)
        inputs = self.prepare_input(params)

        predictions = []
        method = params['method'].lower()
        if method == 'lstm':
            predictions = self.lstm.predict(model_file, inputs)
        elif method == 'rnn':
            predictions = self.rnn.predict(model_file, inputs)
        else:
            predictions = self.xgb.predict(model_file, inputs)
        return {
            'predictions': predictions
        }

    # def put(self):
    #     download_parallel(list_csv='./data/company_list.csv', download_dir='./data/stocks')
    #
    #     return {
    #         "message": "Models are training..."
    #     }

    def get_model_file(self, params: ParseResult):
        method = params['method'].lower()
        if not os.path.exists(f'./models/{method}'):
            abort(HTTPStatus.BAD_REQUEST, f'Not support method {method}')

        symbol = params['symbol'].upper()
        interval = params['interval'].lower()

        trained_models = os.listdir(f'./models/{method}')
        selected_models = filter(lambda m: symbol in m and interval in m,
                                 trained_models)
        model_file = list(selected_models)[0]

        return f'./models/{method}/{model_file}'

    def prepare_input(self, params: ParseResult):
        method = params['method'].lower()
        symbol = params['symbol'].upper()
        interval = params['interval'].lower()
        period = self.get_proper_period(interval)

        stock_data = yfinance.download(symbol, interval=interval, period=period)

        if method == 'lstm' or method == 'rnn':
            inputs = stock_data['Close'].values[-70:]
            inputs = np.reshape(inputs, (-1, 1))
            return inputs
        else:
            inputs = stock_data[['Close', 'High', 'Low']][-10:]
            return inputs

    def get_proper_period(self, interval: str):
        if interval == '1m':
            return '1d'
        elif interval == '60m':
            return '1mo'
        else:
            return '6mo'


@api.errorhandler(Exception)
def handle_root_exception(error: Exception):
    """Return a custom message and 400 status code"""
    return {
               'message': str(error)
           }, HTTPStatus.INTERNAL_SERVER_ERROR


if __name__ == '__main__':
    app.run()
