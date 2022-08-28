import os.path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from flask import Flask, Response
from flask_restx import Resource, Api, reqparse, abort
from flask_restx._http import HTTPStatus
from flask_restx.reqparse import ParseResult

from services.stock_service import read_stock_company_data, \
    format_data
from trainers.long_short_term_model import LongShortTermModel
from trainers.simple_rnn_model import SimpleRNNModel
from trainers.xg_boost_model import XGBoostModel
from utils.api_util import create_basic_request_parser
from namespaces.prediction_history import api as prediction_history_ns

app = Flask(__name__)
api = Api(app, prefix='/api')

api.add_namespace(prediction_history_ns)

parser = create_basic_request_parser()

list_csv = './data/company_list.csv'
# download_parallel(list_csv=list_csv, download_dir='./data/stocks')
company_data = read_stock_company_data(list_csv)

stock_args = [
    ('5y', '1d'),
    ('1y', '60m'),
    ('7d', '1m')
]
stock_data = {}
lstm = LongShortTermModel()
rnn = SimpleRNNModel()
xgb = XGBoostModel()

for index, company in company_data.iterrows():
    for (period, interval) in stock_args:
        for method in ['lstm', 'rnn', 'xgboost']:
            file_name = f'./data/predictions/{method}/{company["Symbol"]}_{period}_{interval}_close.csv'
            df = format_data(pd.read_csv(file_name))
            stock_data[f'{company["Symbol"]}_{method}_{period}_{interval}'] = df


# noinspection PyMethodMayBeStatic
@api.route('/stock-train')
class StockTrainResource(Resource):
    lstm = LongShortTermModel()
    rnn = SimpleRNNModel()
    xgb = XGBoostModel()

    def __init__(self, api=None, *args, **kwargs):
        super().__init__(api, *args, **kwargs)
        global stock_data
        self.stock_data = stock_data

    @api.expect(parser)
    def get(self):
        params = parser.parse_args()

        stock_data = self.get_stock_data(params)

        return Response(
            stock_data[['Date', 'Open', 'Close', 'Low', 'High']].to_json(orient="records"),
            mimetype='application/json',
        )

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

        data = []

        for i in range(1, len(predictions)):
            td = self.timedelta_from_interval(params['interval'].lower(), i)
            data.append(
                {'Close': predictions[i], 'Date': str(datetime.now() + td)})
        return {
            'data': data
        }

    def timedelta_from_interval(self, interval: str, i):
        td = timedelta(minutes=i)
        match interval:
            case '60m':
                td = timedelta(hours=i)
            case '1d':
                td = timedelta(days=i)
        return td

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

        stock_data = self.get_stock_data(params)

        if method == 'lstm' or method == 'rnn':
            inputs = stock_data['Close'].values[-70:]
            inputs = np.reshape(inputs, (-1, 1))
            return inputs
        else:
            inputs = stock_data[['Close', 'High', 'Low']][-10:]
            return inputs

    @api.expect(parser)
    def post(self):
        params = parser.parse_args()
        return Response(
            self.get_data(params).to_json(orient="records"),
            mimetype='application/json',
        )

    def get_stock_data(self, params: ParseResult) -> pd.DataFrame:
        symbol = params['symbol'].upper()
        interval = params['interval'].lower()
        period = self.get_proper_period(interval)
        method = params['method'].lower()
        stock_data = self.stock_data[f'{symbol}_{method}_{period}_{interval}']
        return stock_data[stock_data['Date'] < datetime.now(timezone.utc).timestamp()][-1000:]

    def get_data(self, params: ParseResult) -> pd.DataFrame:
        symbol = params['symbol'].upper()
        interval = params['interval'].lower()
        period = self.get_proper_period(interval)
        method = params['method'].lower()
        stock_data = self.stock_data[f'{symbol}_{method}_{period}_{interval}']
        data = stock_data[stock_data['Date'] > datetime.now(timezone.utc).timestamp()][0:10]
        return pd.concat([stock_data[stock_data['Date'] < datetime.now(timezone.utc).timestamp()][-1000:], data])[
            ['Date', 'Prediction']]

    def get_proper_period(self, interval: str):
        if interval == '1m':
            return '7d'
        elif interval == '60m':
            return '1y'
        else:
            return '5y'


@api.errorhandler(Exception)
def handle_root_exception(error: Exception):
    """Return a custom message and 400 status code"""
    return {
               'message': str(error)
           }, HTTPStatus.INTERNAL_SERVER_ERROR


if __name__ == '__main__':
    app.run(debug=True)
