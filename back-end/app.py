import asyncio
from distutils.log import debug
import os.path
import subprocess

import numpy as np
import pandas as pd
import yfinance
from flask import Flask, jsonify, Response
from flask_restx import Resource, Api, reqparse, abort
from flask_restx._http import HTTPStatus
from flask_restx.reqparse import ParseResult

from services.train_service import train_parallel
from trainers.long_short_term_model import LongShortTermModel
from trainers.simple_rnn_model import SimpleRNNModel
from trainers.xg_boost_model import XGBoostModel
from services.stock_service import download_stock_data_by_interval, download_parallel, read_stock_company_data, format_data
from datetime import datetime, timedelta, timezone

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
    
    def __init__(self, api=None, *args, **kwargs):
        super().__init__(api, *args, **kwargs)
        
        self.stock_data = {}
        
        list_csv='./data/company_list.csv'
        # download_parallel(list_csv=list_csv, download_dir='./data/stocks')
        company_data = read_stock_company_data(list_csv)
 
        stock_args = [
            ('5y', '1d'),
            ('1y', '60m'),
            ('7d', '1m')
        ]

        for index, company in company_data.iterrows():
            for (period, interval) in stock_args:
                file_name = f'./data/stocks/{company["Symbol"]}_{period}_{interval}.csv'
                df = format_data(pd.read_csv(file_name))
                self.stock_data[f'{company["Symbol"]}_{period}_{interval}'] = df

    @api.expect(parser)
    def get(self):
        params = parser.parse_args()
        stock_data = self.get_stock_data(params)
        
        return Response(
            stock_data.to_json(orient="records"),
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

        for i in range(1,len(predictions)): 
            td = self.timedelta_from_interval(params['interval'].lower(), i)
            data.append({'value': predictions[i], 'time':str(datetime.now() + td)})
        return {
            'data': data
        }

    # def put(self):
    #     download_parallel(list_csv='./data/company_list.csv', download_dir='./data/stocks')
    #
    #     return {
    #         "message": "Models are training..."
    #     }
    
    def timedelta_from_interval(self, interval: str, i):
        td = timedelta(minutes=i)
        match interval:
            case '60m':
                td = timedelta(hours=i)
            case '1d':
                td =timedelta(days=i)
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
        
    def get_stock_data(self, params: ParseResult):
        symbol = params['symbol'].upper()
        interval = params['interval'].lower()
        period = self.get_proper_period(interval)

        stock_data = self.stock_data[f'{symbol}_{period}_{interval}']
        
        return stock_data[stock_data['Date'] < datetime.now(timezone.utc)][-1000:]

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
