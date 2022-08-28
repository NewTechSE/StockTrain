from datetime import datetime, timezone

import pandas as pd
from flask import Response
from flask_restx import Namespace, Resource
from flask_restx.reqparse import ParseResult

from trainers.long_short_term_model import LongShortTermModel
from trainers.simple_rnn_model import SimpleRNNModel
from trainers.xg_boost_model import XGBoostModel
from utils.api_util import create_basic_request_parser
from services.stock_service import format_data

api = Namespace('prediction-history', "Prediction history")

parser = create_basic_request_parser()


def get_prediction_data(params: ParseResult) -> pd.DataFrame:
    method = params['method'].lower()
    symbol = params['symbol'].upper()
    interval = params['interval'].lower()

    period_of = {
        '1m': '7d',
        '60m': '1y',
        '1d': '5y',
    }

    csv_file = f'./data/predictions/{method}/{symbol}_{period_of[interval]}_{interval}_close.csv'
    pred_data = format_data(pd.read_csv(csv_file))

    return pred_data[-1000:]


@api.route('/')
class PredictionHistory(Resource):

    @api.expect(parser)
    def get(self):
        data = get_prediction_data(parser.parse_args())
        return Response(
            data.to_json(orient="records"),
            mimetype='application/json',
        )
