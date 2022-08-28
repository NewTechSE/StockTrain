from flask_restx.reqparse import RequestParser


def create_basic_request_parser() -> RequestParser:
    parser = RequestParser()

    parser.add_argument('method', type=str, location='args',
                        required=True, trim=True, help='lstm | rnn | xgboost')
    parser.add_argument('symbol', type=str, location='args',
                        required=True, trim=True, help='adbe | googl | msft')
    parser.add_argument('interval', type=str, location='args',
                        required=True, trim=True, help='1m | 60m | 1d')

    return parser
