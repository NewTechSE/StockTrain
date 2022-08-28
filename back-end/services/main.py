from services.stock_service import download_parallel
from services.train_service import predict_parallel
from trainers.long_short_term_model import LongShortTermModel
from trainers.simple_rnn_model import SimpleRNNModel
from trainers.xg_boost_model import XGBoostModel

if __name__ == '__main__':
    lstm = LongShortTermModel()
    rnn = SimpleRNNModel()
    xgb = XGBoostModel()
    
    predict_parallel(lstm, 4)
    # predict_parallel(rnn, 4)
    # predict_parallel(xgb, 4)

    # download_parallel()
    pass
