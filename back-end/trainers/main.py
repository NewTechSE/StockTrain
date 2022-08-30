import logging

from services.train_service import train_parallel
from trainers.long_short_term_model import LongShortTermModel
from trainers.simple_rnn_model import SimpleRNNModel
from trainers.xg_boost_model import XGBoostModel

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    lstm = LongShortTermModel()
    rnn = SimpleRNNModel()
    xg = XGBoostModel()

    train_parallel(lstm, n_threads=3)
    train_parallel(rnn, n_threads=3)
    train_parallel(xg, n_threads=3)
