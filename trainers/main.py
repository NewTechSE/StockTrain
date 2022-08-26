import asyncio
import logging
import os
from multiprocessing import Process

from joblib import Parallel, delayed

from services.stock_service import download_stock_data_by_interval
from trainers.long_short_term_model import LongShortTermModel
from trainers.model import Model
from trainers.simple_rnn_model import SimpleRNNModel
from trainers.xg_boost_model import XGBoostModel

logging.basicConfig(level=logging.INFO)


async def main():
    lstm = LongShortTermModel()
    rnn = SimpleRNNModel()
    xg = XGBoostModel()

    data_dir = "../data/stocks"
    tasks = []
    for filename in os.listdir(data_dir):
        tasks.append(asyncio.create_task(lstm.train(os.path.join(data_dir, filename))))
        tasks.append(asyncio.create_task(rnn.train(os.path.join(data_dir, filename))))
        tasks.append(asyncio.create_task(xg.train(os.path.join(data_dir, filename))))

    logging.info(f"Tasks: {len(tasks)}")

    await asyncio.gather(*tasks)


# asyncio.run(main())

def download_parallel():
    download_args = [
        ('5y', '1d'),
        ('1y', '60m'),
        ('7d', '1m')
    ]

    download_processes = []
    for args in download_args:
        p = Process(target=asyncio.run(download_stock_data_by_interval(period=args[0], interval=args[1])))
        p.start()
        download_processes.append(p)

    for p in download_processes:
        p.join()


def train_parallel(model: Model, n_threads: int):
    def train_sync(csv_filename):
        asyncio.run(model.train(csv_filename))

    data_dir = "../data/stocks"
    csv_files = []
    for filename in os.listdir(data_dir):
        csv_files.append(os.path.join(data_dir, filename))

    Parallel(n_jobs=n_threads)(delayed(train_sync)(csv_file) for csv_file in csv_files)


if __name__ == '__main__':
    lstm = LongShortTermModel()
    rnn = SimpleRNNModel()
    xg = XGBoostModel()

    train_parallel(model=lstm, n_threads=3)
    train_parallel(model=rnn, n_threads=3)
    train_parallel(model=xg, n_threads=3)
