import asyncio
import os

from joblib import Parallel, delayed

from trainers.model import Model


def train_parallel(model: Model, n_threads: int):
    def train_sync(csv_filename):
        asyncio.run(model.train(csv_filename))

    data_dir = "../data/stocks"
    csv_files = []
    for filename in os.listdir(data_dir):
        csv_files.append(os.path.join(data_dir, filename))

    Parallel(n_jobs=n_threads)(delayed(train_sync)(csv_file) for csv_file in csv_files)
