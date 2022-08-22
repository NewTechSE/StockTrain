import asyncio

from trainers.long_short_term_model import LongShortTermModel
from trainers.xg_boost_model import XGBoostModel


async def main():
    model = XGBoostModel()
    await asyncio.gather(
        model.train('../data/stocks/AAPL_5y_1d.csv'),
    )

asyncio.run(main())
