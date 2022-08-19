import asyncio

from trainers.long_short_term_model import LongShortTermModel


async def main():
    lstm = LongShortTermModel()
    await asyncio.gather(
        lstm.train('../data/stocks/AAPL_5y_1d.csv'),
    )

asyncio.run(main())
