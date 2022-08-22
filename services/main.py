import asyncio

from services.stock_service import download_stock_data_by_interval


async def main():
    # Download some basic data
    await asyncio.gather(
        download_stock_data_by_interval(period='5y', interval='1d'),
        download_stock_data_by_interval(period='1y', interval='60m'),
        download_stock_data_by_interval(period='7d', interval='1m')
    )


asyncio.run(main())
