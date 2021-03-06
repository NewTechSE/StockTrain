import asyncio
import logging
from os.path import exists

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)


def readStockCompanyData(file_path: str):
    """
    Reads the stock company data from a csv file. Return their stock symbols and names
    """

    if exists(file_path) is False:
        raise FileNotFoundError(f"File not found {file_path}")

    data = pd.read_csv(file_path)

    logging.info(f"Read {len(data)} companies")

    return data[['Symbol', 'Name']]


async def downloadStockDataByInterval(period: str, interval: str):
    """
    Downloads stock data of all company from yfinance by interval. Save to data/stocks folder
    """

    company_data = readStockCompanyData('./data/company_list.csv')
    result_list = []
    for index, row in company_data.iterrows():
        stock_data: pd.DataFrame = yf.download(
            row['Symbol'], period=period, interval=interval, auto_adjust=True, threads=True)

        file_name = f'./data/stocks/{row["Symbol"]}_{period}_{interval}.csv'
        stock_data.to_csv(file_name)

        result_list.append(file_name)

        logging.info(f"Downloaded {row['Symbol']} - {row['Name']}")
        
    return result_list


async def main():
    # Download some basic data
    await asyncio.gather(
        downloadStockDataByInterval(period='5y', interval='1d'),
        downloadStockDataByInterval(period='1y', interval='60m'),
        downloadStockDataByInterval(period='7d', interval='1m')
    )

# asyncio.run(main())
