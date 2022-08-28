from stock_service import download_parallel
import time

while True:
    download_parallel()
    time.sleep(10)