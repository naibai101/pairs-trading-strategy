#
# pairs trading strategy for markets
#

import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import matplotlib as plt
import yfinance as yf

#constants
corrnum = 0.8
start = "2000-01-01"
end = "2025-12-31"
interval = "1d"

tickers_list = []
tickers_dict = {}


for ticker in tickers_list:
    t = yf.Ticker(ticker)

    temp = pd.DataFrame.from_dict(t.info, orient='index')

ticker = yf.Ticker("aapl")
historical = ticker.history(start=start, end=end, interval=interval)
historical


