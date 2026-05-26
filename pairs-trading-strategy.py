#
# pairs trading strategy for markets
#

import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import matplotlib as plt
import yfinance as yf
import requests
import os

#constants
corrnum = 0.8
start = "2000-01-01"
end = "2025-12-31"
interval = "1d"

tickers_list = []

#finding the stocks in the s&p 
if os.path.exists("prices.csv"):
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
else:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    table = pd.read_html(response.text)
    tickers_list = table[0]["Symbol"].tolist()

    for ticker in tickers_list:
        ticker = ticker.replace(".", "-")
    df = yf.download(tickers_list, start=start, end=end)
    prices = df['Close']
    prices.to_csv("prices.csv") 

#calculating correlations
corr_matrix = prices.corr()
matrix = np.triu(corr_matrix, k=1)
pairs = matrix[(matrix > 0.8) & (matrix < 1.0)]
print(pairs.reset_index())

