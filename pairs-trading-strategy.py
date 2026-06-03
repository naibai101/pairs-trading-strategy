import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import os

corrnum = 0.8
start = "2000-01-01"
end = "2025-12-31"
z_entry = 2.0
z_exit = 0.5
lookback = 60

if os.path.exists("prices.csv"):
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
else:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    table = pd.read_html(response.text)
    tickers_list = table[0]["Symbol"].tolist()
    tickers_list = [t.replace(".", "-") for t in tickers_list]
    df = yf.download(tickers_list, start=start, end=end)
    prices = df["Close"]
    prices.to_csv("prices.csv")

corr_matrix = prices.corr()
upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
corr_matrix = corr_matrix.where(upper).stack().reset_index()
corr_matrix = corr_matrix.rename(columns={"level_0": "ticker_1", "level_1": "ticker_2", 0: "correlation"})
corr_matrix = corr_matrix[corr_matrix["correlation"] > corrnum].reset_index(drop=True)