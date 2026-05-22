<<<<<<< HEAD
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

#finding the stocks in the s&p 
table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0, attrs={"id": "constituents"})
tickers_list = table[0]

for ticker in tickers_list:
    ticker = ticker.replace(".", "-")

df = yf.download(tickers_list, start=start, end=end)
prices = df['Close']

<<<<<<< Updated upstream

=======
#
# pairs trading strategy for markets
#

import numpy as np
import scipy as sp
import pandas as pd
import statsmodels as sm
import matplotlib as plt
import yfinance as yf

CORRELATION_NUMBER = 0.8

>>>>>>> b63bb781358063faba62a277cdfb89729030140b
=======
prices
>>>>>>> Stashed changes
