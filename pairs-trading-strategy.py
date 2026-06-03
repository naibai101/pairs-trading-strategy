import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import os

corrnum = 0.85
start = "2000-01-01"
end = "2025-12-31"
z_entry = 2.0
z_exit = 0.5
lookback = 60
max_coint_candidates = 100

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
corr_matrix = (
    corr_matrix[corr_matrix["correlation"] > corrnum]
    .sort_values("correlation", ascending=False)
    .head(max_coint_candidates)
    .reset_index(drop=True)
)

coint_results = []
for _, row in corr_matrix.iterrows():
    t1, t2 = row["ticker_1"], row["ticker_2"]
    if t1 not in prices.columns or t2 not in prices.columns:
        continue
    pair_prices = prices[[t1, t2]].dropna()
    if len(pair_prices) < 252:
        continue
    _, pvalue, _ = coint(pair_prices[t1], pair_prices[t2])
    coint_results.append({"ticker_1": t1, "ticker_2": t2, "pvalue": pvalue})

coint_df = pd.DataFrame(coint_results)
coint_df = coint_df[coint_df["pvalue"] < 0.05].sort_values("pvalue").reset_index(drop=True)
top_pairs = coint_df.head(10)


def backtest_pair(prices, t1, t2):
    pair = prices[[t1, t2]].dropna()

    roll_cov = pair[t1].rolling(lookback).cov(pair[t2])
    roll_var = pair[t2].rolling(lookback).var()
    hedge_ratio = roll_cov / roll_var

    spread = pair[t1] - hedge_ratio * pair[t2]
    roll_mean = spread.rolling(lookback).mean()
    roll_std = spread.rolling(lookback).std()
    zscore = (spread - roll_mean) / roll_std

    pair = pair.copy()
    pair["hedge_ratio"] = hedge_ratio
    pair["zscore"] = zscore
    pair = pair.dropna()

    z = pair["zscore"].values
    hr = pair["hedge_ratio"].values
    p1 = pair[t1].values
    p2 = pair[t2].values

    position = 0
    returns = []

    for i in range(1, len(pair)):
        if position == 0:
            if z[i - 1] > z_entry:
                position = -1
            elif z[i - 1] < -z_entry:
                position = 1
        elif position == 1 and z[i - 1] > -z_exit:
            position = 0
        elif position == -1 and z[i - 1] < z_exit:
            position = 0

        r1 = p1[i] / p1[i - 1] - 1
        r2 = p2[i] / p2[i - 1] - 1
        returns.append(position * (r1 - hr[i - 1] * r2))

    return pd.Series(returns)


results = []
for _, row in top_pairs.iterrows():
    t1, t2 = row["ticker_1"], row["ticker_2"]
    rets = backtest_pair(prices, t1, t2)

    total_return = (1 + rets).prod() - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    cumrets = (1 + rets).cumprod()
    max_dd = (cumrets / cumrets.cummax() - 1).min()
    win_rate = (rets > 0).mean()

    results.append({
        "pair": f"{t1}/{t2}",
        "total_return_%": round(total_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_%": round(max_dd * 100, 2),
        "win_rate_%": round(win_rate * 100, 2),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best = top_pairs.iloc[0]
rets = backtest_pair(prices, best["ticker_1"], best["ticker_2"])
equity = (1 + rets).cumprod()

plt.figure(figsize=(12, 5))
plt.plot(equity.values)
plt.title(f"Equity Curve: {best['ticker_1']}/{best['ticker_2']}")
plt.xlabel("Days")
plt.ylabel("Cumulative Return")
plt.tight_layout()
plt.savefig("equity_curve.png", dpi=150)
plt.show()
