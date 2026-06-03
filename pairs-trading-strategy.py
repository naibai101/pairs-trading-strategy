import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import os

corrnum = 0.85
start = "2000-01-01"
train_end = "2010-12-31"
end = "2025-12-31"
z_entry = 2.0
z_exit = 0.5
z_stop = 3.5
lookback_hedge = 252
lookback_zscore = 60
transaction_cost = 0.001
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

log_prices = np.log(prices)

coint_results = []
for _, row in corr_matrix.iterrows():
    t1, t2 = row["ticker_1"], row["ticker_2"]
    if t1 not in log_prices.columns or t2 not in log_prices.columns:
        continue
    pair_prices = log_prices[[t1, t2]].dropna()
    if len(pair_prices) < 252:
        continue
    _, pvalue, _ = coint(pair_prices[t1], pair_prices[t2])
    coint_results.append({"ticker_1": t1, "ticker_2": t2, "pvalue": pvalue})

coint_df = pd.DataFrame(coint_results)
coint_df = coint_df[coint_df["pvalue"] < 0.05].sort_values("pvalue").reset_index(drop=True)
top_pairs = coint_df.head(10)


def backtest_pair(prices, t1, t2):
    pair = np.log(prices[[t1, t2]].dropna())

    roll_cov = pair[t1].rolling(lookback_hedge).cov(pair[t2])
    roll_var = pair[t2].rolling(lookback_hedge).var()
    hedge_ratio = roll_cov / roll_var

    spread = pair[t1] - hedge_ratio * pair[t2]
    roll_mean = spread.rolling(lookback_zscore).mean()
    roll_std = spread.rolling(lookback_zscore).std()
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
        elif position == 1:
            if z[i - 1] > -z_exit or z[i - 1] < -z_stop:
                position = 0
        elif position == -1:
            if z[i - 1] < z_exit or z[i - 1] > z_stop:
                position = 0

        r1 = p1[i] - p1[i - 1]
        r2 = p2[i] - p2[i - 1]
        returns.append(position * (r1 - hr[i - 1] * r2))

    return pd.Series(returns)


results = []
for _, row in top_pairs.iterrows():
    t1, t2 = row["ticker_1"], row["ticker_2"]
    rets = backtest_pair(prices, t1, t2)

    total_return = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    downside = rets[rets < 0].std()
    sortino = rets.mean() / downside * np.sqrt(252) if downside > 0 else 0
    cumrets = (1 + rets).cumprod()
    max_dd = (cumrets / cumrets.cummax() - 1).min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf")
    position_changes = pd.Series(np.diff([0] + [1 if r != 0 else 0 for r in rets.values]))
    num_trades = int((position_changes == 1).sum())

    results.append({
        "pair": f"{t1}/{t2}",
        "ann_return_%": round(ann_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "profit_factor": profit_factor,
        "num_trades": num_trades,
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

palette = [
    "#e040fb", "#ce93d8", "#f48fb1", "#f06292", "#ba68c8",
    "#ab47bc", "#ec407a", "#ff80ab", "#ea80fc", "#b39ddb",
]

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("#0e0e1a")
ax.set_facecolor("#0e0e1a")

for i, (_, row) in enumerate(top_pairs.iterrows()):
    t1, t2 = row["ticker_1"], row["ticker_2"]
    rets = backtest_pair(prices, t1, t2)
    equity = (1 + rets).cumprod()
    color = palette[i % len(palette)]
    ax.plot(equity.values, color=color, linewidth=1.4, alpha=0.85, label=f"{t1} / {t2}")
    ax.annotate(
        f"{t1}/{t2}",
        xy=(len(equity) - 1, equity.iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=color,
        fontsize=7.5,
        va="center",
    )

ax.axhline(1.0, color="#444466", linewidth=0.8, linestyle="--")
ax.set_title("Equity Curves — Top Cointegrated Pairs (S&P 500)", color="#f0e6ff", fontsize=14, pad=14)
ax.set_xlabel("Trading Days", color="#9e8fb2", fontsize=10)
ax.set_ylabel("Cumulative Return", color="#9e8fb2", fontsize=10)
ax.tick_params(colors="#9e8fb2")
for spine in ax.spines.values():
    spine.set_edgecolor("#2a2040")
ax.grid(color="#1e1a2e", linewidth=0.6)
ax.legend(
    loc="upper left",
    fontsize=7.5,
    framealpha=0.2,
    facecolor="#1a1030",
    edgecolor="#3a2060",
    labelcolor="#d8bfff",
)

plt.tight_layout()
plt.savefig("equity_curve.png", dpi=150, facecolor=fig.get_facecolor())
plt.show()
