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

train_prices = prices.loc[:train_end]
test_prices = prices.loc[train_end:]

corr_matrix = train_prices.corr()
upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
corr_matrix = corr_matrix.where(upper).stack().reset_index()
corr_matrix = corr_matrix.rename(columns={"level_0": "ticker_1", "level_1": "ticker_2", 0: "correlation"})
corr_matrix = (
    corr_matrix[(corr_matrix["correlation"] > corrnum) & (corr_matrix["correlation"] < 0.99)]
    .sort_values("correlation", ascending=False)
    .head(max_coint_candidates)
    .reset_index(drop=True)
)

def half_life(spread):
    s = spread.dropna().values
    lag = s[:-1]
    diff = np.diff(s)
    X = np.column_stack([np.ones(len(lag)), lag])
    beta = np.linalg.lstsq(X, diff, rcond=None)[0]
    if beta[1] >= 0:
        return np.nan
    return -np.log(2) / beta[1]

train_log = np.log(train_prices)

coint_results = []
for _, row in corr_matrix.iterrows():
    t1, t2 = row["ticker_1"], row["ticker_2"]
    if t1 not in train_log.columns or t2 not in train_log.columns:
        continue
    pair_prices = train_log[[t1, t2]].dropna()
    if len(pair_prices) < 252:
        continue
    _, pvalue, _ = coint(pair_prices[t1], pair_prices[t2])
    if pvalue >= 0.01:
        continue
    X = np.column_stack([np.ones(len(pair_prices)), pair_prices[t2].values])
    hr = np.linalg.lstsq(X, pair_prices[t1].values, rcond=None)[0][1]
    spread = pair_prices[t1] - hr * pair_prices[t2]
    hl = half_life(spread)
    if np.isnan(hl) or hl < 5 or hl > 60:
        continue
    coint_results.append({"ticker_1": t1, "ticker_2": t2, "pvalue": pvalue, "half_life": round(hl, 1), "hedge_ratio": hr})

coint_df = pd.DataFrame(coint_results) if coint_results else pd.DataFrame(columns=["ticker_1", "ticker_2", "pvalue", "half_life", "hedge_ratio"])
coint_df = coint_df.sort_values("pvalue").reset_index(drop=True)
if coint_df.empty:
    raise ValueError("No cointegrated pairs found. Try lowering corrnum or widening the training window.")
top_pairs = coint_df.head(10)


def backtest_pair(prices, t1, t2):
    raw = prices[[t1, t2]].dropna()
    log_pair = np.log(raw)

    roll_cov = log_pair[t1].rolling(lookback_hedge).cov(log_pair[t2])
    roll_var = log_pair[t2].rolling(lookback_hedge).var()
    hedge_ratio = roll_cov / roll_var

    spread = log_pair[t1] - hedge_ratio * log_pair[t2]
    roll_mean = spread.rolling(lookback_zscore).mean()
    roll_std = spread.rolling(lookback_zscore).std()
    zscore = (spread - roll_mean) / roll_std

    df = raw.copy()
    df["hedge_ratio"] = hedge_ratio
    df["zscore"] = zscore
    df = df.dropna()

    z = df["zscore"].values
    hr = df["hedge_ratio"].values
    p1 = df[t1].values
    p2 = df[t2].values

    position = 0
    prev_position = 0
    returns = []

    for i in range(1, len(df)):
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

        r1 = p1[i] / p1[i - 1] - 1
        r2 = p2[i] / p2[i - 1] - 1
        pnl = position * (r1 - hr[i - 1] * r2)

        if position != prev_position:
            pnl -= transaction_cost * 2

        returns.append(pnl)
        prev_position = position

    return pd.Series(returns)


results = []
for _, row in top_pairs.iterrows():
    t1, t2 = row["ticker_1"], row["ticker_2"]
    rets = backtest_pair(test_prices, t1, t2)

    total_return = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    active = rets[rets != 0]
    sharpe = active.mean() / active.std() * np.sqrt(252) if len(active) > 1 else 0
    downside = active[active < 0].std()
    sortino = active.mean() / downside * np.sqrt(252) if len(active[active < 0]) > 1 else 0
    cumrets = (1 + rets).cumprod()
    max_dd = (cumrets / cumrets.cummax() - 1).min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf")
    position_changes = pd.Series(np.diff([0] + [1 if r != 0 else 0 for r in rets.values]))
    num_trades = int((position_changes == 1).sum())

    hl = top_pairs.loc[top_pairs["ticker_1"] == t1].iloc[0]["half_life"] if "half_life" in top_pairs.columns else np.nan

    results.append({
        "pair": f"{t1}/{t2}",
        "half_life_d": hl,
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
    "#dc143c",
    "#c0c0c0",
    "#00b4d8",
    "#e63946",
    "#4fc3f7",
    "#ff4d6d",
    "#a8dadc",
    "#9e0031",
    "#48cae4",
    "#d4d4d4",
]

BG       = "#0a0a0f"
GRID_COL = "#1a0a0a"
TEXT_COL = "#c8c8c8"
SPINE    = "#3a0a0a"
ZERO     = "#5a1a1a"

fig, ax = plt.subplots(figsize=(15, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

pair_returns = {}
for i, (_, row) in enumerate(top_pairs.iterrows()):
    t1, t2 = row["ticker_1"], row["ticker_2"]
    rets = backtest_pair(test_prices, t1, t2)
    pair_returns[f"{t1}/{t2}"] = rets
    equity = (1 + rets).cumprod()
    color = palette[i % len(palette)]
    ax.plot(equity.values, color=color, linewidth=1.5, alpha=0.9, label=f"{t1} / {t2}")
    ax.annotate(
        f"{t1}/{t2}",
        xy=(len(equity) - 1, equity.iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=color,
        fontsize=7.5,
        va="center",
        fontweight="bold",
    )

results_df = pd.DataFrame(results)
best_sharpe_row = results_df.loc[results_df["sharpe"].idxmax()]
best_return_row = results_df.loc[results_df["ann_return_%"].idxmax()]
best_sortino_row = results_df.loc[results_df["sortino"].idxmax()]

stats_text = (
    f"  Best Sharpe   {best_sharpe_row['pair']}   {best_sharpe_row['sharpe']:.3f}\n"
    f"  Best Ann. Ret  {best_return_row['pair']}   {best_return_row['ann_return_%']:.2f}%\n"
    f"  Best Sortino  {best_sortino_row['pair']}   {best_sortino_row['sortino']:.3f}"
)
ax.text(
    0.01, 0.03, stats_text,
    transform=ax.transAxes,
    fontsize=8,
    color="#c0c0c0",
    verticalalignment="bottom",
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#140505", edgecolor="#8b0000", alpha=0.85),
)

ax.axhline(1.0, color=ZERO, linewidth=0.9, linestyle="--")
ax.set_title("Equity Curves — Top Cointegrated Pairs (S&P 500, 2010–2025)", color=TEXT_COL, fontsize=14, pad=14)
ax.set_xlabel("Trading Days", color=TEXT_COL, fontsize=10)
ax.set_ylabel("Cumulative Return", color=TEXT_COL, fontsize=10)
ax.tick_params(colors=TEXT_COL)
for spine in ax.spines.values():
    spine.set_edgecolor(SPINE)
ax.grid(color=GRID_COL, linewidth=0.6)
ax.legend(
    loc="upper right",
    fontsize=7.5,
    framealpha=0.25,
    facecolor="#0f0505",
    edgecolor="#8b0000",
    labelcolor=TEXT_COL,
)

plt.tight_layout()
plt.savefig("equity_curve.png", dpi=150, facecolor=fig.get_facecolor())
plt.show()
