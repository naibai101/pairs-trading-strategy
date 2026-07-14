import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import yfinance as yf
import requests
import os
import warnings
warnings.filterwarnings("ignore")

ret_corr_min   = 0.40
ret_corr_max   = 0.97
coint_pval     = 0.10
hl_min         = 3
hl_max         = 35
train_months   = 24
z_entry        = 2.0
z_exit         = 0.0
z_stop         = 4.0
tc             = 0.0001
max_cands      = 80
max_pairs      = 30
data_start     = "2003-01-01"
data_end       = "2025-01-01"

if os.path.exists("prices.csv"):
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
else:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    tbl = pd.read_html(requests.get(url, headers=headers).text)[0]
    tickers = [t.replace(".", "-") for t in tbl["Symbol"].tolist()]
    prices = yf.download(tickers, start=data_start, end=data_end)["Close"]
    prices.to_csv("prices.csv")

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}
sp500_tbl = pd.read_html(requests.get(url, headers=headers).text)[0]
sp500_tbl["Symbol"] = sp500_tbl["Symbol"].str.replace(".", "-")
sub_industry = sp500_tbl.set_index("Symbol")["GICS Sub-Industry"].squeeze()
sectors      = sp500_tbl.set_index("Symbol")["GICS Sector"].squeeze()


def half_life(spread):
    s = spread.dropna().values
    if len(s) < 10:
        return np.nan
    lag, diff = s[:-1], np.diff(s)
    X = np.column_stack([np.ones(len(lag)), lag])
    beta = np.linalg.lstsq(X, diff, rcond=None)[0]
    return np.nan if beta[1] >= 0 else -np.log(2) / beta[1]


def find_pairs(prices_window):
    rets  = prices_window.pct_change().dropna()
    log_p = np.log(prices_window)
    results = []
    used = set()

    for sub in sub_industry.unique():
        tickers = [t for t in sub_industry[sub_industry == sub].index
                   if t in prices_window.columns and t in rets.columns]
        if len(tickers) < 2:
            continue
        corr = rets[tickers].corr()
        upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
        cp = corr.where(upper).stack().reset_index()
        cp.columns = ["t1", "t2", "corr"]
        cp = cp[(cp["corr"] > ret_corr_min) & (cp["corr"] < ret_corr_max)]
        cp = cp.sort_values("corr", ascending=False).head(max_cands)
        for _, row in cp.iterrows():
            t1, t2 = row["t1"], row["t2"]
            if t1 in used or t2 in used:
                continue
            pair = log_p[[t1, t2]].dropna()
            if len(pair) < int(252 * 0.6):
                continue
            _, pvalue, _ = coint(pair[t1], pair[t2])
            if pvalue >= coint_pval:
                continue
            r1 = rets[t1].reindex(pair.index).dropna()
            r2 = rets[t2].reindex(pair.index).dropna()
            aln = pd.concat([r1, r2], axis=1).dropna()
            hr = float(
                np.cov(aln.iloc[:, 0], aln.iloc[:, 1])[0, 1]
                / max(np.var(aln.iloc[:, 1]), 1e-10)
            )
            if hr <= 0:
                continue
            spread = pair[t1] - hr * pair[t2]
            hl = half_life(spread)
            if np.isnan(hl) or hl < hl_min or hl > hl_max:
                continue
            results.append({
                "t1": t1, "t2": t2, "pvalue": pvalue,
                "half_life": round(hl, 1), "hedge_ratio": hr,
                "spread_vol": float(spread.diff().std()),
                "sub_industry": sub,
            })
            used.update([t1, t2])
            if len(results) >= max_pairs:
                break
        if len(results) >= max_pairs:
            break

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("pvalue").reset_index(drop=True)


def backtest_pair(train_p, test_p, t1, t2, hr, hl):
    lookback = max(int(hl * 2), 20)
    log_train = np.log(train_p[[t1, t2]].dropna()).tail(lookback * 3)
    log_test  = np.log(test_p[[t1, t2]].dropna())
    combined  = pd.concat([log_train, log_test])
    spread    = combined[t1] - hr * combined[t2]
    rm = spread.rolling(lookback, min_periods=max(lookback // 2, 10)).mean()
    rs = spread.rolling(lookback, min_periods=max(lookback // 2, 10)).std()
    z_full = ((spread - rm) / rs).replace([np.inf, -np.inf], np.nan).fillna(0)
    z_s = z_full.loc[log_test.index]
    p1  = test_p[t1].reindex(log_test.index).values
    p2  = test_p[t2].reindex(log_test.index).values
    z   = z_s.values
    scale = 1.0 + abs(hr)
    position = 0
    prev_pos = 0
    returns  = []
    for i in range(1, len(z)):
        if position == 0:
            if   z[i-1] >  z_entry: position = -1
            elif z[i-1] < -z_entry: position =  1
        elif position == 1:
            if z[i-1] > z_exit or z[i-1] < -z_stop:
                position = 0
        elif position == -1:
            if z[i-1] < -z_exit or z[i-1] > z_stop:
                position = 0
        r1 = p1[i] / p1[i-1] - 1 if p1[i-1] != 0 else 0
        r2 = p2[i] / p2[i-1] - 1 if p2[i-1] != 0 else 0
        pnl = position * (r1 - hr * r2) / scale
        if position != prev_pos:
            pnl -= tc * 2
        returns.append(pnl)
        prev_pos = position
    return pd.Series(returns, index=log_test.index[1:])


quarters  = pd.date_range("2007-01-01", "2024-12-31", freq="QS")
all_rets  = []
pair_log  = []

for i, q_start in enumerate(quarters[:-1]):
    q_end       = quarters[i + 1] - pd.Timedelta(days=1)
    train_end   = q_start - pd.Timedelta(days=1)
    train_start = train_end - pd.DateOffset(months=train_months)

    train_p = prices.loc[str(train_start.date()):str(train_end.date())]
    test_p  = prices.loc[str(q_start.date()):str(q_end.date())]

    if len(train_p) < 200 or len(test_p) < 10:
        continue

    thresh  = int(0.85 * len(train_p))
    avail   = train_p.dropna(axis=1, thresh=thresh).columns
    train_p = train_p[avail]
    test_p  = test_p[[c for c in avail if c in test_p.columns]]

    pairs = find_pairs(train_p)
    if pairs.empty:
        continue

    q_rets, q_vols = [], []
    for _, pr in pairs.iterrows():
        t1, t2 = pr["t1"], pr["t2"]
        if t1 not in test_p.columns or t2 not in test_p.columns:
            continue
        r = backtest_pair(train_p, test_p, t1, t2, pr["hedge_ratio"], pr["half_life"])
        if len(r) > 3:
            q_rets.append(r)
            q_vols.append(pr["spread_vol"])
            pair_log.append({
                "quarter": q_start, "t1": t1, "t2": t2,
                "sub_industry": pr.get("sub_industry", ""),
                "half_life": pr["half_life"], "pvalue": round(pr["pvalue"], 4),
            })

    if not q_rets:
        continue

    inv_vols = np.array([1.0 / max(v, 1e-8) for v in q_vols])
    weights  = inv_vols / inv_vols.sum()
    df   = pd.concat(q_rets, axis=1).fillna(0)
    port = pd.Series(df.values @ weights, index=df.index)
    all_rets.append(port)

portfolio = pd.concat(all_rets).sort_index()
cum       = (1 + portfolio).cumprod()

total   = float(cum.iloc[-1] - 1)
n_yrs   = len(portfolio) / 252
ann     = float((1 + total) ** (1 / n_yrs) - 1)
act     = portfolio[portfolio != 0]
sharpe  = float(act.mean() / act.std() * np.sqrt(252)) if len(act) > 1 else 0
dn      = act[act < 0]
sortino = float(act.mean() / dn.std() * np.sqrt(252)) if len(dn) > 1 else 0
mdd     = float((cum / cum.cummax() - 1).min())
calmar  = float(ann / abs(mdd)) if mdd != 0 else 0
gp      = portfolio[portfolio > 0].sum()
gl      = abs(portfolio[portfolio < 0].sum())
pf      = round(float(gp / gl), 3) if gl > 0 else float("inf")

BG      = "#0a0a0f"
GRID    = "#12121e"
TEXT    = "#c8c8d4"
SPINE   = "#2a1a3a"
CRIMSON = "#dc143c"
SILVER  = "#c0c0c0"
TEAL    = "#00b4d8"
DARK_R  = "#8b0000"

palette = ["#c0c0c0", "#00b4d8", "#4fc3f7", "#a8dadc",
           "#48cae4", "#90e0ef", "#d4d4d4", "#e0fbfc"]

fig = plt.figure(figsize=(16, 8), facecolor=BG)
ax  = fig.add_subplot(111)
ax.set_facecolor(BG)

seen_pairs = {}
for rec in pair_log:
    key = f"{rec['t1']}/{rec['t2']}"
    if key not in seen_pairs:
        seen_pairs[key] = rec

sample = list(seen_pairs.values())[:8]
for idx, rec in enumerate(sample):
    t1, t2   = rec["t1"], rec["t2"]
    q_start  = rec["quarter"]
    q_end    = q_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    train_end   = q_start - pd.Timedelta(days=1)
    train_start = train_end - pd.DateOffset(months=train_months)
    train_p = prices.loc[str(train_start.date()):str(train_end.date())]
    test_p  = prices.loc[str(q_start.date()):str(q_end.date())]
    avail = train_p.dropna(axis=1, thresh=int(0.85 * len(train_p))).columns
    test_p = test_p[[c for c in avail if c in test_p.columns]]
    if t1 not in test_p.columns or t2 not in test_p.columns:
        continue
    ec = (1 + test_p[t1].pct_change().dropna()).cumprod()
    ax.plot(ec.index, ec.values, color=palette[idx % len(palette)],
            linewidth=0.7, alpha=0.30, zorder=2)

ax.plot(cum.index, cum.values, color=CRIMSON, linewidth=2.6,
        alpha=0.95, zorder=5, label="Portfolio")
ax.fill_between(cum.index, 1.0, cum.values,
                where=(cum.values >= 1.0), alpha=0.07, color=CRIMSON, zorder=3)
ax.fill_between(cum.index, 1.0, cum.values,
                where=(cum.values < 1.0), alpha=0.12, color=DARK_R, zorder=3)
ax.axhline(1.0, color=SPINE, linewidth=0.8, linestyle="--", zorder=1)

for yr in range(2007, 2025, 2):
    dt = pd.Timestamp(f"{yr}-01-01")
    if cum.index[0] <= dt <= cum.index[-1]:
        ax.axvline(dt, color="#15152a", linewidth=0.5, alpha=0.7, zorder=1)
        ax.text(dt, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.97,
                str(yr), color="#3a3a5a", fontsize=7, ha="center", va="bottom",
                transform=ax.get_xaxis_transform())

stats_text = (
    f"  Ann. Return    {ann*100:>6.2f}%\n"
    f"  Sharpe Ratio   {sharpe:>6.3f}\n"
    f"  Sortino Ratio  {sortino:>6.3f}\n"
    f"  Calmar Ratio   {calmar:>6.3f}\n"
    f"  Max Drawdown   {mdd*100:>6.2f}%\n"
    f"  Profit Factor  {pf:>6.3f}\n"
    f"  Total Return   {total*100:>6.2f}%"
)
ax.text(
    0.014, 0.975, stats_text,
    transform=ax.transAxes, fontsize=8.5, color=SILVER,
    verticalalignment="top", fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#07070e",
              edgecolor=CRIMSON, alpha=0.92),
    zorder=6,
)

ax.set_title(
    "Walk-Forward Pairs Trading  |  S&P 500  |  2007–2024\n"
    "Quarterly retraining  ·  Sub-industry pairs  ·  Inverse-vol allocation",
    color=TEXT, fontsize=13, pad=12, fontweight="bold",
)
ax.set_xlabel("Date", color=TEXT, fontsize=10)
ax.set_ylabel("Cumulative Return", color=TEXT, fontsize=10)
ax.tick_params(colors=TEXT, labelsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}x"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
for spine in ax.spines.values():
    spine.set_edgecolor(SPINE)
ax.grid(color=GRID, linewidth=0.5, alpha=0.8)

from matplotlib.lines import Line2D
legend_items = [
    Line2D([0], [0], color=CRIMSON, linewidth=2.5, label="Equal-risk portfolio"),
    Line2D([0], [0], color=SILVER, linewidth=0.8, alpha=0.4, label="Sample pair legs"),
]
ax.legend(
    handles=legend_items, loc="upper left", fontsize=8.5,
    framealpha=0.3, facecolor="#0a0a0f", edgecolor=CRIMSON,
    labelcolor=TEXT, bbox_to_anchor=(0.014, 0.60),
)

plt.tight_layout()
plt.savefig("equity_curve.png", dpi=150, facecolor=BG, bbox_inches="tight")
plt.show()
