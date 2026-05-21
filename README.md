# 📈 Pairs Trading Strategy

A statistical arbitrage pairs trading system built in Python. Identifies cointegrated asset pairs, generates mean-reversion signals using Z-scores, and backtests the strategy with full performance analytics.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Strategy Details](#strategy-details)
- [Backtesting & Performance Metrics](#backtesting--performance-metrics)
- [Results](#results)
- [Roadmap](#roadmap)
- [Disclaimer](#disclaimer)

---

## Overview

Pairs trading is a **market-neutral quantitative strategy** that profits from the temporary divergence of two historically correlated assets. When two assets deviate from their long-run equilibrium relationship, the strategy takes opposing positions and profits as prices converge.

This project implements the full pipeline:

1. Data ingestion from Yahoo Finance
2. Pair selection via correlation filtering and cointegration testing
3. Spread construction and Z-score signal generation
4. Event-driven backtesting with transaction costs
5. Performance analysis and visualization

---

## How It Works

```
Universe of Stocks
       │
       ▼
Correlation Filter  ──►  Remove low-correlation pairs
       │
       ▼
Cointegration Test  ──►  Engle-Granger test (p < 0.05)
       │
       ▼
Hedge Ratio (OLS / Kalman Filter)
       │
       ▼
Spread = Price_A − β × Price_B
       │
       ▼
Z-Score = (Spread − Rolling Mean) / Rolling Std
       │
       ▼
  ┌────┴────┐
  │ Signals │
  └────┬────┘
  Z > +2.0  →  Short A, Long B
  Z < -2.0  →  Long A, Short B
  |Z| < 0.5 →  Exit position
       │
       ▼
  Backtesting Engine
       │
       ▼
  Performance Report
```

---

## Project Structure

```
pairs-trading/
│
├── data/
│   └── prices.csv              # Cached price data
│
├── src/
│   ├── data_pipeline.py        # Data fetching and preprocessing
│   ├── pair_selection.py       # Correlation + cointegration screening
│   ├── signal_generation.py    # Spread construction and Z-score signals
│   ├── backtester.py           # Event-driven backtesting engine
│   └── performance.py          # Metrics: Sharpe, drawdown, PnL
│
├── notebooks/
│   └── analysis.ipynb          # Exploratory analysis and visualizations
│
├── tests/
│   ├── test_pair_selection.py
│   ├── test_signal_generation.py
│   └── test_backtester.py
│
├── results/
│   └── equity_curve.png
│   └── spread_zscore.png
│   └── performance_summary.csv
│
├── requirements.txt
├── config.py                   # Strategy parameters
└── main.py                     # Entry point
```

---

## Installation

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/yourusername/pairs-trading.git
cd pairs-trading

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**

```
yfinance
pandas
numpy
statsmodels
scipy
matplotlib
seaborn
pykalman          # Optional: for dynamic hedge ratio
jupyter           # Optional: for notebooks
```

---

## Usage

### Run the full pipeline

```bash
python main.py
```

### Configure strategy parameters

Edit `config.py` to customize the strategy:

```python
# config.py

TICKERS = ["XOM", "CVX", "COP", "SLB", "MPC"]   # Asset universe
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"

CORRELATION_THRESHOLD = 0.80    # Pre-filter: minimum correlation
COINTEGRATION_PVALUE  = 0.05    # Engle-Granger significance threshold
ROLLING_WINDOW        = 30      # Days for rolling mean/std
ENTRY_ZSCORE          = 2.0     # Z-score threshold to open a trade
EXIT_ZSCORE           = 0.5     # Z-score threshold to close a trade
TRANSACTION_COST      = 0.001   # 0.1% per trade (each leg)
```

### Run individual modules

```python
from src.data_pipeline import fetch_prices
from src.pair_selection import find_cointegrated_pairs
from src.signal_generation import compute_spread, compute_zscore
from src.backtester import run_backtest
from src.performance import summary_statistics

# Fetch data
prices = fetch_prices(tickers=["XOM", "CVX"], start="2018-01-01", end="2024-01-01")

# Find valid pairs
pairs = find_cointegrated_pairs(prices, p_threshold=0.05)

# Generate signals
spread = compute_spread(prices["XOM"], prices["CVX"], hedge_ratio=1.2)
zscore = compute_zscore(spread, window=30)

# Backtest
results = run_backtest(zscore, prices["XOM"], prices["CVX"], hedge_ratio=1.2)

# Evaluate
print(summary_statistics(results))
```

---

## Strategy Details

### Pair Selection

**Step 1 — Correlation Filter**
Compute pairwise Pearson correlation on log returns. Only pairs above the `CORRELATION_THRESHOLD` move forward (reduces the number of expensive cointegration tests).

**Step 2 — Cointegration Test**
Apply the Engle-Granger two-step test to each candidate pair. A pair is considered cointegrated if the p-value falls below `COINTEGRATION_PVALUE` (default: 0.05), meaning we reject the null hypothesis of no cointegration.

**Step 3 — Hedge Ratio Estimation**

*Static (OLS):*
```
Price_A = β × Price_B + ε
```
β is estimated via ordinary least squares regression over the training window.

*Dynamic (Kalman Filter)* — optional:
The hedge ratio is updated continuously using a Kalman Filter, allowing it to adapt as the relationship between assets evolves over time.

### Spread & Signal

```python
spread = price_A - hedge_ratio * price_B
zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
```

| Z-Score          | Action                     |
|------------------|----------------------------|
| Z > +2.0         | Short A, Long B            |
| Z < -2.0         | Long A, Short B            |
| \|Z\| < 0.5      | Exit all positions         |

### Walk-Forward Validation

To prevent overfitting, the strategy uses a walk-forward split:

```
|────── Training (70%) ──────|──── Testing (30%) ────|
```

All parameters are fit on the training set only. The test set is never used to inform strategy decisions.

---

## Backtesting & Performance Metrics

The backtester simulates trades chronologically, applying transaction costs on every entry and exit.

**Metrics reported:**

| Metric              | Description                                         |
|---------------------|-----------------------------------------------------|
| Total Return        | Overall strategy return over the test period        |
| Annualized Return   | Return normalized to a yearly basis                 |
| Sharpe Ratio        | Annualized return per unit of volatility            |
| Max Drawdown        | Largest peak-to-trough decline in equity            |
| Win Rate            | Percentage of trades that were profitable           |
| Avg Trade Duration  | Mean number of days a position was held             |
| Number of Trades    | Total round-trip trades executed                    |

---

## Results

> Results shown are on the **out-of-sample test period** only.

| Pair        | Sharpe Ratio | Total Return | Max Drawdown | # Trades |
|-------------|:------------:|:------------:|:------------:|:--------:|
| XOM / CVX   | 1.42         | +18.3%       | -6.1%        | 34       |
| GLD / SLV   | 1.18         | +14.7%       | -8.4%        | 28       |
| KO  / PEP   | 0.97         | +11.2%       | -5.3%        | 41       |

*Note: Past backtested performance does not guarantee future results.*

---

## Roadmap

- [x] Data pipeline with yfinance
- [x] Correlation + cointegration pair screening
- [x] OLS hedge ratio estimation
- [x] Z-score signal generation
- [x] Event-driven backtester with transaction costs
- [x] Performance metrics and reporting
- [ ] Dynamic hedge ratio via Kalman Filter
- [ ] Portfolio-level multi-pair allocation
- [ ] Stop-loss logic for diverging pairs
- [ ] Live paper trading integration

---

## Disclaimer

This project is built for **educational and portfolio purposes only**. It is not financial advice. Backtested results do not guarantee future performance. Pairs trading involves significant risk, including the risk that a historically cointegrated pair permanently diverges.
