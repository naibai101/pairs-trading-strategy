# Pairs Trading Strategy

A statistical arbitrage system built in Python that identifies cointegrated asset pairs and trades their mean-reverting spread.

---

## Overview

Pairs trading is a market-neutral quantitative strategy that profits from the temporary divergence of two historically correlated assets. When two assets drift apart from their long-run equilibrium, the strategy takes opposing positions and profits as prices converge back together.

---

## How It Works

1. **Pair Selection** — Screen a universe of stocks for highly correlated pairs, then apply an Engle-Granger cointegration test to confirm a stable long-run relationship.

2. **Spread Construction** — Compute a hedge ratio via OLS regression and construct the spread between the two assets. The spread is expected to be mean-reverting.

3. **Signal Generation** — Calculate a rolling Z-score of the spread. A high Z-score signals the pair has diverged and a trade should be opened; a low Z-score signals convergence and the trade should be closed.

4. **Backtesting** — Simulate the strategy historically, accounting for transaction costs, and evaluate performance using standard metrics.

---

## Signal Logic

| Z-Score     | Action                  |
|-------------|-------------------------|
| Z > +2.0    | Short Asset A, Long Asset B  |
| Z < -2.0    | Long Asset A, Short Asset B  |
| \|Z\| < 0.5 | Exit position           |

---

## Project Structure

```
pairs-trading/
├── pairs_trading.py   # Everything
├── requirements.txt
└── README.md
```

---

## Performance Metrics

- Total Return
- Annualized Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Number of Trades

---

## Tech Stack

- **Data** — `yfinance`, `pandas`
- **Statistics** — `statsmodels`, `scipy`
- **Numerics** — `numpy`
- **Visualization** — `matplotlib`

---

## Roadmap

- [ ] Data pipeline
- [ ] Pair selection (correlation + cointegration)
- [ ] Spread construction and Z-score signals
- [ ] Backtesting engine
- [ ] Performance reporting and visualization
- [ ] Dynamic hedge ratio via Kalman Filter

---

