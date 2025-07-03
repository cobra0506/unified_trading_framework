# BETS — Backtrader-Based Accurate Strategy Testing System

### ⚡ Modular, Fast, Realistic Strategy Backtesting for Bybit Linear Perpetuals

---

## ✅ Key Features

- Multi-symbol, multi-timeframe backtesting
- Market-order-only simulation with:
  - Slippage
  - Execution delay
  - Leverage
  - Margin requirements
  - Liquidation logic
- Modular strategy support (plug in strategies easily)
- Realistic broker behavior (custom broker)
- Capital tracked across all trades (shared pool)
- Strategy optimization-ready
- Exportable results (CSV, Excel)

---

## 🧱 Project Structure

```bash
.
├── data/                  # Saved .parquet files (fetched OHLCV data)
│   ├── BTCUSDT_1h.parquet
│   └── ETHUSDT_1h.parquet
├── fetch_data.py          # Fetch + clean + save data from Bybit
├── strategy.py            # Modular strategies (SMA, RSI, etc.)
├── backtest.py            # Core backtest runner
└── results/               # (Optional) Folder for CSV/Excel exports
