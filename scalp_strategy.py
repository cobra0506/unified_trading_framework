#scalp_strategy.py
# scalp_strategy.py
from base_strategy import BaseStrategy, SignalType, MarketData, TradeSignal
import pandas as pd
import numpy as np
from typing import Dict, List
from utils import calculate_stoch_rsi, logger
import global_data


class ScalpStrategy(BaseStrategy):
    """
    High-frequency scalping strategy using:
    - EMA crossover
    - StochRSI for confirmation
    - VWAP as trend filter
    - ATR-based SL/TP
    """

    def __init__(self, config: Dict = None):
        default_config = {
            'ema_fast': 5,
            'ema_slow': 13,
            'stoch_rsi_period': 14,
            'stoch_entry_overbought': 80,
            'stoch_entry_oversold': 20,
            'atr_period': 14,
            'atr_mult_sl': 1.0,
            'atr_mult_tp': 2.0,
            'risk_percent': 0.02,
            'max_position_pct': 0.10,
            'required_intervals': ['1', '5'],
            'min_candles': 30
        }
        super().__init__({**default_config, **(config or {})})

    def get_required_intervals(self) -> List[str]:
        return self.config['required_intervals']

    def get_minimum_candles_needed(self) -> int:
        return self.config['min_candles']

    def analyze(self, market_data: MarketData) -> TradeSignal:
        try:
            candles = list(market_data.candles_by_interval.get('1', []))
            if len(candles) < self.get_minimum_candles_needed():
                return TradeSignal(SignalType.HOLD, market_data.symbol, market_data.current_price, reason="Insufficient candles")

            df = pd.DataFrame(candles)
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['volume'] = pd.to_numeric(df['volume'])

            # Indicators
            df['ema_fast'] = df['close'].ewm(span=self.config['ema_fast'], adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(span=self.config['ema_slow'], adjust=False).mean()

            # VWAP
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['cum_vp'] = (df['typical_price'] * df['volume']).cumsum()
            df['cum_vol'] = df['volume'].cumsum()
            df['vwap'] = df['cum_vp'] / df['cum_vol']

            # ATR
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift()).abs(),
                (df['low'] - df['close'].shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(self.config['atr_period']).mean().iloc[-1]

            # StochRSI
            stoch_series = calculate_stoch_rsi(candles[-30:], period=self.config['stoch_rsi_period'], return_series=True)
            if stoch_series is None or len(stoch_series) < 2:
                return TradeSignal(SignalType.HOLD, market_data.symbol, market_data.current_price, reason="Invalid StochRSI")

            curr_k = stoch_series[-1]
            prev_k = stoch_series[-2]

            price = market_data.current_price
            ema_fast = df['ema_fast'].iloc[-1]
            ema_slow = df['ema_slow'].iloc[-1]
            vwap = df['vwap'].iloc[-1]
            current_position = market_data.current_position

            # Entry long
            if ema_fast > ema_slow and prev_k < self.config['stoch_entry_oversold'] and curr_k > prev_k and price > vwap and current_position != 'long':
                sl = price - atr * self.config['atr_mult_sl']
                tp = price + atr * self.config['atr_mult_tp']
                amount_usd = self._calculate_position_size(price, sl)
                return TradeSignal(SignalType.OPEN_LONG, market_data.symbol, price, tp_price=tp, sl_price=sl, amount_usd=amount_usd, reason="Scalp long entry")

            # Entry short
            elif ema_fast < ema_slow and prev_k > self.config['stoch_entry_overbought'] and curr_k < prev_k and price < vwap and current_position != 'short':
                sl = price + atr * self.config['atr_mult_sl']
                tp = price - atr * self.config['atr_mult_tp']
                amount_usd = self._calculate_position_size(price, sl)
                return TradeSignal(SignalType.OPEN_SHORT, market_data.symbol, price, tp_price=tp, sl_price=sl, amount_usd=amount_usd, reason="Scalp short entry")

            # Exit long
            if current_position == 'long' and (price < ema_slow or curr_k > 80):
                return TradeSignal(SignalType.CLOSE_LONG, market_data.symbol, price, reason="Exit long: trend loss or stoch overbought")

            # Exit short
            if current_position == 'short' and (price > ema_slow or curr_k < 20):
                return TradeSignal(SignalType.CLOSE_SHORT, market_data.symbol, price, reason="Exit short: trend loss or stoch oversold")

            return TradeSignal(SignalType.HOLD, market_data.symbol, price, reason="No signal met")

        except Exception as e:
            logger.error(f"ScalpStrategy error on {market_data.symbol}: {str(e)}")
            return TradeSignal(SignalType.HOLD, market_data.symbol, market_data.current_price, reason="Strategy error")

    def _calculate_position_size(self, entry: float, stop: float) -> float:
        risk = self.config['risk_percent'] * global_data.current_balance
        per_unit = abs(entry - stop)
        size = risk / per_unit
        return min(size, global_data.current_balance * self.config['max_position_pct'])
