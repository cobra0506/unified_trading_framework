#EMAVWAPScalpingStrategy1.py
from base_strategy import BaseStrategy, TradeSignal, SignalType, MarketData
from utils import calculate_stoch_rsi, calculate_atr, logger
import pandas as pd
import numpy as np
from global_data import RISK_PERCENT, SL_ATR_MULT, TP_ATR_MULT, MAX_OPEN_POSITIONS, current_balance

class EMAVWAPScalpingStrategy(BaseStrategy):
    def __init__(self, config: dict = None):
        default_config = {
            'ema_fast_period': 5,
            'ema_slow_period': 13,
            'stoch_rsi_period': 14,
            'stoch_rsi_k': 3,
            'stoch_rsi_d': 3,
            'atr_period': 14,
            'risk_percent': 0.02,
            'max_position_percent': 0.1,
            'srsi_overbought': 80,
            'srsi_oversold': 20,
            'account_balance': current_balance,
            'min_sl_distance': 0.01,
            'min_price_movement': 0.001,  # Minimum price movement filter
            'confirmation_period': 3,    # Confirmation period
        }
        super().__init__(config=default_config if config is None else {**default_config, **config})
        self.last_signal_price = None

    def analyze(self, market_data: MarketData) -> TradeSignal:
        symbol = market_data.symbol
        candles = market_data.candles_by_interval.get('1', [])
        
        if not self.validate_data(market_data):
            return TradeSignal(signal_type=SignalType.HOLD, symbol=symbol, price=market_data.current_price, reason="Insufficient data")

        try:
            ema_fast = self.calculate_ema(candles, self.ema_fast_period)
            ema_slow = self.calculate_ema(candles, self.ema_slow_period)
            stoch_rsi_result = calculate_stoch_rsi(candles, self.stoch_rsi_period, self.stoch_rsi_k, self.stoch_rsi_d, return_series=True)
            vwap = self.calculate_vwap(candles)
            atr = calculate_atr(candles, self.atr_period)

            if any(x is None for x in [ema_fast, ema_slow, stoch_rsi_result, vwap, atr]):
                return TradeSignal(signal_type=SignalType.HOLD, symbol=symbol, price=market_data.current_price, reason="Indicator calculation failed")

            current_price = market_data.current_price
            current_position = market_data.current_position

            # Check for minimum price movement
            if self.last_signal_price and abs(current_price - self.last_signal_price) < self.min_price_movement:
                return TradeSignal(signal_type=SignalType.HOLD, symbol=symbol, price=current_price, reason="Price movement too small")

            # Confirm signal over multiple periods
            if len(stoch_rsi_result) < self.confirmation_period:
                return TradeSignal(signal_type=SignalType.HOLD, symbol=symbol, price=current_price, reason="Not enough data for confirmation")

            stoch_rsi_k = stoch_rsi_result[-1]
            prev_stoch_rsi_k = stoch_rsi_result[-2]

            # Exit conditions
            if current_position == 'long':
                if current_price < ema_slow or stoch_rsi_k > self.srsi_overbought:
                    return TradeSignal(signal_type=SignalType.CLOSE_LONG, symbol=symbol, price=current_price, reason="Long exit: Price below slow EMA or StochRSI overbought")
            elif current_position == 'short':
                if current_price > ema_slow or stoch_rsi_k < self.srsi_oversold:
                    return TradeSignal(signal_type=SignalType.CLOSE_SHORT, symbol=symbol, price=current_price, reason="Short exit: Price above slow EMA or StochRSI oversold")

            # Entry conditions for long
            if (ema_fast > ema_slow and
                prev_stoch_rsi_k < self.srsi_oversold and stoch_rsi_k > prev_stoch_rsi_k and
                current_price > vwap and
                current_position is None):
                sl_price = current_price - (atr * SL_ATR_MULT)
                tp_price = current_price + (atr * TP_ATR_MULT)
                self.last_signal_price = current_price
                return TradeSignal(signal_type=SignalType.OPEN_LONG, symbol=symbol, price=current_price, tp_price=tp_price, sl_price=sl_price, reason="Long entry: EMA crossover, StochRSI rising from oversold, above VWAP")

            # Entry conditions for short
            if (ema_fast < ema_slow and
                prev_stoch_rsi_k > self.srsi_overbought and stoch_rsi_k < prev_stoch_rsi_k and
                current_price < vwap and
                current_position is None):
                sl_price = current_price + (atr * SL_ATR_MULT)
                tp_price = current_price - (atr * TP_ATR_MULT)
                self.last_signal_price = current_price
                return TradeSignal(signal_type=SignalType.OPEN_SHORT, symbol=symbol, price=current_price, tp_price=tp_price, sl_price=sl_price, reason="Short entry: EMA crossover, StochRSI falling from overbought, below VWAP")

            return TradeSignal(signal_type=SignalType.HOLD, symbol=symbol, price=current_price, reason="No entry/exit conditions met")

        except Exception as e:
            logger.error(f"{symbol}: Error in EMAVWAPScalpingStrategy analysis: {e}")
            return TradeSignal(signal_type=SignalType.HOLD, symbol=symbol, price=market_data.current_price, reason=f"Analysis error: {str(e)}")

'''class EMAVWAPScalpingStrategy(BaseStrategy):
    def __init__(self, config: dict = None):
        # Default configuration
        default_config = {
            'ema_fast_period': 5,
            'ema_slow_period': 13,
            'stoch_rsi_period': 14,
            'stoch_rsi_k': 3,
            'stoch_rsi_d': 3,
            'atr_period': 14,
            'risk_percent': 0.02,  # 2% risk per trade
            'max_position_percent': 0.1,  # 10% of account max
            'srsi_overbought': 80,
            'srsi_oversold': 20,
            'account_balance': current_balance,  # Use dynamic balance from global_data
            'min_sl_distance': 0.01  # Minimum stop-loss distance in USD
        }
        # Merge default config with any provided config, prioritizing provided values
        super().__init__(config=default_config if config is None else {**default_config, **config})
        self.ema_fast_period = self.get_config_value('ema_fast_period')
        self.ema_slow_period = self.get_config_value('ema_slow_period')
        self.stoch_rsi_period = self.get_config_value('stoch_rsi_period')
        self.stoch_rsi_k = self.get_config_value('stoch_rsi_k')
        self.stoch_rsi_d = self.get_config_value('stoch_rsi_d')
        self.atr_period = self.get_config_value('atr_period')
        self.risk_percent = self.get_config_value('risk_percent')
        self.max_position_percent = self.get_config_value('max_position_percent')
        self.srsi_overbought = self.get_config_value('srsi_overbought')
        self.srsi_oversold = self.get_config_value('srsi_oversold')
        self.min_sl_distance = self.get_config_value('min_sl_distance')


    def get_required_intervals(self) -> list:
        return ['1']  # 1-minute candles for scalping

    def get_minimum_candles_needed(self) -> int:
        return max(self.ema_slow_period, self.stoch_rsi_period + self.stoch_rsi_k + self.stoch_rsi_d, self.atr_period) + 2

    def calculate_ema(self, candles, period):
        if len(candles) < period:
            return None
        closes = [float(c['close']) for c in candles]
        return pd.Series(closes).ewm(span=period, adjust=False).mean().iloc[-1]

    def calculate_vwap(self, candles):
        if len(candles) < 1:
            return None
        df = pd.DataFrame(candles)
        for col in ['close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else None

    def analyze(self, market_data: MarketData) -> TradeSignal:
        symbol = market_data.symbol
        candles = market_data.candles_by_interval.get('1', [])
        
        if not self.validate_data(market_data):
            logger.debug(f"{symbol}: Insufficient data for EMAVWAPScalpingStrategy")
            return TradeSignal(signal_type=SignalType.HOLD, symbol=symbol, price=market_data.current_price, reason="Insufficient data")

        try:
            # Calculate indicators
            ema_fast = self.calculate_ema(candles, self.ema_fast_period)
            ema_slow = self.calculate_ema(candles, self.ema_slow_period)
            
            # Calculate StochRSI for current and previous periods
            stoch_rsi_result = calculate_stoch_rsi(
                candles, self.stoch_rsi_period, self.stoch_rsi_k, self.stoch_rsi_d, return_series=True
            )
            if stoch_rsi_result is None:
                logger.debug(f"{symbol}: StochRSI calculation returned None")
                return TradeSignal(signal_type=SignalType.HOLD, symbol=symbol, price=market_data.current_price, reason="StochRSI calculation failed")
            stoch_rsi_k = stoch_rsi_result[-1] if len(stoch_rsi_result) > 0 else None
            prev_stoch_rsi_k = stoch_rsi_result[-2] if len(stoch_rsi_result) > 1 else stoch_rsi_k
            
            vwap = self.calculate_vwap(candles)
            atr = calculate_atr(candles, self.atr_period)
            
            if any(x is None for x in [ema_fast, ema_slow, stoch_rsi_k, prev_stoch_rsi_k, vwap, atr]):
                logger.debug(f"{symbol}: One or more indicators are None")
                return TradeSignal(signal_type=SignalType.HOLD, symbol=symbol, price=market_data.current_price, reason="Indicator calculation failed")

            current_price = market_data.current_price
            current_position = market_data.current_position

            # Calculate position size based on risk
            account_balance = self.get_config_value('account_balance')
            risk_amount = account_balance * self.risk_percent
            max_position_amount = account_balance * self.max_position_percent
            stop_loss_distance = max(atr * SL_ATR_MULT, self.min_sl_distance)
            amount_usd = min(risk_amount / stop_loss_distance, max_position_amount) if stop_loss_distance > 0 else max_position_amount

            # Exit conditions
            if current_position == 'long':
                if current_price < ema_slow or stoch_rsi_k > self.srsi_overbought:
                    return TradeSignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=symbol,
                        price=current_price,
                        reason="Long exit: Price below slow EMA or StochRSI overbought"
                    )
            elif current_position == 'short':
                if current_price > ema_slow or stoch_rsi_k < self.srsi_oversold:
                    return TradeSignal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=symbol,
                        price=current_price,
                        reason="Short exit: Price above slow EMA or StochRSI oversold"
                    )

            # Entry conditions for long
            if (ema_fast > ema_slow and  # Uptrend
                prev_stoch_rsi_k < self.srsi_oversold and stoch_rsi_k > prev_stoch_rsi_k and  # StochRSI rising from oversold
                current_price > vwap and  # Above VWAP
                current_position is None):  # No existing position
                sl_price = current_price - stop_loss_distance
                tp_price = current_price + (stop_loss_distance * TP_ATR_MULT)

                if sl_price <= 0:
                    logger.warning(f"{symbol}: Invalid stop-loss price {sl_price}")
                    return TradeSignal(signal_type=SignalType.HOLD, symbol=symbol, price=current_price, reason="Invalid stop-loss")

                return TradeSignal(
                    signal_type=SignalType.OPEN_LONG,
                    symbol=symbol,
                    price=current_price,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    amount_usd=amount_usd,
                    reason="Long entry: EMA crossover, StochRSI rising from oversold, above VWAP"
                )

            # Entry conditions for short
            if (ema_fast < ema_slow and  # Downtrend
                prev_stoch_rsi_k > self.srsi_overbought and stoch_rsi_k < prev_stoch_rsi_k and  # StochRSI falling from overbought
                current_price < vwap and  # Below VWAP
                current_position is None):  # No existing position
                sl_price = current_price + stop_loss_distance
                tp_price = current_price - (stop_loss_distance * TP_ATR_MULT)
                return TradeSignal(
                    signal_type=SignalType.OPEN_SHORT,
                    symbol=symbol,
                    price=current_price,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    amount_usd=amount_usd,
                    reason="Short entry: EMA crossover, StochRSI falling from overbought, below VWAP"
                )

            return TradeSignal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=current_price,
                reason="No entry/exit conditions met"
            )

        except Exception as e:
            logger.error(f"{symbol}: Error in EMAVWAPScalpingStrategy analysis: {e}")
            return TradeSignal(signal_type=SignalType.HOLD, symbol=symbol, price=market_data.current_price, reason=f"Analysis error: {str(e)}")'''