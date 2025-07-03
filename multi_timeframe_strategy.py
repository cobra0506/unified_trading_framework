#multi_timeframe_strategy.py
from base_strategy import BaseStrategy, TradeSignal, SignalType, MarketData
from utils import calculate_stoch_rsi, calculate_atr, calculate_sma, logger
import global_data
import pandas as pd
from collections import deque

# Default configuration for the strategy
default_config = {
    'srsi_overbought': 75,
    'srsi_oversold': 25,
    'srsi_entry_oversold': 10,
    'srsi_entry_overbought': 75,
    'risk_percent': 0.02,
    'sl_atr_mult': 2.0,
    'tp_atr_mult': 4.0,
    'adx_threshold': 0,
    'sma_fast_period': 9,
    'sma_slow_period': 21,
    'required_intervals': ['1', '5', '15'],
    'min_candles': 20,
    'stochrsi_period': 14,
    'stochrsi_k': 3,
    'stochrsi_d': 3
}

class MultiTimeframeStrategy(BaseStrategy):
    """Multi-timeframe trading strategy using StochRSI, trend analysis, and ATR-based risk management"""
    
    def __init__(self, config: dict = None):
        # Merge provided config with default_config
        self.config = {**default_config, **(config or {})}
        super().__init__(self.config)
        # Strategy-specific parameters
        self.srsi_oversold = self.get_config_value('srsi_oversold')  # Exit long threshold
        self.srsi_overbought = self.get_config_value('srsi_overbought')  # Exit short threshold
        self.srsi_entry_oversold = self.get_config_value('srsi_entry_oversold')  # Entry long threshold
        self.srsi_entry_overbought = self.get_config_value('srsi_entry_overbought')  # Entry short threshold
        self.risk_percent = self.get_config_value('risk_percent')  # Risk 2% per trade
        self.sl_atr_mult = self.get_config_value('sl_atr_mult')  # Stop loss at 2x ATR
        self.tp_atr_mult = self.get_config_value('tp_atr_mult')  # Take profit at 4x ATR
        self.adx_threshold = self.get_config_value('adx_threshold')  # ADX filter (0 = disabled)
        self.sma_fast_period = self.get_config_value('sma_fast_period')  # 9-period SMA
        self.sma_slow_period = self.get_config_value('sma_slow_period')  # 21-period SMA
        self.stochrsi_period = self.get_config_value('stochrsi_period')  # StochRSI period
        self.stochrsi_k = self.get_config_value('stochrsi_k')  # StochRSI K smoothing
        self.stochrsi_d = self.get_config_value('stochrsi_d')  # StochRSI D smoothing
        logger.info(f"Initialized {self.name} with config: {self.config}")

    def get_required_intervals(self) -> list:
        """Return the required timeframe intervals"""
        return self.get_config_value('required_intervals')

    def get_minimum_candles_needed(self) -> int:
        """Return the minimum number of candles needed for analysis"""
        return self.get_config_value('min_candles')

    def analyze(self, market_data: MarketData) -> TradeSignal:
        symbol = market_data.symbol
        current_price = market_data.current_price
        current_position = market_data.current_position
        candles_by_interval = market_data.candles_by_interval

        # Validate data
        if not self.validate_data(market_data):
            logger.debug(f"{symbol}: Insufficient data for analysis")
            return TradeSignal(SignalType.HOLD, symbol, current_price, reason="Insufficient data")

        try:
            # Calculate StochRSI for all timeframes
            stoch_rsi_values = {}
            for interval in self.get_required_intervals():
                candles = candles_by_interval.get(interval, deque())
                if not isinstance(candles, deque) or len(candles) < self.get_minimum_candles_needed():
                    logger.debug(f"{symbol}/{interval}: Insufficient candles for StochRSI calculation")
                    return TradeSignal(SignalType.HOLD, symbol, current_price, reason="Insufficient candles")

                # Convert deque to list for calculations
                candles_list = list(candles)
                stoch_rsi_result = calculate_stoch_rsi(candles_list, self.stochrsi_period, self.stochrsi_k, self.stochrsi_d, return_series=True)
                if stoch_rsi_result is None or len(stoch_rsi_result) < 2:
                    logger.debug(f"{symbol}/{interval}: StochRSI calculation failed or insufficient data")
                    return TradeSignal(SignalType.HOLD, symbol, current_price, reason="StochRSI calculation failed")

                stoch_rsi_k = stoch_rsi_result[-1]
                prev_stoch_rsi_k = stoch_rsi_result[-2]
                stoch_rsi_values[interval] = {'k': stoch_rsi_k, 'prev_k': prev_stoch_rsi_k}

            # Check StochRSI direction (rising or falling)
            is_rising = {}
            is_falling = {}
            for interval in self.get_required_intervals():
                is_rising[interval] = stoch_rsi_values[interval]['k'] > stoch_rsi_values[interval]['prev_k']
                is_falling[interval] = stoch_rsi_values[interval]['k'] < stoch_rsi_values[interval]['prev_k']

            # Calculate trend on 15-minute chart
            candles_15m = candles_by_interval.get('15', deque())
            if not isinstance(candles_15m, deque) or len(candles_15m) < self.sma_slow_period:
                logger.debug(f"{symbol}: Insufficient candles for 15m SMA calculation")
                return TradeSignal(SignalType.HOLD, symbol, current_price, reason="Insufficient candles for 15m SMA")

            # Convert deque to list for calculations
            candles_15m_list = list(candles_15m)
            closes_15m = [c['close'] for c in candles_15m_list]
            sma_fast = calculate_sma(closes_15m, self.sma_fast_period)
            sma_slow = calculate_sma(closes_15m, self.sma_slow_period)
            if sma_fast is None or sma_slow is None:
                logger.debug(f"{symbol}: SMA calculation failed on 15m")
                return TradeSignal(SignalType.HOLD, symbol, current_price, reason="SMA calculation failed")

            is_bullish = sma_fast > sma_slow
            is_bearish = sma_fast < sma_slow

            # Calculate ATR for position sizing and SL/TP
            atr = calculate_atr(candles_15m_list, period=14)
            if atr is None:
                logger.debug(f"{symbol}: ATR calculation failed")
                return TradeSignal(SignalType.HOLD, symbol, current_price, reason="ATR calculation failed")

            # Risk management calculations
            balance = global_data.current_balance
            risk_amount = balance * self.risk_percent
            amount_usd = min(risk_amount, global_data.Amount)  # Cap at global Amount
            sl_distance = atr * self.sl_atr_mult
            tp_distance = atr * self.tp_atr_mult

            # Exit conditions
            if current_position == 'long':
                if stoch_rsi_values['1']['k'] > self.srsi_overbought or is_bearish:
                    logger.info(f"{symbol}: Closing long position - StochRSI={stoch_rsi_values['1']['k']:.2f}, Bearish={is_bearish}")
                    return TradeSignal(SignalType.CLOSE_LONG, symbol, current_price, reason="Overbought or bearish trend")
                    
            elif current_position == 'short':
                if stoch_rsi_values['1']['k'] < self.srsi_oversold or is_bullish:
                    logger.info(f"{symbol}: Closing short position - StochRSI={stoch_rsi_values['1']['k']:.2f}, Bullish={is_bullish}")
                    return TradeSignal(SignalType.CLOSE_SHORT, symbol, current_price, reason="Oversold or bullish trend")

            # Entry conditions for long
            if (all(stoch_rsi_values[tf]['k'] < self.srsi_entry_oversold for tf in self.get_required_intervals()) and
                all(is_rising[tf] for tf in self.get_required_intervals()) and
                is_bullish):
                sl_price = current_price - sl_distance
                tp_price = current_price + tp_distance
                logger.info(f"{symbol}: Long signal - StochRSI={stoch_rsi_values['1']['k']:.2f}, Bullish trend")
                return TradeSignal(
                    SignalType.OPEN_LONG, symbol, current_price, tp_price, sl_price, amount_usd,
                    reason="Oversold on all timeframes with bullish trend"
                )

            # Entry conditions for short
            if (all(stoch_rsi_values[tf]['k'] > self.srsi_entry_overbought for tf in self.get_required_intervals()) and
                all(is_falling[tf] for tf in self.get_required_intervals()) and
                is_bearish):
                sl_price = current_price + sl_distance
                tp_price = current_price - tp_distance
                logger.info(f"{symbol}: Short signal - StochRSI={stoch_rsi_values['1']['k']:.2f}, Bearish trend")
                return TradeSignal(
                    SignalType.OPEN_SHORT, symbol, current_price, tp_price, sl_price, amount_usd,
                    reason="Overbought on all timeframes with bearish trend"
                )

            # Default: hold position
            return TradeSignal(SignalType.HOLD, symbol, current_price, reason="No entry/exit conditions met")

        except Exception as e:
            logger.error(f"{symbol}: Error in analysis - {str(e)}")
            return TradeSignal(SignalType.HOLD, symbol, current_price, reason=f"Analysis error: {str(e)}")