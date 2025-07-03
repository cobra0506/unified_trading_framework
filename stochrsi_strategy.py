#stockrsi_strategy.py
import pandas as pd
from typing import List, Dict
from base_strategy import BaseStrategy, MarketData, TradeSignal, SignalType
from utils import calculate_stoch_rsi, calculate_sma, calculate_adx, calculate_atr, logger
from collections import deque

class StochRSIStrategy(BaseStrategy):
    """
    StochRSI-based trading strategy with SMA trend confirmation
    """
    
    def __init__(self, config: Dict = None):
        # Default configuration
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
            'min_candles': 50
        }
        
        # Merge user config with defaults
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
    
    def get_required_intervals(self) -> List[str]:
        return self.config['required_intervals']
    
    def get_minimum_candles_needed(self) -> int:
        return self.config['min_candles']
    
    def analyze(self, market_data: MarketData) -> TradeSignal:
        """
        Main strategy logic - analyze market data and return trade signal
        """
        try:
            # Validate input data
            if not self.validate_data(market_data):
                logger.warning(f"{market_data.symbol}: Insufficient data for strategy")
                return TradeSignal(SignalType.HOLD, market_data.symbol, market_data.current_price, reason="Insufficient data")
            
            # Calculate StochRSI for all required intervals
            stoch_k = self._calculate_stoch_rsi_all_intervals(market_data)
            if not stoch_k:
                return TradeSignal(SignalType.HOLD, market_data.symbol, market_data.current_price, reason="StochRSI calculation failed")
            
            # Calculate trend indicators (SMA & ADX on 15m)
            trend_data = self._calculate_trend_indicators(market_data)
            if not trend_data:
                return TradeSignal(SignalType.HOLD, market_data.symbol, market_data.current_price, reason="Trend calculation failed")
            
            # Log current state
            self._log_indicator_values(market_data.symbol, stoch_k, trend_data)
            
            # Generate trade signal based on strategy rules
            signal = self._generate_signal(market_data, stoch_k, trend_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in StochRSI strategy for {market_data.symbol}: {e}")
            return TradeSignal(SignalType.HOLD, market_data.symbol, market_data.current_price, reason=f"Strategy error: {e}")
    
    def _calculate_stoch_rsi_all_intervals(self, market_data: MarketData) -> Dict:
        """Calculate StochRSI for all required intervals"""
        stoch_k = {}
        
        for interval in self.get_required_intervals():
            candles_raw = market_data.candles_by_interval.get(interval)
            if candles_raw is None:
                logger.error(f"{market_data.symbol}: No data for {interval}m interval")
                return None
            
            # Convert deque to list if needed
            if isinstance(candles_raw, deque):
                candles_raw = list(candles_raw)
            
            if not isinstance(candles_raw, list):
                logger.error(f"{market_data.symbol}: Expected list for {interval}m candles, got {type(candles_raw)}")
                return None
            
            # Use last 50 candles for calculation
            candles = candles_raw[-50:]
            
            if len(candles) < 20:
                logger.warning(f"{market_data.symbol}: Not enough {interval}m candles for StochRSI (len={len(candles)})")
                return None
            
            k_series = calculate_stoch_rsi(candles, return_series=True)
            
            if k_series is None or len(k_series) < 2:
                logger.warning(f"{market_data.symbol}: {interval}m StochRSI series too short")
                return None
            
            prev_k = k_series[-2]
            curr_k = k_series[-1]
            
            if pd.isna(prev_k) or pd.isna(curr_k):
                logger.warning(f"{market_data.symbol}: {interval}m StochRSI values contain NaNs")
                return None
            
            stoch_k[interval] = {
                "prev": prev_k,
                "curr": curr_k
            }
        
        return stoch_k
    
    def _calculate_trend_indicators(self, market_data: MarketData) -> Dict:
        """Calculate SMA and ADX for trend analysis"""
        candles_15m = market_data.candles_by_interval.get('15', [])
        if len(candles_15m) < 21:
            logger.warning(f"{market_data.symbol}: Not enough 15m candles for SMA/ADX")
            return None
        
        close_15m = [c['close'] for c in candles_15m]
        fast_sma = calculate_sma(close_15m, self.config['sma_fast_period'])
        slow_sma = calculate_sma(close_15m, self.config['sma_slow_period'])
        adx = calculate_adx(candles_15m)
        
        if fast_sma is None or slow_sma is None or adx is None:
            logger.warning(f"{market_data.symbol}: Could not calculate SMA or ADX")
            return None
        
        return {
            'fast_sma': fast_sma,
            'slow_sma': slow_sma,
            'adx': adx,
            'trend_is_bullish': fast_sma > slow_sma and adx > self.config['adx_threshold'],
            'trend_is_bearish': fast_sma < slow_sma and adx > self.config['adx_threshold']
        }
    
    def _generate_signal(self, market_data: MarketData, stoch_k: Dict, trend_data: Dict) -> TradeSignal:
        """Generate trade signal based on strategy rules"""
        symbol = market_data.symbol
        current_price = market_data.current_price
        current_position = market_data.current_position
        
        # Check for LONG entry conditions
        long_entry_condition = all(
            stoch_k[i]['curr'] < self.config['srsi_entry_oversold'] and 
            stoch_k[i]['curr'] > stoch_k[i]['prev'] 
            for i in self.get_required_intervals()
        )
        
        # Check for SHORT entry conditions  
        short_entry_condition = all(
            stoch_k[i]['curr'] > self.config['srsi_entry_overbought'] and 
            stoch_k[i]['curr'] < stoch_k[i]['prev'] 
            for i in self.get_required_intervals()
        )
        
        # OPEN LONG
        if long_entry_condition and current_position != 'long':
            if not trend_data['trend_is_bullish']:
                return TradeSignal(SignalType.HOLD, symbol, current_price, reason="Trend not bullish for long entry")
            
            # Calculate position sizing and TP/SL
            tp_price, sl_price, amount_usd = self._calculate_position_params(market_data, 'long')
            if tp_price is None:
                return TradeSignal(SignalType.HOLD, symbol, current_price, reason="Could not calculate position parameters")
            
            return TradeSignal(
                SignalType.OPEN_LONG, 
                symbol, 
                current_price,
                tp_price=tp_price,
                sl_price=sl_price,
                amount_usd=amount_usd,
                reason="StochRSI oversold bounce with bullish trend"
            )
        
        # OPEN SHORT
        elif short_entry_condition and current_position != 'short':
            if not trend_data['trend_is_bearish']:
                return TradeSignal(SignalType.HOLD, symbol, current_price, reason="Trend not bearish for short entry")
            
            # Calculate position sizing and TP/SL
            tp_price, sl_price, amount_usd = self._calculate_position_params(market_data, 'short')
            if tp_price is None:
                return TradeSignal(SignalType.HOLD, symbol, current_price, reason="Could not calculate position parameters")
            
            return TradeSignal(
                SignalType.OPEN_SHORT, 
                symbol, 
                current_price,
                tp_price=tp_price,
                sl_price=sl_price,
                amount_usd=amount_usd,
                reason="StochRSI overbought rejection with bearish trend"
            )
        
        # CLOSE LONG
        elif current_position == 'long' and (
            stoch_k['1']['curr'] > self.config['srsi_overbought'] or 
            trend_data['trend_is_bearish']
        ):
            return TradeSignal(
                SignalType.CLOSE_LONG, 
                symbol, 
                current_price,
                reason="StochRSI overbought or trend turned bearish"
            )
        
        # CLOSE SHORT
        elif current_position == 'short' and (
            stoch_k['1']['curr'] < self.config['srsi_oversold'] or 
            trend_data['trend_is_bullish']
        ):
            return TradeSignal(
                SignalType.CLOSE_SHORT, 
                symbol, 
                current_price,
                reason="StochRSI oversold or trend turned bullish"
            )
        
        # HOLD
        return TradeSignal(SignalType.HOLD, symbol, current_price, reason="No signal conditions met")
    
    def _calculate_position_params(self, market_data: MarketData, direction: str):
        """Calculate TP, SL, and position size"""
        try:
            candles_15m = market_data.candles_by_interval.get('15', [])
            atr = calculate_atr(candles_15m)
            if atr is None:
                logger.warning(f"{market_data.symbol}: Could not calculate ATR")
                return None, None, None
            
            current_price = market_data.current_price
            
            if direction == 'long':
                sl_price = current_price - self.config['sl_atr_mult'] * atr
                tp_price = current_price + self.config['tp_atr_mult'] * atr
                
                # Validate
                if tp_price <= current_price or sl_price >= current_price:
                    logger.warning(f"{market_data.symbol}: Invalid TP/SL for long")
                    return None, None, None
                    
            else:  # short
                sl_price = current_price + self.config['sl_atr_mult'] * atr
                tp_price = current_price - self.config['tp_atr_mult'] * atr
                
                # Validate
                if tp_price >= current_price or sl_price <= current_price:
                    logger.warning(f"{market_data.symbol}: Invalid TP/SL for short")
                    return None, None, None
            
            # Calculate position size based on risk
            # This would need to be imported from global_data or passed as parameter
            # For now, using a placeholder
            balance = 10000  # This should come from global_data.current_balance
            sl_pct = atr / current_price
            position_risk = balance * self.config['risk_percent']
            amount_usd = position_risk / sl_pct
            
            return tp_price, sl_price, amount_usd
            
        except Exception as e:
            logger.error(f"Error calculating position params for {market_data.symbol}: {e}")
            return None, None, None
    
    def _log_indicator_values(self, symbol: str, stoch_k: Dict, trend_data: Dict):
        """Log current indicator values for debugging"""
        logger.info(
            f"{symbol}: StochRSI %K - "
            f"1m: curr={stoch_k['1']['curr']:.2f}, prev={stoch_k['1']['prev']:.2f}; "
            f"5m: curr={stoch_k['5']['curr']:.2f}, prev={stoch_k['5']['prev']:.2f}; "
            f"15m: curr={stoch_k['15']['curr']:.2f}, prev={stoch_k['15']['prev']:.2f}"
        )
        
        logger.info(
            f"{symbol}: 15m Fast SMA: {trend_data['fast_sma']:.2f}, "
            f"Slow SMA: {trend_data['slow_sma']:.2f}, ADX: {trend_data['adx']:.2f}"
        )