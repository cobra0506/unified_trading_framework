from typing import List, Dict
from base_strategy import BaseStrategy, MarketData, TradeSignal, SignalType
from utils import logger

class SimpleRSIStrategy(BaseStrategy):
    """
    Example of a simple RSI-based strategy to demonstrate how easy it is 
    to create new strategies with the new framework
    """
    
    def __init__(self, config: Dict = None):
        # Default configuration
        default_config = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'required_intervals': ['5'],  # Only need 5m data
            'min_candles': 20
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
    
    def get_required_intervals(self) -> List[str]:
        return self.config['required_intervals']
    
    def get_minimum_candles_needed(self) -> int:
        return self.config['min_candles']
    
    def analyze(self, market_data: MarketData) -> TradeSignal:
        """
        Simple RSI strategy:
        - Buy when RSI < oversold and previous RSI was also < oversold (confirmation)
        - Sell when RSI > overbought and previous RSI was also > overbought
        - Close positions on opposite signals
        """
        try:
            if not self.validate_data(market_data):
                return TradeSignal(SignalType.HOLD, market_data.symbol, market_data.current_price, 
                                 reason="Insufficient data")
            
            # Get 5m candles
            candles_5m = market_data.candles_by_interval['5']
            
            # Calculate RSI
            rsi_values = self._calculate_rsi(candles_5m, self.config['rsi_period'])
            if len(rsi_values) < 2:
                return TradeSignal(SignalType.HOLD, market_data.symbol, market_data.current_price,
                                 reason="Not enough RSI values")
            
            current_rsi = rsi_values[-1]
            prev_rsi = rsi_values[-2]
            
            logger.info(f"{market_data.symbol}: RSI = {current_rsi:.2f} (prev: {prev_rsi:.2f})")
            
            # Strategy logic
            symbol = market_data.symbol
            current_price = market_data.current_price
            current_position = market_data.current_position
            
            # Long entry: RSI oversold and rising
            if (current_rsi < self.config['rsi_oversold'] and 
                current_rsi > prev_rsi and 
                current_position != 'long'):
                
                return TradeSignal(
                    SignalType.OPEN_LONG,
                    symbol,
                    current_price,
                    reason=f"RSI oversold bounce: {current_rsi:.2f}"
                )
            
            # Short entry: RSI overbought and falling
            elif (current_rsi > self.config['rsi_overbought'] and 
                  current_rsi < prev_rsi and 
                  current_position != 'short'):
                
                return TradeSignal(
                    SignalType.OPEN_SHORT,
                    symbol,
                    current_price,
                    reason=f"RSI overbought rejection: {current_rsi:.2f}"
                )
            
            # Close long: RSI overbought
            elif current_position == 'long' and current_rsi > self.config['rsi_overbought']:
                return TradeSignal(
                    SignalType.CLOSE_LONG,
                    symbol,
                    current_price,
                    reason=f"RSI overbought, close long: {current_rsi:.2f}"
                )
            
            # Close short: RSI oversold
            elif current_position == 'short' and current_rsi < self.config['rsi_oversold']:
                return TradeSignal(
                    SignalType.CLOSE_SHORT,
                    symbol,
                    current_price,
                    reason=f"RSI oversold, close short: {current_rsi:.2f}"
                )
            
            # Hold
            return TradeSignal(SignalType.HOLD, symbol, current_price, reason="No signal conditions met")
            
        except Exception as e:
            logger.error(f"Error in RSI strategy for {market_data.symbol}: {e}")
            return TradeSignal(SignalType.HOLD, market_data.symbol, market_data.current_price, 
                             reason=f"Strategy error: {e}")
    
    def _calculate_rsi(self, candles: List[Dict], period: int = 14) -> List[float]:
        """
        Calculate RSI values
        """
        if len(candles) < period + 1:
            return []
        
        closes = [float(candle['close']) for candle in candles]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # Calculate initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        for i in range(period, len(gains)):
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
            
            # Update averages using Wilder's smoothing
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        return rsi_values