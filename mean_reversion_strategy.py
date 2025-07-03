#mean_reversion_strategy.py
from base_strategy import BaseStrategy, SignalType, MarketData, TradeSignal
import pandas as pd
import numpy as np
from typing import Dict, List
from utils import logger
import global_data

class MeanReversionStrategy(BaseStrategy):
    """
    Institutional Mean Reversion Strategy with:
    - Bollinger Bands + RSI confirmation
    - Volume-weighted entry/exit
    - Adaptive ATR-based stop loss
    - VWAP as trend filter
    """
    
    def __init__(self, config: Dict = None):
        # Default parameters (optimized for 15m timeframe)
        default_config = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bollinger_period': 20,
            'bollinger_dev': 2.0,
            'atr_period': 14,
            'atr_multiplier': 1.5,
            'vwap_confirmation': True,
            'min_volume_ratio': 1.5 , # 1.5x avg volume for entry
            'risk_percent': 0.02,            # <-- Added (2% of balance per trade)
            'max_position_pct': 0.10         # <-- Added (10% cap by default)
        }
        
        # Merge user config
        self.config = {**default_config, **(config or {})}
        super().__init__(self.config)
    
    def get_required_intervals(self) -> List[str]:
        return ['15', '5']  # Primary + confirmation timeframe
    
    def get_minimum_candles_needed(self) -> int:
        return max(self.config['bollinger_period'], 
                  self.config['atr_period']) + 10
    
    def analyze(self, market_data: MarketData) -> TradeSignal:
        try:
            # Get 15m candles (primary timeframe)
            raw_15 = market_data.candles_by_interval.get('15')
            if raw_15 is None:
                logger.warning(f"{market_data.symbol}: No 15m data found")
                return TradeSignal(SignalType.HOLD, market_data.symbol, market_data.current_price, reason="No 15m data")

            candles = list(raw_15)
            if len(candles) < self.get_minimum_candles_needed():
                return TradeSignal(SignalType.HOLD, market_data.symbol, market_data.current_price, reason="Insufficient data")
            
            # Convert to DataFrame
            df = pd.DataFrame(candles[-self.config['bollinger_period']:])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # ------ Core Indicators ------
            # Bollinger Bands
            rolling = df['close'].rolling(self.config['bollinger_period'])
            df['sma'] = rolling.mean()
            df['upper_band'] = df['sma'] + (rolling.std() * self.config['bollinger_dev'])
            df['lower_band'] = df['sma'] - (rolling.std() * self.config['bollinger_dev'])
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(alpha=1/self.config['rsi_period'], adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/self.config['rsi_period'], adjust=False).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # ATR (for stop loss)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(self.config['atr_period']).mean().iloc[-1]
            
            # VWAP (trend filter)
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['cumulative_vp'] = (df['typical_price'] * df['volume']).cumsum()
            df['cumulative_volume'] = df['volume'].cumsum()
            vwap = df['cumulative_vp'].iloc[-1] / df['cumulative_volume'].iloc[-1]
            
            # Volume spike check
            avg_volume = df['volume'].mean()
            last_volume = df['volume'].iloc[-1]
            
            # ------ Signal Logic ------
            current_price = market_data.current_price
            current_rsi = df['rsi'].iloc[-1]
            lower_band = df['lower_band'].iloc[-1]
            upper_band = df['upper_band'].iloc[-1]
            
            # Long Entry: Price at lower band + RSI oversold + Volume spike + Above VWAP (bullish)
            if (current_price <= lower_band and 
                current_rsi < self.config['rsi_oversold'] and
                last_volume >= avg_volume * self.config['min_volume_ratio'] and
                (not self.config['vwap_confirmation'] or current_price > vwap)):
                
                sl = current_price - (atr * self.config['atr_multiplier'])
                tp = df['sma'].iloc[-1]  # Target middle band
                
                return TradeSignal(
                    SignalType.OPEN_LONG,
                    market_data.symbol,
                    current_price,
                    tp_price=tp,
                    sl_price=sl,
                    amount_usd=self.calculate_position_size(current_price, sl),
                    reason=f"Mean reversion long: RSI={current_rsi:.1f}, Price at lower band"
                )
            
            # Short Entry: Price at upper band + RSI overbought + Volume spike + Below VWAP (bearish)
            elif (current_price >= upper_band and 
                  current_rsi > self.config['rsi_overbought'] and
                  last_volume >= avg_volume * self.config['min_volume_ratio'] and
                  (not self.config['vwap_confirmation'] or current_price < vwap)):
                
                sl = current_price + (atr * self.config['atr_multiplier'])
                tp = df['sma'].iloc[-1]  # Target middle band
                
                return TradeSignal(
                    SignalType.OPEN_SHORT,
                    market_data.symbol,
                    current_price,
                    tp_price=tp,
                    sl_price=sl,
                    amount_usd=self.calculate_position_size(current_price, sl),
                    reason=f"Mean reversion short: RSI={current_rsi:.1f}, Price at upper band"
                )
            
            # Close conditions
            current_position = market_data.current_position
            
            # Close long if price reaches SMA or RSI > 50
            if (current_position == 'long' and 
                (current_price >= df['sma'].iloc[-1] or current_rsi > 50)):
                return TradeSignal(
                    SignalType.CLOSE_LONG,
                    market_data.symbol,
                    current_price,
                    reason="Take profit or RSI neutral"
                )
            
            # Close short if price reaches SMA or RSI < 50
            elif (current_position == 'short' and 
                  (current_price <= df['sma'].iloc[-1] or current_rsi < 50)):
                return TradeSignal(
                    SignalType.CLOSE_SHORT,
                    market_data.symbol,
                    current_price,
                    reason="Take profit or RSI neutral"
                )
            
            return TradeSignal(SignalType.HOLD, market_data.symbol, current_price)
            
        except Exception as e:
            logger.error(f"Strategy error for {market_data.symbol}: {str(e)}")
            return TradeSignal(SignalType.HOLD, market_data.symbol, market_data.current_price)

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Risk-based position sizing"""
        risk_per_trade = self.config.get('risk_percent', 0.02) * global_data.current_balance
        risk_per_unit = abs(entry_price - stop_loss)
        max_position_size = self.config.get('max_position_pct', 0.10) * global_data.current_balance
        return min(risk_per_trade / risk_per_unit, max_position_size)