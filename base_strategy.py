#base_startegy.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from utils import logger

class SignalType(Enum):
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"

@dataclass
class TradeSignal:
    signal_type: SignalType
    symbol: str
    price: float
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    amount_usd: Optional[float] = None
    reason: str = ""

@dataclass
class MarketData:
    """Container for all market data needed by strategies"""
    symbol: str
    candles_by_interval: Dict[str, List[Dict]]  # interval -> list of candles
    current_price: float
    current_position: Optional[str]  # 'long', 'short', or None

class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def analyze(self, market_data: MarketData) -> TradeSignal:
        """
        Analyze market data and return a trade signal
        
        Args:
            market_data: MarketData object containing all necessary market information
            
        Returns:
            TradeSignal object indicating what action to take
        """
        pass
    
    @abstractmethod
    def get_required_intervals(self) -> List[str]:
        """Return list of required timeframe intervals (e.g., ['1', '5', '15'])"""
        pass
    
    @abstractmethod
    def get_minimum_candles_needed(self) -> int:
        """Return minimum number of candles needed for analysis"""
        pass
    
    def validate_data(self, market_data: MarketData) -> bool:
        """
        Validate that market data contains everything needed for the strategy
        
        Args:
            market_data: MarketData object to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_intervals = self.get_required_intervals()
        min_candles = self.get_minimum_candles_needed()
        
        for interval in required_intervals:
            candles = market_data.candles_by_interval.get(interval, [])
            if len(candles) < min_candles:
                logger.debug(
                    f"{market_data.symbol}: Validation failed for interval {interval}. "
                    f"Found {len(candles)} candles, need {min_candles}"
                )
                return False
            # Check if candles have required fields
            for candle in candles:
                required_fields = ['timestamp', 'open', 'high', 'low', 'close']
                if not all(field in candle for field in required_fields):
                    logger.debug(
                        f"{market_data.symbol}: Invalid candle in {interval}: {candle}"
                    )
                    return False
        
        logger.debug(f"{market_data.symbol}: Data validation passed. Found sufficient candles for all intervals")
        return True
    
    def get_config_value(self, key: str, default=None):
        """Get configuration value with fallback to default"""
        return self.config.get(key, default)