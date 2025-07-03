#strategy_runner.py
import time
import json
import os
import pandas as pd
import threading
import global_data
from utils import log_signal, logger
from datetime import datetime
import concurrent.futures
from copy import deepcopy
from collections import deque

# Strategy imports
from base_strategy import MarketData, SignalType
from stochrsi_strategy import StochRSIStrategy
from mean_reversion_strategy import MeanReversionStrategy
from multi_timeframe_strategy import MultiTimeframeStrategy


from exchange_handler import handle_buy_signal, close_position, handle_sell_signal, get_account_balance
from global_data import BATCH_SIZE, MAX_WORKERS, SLEEP_BUFFER, POSITION_FILE, MAX_SYMBOL_ERRORS
from global_data import MAX_OPEN_POSITIONS


class PositionTracker:
    def __init__(self):
        self.positions = {}
        self.lock = threading.Lock()
        self.load_positions()

    def load_positions(self):
        try:
            with open(POSITION_FILE, 'r') as f:
                self.positions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.positions = {}

    def save_positions(self):
        with self.lock:
            with open(POSITION_FILE, 'w') as f:
                json.dump(self.positions, f)

    def get_position(self, symbol):
        return self.positions.get(symbol, None)

    def set_position(self, symbol, position_type):
        logger.info(f"Setting position for {symbol}: {position_type}")
        self.positions[symbol] = position_type
        self.save_positions()


class StrategyRunner:
    """
    Strategy execution framework that can work with any strategy implementing BaseStrategy
    """
    
    def __init__(self, strategy_class=None, strategy_config=None):
        self.position_tracker = PositionTracker()
        
        # Initialize strategy - default to StochRSI if none provided
        if strategy_class is None:
            #strategy_class = StochRSIStrategy
            #strategy_class = MeanReversionStrategy
            strategy_class = MultiTimeframeStrategy
        
        self.strategy = strategy_class(strategy_config)
        logger.info(f"Initialized strategy: {self.strategy.name}")
    
    def get_data_snapshot(self):
        """Thread-safe snapshot of candle data for all symbols/intervals."""
        snapshot = {}
        for symbol in global_data.symbols:
            try:
                with global_data.symbol_locks[symbol]:
                    snapshot[symbol] = deepcopy(global_data.candle_data[symbol])
                    # Log the number of candles for each interval
                    for interval in global_data.time_frames:
                        candle_count = len(snapshot[symbol].get(interval, []))
                        logger.debug(
                            f"{symbol}: Snapshot contains {candle_count} candles for interval {interval}"
                        )
            except Exception as e:
                logger.warning(f"Snapshot error for {symbol}: {e}")
        return snapshot
    
    def execute_trade_signal(self, signal):
        """Execute a trade signal"""
        if signal.signal_type == SignalType.OPEN_LONG:
            handle_buy_signal(
                signal.symbol, 
                tp=signal.tp_price, 
                sl=signal.sl_price, 
                amount_usd=signal.amount_usd
            )
            self.position_tracker.set_position(signal.symbol, 'long')
            
        elif signal.signal_type == SignalType.OPEN_SHORT:
            handle_sell_signal(
                signal.symbol, 
                tp=signal.tp_price, 
                sl=signal.sl_price, 
                amount_usd=signal.amount_usd
            )
            self.position_tracker.set_position(signal.symbol, 'short')
            
        elif signal.signal_type == SignalType.CLOSE_LONG:
            self._close_position(signal.symbol)
            self.position_tracker.set_position(signal.symbol, None)
            
        elif signal.signal_type == SignalType.CLOSE_SHORT:
            self._close_position(signal.symbol)
            self.position_tracker.set_position(signal.symbol, None)
    
    def _close_position(self, symbol):
        """Close position for a symbol"""
        from exchange_handler import session, initialize_connection
        if session is None:
            session = initialize_connection()
        close_position(session=session, symbol=symbol)
    
    def run_strategy_for_symbol(self, symbol, candles_by_interval):
        """Run strategy analysis for a single symbol"""
        logger.info(f"Executing strategy for {symbol}")
        
        try:
            # Check symbol health
            if global_data.symbol_health.get(symbol, 0) > MAX_SYMBOL_ERRORS:
                logger.warning(f"Skipping {symbol}: too many errors")
                return
            
            # Prepare market data
            current_price = candles_by_interval['1'][-1]['close']
            current_position = self.position_tracker.get_position(symbol)
            
            market_data = MarketData(
                symbol=symbol,
                candles_by_interval=candles_by_interval,
                current_price=current_price,
                current_position=current_position
            )
            
            # Get signal from strategy
            signal = self.strategy.analyze(market_data)
            
            # Log the signal
            logger.info(f"{symbol}: Strategy signal = {signal.signal_type.value}, Reason: {signal.reason}")
            
            # Check if we should execute the trade
            if signal.signal_type == SignalType.HOLD:
                return
            
            # Additional checks before executing
            if not self._should_execute_signal(signal):
                return
            
            # Execute the trade
            log_signal(symbol, signal.signal_type.value, signal.price)
            self.execute_trade_signal(signal)
            
            logger.info(f"{symbol}: Executed {signal.signal_type.value} at {signal.price}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            global_data.symbol_health[symbol] = global_data.symbol_health.get(symbol, 0) + 1
    
    def _should_execute_signal(self, signal):
        """Additional checks before executing a signal"""
        
        # Check if strategy is enabled
        if not global_data.run_strategy:
            logger.info(f"{signal.symbol}: Strategy execution disabled")
            return False
        
        # Check balance for opening positions
        if signal.signal_type in [SignalType.OPEN_LONG, SignalType.OPEN_SHORT]:
            balance = global_data.current_balance
            if balance <= 0:
                logger.warning(f"{signal.symbol}: Insufficient balance: {balance}")
                return False
            
            # Check max positions limit
            open_positions = sum(1 for p in self.position_tracker.positions.values() if p in ['long', 'short'])
            if open_positions >= MAX_OPEN_POSITIONS:
                logger.info(f"{signal.symbol}: Max positions reached ({open_positions}/{MAX_OPEN_POSITIONS})")
                return False
        
        return True
    
    def run_strategy_loop(self, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS):
        """Main strategy execution loop"""
        logger.info(f"Strategy loop started with {self.strategy.name}")
        global_data.strategy_running = True

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                while True:
                    try:
                        logger.info("Running strategy iteration")
                        start_time = time.time()

                        # Update balance once per tick
                        #get_account_balance()

                        # Get market data snapshot
                        data_snapshot = self.get_data_snapshot()
                        symbols = list(data_snapshot.keys())

                        # Process symbols in batches
                        for i in range(0, len(symbols), batch_size):
                            batch = symbols[i:i + batch_size]
                            futures = {
                                executor.submit(self.run_strategy_for_symbol, symbol, data_snapshot[symbol]): symbol
                                for symbol in batch
                            }
                            concurrent.futures.wait(futures)

                        elapsed = time.time() - start_time
                        logger.info(f"Strategy iteration processed in {elapsed:.2f} seconds")

                        # Sleep until next minute
                        sleep_time = max(0, 60 - (time.time() % 60) + SLEEP_BUFFER)
                        logger.info(f"Sleeping for {sleep_time:.2f} seconds")
                        time.sleep(sleep_time)

                    except Exception as e:
                        logger.error(f"Error in strategy loop: {e}")
                        time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Strategy loop interrupted")

        finally:
            global_data.strategy_running = False
            logger.info("Strategy loop stopped")


# Factory function to create different strategies
def create_strategy(strategy_name: str, config: dict = None):
    """Factory function to create strategy instances"""
    
    strategies = {
        'stochrsi': StochRSIStrategy,
        'mean_reversion': MeanReversionStrategy,
        'multi_timeframe': MultiTimeframeStrategy
        # Add more strategies here as you create them
        # 'macd': MACDStrategy,
        # 'rsi': RSIStrategy,
    }
    
    strategy_class = strategies.get(strategy_name.lower())
    if strategy_class is None:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategy_class(config)


if __name__ == "__main__":
    # Example usage - you can easily change the strategy here
    
    # Option 1: Use default StochRSI strategy
    runner = StrategyRunner()
    
    # Option 2: Use factory function with custom config
    # strategy_config = {
    #     'srsi_overbought': 80,
    #     'srsi_oversold': 20,
    #     'risk_percent': 0.01
    # }
    # strategy = create_strategy('stochrsi', strategy_config)
    # runner = StrategyRunner(strategy_class=type(strategy), strategy_config=strategy_config)
    
    # Test on one symbol first
    global_data.symbols = ["BTCUSDT"]
    runner.run_strategy_loop(batch_size=1, max_workers=1)