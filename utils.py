# utils.py
import os
import json
import threading
import logging
import logging.handlers
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import csv
from collections import Counter
import global_data
from global_data import POSITION_FILE

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
def setup_logging():
    # Create logger
    logger = logging.getLogger('TradingBot')
    logger.setLevel(logging.DEBUG)  # Capture all levels
    logger.handlers = []  # Clear any existing handlers to avoid duplicates

    # Formatter for logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler for ERROR and WARNING only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only WARNING and ERROR
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler for bot.log (INFO and above - reduced from DEBUG)
    bot_log_handler = logging.handlers.RotatingFileHandler(
        os.path.join('logs', 'bot.log'),
        maxBytes=5*1024*1024,  # Reduced to 5MB
        backupCount=3,  # Reduced backup count
        encoding='utf-8'
    )
    bot_log_handler.setLevel(logging.INFO)  # Changed from DEBUG to INFO
    bot_log_handler.setFormatter(formatter)
    logger.addHandler(bot_log_handler)

    # File handler for error.log (ERROR and WARNING only)
    error_log_handler = logging.handlers.RotatingFileHandler(
        os.path.join('logs', 'error.log'),
        maxBytes=2*1024*1024,  # Smaller size for errors only
        backupCount=5,
        encoding='utf-8'
    )
    error_log_handler.setLevel(logging.WARNING)  # WARNING and ERROR only
    error_log_handler.setFormatter(formatter)
    logger.addHandler(error_log_handler)

    # Optional: Create a separate debug log that rotates more frequently
    debug_log_handler = logging.handlers.RotatingFileHandler(
        os.path.join('logs', 'debug.log'),
        maxBytes=1*1024*1024,  # Small 1MB files
        backupCount=2,  # Only keep 2 backup files
        encoding='utf-8'
    )
    debug_log_handler.setLevel(logging.DEBUG)
    debug_log_handler.setFormatter(formatter)
    logger.addHandler(debug_log_handler)

    logger.info("Logging setup complete. Handlers: %s", [h.__class__.__name__ for h in logger.handlers])
    return logger

# Initialize logger
logger = setup_logging()

# Initialize CSV logs at the start of the session
signals_filepath = os.path.join('logs', 'signals_log.csv')
with open(signals_filepath, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'symbol', 'signal', 'price'])  # Write header
    logger.debug("Initialized signals_log.csv")

opened_positions_filepath = os.path.join('logs', 'opened_positions.csv')
with open(opened_positions_filepath, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'symbol', 'action', 'price', 'amount_usd'])  # Updated header to use 'action'
    logger.debug("Initialized opened_positions.csv")

def convert_timestamp_to_readable(timestamp_milliseconds):
    timestamp_seconds = timestamp_milliseconds / 1000
    return datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

def normalize_to_interval(timestamp_ms, interval_minutes):
    interval_ms = interval_minutes * 60 * 1000
    return (timestamp_ms // interval_ms) * interval_ms

def write_candle_data_to_csv(candle_data, output_dir="candle_data"):
    os.makedirs(output_dir, exist_ok=True)
    for symbol, intervals in candle_data.items():
        for interval, candles in intervals.items():
            filepath = os.path.join(output_dir, f"{symbol}-{interval}.csv")
            with global_data.symbol_locks[symbol]:
                unique_candles = {}
                for candle in candles:
                    unique_candles[candle['timestamp']] = candle
                sorted_candles = sorted(unique_candles.values(), key=lambda c: c['timestamp'])
            
            with open(filepath, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['symbol', 'interval', 'timestamp_utc', 'open', 'high', 'low', 'close', 'volume'])
                
                timestamps = [convert_timestamp_to_readable(c['timestamp']) for c in sorted_candles]
                duplicates = [item for item, count in Counter(timestamps).items() if count > 1]
                if duplicates:
                    logger.warning(f"Duplicate timestamps for {symbol}/{interval}: {duplicates}")
                    for ts in duplicates:
                        duplicate_candles = [c for c in sorted_candles if convert_timestamp_to_readable(c['timestamp']) == ts]
                        logger.debug(f"Duplicate candles: {duplicate_candles}")
                
                for candle in sorted_candles:
                    writer.writerow([
                        symbol,
                        interval,
                        convert_timestamp_to_readable(candle['timestamp']),
                        candle['open'],
                        candle['high'],
                        candle['low'],
                        candle['close'],
                        candle.get('volume', 0)
                    ])
                    
def add_candle_uniquely(candle_deque, new_candle, interval_minutes):
    if not candle_deque:
        candle_deque.append(new_candle)
        return
    
    interval_ms = interval_minutes * 60 * 1000
    for existing in candle_deque:
        if existing['timestamp'] == new_candle['timestamp']:
            existing.update(new_candle)
            return
    
    for i, existing in enumerate(candle_deque):
        if new_candle['timestamp'] < existing['timestamp']:
            candle_deque.insert(i, new_candle)
            logger.debug(f"Inserted candle at {new_candle['timestamp']} at index {i}")
            return
    
    candle_deque.append(new_candle)

def validate_candle(candle):
    try:
        o, h, l, c, v = candle['open'], candle['high'], candle['low'], candle['close'], candle.get('volume', 0)
        is_valid = l <= o <= h and l <= c <= h and v >= 0 and all(isinstance(x, (int, float)) for x in [o, h, l, c, v])
        if not is_valid:
            logger.warning(f"Invalid candle data: {candle}")
        return is_valid
    except (KeyError, TypeError) as e:
        logger.error(f"Invalid candle format: {e}, candle: {candle}")
        return False

def has_fresh_data(symbol, current_timestamp_ms):
    interval_ms_map = {
        '1': 60 * 1000,
        '5': 5 * 60 * 1000,
        '15': 15 * 60 * 1000
    }
    
    current_minute = (current_timestamp_ms // (60 * 1000)) % 60
    required_intervals = ['1']
    
    if current_minute % 5 == 0:
        required_intervals.append('5')
        
    if current_minute % 15 == 0:
        required_intervals.append('15')
    
    with global_data.symbol_locks[symbol]:
        for interval in required_intervals:
            if symbol not in global_data.candle_data or interval not in global_data.candle_data[symbol]:
                logger.warning(f"No candle data for {symbol}/{interval}")
                return False
                
            candles = global_data.candle_data[symbol][interval]
            if not candles:
                logger.warning(f"Empty candle data for {symbol}/{interval}")
                return False
                
            interval_ms = interval_ms_map[interval]
            expected_timestamp = (current_timestamp_ms // interval_ms) * interval_ms
            has_candle = any(candle['timestamp'] == expected_timestamp for candle in candles)
            if not has_candle:
                logger.warning(f"No candle for {symbol}/{interval} at timestamp {expected_timestamp}")
                return False
                         
    return True

def calculate_stoch_rsi(candles, period=14, k=3, d=3, return_series=False):

    if len(candles) < period + k + d:
        logger.debug(f"Not enough candles ({len(candles)}) for Stoch RSI calculation")
        return None if return_series else (None, None)
    
    closes = [c['close'] for c in candles]
    df = pd.DataFrame({'close': closes})
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    k_line = stoch_rsi.rolling(k).mean() * 100
    d_line = k_line.rolling(d).mean()
    
    if return_series:
        return k_line.dropna().values
    
    return k_line.iloc[-1], d_line.iloc[-1]

def calculate_adx(candles, period=14):
    if len(candles) < period + 1:
        logger.debug(f"Not enough candles ({len(candles)}) for ADX calculation")
        return None

    df = pd.DataFrame(candles)
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # True Range (TR)
    df['prev_close'] = df['close'].shift()
    df['tr'] = df[['high', 'prev_close']].max(axis=1) - df[['low', 'prev_close']].min(axis=1)

    # Directional Movement (+DM, -DM)
    df['up_move'] = df['high'] - df['high'].shift()
    df['down_move'] = df['low'].shift() - df['low']

    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    # Wilder’s smoothing
    atr = df['tr'].ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * df['plus_dm'].ewm(alpha=1/period, min_periods=period).mean() / atr
    minus_di = 100 * df['minus_dm'].ewm(alpha=1/period, min_periods=period).mean() / atr

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()

    return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else None

def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        logger.debug(f"Not enough candles ({len(candles)}) for ATR calculation")
        return None

    df = pd.DataFrame(candles)
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    df['prev_close'] = df['close'].shift()
    df['tr'] = df[['high', 'prev_close']].max(axis=1) - df[['low', 'prev_close']].min(axis=1)

    atr = df['tr'].rolling(window=period).mean()
    return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else None

def calculate_sma(values, period):
    if len(values) < period:
        logger.debug(f"Not enough values ({len(values)}) for SMA calculation")
        return None
    return pd.Series(values).rolling(window=period).mean().iloc[-1]

def log_signal(symbol, signal_type, price):
    filepath = os.path.join('logs', 'signals_log.csv')
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, symbol, signal_type, price])
            logger.info(f"Logged signal: {symbol}, {signal_type}, {price}")
    except Exception as e:
        logger.error(f"Failed to log signal: {e}")

def log_opened_position(symbol, action, price, amount_usd):
    filepath = os.path.join('logs', 'opened_positions.csv')
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, symbol, action, price, amount_usd])
            logger.info(f"Logged position: {symbol}, {action}, {price}, {amount_usd}")
    except Exception as e:
        logger.error(f"Failed to log position: {e}")

# Additional utility function to help troubleshoot signal execution issues
def log_signal_execution_failure(symbol, signal_type, price, error):
    """Log when a signal fails to execute - this will help troubleshoot the unmatched signals"""
    logger.error(f"SIGNAL EXECUTION FAILED: {symbol} {signal_type} @ {price} - Error: {error}")

def log_position_mismatch(expected_symbol, expected_action, actual_positions):
    """Log when expected position doesn't match actual positions"""
    logger.error(f"POSITION MISMATCH: Expected {expected_symbol} {expected_action}, but found: {actual_positions}")


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

position_tracker = PositionTracker()
