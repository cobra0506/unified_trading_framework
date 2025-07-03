# strategy.py
import time
import json
import os
import pandas as pd
import threading
import global_data
from utils import log_signal
from datetime import datetime
import concurrent.futures
from copy import deepcopy
from collections import deque
from utils import log_signal, calculate_stoch_rsi, calculate_sma, calculate_adx, calculate_atr, has_fresh_data, logger
from exchange_handler import handle_buy_signal, close_position, handle_sell_signal, get_account_balance
from global_data import BATCH_SIZE, MAX_WORKERS, SLEEP_BUFFER, POSITION_FILE, MAX_SYMBOL_ERRORS
from global_data import SRSI_OVERBOUGHT, SRSI_OVERSOLD, MAX_OPEN_POSITIONS, RISK_PERCENT, SL_ATR_MULT, TP_ATR_MULT


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

def get_data_snapshot():
    """Thread-safe snapshot of candle data for all symbols/intervals."""
    snapshot = {}
    for symbol in global_data.symbols:
        try:
            with global_data.symbol_locks[symbol]:
                snapshot[symbol] = deepcopy(global_data.candle_data[symbol])
        except Exception as e:
            logger.warning(f"Snapshot error for {symbol}: {e}")
    return snapshot

def execute_trade(action, symbol):
    action = action.lower()
    if action == "open long":
        handle_buy_signal(symbol)
    elif action == "open short":
        handle_sell_signal(symbol)
    elif action.startswith("close"):
        from exchange_handler import session, initialize_connection
        if session is None:
            session = initialize_connection()
        close_position(session=session, symbol=symbol)


def run_strategy_for_symbol(symbol, candles_by_interval):
    logger.info(f"Executing strategy for {symbol}")
    # Ensure data is fresh to avoid false processing
    ''' current_time_ms = int(time.time() * 1000)
        if not has_fresh_data(symbol, current_time_ms):
            logger.warning(f"{symbol}: Skipping strategy — data is not fresh.")
            return'''
    try:
        if global_data.symbol_health.get(symbol, 0) > MAX_SYMBOL_ERRORS:
            logger.warning(f"Skipping {symbol}: too many errors")
            return

        stoch_k = {}

        # Calculate StochRSI for each required interval
        for interval in ['1', '5', '15']:
            candles_raw = candles_by_interval.get(interval)
            if candles_raw is None:
                logger.error(f"{symbol}: candles_by_interval.get('{interval}') returned None")
                return

            # Convert deque to list
            if isinstance(candles_raw, deque):
                candles_raw = list(candles_raw)

            if not isinstance(candles_raw, list):
                logger.error(f"{symbol}: Expected a list or deque for {interval}m candles, got {type(candles_raw)}")
                return

            needed = 50
            candles = candles_raw[-needed:]

            if not candles or len(candles) < 20:
                logger.warning(f"{symbol}: Not enough {interval}m candles for StochRSI (len={len(candles)})")
                return

            logger.debug(f"{symbol}: {interval}m interval - Total candles = {len(candles)}")
            logger.debug(f"{symbol}: {interval}m interval - Last 3 candles = {candles[-3:]}")

            k_series = calculate_stoch_rsi(candles, return_series=True)

            if k_series is None or len(k_series) < 2:
                logger.warning(f"{symbol}: {interval}m StochRSI series too short")
                return

            prev_k = k_series[-2]
            curr_k = k_series[-1]

            if pd.isna(prev_k) or pd.isna(curr_k):
                logger.warning(f"{symbol}: {interval}m StochRSI values contain NaNs")
                return

            stoch_k[interval] = {
                "prev": prev_k,
                "curr": curr_k
            }

        

        logger.info(
            f"{symbol}: StochRSI %K - "
            f"1m: curr={stoch_k['1']['curr']:.2f}, prev={stoch_k['1']['prev']:.2f}; "
            f"5m: curr={stoch_k['5']['curr']:.2f}, prev={stoch_k['5']['prev']:.2f}; "
            f"15m: curr={stoch_k['15']['curr']:.2f}, prev={stoch_k['15']['prev']:.2f}"
        )

        current_price = candles_by_interval['1'][-1]['close']
        current_position = position_tracker.get_position(symbol)
        logger.info(f"{symbol}: Current position = {current_position}")

        # === SMA & ADX Calculation ===
        candles_15m = candles_by_interval.get('15', [])
        if len(candles_15m) < 21:
            logger.warning(f"{symbol}: Not enough 15m candles for SMA/ADX")
            return

        close_15m = [c['close'] for c in candles_15m]
        fast_sma = calculate_sma(close_15m, 9)
        slow_sma = calculate_sma(close_15m, 21)
        adx = calculate_adx(candles_15m)

        if fast_sma is None or slow_sma is None or adx is None:
            logger.warning(f"{symbol}: Could not calculate SMA or ADX")
            return

        trend_is_bullish = fast_sma > slow_sma and adx > 0
        trend_is_bearish = fast_sma < slow_sma and adx > 0

        logger.info(f"{symbol}: 15m Fast SMA: {fast_sma:.2f}, Slow SMA: {slow_sma:.2f}, ADX: {adx:.2f}")

        open_positions = sum(1 for p in position_tracker.positions.values() if p in ['long', 'short'])

        # === Open Long ===
        if all(stoch_k[i]['curr'] < 10 and stoch_k[i]['curr'] > stoch_k[i]['prev']for i in ['1', '5', '15']):
        #if (stoch_k['1']['curr'] < 10 and stoch_k['1']['curr'] > stoch_k['1']['prev']):
            if current_position == 'long':
                logger.info(f"{symbol}: Already in long position, skipping.")
            elif not trend_is_bullish:
                logger.info(f"{symbol}: Trend is not bullish, skipping Open Long.")
            elif open_positions >= MAX_OPEN_POSITIONS:
                logger.info(f"{symbol}: Max positions reached, skipping Open Long.")
            else:
        
                if not global_data.run_strategy:
                    return

                balance = global_data.current_balance
                if balance <= 0:
                    logger.warning(f"{symbol}: Skipping trade — balance is {balance}")
                    return

                atr = calculate_atr(candles_15m)
                if atr is None:
                    logger.warning(f"{symbol}: Not enough data for ATR")
                    return

                sl_pct = atr / current_price
                position_risk = balance * RISK_PERCENT
                amount_usd = position_risk / sl_pct

                sl_price = current_price - SL_ATR_MULT * atr
                tp_price = current_price + TP_ATR_MULT * atr

                # Validate TP/SL
                if tp_price <= current_price:
                    logger.warning(f"{symbol}: Invalid TP for long — tp={tp_price}, entry={current_price}")
                    return
                if sl_price >= current_price:
                    logger.warning(f"{symbol}: Invalid SL for long — sl={sl_price}, entry={current_price}")
                    return

                log_signal(symbol, "Open Long", current_price)
                handle_buy_signal(symbol, tp=tp_price, sl=sl_price, amount_usd=amount_usd)
                position_tracker.set_position(symbol, 'long')
                logger.info(f"{symbol}: Open Long at {current_price} with USD size: {amount_usd:.2f}, TP: {tp_price}, SL: {sl_price}")

        # === Close Long ===
        elif (stoch_k['1']['curr'] > SRSI_OVERBOUGHT or trend_is_bearish) and current_position == 'long':
            log_signal(symbol, "Close Long", current_price)
            execute_trade("Close Long", symbol)
            position_tracker.set_position(symbol, None)
            logger.info(f"{symbol}: Close Long at {current_price}")


        # === Open Short ===
        if all(stoch_k[i]['curr'] > 75 and stoch_k[i]['curr'] < stoch_k[i]['prev']for i in ['1', '5', '15']):
        #if stoch_k['1']['curr'] > 75 and stoch_k['1']['curr'] < stoch_k['1']['prev']:
            if current_position == 'short':
                logger.info(f"{symbol}: Already in short position, skipping.")
            elif not trend_is_bearish:
                logger.info(f"{symbol}: Trend is not bearish, skipping Open Short.")
            elif open_positions >= MAX_OPEN_POSITIONS:
                logger.info(f"{symbol}: Max positions reached, skipping Open Short.")
            else:
                if not global_data.run_strategy:
                    return
                balance = global_data.current_balance
                if balance <= 0:
                    logger.warning(f"{symbol}: Skipping trade — balance is {balance}")
                    return
                atr = calculate_atr(candles_15m)
                if atr is None:
                    logger.warning(f"{symbol}: Not enough data for ATR")
                    return
                sl_pct = atr / current_price
                position_risk = balance * RISK_PERCENT
                amount_usd = position_risk / sl_pct

                sl_price = current_price + SL_ATR_MULT * atr
                tp_price = current_price - TP_ATR_MULT * atr

                # Validate TP/SL
                if tp_price >= current_price:
                    logger.warning(f"{symbol}: Invalid TP for short — tp={tp_price}, entry={current_price}")
                    return
                if sl_price <= current_price:
                    logger.warning(f"{symbol}: Invalid SL for short — sl={sl_price}, entry={current_price}")
                    return

                log_signal(symbol, "Open Short", current_price)
                handle_sell_signal(symbol, tp=tp_price, sl=sl_price, amount_usd=amount_usd)
                position_tracker.set_position(symbol, 'short')
                logger.info(f"{symbol}: Open Short at {current_price} with USD size: {amount_usd:.2f}, TP: {tp_price}, SL: {sl_price}")

        # === Close Short ===
        elif (stoch_k['1']['curr'] < SRSI_OVERSOLD or trend_is_bullish) and current_position == 'short':
            log_signal(symbol, "Close Short", current_price)
            execute_trade("Close Short", symbol)
            position_tracker.set_position(symbol, None)
            logger.info(f"{symbol}: Close Short at {current_price}")

    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        global_data.symbol_health[symbol] = global_data.symbol_health.get(symbol, 0) + 1


def run_strategy_loop(batch_size=BATCH_SIZE, max_workers=MAX_WORKERS):
    logger.info("Strategy loop started")
    global_data.strategy_running = True

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                try:
                    logger.info("Running strategy iteration")
                    start_time = time.time()

                    # ✅ Update balance once per tick
                    get_account_balance()
                    strategy_balance = global_data.current_balance

                    data_snapshot = get_data_snapshot()
                    symbols = list(data_snapshot.keys())

                    for i in range(0, len(symbols), batch_size):
                        batch = symbols[i:i + batch_size]
                        futures = {
                            executor.submit(run_strategy_for_symbol, symbol, data_snapshot[symbol]): symbol
                            for symbol in batch
                        }
                        concurrent.futures.wait(futures)

                    elapsed = time.time() - start_time
                    logger.info(f"Strategy iteration processed in {elapsed:.2f} seconds")

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

if __name__ == "__main__":
    # Test on one symbol first
    global_data.symbols = ["BTCUSDT"]
    run_strategy_loop(batch_size=1, max_workers=1)

