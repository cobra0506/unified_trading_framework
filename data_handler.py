# data_handler.py
import websocket
import json
import threading
import time
import concurrent.futures
from collections import deque
from queue import Queue
from pybit.unified_trading import HTTP
from utils import convert_timestamp_to_readable, validate_candle, add_candle_uniquely, logger
import global_data


def get_client():
    """Initialize Bybit HTTP client."""
    api_key = global_data.demo_api_key if global_data.demo else global_data.real_api_key
    api_secret = global_data.demo_api_secret if global_data.demo else global_data.real_api_secret
    return HTTP(api_key=api_key, api_secret=api_secret, demo=global_data.demo)

def get_symbols():
    """Fetch all tradable linear perpetual symbols."""
    client = get_client()
    excluded_symbols = ['USDC', 'USDE', 'USTC']
    all_symbols = []
    cursor = None
    
    while True:
        try:
            response = client.get_instruments_info(category="linear", limit=1000, cursor=cursor)
            if response.get("retCode") != 0:
                logger.error(f"Failed to fetch symbols: {response.get('retMsg')}")
                break
            
            items = response['result']['list']
            symbols = [
                item['symbol']
                for item in items
                if not any(exclusion in item['symbol'] for exclusion in excluded_symbols)
                and "-" not in item['symbol']
                and not item['symbol'].endswith("PERP")
            ]
            all_symbols.extend(symbols)
            cursor = response['result'].get('nextPageCursor')
            if not cursor:
                break
        except Exception as e:
            logger.error(f"Exception fetching symbols: {e}")
            break
    
    logger.info(f"Fetched {len(all_symbols)} perpetual symbols")
    return all_symbols

def get_historical_data(symbol, interval, limit=global_data.candle_limit, start_time=None, end_time=None):
    """Fetch historical OHLCV data."""
    try:
        client = get_client()
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time
        
        response = client.get_kline(**params)
        if response.get('retCode') != 0:
            logger.error(f"Failed to fetch historical data for {symbol}/{interval}: {response.get('retMsg')}")
            return []
        
        candles = []
        for item in response['result']['list'][:-1]:  # Exclude open candle
            try:
                candle = {
                    'timestamp': int(item[0]),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5])
                }
                if validate_candle(candle):
                    candles.append(candle)
                else:
                    logger.warning(f"Skipping invalid historical candle for {symbol}/{interval}: {candle}")
            except Exception as e:
                logger.error(f"Error processing candle for {symbol}/{interval}: {e}")
        
        return candles
    
    except Exception as e:
        logger.error(f"Exception fetching historical data for {symbol}/{interval}: {e}")
        return []
    
def fetch_historical_data():
    """Fetch historical data for all symbols and intervals."""
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # Reduced workers
        future_map = {}
        for symbol in global_data.symbols:
            for interval in global_data.time_frames:
                future = executor.submit(get_historical_data, symbol, interval)
                future_map[future] = (symbol, interval)
        
        for future in concurrent.futures.as_completed(future_map):
            symbol, interval = future_map[future]
            try:
                data = future.result()
                if data is None:
                    logger.error(f"{symbol}/{interval}: Historical data returned None")
                    global_data.symbol_health[symbol] += 1
                    continue
                    
                if data:
                    data = sorted(data, key=lambda x: x['timestamp'])
                    with global_data.symbol_locks[symbol]:
                        global_data.candle_data[symbol][interval].extend(data)
                    logger.info(f"{symbol}/{interval}: {len(data)} historical candles fetched")
                else:
                    global_data.symbol_health[symbol] += 1
                    logger.warning(f"{symbol}/{interval}: Empty data returned")
            except Exception as e:
                logger.error(f"{symbol}/{interval}: Error fetching historical data: {str(e)}")
                global_data.symbol_health[symbol] += 1
    
    elapsed = time.time() - start_time
    logger.info(f"Historical data fetched in {int(elapsed // 60)}m {int(elapsed % 60)}s")

    if global_data.ws_connected:
        with global_data.candle_data_lock:
            global_data.run_strategy = True

class BybitWebSocketManager:
    def __init__(self):
        self.data_queue = Queue()
        self.stop_event = threading.Event()
        self.reconnect_delay = 5
        self.last_message_time = time.time()
        self.heartbeat_interval = 20
        self.ws_thread = None
        self.reconnect_attempts = 0

    def hard_recovery(self):
        """Full reset only on connection issues."""
        logger.warning("Initiating hard recovery...")
        with global_data.candle_data_lock:
            global_data.run_strategy = False
        with global_data.candle_data_lock:
            global_data.candle_data = {
                symbol: {interval: deque(maxlen=global_data.candle_limit) 
                         for interval in global_data.time_frames}
                for symbol in global_data.symbols
            }
        self.stop()
        self.start(global_data.symbols, global_data.time_frames)
        fetch_historical_data()
        logger.info("Hard recovery complete!")

    def _subscribe(self, ws, symbols, intervals):
        args = [f"kline.{interval}.{symbol}" for symbol in symbols for interval in intervals]
        for i in range(0, len(args), 500):
            ws.send(json.dumps({"op": "subscribe", "args": args[i:i+500]}))

    def _send_ping(self, ws):
        while not self.stop_event.is_set():
            try:
                ws.send(json.dumps({"op": "ping"}))
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Ping error: {e}")
                break

    def _process_kline(self, data):
        try:
            topic = data["topic"]
            interval = topic.split(".")[1]
            symbol = topic.split(".")[2]
            candles = data["data"]
            confirmed_candles = [c for c in candles if c.get('confirm', False)]
            
            if not confirmed_candles:
                return
            
            with global_data.symbol_locks[symbol]:
                if symbol in global_data.candle_data and interval in global_data.candle_data[symbol]:
                    for candle in confirmed_candles:
                        cleaned = {
                            'timestamp': int(candle['start']),
                            'open': float(candle['open']),
                            'high': float(candle['high']),
                            'low': float(candle['low']),
                            'close': float(candle['close']),
                            'volume': float(candle.get('volume', 0))
                        }
                        if validate_candle(cleaned):
                            candle_deque = global_data.candle_data[symbol][interval]
                            interval_ms = int(interval) * 60 * 1000
                            if candle_deque and cleaned['timestamp'] < candle_deque[-1]['timestamp'] - (3 * interval_ms):
                                logger.warning(
                                    f"Skipping stale WebSocket candle for {symbol}/{interval}: {convert_timestamp_to_readable(cleaned['timestamp'])}"
                                )
                                continue
                            add_candle_uniquely(candle_deque, cleaned, int(interval))
                        else:
                            logger.warning(f"Invalid live candle for {symbol}/{interval}: {cleaned}")
                else:
                    logger.warning(f"Unknown symbol/interval: {symbol}/{interval}")
        except Exception as e:
            logger.error(f"Error processing kline: {e}")

    def _on_message(self, ws, message):
        try:
            self.last_message_time = time.time()
            data = json.loads(message)
            if "topic" in data and "kline" in data["topic"]:
                self._process_kline(data)
            elif data.get('op') == 'pong':
                logger.debug("Pong received")
        except Exception as e:
            logger.error(f"Message processing error: {e}")

    def _run_websocket(self, url, symbols, intervals):
        def on_open(ws):
            logger.info("WebSocket connected")
            global_data.ws_connected = True
            self.last_message_time = time.time()
            self.reconnect_delay = 5
            self._subscribe(ws, symbols, intervals)
            threading.Thread(target=self._send_ping, args=(ws,), daemon=True).start()
            
        while not self.stop_event.is_set():
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_open=on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, e: logger.error(f"WebSocket error: {e}"),
                    on_close=lambda ws, s, m: logger.info(f"WebSocket closed: {s}, {m}")
                )
                self.ws_thread = threading.Thread(
                    target=ws.run_forever,
                    kwargs={"ping_interval": 0, "ping_timeout": 10},
                    daemon=True
                )
                self.ws_thread.start()
                self.ws_thread.join()
                global_data.ws_connected = False
                logger.info("WebSocket disconnected, attempting reconnect")
                
                if not self.stop_event.is_set():
                    time.sleep(self.reconnect_delay)
                    self.reconnect_delay = min(self.reconnect_delay * 2, 60)
            except Exception as e:
                logger.error(f"WebSocket crashed: {e}")
                global_data.ws_connected = False
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)

    def start(self, symbols, intervals):
        self.stop_event.clear()
        self.symbols = symbols
        self.intervals = intervals
        self.ws_thread = threading.Thread(
            target=self._run_websocket,
            args=("wss://stream.bybit.com/v5/public/linear", symbols, intervals),
            daemon=True
        )
        self.ws_thread.start()

    def stop(self):
        self.stop_event.set()
        global_data.ws_connected = False
        with global_data.candle_data_lock:
            global_data.run_strategy = False
        logger.info("WebSocket manager stopped")
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)

    def get_data(self):
        messages = []
        while not self.data_queue.empty():
            messages.append(self.data_queue.get())
        return messages
    
    def _on_close(self, ws, close_status, close_msg):
        logger.error(f"Connection closed: {close_status} - {close_msg}")
        if not self.stop_event.is_set():
            self.hard_recovery()

    def monitor_connection(self):
        while True:
            time.sleep(5)
            if (time.time() - self.last_message_time > 60 and 
                not self.stop_event.is_set() and 
                not global_data.strategy_running):
                logger.warning("No messages for 60s - triggering hard recovery")
                self.hard_recovery()