# global_data.py
import threading
import os
from collections import deque

# Account Settings thystest scalping
demo = True
demo_api_key = "mlD0Z3JrhD4TpRXkga"
demo_api_secret = "s1AL9DKFYVKHlHBH3vpdBVXVbVgn1Ew6X2iq"
real_api_key = "INDLV9HEZL1Q6FeCwh"
real_api_secret = "kJcF3a3z7dFae8Mxl2XGuE0cAboJR0SZRZbY"

# Data Settings
time_frames = ["1", "5", "15"]
candle_limit = 50

# Runtime Data
symbols = []
candle_data = {}
candle_data_lock = threading.Lock()
symbol_locks = {}  # Per-symbol locks
ws_connected = False
symbol_health = {}
run_strategy = False
strategy_running = False

# test settings
start_test_balance = 100
real_start_balance = 0
current_balance = 100

# strategy settings
# Constants
Amount = 1000
Leverage = 10

BATCH_SIZE = 50  # Number of symbols per batch
MAX_WORKERS = min(32, (os.cpu_count() or 1) * 4)  # Optimal thread pool size
SLEEP_BUFFER = 5  # Extra seconds after each minute
POSITION_FILE = 'positions.json'  # File to store current positions
MAX_SYMBOL_ERRORS = 10  # Max error count before skipping a symbol

SRSI_OVERBOUGHT = 80
SRSI_OVERSOLD = 20
MAX_OPEN_POSITIONS = 10  # Add this constant at the top of your script
RISK_PERCENT = 0.01
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0