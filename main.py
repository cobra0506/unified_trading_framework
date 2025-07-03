# main.py

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.clock import Clock
import time
import threading
import atexit
import os
from collections import deque
from strategy_runner import StrategyRunner
from data_handler import get_symbols, get_historical_data, BybitWebSocketManager, fetch_historical_data
from utils import write_candle_data_to_csv, convert_timestamp_to_readable, setup_logging, logger
import global_data
from strategy import run_strategy_loop

# Initialize logger
logger = setup_logging()

# Initialize symbols and candle data
global_data.symbols = get_symbols()
logger.info(f"Total symbols fetched: {len(global_data.symbols)}")
global_data.candle_data = {
    symbol: {interval: deque(maxlen=global_data.candle_limit) for interval in global_data.time_frames}
    for symbol in global_data.symbols
}
global_data.symbol_health = {symbol: 0 for symbol in global_data.symbols}
global_data.symbol_locks = {symbol: threading.Lock() for symbol in global_data.symbols}

class ControlPanel(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=10, spacing=10, **kwargs)

        # First row: Toggle + status + balance
        top_row = BoxLayout(orientation='horizontal', size_hint_y=None, height='40dp', spacing=10)

        self.toggle_button = Button(text="Stop", size_hint=(None, None), size=('120dp', '40dp'))
        self.toggle_button.bind(on_press=self.toggle_strategy)
        top_row.add_widget(self.toggle_button)

        self.status_label = Label(text="Running", size_hint=(None, None), size=('100dp', '40dp'))
        top_row.add_widget(self.status_label)

        self.balance_label = Label(text=f"Balance: ${global_data.start_test_balance:.2f}", halign="right")
        top_row.add_widget(self.balance_label)

        self.add_widget(top_row)

        # Second row: Apply button + balance input
        bottom_row = BoxLayout(orientation='horizontal', size_hint_y=None, height='40dp', spacing=10)

        self.apply_button = Button(text="Apply", size_hint=(None, None), size=('120dp', '40dp'))
        self.apply_button.bind(on_press=self.apply_balance)
        bottom_row.add_widget(self.apply_button)

        self.balance_input = TextInput(
            text=str(global_data.start_test_balance),
            multiline=False,
            size_hint=(1, None),
            height='40dp'
        )
        bottom_row.add_widget(self.balance_input)

        # Periodic UI updates
        Clock.schedule_interval(self.update_ui, 0.5)

    def toggle_strategy(self, instance):
        global_data.run_strategy = not global_data.run_strategy
        logger.info(f"run_strategy set to: {global_data.run_strategy}")

    def apply_balance(self, instance):
        try:
            value = float(self.balance_input.text)
            if value >= 0:
                global_data.start_test_balance = value
                logger.info(f"start_test_balance updated to: {value}")
                self.balance_label.text = f"Balance: ${value:.2f}"
            else:
                logger.warning("Value must be non-negative")
        except ValueError:
            logger.warning("Invalid number entered")

    def update_ui(self, dt):
        self.status_label.text = "Running" if global_data.run_strategy else "Stopping"
        self.toggle_button.text = "Stop" if global_data.run_strategy else "Start"
        self.balance_label.text = f"Balance: ${global_data.current_balance:.2f}"

class TradingBotApp(App):
    def build(self):
        return ControlPanel()
    
    def on_stop(self):
        logger.info("GUI closed. Exiting application...")
        os._exit(0)  # Hard exit to kill all threads

def periodic_csv_dump(interval_seconds=60):
    while True:
        time.sleep(interval_seconds)
        try:
            write_candle_data_to_csv(global_data.candle_data)
            logger.info("Wrote candle data to CSV")
        except Exception as e:
            logger.error(f"Failed to write candle data to CSV: {e}")

def monitor_connection(ws_manager):
    while not global_data.ws_connected:
        time.sleep(1)
    
    while True:
        if time.time() - ws_manager.last_message_time > 60:
            logger.warning("No WebSocket messages for 60s, triggering reconnect")
            try:
                ws_manager.stop()
                ws_manager.start(global_data.symbols, global_data.time_frames)
            except Exception as e:
                logger.error(f"WebSocket reconnect failed: {e}")
        time.sleep(5)

if __name__ == '__main__':
    ws_manager = BybitWebSocketManager()
    atexit.register(ws_manager.stop)
    
    # Start threads
    threading.Thread(target=periodic_csv_dump, daemon=True).start()
    monitor_thread = threading.Thread(
        target=monitor_connection,
        args=(ws_manager,),
        daemon=True
    )
    monitor_thread.start()
        
    # Start WebSocket
    try:
        ws_manager.start(global_data.symbols, global_data.time_frames)
    except Exception as e:
        logger.error(f"Failed to start WebSocket: {e}")
    
    threading.Thread(target=fetch_historical_data, daemon=True).start()

    try:
        runner = StrategyRunner()
        threading.Thread(target=runner.run_strategy_loop, daemon=True).start()
    except Exception as e:
        logger.error(f"Failed to initialize strategy: {e}")
    #threading.Thread(target=run_strategy_loop, daemon=True).start()

    # GUI must run in main thread
    TradingBotApp().run()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully")
