# exchange_handler.py
from pybit.unified_trading import HTTP
from utils import logger, log_opened_position
import global_data
from global_data import demo_api_key, demo_api_secret, real_api_key, real_api_secret, demo, start_test_balance, real_start_balance, current_balance
from global_data import Amount, Leverage

# Configuration
API_KEY = demo_api_key if demo else real_api_key
API_SECRET = demo_api_secret if demo else real_api_secret

# Global variables
session = None
Leverage_amounts = {}

def initialize_connection():
    """Initialize the Bybit connection"""
    global session
    try:
        session = HTTP(
            testnet=False,
            demo=demo,
            api_key=API_KEY,
            api_secret=API_SECRET,
            recv_window=5000
        )
        session.get_server_time()  # Test connection
        logger.info("Connection established successfully")
        return session
    except Exception as e:
        logger.info(f"Connection failed: {str(e)}")
        return None

def get_account_balance():
    """Returns account balance adjusted for demo mode."""
    global session
    try:
        if not session:
            initialize_connection()
        balance_data = session.get_wallet_balance(accountType="UNIFIED")
        if balance_data["retCode"] == 0:
            current_live_balance = float(balance_data["result"]["list"][0]["totalWalletBalance"])
            if demo:
                if global_data.real_start_balance == 0:
                    global_data.real_start_balance = current_live_balance
                    logger.info(f"📌 [DEMO] Real start balance recorded: {global_data.real_start_balance:.2f}")
                balance_change = current_live_balance - global_data.real_start_balance
                adjusted_balance = global_data.start_test_balance + balance_change
                logger.info(f"🧪 [DEMO] Adjusted USDT Balance = {adjusted_balance:.2f}")
                global_data.current_balance = adjusted_balance
                return adjusted_balance
            else:
                logger.info(f"💰 Live USDT Balance = {current_live_balance:.2f}")
                global_data.current_balance = current_live_balance
                return current_live_balance
        else:
            logger.info(f"❌ Failed to get wallet balance: {balance_data['retMsg']}")
            return 0.0
    except Exception as e:
        logger.info(f"❗ Error fetching account balance: {str(e)}")
        return 0.0

def set_leverage(symbol, leverage):
    """Set leverage for a symbol"""
    if not session:
        initialize_connection()
    return adjust_leverage(session, symbol, leverage)

def place_order(symbol, side, amount_usd, leverage):
    """Place market order with proper quantity"""
    if not session:
        initialize_connection()
    return place_smart_order(session, symbol, side, amount_usd, leverage)

def get_symbol_leverage(session, symbol="POPCATUSDT"):
    """Get current leverage for a specific symbol"""
    try:
        position = session.get_positions(category="linear", symbol=symbol)
        if position['retCode'] == 0 and len(position['result']['list']) > 0:
            leverage = float(position['result']['list'][0]['leverage'])
            Leverage_amounts[symbol] = leverage
            logger.info(f"Current leverage for {symbol}: {leverage}x")
            return leverage
        else:
            logger.info(f"No position found for {symbol}")
            return None
    except Exception as e:
        logger.info(f"Failed to get leverage: {str(e)}")
        return None

def adjust_leverage(session, symbol, new_leverage):
    """Adjust leverage using cached value if available"""
    try:
        current_leverage = Leverage_amounts.get(symbol)
        if current_leverage is None:
            logger.info(f"No cached leverage for {symbol}, fetching...")
            current_leverage = get_symbol_leverage(session, symbol)
            if current_leverage is None:
                return False
        if current_leverage == new_leverage:
            logger.info(f"Leverage for {symbol} already at {new_leverage}x")
            return True
        response = session.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=str(new_leverage),
            sellLeverage=str(new_leverage)
        )
        if response['retCode'] == 0:
            Leverage_amounts[symbol] = new_leverage
            logger.info(f"Leverage for {symbol} adjusted to {new_leverage}x")
            return True
        logger.info(f"Failed to adjust leverage: {response['retMsg']}")
        return False
    except Exception as e:
        logger.info(f"Error adjusting leverage: {str(e)}")
        return False

def get_market_price(session, symbol="POPCATUSDT", category="linear"):
    """Get current market price for a symbol"""
    try:
        ticker = session.get_tickers(category=category, symbol=symbol)
        if ticker['retCode'] == 0 and len(ticker['result']['list']) > 0:
            price = float(ticker['result']['list'][0]['lastPrice'])
            logger.info(f"Current {symbol} price: {price}")
            return price
        else:
            logger.info(f"No price data found for {symbol}")
            return None
    except Exception as e:
        logger.info(f"Failed to get price: {str(e)}")
        return None

def get_position_info(session, symbol="POPCATUSDT"):
    """Get position details including status and direction"""
    try:
        position = session.get_positions(category="linear", symbol=symbol)
        if position['retCode'] == 0 and len(position['result']['list']) > 0:
            pos_data = position['result']['list'][0]
            size = float(pos_data['size'])
            if size > 0:
                return {
                    'status': "open",
                    'side': "long" if pos_data['side'] == 'Buy' else "short",
                    'size': size,
                    'entry_price': float(pos_data['avgPrice']),
                    'leverage': float(pos_data['leverage']),
                    'pnl': float(pos_data['unrealisedPnl'])
                }
        return {
            'status': "closed",
            'side': None,
            'size': 0.0,
            'entry_price': 0.0,
            'leverage': 0.0,
            'pnl': 0.0
        }
    except Exception as e:
        logger.info(f"Error getting position info: {str(e)}")
        return None

def place_smart_order(session, symbol, side, amount_usd, leverage=None, tp=None, sl=None):
    try:
        # 1. Adjust leverage if specified
        if leverage is not None:
            adjust_leverage(session, symbol, leverage)
        
        # 2. Get current position info
        position = get_position_info(session, symbol)
        desired_side = side.lower()  # Normalize to lowercase
        
        # 3. Decision tree
        if position['status'] == 'open':
            current_side = position['side']
            if current_side == "long": current_side = "buy"
            if current_side == "short": current_side = "sell"
            
            if current_side == desired_side:
                logger.info(f"Position already exists: {current_side} {position['size']} {symbol}. No action taken.")
            else:
                logger.info(f"Closing opposite position: {current_side} {position['size']} {symbol}")
                return close_position(session, symbol, position)
        
        # 4. Proceed with new order
        price = get_market_price(session, symbol)
        if price is None:
            logger.info(f"Could not retrieve market price for {symbol}")
            return False
        
        # Calculate quantity
        symbol_info = session.get_instruments_info(
            category="linear",
            symbol=symbol
        )
        qty_step = float(symbol_info['result']['list'][0]['lotSizeFilter']['qtyStep'])
        
        # Ensure the order value meets the minimum requirement
        min_order_value = 5  # Minimum order value in USDT
        quantity = max(amount_usd / price, min_order_value / price)
        rounded_qty = round(quantity / qty_step) * qty_step
        rounded_qty = float(f"{rounded_qty:.8f}".rstrip('0').rstrip('.'))
        
        logger.info(f"Market order: {side} {rounded_qty} {symbol} @ ~{price}")
        
        # Calculate stop loss at 2.5%
        stop_loss_price = None
        if side.lower() == "buy":
            stop_loss_price = price * (1 - 0.025)
        elif side.lower() == "sell":
            stop_loss_price = price * (1 + 0.025)

        # Ensure stop loss price is valid
        if side.lower() == "sell" and stop_loss_price < price:
            logger.error(f"Invalid stop loss price for sell order: {stop_loss_price} (must be greater than {price})")
            return False

        # Round stop loss to proper precision (you might want to adjust this based on the symbol's tick size)
        stop_loss_price = float(f"{stop_loss_price:.2f}")

        # Place order
        order = session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(rounded_qty),
            timeInForce="GTC",
            reduceOnly=False,
            stopLoss=str(stop_loss_price)
        )
        
        if order['retCode'] == 0:
            logger.info(f"Order executed: {order['result']['orderId']}")
            # Log opened position
            action = "open_long" if side.lower() == "buy" else "open_short"
            log_opened_position(symbol, action, price, amount_usd)
            return True
        
        logger.error(f"Order failed: {order['retMsg']}")
        return False
        
    except Exception as e:
        logger.error(f"Order error: {str(e)}")
        return False

def close_position(session, symbol, position=None):
    """Close entire position using existing position data if available"""
    try:
        if position is None:
            position = get_position_info(session, symbol)
        if position['status'] != 'open':
            return True
        close_side = "Sell" if position['side'] == 'long' else "Buy"
        response = session.place_order(
            category="linear",
            symbol=symbol,
            side=close_side,
            orderType="Market",
            qty=str(position['size']),
            timeInForce="GTC",
            reduceOnly=True
        )
        if response['retCode'] == 0:
            logger.info(f"Closed {position['size']} {symbol} {position['side']} position")
            action = f"closed_{position['side']}"
            price = get_market_price(session, symbol) or position['entry_price']
            log_opened_position(symbol, action, price, position['size'] * price)
            return True
        logger.error(f"Failed to close position: {response['retMsg']}")
        return False
    except Exception as e:
        logger.error(f"Error closing position: {str(e)}")
        return False

def handle_buy_signal(symbol, tp=None, sl=None, amount_usd=100):
    """Handle buy signal (go long)"""
    global session
    if not session:
        session = initialize_connection()
        if not session:
            logger.info("Failed to initialize connection")
            return False
    symbol = symbol.split('.')[0]
    logger.info(f"Opening long position for {symbol}")
    return place_smart_order(session, symbol, "Buy", amount_usd, Leverage, tp, sl)

def handle_sell_signal(symbol, tp=None, sl=None, amount_usd=100):
    """Handle sell signal (go short)"""
    global session
    if not session:
        session = initialize_connection()
        if not session:
            logger.info("Failed to initialize connection")
            return False
    symbol = symbol.split('.')[0]
    logger.info(f"Opening short position for {symbol}")
    return place_smart_order(session, symbol, "Sell", amount_usd, Leverage, tp, sl)

