"""
Examples of how to use the new modular strategy framework
"""

from strategy_runner import StrategyRunner, create_strategy
from stochrsi_strategy import StochRSIStrategy
# from example_rsi_strategy import SimpleRSIStrategy  # Import your new strategies
import global_data

# ===================================
# Example 1: Use default StochRSI strategy
# ===================================
def run_default_strategy():
    """Run with the default StochRSI strategy"""
    runner = StrategyRunner()
    global_data.symbols = ["BTCUSDT", "ETHUSDT"]
    runner.run_strategy_loop()

# ===================================
# Example 2: Use StochRSI with custom configuration
# ===================================
def run_custom_stochrsi():
    """Run StochRSI strategy with custom parameters"""
    
    # Custom configuration
    custom_config = {
        'srsi_overbought': 80,        # Change from default 75
        'srsi_oversold': 20,          # Change from default 25
        'srsi_entry_oversold': 15,    # Change from default 10
        'risk_percent': 0.01,         # More conservative risk
        'sl_atr_mult': 1.5,           # Tighter stop loss
        'tp_atr_mult': 3.0,           # Lower take profit
    }
    
    runner = StrategyRunner(
        strategy_class=StochRSIStrategy, 
        strategy_config=custom_config
    )
    
    global_data.symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    runner.run_strategy_loop()

# ===================================
# Example 3: Use the factory function
# ===================================
def run_with_factory():
    """Use the factory function to create strategies"""
    
    # First update the factory function in strategy_runner.py to include new strategies:
    # strategies = {
    #     'stochrsi': StochRSIStrategy,
    #     'rsi': SimpleRSIStrategy,  # Add your new strategy here
    # }
    
    config = {
        'rsi_overbought': 75,
        'rsi_oversold': 25,
        'rsi_period': 21
    }
    
    try:
        strategy = create_strategy('rsi', config)  # This would create SimpleRSIStrategy
        runner = StrategyRunner(strategy_class=type(strategy), strategy_config=config)
        
        global_data.symbols = ["BTCUSDT"]
        runner.run_strategy_loop()
        
    except ValueError as e:
        print(f"Strategy creation failed: {e}")

# ===================================
# Example 4: Easy strategy switching
# ===================================
def run_different_strategies():
    """Example of how easy it is to switch between strategies"""
    
    strategies_to_test = [
        {
            'name': 'Conservative StochRSI',
            'class': StochRSIStrategy,
            'config': {
                'risk_percent': 0.005,  # Very conservative
                'sl_atr_mult': 1.0,
                'tp_atr_mult': 2.0,
            }
        },
        {
            'name': 'Aggressive StochRSI', 
            'class': StochRSIStrategy,
            'config': {
                'risk_percent': 0.03,   # More aggressive
                'sl_atr_mult': 3.0,
                'tp_atr_mult': 6.0,
            }
        }
        # You can easily add more strategies here:
        # {
        #     'name': 'Simple RSI',
        #     'class': SimpleRSIStrategy,
        #     'config': {'rsi_period': 14}
        # }
    ]
    
    # For testing, you might run each strategy for a short period
    for strategy_info in strategies_to_test:
        print(f"\n=== Testing {strategy_info['name']} ===")
        
        runner = StrategyRunner(
            strategy_class=strategy_info['class'],
            strategy_config=strategy_info['config']
        )
        
        global_data.symbols = ["BTCUSDT"]
        
        # Run for a short test period (you'd modify this for actual testing)
        # runner.run_strategy_loop()

# ===================================
# Example 5: A/B Testing Framework
# ===================================
class StrategyTester:
    """Simple framework for testing different strategy configurations"""
    
    def __init__(self):
        self.results = {}
    
    def test_strategy(self, name, strategy_class, config, symbols, duration_minutes=60):
        """Test a strategy configuration"""
        print(f"Testing {name}...")
        
        runner = StrategyRunner(strategy_class=strategy_class, strategy_config=config)
        global_data.symbols = symbols
        
        # In a real implementation, you'd run for the specified duration
        # and collect performance metrics
        
        # Placeholder for actual testing logic
        self.results[name] = {
            'config': config,
            'symbols': symbols,
            'duration': duration_minutes,
            # Add actual metrics here: win_rate, profit_loss, max_drawdown, etc.
        }
        
        print(f"Completed testing {name}")
    
    def compare_results(self):
        """Compare results from different strategy tests"""
        print("\n=== Strategy Comparison ===")
        for name, results in self.results.items():
            print(f"{name}: {results}")

# ===================================
# Main execution examples
# ===================================
if __name__ == "__main__":
    # Choose which example to run:
    
    # Example 1: Default strategy
    # run_default_strategy()
    
    # Example 2: Custom StochRSI
    # run_custom_stochrsi()
    
    # Example 3: Factory pattern
    # run_with_factory()
    
    # Example 4: Strategy switching demo
    # run_different_strategies()
    
    # Example 5: A/B testing
    tester = StrategyTester()
    
    # Test conservative vs aggressive StochRSI
    tester.test_strategy(
        "Conservative", 
        StochRSIStrategy, 
        {'risk_percent': 0.01, 'sl_atr_mult': 1.5},
        ["BTCUSDT"]
    )
    
    tester.test_strategy(
        "Aggressive", 
        StochRSIStrategy, 
        {'risk_percent': 0.03, 'sl_atr_mult': 2.5},
        ["BTCUSDT"]
    )
    
    tester.compare_results()