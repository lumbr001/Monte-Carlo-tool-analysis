import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from Trading.com"""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']

def calculate_daily_returns(data):
    """Calculate daily returns and relevant statistics"""
    returns = data.pct_change().dropna()
    return returns

def monte_carlo_simulation(data, days=252, simulations=1000):
    """Monte Carlo simulation for future price projections"""
    returns = calculate_daily_returns(data)
    last_price = data[-1]
    
    # Calculate drift and volatility
    u = returns.mean()
    var = returns.var()
    drift = u - (0.5 * var)
    volatility = returns.std()
    
    # Generate daily returns
    daily_returns = np.exp(drift + volatility * norm.ppf(np.random.rand(days, simulations)))
    
    # Create price paths
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = last_price
    
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    return price_paths

def calculate_probabilities(price_paths, target_price):
    """Calculate various probabilities from simulated paths"""
    final_prices = price_paths[-1]
    
    # Probability calculations
    prob_above_target = np.sum(final_prices > target_price) / len(final_prices)
    prob_below_target = 1 - prob_above_target
    avg_return = np.mean(final_prices) / price_paths[0][0] - 1
    var_95 = np.percentile(final_prices, 5)
    
    return {
        'probability_above_target': prob_above_target,
        'probability_below_target': prob_below_target,
        'average_expected_return': avg_return,
        'value_at_risk_95': var_95
    }
def get_permutations(lst):
    # Base cases
    if len(lst) == 0:
        return []  # Return empty list for empty input
    elif len(lst) == 1:
        return [lst]  # Return list with single element as the only permutation
    
    # Recursive case to generate permutations
    permutations = []
    for i in range(len(lst)):
        # Extract the current element to fix at the first position
        current_element = lst[i]
        # Remaining list after excluding the current_element
        remaining_list = lst[:i] + lst[i+1:]
        # Generate permutations of the remaining elements
        for p in get_permutations(remaining_list):
            # Combine the current_element with each permutation of the remaining elements
            permutations.append([current_element] + p)
    return permutations

# Example usage
example_list = [1, 2, 3]
result = get_permutations(example_list)
print("All permutations of", example_list, "are:")
for perm in result:
    print(perm)
    
def moving_average_analysis(data, short_window=50, long_window=200):
    """Calculate moving average crossover probabilities"""
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data
    signals['short_mavg'] = data.rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = data.rolling(window=long_window, min_periods=1).mean()
    signals['signal'] = np.where(signals['short_mavg'] > signals['long_mavg'], 1, 0)
    return signals

def plot_results(data, simulations, signals):
    """Visualize the analysis results"""
    plt.figure(figsize=(15, 10))
    
    # Plot historical data and moving averages
    plt.subplot(2, 2, 1)
    plt.plot(data, label='Price')
    plt.plot(signals['short_mavg'], label='50-day MA')
    plt.plot(signals['long_mavg'], label='200-day MA')
    plt.title('Price and Moving Averages')
    plt.legend()
    
    # Plot Monte Carlo simulations
    plt.subplot(2, 2, 2)
    plt.plot(simulations)
    plt.title('Monte Carlo Simulation Projections')
    
    # Plot final price distribution
    plt.subplot(2, 2, 3)
    plt.hist(simulations[-1], bins=50, density=True)
    plt.title('Final Price Distribution')
    
    # Plot daily returns distribution
    plt.subplot(2, 2, 4)
    returns = calculate_daily_returns(data)
    plt.hist(returns, bins=50, density=True)
    plt.title('Daily Returns Distribution')
    
    plt.tight_layout()
    plt.show()

# Main analysis function
def analyze_stock(ticker, start_date, end_date, target_price, simulations=1000, forecast_days=252):
    # Fetch and prepare data
    data = fetch_stock_data(ticker, start_date, end_date)
    
    # Perform Monte Carlo simulation
    price_paths = monte_carlo_simulation(data, days=forecast_days, simulations=simulations)
    
    # Calculate probabilities
    probabilities = calculate_probabilities(price_paths, target_price)
    
    # Moving average analysis
    ma_signals = moving_average_analysis(data)
    
    # Plot results
    plot_results(data, price_paths, ma_signals)
    
    return probabilities

# Example usage
if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2023-01-01'
    target_price = 200  # Example target price
    
    results = analyze_stock(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        target_price=target_price,
        simulations=1000,
        forecast_days=252
    )
    
    print("\nStock Analysis Results:")
    print(f"Probability of price above {target_price}: {results['probability_above_target']:.2%}")
    print(f"Probability of price below {target_price}: {results['probability_below_target']:.2%}")
    print(f"Average expected return: {results['average_expected_return']:.2%}")
    print(f"Value at Risk (95% confidence): ${results['value_at_risk_95']:.2f}")
