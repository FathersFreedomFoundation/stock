import yfinance as yf
import pandas as pd

# Download historical data for a specific stock (e.g., Apple)
data = yf.download('AAPL', start='2010-01-01', end='2023-09-01')

# Save the data as a CSV
data.to_csv('stock_data.csv')
