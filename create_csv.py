import pandas as pd
import random

# Sample stock data
stocks = [
    ("AAPL", "Apple Inc.", "Technology"),
    ("GOOGL", "Alphabet Inc.", "Technology"),
    ("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
    ("MSFT", "Microsoft Corp.", "Technology"),
    ("TSLA", "Tesla Inc.", "Consumer Discretionary"),
    ("META", "Meta Platforms Inc.", "Communication Services"),
    ("JPM", "JPMorgan Chase & Co.", "Financials"),
    ("NVDA", "NVIDIA Corp.", "Technology"),
    ("NFLX", "Netflix Inc.", "Communication Services"),
    ("BRK.B", "Berkshire Hathaway Inc.", "Financials"),
]

# Generate random portfolio data
def generate_portfolio(n=10):
    data = []
    for symbol, name, sector in random.sample(stocks, k=n):
        shares = random.randint(5, 200)
        avg_price = round(random.uniform(100, 500), 2)
        current_price = round(avg_price * random.uniform(0.8, 1.2), 2)
        data.append({
            "Symbol": symbol,
            "Company": name,
            "Sector": sector,
            "Shares": shares,
            "Average_Buy_Price": avg_price,
            "Current_Price": current_price,
            "Market_Value": round(shares * current_price, 2),
            "Unrealized_PnL": round((current_price - avg_price) * shares, 2)
        })
    return pd.DataFrame(data)

# Create 4 portfolios
for i in range(1, 5):
    df = generate_portfolio(random.randint(5, 8))
    df.to_csv(f"portfolio_{i}.csv", index=False)

import os
os.listdir()
