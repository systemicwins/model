#!/usr/bin/env python3
"""
Fetch comprehensive list of all tradeable stock tickers from multiple sources
including NYSE, NASDAQ, AMEX, and OTC markets
"""

import json
import requests
import pandas as pd
from typing import Set, List
import time

def fetch_nasdaq_listed():
    """Fetch all NASDAQ listed symbols"""
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&exchange=nasdaq"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        rows = data['data']['rows']
        return {row['symbol'] for row in rows if 'symbol' in row}
    except:
        print("Failed to fetch NASDAQ tickers")
        return set()

def fetch_nyse_listed():
    """Fetch all NYSE listed symbols"""
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&exchange=nyse"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        rows = data['data']['rows']
        return {row['symbol'] for row in rows if 'symbol' in row}
    except:
        print("Failed to fetch NYSE tickers")
        return set()

def fetch_amex_listed():
    """Fetch all AMEX listed symbols"""
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&exchange=amex"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        rows = data['data']['rows']
        return {row['symbol'] for row in rows if 'symbol' in row}
    except:
        print("Failed to fetch AMEX tickers")
        return set()

def fetch_ftp_nasdaq():
    """Fetch from NASDAQ FTP alternative source"""
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt"
    try:
        response = requests.get(url)
        tickers = response.text.strip().split('\n')
        return {t.strip() for t in tickers if t.strip()}
    except:
        print("Failed to fetch FTP NASDAQ tickers")
        return set()

def fetch_ftp_nyse():
    """Fetch from NYSE FTP alternative source"""
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.txt"
    try:
        response = requests.get(url)
        tickers = response.text.strip().split('\n')
        return {t.strip() for t in tickers if t.strip()}
    except:
        print("Failed to fetch FTP NYSE tickers")
        return set()

def fetch_ftp_amex():
    """Fetch from AMEX FTP alternative source"""
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_tickers.txt"
    try:
        response = requests.get(url)
        tickers = response.text.strip().split('\n')
        return {t.strip() for t in tickers if t.strip()}
    except:
        print("Failed to fetch FTP AMEX tickers")
        return set()

def load_russell2000():
    """Load existing Russell 2000 tickers"""
    try:
        with open('/Users/alex/relentless/client/russell2000_tickers.txt', 'r') as f:
            return {line.strip() for line in f if line.strip()}
    except:
        print("Failed to load Russell 2000 tickers")
        return set()

def fetch_polygon_tickers():
    """Fetch from Polygon.io (requires API key)"""
    # Using a free tier endpoint
    url = "https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&limit=1000"
    headers = {
        'Authorization': 'Bearer YOUR_POLYGON_API_KEY'  # Would need actual API key
    }
    # Skipping for now as it requires API key
    return set()

def fetch_sp500():
    """Fetch S&P 500 tickers"""
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    try:
        df = pd.read_csv(url)
        return set(df['Symbol'].tolist())
    except:
        print("Failed to fetch S&P 500 tickers")
        return set()

def clean_ticker(ticker: str) -> str:
    """Clean and validate ticker symbol"""
    # Remove any whitespace
    ticker = ticker.strip().upper()
    
    # Remove special suffixes for preferred shares, warrants, etc.
    # but keep them as separate entries
    base_ticker = ticker.split('.')[0].split('-')[0].split('^')[0]
    
    # Only keep alphanumeric characters
    base_ticker = ''.join(c for c in base_ticker if c.isalnum())
    
    return base_ticker

def main():
    print("Fetching comprehensive ticker list from multiple sources...")
    
    all_tickers = set()
    
    # Fetch from primary sources
    print("Fetching from NASDAQ API...")
    nasdaq_tickers = fetch_nasdaq_listed()
    print(f"  Found {len(nasdaq_tickers)} NASDAQ tickers")
    all_tickers.update(nasdaq_tickers)
    time.sleep(1)  # Rate limiting
    
    print("Fetching from NYSE API...")
    nyse_tickers = fetch_nyse_listed()
    print(f"  Found {len(nyse_tickers)} NYSE tickers")
    all_tickers.update(nyse_tickers)
    time.sleep(1)
    
    print("Fetching from AMEX API...")
    amex_tickers = fetch_amex_listed()
    print(f"  Found {len(amex_tickers)} AMEX tickers")
    all_tickers.update(amex_tickers)
    time.sleep(1)
    
    # Fetch from alternative sources
    print("Fetching from alternative sources...")
    ftp_nasdaq = fetch_ftp_nasdaq()
    print(f"  Found {len(ftp_nasdaq)} alternative NASDAQ tickers")
    all_tickers.update(ftp_nasdaq)
    
    ftp_nyse = fetch_ftp_nyse()
    print(f"  Found {len(ftp_nyse)} alternative NYSE tickers")
    all_tickers.update(ftp_nyse)
    
    ftp_amex = fetch_ftp_amex()
    print(f"  Found {len(ftp_amex)} alternative AMEX tickers")
    all_tickers.update(ftp_amex)
    
    # Add Russell 2000
    print("Loading Russell 2000 tickers...")
    russell2000 = load_russell2000()
    print(f"  Found {len(russell2000)} Russell 2000 tickers")
    all_tickers.update(russell2000)
    
    # Add S&P 500
    print("Fetching S&P 500 tickers...")
    sp500 = fetch_sp500()
    print(f"  Found {len(sp500)} S&P 500 tickers")
    all_tickers.update(sp500)
    
    # Clean all tickers
    print("\nCleaning ticker symbols...")
    cleaned_tickers = set()
    special_tickers = set()  # For warrants, preferred shares, etc.
    
    for ticker in all_tickers:
        if not ticker:
            continue
            
        # Keep original if it has special characters (warrants, preferred, etc.)
        if '.' in ticker or '-' in ticker or '^' in ticker:
            special_tickers.add(ticker)
        
        # Always add the cleaned base ticker
        cleaned = clean_ticker(ticker)
        if cleaned and len(cleaned) <= 5:  # Most tickers are 1-5 characters
            cleaned_tickers.add(cleaned)
    
    # Combine both sets
    all_final_tickers = cleaned_tickers | special_tickers
    
    # Remove empty strings and sort
    all_final_tickers = sorted([t for t in all_final_tickers if t])
    
    print(f"\nTotal unique tickers collected: {len(all_final_tickers)}")
    print(f"  Base tickers: {len(cleaned_tickers)}")
    print(f"  Special tickers (warrants, preferred, etc.): {len(special_tickers)}")
    
    # Save to file
    output_file = '/Users/alex/relentless/model/data/all_tickers.txt'
    with open(output_file, 'w') as f:
        for ticker in all_final_tickers:
            f.write(f"{ticker}\n")
    
    print(f"\nSaved {len(all_final_tickers)} tickers to {output_file}")
    
    # Also save as JSON with metadata
    ticker_data = {
        'total_count': len(all_final_tickers),
        'base_tickers': sorted(list(cleaned_tickers)),
        'special_tickers': sorted(list(special_tickers)),
        'all_tickers': all_final_tickers
    }
    
    json_file = '/Users/alex/relentless/model/data/all_tickers.json'
    with open(json_file, 'w') as f:
        json.dump(ticker_data, f, indent=2)
    
    print(f"Saved ticker data with metadata to {json_file}")
    
    return all_final_tickers

if __name__ == "__main__":
    tickers = main()
    
    # Show some examples
    print("\nSample tickers:")
    for ticker in list(tickers)[:20]:
        print(f"  {ticker}")