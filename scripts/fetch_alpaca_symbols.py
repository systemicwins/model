#!/usr/bin/env python3
"""
Fetch all tradeable symbols from Alpaca Markets API
"""

import requests
import json
import os
from typing import List, Dict

# Alpaca API configuration
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading endpoint

def fetch_alpaca_assets() -> List[Dict]:
    """Fetch all assets from Alpaca"""
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY
    }
    
    url = f"{ALPACA_BASE_URL}/v2/assets"
    params = {
        'status': 'active',
        'asset_class': 'us_equity'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from Alpaca: {e}")
        return []

def extract_symbols(assets: List[Dict]) -> List[str]:
    """Extract tradeable symbols from assets"""
    symbols = []
    for asset in assets:
        if asset.get('tradable', False):
            symbol = asset.get('symbol', '')
            if symbol:
                symbols.append(symbol)
                # Also add common variations
                if '.' in symbol:
                    # Handle class shares like BRK.A
                    symbols.append(symbol.replace('.', '_'))
                if '-' in symbol:
                    # Handle preferred shares and special tickers
                    symbols.append(symbol.replace('-', '_'))
    
    return sorted(list(set(symbols)))

def save_symbols(symbols: List[str], output_file: str):
    """Save symbols to file"""
    with open(output_file, 'w') as f:
        f.write("# Alpaca Tradeable Symbols\n")
        f.write(f"# Total: {len(symbols)} symbols\n")
        f.write("# Generated from Alpaca Markets API\n\n")
        for symbol in symbols:
            f.write(f"{symbol}\n")
    print(f"Saved {len(symbols)} symbols to {output_file}")

def generate_cpp_token_list(symbols: List[str], output_file: str):
    """Generate C++ code snippet with symbols"""
    with open(output_file, 'w') as f:
        f.write("// Alpaca tradeable symbols for tokenizer\n")
        f.write(f"// Total: {len(symbols)} symbols\n\n")
        f.write("std::vector<std::string> alpaca_symbols = {\n")
        
        # Write symbols in chunks for readability
        for i, symbol in enumerate(symbols):
            if i > 0:
                f.write(",")
            if i % 10 == 0:
                f.write("\n    ")
            f.write(f'"{symbol}"')
        
        f.write("\n};\n")
    print(f"Generated C++ code in {output_file}")

def main():
    # Check if API keys are set
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Warning: ALPACA_API_KEY or ALPACA_SECRET_KEY not set")
        print("Using sample common symbols instead...")
        
        # Fallback to common symbols
        symbols = [
            # Major indices and ETFs
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "EFA", "EEM", "AGG", "GLD", "SLV",
            
            # Tech giants
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC",
            "CRM", "ORCL", "ADBE", "NFLX", "PYPL", "SQ", "SHOP", "UBER", "LYFT", "SNAP",
            
            # Financial
            "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "COF", "AXP",
            "V", "MA", "BLK", "SCHW", "SPGI", "CME", "ICE", "COIN", "HOOD",
            
            # Healthcare
            "JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT", "CVS", "LLY", "MRK", "MDT",
            
            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO", "PSX", "KMI",
            
            # Consumer
            "WMT", "AMZN", "HD", "PG", "KO", "PEP", "COST", "NKE", "MCD", "SBUX",
            "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS",
            
            # Industrial
            "BA", "CAT", "HON", "UPS", "UNP", "LMT", "RTX", "GE", "MMM", "DE",
            
            # Real Estate
            "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "WELL", "AVB", "EQR", "DLR",
            
            # Crypto-related stocks
            "COIN", "MARA", "RIOT", "MSTR", "SQ", "PYPL", "GBTC", "BITO",
            
            # Meme stocks
            "GME", "AMC", "BB", "NOK", "BBBY", "CLOV", "WISH", "PLTR"
        ]
    else:
        # Fetch from Alpaca API
        print("Fetching assets from Alpaca...")
        assets = fetch_alpaca_assets()
        
        if not assets:
            print("No assets fetched. Check API credentials.")
            return
        
        print(f"Fetched {len(assets)} assets")
        symbols = extract_symbols(assets)
    
    # Save outputs
    output_dir = "/Users/alex/relentless/model/data"
    os.makedirs(output_dir, exist_ok=True)
    
    save_symbols(symbols, os.path.join(output_dir, "alpaca_symbols.txt"))
    generate_cpp_token_list(symbols, os.path.join(output_dir, "alpaca_symbols.cpp"))
    
    print(f"\nTop 20 symbols: {symbols[:20]}")
    print(f"Total tradeable symbols: {len(symbols)}")

if __name__ == "__main__":
    main()