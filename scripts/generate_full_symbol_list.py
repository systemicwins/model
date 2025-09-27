#!/usr/bin/env python3
"""
Generate comprehensive list of tradeable symbols for tokenizer
Includes S&P 500, NASDAQ 100, Russell 2000 components, and major ETFs
"""

import json

def get_comprehensive_symbols():
    """Return a comprehensive list of major tradeable symbols"""
    
    # S&P 500 components (top representatives)
    sp500 = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "BRK.B", "UNH",
        "JNJ", "JPM", "V", "PG", "XOM", "MA", "HD", "CVX", "LLY", "ABBV",
        "PFE", "BAC", "KO", "PEP", "MRK", "AVGO", "TMO", "COST", "WMT", "DIS",
        "CSCO", "ACN", "ABT", "MCD", "VZ", "ADBE", "WFC", "CMCSA", "CRM", "NEE",
        "NKE", "TXN", "DHR", "BMY", "NFLX", "COP", "PM", "UNP", "QCOM", "RTX",
        "T", "LIN", "LOW", "INTC", "HON", "AMD", "CVS", "UPS", "INTU", "IBM",
        "MS", "SCHW", "CAT", "BA", "SPGI", "GS", "BLK", "AMGN", "DE", "SBUX",
        "PLD", "GILD", "AXP", "MDLZ", "C", "ADI", "ISRG", "AMT", "LMT", "MMC",
        "PYPL", "TJX", "SYK", "NOW", "CB", "CI", "MO", "DUK", "BDX", "VRTX",
        "ZTS", "SO", "REGN", "EOG", "SLB", "TMUS", "PGR", "AON", "BSX", "ITW",
        "MU", "ETN", "NOC", "FISV", "CSX", "HUM", "GD", "CL", "USB", "TGT",
        "MRNA", "FCX", "MMM", "PNC", "GE", "ICE", "LRCX", "EL", "F", "D",
        "MCO", "PSA", "GM", "COF", "SHW", "APD", "CCI", "FDX", "NSC", "HCA",
        "EMR", "ADP", "MET", "KLAC", "CNC", "MAR", "MCK", "ROP", "ORLY", "PH",
        "MSI", "SNPS", "CTSH", "CDNS", "KMB", "AEP", "CARR", "PAYX", "SRE", "AIG"
    ]
    
    # NASDAQ 100 additions (not in S&P 500)
    nasdaq100 = [
        "ASML", "AZN", "PDD", "MRVL", "WDAY", "PANW", "MELI", "DXCM", "MNST", "ODFL",
        "FTNT", "CTAS", "ADSK", "LULU", "CPRT", "KDP", "PCAR", "AEP", "PAYX", "ROST",
        "IDXX", "ABNB", "BIIB", "KHC", "NXPI", "MCHP", "CRWD", "EXC", "GEHC", "CSGP",
        "TEAM", "VRSK", "ANSS", "DDOG", "BKR", "ON", "CDW", "TTD", "FAST", "ZS",
        "CEG", "FANG", "WBD", "ILMN", "WBA", "XEL", "ALGN", "CTVA", "EBAY", "SGEN"
    ]
    
    # Popular ETFs
    etfs = [
        "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "EEM", "EFA", "IVV", "VEA",
        "AGG", "BND", "VWO", "IEMG", "VUG", "IJH", "IJR", "VTV", "IWF", "IWD",
        "VIG", "VYM", "GLD", "SLV", "USO", "GDX", "GDXJ", "XLF", "XLK", "XLE",
        "XLV", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE", "VNQ", "IYR", "SCHD",
        "HYG", "LQD", "TLT", "IEF", "SHY", "TIP", "EMB", "MUB", "VCIT", "VCSH",
        "ARKK", "ARKG", "ARKQ", "ARKW", "ARKF", "SOXX", "SMH", "IGV", "VGT", "XBI",
        "IBB", "ICLN", "TAN", "FAN", "PBW", "JETS", "SKYY", "HACK", "FINX", "BOTZ",
        "ROBO", "LIT", "REMX", "URA", "COPX", "SIL", "PAVE", "MJ", "YOLO", "MSOS",
        "VXX", "UVXY", "SVXY", "SQQQ", "TQQQ", "SPXU", "SPXL", "TMF", "TBT", "TMV",
        "UPRO", "UDOW", "URTY", "SOXL", "SOXS", "LABU", "LABD", "TECS", "TECL", "ERX",
        "JNUG", "JDST", "NUGT", "DUST", "GUSH", "DRIP", "FAZ", "FAS", "TNA", "TZA"
    ]
    
    # Crypto-related stocks
    crypto_stocks = [
        "COIN", "MSTR", "MARA", "RIOT", "HUT", "BITF", "HIVE", "BTBT", "CLSK", "CAN",
        "GBTC", "ETHE", "BITO", "BITI", "BTF", "BTCR", "BLOK", "BITQ", "DAPP", "LEGR"
    ]
    
    # Popular meme and retail favorites
    retail_favorites = [
        "GME", "AMC", "BB", "NOK", "BBBY", "CLOV", "WISH", "PLTR", "SOFI", "HOOD",
        "LCID", "RIVN", "NIO", "XPEV", "LI", "FSR", "RIDE", "NKLA", "GOEV", "WKHS",
        "SPCE", "RKT", "UWMC", "OPEN", "AFRM", "UPST", "ROOT", "LMND", "MILE", "PSFE",
        "TLRY", "SNDL", "ACB", "CGC", "CRON", "HEXO", "OGI", "APHA", "VFF", "GRWG"
    ]
    
    # Major international ADRs
    international = [
        "TSM", "BABA", "NVO", "SHEL", "TM", "SAP", "SNY", "NVS", "HSBC", "TD",
        "UL", "SONY", "SHOP", "RY", "BP", "TOT", "GSK", "DEO", "BTI", "RIO",
        "BHP", "VALE", "BBD", "GOLD", "AEM", "SU", "CNQ", "CP", "CNI", "BMO",
        "BNS", "CM", "BCE", "TU", "SQ", "SE", "BILI", "JD", "PDD", "NTES",
        "BIDU", "TME", "WB", "IQ", "TAL", "EDU", "LI", "XPEV", "NIO", "KNDI"
    ]
    
    # Combine all symbols
    all_symbols = list(set(
        sp500 + nasdaq100 + etfs + crypto_stocks + retail_favorites + international
    ))
    
    return sorted(all_symbols)

def generate_cpp_code():
    """Generate C++ code to add symbols to tokenizer"""
    symbols = get_comprehensive_symbols()
    
    cpp_code = '''// Add trading symbols to vocabulary
    std::vector<std::string> trading_symbols = {
'''
    
    # Add symbols in rows of 10 for readability
    for i in range(0, len(symbols), 10):
        chunk = symbols[i:i+10]
        line = '        ' + ', '.join(f'"{s}"' for s in chunk)
        if i + 10 < len(symbols):
            line += ','
        cpp_code += line + '\n'
    
    cpp_code += '''    };
    
    // Add all trading symbols to vocabulary
    for (const auto& symbol : trading_symbols) {
        if (current_id >= vocab_size_) break;
        if (token_to_id_.find(symbol) == token_to_id_.end()) {
            token_to_id_[symbol] = current_id;
            id_to_token_[current_id] = symbol;
            current_id++;
        }
    }'''
    
    return cpp_code, len(symbols)

def main():
    cpp_code, count = generate_cpp_code()
    
    # Save the C++ code
    with open('/Users/alex/relentless/model/data/trading_symbols.cpp', 'w') as f:
        f.write(cpp_code)
    
    # Save the raw symbol list
    symbols = get_comprehensive_symbols()
    with open('/Users/alex/relentless/model/data/trading_symbols.txt', 'w') as f:
        f.write("# Comprehensive Trading Symbols\n")
        f.write(f"# Total: {len(symbols)} symbols\n\n")
        for symbol in symbols:
            f.write(f"{symbol}\n")
    
    print(f"Generated {count} trading symbols")
    print(f"Files saved to:")
    print(f"  - /Users/alex/relentless/model/data/trading_symbols.cpp")
    print(f"  - /Users/alex/relentless/model/data/trading_symbols.txt")
    print(f"\nSample symbols: {symbols[:20]}")

if __name__ == "__main__":
    main()