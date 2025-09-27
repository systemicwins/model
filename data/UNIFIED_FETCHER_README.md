# 🚀 Unified Financial Data Fetcher

## 📊 Overview

This JavaScript application fetches comprehensive financial data from both **Federal Reserve Economic Data (FRED)** and **SEC EDGAR** APIs, covering all data from January 1, 2000 to present (21st century).

## 🎯 Features

### ✅ **Complete Data Coverage**
- **FRED Data**: Interest rates, economic indicators, Treasury yields, corporate bond yields
- **SEC EDGAR Data**: All company filings (10-K, 10-Q, 8-K) for 6,700+ companies
- **Time Range**: 2000-01-01 to present (25+ years of data)

### ✅ **Robust Implementation**
- **Rate Limiting**: Proper API rate limiting for both services
- **Error Handling**: Comprehensive error handling and retry logic
- **Progress Tracking**: Real-time progress reporting
- **Data Validation**: Input validation and data integrity checks

### ✅ **Flexible Usage**
- **Unified Fetch**: Get both FRED and SEC data in one command
- **Selective Fetch**: Fetch only FRED or only SEC data
- **Command Line Interface**: Easy-to-use CLI with help system

---

## 📋 Prerequisites

### **1. Node.js**
```bash
# Check if Node.js is installed
node --version

# Install Node.js 18+ if needed
# Download from https://nodejs.org/
```

### **2. API Keys Required**
```bash
# Get your free FRED API key from:
# https://fred.stlouisfed.org/docs/api/api_key.html
export FRED_API_KEY=your_fred_api_key_here

# Get your free Alpha Vantage API key from:
# https://www.alphavantage.co/support/#api-key
export ALPHA_VANTAGE_API_KEY=AH672EQB2ABNTHX8

# Optional: Get your free Twelve Data API key from:
# https://twelvedata.com/
export TWELVE_DATA_API_KEY=your_twelve_data_api_key_here
```

### **3. Permanent Environment Setup**
**Option A: Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)**
```bash
echo 'export FRED_API_KEY=your_fred_api_key_here' >> ~/.zshrc
echo 'export ALPHA_VANTAGE_API_KEY=AH672EQB2ABNTHX8' >> ~/.zshrc
echo 'export TWELVE_DATA_API_KEY=your_twelve_data_api_key_here' >> ~/.zshrc
source ~/.zshrc
```

**Option B: Create local .env file in model/data directory**
```bash
cd /Users/alex/relentless/model/data
cat > .env << 'EOF'
FRED_API_KEY=your_fred_api_key_here
ALPHA_VANTAGE_API_KEY=AH672EQB2ABNTHX8
TWELVE_DATA_API_KEY=your_twelve_data_api_key_here
EOF
```

**Option C: Set for current session only**
```bash
export FRED_API_KEY=your_fred_api_key_here
export ALPHA_VANTAGE_API_KEY=AH672EQB2ABNTHX8
export TWELVE_DATA_API_KEY=your_twelve_data_api_key_here
```

### **4. Install Dependencies**
```bash
cd /Users/alex/relentless/model/data
npm install
```

---

## 🚀 Quick Start

### **1. Full Data Collection (Recommended)**
```bash
# Fetch both FRED and SEC data
cd /Users/alex/relentless/model/data
node unified_data_fetcher.js
```

### **2. FRED Data Only**
```bash
# Interest rates and economic indicators only
node unified_data_fetcher.js fred-only
```

### **3. SEC Data Only**
```bash
# Company filings only
node unified_data_fetcher.js sec-only
```

### **4. Congressional Data Only**
```bash
# Congressional trading disclosures only
node unified_data_fetcher.js congressional-only
```

### **5. Market Data Only**
```bash
# Historical stock prices (no API key needed)
node unified_data_fetcher.js market-only
```

### **6. Show Help**
```bash
# Display available commands
node unified_data_fetcher.js help
```

---

## 📊 Data Collection Details

### **FRED Data (Federal Reserve)**
**Interest Rates:**
- Federal Funds Rate (Daily & Monthly)
- Treasury Yields (3MO, 6MO, 1Y, 2Y, 5Y, 10Y, 30Y)
- Prime Rate, SOFR Rate
- Corporate Bond Yields (AAA, BAA)

**Economic Indicators:**
- GDP (Nominal & Real)
- Unemployment Rate
- Consumer Price Index (CPI)
- Industrial Production
- Nonfarm Payrolls
- Exchange Rates (EUR, JPY, CNY, GBP, CAD)

### **SEC EDGAR Data**
**Filing Types:**
- 10-K (Annual Reports)
- 10-Q (Quarterly Reports)
- 8-K (Current Events)
- And all other filing types

**Coverage:**
- 6,700+ publicly traded companies
- All filings from 2000-present
- Complete submission metadata
- Organized by CIK (company ID)

### **🏛️ Congressional Trading Data**
**Sources:**
- House of Representatives Periodic Transaction Reports (PTR)
- Senate Financial Disclosure Reports
- Quiver Quantitative API (comprehensive congressional trading data)

**Coverage:**
- All disclosed trades by House Representatives
- All disclosed trades by Senators
- All trades from 2000-present (25+ years)
- Transaction details: ticker, amount, date, type
- Organized by chamber and representative/senator

### **📈 Historical Market Data**
**Sources:**
- Yahoo Finance (primary, no API key needed)
- Alpha Vantage (backup, free API key required)
- Twelve Data (backup, free API key required)

**Coverage:**
- S&P 500 companies (50 most liquid stocks)
- Major market indices (^GSPC, ^IXIC, ^DJI, ^RUT)
- All trading days from 2000-present (25+ years)
- OHLCV data: Open, High, Low, Close, Volume
- Multiple data sources for redundancy

---

## 📁 Output Structure

```
unified_data/
├── FRED/
│   └── comprehensive_fred_data.json
│       # All FRED series with observations
├── SEC/
│   └── comprehensive_sec_data.json
│       # All company filings and metadata
├── CONGRESSIONAL/
│   └── comprehensive_congressional_data.json
│       # All congressional trading disclosures
├── MARKET/
│   └── comprehensive_market_data.json
│       # Historical stock prices and trading data
└── unified_data_summary.json
    # High-level summary of collection
```

### **Data Format Examples**

**FRED Data:**
```json
{
  "interest_rates": {
    "DGS10": {
      "series_id": "DGS10",
      "title": "10-Year Treasury Constant Maturity Rate",
      "observations": [
        {"date": "2000-01-03", "value": 6.58, "series_id": "DGS10"},
        {"date": "2000-01-04", "value": 6.49, "series_id": "DGS10"}
      ]
    }
  }
}
```

**SEC Data:**
```json
{
  "companies": [
    {
      "cik": "0000320193",
      "ticker": "AAPL",
      "name": "Apple Inc.",
      "recentFilings": [
        {
          "form": "10-K",
          "filingDate": "2024-09-10",
          "accessionNumber": "0000320193-24-000100"
        }
      ]
    }
  ]
}
```

**Congressional Trading Data:**
```json
{
  "house_disclosures": [
    {
      "source": "house",
      "chamber": "house",
      "representative": "Nancy Pelosi",
      "ticker": "TSLA",
      "company": "Tesla Inc.",
      "transaction_type": "purchase",
      "amount": 1000000,
      "trade_date": "2024-01-15",
      "disclosure_date": "2024-01-20"
    }
  ],
  "senate_disclosures": [...],
  "quiver_quant_data": [...]
}
```

**Market Data:**
```json
{
  "yahoo_finance": {
    "AAPL": {
      "symbol": "AAPL",
      "source": "yahoo_finance",
      "data": [
        {
          "symbol": "AAPL",
          "date": "2000-01-03",
          "open": 0.85,
          "high": 0.89,
          "low": 0.79,
          "close": 0.85,
          "volume": 535796800
        }
      ],
      "count": 6250
    }
  },
  "alpha_vantage": {...},
  "twelve_data": {...}
}
```

---

## ⚙️ Configuration

### **Environment Variables**
```bash
# Required
export FRED_API_KEY=your_fred_api_key_here

# Optional (defaults shown)
export START_DATE=2000-01-01
export END_DATE=2024-09-21
export OUTPUT_DIR=./unified_data
export FRED_REQUESTS_PER_MINUTE=120
export SEC_REQUESTS_PER_SECOND=10
```

### **Command Line Options**
```bash
# Custom date range
node unified_data_fetcher.js --start-date 2000-01-01 --end-date 2024-12-31

# Custom output directory
node unified_data_fetcher.js --output-dir ./my_data
```

---

## 📈 Performance & Scale

### **API Usage**
- **FRED API**: ~170 requests for complete dataset
- **SEC API**: ~6,700 requests for all companies
- **Processing Time**: 30-60 minutes for full collection
- **Data Volume**: 100-500+ MB depending on date range

### **Rate Limiting**
- **FRED**: 120 requests/minute (automatic)
- **SEC**: 10 requests/second (automatic)
- **Retry Logic**: Exponential backoff for failures
- **Resume Capability**: Can resume interrupted collections

---

## 🛠️ Development

### **Project Structure**
```
model/data/
├── unified_data_fetcher.js    # Main application
├── package.json              # Dependencies and scripts
└── UNIFIED_FETCHER_README.md # This documentation
```

### **Adding New Data Sources**
```javascript
// Example: Add new FRED series
const newSeries = {
    'NEWCODE': 'New Economic Indicator'
};

class FREDFetcher {
    getCustomIndicators() {
        return newSeries;
    }
}
```

---

## 📋 Available Scripts

| Script | Command | Description |
|--------|---------|-------------|
| **Full Collection** | `npm start` | FRED + SEC + Congressional + Market data |
| **FRED Only** | `npm run fred` | Interest rates + indicators |
| **SEC Only** | `npm run sec` | Company filings |
| **Congressional Only** | `npm run congressional` | Congressional trading disclosures |
| **Market Only** | `npm run market` | Historical stock prices |
| **Help** | `npm run help` | Show usage information |

---

## 🎯 Use Cases

### **1. Financial Language Model Training**
- Massive corpus of financial documents
- Economic indicators for context
- Market data for analysis

### **2. Market Research**
- Historical filing patterns
- Economic trend analysis
- Company performance tracking

### **3. Risk Assessment**
- Regulatory filing analysis
- Economic indicator monitoring
- Market condition tracking

### **4. Quantitative Analysis**
- Interest rate modeling
- Economic forecasting
- Statistical analysis

### **5. Congressional Trading Analysis**
- Insider trading pattern detection
- Political influence on markets
- Regulatory impact analysis
- Portfolio performance comparison

### **6. Market Analysis & Trading**
- Historical price pattern recognition
- Technical analysis signal generation
- Market correlation studies
- Algorithmic trading strategy development
- Risk modeling and portfolio optimization

---

## 📊 Monitoring & Debugging

### **Progress Tracking**
- Real-time console output
- Request counting and timing
- Error reporting and retry status

### **Log Levels**
```bash
# Enable debug logging
DEBUG=* node unified_data_fetcher.js

# JSON-only output (quiet mode)
QUIET=true node unified_data_fetcher.js
```

---

## 🔧 Troubleshooting

### **Common Issues**
1. **FRED API Key Missing**
   ```bash
   export FRED_API_KEY=your_key_here
   ```

2. **Rate Limiting**
   - Script handles this automatically
   - Check console for "Rate limited" messages

3. **Network Issues**
   - Automatic retry with exponential backoff
   - Resume capability for interruptions

4. **Permission Errors**
   ```bash
   # Ensure output directory is writable
   chmod 755 /path/to/unified_data
   ```

---

## 📈 Data Statistics

### **Expected Results**
- **FRED Series**: 50+ economic indicators
- **SEC Companies**: 6,700+ publicly traded companies
- **Congressional Trades**: 10,000+ disclosed trades
- **Market Tickers**: 54 (S&P 500 + major indices)
- **Time Period**: 25+ years (2000-present)
- **Data Points**: Millions of observations
- **File Size**: 500-2000+ MB (with market data)

### **Sample Output**
```
✅ Data collection completed!
📁 FRED data: 52 series
📁 SEC data: 6789 companies, 1250000 filings
🏛️ Congressional data: 10000 trades
📈 Market data: 54 tickers, 1,250,000 data points
📊 Summary saved to: ./unified_data/unified_data_summary.json
```

---

## 🎉 Getting Started

1. **Install dependencies**: `npm install`
2. **Set API key**: `export FRED_API_KEY=your_key`
3. **Run collection**: `npm start`
4. **Check results**: Look in `./unified_data/` directory

**Happy data collecting! 🚀**
