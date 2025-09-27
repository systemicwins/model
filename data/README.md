# Model Training Data

This directory contains scripts to fetch training data for the transformer model from SEC EDGAR and Federal Reserve (FRED) APIs.

## Directory Structure

```
model/data/
├── SEC/                    # SEC filing data
│   ├── CIK*/             # Company-specific directories
│   └── *.json            # Submission metadata
├── FED/                   # Federal Reserve economic data
│   ├── *.csv             # Time series data
│   └── *_metadata.json   # Series metadata
├── edgar_fetcher.py      # SEC EDGAR API client
├── fred_fetcher.py       # FRED API client
└── requirements.txt      # Python dependencies
```

## Setup

1. **Install dependencies:**
```bash
cd model/data
pip install -r requirements.txt
```

2. **Configure API keys:**
```bash
cp .env.example .env
# Edit .env with your FRED API key
```

Get your free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html

3. **Update SEC User-Agent:**
Edit the `HEADERS` in `edgar_fetcher.py` with your contact information (required by SEC).

## Usage

### Fetch SEC Filings

```python
from edgar_fetcher import EDGARFetcher

fetcher = EDGARFetcher(output_dir="SEC")

# Fetch recent filings for a company
filings = fetcher.fetch_recent_filings(
    cik="0000320193",  # Apple
    form_types=['10-K', '10-Q', '8-K'],
    limit=5
)

# Bulk fetch for multiple companies
cik_list = ["0000320193", "0000789019", "0001318605"]
fetcher.fetch_bulk_companies(cik_list)
```

### Fetch Federal Reserve Data

```python
from fred_fetcher import FREDFetcher

fetcher = FREDFetcher(output_dir="FED")

# Fetch specific series
fed_funds = fetcher.fetch_series('DFF', start_date='2020-01-01')

# Fetch all interest rates
interest_rates = fetcher.fetch_all_interest_rates(start_date='2000-01-01')

# Create complete training dataset
training_data = fetcher.create_training_dataset(start_date='2000-01-01')
```

## Rate Limits

### SEC EDGAR
- **Limit:** 10 requests per second
- **User-Agent:** Required with contact info
- Automatic retry with exponential backoff
- Courtesy delays between companies

### FRED API
- **Limit:** 120 requests per minute
- **API Key:** Required (free)
- Automatic rate limiting
- Batch fetching supported

## Available Data Series

### Interest Rates (FRED)
- Federal Funds Rate (DFF, FEDFUNDS)
- Treasury Yields (3MO, 6MO, 1Y, 2Y, 5Y, 10Y, 30Y)
- Prime Rate (DPRIME)
- SOFR Rate
- Discount Rate
- Corporate Bond Yields (AAA, BAA)

### Economic Indicators (FRED)
- Consumer Price Index (CPI)
- Unemployment Rate
- GDP (Nominal and Real)
- Industrial Production
- Nonfarm Payrolls
- Housing Starts
- Exchange Rates
- Commodity Prices

### SEC Filings
- 10-K (Annual Reports)
- 10-Q (Quarterly Reports)
- 8-K (Current Reports)
- DEF 14A (Proxy Statements)
- S-1 (Registration Statements)

## Data Format

### FRED Data
- CSV files with date index
- Columns: date, value
- Metadata in JSON format
- Combined datasets available

### SEC Data
- Raw filing documents (HTML/TXT)
- Submission metadata in JSON
- Organized by CIK (company ID)

## Training Dataset Creation

The `fred_fetcher.py` script can create a combined training dataset:

```python
# Creates training_dataset_YYYY-MM-DD.csv
training_data = fetcher.create_training_dataset(start_date='2000-01-01')
```

This combines:
- All interest rate series
- Economic indicators
- Forward-filled missing values
- Date-aligned observations

## Example Scripts

Run the example fetchers:

```bash
# Fetch sample SEC data
python edgar_fetcher.py

# Fetch sample FRED data (requires API key)
python fred_fetcher.py
```

## Notes

- Data is cached locally to avoid redundant API calls
- Both fetchers implement exponential backoff for reliability
- SEC requires proper User-Agent identification
- FRED data goes back decades for most series
- Consider data storage requirements for bulk fetching