# 🚀 SEC EDGAR Historical Data Collection System

## 📊 Project Overview

This system collects **ALL SEC filings from the 21st century (2000-present)** for **6,700+ publicly traded companies**. This represents a massive dataset of financial documents that will provide unprecedented training data for financial language models.

## 🎯 Scope & Scale

### **Data Collection Targets**
- **📅 Time Period**: January 1, 2000 → Present (25+ years)
- **🏢 Companies**: 6,700+ publicly traded companies
- **📄 Filing Types**: 10-K, 10-Q, 8-K (annual, quarterly, current reports)
- **📊 Estimated Filings**: 500,000 - 2,000,000+ individual filings
- **💾 Storage**: 10-50+ GB of structured data

### **API Requirements**
- **🌐 SEC EDGAR API**: 10 requests/second rate limit
- **🔑 Authentication**: Requires User-Agent header
- **⚡ Processing**: 670,000+ API calls required
- **⏱️ Timeline**: 48-72 hours for full collection

## 🚀 Quick Start Guide

### **1. Install Dependencies**
```bash
cd /Users/alex/relentless/model/data
pip3 install -r requirements.txt
```

### **2. Test the System**
```bash
# Quick test with 5 major companies (2020-present)
python3 edgar_fetcher.py sample

# Demo with 3 companies (recent filings)
python3 edgar_fetcher.py demo
```

### **3. Full Historical Collection**
```bash
# Interactive mode - requires confirmation
python3 edgar_fetcher.py

# Or use the convenience script
./fetch_sec_data.sh main
```

## 📋 Available Commands

| Command | Description | Scale | Time | Use Case |
|---------|-------------|-------|------|----------|
| `sample` | 5 major companies, 2020-present | Small | 2-3 min | Testing |
| `demo` | 3 companies, recent filings | Small | 1-2 min | Demo |
| `main` | ALL 6700+ companies, 2000-present | Massive | 48-72h | Full training data |
| `resume` | Resume interrupted collection | Variable | Variable | Recovery |

## 📁 Data Organization

### **Directory Structure**
```
SEC/
├── CIK0000320193_submissions.json  # Company submission metadata
├── CIK0000789019_submissions.json
├── ...
├── historical_results_batch_001.json  # Incremental results
├── historical_results_batch_002.json
├── ...
├── all_historical_data.json           # Complete dataset
├── collection_summary.json            # Statistics & reports
├── cik_mapping.json                   # CIK cache
├── fetch_progress.json               # Progress tracking
├── known_ciks.json                   # Built-in CIK database
└── historical_collection_summary.json # Comprehensive report
```

### **Data Format**
```json
{
  "AAPL": {
    "cik": "0000320193",
    "ticker": "AAPL",
    "filings": [
      {
        "form": "10-K",
        "filingDate": "2024-09-28",
        "accessionNumber": "0000320193-24-000123",
        "primaryDocument": "aapl-20240928.htm",
        "reportDate": "2024-09-28",
        "size": 1234567
      },
      {
        "form": "10-Q",
        "filingDate": "2024-08-03",
        "accessionNumber": "0000320193-24-000089",
        "primaryDocument": "aapl-20240803.htm",
        "reportDate": "2024-06-29",
        "size": 890123
      }
    ],
    "total_filings_found": 96,
    "fetch_time": "2025-09-21T06:07:41",
    "success": true
  }
}
```

## 🎯 Collection Features

### **✅ Robust Error Handling**
- **🔄 Automatic Retries**: Exponential backoff for failed requests
- **📊 Progress Tracking**: Save progress every 10 companies
- **🔁 Resume Capability**: Continue from interruptions
- **⚠️ Graceful Degradation**: Handle missing documents

### **✅ Rate Limiting Compliance**
- **⏱️ 10 requests/second**: Strictly respects SEC limits
- **🕐 Delays**: 0.5s between companies, 2s between batches
- **📈 Monitoring**: Real-time progress and ETA calculation

### **✅ Comprehensive Reporting**
- **📈 Success Metrics**: Detailed statistics and success rates
- **📊 Form Breakdown**: Count by filing type (10-K, 10-Q, 8-K)
- **🏢 Top Companies**: Companies with most filings
- **📅 Timeline Distribution**: Filing counts by year

### **✅ Data Quality**
- **🔍 Date Filtering**: Precise date range filtering
- **📋 Form Validation**: Only collect specified filing types
- **📝 Metadata Extraction**: Comprehensive filing information
- **🗂️ Organization**: Structured by company and date

## 🚨 Important Considerations

### **⚠️ System Requirements**
- **💻 Hardware**: Modern computer with good internet connection
- **💾 Storage**: 10-50+ GB available disk space
- **🌐 Network**: Stable internet (will make 670k+ API calls)
- **⏱️ Time**: 48-72 hours for complete collection
- **🔋 Power**: Ensure system won't sleep/hibernate

### **📋 Before Starting**
1. **Backup**: Ensure you have adequate disk space
2. **Network**: Test internet connectivity
3. **Power**: Disable sleep mode
4. **Monitoring**: Consider running in screen/tmux for remote monitoring

### **🎛️ Control & Monitoring**
- **📊 Real-time Progress**: Shows current batch, ETA, success rate
- **🔄 Resume Support**: Can restart from interruptions
- **⚡ Speed Control**: Configurable batch sizes and delays
- **📈 Statistics**: Comprehensive reporting at completion

## 📊 Expected Results

### **Sample Output Statistics**
```json
{
  "collection_summary": {
    "start_date": "2000-01-01",
    "end_date": "2025-09-21",
    "total_tickers_processed": 6700,
    "successful_companies": 6500,
    "failed_companies": 200,
    "success_rate": "97.0%",
    "total_filings_collected": 850000,
    "average_filings_per_company": 127
  },
  "form_type_breakdown": {
    "10-K": 162000,
    "10-Q": 486000,
    "8-K": 202000
  }
}
```

### **Timeline Distribution**
- **📈 2000-2010**: Building phase (~50k filings)
- **📊 2010-2020**: Growth phase (~300k filings)
- **🚀 2020-Present**: Modern era (~500k filings)

## 🎯 Use Cases

### **🤖 Machine Learning Training**
- **Language Models**: Financial document understanding
- **Classification**: Filing type and content classification
- **Information Extraction**: Financial data extraction
- **Sentiment Analysis**: Market sentiment from filings

### **📊 Financial Analysis**
- **Company Research**: Historical filing analysis
- **Trend Analysis**: Multi-year financial trends
- **Comparative Analysis**: Cross-company comparisons
- **Risk Assessment**: Financial risk indicators

### **🔍 Research Applications**
- **Academic Research**: Financial text analysis
- **Regulatory Analysis**: SEC filing patterns
- **Market Research**: Industry trend analysis
- **Compliance**: Regulatory filing studies

## 🚨 Safety & Best Practices

### **✅ SEC Compliance**
- **⏱️ Rate Limiting**: Strictly follows 10 req/sec limit
- **👤 Proper Headers**: Includes required User-Agent
- **🔄 Error Handling**: Graceful handling of failures
- **📊 Monitoring**: Real-time progress tracking

### **🛡️ System Protection**
- **💾 Incremental Saves**: Progress saved every 10 companies
- **🔄 Resume Support**: Can restart from interruptions
- **⚡ Configurable Speed**: Adjustable batch sizes and delays
- **📊 Resource Monitoring**: Memory and disk usage tracking

### **🎯 Data Quality**
- **🔍 Validation**: CIK verification and data validation
- **📅 Date Filtering**: Precise date range enforcement
- **📋 Form Filtering**: Only specified filing types
- **📝 Metadata**: Comprehensive filing information

## 🚀 Ready to Start?

1. **Test First**: Run `python3 edgar_fetcher.py sample` to verify setup
2. **Check Resources**: Ensure adequate disk space and stable internet
3. **Start Collection**: Run `python3 edgar_fetcher.py` for full collection
4. **Monitor Progress**: Check logs and progress files regularly
5. **Use Results**: Leverage this massive dataset for ML training!

**This system will provide the most comprehensive SEC filing dataset available for financial AI training!** 📈🤖

## 📞 Support

For issues or questions:
- Check the logs for detailed error messages
- Review `historical_collection_summary.json` for statistics
- Use `resume` command to continue interrupted collections
- Monitor system resources during long-running operations
