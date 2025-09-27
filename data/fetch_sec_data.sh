#!/bin/bash

# SEC EDGAR Data Collection Script
# Fetches SEC filings for all companies in the ticker database

echo "🔍 SEC EDGAR Data Collection for Financial Transformer"
echo "====================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Function to handle errors gracefully
handle_error() {
    echo "❌ Error occurred: $1"
    echo "💡 Check the logs above for more details"
    exit 1
}

# Install dependencies if needed
echo "📦 Checking Python dependencies..."
if ! python3 -c "import requests, pandas, numpy, ratelimit, backoff, bs4, lxml, fredapi" &>/dev/null; then
    echo "Installing missing dependencies..."
    pip3 install -r requirements.txt
else
    echo "✅ All dependencies already installed"
fi

# Change to model/data directory (use script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if edgar_fetcher.py exists
if [ ! -f "edgar_fetcher.py" ]; then
    echo "❌ edgar_fetcher.py not found in current directory"
    echo "Current directory: $(pwd)"
    echo "Expected: edgar_fetcher.py should be in the same directory as this script"
    exit 1
fi

echo ""
echo "📋 AVAILABLE COMMANDS:"
echo "   sample  - Quick test with 5 major companies (2020-present)"
echo "   demo    - Demo with 3 companies (recent filings only)"
echo "   main    - FULL historical collection (2000-present, ALL companies)"
echo "   resume  - Resume interrupted collection"
echo ""

# Default to sample if no argument provided
COMMAND=${1:-sample}

echo "🎯 Starting SEC data collection with command: $COMMAND"
echo ""

case $COMMAND in
    "sample")
        echo "🧪 Running QUICK SAMPLE mode:"
        echo "   - 5 major companies (AAPL, MSFT, GOOGL, TSLA, NVDA)"
        echo "   - 2020-present filings only"
        echo "   - Fast execution (~2-3 minutes)"
        echo ""
        if ! python3 edgar_fetcher.py sample; then
            handle_error "Sample mode failed"
        fi
        ;;

    "demo")
        echo "🎬 Running DEMO mode:"
        echo "   - 3 companies (AAPL, MSFT, TSLA)"
        echo "   - Recent filings only"
        echo "   - Quick test (~1-2 minutes)"
        echo ""
        if ! python3 edgar_fetcher.py demo; then
            handle_error "Demo mode failed"
        fi
        ;;

    "main")
        echo "🚀 Running FULL HISTORICAL COLLECTION:"
        echo "   - ALL 6700+ companies in database"
        echo "   - 2000-present (21st century)"
        echo "   - ALL available filings"
        echo "   - LONG operation (48-72 hours)"
        echo ""
        echo "📊 EXPECTED RESULTS:"
        echo "   • 500,000 - 2,000,000+ SEC filings"
        echo "   • 10-50+ GB of data"
        echo "   • 670,000+ API requests"
        echo ""

        read -p "⚠️  This is a LARGE-SCALE operation. Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🎯 Starting full historical collection..."
            echo "💡 Check SEC_DATA_COLLECTION_README.md for detailed documentation"
            echo "📝 Progress will be saved and can be resumed if interrupted"
            echo ""
            if ! python3 edgar_fetcher.py; then
                handle_error "Main collection mode failed"
            fi
        else
            echo "❌ Full collection cancelled"
            echo "💡 Try 'sample' mode first to test the system"
            exit 1
        fi
        ;;

    "resume")
        echo "🔄 RESUMING INTERRUPTED COLLECTION:"
        echo "   - Continue from where previous run stopped"
        echo "   - Uses progress tracking"
        echo ""
        if ! python3 edgar_fetcher.py resume; then
            handle_error "Resume mode failed"
        fi
        ;;

    *)
        echo "❌ Unknown command: $COMMAND"
        echo "Available commands: sample, demo, main, resume"
        exit 1
        ;;
esac

echo ""
echo "✅ Data collection completed!"
echo "📁 Check model/data/SEC/ for downloaded data"
echo "📊 See historical_collection_summary.json for detailed statistics"
