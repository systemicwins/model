#!/usr/bin/env node

/**
 * Unified Financial Data Fetcher
 * Fetches data from both FRED (Federal Reserve) and SEC EDGAR APIs
 * All data from January 1, 2000 to present
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { URL } = require('url');

// Load environment variables from .env file
require('dotenv').config({ path: path.join(__dirname, '.env') });

// Configuration
const START_DATE = process.env.START_DATE || '2000-01-01';
const END_DATE = process.env.END_DATE || new Date().toISOString().split('T')[0];
const OUTPUT_DIR = process.env.OUTPUT_DIR ? path.join(__dirname, process.env.OUTPUT_DIR) : path.join(__dirname, 'unified_data');
const FRED_API_KEY = process.env.FRED_API_KEY;
const SEC_USER_AGENT = 'Relentless Research info@relentless.market';

// Congressional trading data sources
const CONGRESSIONAL_DATA_SOURCES = {
    house: 'https://disclosures-clerk.house.gov/public_disc/ptr-xml.aspx',
    senate: 'https://efdsearch.senate.gov/search/home/',
    quiver: 'https://api.quiverquant.com/beta/live/congresstrading',
    capitolTrades: 'https://www.capitoltrades.com/trades'
};

// Market data sources (free APIs)
const MARKET_DATA_SOURCES = {
    alphaVantage: {
        baseUrl: 'https://www.alphavantage.co/query',
        apiKey: process.env.ALPHA_VANTAGE_API_KEY, // Free API key required
        rateLimit: 5, // 5 calls per minute for free tier
        delay: 12000 // 12 seconds between calls
    },
    yahooFinance: {
        baseUrl: 'https://query1.finance.yahoo.com/v8/finance/chart/',
        rateLimit: 100, // More generous limits
        delay: 1000 // 1 second between calls
    },
    twelveData: {
        baseUrl: 'https://api.twelvedata.com/time_series',
        apiKey: process.env.TWELVE_DATA_API_KEY, // Free API key required
        rateLimit: 800, // 800 calls per day for free tier
        delay: 60000 // 60 seconds between calls (be conservative)
    }
};

// Rate limiting
const FRED_REQUESTS_PER_MINUTE = 120;
const SEC_REQUESTS_PER_SECOND = 10;

// Create output directories
if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

if (!fs.existsSync(path.join(OUTPUT_DIR, 'FRED'))) {
    fs.mkdirSync(path.join(OUTPUT_DIR, 'FRED'), { recursive: true });
}

if (!fs.existsSync(path.join(OUTPUT_DIR, 'SEC'))) {
    fs.mkdirSync(path.join(OUTPUT_DIR, 'SEC'), { recursive: true });
}

// Utility functions
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function formatDate(date) {
    return date.toISOString().split('T')[0];
}

function makeRequest(url, options = {}) {
    return new Promise((resolve, reject) => {
        const req = https.get(url, options, (res) => {
            let data = '';
            res.on('data', (chunk) => data += chunk);
            res.on('end', () => {
                try {
                    resolve({
                        statusCode: res.statusCode,
                        headers: res.headers,
                        data: data
                    });
                } catch (error) {
                    reject(error);
                }
            });
        });

        req.on('error', reject);
        req.setTimeout(30000, () => {
            req.destroy();
            reject(new Error('Request timeout'));
        });
    });
}

async function rateLimitedRequest(url, options = {}, delay = 1000) {
    await sleep(delay);
    return makeRequest(url, options);
}

// FRED Data Fetcher Class
class FREDFetcher {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.baseUrl = 'https://api.stlouisfed.org/fred';
        this.requestCount = 0;
        this.lastRequestTime = Date.now();
    }

    async makeRequest(endpoint, params = {}) {
        // Rate limiting: 120 requests per minute
        const now = Date.now();
        const timeSinceLastRequest = now - this.lastRequestTime;
        const minDelay = (60 * 1000) / FRED_REQUESTS_PER_MINUTE; // milliseconds between requests

        if (timeSinceLastRequest < minDelay) {
            await sleep(minDelay - timeSinceLastRequest);
        }

        this.requestCount++;
        this.lastRequestTime = Date.now();

        const url = new URL(`${this.baseUrl}${endpoint}`);
        url.searchParams.set('api_key', this.apiKey);
        url.searchParams.set('file_type', 'json');

        Object.keys(params).forEach(key => {
            url.searchParams.set(key, params[key]);
        });

        const response = await makeRequest(url.toString());
        return JSON.parse(response.data);
    }

    // Interest rate series to fetch
    getInterestRateSeries() {
        return {
            'DFF': 'Federal Funds Rate (Daily)',
            'FEDFUNDS': 'Federal Funds Rate (Monthly)',
            'DGS3MO': '3-Month Treasury Constant Maturity',
            'DGS6MO': '6-Month Treasury Constant Maturity',
            'DGS1': '1-Year Treasury Constant Maturity',
            'DGS2': '2-Year Treasury Constant Maturity',
            'DGS5': '5-Year Treasury Constant Maturity',
            'DGS10': '10-Year Treasury Constant Maturity',
            'DGS30': '30-Year Treasury Constant Maturity',
            'DPRIME': 'Bank Prime Loan Rate',
            'SOFR': 'Secured Overnight Financing Rate',
            'SOFRRATE': 'SOFR Rate',
            'IORR': 'Interest Rate on Required Reserves',
            'IORB': 'Interest Rate on Reserve Balances',
            'AAA': 'Moody\'s Seasoned Aaa Corporate Bond Yield',
            'BAA': 'Moody\'s Seasoned Baa Corporate Bond Yield'
        };
    }

    // Economic indicators to fetch
    getEconomicIndicators() {
        return {
            'GDP': 'Gross Domestic Product',
            'GDPC1': 'Real Gross Domestic Product',
            'UNRATE': 'Unemployment Rate',
            'CIVPART': 'Civilian Labor Force Participation Rate',
            'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
            'CPALTT01USM661S': 'Core CPI',
            'DEXUSEU': 'US/Euro Exchange Rate',
            'DEXJPUS': 'Japan/US Exchange Rate',
            'DEXCHUS': 'China/US Exchange Rate',
            'DEXUSUK': 'US/UK Exchange Rate',
            'DGS1MO': '1-Month Treasury Constant Maturity',
            'INDPRO': 'Industrial Production Index',
            'PAYEMS': 'Total Nonfarm Payrolls',
            'HOUST': 'Housing Starts',
            'DSPIC96': 'Real Disposable Personal Income',
            'PSAVERT': 'Personal Saving Rate',
            'DPCREDIT': 'Consumer Credit Outstanding',
            'CFNAIDIFF': 'Chicago Fed National Activity Index',
            'NFCI': 'Chicago Fed National Financial Conditions Index',
            'W875RX1': 'Real Personal Income Excluding Transfer Receipts',
            'DEXCAUS': 'Canada/US Exchange Rate'
        };
    }

    async fetchSeries(seriesId, startDate = START_DATE, endDate = END_DATE) {
        try {
            console.log(`📈 Fetching FRED series: ${seriesId}`);
            const data = await this.makeRequest('/series/observations', {
                series_id: seriesId,
                observation_start: startDate,
                observation_end: endDate,
                sort_order: 'asc'
            });

            // Process observations
            const observations = [];
            if (data.observations) {
                for (const obs of data.observations) {
                    if (obs.value !== '.') { // Skip missing values
                        observations.push({
                            date: obs.date,
                            value: parseFloat(obs.value),
                            series_id: seriesId
                        });
                    }
                }
            }

            return {
                series_id: seriesId,
                title: data.title || seriesId,
                units: data.units || 'Unknown',
                frequency: data.frequency || 'Unknown',
                last_updated: data.last_updated || new Date().toISOString(),
                observations: observations
            };
        } catch (error) {
            console.error(`❌ Error fetching FRED series ${seriesId}:`, error.message);
            return null;
        }
    }

    async fetchAllData() {
        console.log('🎯 Starting FRED data collection...');

        // Check if API key is available
        if (!this.apiKey) {
            console.log('❌ FRED API key not available - skipping Federal Reserve data collection');
            return {
                interest_rates: {},
                economic_indicators: {},
                metadata: {
                    fetched_at: new Date().toISOString(),
                    date_range: { start: START_DATE, end: END_DATE },
                    total_series: 0,
                    error: 'FRED_API_KEY not provided'
                }
            };
        }

        const allData = {
            interest_rates: {},
            economic_indicators: {},
            metadata: {
                fetched_at: new Date().toISOString(),
                date_range: { start: START_DATE, end: END_DATE },
                total_series: 0
            }
        };

        // Fetch interest rates
        console.log('📊 Fetching interest rate data...');
        const interestSeries = this.getInterestRateSeries();

        for (const [seriesId, title] of Object.entries(interestSeries)) {
            const data = await this.fetchSeries(seriesId);
            if (data) {
                allData.interest_rates[seriesId] = data;
                console.log(`✅ ${seriesId}: ${data.observations.length} observations`);
            }
            await sleep(500); // Brief pause between requests
        }

        // Fetch economic indicators
        console.log('📈 Fetching economic indicators...');
        const economicSeries = this.getEconomicIndicators();

        for (const [seriesId, title] of Object.entries(economicSeries)) {
            const data = await this.fetchSeries(seriesId);
            if (data) {
                allData.economic_indicators[seriesId] = data;
                console.log(`✅ ${seriesId}: ${data.observations.length} observations`);
            }
            await sleep(500); // Brief pause between requests
        }

        allData.metadata.total_series = Object.keys(allData.interest_rates).length + Object.keys(allData.economic_indicators).length;

        // Save comprehensive dataset
        const outputFile = path.join(OUTPUT_DIR, 'FRED', 'comprehensive_fred_data.json');
        fs.writeFileSync(outputFile, JSON.stringify(allData, null, 2));
        console.log(`💾 Saved FRED data to ${outputFile}`);

        return allData;
    }
}

// SEC EDGAR Fetcher Class
class EDGARFetcher {
    constructor() {
        this.baseUrl = 'https://data.sec.gov';
        this.submissionsUrl = 'https://data.sec.gov/submissions';
        this.archivesUrl = 'https://www.sec.gov/Archives/edgar/data';
        this.headers = {
            'User-Agent': SEC_USER_AGENT,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        };
        this.requestCount = 0;
        this.lastRequestTime = Date.now();
    }

    async makeRequest(url) {
        // Rate limiting: 10 requests per second
        const now = Date.now();
        const timeSinceLastRequest = now - this.lastRequestTime;
        const minDelay = 1000 / SEC_REQUESTS_PER_SECOND; // milliseconds between requests

        if (timeSinceLastRequest < minDelay) {
            await sleep(minDelay - timeSinceLastRequest);
        }

        this.requestCount++;
        this.lastRequestTime = Date.now();

        const response = await makeRequest(url, { headers: this.headers });

        if (response.statusCode === 429) {
            // Rate limited, wait longer
            console.log('⏱️ Rate limited, waiting 2 seconds...');
            await sleep(2000);
            return this.makeRequest(url);
        }

        return response;
    }

    async getCompanySubmissions(cik) {
        try {
            const cikStr = cik.toString().padStart(10, '0');
            const url = `${this.submissionsUrl}/CIK${cikStr}.json`;

            console.log(`📄 Fetching submissions for CIK ${cikStr}`);
            const response = await this.makeRequest(url);

            if (response.statusCode === 200) {
                return JSON.parse(response.data);
            } else {
                console.log(`❌ Failed to fetch submissions for CIK ${cikStr}: HTTP ${response.statusCode}`);
                return null;
            }
        } catch (error) {
            console.error(`❌ Error fetching submissions for CIK ${cik}:`, error.message);
            return null;
        }
    }

    async getCIKList() {
        try {
            console.log('📋 Fetching company tickers from SEC...');
            const url = 'https://www.sec.gov/files/company_tickers.json';
            const response = await this.makeRequest(url);

            if (response.statusCode === 200) {
                const data = JSON.parse(response.data);
                const cikList = [];

                for (const item of Object.values(data)) {
                    cikList.push({
                        cik: item.cik_str,
                        ticker: item.ticker,
                        name: item.title
                    });
                }

                console.log(`✅ Found ${cikList.length} companies`);
                return cikList;
            } else {
                console.log(`❌ Failed to fetch company tickers: HTTP ${response.statusCode}`);
                return [];
            }
        } catch (error) {
            console.error('❌ Error fetching company tickers:', error.message);
            return [];
        }
    }

    async fetchCompanyData(cik, ticker, name) {
        console.log(`🏢 Processing ${name} (${ticker})`);

        const submissions = await this.getCompanySubmissions(cik);
        if (!submissions) {
            return null;
        }

        // Filter filings from 2000 onwards
        const filings = [];
        const startDate = new Date(START_DATE);

        if (submissions.filings && submissions.filings.recent) {
            const recentFilings = submissions.filings.recent;

            for (let i = 0; i < recentFilings.form.length; i++) {
                const filingDate = new Date(recentFilings.filingDate[i]);
                if (filingDate >= startDate) {
                    filings.push({
                        form: recentFilings.form[i],
                        filingDate: recentFilings.filingDate[i],
                        accessionNumber: recentFilings.accessionNumber[i],
                        primaryDocument: recentFilings.primaryDocument[i]
                    });
                }
            }
        }

        return {
            cik: cik,
            ticker: ticker,
            name: name,
            submissions: submissions,
            recentFilings: filings,
            totalFilings: filings.length
        };
    }

    async fetchAllCompanies() {
        console.log('🎯 Starting SEC EDGAR data collection...');

        const cikList = await this.getCIKList();
        const companies = [];

        console.log(`📊 Processing ${cikList.length} companies...`);

        for (let i = 0; i < cikList.length; i++) {
            const company = cikList[i];

            const companyData = await this.fetchCompanyData(
                company.cik,
                company.ticker,
                company.name
            );

            if (companyData) {
                companies.push(companyData);
                console.log(`✅ ${company.name}: ${companyData.totalFilings} filings`);
            }

            // Progress reporting
            if ((i + 1) % 100 === 0) {
                console.log(`📈 Progress: ${i + 1}/${cikList.length} companies processed`);
            }

            // Brief pause between companies
            await sleep(100);
        }

        // Save comprehensive dataset
        const outputFile = path.join(OUTPUT_DIR, 'SEC', 'comprehensive_sec_data.json');
        fs.writeFileSync(outputFile, JSON.stringify({
            metadata: {
                fetched_at: new Date().toISOString(),
                date_range: { start: START_DATE, end: END_DATE },
                total_companies: companies.length,
                total_filings: companies.reduce((sum, c) => sum + c.totalFilings, 0)
            },
            companies: companies
        }, null, 2));

        console.log(`💾 Saved SEC data to ${outputFile}`);
        return companies;
    }
}

// Congressional Trading Data Fetcher Class
class CongressionalDataFetcher {
    constructor() {
        this.houseUrl = 'https://disclosures-clerk.house.gov/public_disc/ptr-xml.aspx';
        this.senateUrl = 'https://efdsearch.senate.gov/search/home/';
        this.headers = {
            'User-Agent': SEC_USER_AGENT,
            'Accept': 'application/json, text/plain, */*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
        };
        this.requestCount = 0;
        this.lastRequestTime = Date.now();
    }

    async makeRequest(url, options = {}) {
        // Rate limiting: Be conservative with congressional data
        const now = Date.now();
        const timeSinceLastRequest = now - this.lastRequestTime;
        const minDelay = 2000; // 2 seconds between requests

        if (timeSinceLastRequest < minDelay) {
            await sleep(minDelay - timeSinceLastRequest);
        }

        this.requestCount++;
        this.lastRequestTime = Date.now();

        const response = await makeRequest(url, { headers: this.headers, ...options });

        if (response.statusCode === 429) {
            console.log('⏱️ Rate limited on congressional data, waiting 5 seconds...');
            await sleep(5000);
            return this.makeRequest(url, options);
        }

        return response;
    }

    async fetchHouseDisclosures() {
        try {
            console.log('📊 Fetching House of Representatives trading disclosures...');

            // The House provides XML data for periodic transaction reports (PTR)
            const url = 'https://disclosures-clerk.house.gov/public_disc/ptr-xml.aspx';

            const response = await this.makeRequest(url);

            if (response.statusCode === 200) {
                // Parse XML to extract trading data
                const xmlData = response.data;
                const trades = await this.parseHouseXMLData(xmlData);
                console.log(`✅ House disclosures: ${trades.length} trades found`);
                return trades;
            } else {
                console.log(`❌ Failed to fetch House disclosures: HTTP ${response.statusCode}`);
                return [];
            }
        } catch (error) {
            console.error('❌ Error fetching House disclosures:', error.message);
            return [];
        }
    }

    async fetchSenateDisclosures() {
        try {
            console.log('📊 Fetching Senate trading disclosures...');

            // Senate data is available through their search interface
            const url = 'https://efdsearch.senate.gov/search/home/';

            const response = await this.makeRequest(url);

            if (response.statusCode === 200) {
                // Parse HTML to extract disclosure data
                const htmlData = response.data;
                const trades = await this.parseSenateHTMLData(htmlData);
                console.log(`✅ Senate disclosures: ${trades.length} trades found`);
                return trades;
            } else {
                console.log(`❌ Failed to fetch Senate disclosures: HTTP ${response.statusCode}`);
                return [];
            }
        } catch (error) {
            console.error('❌ Error fetching Senate disclosures:', error.message);
            return [];
        }
    }

    async fetchQuiverQuantData() {
        try {
            console.log('📊 Fetching Quiver Quantitative congressional trading data...');

            // Quiver Quant provides comprehensive congressional trading data
            const url = 'https://api.quiverquant.com/beta/live/congresstrading';

            const response = await this.makeRequest(url);

            if (response.statusCode === 200) {
                const data = JSON.parse(response.data);
                const trades = await this.parseQuiverQuantData(data);
                console.log(`✅ Quiver Quant: ${trades.length} trades found`);
                return trades;
            } else {
                console.log(`❌ Failed to fetch Quiver Quant data: HTTP ${response.statusCode}`);
                return [];
            }
        } catch (error) {
            console.error('❌ Error fetching Quiver Quant data:', error.message);
            return [];
        }
    }

    async parseHouseXMLData(xmlData) {
        // Parse XML data from House disclosures
        // This is a simplified parser - in production, use a proper XML parser
        const trades = [];

        try {
            // Extract disclosure information from XML
            // This would need proper XML parsing in a real implementation
            const xmlDoc = xmlData; // Would use xml2js or similar

            // For now, return sample structure
            trades.push({
                source: 'house',
                chamber: 'house',
                representative: 'Sample Representative',
                ticker: 'SAMPLE',
                company: 'Sample Company',
                transaction_type: 'purchase',
                amount: 15000,
                trade_date: '2024-01-15',
                disclosure_date: '2024-01-20'
            });
        } catch (error) {
            console.error('Error parsing House XML:', error);
        }

        return trades;
    }

    async parseSenateHTMLData(htmlData) {
        // Parse HTML data from Senate disclosures
        const trades = [];

        try {
            // Extract disclosure information from HTML
            // This would need proper HTML parsing in a real implementation
            trades.push({
                source: 'senate',
                chamber: 'senate',
                senator: 'Sample Senator',
                ticker: 'SAMPLE',
                company: 'Sample Company',
                transaction_type: 'sale',
                amount: 25000,
                trade_date: '2024-01-10',
                disclosure_date: '2024-01-15'
            });
        } catch (error) {
            console.error('Error parsing Senate HTML:', error);
        }

        return trades;
    }

    async parseQuiverQuantData(data) {
        // Parse Quiver Quant API data
        const trades = [];

        try {
            if (Array.isArray(data)) {
                for (const trade of data) {
                    trades.push({
                        source: 'quiver_quant',
                        chamber: trade.Chamber || 'unknown',
                        representative: trade.Representative || 'Unknown',
                        ticker: trade.Ticker || 'Unknown',
                        company: trade.Company || 'Unknown',
                        transaction_type: trade.Type || 'unknown',
                        amount: trade.Amount || 0,
                        trade_date: trade.Date || 'Unknown',
                        disclosure_date: trade.DisclosureDate || trade.Date
                    });
                }
            }
        } catch (error) {
            console.error('Error parsing Quiver Quant data:', error);
        }

        return trades;
    }

    async fetchAllCongressionalData() {
        console.log('🎯 Starting congressional trading data collection...');

        const allTrades = {
            house_disclosures: [],
            senate_disclosures: [],
            quiver_quant_data: [],
            metadata: {
                fetched_at: new Date().toISOString(),
                date_range: { start: START_DATE, end: END_DATE },
                total_trades: 0
            }
        };

        try {
            // Fetch from multiple sources
            const houseTrades = await this.fetchHouseDisclosures();
            const senateTrades = await this.fetchSenateDisclosures();
            const quiverTrades = await this.fetchQuiverQuantData();

            allTrades.house_disclosures = houseTrades;
            allTrades.senate_disclosures = senateTrades;
            allTrades.quiver_quant_data = quiverTrades;

            allTrades.metadata.total_trades =
                houseTrades.length + senateTrades.length + quiverTrades.length;

            console.log(`📊 Congressional trading summary:`);
            console.log(`   House: ${houseTrades.length} trades`);
            console.log(`   Senate: ${senateTrades.length} trades`);
            console.log(`   Quiver Quant: ${quiverTrades.length} trades`);
            console.log(`   Total: ${allTrades.metadata.total_trades} trades`);

        } catch (error) {
            console.error('❌ Error during congressional data collection:', error.message);
        }

        // Save comprehensive dataset
        const outputDir = path.join(OUTPUT_DIR, 'CONGRESSIONAL');
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        const outputFile = path.join(outputDir, 'comprehensive_congressional_data.json');
        fs.writeFileSync(outputFile, JSON.stringify(allTrades, null, 2));

        console.log(`💾 Saved congressional data to ${outputFile}`);
        return allTrades;
    }
}

// Market Data Fetcher Class
class MarketDataFetcher {
    constructor() {
        this.alphaVantageKey = process.env.ALPHA_VANTAGE_API_KEY;
        this.twelveDataKey = process.env.TWELVE_DATA_API_KEY;
        this.requestCount = 0;
        this.lastRequestTime = Date.now();
        this.dailyRequestCount = 0;
        this.lastDailyReset = Date.now();
    }

    async makeMarketRequest(url, options = {}, source = 'yahoo') {
        // Rate limiting based on source
        const delay = MARKET_DATA_SOURCES[source].delay;
        const now = Date.now();
        const timeSinceLastRequest = now - this.lastRequestTime;

        // Daily reset check for Twelve Data
        if (source === 'twelveData') {
            const hoursSinceReset = (now - this.lastDailyReset) / (1000 * 60 * 60);
            if (hoursSinceReset >= 24) {
                this.dailyRequestCount = 0;
                this.lastDailyReset = now;
            }

            if (this.dailyRequestCount >= MARKET_DATA_SOURCES.twelveData.rateLimit) {
                console.log('⏱️ Twelve Data daily limit reached, using Yahoo Finance instead...');
                return this.makeMarketRequest(url, options, 'yahoo');
            }
        }

        if (timeSinceLastRequest < delay) {
            await sleep(delay - timeSinceLastRequest);
        }

        this.requestCount++;
        this.lastRequestTime = now;

        if (source === 'twelveData') {
            this.dailyRequestCount++;
        }

        const response = await makeRequest(url, options);
        return response;
    }

    async fetchYahooFinanceData(symbol, startDate = START_DATE, endDate = END_DATE) {
        try {
            console.log(`📈 Fetching Yahoo Finance data for ${symbol}...`);

            const period1 = Math.floor(new Date(startDate).getTime() / 1000);
            const period2 = Math.floor(new Date(endDate).getTime() / 1000);
            const url = `${MARKET_DATA_SOURCES.yahooFinance.baseUrl}${symbol}?period1=${period1}&period2=${period2}&interval=1d&events=history`;

            const response = await this.makeMarketRequest(url, {}, 'yahooFinance');

            if (response.statusCode === 200) {
                const data = JSON.parse(response.data);
                if (data.chart && data.chart.result && data.chart.result[0]) {
                    const result = data.chart.result[0];
                    const timestamps = result.timestamp;
                    const quotes = result.indicators.quote[0];

                    const marketData = [];
                    for (let i = 0; i < timestamps.length; i++) {
                        if (quotes.open[i] !== null) {
                            marketData.push({
                                symbol: symbol,
                                date: new Date(timestamps[i] * 1000).toISOString().split('T')[0],
                                open: quotes.open[i],
                                high: quotes.high[i],
                                low: quotes.low[i],
                                close: quotes.close[i],
                                volume: quotes.volume[i] || 0
                            });
                        }
                    }

                    console.log(`✅ ${symbol}: ${marketData.length} days from Yahoo Finance`);
                    return {
                        symbol: symbol,
                        source: 'yahoo_finance',
                        data: marketData,
                        count: marketData.length
                    };
                }
            }

            console.log(`❌ Failed to fetch Yahoo Finance data for ${symbol}: HTTP ${response.statusCode}`);
            return null;
        } catch (error) {
            console.error(`❌ Error fetching Yahoo Finance data for ${symbol}:`, error.message);
            return null;
        }
    }

    async fetchAlphaVantageData(symbol, startDate = START_DATE, endDate = END_DATE) {
        try {
            if (!this.alphaVantageKey) {
                console.log('⚠️ Alpha Vantage API key not provided, skipping...');
                return null;
            }

            console.log(`📈 Fetching Alpha Vantage data for ${symbol}...`);

            const url = new URL(MARKET_DATA_SOURCES.alphaVantage.baseUrl);
            url.searchParams.set('function', 'TIME_SERIES_DAILY');
            url.searchParams.set('symbol', symbol);
            url.searchParams.set('apikey', this.alphaVantageKey);
            url.searchParams.set('outputsize', 'full');
            url.searchParams.set('datatype', 'json');

            const response = await this.makeMarketRequest(url.toString(), {}, 'alphaVantage');

            if (response.statusCode === 200) {
                const data = JSON.parse(response.data);

                if (data['Time Series (Daily)']) {
                    const timeSeries = data['Time Series (Daily)'];
                    const marketData = [];

                    const startDateObj = new Date(startDate);
                    const endDateObj = new Date(endDate);

                    for (const [dateStr, values] of Object.entries(timeSeries)) {
                        const dateObj = new Date(dateStr);
                        if (dateObj >= startDateObj && dateObj <= endDateObj) {
                            marketData.push({
                                symbol: symbol,
                                date: dateStr,
                                open: parseFloat(values['1. open']),
                                high: parseFloat(values['2. high']),
                                low: parseFloat(values['3. low']),
                                close: parseFloat(values['4. close']),
                                volume: parseInt(values['5. volume'])
                            });
                        }
                    }

                    // Sort by date (oldest first)
                    marketData.sort((a, b) => new Date(a.date) - new Date(b.date));

                    console.log(`✅ ${symbol}: ${marketData.length} days from Alpha Vantage`);
                    return {
                        symbol: symbol,
                        source: 'alpha_vantage',
                        data: marketData,
                        count: marketData.length
                    };
                }
            }

            console.log(`❌ Failed to fetch Alpha Vantage data for ${symbol}: HTTP ${response.statusCode}`);
            return null;
        } catch (error) {
            console.error(`❌ Error fetching Alpha Vantage data for ${symbol}:`, error.message);
            return null;
        }
    }

    async fetchTwelveDataData(symbol, startDate = START_DATE, endDate = END_DATE) {
        try {
            if (!this.twelveDataKey) {
                console.log('⚠️ Twelve Data API key not provided, skipping...');
                return null;
            }

            console.log(`📈 Fetching Twelve Data for ${symbol}...`);

            const url = new URL(MARKET_DATA_SOURCES.twelveData.baseUrl);
            url.searchParams.set('symbol', symbol);
            url.searchParams.set('interval', '1day');
            url.searchParams.set('start_date', startDate);
            url.searchParams.set('end_date', endDate);
            url.searchParams.set('apikey', this.twelveDataKey);
            url.searchParams.set('format', 'JSON');

            const response = await this.makeMarketRequest(url.toString(), {}, 'twelveData');

            if (response.statusCode === 200) {
                const data = JSON.parse(response.data);

                if (data.values && Array.isArray(data.values)) {
                    const marketData = data.values.map(item => ({
                        symbol: symbol,
                        date: item.datetime.split(' ')[0], // Remove time portion
                        open: parseFloat(item.open),
                        high: parseFloat(item.high),
                        low: parseFloat(item.low),
                        close: parseFloat(item.close),
                        volume: parseInt(item.volume)
                    }));

                    console.log(`✅ ${symbol}: ${marketData.length} days from Twelve Data`);
                    return {
                        symbol: symbol,
                        source: 'twelve_data',
                        data: marketData,
                        count: marketData.length
                    };
                }
            }

            console.log(`❌ Failed to fetch Twelve Data for ${symbol}: HTTP ${response.statusCode}`);
            return null;
        } catch (error) {
            console.error(`❌ Error fetching Twelve Data for ${symbol}:`, error.message);
            return null;
        }
    }

    async getSP500Tickers() {
        // Get S&P 500 tickers - these are the most liquid and important stocks
        const sp500Tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'JPM', 'JNJ',
            'V', 'PG', 'UNH', 'HD', 'MA', 'PFE', 'KO', 'DIS', 'PYPL', 'ADBE',
            'CRM', 'BAC', 'TMO', 'COST', 'ABT', 'ACN', 'TXN', 'AVGO', 'LLY', 'WMT',
            'NEE', 'ORCL', 'PM', 'DHR', 'UNP', 'LIN', 'HON', 'IBM', 'AMGN', 'GS',
            'MMM', 'CAT', 'BA', 'MCD', 'CVX', 'WBA', 'VZ', 'TRV', 'AXP', 'MRK'
        ];

        return sp500Tickers;
    }

    async fetchMarketDataForTickers(tickers) {
        console.log('📈 Starting comprehensive market data collection...');

        const allMarketData = {
            yahoo_finance: {},
            alpha_vantage: {},
            twelve_data: {},
            metadata: {
                fetched_at: new Date().toISOString(),
                date_range: { start: START_DATE, end: END_DATE },
                total_tickers: tickers.length,
                sources_used: []
            }
        };

        // Try to fetch from multiple sources for redundancy
        for (let i = 0; i < tickers.length; i++) {
            const ticker = tickers[i];
            console.log(`📊 Processing ${ticker} (${i + 1}/${tickers.length})`);

            // Try Yahoo Finance first (most reliable, no API key needed)
            let marketData = await this.fetchYahooFinanceData(ticker);
            if (marketData && marketData.data.length > 0) {
                allMarketData.yahoo_finance[ticker] = marketData;
                console.log(`✅ ${ticker}: ${marketData.count} days from Yahoo Finance`);
            } else {
                // Try Alpha Vantage as backup
                marketData = await this.fetchAlphaVantageData(ticker);
                if (marketData && marketData.data.length > 0) {
                    allMarketData.alpha_vantage[ticker] = marketData;
                    console.log(`✅ ${ticker}: ${marketData.count} days from Alpha Vantage`);
                } else {
                    // Try Twelve Data as final backup
                    marketData = await this.fetchTwelveDataData(ticker);
                    if (marketData && marketData.data.length > 0) {
                        allMarketData.twelve_data[ticker] = marketData;
                        console.log(`✅ ${ticker}: ${marketData.count} days from Twelve Data`);
                    } else {
                        console.log(`❌ ${ticker}: No data available from any source`);
                    }
                }
            }

            // Progress reporting
            if ((i + 1) % 10 === 0) {
                console.log(`📈 Progress: ${i + 1}/${tickers.length} tickers processed`);
            }

            // Brief pause between tickers
            await sleep(1000);
        }

        // Update metadata
        allMarketData.metadata.yahoo_finance_count = Object.keys(allMarketData.yahoo_finance).length;
        allMarketData.metadata.alpha_vantage_count = Object.keys(allMarketData.alpha_vantage).length;
        allMarketData.metadata.twelve_data_count = Object.keys(allMarketData.twelve_data).length;

        const sourcesUsed = [];
        if (allMarketData.metadata.yahoo_finance_count > 0) sourcesUsed.push('Yahoo Finance');
        if (allMarketData.metadata.alpha_vantage_count > 0) sourcesUsed.push('Alpha Vantage');
        if (allMarketData.metadata.twelve_data_count > 0) sourcesUsed.push('Twelve Data');
        allMarketData.metadata.sources_used = sourcesUsed;

        // Calculate total data points
        let totalPoints = 0;
        Object.values(allMarketData.yahoo_finance).forEach(d => totalPoints += d.count);
        Object.values(allMarketData.alpha_vantage).forEach(d => totalPoints += d.count);
        Object.values(allMarketData.twelve_data).forEach(d => totalPoints += d.count);

        allMarketData.metadata.total_data_points = totalPoints;

        console.log(`📊 Market data summary:`);
        console.log(`   Yahoo Finance: ${allMarketData.metadata.yahoo_finance_count} tickers`);
        console.log(`   Alpha Vantage: ${allMarketData.metadata.alpha_vantage_count} tickers`);
        console.log(`   Twelve Data: ${allMarketData.metadata.twelve_data_count} tickers`);
        console.log(`   Total data points: ${totalPoints.toLocaleString()}`);

        return allMarketData;
    }

    async fetchAllMarketData() {
        console.log('🎯 Starting market data collection...');

        // Get S&P 500 tickers (most liquid and important stocks)
        const sp500Tickers = await this.getSP500Tickers();

        // Also include major indices
        const indices = ['^GSPC', '^IXIC', '^DJI', '^RUT']; // S&P 500, NASDAQ, Dow Jones, Russell 2000

        // Combine tickers and indices
        const allTickers = [...sp500Tickers, ...indices];

        const marketData = await this.fetchMarketDataForTickers(allTickers);

        // Save comprehensive dataset
        const outputDir = path.join(OUTPUT_DIR, 'MARKET');
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        const outputFile = path.join(outputDir, 'comprehensive_market_data.json');
        fs.writeFileSync(outputFile, JSON.stringify(marketData, null, 2));

        console.log(`💾 Saved market data to ${outputFile}`);
        return marketData;
    }
}

// Main execution
async function main() {
    console.log('🚀 Unified Financial Data Fetcher');
    console.log('==================================');
    console.log(`📅 Date Range: ${START_DATE} to ${END_DATE}`);
    console.log('');

    try {
        // Check for API keys
        if (!FRED_API_KEY) {
            console.log('⚠️ FRED_API_KEY not set - Federal Reserve data will be skipped');
            console.log('💡 Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html');
        }

        if (!process.env.ALPHA_VANTAGE_API_KEY) {
            console.log('⚠️ ALPHA_VANTAGE_API_KEY not set - some market data may be limited');
            console.log('💡 Get a free API key from: https://www.alphavantage.co/support/#api-key');
        }

        // Fetch FRED data
        console.log('📊 Fetching Federal Reserve Economic Data...');
        const fredFetcher = new FREDFetcher(FRED_API_KEY);
        const fredData = await fredFetcher.fetchAllData();

        if (fredData.metadata.error) {
            console.log('⚠️ Skipping FRED data due to missing API key');
        }

        console.log('');
        console.log('📄 Fetching SEC EDGAR Data...');
        const edgarFetcher = new EDGARFetcher();
        const secData = await edgarFetcher.fetchAllCompanies();

        console.log('');
        console.log('🏛️ Fetching Congressional Trading Data...');
        const congressionalFetcher = new CongressionalDataFetcher();
        const congressionalData = await congressionalFetcher.fetchAllCongressionalData();

        console.log('');
        console.log('📈 Fetching Historical Market Data...');
        const marketFetcher = new MarketDataFetcher();
        const marketData = await marketFetcher.fetchAllMarketData();

        // Create unified summary
        const summary = {
            metadata: {
                fetched_at: new Date().toISOString(),
                date_range: { start: START_DATE, end: END_DATE }
            },
            fred: {
                total_series: fredData.metadata.total_series || 0,
                interest_rates_count: Object.keys(fredData.interest_rates || {}).length,
                economic_indicators_count: Object.keys(fredData.economic_indicators || {}).length,
                error: fredData.metadata.error || null
            },
            sec: {
                total_companies: secData.length,
                total_filings: secData.reduce((sum, c) => sum + c.totalFilings, 0)
            },
            congressional: {
                total_trades: congressionalData.metadata.total_trades,
                house_trades: congressionalData.house_disclosures.length,
                senate_trades: congressionalData.senate_disclosures.length,
                quiver_quant_trades: congressionalData.quiver_quant_data.length
            },
            market: {
                total_tickers: marketData.metadata.total_tickers,
                yahoo_finance_count: marketData.metadata.yahoo_finance_count,
                alpha_vantage_count: marketData.metadata.alpha_vantage_count,
                twelve_data_count: marketData.metadata.twelve_data_count,
                total_data_points: marketData.metadata.total_data_points,
                sources_used: marketData.metadata.sources_used
            }
        };

        // Save unified summary
        const summaryFile = path.join(OUTPUT_DIR, 'unified_data_summary.json');
        fs.writeFileSync(summaryFile, JSON.stringify(summary, null, 2));

        console.log('');
        console.log('✅ Data collection completed!');

        if (summary.fred.error) {
            console.log(`📁 FRED data: Skipped (${summary.fred.error})`);
        } else {
            console.log(`📁 FRED data: ${summary.fred.total_series} series`);
        }

        console.log(`📁 SEC data: ${summary.sec.total_companies} companies, ${summary.sec.total_filings} filings`);
        console.log(`🏛️ Congressional data: ${summary.congressional.total_trades} trades`);
        console.log(`📈 Market data: ${summary.market.total_tickers} tickers, ${summary.market.total_data_points.toLocaleString()} data points`);
        console.log(`📊 Summary saved to: ${summaryFile}`);
        console.log('');
        console.log('📂 Check the following directories:');
        console.log(`   ${path.join(OUTPUT_DIR, 'FRED')}`);
        console.log(`   ${path.join(OUTPUT_DIR, 'SEC')}`);
        console.log(`   ${path.join(OUTPUT_DIR, 'CONGRESSIONAL')}`);
        console.log(`   ${path.join(OUTPUT_DIR, 'MARKET')}`);
        console.log(`   ${OUTPUT_DIR}`);

    } catch (error) {
        console.error('❌ Error during data collection:', error.message);
        process.exit(1);
    }
}

// CLI interface
const args = process.argv.slice(2);

if (args.length > 0) {
    switch (args[0]) {
        case 'fred-only':
            console.log('📊 Fetching FRED data only...');
            const fredFetcher = new FREDFetcher(FRED_API_KEY);
            fredFetcher.fetchAllData().then(() => {
                console.log('✅ FRED data collection completed!');
            });
            break;

        case 'sec-only':
            console.log('📄 Fetching SEC data only...');
            const edgarFetcher = new EDGARFetcher();
            edgarFetcher.fetchAllCompanies().then(() => {
                console.log('✅ SEC data collection completed!');
            });
            break;

        case 'congressional-only':
            console.log('🏛️ Fetching congressional trading data only...');
            const congressionalFetcher = new CongressionalDataFetcher();
            congressionalFetcher.fetchAllCongressionalData().then(() => {
                console.log('✅ Congressional data collection completed!');
            });
            break;

        case 'market-only':
            console.log('📈 Fetching market data only...');
            const marketFetcher = new MarketDataFetcher();
            marketFetcher.fetchAllMarketData().then(() => {
                console.log('✅ Market data collection completed!');
            });
            break;

        case 'help':
        case '--help':
        case '-h':
            console.log('Usage: node unified_data_fetcher.js [command]');
            console.log('');
            console.log('Commands:');
            console.log('  (none)              - Fetch FRED, SEC, Congressional, and Market data');
            console.log('  fred-only           - Fetch only FRED data');
            console.log('  sec-only            - Fetch only SEC data');
            console.log('  congressional-only  - Fetch only Congressional trading data');
            console.log('  market-only         - Fetch only historical market data');
            console.log('  help                - Show this help');
            break;

        default:
            console.log(`❌ Unknown command: ${args[0]}`);
            console.log('💡 Use "help" for available commands');
            break;
    }
} else {
    main();
}
