#!/usr/bin/env python3
"""
SEC EDGAR API Data Fetcher
Fetches SEC filings with proper rate limiting (10 requests/second max)
"""

import os
import json
import time
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
from ratelimit import limits, sleep_and_retry
import backoff
from pathlib import Path
import csv
from urllib.parse import quote

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EDGARFetcher:
    """Fetches SEC filings from EDGAR API with rate limiting"""

    BASE_URL = "https://data.sec.gov"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
    COMPANY_SEARCH_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

    # SEC requires User-Agent with contact info
    HEADERS = {
        "User-Agent": "Relentless Research info@relentless.market",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov"
    }

    # Rate limits
    CALLS_PER_SECOND = 10  # SEC allows 10 requests/second
    BULK_FETCH_DELAY = 2   # Delay between companies in bulk fetch
    CIK_LOOKUP_DELAY = 0.5 # Delay between CIK lookups

    # Make these available at module level for decorators
    CALLS_PER_SECOND = 10

    # Cache file for CIK mappings
    CIK_CACHE_FILE = "cik_mapping.json"
    
    def __init__(self, output_dir: str = "SEC"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.cik_cache = self.load_cik_cache()

    def load_cik_cache(self) -> Dict[str, str]:
        """Load CIK cache from file"""
        cache_file = self.output_dir / self.CIK_CACHE_FILE
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_cik_cache(self):
        """Save CIK cache to file"""
        cache_file = self.output_dir / self.CIK_CACHE_FILE
        with open(cache_file, 'w') as f:
            json.dump(self.cik_cache, f, indent=2)

    def lookup_cik(self, ticker: str) -> Optional[str]:
        """
        Lookup CIK for a ticker symbol
        Uses known CIKs first, then cache, then SEC company search API

        Args:
            ticker: Stock ticker symbol

        Returns:
            CIK number as string or None if not found
        """
        # Check known CIKs first (for major companies)
        known_ciks = self.load_cik_from_cache_or_file()
        if ticker in known_ciks:
            cik = known_ciks[ticker]
            logger.info(f"Found known CIK for {ticker}: {cik}")
            return cik

        # Check cache first
        if ticker in self.cik_cache:
            return self.cik_cache[ticker]

        try:
            # Use SEC company search API
            cik = self._lookup_cik_api(ticker)
            if cik:
                self.cik_cache[ticker] = cik
                self.save_cik_cache()
                logger.info(f"Found CIK for {ticker}: {cik}")
                return cik
            else:
                logger.warning(f"No CIK found for ticker: {ticker}")
                return None

        except Exception as e:
            logger.error(f"Error looking up CIK for {ticker}: {e}")
            return None

    def _lookup_cik_api(self, ticker: str) -> Optional[str]:
        """Lookup CIK using multiple SEC API methods and fallback approaches"""
        cik = None

        # Method 1: Try the official SEC company search API
        try:
            cik = self._lookup_cik_sec_search(ticker)
            if cik:
                logger.info(f"Found CIK via SEC search API for {ticker}: {cik}")
                return cik
        except Exception as e:
            logger.warning(f"SEC search API failed for {ticker}: {e}")

        # Method 2: Try the submissions API directly with CIK lookup
        try:
            cik = self._lookup_cik_submissions_api(ticker)
            if cik:
                logger.info(f"Found CIK via submissions API for {ticker}: {cik}")
                return cik
        except Exception as e:
            logger.warning(f"Submissions API lookup failed for {ticker}: {e}")

        # Method 3: Try company search with different parameters
        try:
            cik = self._lookup_cik_company_search(ticker)
            if cik:
                logger.info(f"Found CIK via company search for {ticker}: {cik}")
                return cik
        except Exception as e:
            logger.warning(f"Company search failed for {ticker}: {e}")

        logger.warning(f"No CIK found for {ticker} via any API method")
        return None

    def _lookup_cik_sec_search(self, ticker: str) -> Optional[str]:
        """Use SEC's search API with robust parsing"""
        company_search_url = "https://www.sec.gov/edgar/search"

        params = {
            'q': ticker,
            'category': 'custom',
            'entityNameType': 'C',
            'output': 'atom',
            'count': '5'  # Get more results for better matching
        }

        try:
            response = self.session.get(company_search_url, params=params, timeout=30)
            response.raise_for_status()

            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)

            # Look for entry elements (ATOM feed format)
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')

            for entry in entries:
                # Try multiple extraction methods
                cik = self._extract_cik_from_entry(entry, ticker)
                if cik:
                    return cik

            return None

        except Exception as e:
            logger.error(f"SEC search API error for {ticker}: {e}")
            return None

    def _extract_cik_from_entry(self, entry, ticker: str) -> Optional[str]:
        """Extract CIK from XML entry with multiple strategies"""
        import re

        # Strategy 1: Look for CIK in content text
        content = entry.find('.//{http://www.w3.org/2005/Atom}content')
        if content is not None:
            content_text = content.text or ''
            cik_match = re.search(r'CIK:\s*(\d+)', content_text)
            if cik_match:
                return cik_match.group(1)

        # Strategy 2: Look for CIK in summary
        summary = entry.find('.//{http://www.w3.org/2005/Atom}summary')
        if summary is not None:
            summary_text = summary.text or ''
            cik_match = re.search(r'CIK:\s*(\d+)', summary_text)
            if cik_match:
                return cik_match.group(1)

        # Strategy 3: Look for CIK in title
        title = entry.find('.//{http://www.w3.org/2005/Atom}title')
        if title is not None:
            title_text = title.text or ''
            cik_match = re.search(r'CIK:\s*(\d+)', title_text)
            if cik_match:
                return cik_match.group(1)

        # Strategy 4: Look in links (most reliable)
        links = entry.findall('.//{http://www.w3.org/2005/Atom}link')
        for link in links:
            href = link.get('href', '')
            cik_match = re.search(r'/edgar/data/(\d+)/', href)
            if cik_match:
                return cik_match.group(1)

        # Strategy 5: Try to find any 10-digit number that could be a CIK
        all_text = ET.tostring(entry, encoding='unicode', method='text')
        cik_matches = re.findall(r'\b(\d{10})\b', all_text)
        if cik_matches:
            # Return the first 10-digit number found (most likely to be CIK)
            return cik_matches[0]

        return None

    def _lookup_cik_submissions_api(self, ticker: str) -> Optional[str]:
        """Try to use the submissions API with different CIK lookup strategies"""
        # First, try to get CIK from company tickers JSON
        try:
            company_tickers_url = "https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(company_tickers_url, timeout=30)
            response.raise_for_status()

            company_data = response.json()
            for item in company_data.values():
                if item.get('ticker', '').upper() == ticker.upper():
                    return str(item.get('cik_str', '')).zfill(10)

        except Exception as e:
            logger.warning(f"Company tickers JSON lookup failed for {ticker}: {e}")

        return None

    def _lookup_cik_company_search(self, ticker: str) -> Optional[str]:
        """Alternative company search approach"""
        company_search_url = "https://www.sec.gov/cgi-bin/browse-edgar"

        params = {
            'action': 'getcompany',
            'CIK': ticker,
            'type': '10-K',
            'dateb': '20240101',
            'owner': 'exclude',
            'count': '1'
        }

        try:
            response = self.session.get(company_search_url, params=params, timeout=30)
            response.raise_for_status()

            # Look for CIK in the HTML response
            import re
            cik_match = re.search(r'CIK=(\d{10})', response.text)
            if cik_match:
                return cik_match.group(1)

        except Exception as e:
            logger.warning(f"Company browse search failed for {ticker}: {e}")

        return None

    def load_cik_from_cache_or_file(self) -> Dict[str, str]:
        """Load CIK mappings from cache or try to load from known sources"""
        cache_file = self.output_dir / "known_ciks.json"

        # Try to load from cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and len(data) > 10:  # Ensure it's a valid CIK mapping
                        logger.info(f"Loaded {len(data)} CIK mappings from cache")
                        return data
            except Exception as e:
                logger.warning(f"Failed to load CIK cache: {e}")

        # Try to create comprehensive CIK database
        try:
            comprehensive_ciks = self._create_comprehensive_cik_database()
            if comprehensive_ciks:
                logger.info(f"Created comprehensive CIK database with {len(comprehensive_ciks)} entries")
                return comprehensive_ciks
        except Exception as e:
            logger.warning(f"Failed to create comprehensive CIK database: {e}")

        # Fallback: Use hardcoded CIKs for major companies
        logger.warning("Using fallback CIK mappings")
        return self._get_fallback_cik_mappings()

    def _create_comprehensive_cik_database(self) -> Dict[str, str]:
        """Create a comprehensive CIK database from multiple sources"""
        cik_database = {}

        # Source 1: Load existing known_ciks.json if available
        cache_file = self.output_dir / "known_ciks.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cik_database.update(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load known_ciks.json: {e}")

        # Source 2: Try to load SEC's official company tickers JSON
        try:
            cik_database.update(self._load_sec_company_tickers())
        except Exception as e:
            logger.warning(f"Could not load SEC company tickers: {e}")

        # Source 3: Add fallback mappings for common cases
        cik_database.update(self._get_fallback_cik_mappings())

        # Source 4: Try to load from local ticker files
        try:
            cik_database.update(self._load_from_ticker_files())
        except Exception as e:
            logger.warning(f"Could not load from ticker files: {e}")

        # Save the comprehensive database
        if cik_database:
            with open(cache_file, 'w') as f:
                json.dump(cik_database, f, indent=2)
            logger.info(f"Saved comprehensive CIK database with {len(cik_database)} entries")

        return cik_database

    def _load_sec_company_tickers(self) -> Dict[str, str]:
        """Load CIK mappings from SEC's official company tickers JSON"""
        cik_mappings = {}

        try:
            company_tickers_url = "https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(company_tickers_url, timeout=30)
            response.raise_for_status()

            company_data = response.json()
            for item in company_data.values():
                ticker = item.get('ticker', '').upper()
                cik = str(item.get('cik_str', '')).zfill(10)
                if ticker and cik:
                    cik_mappings[ticker] = cik

            logger.info(f"Loaded {len(cik_mappings)} CIK mappings from SEC")
        except Exception as e:
            logger.error(f"Failed to load SEC company tickers: {e}")

        return cik_mappings

    def _get_fallback_cik_mappings(self) -> Dict[str, str]:
        """Get fallback CIK mappings for major companies"""
        return {
            "AAPL": "0000320193",
            "MSFT": "0000789019",
            "GOOGL": "0001652044",
            "TSLA": "0001318605",
            "NVDA": "0001045810",
            "AMZN": "0001018724",
            "META": "0001326801",
            "NFLX": "0001065280",
            "CRM": "0001108524",
            "AMD": "0000002488",
            "INTC": "0000050863",
            "IBM": "0000051143",
            "ORCL": "0001341439",
            "CSCO": "0000858877",
            "JPM": "0000019617",
            "BAC": "0000070858",
            "WFC": "0000072971",
            "GS": "0000886982",
            "MS": "0000895421",
            "C": "0000831001",
            "V": "0001403161",
            "MA": "0001141391",
            "PYPL": "0001633917",
            "SQ": "0001512673",
            "SHOP": "0001594805",
            "SPOT": "0001639920",
            "UBER": "0001543151",
            "LYFT": "0001759509",
            "ZM": "0001585521",
            "DOCU": "0001261333",
            "ROKU": "0001428439",
            "PLTR": "0001321655",
            "SNOW": "0001640147",
            "CRWD": "0001535527",
            "NET": "0001477333",
            "DDOG": "0001561550",
            "MDB": "0001441816",
            "OKTA": "0001660134",
            "TWLO": "0001447669",
            "ZS": "0001713683",
            "PANW": "0001327567",
            "FTNT": "0001262039",
            "AKAM": "0001086222",
            "ETSY": "0001370637",
            "PINS": "0001506293",
            "SNAP": "0001564408",
            "TWTR": "0001418091",
            "FB": "0001326801",
            "TSM": "0001046179",
            "ASML": "0000937966",
            "BABA": "0001577552",
            "TCEHY": "0001293451",
            "NIO": "0001736541",
            "LI": "0001811210",
            "XPEV": "0001810997",
            "BYDDF": "0001544972",
            "JD": "0001545158",
            "PDD": "0001737806",
            "BIDU": "0001329099",
            "NTES": "0001110646",
            "WB": "0001595761",
            "MOMO": "0001610601",
            "TME": "0001744676",
            "HUYA": "0001728190",
            "DOYU": "0001762412",
            "BILI": "0001723690",
            "IQ": "0001692349",
            "YY": "0001530238",
            "VIPS": "0001529192",
            "SE": "0001703399",
            "FUTU": "0001754581",
            "TIGR": "0001799448",
            "UPST": "0001641631",
            "SOFI": "0001818874",
            "AFRM": "0001824434",
            "COIN": "0001679788",
            "MSTR": "0001050446",
            "RIOT": "0001167419",
            "MARA": "0001507605",
            "HUT": "0001840292",
            "BITF": "0001812477",
            "HIVE": "0001720421",
            "ARBK": "0001841675",
            "IREN": "0001880419",
            "WULF": "0001083301",
            "CIFR": "0001822479",
            "BTBT": "0001752828",
            "GREE": "0001830029",
            "WISH": "0001830043",
            "CVNA": "0001690820",
            "LCID": "0001811210",
            "RIVN": "0001874178",
            "F": "0000037996",
            "GM": "0001467858",
            "FCAU": "0001605484",
            "STLA": "0001605484",
            "TM": "0001094517",
            "HMC": "0000715153",
            "NSANY": "0001191331",
            "MBGYY": "0001446457",
            "BMWYY": "0001022837",
            "VLKAY": "0000931428",
            "RACE": "0001648416",
            "POAHY": "0001446702",
            "JNJ": "0000200406",
            "PFE": "0000780571",
            "MRNA": "0001682868",
            "BNTX": "0001776985",
            "NVAX": "0001000694",
            "GILD": "0000882095",
            "AMGN": "0000318154",
            "REGN": "0000872589",
            "VRTX": "0000875320",
            "ILMN": "0001110803",
            "CRSP": "0001674416",
            "EDIT": "0001650664",
            "NTLA": "0001641991",
            "BEAM": "0001745999",
            "SGEN": "0001023796",
            "EXAS": "0001124140",
            "GH": "0001576280",
            "PACB": "0001299130",
            "TXG": "0001770787",
            "A": "0001090872",
            "TMO": "0000097745",
            "DHR": "0000313616",
            "ABT": "0000001800",
            "MDT": "0001613103",
            "SYK": "0000310764",
            "BSX": "0000855787",
            "ISRG": "0001035267",
            "DXCM": "0001093557",
            "PODD": "0001145197",
            "TNDM": "0001438133",
            "ABMD": "0000815094",
            "EW": "0001099800",
            "ZBH": "0001136869",
            "SNN": "0000849395",
            "ZTS": "0001555280",
            "IDXX": "0000874716",
            "WST": "0000105770",
            "RMD": "0000943819",
            "COO": "0000711404",
            "XRAY": "0000818479",
            "ALGN": "0001097149",
            "MTD": "0001037646",
            "WAT": "0001000697",
            "PKI": "0000073124",
            "BRKR": "0001109354",
            "TECH": "0000842023",
            "BIO": "0000012208",
            "AVTR": "0001722482",
            "CTLT": "0001596783",
            "BMRN": "0001048477",
            "ALNY": "0001178670",
            "IONS": "0000874016",
            "PTCT": "0001070081",
            "SRPT": "0000873303",
            "INCY": "0000879169",
            "EXEL": "0000939767",
            "HALO": "0001159036",
            "RARE": "0001616262",
            "FOLD": "0001178879",
            "AGIO": "0001439222",
            "BLUE": "0001293971",
            "SGMO": "0001123142",
            "CRBU": "0001844048",
            "VERV": "0001840574",
            "CRIS": "0001107332",
            "ADVM": "0001501756",
            "RCEL": "0001762303",
            "ELEV": "0001783032",
            "ADPT": "0001455665",
            "RGNX": "0001547859",
            "QURE": "0001590560",
            "PSNL": "0001527753",
            "NTRA": "0001604821",
            "MYGN": "0000899923",
            "VCYT": "0001384101",
            "QDEL": "0001196853",
            "NEOG": "0000715915",
            "HSIC": "0001000228",
            "PDCO": "0000898875",
            "CAH": "0000721371",
            "MCK": "0000927653",
        }

    def _load_from_ticker_files(self) -> Dict[str, str]:
        """Load CIK mappings from local ticker files"""
        cik_mappings = {}

        # Try to load from all_tickers.json
        ticker_file = Path("/Users/alex/relentless/model/data/all_tickers.json")
        if ticker_file.exists():
            try:
                with open(ticker_file, 'r') as f:
                    ticker_data = json.load(f)
                    if isinstance(ticker_data, dict) and 'base_tickers' in ticker_data:
                        # For now, just add the first few as examples
                        # In a real implementation, you'd want to cross-reference with CIK data
                        for ticker in ticker_data['base_tickers'][:100]:  # Limit to avoid too many API calls
                            if ticker not in cik_mappings:
                                # This would need actual CIK lookup, but for now just skip
                                pass
            except Exception as e:
                logger.warning(f"Failed to load from all_tickers.json: {e}")

        return cik_mappings

    def _get_historical_ticker_mappings(self) -> Dict[str, str]:
        """Get historical ticker mappings for delisted/changed companies"""
        return {
            # Companies that changed tickers
            "FB": "0001326801",      # Facebook -> Meta
            "TWTR": "0001418091",    # Twitter -> X Corp
            "GILD": "0000882095",    # Gilead Sciences
            "CELG": "0001045520",    # Celgene (acquired by Bristol Myers)
            "AGN": "0000008848",     # Allergan (acquired by AbbVie)
            "MYL": "0000062215",     # Mylan (merged with Pfizer)
            "RTN": "0000082811",     # Raytheon (merged with United Technologies)
            "UTX": "0000101829",     # United Technologies (merged with Raytheon)
            "DWDP": "0000030554",    # DowDuPont (split into DOW, DD, CTVA)
            "MON": "0001110783",     # Monsanto (acquired by Bayer)
            "FOXA": "0001308161",    # 21st Century Fox (acquired by Disney)
            "CELGZ": "0001045520",   # Celgene (preferred shares)
            "FCAU": "0001605484",    # Fiat Chrysler (merged with PSA to form Stellantis)
            "STLA": "0001605484",    # Stellantis
            "ABMD": "0000815094",    # Abiomed (acquired by Johnson & Johnson)
            "SGEN": "0001023796",    # Seagen (acquired by Pfizer)
            "ATVI": "0000718877",    # Activision Blizzard (acquired by Microsoft)
            "CTXS": "0000877890",    # Citrix (acquired by Vista/TB)
            "VMW": "0001124610",     # VMware (acquired by Broadcom)
            "NUAN": "0001002517",    # Nuance (acquired by Microsoft)
            "ZEN": "0001450704",     # Zendesk (acquired by private equity)
            "RNG": "0001384905",     # RingCentral
            "ZM": "0001585521",      # Zoom Video Communications
            "DOCU": "0001261333",    # DocuSign
            "DBX": "0001467623",     # Dropbox
            "OKTA": "0001660134",    # Okta
            "TWLO": "0001447669",    # Twilio
            "SNAP": "0001564408",    # Snap Inc.
            "PINS": "0001506293",    # Pinterest
            "ETSY": "0001370637",    # Etsy
            "SPOT": "0001639920",    # Spotify
            "ROKU": "0001428439",    # Roku
            "SQ": "0001512673",      # Block (formerly Square)
            "SHOP": "0001594805",    # Shopify
            "CRWD": "0001535527",    # CrowdStrike
            "ZS": "0001713683",      # Zscaler
            "PANW": "0001327567",    # Palo Alto Networks
            "FTNT": "0001262039",    # Fortinet
            "NET": "0001477333",     # Cloudflare
            "SNOW": "0001640147",    # Snowflake
            "MDB": "0001441816",     # MongoDB
            "DDOG": "0001561550",    # Datadog
            "PLTR": "0001321655",    # Palantir
            "U": "0001754195",       # Unity Software
            "RBLX": "0001315098",    # Roblox
            "COIN": "0001679788",    # Coinbase
            "MSTR": "0001050446",    # MicroStrategy
            "RIOT": "0001167419",    # Riot Blockchain
            "MARA": "0001507605",    # Marathon Digital Holdings
            "HUT": "0001840292",     # Hut 8 Mining
            "BITF": "0001812477",    # Bitfarms
            "HIVE": "0001720421",    # HIVE Blockchain
            "ARBK": "0001841675",    # Argo Blockchain
            "IREN": "0001880419",    # Iris Energy
            "WULF": "0001083301",    # TeraWulf
            "CIFR": "0001822479",    # Cipher Mining
            "BTBT": "0001752828",    # Bit Digital
            "GREE": "0001830029",    # Greenidge Generation
            "WISH": "0001830043",    # ContextLogic (Wish)
            "CVNA": "0001690820",    # Carvana
            "LCID": "0001811210",    # Lucid Group
            "RIVN": "0001874178",    # Rivian
            "TSLA": "0001318605",    # Tesla
            "NVDA": "0001045810",    # NVIDIA
            "AMD": "0000002488",     # Advanced Micro Devices
            "INTC": "0000050863",    # Intel
            "QCOM": "0000804328",    # Qualcomm
            "TXN": "0000097476",     # Texas Instruments
            "AVGO": "0001730168",    # Broadcom
            "MU": "0000723125",      # Micron Technology
            "MRVL": "0001058057",    # Marvell Technology
            "SWKS": "0000004127",    # Skyworks Solutions
            "QRVO": "0001604778",    # Qorvo
            "NXPI": "0001413447",    # NXP Semiconductors
            "ADI": "0000006281",     # Analog Devices
            "MCHP": "0000827054",    # Microchip Technology
            "XLNX": "0000743988",    # Xilinx (acquired by AMD)
            "LSCC": "0000885648",    # Lattice Semiconductor
            "ON": "0001097864",      # ON Semiconductor
            "TER": "0000972162",     # Teradyne
            "ENTG": "0001101302",    # Entegris
            "AMAT": "0000006951",    # Applied Materials
            "LRCX": "0000707549",    # Lam Research
            "KLAC": "0000319201",    # KLA Corporation
            "ASML": "0000937966",    # ASML Holding
            "TSM": "0001046179",     # Taiwan Semiconductor
            "UMC": "0001033767",     # United Microelectronics
            "ASX": "0001122411",     # ASE Technology
        }

    def lookup_cik_robust(self, ticker: str) -> Optional[str]:
        """Robust CIK lookup with multiple fallback strategies"""
        original_ticker = ticker

        # Strategy 1: Check cache first (fastest)
        if ticker in self.cik_cache:
            return self.cik_cache[ticker]

        # Strategy 2: Try exact ticker match
        cik = self.lookup_cik(ticker)
        if cik:
            return cik

        # Strategy 3: Try historical ticker mappings
        historical_ciks = self._get_historical_ticker_mappings()
        if ticker in historical_ciks:
            cik = historical_ciks[ticker]
            logger.info(f"Found historical CIK mapping for {ticker}: {cik}")
            self.cik_cache[ticker] = cik
            self.save_cik_cache()
            return cik

        # Strategy 4: Try common ticker variations
        variations = [
            ticker.replace('.', ''),  # Remove dots
            ticker.split('.')[0],     # Take part before dot
            ticker.replace('-', ''),  # Remove dashes
            ticker.upper(),           # Ensure uppercase
            ticker.lower(),           # Try lowercase
        ]

        for variation in variations:
            if variation != ticker:
                cik = self.lookup_cik(variation)
                if cik:
                    logger.info(f"Found CIK via ticker variation {variation}: {cik}")
                    self.cik_cache[ticker] = cik
                    self.save_cik_cache()
                    return cik

        # Strategy 5: Try removing common suffixes
        if ticker.endswith(('A', 'B', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')):
            base_ticker = ticker[:-1]
            cik = self.lookup_cik(base_ticker)
            if cik:
                logger.info(f"Found CIK via base ticker {base_ticker}: {cik}")
                self.cik_cache[ticker] = cik
                self.save_cik_cache()
                return cik

        logger.warning(f"No CIK found for ticker {original_ticker} using any strategy")
        return None

    def _fetch_company_historical_data(self, ticker: str, form_types: List[str],
                                      start_date: datetime, end_date: datetime,
                                      max_filings: int) -> Optional[Dict]:
        """Fetch historical data for a single company"""
        try:
            # Lookup CIK using robust method
            cik = self.lookup_cik_robust(ticker)
            if not cik:
                return None

            # Fetch company submissions
            submissions = self.fetch_company_submissions(cik)
            if not submissions or 'filings' not in submissions:
                return None

            # Filter filings by date and form type
            filtered_filings = self._filter_filings_by_date_and_type(
                submissions['filings'], form_types, start_date, end_date, max_filings)

            if not filtered_filings:
                return None

            return {
                'cik': cik,
                'ticker': ticker,
                'filings': filtered_filings,
                'total_filings_found': len([f for f in submissions['filings']['recent'].get('form', [])
                                          if f in form_types]),
                'fetch_time': datetime.now().isoformat(),
                'success': True
            }

        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return None

    def _filter_filings_by_date_and_type(self, filings: Dict, form_types: List[str],
                                        start_date: datetime, end_date: datetime,
                                        max_filings: int) -> List[Dict]:
        """Filter filings by date range and form types"""
        filtered = []

        # The filings data is nested under filings.recent
        recent_filings = filings.get('recent', {})
        filing_dates = recent_filings.get('filingDate', [])
        filing_forms = recent_filings.get('form', [])

        logger.info(f"Filtering {len(filing_dates)} filings for forms {form_types}")
        logger.info(f"Date range: {start_date} to {end_date}")

        for i, (date_str, form) in enumerate(zip(filing_dates, filing_forms)):
            if form not in form_types:
                continue

            try:
                filing_date = datetime.strptime(date_str, "%Y-%m-%d")
                if filing_date < start_date or filing_date > end_date:
                    logger.debug(f"Date {date_str} is outside range {start_date} to {end_date}")
                    continue
            except ValueError as e:
                logger.warning(f"Could not parse date {date_str}: {e}")
                continue

            logger.info(f"Found matching filing: {form} on {date_str}")

            # Collect filing metadata from the recent filings
            filing = {
                'form': form,
                'filingDate': date_str,
                'accessionNumber': recent_filings.get('accessionNumber', ['' for _ in filing_dates])[i],
                'primaryDocument': recent_filings.get('primaryDocument', ['' for _ in filing_dates])[i],
                'reportDate': recent_filings.get('reportDate', ['' for _ in filing_dates])[i] if 'reportDate' in recent_filings else '',
                'size': recent_filings.get('size', [0 for _ in filing_dates])[i] if 'size' in recent_filings else 0
            }

            filtered.append(filing)

            # Limit number of filings per company if specified
            if max_filings and len(filtered) >= max_filings:
                break

        logger.info(f"Found {len(filtered)} matching filings")
        return filtered

    @sleep_and_retry
    @limits(calls=CALLS_PER_SECOND, period=1)
    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
    def fetch_company_submissions(self, cik: str) -> Dict:
        """
        Fetch company submission history

        Args:
            cik: Central Index Key (10-digit with leading zeros)

        Returns:
            Dict containing submission metadata
        """
        # Ensure CIK is 10 digits with leading zeros
        cik = str(cik).zfill(10)
        url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"

        try:
            response = self._make_request(url)
            data = response.json()

            # Save to file
            output_file = self.output_dir / f"CIK{cik}_submissions.json"
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved submissions for CIK {cik} to {output_file}")
            return data

        except Exception as e:
            logger.error(f"Error fetching CIK {cik}: {e}")
            return {}

    @sleep_and_retry
    @limits(calls=CALLS_PER_SECOND, period=1)
    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
    def _make_request(self, url: str) -> requests.Response:
        """Make rate-limited request to SEC EDGAR"""
        logger.debug(f"Fetching: {url}")
        response = self.session.get(url)
        response.raise_for_status()
        return response


def quick_historical_sample():
    """Quick sample of historical data for testing"""
    fetcher = EDGARFetcher()

    # Sample with just a few major companies for testing
    SAMPLE_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    FORM_TYPES = ['10-K', '10-Q']
    START_DATE = "2020-01-01"  # Recent years only for testing
    MAX_FILINGS = 5  # Limit for testing

    logger.info("🧪 Running QUICK HISTORICAL SAMPLE with major companies")
    logger.info(f"   📊 Companies: {SAMPLE_TICKERS}")
    logger.info(f"   📅 Date Range: {START_DATE} onwards")
    logger.info(f"   📄 Max Filings/Company: {MAX_FILINGS}")

    results = {}
    for ticker in SAMPLE_TICKERS:
        logger.info(f"Processing {ticker}...")
        try:
            result = fetcher._fetch_company_historical_data(ticker, FORM_TYPES,
                                                          datetime.strptime(START_DATE, "%Y-%m-%d"),
                                                          datetime.now(), MAX_FILINGS)
            if result:
                results[ticker] = result
                logger.info(f"  ✅ {ticker}: {len(result['filings'])} filings")
            else:
                logger.warning(f"  ❌ {ticker}: No data found")
        except Exception as e:
            logger.error(f"  ❌ {ticker}: Error - {e}")

    # Save sample results
    sample_file = Path("SEC") / "historical_sample.json"
    sample_file.parent.mkdir(exist_ok=True)
    with open(sample_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"📁 Sample results saved to {sample_file}")


def demo_small_batch():
    """Demo with a small batch of companies for testing"""
    fetcher = EDGARFetcher()

    # Small batch for testing
    sample_tickers = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'NVDA'][:3]  # First 3

    logger.info(f"Running demo with {len(sample_tickers)} companies: {sample_tickers}")

    results = {}
    for ticker in sample_tickers:
        logger.info(f"Processing {ticker}")

        try:
            cik = fetcher.lookup_cik(ticker)
            if cik:
                filings = fetcher.fetch_recent_filings(cik, ['10-K', '10-Q'], 2)
                results[ticker] = {
                    'cik': cik,
                    'filings': filings,
                    'fetch_time': datetime.now().isoformat()
                }
                logger.info(f"  Found {len(filings)} filings for {ticker}")
            else:
                logger.warning(f"  No CIK found for {ticker}")

        except Exception as e:
            logger.error(f"  Error processing {ticker}: {e}")

        time.sleep(2)  # Respectful delay

    # Save demo results
    demo_file = Path("SEC") / "demo_results.json"
    demo_file.parent.mkdir(exist_ok=True)
    with open(demo_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Demo results saved to {demo_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "sample":
            quick_historical_sample()
        elif sys.argv[1] == "demo":
            demo_small_batch()
        else:
            logger.error(f"❌ Unknown command: {sys.argv[1]}")
            logger.info("Available commands: sample, demo")
            sys.exit(1)
    else:
        logger.error("❌ No command provided")
        logger.info("Usage: python3 edgar_fetcher.py [sample|demo]")
        sys.exit(1)
