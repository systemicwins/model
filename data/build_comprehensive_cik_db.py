#!/usr/bin/env python3
"""
Comprehensive CIK Database Builder
Builds a robust CIK database from multiple sources including:
- SEC's official company tickers JSON
- Historical ticker mappings
- Manual corrections for delisted companies
- Common ticker variations
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, Set, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CIKDatabaseBuilder:
    def __init__(self, output_dir: str = "SEC"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

        # Set proper headers
        self.session.headers.update({
            "User-Agent": "CIK Database Builder info@relentless.market",
            "Accept-Encoding": "gzip, deflate",
        })

    def build_comprehensive_database(self) -> Dict[str, str]:
        """Build comprehensive CIK database from multiple sources"""
        cik_database = {}

        logger.info("Building comprehensive CIK database...")

        # Source 1: Load existing CIK mappings
        cik_database.update(self._load_existing_ciks())

        # Source 2: Load from SEC's official company tickers
        cik_database.update(self._load_sec_company_tickers())

        # Source 3: Add historical mappings
        cik_database.update(self._get_historical_mappings())

        # Source 4: Add common variations and corrections
        cik_database.update(self._get_common_variations())

        # Source 5: Try to resolve missing tickers via API
        cik_database.update(self._resolve_missing_tickers(cik_database))

        # Save the database
        self._save_database(cik_database)

        logger.info(f"Built comprehensive CIK database with {len(cik_database)} entries")
        return cik_database

    def _load_existing_ciks(self) -> Dict[str, str]:
        """Load existing CIK mappings from files"""
        ciks = {}

        # Load from known_ciks.json
        known_ciks_file = self.output_dir / "known_ciks.json"
        if known_ciks_file.exists():
            try:
                with open(known_ciks_file, 'r') as f:
                    ciks.update(json.load(f))
                logger.info(f"Loaded {len(ciks)} CIK mappings from known_ciks.json")
            except Exception as e:
                logger.warning(f"Failed to load known_ciks.json: {e}")

        return ciks

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

    def _get_historical_mappings(self) -> Dict[str, str]:
        """Get historical ticker mappings for companies that changed tickers"""
        return {
            # Companies that changed tickers or were acquired
            "FB": "0001326801",      # Facebook -> Meta
            "TWTR": "0001418091",    # Twitter -> X Corp
            "CELG": "0001045520",    # Celgene (acquired by Bristol Myers)
            "AGN": "0000008848",     # Allergan (acquired by AbbVie)
            "MYL": "0000062215",     # Mylan (merged with Pfizer)
            "RTN": "0000082811",     # Raytheon (merged with United Technologies)
            "UTX": "0000101829",     # United Technologies (merged with Raytheon)
            "DWDP": "0000030554",    # DowDuPont (split into DOW, DD, CTVA)
            "MON": "0001110783",     # Monsanto (acquired by Bayer)
            "FOXA": "0001308161",    # 21st Century Fox (acquired by Disney)
            "CELGZ": "0001045520",   # Celgene (preferred shares)
            "FCAU": "0001605484",    # Fiat Chrysler (merged with PSA)
            "STLA": "0001605484",    # Stellantis
            "ABMD": "0000815094",    # Abiomed (acquired by J&J)
            "SGEN": "0001023796",    # Seagen (acquired by Pfizer)
            "ATVI": "0000718877",    # Activision Blizzard (acquired by Microsoft)
            "CTXS": "0000877890",    # Citrix (acquired by Vista/TB)
            "VMW": "0001124610",     # VMware (acquired by Broadcom)
            "NUAN": "0001002517",    # Nuance (acquired by Microsoft)
            "ZEN": "0001450704",     # Zendesk (acquired by private equity)
            "XLNX": "0000743988",    # Xilinx (acquired by AMD)
            "CTVA": "0001755672",    # Corteva (DowDuPont spin-off)
            "DD": "0000030554",      # DuPont (DowDuPont spin-off)
            "DOW": "0001751788",     # Dow Inc (DowDuPont spin-off)
        }

    def _get_common_variations(self) -> Dict[str, str]:
        """Get common ticker variations and corrections"""
        return {
            # Preferred shares and other variations
            "BRK.A": "0001067983",   # Berkshire Hathaway Class A
            "BRK.B": "0001067983",   # Berkshire Hathaway Class B
            "BRK": "0001067983",     # Berkshire Hathaway (default to Class B)
            "BF.A": "0000014943",    # Brown-Forman Class A
            "BF.B": "0000014943",    # Brown-Forman Class B
            "GOOG": "0001652044",    # Alphabet Class C
            "GOOGL": "0001652044",   # Alphabet Class A
            "FWONK": "0001579491",   # Liberty Formula One Group
            "FWONA": "0001579491",   # Liberty Formula One Group
            "FWONB": "0001579491",   # Liberty Formula One Group
            "LBRDA": "0001579491",   # Liberty Broadband Class A
            "LBRDK": "0001579491",   # Liberty Broadband Class C
            "LBRDB": "0001579491",   # Liberty Broadband Class B
            "BATRA": "0001560385",   # Liberty Braves Group
            "BATRK": "0001560385",   # Liberty Braves Group
            "BATRB": "0001560385",   # Liberty Braves Group
        }

    def _resolve_missing_tickers(self, existing_ciks: Dict[str, str]) -> Dict[str, str]:
        """Try to resolve CIKs for tickers that are missing"""
        resolved = {}

        # Common tickers that might be missing
        missing_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
            "CRM", "AMD", "INTC", "IBM", "ORCL", "CSCO", "JPM", "BAC", "WFC",
            "GS", "MS", "C", "V", "MA", "PYPL", "SQ", "SHOP", "SPOT", "UBER"
        ]

        for ticker in missing_tickers:
            if ticker not in existing_ciks:
                cik = self._lookup_single_ticker(ticker)
                if cik:
                    resolved[ticker] = cik
                    logger.info(f"Resolved missing ticker {ticker}: {cik}")

        return resolved

    def _lookup_single_ticker(self, ticker: str) -> Optional[str]:
        """Lookup CIK for a single ticker using multiple methods"""
        # Try SEC company tickers first
        try:
            company_tickers_url = "https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(company_tickers_url, timeout=30)
            response.raise_for_status()

            company_data = response.json()
            for item in company_data.values():
                if item.get('ticker', '').upper() == ticker.upper():
                    return str(item.get('cik_str', '')).zfill(10)
        except Exception:
            pass

        # Try search API
        try:
            search_url = "https://www.sec.gov/edgar/search"
            params = {
                'q': ticker,
                'category': 'custom',
                'entityNameType': 'C',
                'output': 'atom',
                'count': '1'
            }

            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()

            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')

            for entry in entries:
                links = entry.findall('.//{http://www.w3.org/2005/Atom}link')
                for link in links:
                    href = link.get('href', '')
                    import re
                    cik_match = re.search(r'/edgar/data/(\d+)/', href)
                    if cik_match:
                        return cik_match.group(1)

        except Exception:
            pass

        return None

    def _save_database(self, cik_database: Dict[str, str]):
        """Save the comprehensive CIK database"""
        output_file = self.output_dir / "comprehensive_ciks.json"

        # Create a more detailed database with metadata
        database_with_metadata = {
            "metadata": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_entries": len(cik_database),
                "sources": [
                    "SEC Official Company Tickers",
                    "Historical Ticker Mappings",
                    "Manual Corrections",
                    "API Resolution"
                ]
            },
            "cik_mappings": cik_database
        }

        with open(output_file, 'w') as f:
            json.dump(database_with_metadata, f, indent=2)

        logger.info(f"Saved comprehensive CIK database to {output_file}")


def main():
    builder = CIKDatabaseBuilder()
    cik_database = builder.build_comprehensive_database()

    print(f"✅ Built comprehensive CIK database with {len(cik_database)} entries")
    print("The database includes:")
    print("  • Official SEC company tickers")
    print("  • Historical ticker mappings for delisted companies")
    print("  • Common ticker variations and preferred shares")
    print("  • Manual corrections for acquired companies")


if __name__ == "__main__":
    main()
