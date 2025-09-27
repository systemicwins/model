#!/usr/bin/env python3
"""
Federal Reserve Economic Data (FRED) API Fetcher
Fetches historical interest rate data with proper rate limiting
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from ratelimit import limits, sleep_and_retry
import backoff
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FREDFetcher:
    """Fetches Federal Reserve Economic Data with rate limiting"""
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    # FRED rate limit: 120 requests per minute
    CALLS_PER_MINUTE = 120
    
    # Key interest rate series
    INTEREST_RATE_SERIES = {
        'DFF': 'Federal Funds Rate (Daily)',
        'FEDFUNDS': 'Federal Funds Rate (Monthly)',
        'DGS3MO': '3-Month Treasury',
        'DGS6MO': '6-Month Treasury',
        'DGS1': '1-Year Treasury',
        'DGS2': '2-Year Treasury',
        'DGS5': '5-Year Treasury',
        'DGS10': '10-Year Treasury',
        'DGS30': '30-Year Treasury',
        'DPRIME': 'Prime Rate',
        'SOFR': 'SOFR Rate',
        'DISCOUNTRATE': 'Discount Rate',
        'TB3MS': '3-Month Treasury Bill (Secondary Market)',
        'AAA': 'Moody\'s Aaa Corporate Bond Yield',
        'BAA': 'Moody\'s Baa Corporate Bond Yield',
    }
    
    # Economic indicators for context
    ECONOMIC_INDICATORS = {
        'CPIAUCSL': 'Consumer Price Index',
        'UNRATE': 'Unemployment Rate',
        'GDP': 'Gross Domestic Product',
        'GDPC1': 'Real GDP',
        'INDPRO': 'Industrial Production Index',
        'PAYEMS': 'Nonfarm Payrolls',
        'HOUST': 'Housing Starts',
        'DEXUSEU': 'US/Euro Exchange Rate',
        'DCOILWTICO': 'WTI Crude Oil Price',
        'GOLDAMGBD228NLBM': 'Gold Price',
    }
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "FED"):
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED API key required. Set FRED_API_KEY environment variable.")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
    
    @sleep_and_retry
    @limits(calls=CALLS_PER_MINUTE, period=60)
    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
    def _make_request(self, endpoint: str, params: Dict) -> requests.Response:
        """Make rate-limited request to FRED API"""
        url = f"{self.BASE_URL}/{endpoint}"
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        logger.debug(f"Fetching: {url} with params: {params}")
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response
    
    def fetch_series(self, series_id: str, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    frequency: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch time series data
        
        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency (d, w, m, q, a)
        
        Returns:
            DataFrame with date and value columns
        """
        params = {'series_id': series_id}
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
        if frequency:
            params['frequency'] = frequency
        
        try:
            response = self._make_request('series/observations', params)
            data = response.json()
            
            # Convert to DataFrame
            observations = data.get('observations', [])
            if observations:
                df = pd.DataFrame(observations)
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df[['date', 'value']].dropna()
                df.set_index('date', inplace=True)
                
                # Save to CSV
                filename = f"{series_id}_{frequency or 'default'}.csv"
                if start_date or end_date:
                    date_suffix = f"_{start_date or 'start'}_{end_date or 'end'}"
                    filename = filename.replace('.csv', f"{date_suffix}.csv")
                
                output_file = self.output_dir / filename
                df.to_csv(output_file)
                logger.info(f"Saved {series_id} to {output_file}")
                
                return df
            else:
                logger.warning(f"No data found for {series_id}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return pd.DataFrame()
    
    def fetch_series_metadata(self, series_id: str) -> Dict:
        """Fetch metadata for a series"""
        params = {'series_id': series_id}
        
        try:
            response = self._make_request('series', params)
            data = response.json()
            
            if 'seriess' in data and data['seriess']:
                metadata = data['seriess'][0]
                
                # Save metadata
                output_file = self.output_dir / f"{series_id}_metadata.json"
                with open(output_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Saved metadata for {series_id}")
                return metadata
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching metadata for {series_id}: {e}")
            return {}
    
    def fetch_all_interest_rates(self, start_date: str = "2000-01-01") -> Dict[str, pd.DataFrame]:
        """
        Fetch all interest rate series
        
        Args:
            start_date: Start date for historical data
        
        Returns:
            Dictionary of DataFrames keyed by series ID
        """
        results = {}
        
        logger.info(f"Fetching {len(self.INTEREST_RATE_SERIES)} interest rate series")
        
        for series_id, description in self.INTEREST_RATE_SERIES.items():
            logger.info(f"Fetching {series_id}: {description}")
            
            # Get metadata first
            self.fetch_series_metadata(series_id)
            
            # Fetch data
            df = self.fetch_series(series_id, start_date=start_date)
            if not df.empty:
                results[series_id] = df
            
            # Courtesy delay
            time.sleep(0.5)
        
        # Create combined dataset
        if results:
            combined = pd.DataFrame()
            for series_id, df in results.items():
                if combined.empty:
                    combined = df.rename(columns={'value': series_id})
                else:
                    combined = combined.join(
                        df.rename(columns={'value': series_id}),
                        how='outer'
                    )
            
            # Save combined data
            output_file = self.output_dir / f"combined_interest_rates_{start_date}.csv"
            combined.to_csv(output_file)
            logger.info(f"Saved combined data to {output_file}")
        
        return results
    
    def fetch_economic_context(self, start_date: str = "2000-01-01") -> Dict[str, pd.DataFrame]:
        """
        Fetch economic indicator data for context
        
        Args:
            start_date: Start date for historical data
        
        Returns:
            Dictionary of DataFrames keyed by series ID
        """
        results = {}
        
        logger.info(f"Fetching {len(self.ECONOMIC_INDICATORS)} economic indicators")
        
        for series_id, description in self.ECONOMIC_INDICATORS.items():
            logger.info(f"Fetching {series_id}: {description}")
            
            # Most economic indicators are monthly or quarterly
            df = self.fetch_series(series_id, start_date=start_date, frequency='m')
            if not df.empty:
                results[series_id] = df
            
            time.sleep(0.5)
        
        return results
    
    def create_training_dataset(self, start_date: str = "2000-01-01") -> pd.DataFrame:
        """
        Create combined dataset for model training
        
        Args:
            start_date: Start date for data
        
        Returns:
            Combined DataFrame with all features
        """
        logger.info("Creating training dataset")
        
        # Fetch interest rates
        interest_rates = self.fetch_all_interest_rates(start_date)
        
        # Fetch economic indicators
        economic_data = self.fetch_economic_context(start_date)
        
        # Combine all data
        all_data = pd.DataFrame()
        
        # Add interest rates
        for series_id, df in interest_rates.items():
            if all_data.empty:
                all_data = df.rename(columns={'value': series_id})
            else:
                all_data = all_data.join(
                    df.rename(columns={'value': series_id}),
                    how='outer'
                )
        
        # Add economic indicators
        for series_id, df in economic_data.items():
            all_data = all_data.join(
                df.rename(columns={'value': series_id}),
                how='outer'
            )
        
        # Forward fill missing values (common for daily series with monthly indicators)
        all_data = all_data.fillna(method='ffill')
        
        # Save complete dataset
        output_file = self.output_dir / f"training_dataset_{start_date}.csv"
        all_data.to_csv(output_file)
        logger.info(f"Saved training dataset to {output_file}")
        
        # Create summary statistics
        summary = {
            'start_date': all_data.index.min().isoformat() if not all_data.empty else None,
            'end_date': all_data.index.max().isoformat() if not all_data.empty else None,
            'total_observations': len(all_data),
            'features': list(all_data.columns),
            'missing_values': all_data.isnull().sum().to_dict()
        }
        
        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Dataset summary saved to {summary_file}")
        
        return all_data


def main():
    """Example usage"""
    # Initialize fetcher (requires FRED_API_KEY env variable)
    try:
        fetcher = FREDFetcher()
    except ValueError as e:
        logger.error(f"Initialization failed: {e}")
        logger.info("Please set FRED_API_KEY environment variable")
        logger.info("Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return
    
    # Fetch recent data (last 5 years for demo)
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    # Fetch key interest rates
    logger.info("Fetching Federal Funds Rate")
    fed_funds = fetcher.fetch_series('DFF', start_date=start_date)
    
    if not fed_funds.empty:
        print(f"\nFederal Funds Rate - Recent Data:")
        print(fed_funds.tail())
        print(f"\nCurrent Rate: {fed_funds.iloc[-1]['value']:.2f}%")
        print(f"5-Year Range: {fed_funds['value'].min():.2f}% - {fed_funds['value'].max():.2f}%")
    
    # Fetch 10-year Treasury
    logger.info("Fetching 10-Year Treasury Yield")
    treasury_10y = fetcher.fetch_series('DGS10', start_date=start_date)
    
    if not treasury_10y.empty:
        print(f"\n10-Year Treasury Yield - Recent Data:")
        print(treasury_10y.tail())
    
    # Create smaller training dataset for demo
    logger.info("\nCreating training dataset (this may take a few minutes)...")
    # Uncomment to create full dataset:
    # training_data = fetcher.create_training_dataset(start_date)
    
    logger.info("Data fetching completed!")


if __name__ == "__main__":
    main()