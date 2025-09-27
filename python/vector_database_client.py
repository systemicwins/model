#!/usr/bin/env python3
"""
PostgreSQL + pgvector client for financial embeddings
Designed for Google Cloud SQL integration
"""

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
import logging
from dataclasses import dataclass
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockEmbedding:
    """Container for stock embeddings at multiple dimensions"""
    ticker: str
    embeddings: Dict[int, np.ndarray]  # {64: array, 256: array, ...}
    metadata: Dict[str, any] = None

class FinancialVectorDB:
    """Client for PostgreSQL with pgvector on Google Cloud SQL"""
    
    def __init__(self, 
                 host: str = None,
                 database: str = None,
                 user: str = None,
                 password: str = None,
                 port: int = 5432,
                 cloud_sql_connection_name: str = None):
        """
        Initialize connection to Cloud SQL with pgvector
        
        Args:
            host: Database host (use localhost for Cloud SQL proxy)
            database: Database name
            user: Database user
            password: Database password
            port: Database port (default 5432)
            cloud_sql_connection_name: For direct Cloud SQL connection
        """
        # Use environment variables if not provided
        self.host = host or os.getenv('DB_HOST', 'localhost')
        self.database = database or os.getenv('DB_NAME', 'financial_vectors')
        self.user = user or os.getenv('DB_USER', 'postgres')
        self.password = password or os.getenv('DB_PASSWORD')
        self.port = port or int(os.getenv('DB_PORT', 5432))
        
        # For Cloud SQL Unix socket connection
        self.cloud_sql_connection_name = cloud_sql_connection_name or os.getenv('CLOUD_SQL_CONNECTION_NAME')
        
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish connection to the database"""
        try:
            if self.cloud_sql_connection_name:
                # Cloud SQL Unix socket connection
                self.conn = psycopg2.connect(
                    host=f'/cloudsql/{self.cloud_sql_connection_name}',
                    database=self.database,
                    user=self.user,
                    password=self.password
                )
            else:
                # Standard TCP connection (Cloud SQL Proxy or direct)
                self.conn = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password
                )
            
            # Register pgvector extension
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.conn.commit()
                
            logger.info(f"Connected to database: {self.database}")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # ============== INSERT OPERATIONS ==============
    
    def upsert_stock_embeddings(self, stock_embedding: StockEmbedding):
        """
        Insert or update stock embeddings for all dimensions
        
        Args:
            stock_embedding: StockEmbedding object with ticker and embeddings
        """
        with self.conn.cursor() as cur:
            # Build the SQL dynamically based on available dimensions
            dimensions = stock_embedding.embeddings.keys()
            
            # Prepare column names and values
            embedding_columns = [f"embedding_{dim}" for dim in dimensions]
            embedding_values = [stock_embedding.embeddings[dim].tolist() for dim in dimensions]
            
            # Add metadata if provided
            metadata = stock_embedding.metadata or {}
            
            sql = f"""
                INSERT INTO stock_embeddings (
                    ticker,
                    company_name,
                    sector,
                    industry,
                    market_cap,
                    {','.join(embedding_columns)}
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    {','.join(['%s' for _ in embedding_columns])}
                )
                ON CONFLICT (ticker) DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    market_cap = EXCLUDED.market_cap,
                    {','.join([f'{col} = EXCLUDED.{col}' for col in embedding_columns])},
                    updated_at = CURRENT_TIMESTAMP
            """
            
            values = [
                stock_embedding.ticker,
                metadata.get('company_name'),
                metadata.get('sector'),
                metadata.get('industry'),
                metadata.get('market_cap')
            ] + embedding_values
            
            cur.execute(sql, values)
            self.conn.commit()
            
            logger.info(f"Upserted embeddings for {stock_embedding.ticker}")
    
    def batch_upsert_embeddings(self, embeddings: List[StockEmbedding], batch_size: int = 100):
        """Batch insert multiple stock embeddings"""
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            for embedding in batch:
                self.upsert_stock_embeddings(embedding)
            logger.info(f"Processed batch {i//batch_size + 1}")
    
    def insert_filing_embedding(self, 
                               ticker: str,
                               filing_type: str,
                               filing_date: str,
                               accession_number: str,
                               embeddings: Dict[int, np.ndarray],
                               metadata: Dict = None):
        """
        Insert SEC filing embeddings
        
        Args:
            ticker: Stock ticker symbol
            filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
            filing_date: Date of filing
            accession_number: SEC accession number (e.g., 0000950103-24-016820)
            embeddings: Dictionary of embeddings by dimension
            metadata: Optional metadata including CIK, URL, sentiment scores
        """
        with self.conn.cursor() as cur:
            sql = """
                INSERT INTO filing_embeddings (
                    ticker, filing_type, filing_date, accession_number, cik,
                    embedding_256, embedding_512, embedding_1536,
                    filing_url, sentiment_score, risk_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO UPDATE SET
                    embedding_256 = EXCLUDED.embedding_256,
                    embedding_512 = EXCLUDED.embedding_512,
                    embedding_1536 = EXCLUDED.embedding_1536,
                    sentiment_score = EXCLUDED.sentiment_score,
                    risk_score = EXCLUDED.risk_score
                RETURNING filing_id
            """
            
            # Extract CIK from accession number if not provided
            cik = metadata.get('cik') if metadata else None
            if not cik and accession_number:
                # First 10 digits of accession number is the CIK
                cik = accession_number[:10].lstrip('0')
            
            values = [
                ticker, filing_type, filing_date, accession_number, cik,
                embeddings.get(256, None),
                embeddings.get(512, None),
                embeddings.get(1536, None),
                metadata.get('filing_url') if metadata else None,
                metadata.get('sentiment_score') if metadata else None,
                metadata.get('risk_score') if metadata else None
            ]
            
            cur.execute(sql, values)
            filing_id = cur.fetchone()[0]
            self.conn.commit()
            
            logger.info(f"Inserted filing {accession_number} for {ticker}")
            return filing_id
    
    # ============== SEARCH OPERATIONS ==============
    
    def find_similar_stocks(self, 
                           ticker: str,
                           dimension: int = 256,
                           limit: int = 10,
                           sector_filter: str = None) -> List[Dict]:
        """
        Find stocks similar to the given ticker
        
        Args:
            ticker: Reference ticker symbol
            dimension: Embedding dimension to use (64, 256, 512, etc.)
            limit: Number of results to return
            sector_filter: Optional sector filter
            
        Returns:
            List of similar stocks with similarity scores
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            embedding_col = f"embedding_{dimension}"
            
            sql = f"""
                WITH reference AS (
                    SELECT {embedding_col} as ref_embedding, sector
                    FROM stock_embeddings
                    WHERE ticker = %s
                )
                SELECT 
                    s.ticker,
                    s.company_name,
                    s.sector,
                    s.market_cap,
                    1 - (s.{embedding_col} <=> r.ref_embedding) as similarity
                FROM stock_embeddings s, reference r
                WHERE s.ticker != %s
                    {f"AND s.sector = %s" if sector_filter else ""}
                ORDER BY s.{embedding_col} <=> r.ref_embedding
                LIMIT %s
            """
            
            params = [ticker, ticker]
            if sector_filter:
                params.append(sector_filter)
            params.append(limit)
            
            cur.execute(sql, params)
            return cur.fetchall()
    
    def search_by_embedding(self, 
                           query_embedding: np.ndarray,
                           dimension: int = None,
                           limit: int = 50,
                           filters: Dict = None) -> List[Dict]:
        """
        Search stocks by a query embedding
        
        Args:
            query_embedding: Query vector
            dimension: Dimension (auto-detected if None)
            limit: Number of results
            filters: Optional filters (sector, min_market_cap, etc.)
            
        Returns:
            List of matching stocks with similarity scores
        """
        if dimension is None:
            dimension = len(query_embedding)
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            embedding_col = f"embedding_{dimension}"
            
            # Build WHERE clause from filters
            where_clauses = []
            params = [query_embedding.tolist()]
            
            if filters:
                if 'sector' in filters:
                    where_clauses.append("sector = %s")
                    params.append(filters['sector'])
                
                if 'min_market_cap' in filters:
                    where_clauses.append("market_cap >= %s")
                    params.append(filters['min_market_cap'])
                
                if 'max_market_cap' in filters:
                    where_clauses.append("market_cap <= %s")
                    params.append(filters['max_market_cap'])
            
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            
            sql = f"""
                SELECT 
                    ticker,
                    company_name,
                    sector,
                    market_cap,
                    1 - ({embedding_col} <=> %s) as similarity
                FROM stock_embeddings
                {where_sql}
                ORDER BY {embedding_col} <=> %s
                LIMIT %s
            """
            
            params.extend([query_embedding.tolist(), limit])
            
            cur.execute(sql, params)
            return cur.fetchall()
    
    def get_filing_by_accession(self, accession_number: str) -> Optional[Dict]:
        """Get filing details by SEC accession number"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = """
                SELECT 
                    filing_id, ticker, filing_type, filing_date,
                    accession_number, cik, filing_url,
                    sentiment_score, risk_score, created_at
                FROM filing_embeddings
                WHERE accession_number = %s
            """
            cur.execute(sql, [accession_number])
            return cur.fetchone()
    
    def find_similar_filings(self,
                            ticker: str = None,
                            filing_date: str = None,
                            accession_number: str = None,
                            limit: int = 10) -> List[Dict]:
        """
        Find similar SEC filings across all companies
        Can search by ticker+date or by accession number
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            if accession_number:
                # Search by accession number
                sql = """
                    WITH reference AS (
                        SELECT embedding_512 as ref_embedding, ticker, filing_date
                        FROM filing_embeddings
                        WHERE accession_number = %s
                        LIMIT 1
                    )
                    SELECT 
                        f.ticker,
                        f.filing_type,
                        f.filing_date,
                        f.accession_number,
                        f.sentiment_score,
                        f.risk_score,
                        1 - (f.embedding_512 <=> r.ref_embedding) as similarity
                    FROM filing_embeddings f, reference r
                    WHERE f.accession_number != %s
                    ORDER BY f.embedding_512 <=> r.ref_embedding
                    LIMIT %s
                """
                cur.execute(sql, [accession_number, accession_number, limit])
            else:
                # Search by ticker and date
                sql = """
                    WITH reference AS (
                        SELECT embedding_512 as ref_embedding
                        FROM filing_embeddings
                        WHERE ticker = %s AND filing_date = %s
                        LIMIT 1
                    )
                    SELECT 
                        f.ticker,
                        f.filing_type,
                        f.filing_date,
                        f.accession_number,
                        f.sentiment_score,
                        f.risk_score,
                        1 - (f.embedding_512 <=> r.ref_embedding) as similarity
                    FROM filing_embeddings f, reference r
                    WHERE NOT (f.ticker = %s AND f.filing_date = %s)
                    ORDER BY f.embedding_512 <=> r.ref_embedding
                    LIMIT %s
                """
                cur.execute(sql, [ticker, filing_date, ticker, filing_date, limit])
            
            return cur.fetchall()
    
    # ============== ANALYTICS OPERATIONS ==============
    
    def get_sector_similarity(self, ticker: str) -> List[Dict]:
        """Get similarity of a stock to various sectors"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = """
                WITH stock_embedding AS (
                    SELECT embedding_256
                    FROM stock_embeddings
                    WHERE ticker = %s
                )
                SELECT 
                    s.sector,
                    COUNT(*) as stock_count,
                    AVG(1 - (s.embedding_256 <=> se.embedding_256)) as avg_similarity
                FROM stock_embeddings s, stock_embedding se
                WHERE s.sector IS NOT NULL
                GROUP BY s.sector
                ORDER BY avg_similarity DESC
            """
            
            cur.execute(sql, [ticker])
            return cur.fetchall()
    
    def detect_outliers(self, 
                        sector: str = None,
                        dimension: int = 256,
                        threshold: float = 0.3) -> List[Dict]:
        """Detect outlier stocks that don't fit their sector pattern"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            embedding_col = f"embedding_{dimension}"
            
            if sector:
                sql = f"""
                    WITH sector_centroid AS (
                        SELECT AVG({embedding_col}) as centroid
                        FROM stock_embeddings
                        WHERE sector = %s
                    )
                    SELECT 
                        s.ticker,
                        s.company_name,
                        s.sector,
                        1 - (s.{embedding_col} <=> sc.centroid) as similarity_to_sector
                    FROM stock_embeddings s, sector_centroid sc
                    WHERE s.sector = %s
                        AND (1 - (s.{embedding_col} <=> sc.centroid)) < %s
                    ORDER BY similarity_to_sector ASC
                """
                params = [sector, sector, threshold]
            else:
                # Find outliers across all sectors
                sql = f"""
                    WITH sector_centroids AS (
                        SELECT 
                            sector,
                            AVG({embedding_col}) as centroid
                        FROM stock_embeddings
                        WHERE sector IS NOT NULL
                        GROUP BY sector
                    )
                    SELECT 
                        s.ticker,
                        s.company_name,
                        s.sector,
                        1 - (s.{embedding_col} <=> sc.centroid) as similarity_to_sector
                    FROM stock_embeddings s
                    JOIN sector_centroids sc ON s.sector = sc.sector
                    WHERE (1 - (s.{embedding_col} <=> sc.centroid)) < %s
                    ORDER BY similarity_to_sector ASC
                """
                params = [threshold]
            
            cur.execute(sql, params)
            return cur.fetchall()
    
    # ============== REAL-TIME OPERATIONS ==============
    
    def get_fast_embedding(self, ticker: str) -> Optional[np.ndarray]:
        """Get 64-dim embedding for real-time operations"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT embedding_64 FROM stock_embeddings WHERE ticker = %s", [ticker])
            result = cur.fetchone()
            return np.array(result[0]) if result else None
    
    def batch_get_embeddings(self, 
                            tickers: List[str],
                            dimension: int = 256) -> Dict[str, np.ndarray]:
        """Get embeddings for multiple tickers efficiently"""
        with self.conn.cursor() as cur:
            embedding_col = f"embedding_{dimension}"
            
            sql = f"""
                SELECT ticker, {embedding_col}
                FROM stock_embeddings
                WHERE ticker = ANY(%s)
            """
            
            cur.execute(sql, [tickers])
            results = cur.fetchall()
            
            return {
                ticker: np.array(embedding)
                for ticker, embedding in results
            }
    
    # ============== MAINTENANCE OPERATIONS ==============
    
    def refresh_materialized_views(self):
        """Refresh materialized views for better performance"""
        with self.conn.cursor() as cur:
            cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sector_averages")
            self.conn.commit()
            logger.info("Refreshed materialized views")
    
    def optimize_indexes(self):
        """Re-index for better search performance"""
        with self.conn.cursor() as cur:
            # Get all ivfflat indexes
            cur.execute("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE indexdef LIKE '%ivfflat%'
            """)
            indexes = cur.fetchall()
            
            for index in indexes:
                cur.execute(f"REINDEX INDEX {index[0]}")
                logger.info(f"Reindexed {index[0]}")
            
            self.conn.commit()
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM stock_embeddings) as total_stocks,
                    (SELECT COUNT(*) FROM filing_embeddings) as total_filings,
                    (SELECT COUNT(*) FROM news_embeddings) as total_news,
                    (SELECT COUNT(DISTINCT sector) FROM stock_embeddings) as total_sectors,
                    (SELECT pg_size_pretty(pg_database_size(current_database()))) as database_size
            """)
            return cur.fetchone()


# Example usage
if __name__ == "__main__":
    # Initialize connection
    db = FinancialVectorDB(
        host="localhost",  # Use Cloud SQL Proxy
        database="financial_vectors",
        user="postgres",
        password="your_password"
    )
    
    # Example: Upsert stock embeddings
    from your_model import generate_embeddings  # Your model
    
    text = "Apple Inc Q3 earnings report..."
    embeddings = generate_embeddings(text)  # Returns dict of dimensions
    
    stock_emb = StockEmbedding(
        ticker="AAPL",
        embeddings=embeddings,
        metadata={
            "company_name": "Apple Inc",
            "sector": "Technology",
            "market_cap": 3000000000000
        }
    )
    
    db.upsert_stock_embeddings(stock_emb)
    
    # Find similar stocks
    similar = db.find_similar_stocks("AAPL", dimension=256, limit=10)
    for stock in similar:
        print(f"{stock['ticker']}: {stock['similarity']:.3f}")
    
    # Clean up
    db.close()