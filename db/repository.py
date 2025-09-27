"""
Repository pattern for clean database operations
Works with both PostgreSQL and YugabyteDB
"""

from typing import List, Dict, Optional, Any
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from pgvector.sqlalchemy import Vector

from .models import StockEmbedding, FilingEmbedding, NewsEmbedding, PricePatternEmbedding
from .database import session_scope

class StockRepository:
    """Repository for stock-related database operations"""
    
    def __init__(self, session: Session = None):
        self.session = session
    
    def upsert_stock(self, ticker: str, embeddings: Dict[int, np.ndarray], 
                     metadata: Dict = None) -> StockEmbedding:
        """Insert or update stock with embeddings"""
        stock = self.session.query(StockEmbedding).filter_by(ticker=ticker).first()
        
        if not stock:
            stock = StockEmbedding(ticker=ticker)
            self.session.add(stock)
        
        # Update embeddings
        for dim, embedding in embeddings.items():
            if hasattr(stock, f'embedding_{dim}'):
                setattr(stock, f'embedding_{dim}', embedding)
        
        # Update metadata
        if metadata:
            for key, value in metadata.items():
                if hasattr(stock, key):
                    setattr(stock, key, value)
        
        self.session.commit()
        return stock
    
    def find_similar_stocks(self, ticker: str, dimension: int = 256, 
                           limit: int = 10, filters: Dict = None) -> List[Dict]:
        """Find stocks similar to given ticker"""
        reference = self.session.query(StockEmbedding).filter_by(ticker=ticker).first()
        if not reference:
            return []
        
        embedding_col = getattr(StockEmbedding, f'embedding_{dimension}')
        reference_embedding = getattr(reference, f'embedding_{dimension}')
        
        query = self.session.query(
            StockEmbedding.ticker,
            StockEmbedding.company_name,
            StockEmbedding.sector,
            StockEmbedding.market_cap,
            (1 - embedding_col.cosine_distance(reference_embedding)).label('similarity')
        ).filter(StockEmbedding.ticker != ticker)
        
        # Apply filters
        if filters:
            if 'sector' in filters:
                query = query.filter(StockEmbedding.sector == filters['sector'])
            if 'min_market_cap' in filters:
                query = query.filter(StockEmbedding.market_cap >= filters['min_market_cap'])
        
        return query.order_by(embedding_col.cosine_distance(reference_embedding))\
                   .limit(limit)\
                   .all()
    
    def search_by_embedding(self, query_embedding: np.ndarray, 
                           dimension: int = None, limit: int = 50) -> List[Dict]:
        """Search stocks by query embedding"""
        if dimension is None:
            dimension = len(query_embedding)
        
        embedding_col = getattr(StockEmbedding, f'embedding_{dimension}')
        
        results = self.session.query(
            StockEmbedding.ticker,
            StockEmbedding.company_name,
            StockEmbedding.sector,
            (1 - embedding_col.cosine_distance(query_embedding)).label('similarity')
        ).order_by(embedding_col.cosine_distance(query_embedding))\
         .limit(limit)\
         .all()
        
        return [dict(r) for r in results]


class FilingRepository:
    """Repository for SEC filing operations"""
    
    def __init__(self, session: Session = None):
        self.session = session
    
    def insert_filing(self, ticker: str, filing_type: str, filing_date: str,
                     accession_number: str, embeddings: Dict[int, np.ndarray],
                     metadata: Dict = None) -> FilingEmbedding:
        """Insert or update filing"""
        filing = self.session.query(FilingEmbedding)\
                            .filter_by(accession_number=accession_number)\
                            .first()
        
        if not filing:
            filing = FilingEmbedding(
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing_date,
                accession_number=accession_number
            )
            self.session.add(filing)
        
        # Update embeddings
        for dim, embedding in embeddings.items():
            if hasattr(filing, f'embedding_{dim}'):
                setattr(filing, f'embedding_{dim}', embedding)
        
        # Extract CIK from accession number
        if accession_number:
            filing.cik = accession_number[:10].lstrip('0')
        
        # Update metadata
        if metadata:
            for key, value in metadata.items():
                if hasattr(filing, key):
                    setattr(filing, key, value)
        
        self.session.commit()
        return filing
    
    def get_by_accession(self, accession_number: str) -> Optional[FilingEmbedding]:
        """Get filing by accession number"""
        return self.session.query(FilingEmbedding)\
                          .filter_by(accession_number=accession_number)\
                          .first()
    
    def find_similar_filings(self, accession_number: str = None,
                           ticker: str = None, filing_date: str = None,
                           dimension: int = 512, limit: int = 10) -> List[Dict]:
        """Find similar filings"""
        if accession_number:
            reference = self.get_by_accession(accession_number)
        elif ticker and filing_date:
            reference = self.session.query(FilingEmbedding)\
                                   .filter_by(ticker=ticker, filing_date=filing_date)\
                                   .first()
        else:
            return []
        
        if not reference:
            return []
        
        embedding_col = getattr(FilingEmbedding, f'embedding_{dimension}')
        reference_embedding = getattr(reference, f'embedding_{dimension}')
        
        results = self.session.query(
            FilingEmbedding.ticker,
            FilingEmbedding.filing_type,
            FilingEmbedding.filing_date,
            FilingEmbedding.accession_number,
            (1 - embedding_col.cosine_distance(reference_embedding)).label('similarity')
        ).filter(FilingEmbedding.filing_id != reference.filing_id)\
         .order_by(embedding_col.cosine_distance(reference_embedding))\
         .limit(limit)\
         .all()
        
        return [dict(r) for r in results]


# Convenience functions using context manager
def upsert_stock_embeddings(ticker: str, embeddings: Dict, metadata: Dict = None):
    """Upsert stock embeddings with automatic session management"""
    with session_scope() as session:
        repo = StockRepository(session)
        return repo.upsert_stock(ticker, embeddings, metadata)

def find_similar_stocks(ticker: str, **kwargs):
    """Find similar stocks with automatic session management"""
    with session_scope() as session:
        repo = StockRepository(session)
        return repo.find_similar_stocks(ticker, **kwargs)

def insert_filing_embedding(ticker: str, filing_type: str, filing_date: str,
                           accession_number: str, embeddings: Dict, metadata: Dict = None):
    """Insert filing with automatic session management"""
    with session_scope() as session:
        repo = FilingRepository(session)
        return repo.insert_filing(ticker, filing_type, filing_date, 
                                 accession_number, embeddings, metadata)