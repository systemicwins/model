"""
Financial embeddings ORM models using SQLAlchemy
Supports PostgreSQL with pgvector and YugabyteDB
"""

from sqlalchemy import Column, String, Integer, BigInteger, Float, Date, DateTime, ForeignKey, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid

Base = declarative_base()

class StockEmbedding(Base):
    """Main stock embeddings with Matryoshka dimensions"""
    __tablename__ = 'stock_embeddings'
    
    ticker = Column(String(10), primary_key=True)
    company_name = Column(String(255))
    sector = Column(String(50), index=True)
    industry = Column(String(100))
    market_cap = Column(BigInteger, index=True)
    
    # Matryoshka embeddings at different dimensions
    embedding_64 = Column(Vector(64))      # Real-time trading
    embedding_128 = Column(Vector(128))    # Quick screening
    embedding_256 = Column(Vector(256))    # Similarity search
    embedding_512 = Column(Vector(512))    # Semantic analysis
    embedding_768 = Column(Vector(768))    # High-quality retrieval
    embedding_1024 = Column(Vector(1024))  # Detailed analysis
    embedding_1536 = Column(Vector(1536))  # Full fidelity
    
    # Metadata
    last_filing_date = Column(Date)
    last_filing_type = Column(String(20))
    embedding_version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    filings = relationship("FilingEmbedding", back_populates="stock")
    news = relationship("NewsEmbedding", back_populates="stock")
    patterns = relationship("PricePatternEmbedding", back_populates="stock")
    
    def find_similar(self, session, dimension=256, limit=10):
        """Find similar stocks using vector similarity"""
        embedding_col = getattr(self.__class__, f'embedding_{dimension}')
        query_embedding = getattr(self, f'embedding_{dimension}')
        
        return session.query(StockEmbedding)\
            .filter(StockEmbedding.ticker != self.ticker)\
            .order_by(embedding_col.cosine_distance(query_embedding))\
            .limit(limit)\
            .all()


class FilingEmbedding(Base):
    """SEC filing embeddings"""
    __tablename__ = 'filing_embeddings'
    
    filing_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), ForeignKey('stock_embeddings.ticker'), nullable=False, index=True)
    filing_type = Column(String(20), nullable=False)
    filing_date = Column(Date, nullable=False, index=True)
    accession_number = Column(String(20), unique=True, nullable=False, index=True)
    cik = Column(String(10), index=True)
    
    # Multi-dimensional embeddings
    embedding_256 = Column(Vector(256))
    embedding_512 = Column(Vector(512))
    embedding_1536 = Column(Vector(1536))
    
    # Metadata
    filing_url = Column(String)
    file_size_bytes = Column(BigInteger)
    word_count = Column(Integer)
    sentiment_score = Column(Float)
    risk_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("StockEmbedding", back_populates="filings")
    
    # Composite index for ticker+date queries
    __table_args__ = (
        Index('idx_filing_ticker_date', 'ticker', 'filing_date'),
    )


class NewsEmbedding(Base):
    """News and sentiment embeddings"""
    __tablename__ = 'news_embeddings'
    
    news_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), ForeignKey('stock_embeddings.ticker'), index=True)
    headline = Column(String)
    source = Column(String(100))
    published_at = Column(DateTime, index=True)
    
    # Smaller embeddings for news
    embedding_64 = Column(Vector(64))
    embedding_256 = Column(Vector(256))
    
    sentiment = Column(String(20))  # bullish, bearish, neutral
    relevance_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("StockEmbedding", back_populates="news")


class PricePatternEmbedding(Base):
    """Technical pattern embeddings"""
    __tablename__ = 'price_pattern_embeddings'
    
    pattern_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), ForeignKey('stock_embeddings.ticker'), nullable=False, index=True)
    timeframe = Column(String(20))  # 1min, 5min, 1hour, 1day
    pattern_date = Column(Date, index=True)
    
    embedding_128 = Column(Vector(128))
    
    pattern_type = Column(String(50))  # breakout, reversal, consolidation
    volume_ratio = Column(Float)
    price_change_pct = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("StockEmbedding", back_populates="patterns")


class SectorEmbedding(Base):
    """Aggregated sector embeddings"""
    __tablename__ = 'sector_embeddings'
    
    sector = Column(String(50), primary_key=True)
    embedding_256 = Column(Vector(256))
    embedding_512 = Column(Vector(512))
    
    constituent_count = Column(Integer)
    total_market_cap = Column(BigInteger)
    last_updated = Column(DateTime, default=datetime.utcnow)