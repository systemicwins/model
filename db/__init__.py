"""
Financial Vector Database ORM
Supports PostgreSQL with pgvector and YugabyteDB
"""

from .database import (
    DatabaseConfig,
    DatabaseManager,
    get_db_manager,
    get_session,
    session_scope
)

from .models import (
    Base,
    StockEmbedding,
    FilingEmbedding,
    NewsEmbedding,
    PricePatternEmbedding,
    SectorEmbedding
)

from .repository import (
    StockRepository,
    FilingRepository,
    upsert_stock_embeddings,
    find_similar_stocks,
    insert_filing_embedding
)

__all__ = [
    # Database management
    'DatabaseConfig',
    'DatabaseManager',
    'get_db_manager',
    'get_session',
    'session_scope',
    
    # Models
    'Base',
    'StockEmbedding',
    'FilingEmbedding',
    'NewsEmbedding',
    'PricePatternEmbedding',
    'SectorEmbedding',
    
    # Repositories
    'StockRepository',
    'FilingRepository',
    'upsert_stock_embeddings',
    'find_similar_stocks',
    'insert_filing_embedding'
]