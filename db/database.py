"""
Database connection and session management
Supports both PostgreSQL and YugabyteDB
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool
import os
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration that supports both PostgreSQL and YugabyteDB"""
    
    def __init__(self, db_type='postgresql'):
        self.db_type = db_type
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = os.getenv('DB_PORT', 5432)
        self.database = os.getenv('DB_NAME', 'financial_vectors')
        self.user = os.getenv('DB_USER', 'postgres')
        self.password = os.getenv('DB_PASSWORD')
        
        # YugabyteDB specific settings
        if db_type == 'yugabyte':
            self.port = os.getenv('DB_PORT', 5433)  # YugabyteDB default
            self.load_balance = os.getenv('YB_LOAD_BALANCE', 'true')
            self.topology_keys = os.getenv('YB_TOPOLOGY_KEYS', '')
    
    def get_connection_string(self):
        """Build connection string for SQLAlchemy"""
        base_url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        if self.db_type == 'yugabyte':
            # Add YugabyteDB specific parameters
            params = []
            if hasattr(self, 'load_balance'):
                params.append(f"load_balance={self.load_balance}")
            if hasattr(self, 'topology_keys') and self.topology_keys:
                params.append(f"topology_keys={self.topology_keys}")
            
            if params:
                base_url += "?" + "&".join(params)
        
        return base_url


class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self, config: DatabaseConfig = None):
        if config is None:
            # Auto-detect YugabyteDB vs PostgreSQL
            db_type = os.getenv('DB_TYPE', 'postgresql')
            config = DatabaseConfig(db_type)
        
        self.config = config
        self.engine = None
        self.Session = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize SQLAlchemy engine with appropriate settings"""
        connection_string = self.config.get_connection_string()
        
        # Engine configuration optimized for vector operations
        engine_config = {
            'pool_pre_ping': True,  # Verify connections before use
            'pool_size': 10,
            'max_overflow': 20,
            'echo': os.getenv('SQL_ECHO', 'false').lower() == 'true'
        }
        
        if self.config.db_type == 'yugabyte':
            # YugabyteDB performs better with connection pooling disabled
            # for distributed queries
            engine_config['poolclass'] = NullPool
        
        self.engine = create_engine(connection_string, **engine_config)
        
        # Add pgvector extension on connect
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            with dbapi_conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create session factory
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        
        logger.info(f"Connected to {self.config.db_type} database: {self.config.database}")
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope for database operations"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def create_all_tables(self):
        """Create all tables defined in models"""
        from .models import Base
        Base.metadata.create_all(self.engine)
        logger.info("Created all database tables")
    
    def drop_all_tables(self):
        """Drop all tables - use with caution!"""
        from .models import Base
        Base.metadata.drop_all(self.engine)
        logger.info("Dropped all database tables")
    
    def get_engine(self):
        """Get the SQLAlchemy engine"""
        return self.engine
    
    def get_session(self):
        """Get a new session"""
        return self.Session()
    
    def close(self):
        """Close all connections"""
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()


# Global database manager instance
_db_manager = None

def get_db_manager():
    """Get or create the global database manager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def get_session():
    """Get a database session"""
    return get_db_manager().get_session()

@contextmanager
def session_scope():
    """Context manager for database sessions"""
    manager = get_db_manager()
    with manager.session_scope() as session:
        yield session