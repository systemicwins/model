-- PostgreSQL with pgvector schema for financial embeddings
-- Designed for Google Cloud SQL with pgvector extension

-- Enable pgvector extension (must be done by Cloud SQL admin)
CREATE EXTENSION IF NOT EXISTS vector;

-- Main table for stock embeddings with multiple dimensions
CREATE TABLE IF NOT EXISTS stock_embeddings (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255),
    sector VARCHAR(50),
    industry VARCHAR(100),
    market_cap BIGINT,
    
    -- Matryoshka embeddings at different dimensions
    embedding_64 vector(64),      -- Real-time trading (fastest)
    embedding_128 vector(128),     -- Quick screening
    embedding_256 vector(256),     -- Similarity search
    embedding_512 vector(512),     -- Semantic analysis
    embedding_768 vector(768),     -- High-quality retrieval
    embedding_1024 vector(1024),   -- Detailed analysis
    embedding_1536 vector(1536),   -- Full fidelity
    
    -- Metadata
    last_filing_date DATE,
    last_filing_type VARCHAR(20),  -- 10-K, 10-Q, 8-K
    embedding_version INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- SEC filings embeddings (separate table for historical data)
CREATE TABLE IF NOT EXISTS filing_embeddings (
    filing_id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    filing_type VARCHAR(20) NOT NULL,
    filing_date DATE NOT NULL,
    accession_number VARCHAR(20) UNIQUE NOT NULL,  -- SEC's unique ID (e.g., 0000950103-24-016820)
    cik VARCHAR(10),                                -- Central Index Key of filer
    
    -- Store different dimensions for different use cases
    embedding_256 vector(256),    -- For quick search
    embedding_512 vector(512),    -- For analysis
    embedding_1536 vector(1536),  -- For detailed comparison
    
    -- Filing metadata
    filing_url TEXT,
    file_size_bytes BIGINT,
    word_count INT,
    sentiment_score FLOAT,        -- Pre-computed sentiment
    risk_score FLOAT,             -- Pre-computed risk assessment
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (ticker) REFERENCES stock_embeddings(ticker)
);

-- News and market sentiment embeddings
CREATE TABLE IF NOT EXISTS news_embeddings (
    news_id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    headline TEXT,
    source VARCHAR(100),
    published_at TIMESTAMP,
    
    -- Smaller embeddings for news (more volume, less depth needed)
    embedding_64 vector(64),      -- Quick classification
    embedding_256 vector(256),    -- Sentiment analysis
    
    sentiment VARCHAR(20),        -- bullish, bearish, neutral
    relevance_score FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (ticker) REFERENCES stock_embeddings(ticker)
);

-- Intraday price movement patterns
CREATE TABLE IF NOT EXISTS price_pattern_embeddings (
    pattern_id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timeframe VARCHAR(20),        -- '1min', '5min', '1hour', '1day'
    pattern_date DATE,
    
    -- Technical pattern embedding
    embedding_128 vector(128),    -- Price patterns don't need high dimensions
    
    -- Pattern metadata
    pattern_type VARCHAR(50),     -- 'breakout', 'reversal', 'consolidation'
    volume_ratio FLOAT,
    price_change_pct FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (ticker) REFERENCES stock_embeddings(ticker)
);

-- Sector/Industry aggregated embeddings for comparison
CREATE TABLE IF NOT EXISTS sector_embeddings (
    sector VARCHAR(50) PRIMARY KEY,
    
    -- Sector-wide embeddings (averaged from constituents)
    embedding_256 vector(256),
    embedding_512 vector(512),
    
    constituent_count INT,
    total_market_cap BIGINT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for vector similarity search
-- Using IVFFlat for better performance on large datasets

-- For stock_embeddings (different indexes for different use cases)
CREATE INDEX idx_stock_embed_64_ivfflat 
ON stock_embeddings USING ivfflat (embedding_64 vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX idx_stock_embed_256_ivfflat 
ON stock_embeddings USING ivfflat (embedding_256 vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX idx_stock_embed_512_ivfflat 
ON stock_embeddings USING ivfflat (embedding_512 vector_cosine_ops)
WITH (lists = 100);

-- For filing_embeddings
CREATE INDEX idx_filing_embed_256_ivfflat 
ON filing_embeddings USING ivfflat (embedding_256 vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX idx_filing_embed_512_ivfflat 
ON filing_embeddings USING ivfflat (embedding_512 vector_cosine_ops)
WITH (lists = 50);

-- For news_embeddings (high volume, need fast search)
CREATE INDEX idx_news_embed_64_ivfflat 
ON news_embeddings USING ivfflat (embedding_64 vector_cosine_ops)
WITH (lists = 200);

-- Regular indexes for filtering
CREATE INDEX idx_stock_sector ON stock_embeddings(sector);
CREATE INDEX idx_stock_market_cap ON stock_embeddings(market_cap);
CREATE INDEX idx_filing_date ON filing_embeddings(filing_date);
CREATE INDEX idx_filing_ticker_date ON filing_embeddings(ticker, filing_date DESC);
CREATE INDEX idx_filing_accession ON filing_embeddings(accession_number);  -- Fast lookup by SEC ID
CREATE INDEX idx_filing_cik ON filing_embeddings(cik);                      -- Lookup by company CIK
CREATE INDEX idx_news_ticker_date ON news_embeddings(ticker, published_at DESC);
CREATE INDEX idx_pattern_ticker_date ON price_pattern_embeddings(ticker, pattern_date DESC);

-- Materialized view for fast sector comparisons
CREATE MATERIALIZED VIEW mv_sector_averages AS
SELECT 
    s.sector,
    COUNT(DISTINCT s.ticker) as stock_count,
    AVG(s.market_cap) as avg_market_cap,
    -- Average embeddings for the sector (using 256-dim for efficiency)
    AVG(s.embedding_256) as sector_embedding_256
FROM stock_embeddings s
WHERE s.sector IS NOT NULL
GROUP BY s.sector;

CREATE INDEX idx_mv_sector_embedding 
ON mv_sector_averages USING ivfflat (sector_embedding_256 vector_cosine_ops)
WITH (lists = 10);

-- Function to find similar stocks
CREATE OR REPLACE FUNCTION find_similar_stocks(
    query_ticker VARCHAR(10),
    dimension INT DEFAULT 256,
    limit_count INT DEFAULT 10
)
RETURNS TABLE(
    ticker VARCHAR(10),
    company_name VARCHAR(255),
    sector VARCHAR(50),
    similarity FLOAT
) AS $$
BEGIN
    IF dimension = 64 THEN
        RETURN QUERY
        SELECT 
            s.ticker,
            s.company_name,
            s.sector,
            1 - (s.embedding_64 <=> q.embedding_64) as similarity
        FROM stock_embeddings s, stock_embeddings q
        WHERE q.ticker = query_ticker
            AND s.ticker != query_ticker
        ORDER BY s.embedding_64 <=> q.embedding_64
        LIMIT limit_count;
    ELSIF dimension = 256 THEN
        RETURN QUERY
        SELECT 
            s.ticker,
            s.company_name,
            s.sector,
            1 - (s.embedding_256 <=> q.embedding_256) as similarity
        FROM stock_embeddings s, stock_embeddings q
        WHERE q.ticker = query_ticker
            AND s.ticker != query_ticker
        ORDER BY s.embedding_256 <=> q.embedding_256
        LIMIT limit_count;
    ELSIF dimension = 512 THEN
        RETURN QUERY
        SELECT 
            s.ticker,
            s.company_name,
            s.sector,
            1 - (s.embedding_512 <=> q.embedding_512) as similarity
        FROM stock_embeddings s, stock_embeddings q
        WHERE q.ticker = query_ticker
            AND s.ticker != query_ticker
        ORDER BY s.embedding_512 <=> q.embedding_512
        LIMIT limit_count;
    ELSE
        RAISE EXCEPTION 'Unsupported dimension: %', dimension;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to screen stocks by criteria embedding
CREATE OR REPLACE FUNCTION screen_stocks_by_embedding(
    query_embedding vector,
    min_market_cap BIGINT DEFAULT 0,
    sector_filter VARCHAR(50) DEFAULT NULL,
    limit_count INT DEFAULT 50
)
RETURNS TABLE(
    ticker VARCHAR(10),
    company_name VARCHAR(255),
    sector VARCHAR(50),
    market_cap BIGINT,
    similarity FLOAT
) AS $$
BEGIN
    -- Determine dimension from query_embedding
    IF vector_dims(query_embedding) = 256 THEN
        RETURN QUERY
        SELECT 
            s.ticker,
            s.company_name,
            s.sector,
            s.market_cap,
            1 - (s.embedding_256 <=> query_embedding) as similarity
        FROM stock_embeddings s
        WHERE s.market_cap >= min_market_cap
            AND (sector_filter IS NULL OR s.sector = sector_filter)
        ORDER BY s.embedding_256 <=> query_embedding
        LIMIT limit_count;
    ELSE
        RAISE EXCEPTION 'Query embedding must be 256 dimensions';
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_stock_embeddings_updated_at 
    BEFORE UPDATE ON stock_embeddings 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed for Cloud SQL)
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO readwrite_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO readwrite_user;