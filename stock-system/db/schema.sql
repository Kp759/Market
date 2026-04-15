-- SQLite schema for the stock-analysis system
-- All tables share a (ticker, fetched_at, ttl_expires_at) caching contract.

-- ---------------------------------------------------------------------------
-- prices — daily OHLCV (1-year history, 15-min TTL)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prices (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    data           TEXT    NOT NULL,   -- JSON array of OHLCV dicts
    fetched_at     REAL    NOT NULL,   -- Unix timestamp
    ttl_expires_at REAL    NOT NULL    -- fetched_at + 900
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);
CREATE INDEX IF NOT EXISTS idx_prices_ttl    ON prices(ticker, ttl_expires_at);

-- ---------------------------------------------------------------------------
-- fundamentals — P/E, EPS, revenue growth, market cap (15-min TTL)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fundamentals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    data           TEXT    NOT NULL,   -- JSON object
    fetched_at     REAL    NOT NULL,
    ttl_expires_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker ON fundamentals(ticker);
CREATE INDEX IF NOT EXISTS idx_fundamentals_ttl    ON fundamentals(ticker, ttl_expires_at);

-- ---------------------------------------------------------------------------
-- news_articles — NewsAPI articles (1-hour TTL)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_articles (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    data           TEXT    NOT NULL,   -- JSON array of article dicts
    fetched_at     REAL    NOT NULL,
    ttl_expires_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_articles(ticker);
CREATE INDEX IF NOT EXISTS idx_news_ttl    ON news_articles(ticker, ttl_expires_at);

-- ---------------------------------------------------------------------------
-- iv_data — ATM implied volatility from nearest-expiry options (1-hour TTL)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS iv_data (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    data           TEXT    NOT NULL,   -- JSON: {"iv": 0.32}
    fetched_at     REAL    NOT NULL,
    ttl_expires_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_iv_ticker ON iv_data(ticker);
CREATE INDEX IF NOT EXISTS idx_iv_ttl    ON iv_data(ticker, ttl_expires_at);

-- ---------------------------------------------------------------------------
-- recommendations — LLM-generated ranked output
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recommendations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    score          REAL,
    recommendation TEXT,               -- 'buy' | 'hold' | 'sell'
    justification  TEXT,
    risk_flag      TEXT,
    signals        TEXT,               -- JSON blob of all input signals
    fetched_at     REAL    NOT NULL,
    ttl_expires_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rec_ticker ON recommendations(ticker);
CREATE INDEX IF NOT EXISTS idx_rec_ttl    ON recommendations(ticker, ttl_expires_at);
CREATE INDEX IF NOT EXISTS idx_rec_score  ON recommendations(score DESC);
