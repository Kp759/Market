-- SQLite schema for stock data caching

CREATE TABLE IF NOT EXISTS prices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    data        TEXT    NOT NULL,  -- JSON-encoded OHLCV records
    fetched_at      REAL NOT NULL,  -- Unix timestamp
    ttl_expires_at  REAL NOT NULL   -- Unix timestamp (fetched_at + 900s)
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);
CREATE INDEX IF NOT EXISTS idx_prices_ttl   ON prices(ticker, ttl_expires_at);

-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fundamentals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    pe_ratio    REAL,
    eps         REAL,
    revenue     REAL,
    data        TEXT    NOT NULL,  -- full JSON blob for extensibility
    fetched_at      REAL NOT NULL,
    ttl_expires_at  REAL NOT NULL   -- fetched_at + 900s (same TTL as prices)
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker ON fundamentals(ticker);
CREATE INDEX IF NOT EXISTS idx_fundamentals_ttl    ON fundamentals(ticker, ttl_expires_at);

-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS news_articles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    data        TEXT    NOT NULL,  -- JSON array of article objects
    fetched_at      REAL NOT NULL,
    ttl_expires_at  REAL NOT NULL   -- fetched_at + 3600s (1 hour)
);

CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_articles(ticker);
CREATE INDEX IF NOT EXISTS idx_news_ttl    ON news_articles(ticker, ttl_expires_at);
