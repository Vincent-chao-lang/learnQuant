PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS radar_runs (
  run_id TEXT PRIMARY KEY,
  ts_hour TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL,        -- ok / degraded / fail
  error_code TEXT,
  error_msg TEXT,
  confidence REAL NOT NULL,
  sources_used TEXT NOT NULL   -- JSON
);

CREATE TABLE IF NOT EXISTS radar_hourly_ohlcv (
  ts_hour TEXT PRIMARY KEY,
  open REAL,
  high REAL,
  low REAL,
  close REAL NOT NULL,
  volume REAL,
  quality_flags TEXT NOT NULL  -- JSON
);

CREATE TABLE IF NOT EXISTS radar_events (
  event_id TEXT PRIMARY KEY,
  ts_hour TEXT NOT NULL,
  event_type TEXT NOT NULL,    -- FREEZE/CAUTIOUS/DATA_OUTAGE/SHOCK_MOVE...
  severity TEXT NOT NULL,      -- info/warn/critical
  title TEXT NOT NULL,
  details TEXT NOT NULL        -- JSON
);

CREATE TABLE IF NOT EXISTS radar_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL          -- JSON
);
