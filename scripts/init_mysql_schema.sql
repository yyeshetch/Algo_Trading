-- Algo Trading session-pipeline tables (run in InterServer phpMyAdmin)
-- Database: st83583_stocks_analysis

CREATE TABLE IF NOT EXISTS signals (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  trade_date DATE NOT NULL,
  bar_timestamp VARCHAR(32) NOT NULL,
  underlying VARCHAR(32) NOT NULL,
  asset_class VARCHAR(16) NOT NULL,
  row_json JSON NOT NULL,
  KEY idx_signals_lookup (trade_date, underlying, asset_class),
  KEY idx_signals_ts (trade_date, bar_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS market_snapshots (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  trade_date DATE NOT NULL,
  bar_timestamp VARCHAR(32) NOT NULL,
  underlying VARCHAR(32) NOT NULL,
  asset_class VARCHAR(16) NOT NULL,
  row_json JSON NOT NULL,
  KEY idx_snapshots_lookup (trade_date, underlying, asset_class),
  KEY idx_snapshots_ts (trade_date, bar_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS option_chain_rows (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  trade_date DATE NOT NULL,
  bar_timestamp VARCHAR(32) NOT NULL,
  underlying VARCHAR(32) NOT NULL,
  row_json JSON NOT NULL,
  KEY idx_option_chain_lookup (trade_date, underlying, bar_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS json_artifacts (
  artifact_type VARCHAR(64) NOT NULL,
  trade_date DATE NOT NULL,
  underlying VARCHAR(32) NOT NULL DEFAULT '',
  payload JSON NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (artifact_type, trade_date, underlying)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
