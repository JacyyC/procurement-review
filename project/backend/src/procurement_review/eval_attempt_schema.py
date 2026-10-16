"""SQLite schema for the V1-07 durable attempt ledger."""

SCHEMA_SQL = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE budget (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    max_requests INTEGER NOT NULL CHECK (max_requests > 0),
    max_input_tokens INTEGER CHECK (max_input_tokens > 0),
    max_output_tokens INTEGER CHECK (max_output_tokens > 0),
    reserved_requests INTEGER NOT NULL DEFAULT 0,
    reserved_input_tokens INTEGER NOT NULL DEFAULT 0,
    reserved_output_tokens INTEGER NOT NULL DEFAULT 0,
    known_input_tokens INTEGER NOT NULL DEFAULT 0,
    known_output_tokens INTEGER NOT NULL DEFAULT 0,
    unknown_usage_attempts INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE tasks (
    task_key TEXT PRIMARY KEY,
    payload_sha256 TEXT NOT NULL,
    state TEXT NOT NULL,
    committed_attempt_id INTEGER,
    projection_json TEXT,
    projection_sha256 TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_key TEXT NOT NULL REFERENCES tasks(task_key),
    attempt_no INTEGER NOT NULL CHECK (attempt_no > 0),
    request_sha256 TEXT NOT NULL,
    state TEXT NOT NULL,
    reserved_input_tokens INTEGER NOT NULL CHECK (reserved_input_tokens >= 0),
    reserved_output_tokens INTEGER NOT NULL CHECK (reserved_output_tokens >= 0),
    receipt_json TEXT,
    receipt_sha256 TEXT,
    projection_json TEXT,
    projection_sha256 TEXT,
    created_at TEXT NOT NULL,
    sent_at TEXT,
    terminal_at TEXT,
    UNIQUE(task_key, attempt_no)
);
CREATE TABLE events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    task_key TEXT,
    attempt_id INTEGER,
    occurred_at TEXT NOT NULL,
    detail_json TEXT NOT NULL
);
"""

__all__ = ["SCHEMA_SQL"]
