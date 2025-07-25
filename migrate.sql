PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS employees (
    employee_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE,
    department TEXT,
    employee_no TEXT UNIQUE,
    join_date DATE
);

CREATE TABLE IF NOT EXISTS face_templates (
    template_id TEXT PRIMARY KEY,
    employee_id TEXT NOT NULL,
    embedding BLOB,
    confidence REAL,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

CREATE TABLE IF NOT EXISTS attendance_logs (
    log_id TEXT PRIMARY KEY,
    employee_id TEXT NOT NULL,
    event_type TEXT CHECK(event_type IN ('ENTRY','EXIT')),
    timestamp DATETIME,
    device_id TEXT,
    confidence_score REAL,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_logs(date(timestamp));
