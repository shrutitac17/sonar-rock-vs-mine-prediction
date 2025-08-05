# logger.py

import sqlite3
from datetime import datetime

DB_FILE = "logs.db"

# Create the logs table if not exists
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_data TEXT,
                    prediction TEXT,
                    confidence REAL,
                    timestamp TEXT
                )''')
    conn.commit()
    conn.close()

# Log a new prediction
def log_prediction(input_data, prediction, confidence):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO predictions (input_data, prediction, confidence, timestamp) VALUES (?, ?, ?, ?)",
              (str(input_data), prediction, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# Retrieve all logs
def get_logs():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows
