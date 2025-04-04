import sqlite3
from datetime import datetime

# Connect to the SQLite database (or create it if it doesn't exist)
def get_db_connection():
    conn = sqlite3.connect("spam.db")
    conn.row_factory = sqlite3.Row  # Enable row access by column name
    return conn

# Create the emails table
def create_table():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

# Insert a new email and prediction into the database
def insert_email(email_text, prediction):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO emails (text, prediction) VALUES (?, ?)",
        (email_text, prediction),
    )
    conn.commit()
    conn.close()

# Fetch all emails from the database
def fetch_all_emails():
    conn = get_db_connection()
    emails = conn.execute("SELECT * FROM emails").fetchall()
    conn.close()
    return emails

# Main function to initialize the database
if __name__ == "__main__":
    create_table()
    print("Database and table created successfully.")