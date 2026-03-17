"""
Database Reset Script
Run this to clear all data and start fresh with a clean database.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_flask import app, db

def reset_database():
    """Reset the database by deleting and recreating all tables."""
    with app.app_context():
        print("🗑️  Dropping all database tables...")
        db.drop_all()
        print("✅ Tables dropped successfully")
        
        print("\n📝 Creating new database tables...")
        db.create_all()
        print("✅ New tables created successfully")
        
        print("\n✨ Database reset complete!")
        print("You can now register a new account and test the app.")

if __name__ == '__main__':
    confirm = input("⚠️  WARNING: This will delete ALL data! Are you sure? (yes/no): ").strip().lower()
    if confirm == 'yes':
        reset_database()
    else:
        print("Reset cancelled.")
