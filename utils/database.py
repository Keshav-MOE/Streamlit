import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import json
import streamlit as st

class TicketDB:
    def __init__(self):
        # Use SQLite database (file-based for Streamlit Cloud)
        self.engine = create_engine('sqlite:///ticket_analysis.db')
        self.create_tables()
    
    def create_tables(self):
        """Create necessary tables"""
        create_analysis_table = """
        CREATE TABLE IF NOT EXISTS ticket_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            month TEXT,
            batch_number INTEGER,
            ticket_id TEXT,
            ticket_category TEXT,
            analysis_json TEXT,
            sdk_issues TEXT,
            improvement_suggestions TEXT,
            priority_score INTEGER,
            resolution_time_hours REAL,
            customer_satisfaction INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        create_summary_table = """
        CREATE TABLE IF NOT EXISTS monthly_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            month TEXT,
            total_tickets INTEGER,
            processed_tickets INTEGER,
            top_sdk_issues TEXT,
            improvement_priorities TEXT,
            overall_score REAL,
            avg_priority REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(create_analysis_table))
            conn.execute(text(create_summary_table))
            conn.commit()
    
    @st.cache_data(ttl=60)  # Cache for 1 minute
    def get_analysis_data(_self, month=None):
        """Get analysis data with optional month filter"""
        query = "SELECT * FROM ticket_analysis"
        params = {}
        
        if month:
            query += " WHERE month = :month"
            params['month'] = month
            
        query += " ORDER BY created_at DESC"
        
        return pd.read_sql(query, _self.engine, params=params)
    
    def save_batch_analysis(self, month, batch_num, analysis_results):
        """Save batch analysis results"""
        records = []
        
        for result in analysis_results:
            record = {
                'month': month,
                'batch_number': batch_num,
                'ticket_id': result.get('ticket_id', ''),
                'ticket_category': result.get('ticket_category', ''),
                'analysis_json': json.dumps(result),
                'sdk_issues': json.dumps(result.get('sdk_issues', [])),
                'improvement_suggestions': json.dumps(result.get('improvement_suggestions', [])),
                'priority_score': result.get('priority_score', 5),
                'resolution_time_hours': result.get('resolution_time_hours', 0),
                'customer_satisfaction': result.get('customer_satisfaction', 3),
                'created_at': datetime.now()
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_sql('ticket_analysis', self.engine, if_exists='append', index=False)
    
    def get_total_tickets(self):
        """Get total number of analyzed tickets"""
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) as count FROM ticket_analysis"))
            return result.fetchone()[0]
    
    def get_monthly_summaries(self):
        """Get all monthly summaries"""
        return pd.read_sql("SELECT * FROM monthly_summary ORDER BY created_at DESC", self.engine)
    
    def generate_monthly_summary(self, month):
        """Generate summary for a specific month"""
        df = self.get_analysis_data(month)
        
        if df.empty:
            return {}
        
        # Parse SDK issues
        all_sdk_issues = []
        for issues_str in df['sdk_issues'].dropna():
            try:
                issues = json.loads(issues_str)
                all_sdk_issues.extend(issues)
            except:
                continue
        
        # Parse improvement suggestions
        all_improvements = []
        for imp_str in df['improvement_suggestions'].dropna():
            try:
                improvements = json.loads(imp_str)
                all_improvements.extend(improvements)
            except:
                continue
        
        # Calculate summary metrics
        summary = {
            'total_tickets': len(df),
            'processed_tickets': len(df),
            'avg_priority': df['priority_score'].mean(),
            'avg_resolution_time': df['resolution_time_hours'].mean(),
            'avg_satisfaction': df['customer_satisfaction'].mean(),
            'top_sdk_issues': pd.Series(all_sdk_issues).value_counts().head(10).to_dict() if all_sdk_issues else {},
            'improvement_priorities': pd.Series(all_improvements).value_counts().head(10).to_dict() if all_improvements else {},
            'overall_score': df['customer_satisfaction'].mean() * df['priority_score'].mean() / 10
        }
        
        return summary
    
    def save_monthly_summary(self, month, summary):
        """Save monthly summary"""
        record = {
            'month': month,
            'total_tickets': summary.get('total_tickets', 0),
            'processed_tickets': summary.get('processed_tickets', 0),
            'top_sdk_issues': json.dumps(summary.get('top_sdk_issues', {})),
            'improvement_priorities': json.dumps(summary.get('improvement_priorities', {})),
            'overall_score': summary.get('overall_score', 0),
            'avg_priority': summary.get('avg_priority', 0),
            'created_at': datetime.now()
        }
        
        df = pd.DataFrame([record])
        df.to_sql('monthly_summary', self.engine, if_exists='append', index=False)
        # Add these methods to the TicketDB class

def get_recent_activity(self, limit=10):
    """Get recent activity for home page"""
    query = """
    SELECT month, ticket_category, COUNT(*) as count, 
           AVG(priority_score) as avg_priority,
           MAX(created_at) as last_updated
    FROM ticket_analysis 
    GROUP BY month, ticket_category 
    ORDER BY last_updated DESC 
    LIMIT :limit
    """
    return pd.read_sql(query, self.engine, params={'limit': limit})

def get_quick_stats(self):
    """Get quick statistics for sidebar"""
    with self.engine.connect() as conn:
        # Total months
        total_months = conn.execute(text(
            "SELECT COUNT(DISTINCT month) as count FROM ticket_analysis"
        )).fetchone()[0]
        
        # Current month tickets (latest month)
        latest_month = conn.execute(text(
            "SELECT month FROM ticket_analysis ORDER BY created_at DESC LIMIT 1"
        )).fetchone()
        
        if latest_month:
            current_month_tickets = conn.execute(text(
                "SELECT COUNT(*) as count FROM ticket_analysis WHERE month = :month"
            ), {'month': latest_month[0]}).fetchone()[0]
        else:
            current_month_tickets = 0
        
        # Average priority
        avg_priority = conn.execute(text(
            "SELECT AVG(priority_score) as avg FROM ticket_analysis"
        )).fetchone()[0] or 0
        
        # Top category
        top_category = conn.execute(text(
            """SELECT ticket_category, COUNT(*) as count 
               FROM ticket_analysis 
               GROUP BY ticket_category 
               ORDER BY count DESC 
               LIMIT 1"""
        )).fetchone()
        
        return {
            'total_months': total_months,
            'current_month_tickets': current_month_tickets,
            'avg_priority': round(avg_priority, 1),
            'top_category': top_category[0] if top_category else 'N/A'
        }

def get_database_size(self):
    """Get database size info"""
    try:
        import os
        if os.path.exists('ticket_analysis.db'):
            size = os.path.getsize('ticket_analysis.db')
            if size < 1024:
                return f"{size} bytes"
            elif size < 1024*1024:
                return f"{size/1024:.1f} KB"
            else:
                return f"{size/(1024*1024):.1f} MB"
        return "Unknown"
    except:
        return "Unknown"
