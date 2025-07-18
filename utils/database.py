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
    
    def clear_all_data(self):
        """Clear all data from the database"""
        try:
            with self.engine.connect() as conn:
                # Clear ticket analysis data
                conn.execute(text("DELETE FROM ticket_analysis"))
                # Clear monthly summaries
                conn.execute(text("DELETE FROM monthly_summary"))
                conn.commit()
            return True
        except Exception as e:
            st.error(f"Failed to clear database: {e}")
            return False
    
    def get_table_counts(self):
        """Get count of records in each table"""
        try:
            with self.engine.connect() as conn:
                ticket_count = conn.execute(text("SELECT COUNT(*) FROM ticket_analysis")).fetchone()[0]
                summary_count = conn.execute(text("SELECT COUNT(*) FROM monthly_summary")).fetchone()[0]
                
                return {
                    'tickets': ticket_count,
                    'summaries': summary_count,
                    'total': ticket_count + summary_count
                }
        except Exception as e:
            return {'tickets': 0, 'summaries': 0, 'total': 0}
    
    def vacuum_database(self):
        """Optimize database after clearing data"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("VACUUM"))
            return True
        except Exception as e:
            st.warning(f"Database optimization failed: {e}")
            return False
    
    def get_analysis_data(self, month=None):
        """Get analysis data with optional month filter - NO CACHING"""
        query = "SELECT * FROM ticket_analysis"
        params = {}
        
        if month:
            query += " WHERE month = :month"
            params['month'] = month
            
        query += " ORDER BY created_at DESC"
        
        try:
            return pd.read_sql(query, self.engine, params=params)
        except Exception as e:
            st.error(f"Database query failed: {e}")
            return pd.DataFrame()
    
    def save_batch_analysis(self, month, batch_num, analysis_results):
        """Save batch analysis results"""
        try:
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
            
            if records:
                df = pd.DataFrame(records)
                df.to_sql('ticket_analysis', self.engine, if_exists='append', index=False)
                st.success(f"✅ Saved batch {batch_num} with {len(records)} tickets for {month}")
            
        except Exception as e:
            st.error(f"Failed to save batch analysis: {e}")
            raise e
    
    def get_total_tickets(self):
        """Get total number of analyzed tickets"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) as count FROM ticket_analysis"))
                return result.fetchone()[0]
        except Exception as e:
            st.error(f"Failed to get total tickets: {e}")
            return 0
    
    def get_monthly_summaries(self):
        """Get all monthly summaries"""
        try:
            return pd.read_sql("SELECT * FROM monthly_summary ORDER BY created_at DESC", self.engine)
        except Exception as e:
            st.error(f"Failed to get monthly summaries: {e}")
            return pd.DataFrame()
    
    def generate_monthly_summary(self, month):
        """Generate summary for a specific month"""
        try:
            df = self.get_analysis_data(month)
            
            if df.empty:
                return {
                    'total_tickets': 0,
                    'processed_tickets': 0,
                    'avg_priority': 0,
                    'avg_resolution_time': 0,
                    'avg_satisfaction': 0,
                    'top_sdk_issues': {},
                    'improvement_priorities': {},
                    'overall_score': 0
                }
            
            # Parse SDK issues safely
            all_sdk_issues = []
            for issues_str in df['sdk_issues'].dropna():
                try:
                    issues = json.loads(issues_str) if isinstance(issues_str, str) else issues_str
                    if isinstance(issues, list):
                        all_sdk_issues.extend(issues)
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
            
            # Parse improvement suggestions safely
            all_improvements = []
            for imp_str in df['improvement_suggestions'].dropna():
                try:
                    improvements = json.loads(imp_str) if isinstance(imp_str, str) else imp_str
                    if isinstance(improvements, list):
                        all_improvements.extend(improvements)
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
            
            # Calculate summary metrics
            summary = {
                'total_tickets': len(df),
                'processed_tickets': len(df),
                'avg_priority': float(df['priority_score'].mean()) if not df['priority_score'].empty else 0,
                'avg_resolution_time': float(df['resolution_time_hours'].mean()) if not df['resolution_time_hours'].empty else 0,
                'avg_satisfaction': float(df['customer_satisfaction'].mean()) if not df['customer_satisfaction'].empty else 0,
                'top_sdk_issues': dict(pd.Series(all_sdk_issues).value_counts().head(10)) if all_sdk_issues else {},
                'improvement_priorities': dict(pd.Series(all_improvements).value_counts().head(10)) if all_improvements else {},
            }
            
            # Calculate overall score
            if summary['avg_satisfaction'] > 0 and summary['avg_priority'] > 0:
                summary['overall_score'] = float(summary['avg_satisfaction'] * summary['avg_priority'] / 10)
            else:
                summary['overall_score'] = 0.0
            
            return summary
            
        except Exception as e:
            st.error(f"Failed to generate monthly summary: {e}")
            return {}
    
    def save_monthly_summary(self, month, summary):
        """Save monthly summary"""
        try:
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
            st.success(f"✅ Monthly summary saved for {month}")
            
        except Exception as e:
            st.error(f"Failed to save monthly summary: {e}")
    
    def get_recent_activity(self, limit=10):
        """Get recent activity for home page"""
        try:
            query = """
            SELECT month, ticket_category, COUNT(*) as count, 
                   AVG(priority_score) as avg_priority,
                   MAX(created_at) as last_updated
            FROM ticket_analysis 
            GROUP BY month, ticket_category 
            ORDER BY last_updated DESC 
            LIMIT ?
            """
            return pd.read_sql(query, self.engine, params=[limit])
        except Exception as e:
            st.warning(f"Could not load recent activity: {e}")
            return pd.DataFrame()
    
    def get_quick_stats(self):
        """Get quick statistics for sidebar"""
        try:
            with self.engine.connect() as conn:
                # Total months
                total_months_result = conn.execute(text(
                    "SELECT COUNT(DISTINCT month) as count FROM ticket_analysis"
                ))
                total_months = total_months_result.fetchone()[0] or 0
                
                # Current month tickets (latest month)
                latest_month_result = conn.execute(text(
                    "SELECT month FROM ticket_analysis ORDER BY created_at DESC LIMIT 1"
                ))
                latest_month_row = latest_month_result.fetchone()
                
                if latest_month_row:
                    current_month_tickets_result = conn.execute(text(
                        "SELECT COUNT(*) as count FROM ticket_analysis WHERE month = :month"
                    ), {'month': latest_month_row[0]})
                    current_month_tickets = current_month_tickets_result.fetchone()[0] or 0
                else:
                    current_month_tickets = 0
                
                # Average priority
                avg_priority_result = conn.execute(text(
                    "SELECT AVG(priority_score) as avg FROM ticket_analysis"
                ))
                avg_priority = avg_priority_result.fetchone()[0] or 0
                
                # Top category
                top_category_result = conn.execute(text(
                    """SELECT ticket_category, COUNT(*) as count 
                       FROM ticket_analysis 
                       GROUP BY ticket_category 
                       ORDER BY count DESC 
                       LIMIT 1"""
                ))
                top_category_row = top_category_result.fetchone()
                top_category = top_category_row[0] if top_category_row else 'N/A'
                
                return {
                    'total_months': total_months,
                    'current_month_tickets': current_month_tickets,
                    'avg_priority': round(avg_priority, 1),
                    'top_category': top_category
                }
                
        except Exception as e:
            st.warning(f"Could not load quick stats: {e}")
            return {
                'total_months': 0,
                'current_month_tickets': 0,
                'avg_priority': 0,
                'top_category': 'N/A'
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
            return "0 KB"
        except Exception:
            return "Unknown"
    
    def clear_month_data(self, month):
        """Clear data for a specific month (useful for re-processing)"""
        try:
            with self.engine.connect() as conn:
                # Delete analysis data
                conn.execute(text("DELETE FROM ticket_analysis WHERE month = :month"), {'month': month})
                # Delete summary data
                conn.execute(text("DELETE FROM monthly_summary WHERE month = :month"), {'month': month})
                conn.commit()
                st.success(f"✅ Cleared data for {month}")
        except Exception as e:
            st.error(f"Failed to clear data for {month}: {e}")
    
    def get_processing_status(self, month):
        """Get processing status for a month"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT COUNT(*) as count FROM ticket_analysis WHERE month = :month"
                ), {'month': month})
                count = result.fetchone()[0]
                return count > 0
        except Exception:
            return False
