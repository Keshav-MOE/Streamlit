import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text, Engine
from datetime import datetime
import json
import os
import numpy as np
from typing import Any, Dict, List, Optional, Union

# --- Consolidated Data Conversion Utilities ---

def _to_native_type(obj: Any) -> Any:
    """
    Recursively converts pandas/numpy data types to native Python types.
    This single utility replaces the multiple conversion functions from the original code.
    """
    if pd.isna(obj) or obj is None:
        return None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.ndarray, list, tuple, pd.Series)):
        # Consistently return a list, don't unwrap single-element arrays
        return [_to_native_type(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _to_native_type(v) for k, v in obj.items()}
    if hasattr(obj, 'item'): # Catches other numpy scalar types
        return obj.item()
    return obj

def _safe_json_dumps(data: Any) -> str:
    """
    Safely serializes an object to a JSON string after converting it to native types.
    Handles potential serialization errors gracefully.
    """
    try:
        native_data = _to_native_type(data)
        return json.dumps(native_data, ensure_ascii=False)
    except (TypeError, OverflowError):
        # Fallback for complex, non-serializable objects
        return json.dumps(str(data))

# --- Refactored Database Class ---

class TicketDB:
    """
    A refactored class to manage ticket analysis data in a SQLite database.
    This version is decoupled from Streamlit and uses more robust data handling.
    """
    def __init__(self, db_path: str = 'ticket_analysis.db'):
        """Initializes the database engine and creates tables if they don't exist."""
        self.db_path = db_path
        self.engine: Engine = create_engine(f'sqlite:///{db_path}')
        self._create_tables()

    def _create_tables(self):
        """Creates the necessary database tables using 'IF NOT EXISTS'."""
        with self.engine.connect() as conn:
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ticket_analysis (
                id INTEGER PRIMARY KEY,
                month TEXT NOT NULL,
                batch_number INTEGER NOT NULL,
                ticket_id TEXT NOT NULL,
                ticket_category TEXT,
                analysis_json TEXT,
                sdk_issues TEXT,
                improvement_suggestions TEXT,
                priority_score INTEGER,
                resolution_time_hours REAL,
                customer_satisfaction INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """))
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS monthly_summary (
                id INTEGER PRIMARY KEY,
                month TEXT UNIQUE NOT NULL,
                total_tickets INTEGER,
                processed_tickets INTEGER,
                top_sdk_issues TEXT,
                improvement_priorities TEXT,
                overall_score REAL,
                avg_priority REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """))
            conn.commit()

    def clear_all_data(self) -> bool:
        """
        Clears all data from all tables using SQLAlchemy.
        Returns True on success, False on failure.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM ticket_analysis"))
                conn.execute(text("DELETE FROM monthly_summary"))
                # Reset autoincrement counters in SQLite
                conn.execute(text("DELETE FROM sqlite_sequence WHERE name IN ('ticket_analysis', 'monthly_summary')"))
                conn.commit()
            return True
        except Exception:
            return False

    def get_total_tickets(self) -> int:
        """Gets the total number of tickets from the database."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM ticket_analysis")).scalar_one_or_none()
                return result or 0
        except Exception:
            return 0

    def get_analysis_data(self, month: Optional[str] = None) -> pd.DataFrame:
        """Retrieves analysis data, optionally filtered by month."""
        query = "SELECT * FROM ticket_analysis"
        params = {}
        if month:
            query += " WHERE month = :month"
            params['month'] = month
        query += " ORDER BY created_at DESC"
        
        try:
            return pd.read_sql(query, self.engine, params=params)
        except Exception:
            return pd.DataFrame()

    def save_batch_analysis(self, month: str, batch_num: int, analysis_results: List[Dict[str, Any]]) -> tuple[bool, int]:
        """
        Saves a batch of analysis results to the database.
        This method is now simpler and expects cleaner input data.
        Returns a tuple of (success_status, number_of_records_saved).
        """
        if not isinstance(analysis_results, list) or not analysis_results:
            return False, 0

        records = []
        for i, result in enumerate(analysis_results):
            if not isinstance(result, dict):
                continue  # Skip invalid entries

            # Simplified data extraction assuming a predictable structure.
            # Type validation and cleaning should happen before this stage.
            record = {
                'month': str(month),
                'batch_number': int(batch_num),
                'ticket_id': str(result.get('ticket_id', f'unknown_{i}')),
                'ticket_category': str(result.get('ticket_category', 'General')),
                'analysis_json': _safe_json_dumps(result),
                'sdk_issues': _safe_json_dumps(result.get('sdk_issues', [])),
                'improvement_suggestions': _safe_json_dumps(result.get('improvement_suggestions', [])),
                'priority_score': int(result.get('priority_score', 5)),
                'resolution_time_hours': float(result.get('resolution_time_hours', 0.0)),
                'customer_satisfaction': int(result.get('customer_satisfaction', 3)),
                'created_at': datetime.now()
            }
            records.append(record)

        if not records:
            return False, 0

        try:
            df = pd.DataFrame(records)
            df.to_sql('ticket_analysis', self.engine, if_exists='append', index=False)
            return True, len(records)
        except Exception:
            return False, 0

    def save_single_analysis(self, month: str, ticket_result: dict) -> bool:
        """Save a single analysis result to the database."""
        from datetime import datetime
        if not isinstance(ticket_result, dict):
            return False
        record = {
            'month': str(month),
            'batch_number': 1,
            'ticket_id': str(ticket_result.get('ticket_id', 'unknown')),
            'ticket_category': str(ticket_result.get('ticket_category', 'General')),
            'analysis_json': _safe_json_dumps(ticket_result),
            'sdk_issues': _safe_json_dumps(ticket_result.get('sdk_issues', [])),
            'improvement_suggestions': _safe_json_dumps(ticket_result.get('improvement_suggestions', [])),
            'priority_score': int(ticket_result.get('priority_score', 5)),
            'resolution_time_hours': float(ticket_result.get('resolution_time_hours', 0.0)),
            'customer_satisfaction': int(ticket_result.get('customer_satisfaction', 3)),
            'created_at': datetime.now()
        }
        try:
            df = pd.DataFrame([record])
            df.to_sql('ticket_analysis', self.engine, if_exists='append', index=False)
            return True
        except Exception:
            return False

    def generate_monthly_summary(self, month: str) -> Optional[Dict[str, Any]]:
        """Generates and saves a summary for a specific month."""
        df = self.get_analysis_data(month)
        if df.empty:
            return None

        # Helper to safely parse JSON strings from the database
        def _parse_json_col(series: pd.Series) -> List[str]:
            items = []
            for json_str in series.dropna():
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        items.extend(str(item) for item in data if item)
                except (json.JSONDecodeError, TypeError):
                    continue
            return items

        all_sdk_issues = _parse_json_col(df['sdk_issues'])
        all_improvements = _parse_json_col(df['improvement_suggestions'])
        
        # Calculate summary metrics safely
        avg_priority = pd.to_numeric(df['priority_score'], errors='coerce').mean()
        avg_satisfaction = pd.to_numeric(df['customer_satisfaction'], errors='coerce').mean()
        overall_score = (avg_satisfaction * avg_priority / 10) if pd.notna(avg_satisfaction) and pd.notna(avg_priority) else 0.0

        summary = {
            'month': month,
            'total_tickets': len(df),
            'processed_tickets': len(df),
            'top_sdk_issues': _safe_json_dumps(pd.Series(all_sdk_issues).value_counts().head(10).to_dict()),
            'improvement_priorities': _safe_json_dumps(pd.Series(all_improvements).value_counts().head(10).to_dict()),
            'overall_score': float(overall_score),
            'avg_priority': float(avg_priority),
            'created_at': datetime.now()
        }
        
        return _to_native_type(summary)

    def save_monthly_summary(self, summary: Dict[str, Any]) -> bool:
        """
        Saves a monthly summary. Uses 'REPLACE' logic by deleting an existing record.
        """
        if not summary or 'month' not in summary:
            return False

        month = summary['month']
        try:
            with self.engine.connect() as conn:
                # Use INSERT OR REPLACE behavior by deleting first
                conn.execute(text("DELETE FROM monthly_summary WHERE month = :month"), {'month': month})
                conn.commit()
            
            df = pd.DataFrame([summary])
            df.to_sql('monthly_summary', self.engine, if_exists='append', index=False)
            return True
        except Exception:
            return False
            
    def get_monthly_summaries(self) -> pd.DataFrame:
        """Gets all monthly summaries from the database."""
        try:
            return pd.read_sql("SELECT * FROM monthly_summary ORDER BY month DESC", self.engine)
        except Exception:
            return pd.DataFrame()

    def get_quick_stats(self) -> Dict[str, Any]:
        """Retrieves quick statistics for a dashboard display."""
        # Get the most recent month from the summary table for accuracy
        latest_month_query = "SELECT month FROM monthly_summary ORDER BY month DESC LIMIT 1"
        
        stats_query = """
        SELECT
            COUNT(DISTINCT month) as total_months,
            (SELECT COUNT(*) FROM ticket_analysis) as total_tickets,
            AVG(priority_score) as avg_priority,
            (SELECT ticket_category FROM ticket_analysis GROUP BY ticket_category ORDER BY COUNT(*) DESC LIMIT 1) as top_category
        FROM ticket_analysis;
        """
        
        try:
            with self.engine.connect() as conn:
                latest_month_result = conn.execute(text(latest_month_query)).scalar_one_or_none()
                latest_month = latest_month_result if latest_month_result else ""

                current_month_tickets = 0
                if latest_month:
                    current_month_tickets_query = "SELECT COUNT(*) FROM ticket_analysis WHERE month = :month"
                    current_month_tickets = conn.execute(text(current_month_tickets_query), {'month': latest_month}).scalar_one()

                result = conn.execute(text(stats_query)).fetchone()

            if result:
                stats = {
                    'total_months': result[0] or 0,
                    'total_tickets': result[1] or 0,
                    'avg_priority': round(result[2] or 0, 1),
                    'top_category': result[3] or 'N/A',
                    'current_month_tickets': current_month_tickets or 0
                }
                return _to_native_type(stats)
            
        except Exception:
            pass # Fallthrough to return default
        
        return {'total_months': 0, 'total_tickets': 0, 'avg_priority': 0.0, 'top_category': 'N/A', 'current_month_tickets': 0}

    def get_processing_status(self, month: str) -> bool:
        """Checks if a given month has already been processed."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT 1 FROM ticket_analysis WHERE month = :month LIMIT 1"),
                    {'month': month}
                ).scalar_one_or_none()
                return result is not None
        except Exception:
            return False

    def clear_month_data(self, month: str) -> bool:
        """Clears all data for a specific month."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM ticket_analysis WHERE month = :month"), {'month': month})
                conn.execute(text("DELETE FROM monthly_summary WHERE month = :month"), {'month': month})
                conn.commit()
            return True
        except Exception:
            return False

    def debug_database_contents(self) -> List[str]:
        """Returns a list of strings with debug information about the database."""
        debug_info = []
        try:
            with self.engine.connect() as conn:
                tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()
                debug_info.append(f"✅ Tables found: {[table[0] for table in tables]}")
                ticket_count = conn.execute(text("SELECT COUNT(*) FROM ticket_analysis")).scalar_one()
                debug_info.append(f"✅ ticket_analysis rows: {ticket_count}")
                if ticket_count > 0:
                    latest_ticket = conn.execute(text("SELECT created_at FROM ticket_analysis ORDER BY created_at DESC LIMIT 1")).scalar_one()
                    debug_info.append(f"✅ Latest ticket entry: {latest_ticket}")
                summary_count = conn.execute(text("SELECT COUNT(*) FROM monthly_summary")).scalar_one()
                debug_info.append(f"✅ monthly_summary rows: {summary_count}")
                if summary_count > 0:
                    latest_summary = conn.execute(text("SELECT month FROM monthly_summary ORDER BY month DESC LIMIT 1")).scalar_one()
                    debug_info.append(f"✅ Latest summary month: {latest_summary}")
        except Exception as e:
            debug_info.append(f"❌ Error during debug check: {e}")
        return debug_info

    def get_recent_activity(self, limit: int = 5) -> pd.DataFrame:
        """Retrieves the most recent analysis activities."""
        query = "SELECT created_at, month, batch_number, ticket_id, ticket_category, priority_score FROM ticket_analysis ORDER BY created_at DESC LIMIT :limit"
        try:
            return pd.read_sql(query, self.engine, params={'limit': limit})
        except Exception:
            return pd.DataFrame()

    def get_database_size(self) -> str:
        """Returns the database file size as a formatted string."""
        try:
            size_bytes = os.path.getsize(self.db_path)
            if size_bytes < 1024: return f"{size_bytes} B"
            elif size_bytes < 1024**2: return f"{size_bytes/1024:.2f} KB"
            else: return f"{size_bytes/1024**2:.2f} MB"
        except OSError:
            return "N/A"

    def get_table_counts(self) -> Dict[str, int]:
        """Returns the row counts for main tables."""
        counts = {'tickets': 0, 'summaries': 0}
        try:
            with self.engine.connect() as conn:
                counts['tickets'] = conn.execute(text("SELECT COUNT(*) FROM ticket_analysis")).scalar_one()
                counts['summaries'] = conn.execute(text("SELECT COUNT(*) FROM monthly_summary")).scalar_one()
        except Exception: pass
        return counts

    def vacuum_database(self):
        """Runs the VACUUM command to rebuild the database file."""
        try:
            with self.engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT").execute(text("VACUUM"))
        except Exception: pass

    def force_verify_data(self) -> Dict[str, Any]:
        """Performs a quick verification of database contents."""
        try:
            with self.engine.connect() as conn:
                ticket_count = conn.execute(text("SELECT COUNT(*) FROM ticket_analysis")).scalar_one()
                summary_count = conn.execute(text("SELECT COUNT(*) FROM monthly_summary")).scalar_one()
                months = conn.execute(text("SELECT DISTINCT month FROM ticket_analysis ORDER BY month DESC")).scalars().all()
                return {'ticket_count': ticket_count, 'summary_count': summary_count, 'months': months, 'verified_at': datetime.now().isoformat()}
        except Exception as e:
            return {'error': str(e)}