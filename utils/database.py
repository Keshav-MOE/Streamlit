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
        query = """
        SELECT
            COUNT(DISTINCT month) as total_months,
            (SELECT COUNT(*) FROM ticket_analysis) as total_tickets,
            AVG(priority_score) as avg_priority,
            (SELECT ticket_category FROM ticket_analysis GROUP BY ticket_category ORDER BY COUNT(*) DESC LIMIT 1) as top_category
        FROM ticket_analysis;
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
            if result:
                stats = {
                    'total_months': result[0] or 0,
                    'total_tickets': result[1] or 0,
                    'avg_priority': round(result[2] or 0, 1),
                    'top_category': result[3] or 'N/A'
                }
                return _to_native_type(stats)
            
        except Exception:
            pass # Fallthrough to return default
        
        return {'total_months': 0, 'total_tickets': 0, 'avg_priority': 0.0, 'top_category': 'N/A'}