import pandas as pd
import streamlit as st

class DataProcessor:
    def __init__(self):
        pass
    
    def merge_ticket_data(self, details_df: pd.DataFrame, comments_df: pd.DataFrame) -> pd.DataFrame:
        """Merge ticket details with comments/conversations"""
        
        # Try different common column names for joining
        possible_join_keys = [
            'ticket_id', 
            'Ticket ID', 
            'id', 
            'ID',
            'ticket_number',
            'TicketID'
        ]
        
        join_key = None
        for key in possible_join_keys:
            if key in details_df.columns and key in comments_df.columns:
                join_key = key
                break
        
        if not join_key:
            st.error(f"""
            **Cannot merge files - no common ID column found!**
            
            Details file columns: {list(details_df.columns)}
            Comments file columns: {list(comments_df.columns)}
            
            Please ensure both files have a common column like 'ticket_id', 'ID', etc.
            """)
            raise ValueError("No common join key found between the two files")
        
        st.info(f"Merging files using column: **{join_key}**")
        
        # Perform the merge
        try:
            merged_df = pd.merge(
                details_df, 
                comments_df, 
                on=join_key, 
                how='left'
            )
            
            # Clean up column names
            merged_df = self._standardize_columns(merged_df)
            
            st.success(f"✅ Successfully merged {len(merged_df)} records")
            
            return merged_df
            
        except Exception as e:
            st.error(f"Error during merge: {e}")
            raise
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistent processing"""
        
        # Create column mapping
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower().strip()
            
            # Map common variations to standard names
            if any(x in col_lower for x in ['ticket_id', 'ticket id', 'ticketid', 'id']):
                if 'ticket_id' not in column_mapping.values():
                    column_mapping[col] = 'ticket_id'
            
            elif any(x in col_lower for x in ['subject', 'title', 'sub_name', 'ticket_sub_name']):
                if 'ticket_sub_name' not in column_mapping.values():
                    column_mapping[col] = 'ticket_sub_name'
            
            elif any(x in col_lower for x in ['resolution_time', 'resolution time', 'time_to_resolve']):
                if 'resolution_time' not in column_mapping.values():
                    column_mapping[col] = 'resolution_time'
            
            elif any(x in col_lower for x in ['conversation', 'comments', 'description', 'ticket_conversation']):
                if 'ticket_conversation' not in column_mapping.values():
                    column_mapping[col] = 'ticket_conversation'
            
            elif any(x in col_lower for x in ['call', 'phone', 'did_call', 'customer_call']):
                if 'did_call_happened' not in column_mapping.values():
                    column_mapping[col] = 'did_call_happened'
            
            elif any(x in col_lower for x in ['category', 'type', 'issue_type']):
                if 'category' not in column_mapping.values():
                    column_mapping[col] = 'category'
        
        # Apply mapping
        if column_mapping:
            df = df.rename(columns=column_mapping)
            st.info(f"Standardized columns: {dict(column_mapping)}")
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> dict:
        """Validate data quality and return quality metrics"""
        
        quality_report = {
            'total_records': len(df),
            'missing_ticket_ids': df['ticket_id'].isna().sum() if 'ticket_id' in df.columns else 0,
            'missing_subjects': df['ticket_sub_name'].isna().sum() if 'ticket_sub_name' in df.columns else 0,
            'missing_conversations': df['ticket_conversation'].isna().sum() if 'ticket_conversation' in df.columns else 0,
            'duplicate_tickets': df.duplicated(subset=['ticket_id']).sum() if 'ticket_id' in df.columns else 0,
            'empty_conversations': 0,
            'data_quality_score': 0
        }
        
        # Check for empty conversations
        if 'ticket_conversation' in df.columns:
            quality_report['empty_conversations'] = (
                df['ticket_conversation'].fillna('').str.strip().eq('').sum()
            )
        
        # Calculate quality score (0-100)
        total_fields = len(df) * 3  # ticket_id, subject, conversation
        missing_fields = (
            quality_report['missing_ticket_ids'] + 
            quality_report['missing_subjects'] + 
            quality_report['missing_conversations']
        )
        
        quality_report['data_quality_score'] = max(0, 100 - (missing_fields / total_fields * 100))
        
        return quality_report
