import pandas as pd
import streamlit as st
import re

class DataProcessor:
    def __init__(self):
        self.processed_count = 0
        self.merge_log = []
        
    def merge_ticket_data(self, details_df: pd.DataFrame, comments_df: pd.DataFrame) -> pd.DataFrame:
        """Merge ticket details with comments/conversations with debug info"""
        
        st.info("🔄 Starting data merge process...")
        
        # Debug: Show input data info
        st.markdown(f"""
        **Input Data Info:**
        - Details file: {len(details_df)} rows, {len(details_df.columns)} columns
        - Comments file: {len(comments_df)} rows, {len(comments_df.columns)} columns
        """)
        
        # Try different common column names for joining
        possible_join_keys = [
            'Ticket_ID',        # Your actual column name
            'ticket_id', 
            'Ticket ID', 
            'TicketID',
            'id', 
            'ID',
            'ticket_number',
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
            
            Expected: 'Ticket_ID' in both files
            """)
            raise ValueError("No common join key found between the two files")
        
        st.success(f"✅ Found common column: **{join_key}**")
        
        # Show sample data before merge
        with st.expander("🔍 Sample Data Preview"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Details Sample:**")
                st.dataframe(details_df.head(3))
            with col2:
                st.markdown("**Comments Sample:**")
                st.dataframe(comments_df.head(3))
        
        # Check for duplicates
        details_dupes = details_df[join_key].duplicated().sum()
        comments_dupes = comments_df[join_key].duplicated().sum()
        
        if details_dupes > 0:
            st.warning(f"⚠️ Found {details_dupes} duplicate ticket IDs in details file")
        if comments_dupes > 0:
            st.warning(f"⚠️ Found {comments_dupes} duplicate ticket IDs in comments file")
        
        # Perform the merge
        try:
            st.info(f"🔗 Merging on column '{join_key}'...")
            
            merged_df = pd.merge(
                details_df, 
                comments_df, 
                on=join_key, 
                how='left'  # Keep all tickets, even if no comments
            )
            
            # Clean up column names
            merged_df = self._standardize_columns_for_your_data(merged_df)
            
            # Show merge results
            original_count = len(details_df)
            merged_count = len(merged_df)
            matched_count = len(merged_df[merged_df['ticket_conversation'].notna()])
            
            st.success(f"✅ Merge completed!")
            st.markdown(f"""
            **Merge Results:**
            - Original tickets: {original_count}
            - After merge: {merged_count}
            - With comments: {matched_count}
            - Missing comments: {merged_count - matched_count}
            """)
            
            if merged_count - matched_count > 0:
                st.warning(f"⚠️ {merged_count - matched_count} tickets have no matching comments")
            
            # Final data quality check
            quality_report = self.validate_data_quality(merged_df)
            self._show_quality_report(quality_report)
            
            return merged_df
            
        except Exception as e:
            st.error(f"❌ Error during merge: {e}")
            raise

    def _standardize_columns_for_your_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for your specific data structure"""
        
        # Create column mapping for your exact column names
        column_mapping = {
            'Ticket_ID': 'ticket_id',
            'Ticket subject': 'ticket_sub_name',
            'Full Resolution Time': 'resolution_time',
            'Comments': 'ticket_conversation',
            'Call Happened for the Customer?': 'did_call_happened',
            'Customer Tier ': 'customer_tier',
            'SDK': 'sdk_name',
            'SDK Issue Types': 'sdk_issue_category',
            'Ticket organization name': 'organization_name'
        }
        
        # Apply mapping only for columns that exist
        existing_mapping = {old: new for old, new in column_mapping.items() if old in df.columns}
        
        if existing_mapping:
            df = df.rename(columns=existing_mapping)
            st.info(f"🔄 Standardized columns: {list(existing_mapping.keys())}")
        
        # Handle resolution time - convert to hours if needed
        if 'resolution_time' in df.columns:
            df = self._convert_resolution_time(df)
        
        # Fill missing comments with default text
        if 'ticket_conversation' in df.columns:
            df['ticket_conversation'] = df['ticket_conversation'].fillna('No conversation available')
        
        return df
    
    def _convert_resolution_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert resolution time to hours with debug info"""
        
        def parse_resolution_time(time_str):
            """Parse various time formats to hours"""
            if pd.isna(time_str) or time_str == '':
                return 0.0
            
            time_str = str(time_str).lower().strip()
            
            try:
                # If already a number, assume it's hours
                if time_str.replace('.', '').replace('-', '').replace('+', '').isdigit():
                    return float(time_str)
                
                # Parse formats like "2 days 3 hours", "5h 30m", etc.
                hours = 0.0
                
                # Days
                if 'day' in time_str:
                    days_match = re.search(r'(\d+\.?\d*)\s*day', time_str)
                    if days_match:
                        hours += float(days_match.group(1)) * 24
                
                # Hours
                if 'hour' in time_str or 'hr' in time_str or 'h' in time_str:
                    hours_match = re.search(r'(\d+\.?\d*)\s*(?:hour|hr|h)', time_str)
                    if hours_match:
                        hours += float(hours_match.group(1))
                
                # Minutes
                if 'min' in time_str or 'm' in time_str:
                    minutes_match = re.search(r'(\d+\.?\d*)\s*(?:min|m)', time_str)
                    if minutes_match:
                        hours += float(minutes_match.group(1)) / 60
                
                return hours if hours > 0 else 0.0
                
            except Exception:
                return 0.0
        
        # Show sample resolution times before conversion
        sample_times = df['resolution_time'].head(5).tolist()
        st.info(f"🔄 Converting resolution times. Sample: {sample_times}")
        
        # Apply conversion
        df['resolution_time_hours'] = df['resolution_time'].apply(parse_resolution_time)
        
        # Show conversion results
        converted_sample = df['resolution_time_hours'].head(5).tolist()
        st.success(f"✅ Converted to hours. Sample: {converted_sample}")
        
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
            'data_quality_score': 0,
            'columns_found': list(df.columns),
            'resolution_time_stats': {}
        }
        
        # Check for empty conversations
        if 'ticket_conversation' in df.columns:
            quality_report['empty_conversations'] = (
                df['ticket_conversation'].fillna('').str.strip().eq('').sum()
            )
        
        # Resolution time statistics
        if 'resolution_time_hours' in df.columns:
            quality_report['resolution_time_stats'] = {
                'mean': df['resolution_time_hours'].mean(),
                'median': df['resolution_time_hours'].median(),
                'max': df['resolution_time_hours'].max(),
                'min': df['resolution_time_hours'].min()
            }
        
        # Calculate quality score (0-100)
        total_fields = len(df) * 3  # ticket_id, subject, conversation
        missing_fields = (
            quality_report['missing_ticket_ids'] + 
            quality_report['missing_subjects'] + 
            quality_report['missing_conversations']
        )
        
        if total_fields > 0:
            quality_report['data_quality_score'] = max(0, 100 - (missing_fields / total_fields * 100))
        
        return quality_report
    
    def _show_quality_report(self, quality_report):
        """Display data quality report"""
        
        st.markdown("### 📊 Data Quality Report")
        
        # Overall score
        score = quality_report['data_quality_score']
        if score >= 90:
            score_color = "🟢"
        elif score >= 70:
            score_color = "🟡"
        else:
            score_color = "🔴"
        
        st.markdown(f"**Overall Quality Score: {score_color} {score:.1f}%**")
        
        # Detailed metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Data Completeness:**")
            st.write(f"• Total records: {quality_report['total_records']:,}")
            st.write(f"• Missing ticket IDs: {quality_report['missing_ticket_ids']}")
            st.write(f"• Missing subjects: {quality_report['missing_subjects']}")
            st.write(f"• Missing conversations: {quality_report['missing_conversations']}")
            st.write(f"• Duplicate tickets: {quality_report['duplicate_tickets']}")
        
        with col2:
            st.markdown("**📈 Resolution Time Stats:**")
            if quality_report['resolution_time_stats']:
                stats = quality_report['resolution_time_stats']
                st.write(f"• Average: {stats['mean']:.1f} hours")
                st.write(f"• Median: {stats['median']:.1f} hours")
                st.write(f"• Range: {stats['min']:.1f} - {stats['max']:.1f} hours")
            else:
                st.write("• No resolution time data")
        
        # Show issues if any
        issues = []
        if quality_report['missing_ticket_ids'] > 0:
            issues.append(f"Missing ticket IDs: {quality_report['missing_ticket_ids']}")
        if quality_report['duplicate_tickets'] > 0:
            issues.append(f"Duplicate tickets: {quality_report['duplicate_tickets']}")
        if quality_report['missing_conversations'] > 0:
            issues.append(f"Missing conversations: {quality_report['missing_conversations']}")
        
        if issues:
            st.warning("⚠️ **Data Issues Found:**")
            for issue in issues:
                st.write(f"• {issue}")
        else:
            st.success("✅ **No major data issues found**")
