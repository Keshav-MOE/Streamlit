import streamlit as st
import pandas as pd
import time
from utils.data_processor import DataProcessor
from sqlalchemy import create_engine
import io

def main():
    st.title("📁 Upload Monthly Ticket Data")
    
    # Check prerequisites
    if not st.session_state.get('api_configured', False):
        st.error("🚨 Please configure Gemini API key in the main page first!")
        st.markdown("""
        **To get started:**
        1. Go back to the Home page
        2. Enter your Gemini API key in the sidebar
        3. Return here to upload data
        """)
        if st.button("🏠 Go to Home", use_container_width=True):
            st.switch_page("streamlit_app.py")
        return
    
    # System status check
    show_system_status()
    
    # Month and Year selection
    st.markdown("### 📅 Select Time Period")
    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox(
            "Select Month:",
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"],
            index=4  # Default to May
        )
    with col2:
        year = st.number_input("Year:", min_value=2020, max_value=2030, value=2024)
    
    month_year = f"{month}_{year}"
    
    st.divider()
    
    # File upload section
    st.markdown("### 📂 Upload Your CSV Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎫 Ticket Details")
        st.markdown("**Expected columns:**")
        st.markdown("""
        - `Ticket_ID` or `ticket_id` *(Required)*
        - `Ticket subject` or `ticket_sub_name`  
        - `Full Resolution Time` or `resolution_time`
        - `SDK` (SDK name)
        - `SDK Issue Types` (issue categories)
        - `Customer Tier ` (customer level)
        - `Ticket organization name`
        """)
        
        details_file = st.file_uploader(
            "Upload ticket details CSV file",
            type=['csv'],
            key="details_file",
            help="CSV file containing ticket ID, subject, resolution time, etc."
        )
        
        if details_file:
            try:
                # Show file info
                file_size = details_file.size
                st.info(f"📄 File size: {file_size:,} bytes")
                
                details_df = pd.read_csv(details_file)
                st.success(f"✅ Loaded {len(details_df)} ticket records")
                
                # Show data quality info
                show_data_preview(details_df, "Details")
                
            except Exception as e:
                st.error(f"❌ Error reading details file: {e}")
                st.info("💡 Make sure your file is properly formatted CSV")
        
    with col2:
        st.subheader("💬 Comments/Conversations") 
        st.markdown("**Expected columns:**")
        st.markdown("""
        - `Ticket_ID` or `ticket_id` *(Required - must match details)*
        - `Comments` or `ticket_conversation`
        """)
        
        comments_file = st.file_uploader(
            "Upload comments CSV file",
            type=['csv'],
            key="comments_file",
            help="CSV file containing ticket conversations and comments"
        )
        
        if comments_file:
            try:
                # Show file info
                file_size = comments_file.size
                st.info(f"📄 File size: {file_size:,} bytes")
                
                comments_df = pd.read_csv(comments_file)
                st.success(f"✅ Loaded {len(comments_df)} conversation records")
                
                # Show data quality info
                show_data_preview(comments_df, "Comments")
                
            except Exception as e:
                st.error(f"❌ Error reading comments file: {e}")
                st.info("💡 Make sure your file is properly formatted CSV")
    
    # Processing section
    if details_file and comments_file:
        st.divider()
        
        # Pre-processing validation
        st.markdown("### 🔍 Pre-Processing Validation")
        
        # Check for common ID column
        details_columns = set(details_df.columns)
        comments_columns = set(comments_df.columns)
        
        possible_id_columns = ['Ticket_ID', 'ticket_id', 'Ticket ID', 'TicketID', 'ID', 'id']
        common_id_columns = []
        
        for col in possible_id_columns:
            if col in details_columns and col in comments_columns:
                common_id_columns.append(col)
        
        if common_id_columns:
            st.success(f"✅ Found common ID column(s): {common_id_columns}")
            
            # Show merge preview
            id_col = common_id_columns[0]
            details_ids = set(details_df[id_col].astype(str))
            comments_ids = set(comments_df[id_col].astype(str))
            
            matching_ids = details_ids.intersection(comments_ids)
            
            st.info(f"📊 Merge Preview: {len(matching_ids)} tickets will have comments out of {len(details_df)} total tickets")
            
            if len(matching_ids) < len(details_df) * 0.5:  # Less than 50% match
                st.warning(f"⚠️ Low match rate: Only {len(matching_ids)/len(details_df)*100:.1f}% of tickets will have comments")
        else:
            st.error("❌ No common ID column found between files!")
            st.markdown("**Available columns:**")
            st.write(f"Details: {list(details_columns)}")
            st.write(f"Comments: {list(comments_columns)}")
            return
        
        # Configuration options
        with st.expander("⚙️ Processing Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                batch_size = st.slider("Batch Size:", 50, 200, 100, step=10,
                                     help="Number of tickets to process at once")
                max_tickets = st.number_input(
                    "Max tickets to process (0 = all):", 
                    min_value=0, 
                    value=0,
                    help="Limit processing for testing"
                )
            
            with col2:
                # Check if month already processed
                if st.session_state.get('db_initialized'):
                    existing_data = st.session_state.db.get_processing_status(month_year)
                    if existing_data:
                        st.warning(f"⚠️ Data for {month_year} already exists!")
                        
                        # Get existing count
                        existing_df = st.session_state.db.get_analysis_data(month_year)
                        st.info(f"📊 Existing records: {len(existing_df)} tickets")
                        
                        if st.checkbox("Clear existing data and reprocess"):
                            if st.button("🗑️ Clear Existing Data", type="secondary"):
                                st.session_state.db.clear_month_data(month_year)
                                st.success("✅ Existing data cleared")
                                st.rerun()
        
        # Process button
        st.markdown("### 🚀 Start Processing")
        
        if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
            process_tickets(details_df, comments_df, month_year, batch_size, max_tickets)

def show_system_status():
    """Show current system status"""
    
    with st.expander("🔧 System Status"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Database status
            db_status = st.session_state.get('db_initialized', False)
            if db_status:
                try:
                    total_tickets = st.session_state.db.get_total_tickets()
                    st.success(f"✅ Database: {total_tickets:,} tickets")
                except:
                    st.warning("⚠️ Database: Connected but query failed")
            else:
                st.error("❌ Database: Not connected")
        
        with col2:
            # API status
            api_status = st.session_state.get('api_configured', False)
            analyzer_status = st.session_state.get('analyzer_initialized', False)
            
            if api_status and analyzer_status:
                st.success("✅ AI: Ready")
            elif api_status:
                st.warning("⚠️ AI: Key configured, analyzer pending")
            else:
                st.error("❌ AI: Not configured")
        
        with col3:
            # Storage status
            try:
                import os
                if os.path.exists('ticket_analysis.db'):
                    size = os.path.getsize('ticket_analysis.db')
                    if size < 1024:
                        size_str = f"{size} bytes"
                    elif size < 1024*1024:
                        size_str = f"{size/1024:.1f} KB"
                    else:
                        size_str = f"{size/(1024*1024):.1f} MB"
                    
                    st.success(f"✅ Storage: {size_str}")
                else:
                    st.info("📝 Storage: New database")
            except:
                st.warning("⚠️ Storage: Unknown")

def show_data_preview(df, file_type):
    """Show data preview and quality info"""
    
    with st.expander(f"🔍 {file_type} Data Preview"):
        # Basic info
        st.markdown(f"**📊 {file_type} Statistics:**")
        st.write(f"• Rows: {len(df):,}")
        st.write(f"• Columns: {len(df.columns)}")
        
        # Column info
        st.markdown("**📋 Columns:**")
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            null_percentage = (len(df) - non_null_count) / len(df) * 100
            
            if null_percentage > 50:
                status = "🔴"
            elif null_percentage > 20:
                status = "🟡"
            else:
                status = "🟢"
                
            st.write(f"{status} `{col}`: {non_null_count:,}/{len(df):,} ({100-null_percentage:.1f}% complete)")
        
        # Sample data
        st.markdown("**📄 Sample Data:**")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Data quality issues
        issues = []
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            issues.append(f"Empty columns: {empty_cols}")
        
        # Check for high null percentage
        high_null_cols = []
        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df) * 100
            if null_pct > 80:
                high_null_cols.append(f"{col} ({null_pct:.1f}%)")
        
        if high_null_cols:
            issues.append(f"High null columns: {high_null_cols}")
        
        # Check for duplicates if ID column exists
        if file_type == "Details":
            id_cols = ['Ticket_ID', 'ticket_id', 'Ticket ID', 'TicketID']
            for col in id_cols:
                if col in df.columns:
                    dupes = df[col].duplicated().sum()
                    if dupes > 0:
                        issues.append(f"Duplicate IDs in {col}: {dupes}")
                    break
        
        if issues:
            st.warning("⚠️ **Data Quality Issues:**")
            for issue in issues:
                st.write(f"• {issue}")
        else:
            st.success("✅ **No major data quality issues found**")

def process_tickets(details_df, comments_df, month_year, batch_size, max_tickets):
    """Main ticket processing function with single-ticket Gemini calls and immediate DB save."""
    st.markdown("---")
    st.markdown("## 🔄 Processing Pipeline")
    # Step 1: Merge data
    st.markdown("### Step 1: Data Merging")
    try:
        with st.spinner("🔗 Merging ticket data..."):
            merged_df = st.session_state.processor.merge_ticket_data(details_df, comments_df)
            if max_tickets > 0:
                original_count = len(merged_df)
                merged_df = merged_df.head(max_tickets)
                st.info(f"📊 Limited processing to {len(merged_df)} tickets (from {original_count})")
    except Exception as e:
        st.error(f"❌ Data merging failed: {e}")
        st.stop()
    st.markdown("### Step 2: AI Analysis (One Ticket at a Time)")
    progress_bar = st.progress(0, text="Starting processing...")
    log_container = st.container()
    total_tickets = len(merged_df)
    processed = 0
    failed = 0
    processing_times = []
    for idx, row in merged_df.iterrows():
        ticket_row = row.to_dict()
        start_time = time.time()
        try:
            result = st.session_state.analyzer.analyze_ticket(ticket_row)
            save_success = st.session_state.db.save_single_analysis(month_year, result)
            if save_success:
                status = f"✅ Ticket {ticket_row.get('ticket_id', ticket_row.get('Ticket_ID', idx))} saved"
            else:
                status = f"❌ Ticket {ticket_row.get('ticket_id', ticket_row.get('Ticket_ID', idx))} DB save failed"
        except Exception as e:
            failed += 1
            status = f"❌ Ticket {ticket_row.get('ticket_id', ticket_row.get('Ticket_ID', idx))} failed: {e}"
        processing_times.append(time.time() - start_time)
        processed += 1
        progress_bar.progress(processed / total_tickets, text=f"Processed {processed}/{total_tickets} tickets")
        with log_container:
            st.text(status)
        time.sleep(1)  # brief pause to avoid rate limiting
    st.success(f"🎉 Processing completed! {processed - failed}/{total_tickets} tickets processed successfully.")

if __name__ == "__main__":
    main()
