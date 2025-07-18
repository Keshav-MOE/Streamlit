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
    """Main ticket processing function with comprehensive debugging"""
    
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
    
    # Step 2: Setup processing
    st.markdown("### Step 2: AI Processing Setup")
    
    total_batches = (len(merged_df) + batch_size - 1) // batch_size
    st.info(f"📦 Will process {len(merged_df)} tickets in {total_batches} batches of {batch_size}")
    
    # Processing containers
    progress_container = st.container()
    status_container = st.empty()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0, text="Starting processing...")
        processing_metrics = st.empty()
    
    # Step 3: Process batches
    st.markdown("### Step 3: AI Analysis")
    
    successful_batches = 0
    total_processed = 0
    failed_batches = []
    processing_times = []
    
    # Create expandable log
    with st.expander("📋 Processing Log", expanded=True):
        log_container = st.container()
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(merged_df))
        batch_df = merged_df.iloc[start_idx:end_idx]
        
        # Update progress
        progress_bar.progress(
            batch_num / total_batches, 
            text=f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_df)} tickets)"
        )
        
        # Log current batch
        with log_container:
            st.text(f"🔄 Batch {batch_num + 1}/{total_batches}: Processing {len(batch_df)} tickets...")
        
        batch_start_time = time.time()
        
        try:
            # Analyze batch using Gemini
            analysis_results = st.session_state.analyzer.analyze_batch(batch_df)
            
            if not analysis_results:
                raise ValueError("No analysis results returned")
            
            # Save to database with verification
            save_success, num_saved = st.session_state.db.save_batch_analysis(
                month_year, 
                batch_num + 1, # Batch numbers are 1-based
                analysis_results
            )
            
            if save_success and num_saved > 0:
                batch_time = time.time() - batch_start_time
                processing_times.append(batch_time)
                
                successful_batches += 1
                total_processed += len(batch_df)
                
                # Update metrics
                avg_time = sum(processing_times) / len(processing_times)
                estimated_remaining = (total_batches - batch_num - 1) * avg_time
                
                with processing_metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Processed", f"{total_processed}")
                    with col2:
                        st.metric("Success Rate", f"{successful_batches/max(1, batch_num + 1)*100:.1f}%")
                    with col3:
                        st.metric("Avg Time/Batch", f"{avg_time:.1f}s")
                    with col4:
                        st.metric("Est. Remaining", f"{estimated_remaining/60:.1f}m")
                
                with log_container:
                    st.text(f"✅ Batch {batch_num + 1}: Success in {batch_time:.1f}s")
            else:
                raise ValueError(f"Database save failed. Success: {save_success}, Saved: {num_saved}")
            
        except Exception as e:
            batch_time = time.time() - batch_start_time
            failed_batches.append({
                'batch': batch_num + 1,
                'error': str(e),
                'time': batch_time
            })
            
            with log_container:
                st.text(f"❌ Batch {batch_num + 1}: Failed - {str(e)[:100]}")
            
            # Continue processing other batches
            continue
        
        # Update progress
        progress_bar.progress((batch_num + 1) / total_batches)
        
        # Brief pause to avoid rate limiting
        time.sleep(1)
    
    # Step 4: Final Results
    st.markdown("### Step 4: Processing Results")
    
    progress_bar.progress(1.0, text="Processing completed!")
    
    if successful_batches > 0:
        # Success summary
        success_rate = successful_batches / total_batches * 100
        total_time = sum(processing_times)
        
        status_container.success(
            f"🎉 Processing completed! {total_processed}/{len(merged_df)} tickets processed "
            f"({successful_batches}/{total_batches} batches, {success_rate:.1f}% success rate)"
        )
        
        # Enhanced debug and verification section
        st.markdown("### 🔍 Upload Verification & Debug")
        
        # Force refresh database connection to get latest data
        st.info("🔄 Refreshing database connection...")
        
        try:
            st.session_state.db.engine.dispose()
            st.session_state.db.engine = create_engine('sqlite:///ticket_analysis.db')
            
            # Get comprehensive debug info
            debug_info = st.session_state.db.debug_database_contents()
            verification = st.session_state.db.force_verify_data()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📊 Database Status:**")
                for info in debug_info:
                    if "❌" in info:
                        st.error(info)
                    elif "⚠️" in info:
                        st.warning(info)
                    else:
                        st.success(info)
            
            with col2:
                st.markdown("**✅ Data Verification:**")
                if 'error' not in verification:
                    st.success(f"✅ Total tickets: {verification['ticket_count']:,}")
                    st.success(f"✅ Summary records: {verification['summary_count']}")
                    st.success(f"✅ Months available: {len(verification['months'])}")
                    
                    if verification['months']:
                        st.info(f"📅 Latest month: {verification['months'][0]}")
                    
                    st.text(f"🕒 Verified: {verification['verified_at']}")
                else:
                    st.error(f"❌ Verification failed: {verification['error']}")
            
            with col3:
                st.markdown("**🧪 Live Data Tests:**")
                try:
                    # Test live data retrieval
                    fresh_total = st.session_state.db.get_total_tickets()
                    fresh_stats = st.session_state.db.get_quick_stats()
                    month_data = st.session_state.db.get_analysis_data(month_year)
                    
                    st.success(f"✅ Live total: {fresh_total:,}")
                    st.success(f"✅ Month count: {fresh_stats.get('total_months', 0)}")
                    st.success(f"✅ This month: {len(month_data)} tickets")
                    
                    # Update session state to ensure home page reflects changes
                    if 'db_last_refresh' in st.session_state:
                        del st.session_state['db_last_refresh']
                    
                    st.session_state.db_last_refresh = time.time()
                    
                except Exception as e:
                    st.error(f"❌ Live test failed: {e}")
        
        except Exception as e:
            st.error(f"❌ Verification process failed: {e}")
        
        # Generate and save monthly summary
        try:
            with st.spinner("📊 Generating monthly summary..."):
                summary = st.session_state.db.generate_monthly_summary(month_year)
                if summary and summary.get('total_tickets', 0) > 0: # Check if summary is valid
                    st.session_state.db.save_monthly_summary(summary)
                    
                    # Show summary preview
                    with results_container:
                        st.subheader("📋 Month Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Tickets Processed", f"{summary['total_tickets']:,}")
                        with col2:
                            st.metric("Avg Priority", f"{summary['avg_priority']:.1f}/10")
                        
                        # Show top issues and improvements
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if summary.get('top_sdk_issues'):
                                st.markdown("**🔥 Top SDK Issues:**")
                                top_issues = json.loads(summary['top_sdk_issues'])
                                for issue, count in list(top_issues.items())[:5]:
                                    st.write(f"• {issue}: {count} tickets")
                            else:
                                st.info("No SDK issues data available")
                        
                        with col2:
                            if summary.get('improvement_priorities'):
                                st.markdown("**💡 Top Improvements:**")
                                top_improvements = json.loads(summary['improvement_priorities'])
                                for improvement, count in list(top_improvements.items())[:5]:
                                    st.write(f"• {improvement}: {count} mentions")
                            else:
                                st.info("No improvement data available")
                    
                    st.balloons()
                else:
                    st.warning("⚠️ Could not generate summary - no valid data found")
        
        except Exception as e:
            st.warning(f"Summary generation had issues: {e}")
            st.info("Don't worry - your ticket analysis data was saved successfully!")
        
        # Show failed batches if any
        if failed_batches:
            st.markdown("### ⚠️ Failed Batches")
            st.warning(f"The following {len(failed_batches)} batches failed:")
            
            for failure in failed_batches:
                with st.expander(f"❌ Batch {failure['batch']} - {failure['error'][:50]}..."):
                    st.write(f"**Error:** {failure['error']}")
                    st.write(f"**Time taken:** {failure['time']:.1f}s")
        
        # Processing statistics
        st.markdown("### 📈 Processing Statistics")
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.metric("Total Time", f"{total_time:.1f}s")
            st.metric("Avg per Ticket", f"{total_time/max(1, total_processed):.2f}s")
        
        with stats_col2:
            st.metric("Success Rate", f"{success_rate:.1f}%")
            st.metric("Tickets/Min", f"{total_processed/(total_time/60):.1f}")
        
        with stats_col3:
            st.metric("Failed Batches", len(failed_batches))
            st.metric("Data Quality", "🟢 High" if success_rate > 80 else "🟡 Medium")
        
        # Navigation
        st.markdown("### 🎯 What's Next?")
        st.success("✅ Your ticket data has been successfully analyzed and is ready for insights!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🏠 Back to Home", use_container_width=True):
                st.switch_page("streamlit_app.py")
        
        with col2:
            if st.button("📊 View Dashboard", use_container_width=True):
                st.switch_page("pages/2_📊_Dashboard.py")
        
        with col3:
            if st.button("🛠️ SDK Insights", use_container_width=True):
                st.switch_page("pages/3_🛠️_SDK_Insights.py")
        
        with col4:
            if st.button("📈 View Trends", use_container_width=True):
                st.switch_page("pages/4_📈_Trends.py")
    
    else:
        # Complete failure
        status_container.error("❌ Processing failed completely!")
        
        st.markdown("### 🔧 Troubleshooting")
        st.error("**No batches were processed successfully. Here's what to check:**")
        
        st.markdown("""
        1. **API Key Issues:**
           - Verify your Gemini API key is correct
           - Check if you've hit rate limits
           - Try with a smaller batch size
        
        2. **Data Issues:**
           - Ensure CSV files have the required columns
           - Check for data formatting problems
           - Try with a smaller dataset first
        
        3. **Technical Issues:**
           - Check your internet connection
           - Try refreshing the page and retry
           - Contact support if issues persist
        """)
        
        # Show detailed error log
        if failed_batches:
            st.markdown("### 📋 Detailed Error Log")
            for failure in failed_batches:
                st.error(f"Batch {failure['batch']}: {failure['error']}")

if __name__ == "__main__":
    main()
