import streamlit as st
import pandas as pd
import time
from utils.data_processor import DataProcessor
import io

st.set_page_config(page_title="Upload Data", page_icon="📁")

def main():
    st.title("📁 Upload Monthly Ticket Data")
    
    # Check prerequisites
    if not st.session_state.get('api_configured', False):
        st.error("🚨 Please configure Gemini API key in the main page first!")
        return
    
    # Month and Year selection
    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox(
            "Select Month:",
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]
        )
    with col2:
        year = st.number_input("Year:", min_value=2020, max_value=2030, value=2024)
    
    month_year = f"{month}_{year}"
    
    st.divider()
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎫 Ticket Details")
        details_file = st.file_uploader(
            "Upload ticket details CSV file",
            type=['csv'],
            key="details_file",
            help="CSV file containing ticket ID, subject, resolution time, etc."
        )
        
        if details_file:
            try:
                details_df = pd.read_csv(details_file)
                st.success(f"✅ Loaded {len(details_df)} ticket records")
                
                # Show preview
                with st.expander("Preview Data"):
                    st.dataframe(details_df.head(10))
                
                # Show columns
                st.info(f"**Columns found:** {', '.join(details_df.columns.tolist())}")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
    with col2:
        st.subheader("💬 Comments/Conversations") 
        comments_file = st.file_uploader(
            "Upload comments CSV file",
            type=['csv'],
            key="comments_file",
            help="CSV file containing ticket conversations and comments"
        )
        
        if comments_file:
            try:
                comments_df = pd.read_csv(comments_file)
                st.success(f"✅ Loaded {len(comments_df)} conversation records")
                
                # Show preview
                with st.expander("Preview Data"):
                    st.dataframe(comments_df.head(10))
                
                # Show columns
                st.info(f"**Columns found:** {', '.join(comments_df.columns.tolist())}")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Processing section
    if details_file and comments_file:
        st.divider()
        
        # Configuration options
        with st.expander("⚙️ Processing Options"):
            batch_size = st.slider("Batch Size:", 50, 200, 100)
            max_tickets = st.number_input(
                "Max tickets to process (0 = all):", 
                min_value=0, 
                value=0
            )
        
        # Process button
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            
            # Validate files and merge
            with st.spinner("Preparing data..."):
                try:
                    merged_df = st.session_state.processor.merge_ticket_data(
                        details_df, comments_df
                    )
                    
                    if max_tickets > 0:
                        merged_df = merged_df.head(max_tickets)
                    
                    st.info(f"Processing {len(merged_df)} tickets in batches of {batch_size}")
                    
                except Exception as e:
                    st.error(f"Error merging data: {e}")
                    return
            
            # Process in batches
            total_batches = (len(merged_df) + batch_size - 1) // batch_size
            
            progress_bar = st.progress(0)
            status_container = st.empty()
            results_container = st.empty()
            
            successful_batches = 0
            total_processed = 0
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(merged_df))
                batch_df = merged_df.iloc[start_idx:end_idx]
                
                status_container.info(
                    f"🔄 Processing batch {batch_num + 1}/{total_batches} "
                    f"({len(batch_df)} tickets)"
                )
                
                try:
                    # Analyze batch using Gemini
                    analysis_results = st.session_state.analyzer.analyze_batch(batch_df)
                    
                    # Save to database
                    st.session_state.db.save_batch_analysis(
                        month_year, 
                        batch_num,
                        analysis_results
                    )
                    
                    successful_batches += 1
                    total_processed += len(batch_df)
                    
                    # Update progress
                    progress_bar.progress((batch_num + 1) / total_batches)
                    
                    # Brief pause to avoid rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    st.error(f"Error processing batch {batch_num + 1}: {e}")
                    continue
            
            # Final results
            progress_bar.progress(1.0)
            
            if successful_batches > 0:
                status_container.success(
                    f"🎉 Successfully processed {total_processed} tickets "
                    f"({successful_batches}/{total_batches} batches) for {month_year}"
                )
                
                # Generate monthly summary
                try:
                    summary = st.session_state.db.generate_monthly_summary(month_year)
                    st.session_state.db.save_monthly_summary(month_year, summary)
                    
                    st.balloons()
                    
                    # Show quick summary
                    with st.container():
                        st.subheader("📋 Processing Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Tickets Processed", total_processed)
                        with col2:
                            st.metric("Successful Batches", successful_batches)
                        with col3:
                            st.metric("Success Rate", f"{(successful_batches/total_batches)*100:.1f}%")
                        with col4:
                            st.metric("Avg Priority Score", f"{summary.get('avg_priority', 0):.1f}")
                    
                except Exception as e:
                    st.warning(f"Summary generation failed: {e}")
            
            else:
                status_container.error("❌ No batches were processed successfully")

if __name__ == "__main__":
    main()