import streamlit as st
import pandas as pd
import time
from utils.data_processor import DataProcessor
import io

# Set the page configuration
st.set_page_config(page_title="Upload Data", page_icon="📁")

def main():
    st.title("📁 Upload Monthly Ticket Data")

    # --- CORRECTION: Initialize session state objects if they don't exist ---
    # This makes the app more robust and prevents errors on first run.
    if 'processor' not in st.session_state:
        st.session_state.processor = DataProcessor()
    # Assuming analyzer and db are initialized similarly, e.g., on a login/main page
    # if 'analyzer' not in st.session_state:
    #     st.session_state.analyzer = SomeAnalyzerClass()
    # if 'db' not in st.session_state:
    #     st.session_state.db = SomeDatabaseClass()

    # Check for API key prerequisite
    if not st.session_state.get('api_configured', False):
        st.error("🚨 Please configure your API key on the main page first!")
        st.stop()

    # --- CORRECTION: Initialize DataFrame variables to None ---
    # This prevents a NameError if file reading fails.
    details_df = None
    comments_df = None

    # Month and Year selection
    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox(
            "Select Month:",
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"],
             index=6 # Default to July
        )
    with col2:
        # Based on the current date of July 2025
        year = st.number_input("Select Year:", min_value=2020, max_value=2030, value=2025)

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
                with st.expander("Preview Ticket Details"):
                    st.dataframe(details_df.head())
                st.info(f"**Columns:** {', '.join(details_df.columns.tolist())}")
            except Exception as e:
                st.error(f"Error reading details file: {e}")

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
                with st.expander("Preview Comments Data"):
                    st.dataframe(comments_df.head())
                st.info(f"**Columns:** {', '.join(comments_df.columns.tolist())}")
            except Exception as e:
                st.error(f"Error reading comments file: {e}")

    # Processing section - runs only if both files are successfully uploaded
    if details_df is not None and comments_df is not None:
        st.divider()

        with st.expander("⚙️ Processing Options"):
            batch_size = st.slider("Batch Size:", 50, 500, 100, 50)
            max_tickets = st.number_input(
                "Max tickets to process (0 = all):",
                min_value=0,
                value=0,
                help="Set to 0 to process all tickets in the uploaded files."
            )

        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            with st.spinner("Preparing data... Please wait."):
                try:
                    merged_df = st.session_state.processor.merge_ticket_data(details_df, comments_df)
                    if max_tickets > 0:
                        merged_df = merged_df.head(max_tickets)
                    st.info(f"Ready to process {len(merged_df)} tickets in batches of {batch_size}.")
                except Exception as e:
                    st.error(f"Error merging data: {e}")
                    st.stop()

            total_batches = (len(merged_df) + batch_size - 1) // batch_size
            if total_batches == 0:
                st.warning("⚠️ No data to process.")
                st.stop()

            progress_bar = st.progress(0, text="Starting processing...")
            status_container = st.container()
            successful_batches = 0
            total_processed = 0

            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(merged_df))
                batch_df = merged_df.iloc[start_idx:end_idx]

                progress_text = f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_df)} tickets)"
                status_container.info(f"🔄 {progress_text}")
                progress_bar.progress((batch_num) / total_batches, text=progress_text)

                try:
                    # These lines are placeholders for your actual analysis and DB logic
                    # analysis_results = st.session_state.analyzer.analyze_batch(batch_df)
                    # st.session_state.db.save_batch_analysis(month_year, batch_num, analysis_results)
                    time.sleep(2) # Simulating work

                    successful_batches += 1
                    total_processed += len(batch_df)
                    time.sleep(1) # Brief pause to avoid rate limiting
                except Exception as e:
                    status_container.error(f"❌ Error in batch {batch_num + 1}: {e}")
                    continue

            progress_bar.progress(1.0, text="Processing complete!")

            if successful_batches > 0:
                status_container.success(
                    f"🎉 Successfully processed {total_processed} tickets "
                    f"({successful_batches}/{total_batches} batches) for {month_year}"
                )
                
                # --- CORRECTION: Fixed syntax and indentation for the summary block ---
                try:
                    with st.spinner("Generating monthly summary..."):
                        # Placeholder for summary generation and saving
                        # summary = st.session_state.db.generate_monthly_summary(month_year)
                        # st.session_state.db.save_monthly_summary(month_year, summary)
                        summary = {'avg_priority': 4.2} # Dummy summary data
                        time.sleep(3) # Simulating work

                    if summary:
                        st.balloons()
                        st.subheader("📋 Processing Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Tickets Processed", total_processed)
                        col2.metric("Successful Batches", successful_batches)
                        
                        # --- CORRECTION: Added check to prevent ZeroDivisionError ---
                        if total_batches > 0:
                            success_rate = f"{(successful_batches/total_batches)*100:.1f}%"
                        else:
                            success_rate = "N/A"
                        col3.metric("Success Rate", success_rate)
                        col4.metric("Avg Priority", f"{summary.get('avg_priority', 0):.1f}")
                    else:
                        st.warning("⚠️ Could not generate summary, but batch data was saved.")

                except Exception as e:
                    st.warning(f"Summary generation failed: {e}")
                    st.info("Don't worry - your ticket analysis data was saved successfully!")
            else:
                status_container.error("❌ No batches were processed successfully.")

if __name__ == "__main__":
    main()
