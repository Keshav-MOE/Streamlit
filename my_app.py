import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.analyzer import GeminiTicketAnalyzer
from utils.database import TicketDB
from utils.data_processor import DataProcessor
import json
import os

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Ticket Analytics Dashboard", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
    }
    .status-indicator {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎫 Support Ticket Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Main content area
    show_home_page()
    
    # Sidebar
    show_sidebar()

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Database initialization
    if 'db_initialized' not in st.session_state:
        try:
            st.session_state.db = TicketDB()
            st.session_state.db_initialized = True
        except Exception as e:
            st.session_state.db_initialized = False
            st.session_state.db_error = str(e)
    
    # API Configuration
    if 'api_configured' not in st.session_state:
        # Try to get API key from Streamlit secrets first
        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY", "")
            if gemini_key:
                st.session_state.gemini_api_key = gemini_key
                st.session_state.api_configured = True
            else:
                st.session_state.api_configured = False
        except Exception:
            st.session_state.api_configured = False
    
    # Analyzer initialization
    if ('analyzer_initialized' not in st.session_state and 
        st.session_state.get('api_configured', False)):
        try:
            st.session_state.analyzer = GeminiTicketAnalyzer(
                api_key=st.session_state.get('gemini_api_key')
            )
            st.session_state.analyzer_initialized = True
        except Exception as e:
            st.session_state.analyzer_initialized = False
            st.session_state.analyzer_error = str(e)
    
    # Data processor
    if 'processor' not in st.session_state:
        st.session_state.processor = DataProcessor()

def show_home_page():
    """Show the main home page content"""
    
    # Welcome message
    st.markdown("""
    ### Welcome to the Support Ticket Analytics Platform! 🚀
    
    This platform helps you analyze support tickets using AI to identify SDK improvement opportunities,
    track trends, and generate actionable insights for better customer support.
    
    **How to get started:**
    1. 📁 Upload your monthly ticket data (details + conversations)
    2. 🔄 Let AI analyze the tickets in batches
    3. 📊 Explore insights in the dashboard
    4. 📈 Track trends across months
    """)
    
    # Quick Status Overview
    st.markdown("### 🔧 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        db_status = "✅ Connected" if st.session_state.get('db_initialized') else "❌ Not Connected"
        db_class = "status-success" if st.session_state.get('db_initialized') else "status-error"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Database</h4>
            <span class="status-indicator {db_class}">{db_status}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        api_status = "✅ Configured" if st.session_state.get('api_configured') else "❌ Not Configured"
        api_class = "status-success" if st.session_state.get('api_configured') else "status-error"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Gemini AI</h4>
            <span class="status-indicator {api_class}">{api_status}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.session_state.get('db_initialized'):
            try:
                total_tickets = st.session_state.db.get_total_tickets()
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Total Tickets Analyzed</h4>
                    <h2 style="color: #1f77b4; margin: 0;">{total_tickets:,}</h2>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Total Tickets Analyzed</h4>
                    <h2 style="color: #1f77b4; margin: 0;">0</h2>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Tickets Analyzed</h4>
                <h2 style="color: #1f77b4; margin: 0;">0</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📁 Upload New Data", use_container_width=True):
            st.switch_page("pages/1_📁_Upload_Data.py")
    
    with col2:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.switch_page("pages/2_📊_Dashboard.py")
    
    with col3:
        if st.button("🛠️ SDK Insights", use_container_width=True):
            st.switch_page("pages/3_🛠️_SDK_Insights.py")
    
    with col4:
        if st.button("📈 View Trends", use_container_width=True):
            st.switch_page("pages/4_📈_Trends.py")
    
    # Recent Activity
    if st.session_state.get('db_initialized'):
        st.markdown("### 📋 Recent Activity")
        try:
            recent_data = st.session_state.db.get_recent_activity()
            if not recent_data.empty:
                st.dataframe(
                    recent_data.head(10),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No recent activity. Upload some data to get started!")
        except Exception as e:
            st.info("No activity data available yet.")

def show_sidebar():
    """Show sidebar content"""
    
    with st.sidebar:
        st.title("🚀 Control Panel")
        
        # API Key Configuration
        if not st.session_state.get('api_configured', False):
            with st.expander("⚙️ Configure Gemini API", expanded=True):
                st.warning("⚠️ Gemini API key required for AI analysis")
                api_key_input = st.text_input(
                    "Enter Gemini API Key:", 
                    type="password",
                    help="Get your API key from Google AI Studio (https://makersuite.google.com/app/apikey)"
                )
                
                if st.button("💾 Save API Key", use_container_width=True):
                    if api_key_input.strip():
                        st.session_state.gemini_api_key = api_key_input.strip()
                        st.session_state.api_configured = True
                        st.success("✅ API Key saved successfully!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid API key")
        else:
            st.success("✅ API Key configured")
            if st.button("🔄 Reset API Key"):
                st.session_state.api_configured = False
                st.session_state.pop('gemini_api_key', None)
                st.rerun()
        
        st.divider()
        
        # Quick Stats
        st.markdown("### 📊 Quick Stats")
        
        if st.session_state.get('db_initialized'):
            try:
                stats = st.session_state.db.get_quick_stats()
                
                st.metric("Total Months", stats.get('total_months', 0))
                st.metric("This Month's Tickets", stats.get('current_month_tickets', 0))
                st.metric("Avg Priority Score", f"{stats.get('avg_priority', 0):.1f}")
                st.metric("Top Issue Category", stats.get('top_category', 'N/A'))
                
            except Exception as e:
                st.info("Stats will appear after processing tickets")
        
        st.divider()
        
        # Navigation Help
        st.markdown("### 📖 Navigation Guide")
        st.markdown("""
        **📁 Upload Data**: Import monthly CSV files
        
        **📊 Dashboard**: View analytics & charts
        
        **🛠️ SDK Insights**: AI-powered improvement suggestions
        
        **📈 Trends**: Historical analysis & patterns
        """)
        
        st.divider()
        
        # System Info
        with st.expander("ℹ️ System Information"):
            st.text(f"Streamlit version: {st.__version__}")
            st.text(f"Database: SQLite")
            st.text(f"AI Model: Gemini Pro")
            
            if st.session_state.get('db_initialized'):
                try:
                    db_size = st.session_state.db.get_database_size()
                    st.text(f"Database size: {db_size}")
                except:
                    st.text("Database size: Unknown")

if __name__ == "__main__":
    main()
