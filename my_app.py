import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.analyzer import GeminiTicketAnalyzer
from utils.database import TicketDB
from utils.data_processor import DataProcessor
import json
import os

# Page config
st.set_page_config(
    page_title="Ticket Analytics Dashboard", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎫 Support Ticket Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🚀 Quick Actions")
        
        # API Key input (for first-time setup)
        if not st.session_state.get('api_configured', False):
            with st.expander("⚙️ API Configuration"):
                api_key_input = st.text_input(
                    "Enter Gemini API Key:", 
                    type="password",
                    help="Get your API key from Google AI Studio"
                )
                if st.button("Save API Key") and api_key_input:
                    st.session_state.gemini_api_key = api_key_input
                    st.session_state.api_configured = True
                    st.success("API Key saved!")
                    st.rerun()
        
        # Status indicators
        st.info("📊 **Dashboard Status**")
        db_status = "✅ Connected" if st.session_state.get('db_initialized') else "❌ Not Connected"
        st.write(f"Database: {db_status}")
        
        api_status = "✅ Configured" if st.session_state.get('api_configured') else "❌ Not Configured"
        st.write(f"Gemini API: {api_status}")
        
        # Quick stats
        if st.session_state.get('db_initialized'):
            try:
                total_tickets = st.session_state.db.get_total_tickets()
                st.metric("Total Analyzed Tickets", total_tickets)
            except:
                st.metric("Total Analyzed Tickets", 0)

def initialize_session_state():
    """Initialize session state variables"""
    
    # Database
    if 'db_initialized' not in st.session_state:
        try:
            st.session_state.db = TicketDB()
            st.session_state.db_initialized = True
        except Exception as e:
            st.error(f"Database initialization failed: {e}")
            st.session_state.db_initialized = False
    
    # API Configuration
    if 'api_configured' not in st.session_state:
        # Check if API key is in Streamlit secrets
        try:
            gemini_key = st.secrets["GEMINI_API_KEY"]
            st.session_state.gemini_api_key = gemini_key
            st.session_state.api_configured = True
        except:
            st.session_state.api_configured = False
    
    # Analyzer
    if 'analyzer_initialized' not in st.session_state and st.session_state.get('api_configured'):
        try:
            st.session_state.analyzer = GeminiTicketAnalyzer(
                api_key=st.session_state.gemini_api_key
            )
            st.session_state.analyzer_initialized = True
        except Exception as e:
            st.error(f"Analyzer initialization failed: {e}")
            st.session_state.analyzer_initialized = False
    
    # Data processor
    if 'processor' not in st.session_state:
        st.session_state.processor = DataProcessor()

if __name__ == "__main__":
    main()
