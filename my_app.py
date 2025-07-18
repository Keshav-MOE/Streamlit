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
    .api-key-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .danger-zone {
        background-color: #fff5f5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #e53e3e;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fffdf0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f6ad55;
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
            if gemini_key and gemini_key.strip():
                st.session_state.gemini_api_key = gemini_key.strip()
                st.session_state.api_configured = True
                test_api_key_validity()
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

def test_api_key_validity():
    """Test if the current API key is valid"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.session_state.gemini_api_key)
        # Simple test - just configure, don't call API yet
        return True
    except Exception:
        st.session_state.api_configured = False
        return False

def show_home_page():
    """Show the main home page content"""
    
    # Welcome message
    st.markdown("""
    ### Welcome to the Support Ticket Analytics Platform! 🚀
    
    This platform helps you analyze support tickets using AI to identify SDK improvement opportunities,
    track trends, and generate actionable insights for better customer support.
    
    **How to get started:**
    1. 🔑 Configure your Gemini API key in the sidebar
    2. 📁 Upload your monthly ticket data (details + conversations)
    3. 🔄 Let AI analyze the tickets in batches  
    4. 📊 Explore insights in the dashboard
    5. 📈 Track trends across months
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
    
    # Show API configuration notice if not configured
    if not st.session_state.get('api_configured', False):
        st.markdown("""
        <div class="api-key-box">
            <h4>🔑 API Key Required</h4>
            <p>To analyze tickets with AI, you need to configure your Gemini API key in the sidebar.</p>
            <p><strong>Get your free API key:</strong> <a href="https://makersuite.google.com/app/apikey" target="_blank">Google AI Studio</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent Activity
    if st.session_state.get('db_initialized'):
        st.markdown("### 📋 Recent Activity")
        try:
            recent_data = st.session_state.db.get_recent_activity()
            if not recent_data.empty:
                st.dataframe(
                    recent_data.head(5),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No recent activity. Upload some data to get started!")
        except Exception as e:
            st.info("Upload and analyze some tickets to see recent activity here.")

def show_sidebar():
    """Show sidebar content"""
    
    with st.sidebar:
        st.title("🚀 Control Panel")
        
        # API Key Configuration
        if not st.session_state.get('api_configured', False):
            with st.expander("🔑 Configure Gemini API", expanded=True):
                st.warning("⚠️ Gemini API key required for AI analysis")
                
                st.markdown("""
                **Get Your Free API Key:**
                1. Visit: https://makersuite.google.com/app/apikey
                2. Sign in with Google account
                3. Click "Create API Key" 
                4. Copy the key (starts with 'AIza...')
                5. Paste it below:
                """)
                
                api_key_input = st.text_input(
                    "Enter Gemini API Key:", 
                    type="password",
                    placeholder="AIzaSy...",
                    help="Your API key should start with 'AIza'"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save Key", use_container_width=True):
                        if api_key_input.strip():
                            if api_key_input.startswith('AIza'):
                                try:
                                    # Test the API key immediately
                                    import google.generativeai as genai
                                    genai.configure(api_key=api_key_input.strip())
                                    
                                    # Try to create analyzer
                                    test_analyzer = GeminiTicketAnalyzer(api_key=api_key_input.strip())
                                    
                                    # If we get here, the API key works
                                    st.session_state.gemini_api_key = api_key_input.strip()
                                    st.session_state.api_configured = True
                                    st.session_state.analyzer = test_analyzer
                                    st.session_state.analyzer_initialized = True
                                    
                                    st.success("✅ API Key saved and validated!")
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ API Key validation failed: {str(e)}")
                            else:
                                st.error("❌ Invalid format. API key should start with 'AIza'")
                        else:
                            st.error("❌ Please enter an API key")
                
                with col2:
                    if st.button("🧪 Test Key", use_container_width=True):
                        if api_key_input.strip():
                            try:
                                import google.generativeai as genai
                                genai.configure(api_key=api_key_input.strip())
                                models = list(genai.list_models())
                                st.success(f"✅ Valid! Found {len(models)} models")
                            except Exception as e:
                                st.error(f"❌ Invalid key: {str(e)}")
                        else:
                            st.error("❌ Enter key to test")
        else:
            st.success("✅ API Key Configured")
            
            # Show masked API key
            if hasattr(st.session_state, 'gemini_api_key'):
                masked_key = st.session_state.gemini_api_key[:12] + "..." if len(st.session_state.gemini_api_key) > 12 else "***"
                st.code(f"Key: {masked_key}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reset Key", use_container_width=True):
                    st.session_state.api_configured = False
                    st.session_state.analyzer_initialized = False
                    for key in ['gemini_api_key', 'analyzer']:
                        st.session_state.pop(key, None)
                    st.rerun()
            
            with col2:
                if st.button("🧪 Test API", use_container_width=True):
                    try:
                        if st.session_state.get('analyzer_initialized'):
                            st.success("✅ API is working!")
                        else:
                            st.error("❌ Analyzer not initialized")
                    except Exception as e:
                        st.error(f"❌ API Error: {e}")
        
        st.divider()
        
        # Quick Stats
        st.markdown("### 📊 Quick Stats")
        
        if st.session_state.get('db_initialized'):
            try:
                stats = st.session_state.db.get_quick_stats()
                
                st.metric("Total Months", stats.get('total_months', 0))
                st.metric("Current Month Tickets", stats.get('current_month_tickets', 0))
                st.metric("Avg Priority Score", f"{stats.get('avg_priority', 0):.1f}")
                
                top_cat = stats.get('top_category', 'N/A')
                if len(top_cat) > 15:
                    top_cat = top_cat[:15] + "..."
                st.metric("Top Category", top_cat)
                
            except Exception as e:
                st.info("Stats will appear after processing tickets")
        else:
            st.info("Database not initialized")
        
        st.divider()
        
        # Database Management Section
        st.markdown("### 🗄️ Database Management")
        
        if st.session_state.get('db_initialized'):
            try:
                db_size = st.session_state.db.get_database_size()
                st.metric("Database Size", db_size)
                
                total_records = st.session_state.db.get_total_tickets()
                st.metric("Total Records", total_records)
                
            except Exception as e:
                st.info("Database info unavailable")
        
        # Clear Database Section - DANGER ZONE
        with st.expander("⚠️ Danger Zone", expanded=False):
            st.markdown("""
            <div class="danger-zone">
                <h4>🚨 Clear All Data</h4>
                <p><strong>Warning:</strong> This will permanently delete all analyzed tickets, summaries, and insights from the database.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Two-step confirmation for safety
            if 'clear_db_confirm_step' not in st.session_state:
                st.session_state.clear_db_confirm_step = 0
            
            if st.session_state.clear_db_confirm_step == 0:
                if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
                    st.session_state.clear_db_confirm_step = 1
                    st.rerun()
            
            elif st.session_state.clear_db_confirm_step == 1:
                st.markdown("""
                <div class="warning-box">
                    <p><strong>⚠️ Are you absolutely sure?</strong></p>
                    <p>This action cannot be undone. All your analyzed data will be lost.</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state.clear_db_confirm_step = 0
                        st.rerun()
                
                with col2:
                    if st.button("💥 YES, DELETE ALL", type="primary", use_container_width=True):
                        clear_database()
        
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
            st.text(f"Streamlit: {st.__version__}")
            st.text(f"Database: SQLite")
            st.text(f"AI Model: Gemini 1.5 Flash")
            
            if st.session_state.get('db_initialized'):
                try:
                    db_size = st.session_state.db.get_database_size()
                    st.text(f"DB Size: {db_size}")
                except:
                    st.text("DB Size: Unknown")

def clear_database():
    """Clear all data from the database"""
    
    try:
        with st.spinner("🗑️ Clearing database..."):
            # Get database instance
            db = st.session_state.db
            
            # Clear all tables
            with db.engine.connect() as conn:
                # Delete all ticket analysis data
                conn.execute(db.engine.execute("DELETE FROM ticket_analysis"))
                
                # Delete all monthly summaries
                conn.execute(db.engine.execute("DELETE FROM monthly_summary"))
                
                # Commit the changes
                conn.commit()
            
            st.success("✅ Database cleared successfully!")
            
            # Reset confirmation step
            st.session_state.clear_db_confirm_step = 0
            
            # Show success message with balloons
            st.balloons()
            
            # Auto-refresh after a moment
            time.sleep(2)
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error clearing database: {e}")
        st.session_state.clear_db_confirm_step = 0

if __name__ == "__main__":
    main()
