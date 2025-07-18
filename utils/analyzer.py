import google.generativeai as genai
import pandas as pd
import json
import time
import streamlit as st
from typing import List, Dict
import os
import re

class GeminiTicketAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None
        
        # Try to configure API key
        if not self._configure_api_key():
            raise ValueError("❌ Gemini API key configuration failed")
        
        # Initialize model
        try:
            # Try different model names in order of preference
            model_names = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-1.5-flash']
            
            for model_name in model_names:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    st.info(f"✅ Using model: {model_name}")
                    break
                except Exception as e:
                    continue
            
            if not self.model:
                raise ValueError("No available Gemini models found")
                
        except Exception as e:
            st.error(f"❌ Failed to initialize Gemini model: {e}")
            raise ValueError(f"Model initialization failed: {e}")
        
        # SDK-focused analysis criteria
        self.sdk_focus_areas = [
            "API Integration Issues",
            "Documentation Problems", 
            "Code Examples Missing",
            "Error Handling Issues",
            "Performance Problems",
            "Authentication Issues",
            "SDK Bug Reports",
            "Feature Requests",
            "Developer Experience Issues",
            "Compatibility Problems"
        ]
    
    def _configure_api_key(self) -> bool:
        """Configure Gemini API key with multiple fallback options"""
        
        api_key = None
        
        try:
            # 1. Direct parameter
            if self.api_key:
                api_key = self.api_key
            
            # 2. Streamlit secrets
            elif hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
                api_key = st.secrets['GEMINI_API_KEY']
            
            # 3. Environment variable
            elif os.getenv('GEMINI_API_KEY'):
                api_key = os.getenv('GEMINI_API_KEY')
            
            # 4. Session state (if user entered it)
            elif hasattr(st.session_state, 'gemini_api_key'):
                api_key = st.session_state.gemini_api_key
            
            if not api_key:
                st.error("❌ No Gemini API key found!")
                return False
            
            # Validate API key format
            if not api_key.startswith('AIza'):
                st.error("❌ Invalid API key format. Gemini API keys should start with 'AIza'")
                return False
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Test the API key with a simple call
            self._test_api_key()
            
            return True
            
        except Exception as e:
            st.error(f"❌ API key configuration error: {e}")
            return False
    
    def _test_api_key(self):
        """Test if the API key works"""
        try:
            # Try to list models to test the API key
            models = list(genai.list_models())
            available_models = [m.name for m in models]
            
            # Check if our preferred model is available
            preferred_models = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
            
            found_model = False
            for model in preferred_models:
                if model in available_models:
                    found_model = True
                    break
            
            if not found_model:
                st.warning(f"⚠️ Preferred models not found. Available: {available_models[:3]}")
                
        except Exception as e:
            st.error(f"❌ API key test failed: {e}")
            raise ValueError(f"Invalid API key: {e}")
    
    def analyze_batch(self, batch_df: pd.DataFrame) -> List[Dict]:
        """Analyze a batch of tickets with focus on SDK improvements"""
        
        # Limit batch size to avoid token limits
        processing_df = batch_df.head(10)  # Process 10 tickets at a time
        
        try:
            # Create focused prompt for SDK analysis
            prompt = self._create_sdk_focused_prompt(processing_df)
            
            # Call Gemini API with updated configuration
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Lower temperature for more consistent analysis
                    max_output_tokens=4096,
                    top_p=0.8,
                    top_k=40
                )
            )
            
            # Parse response
            analysis_results = self._parse_gemini_response(response.text, processing_df)
            
            return analysis_results
            
        except Exception as e:
            st.warning(f"⚠️ Gemini API Error: {e}")
            st.info("Using fallback analysis...")
            # Return fallback analysis
            return self._create_fallback_analysis(processing_df)
    
    def _create_sdk_focused_prompt(self, df: pd.DataFrame) -> str:
        """Create a comprehensive SDK-focused analysis prompt"""
        
        ticket_summaries = []
        for idx, row in df.iterrows():
            # Handle both original and standardized column names
            summary = {
                'id': str(row.get('ticket_id', row.get('Ticket_ID', f'ticket_{idx}'))),
                'subject': str(row.get('ticket_sub_name', row.get('Ticket subject', 'No Subject')))[:150],
                'organization': str(row.get('organization_name', row.get('Ticket organization name', 'Unknown')))[:50],
                'customer_tier': str(row.get('customer_tier', row.get('Customer Tier ', 'Unknown')))[:20],
                'sdk': str(row.get('sdk_name', row.get('SDK', 'Unknown SDK')))[:50],
                'sdk_issue_type': str(row.get('sdk_issue_category', row.get('SDK Issue Types', 'General')))[:100],
                'resolution_time': str(row.get('resolution_time_hours', row.get('Full Resolution Time', 'Unknown'))),
                'call_happened': str(row.get('did_call_happened', row.get('Call Happened for the Customer?', 'Unknown'))),
                'conversation': str(row.get('ticket_conversation', row.get('Comments', 'No conversation')))[:400]
            }
            ticket_summaries.append(summary)
        
        # Create focused prompt
        prompt = f"""
You are an expert SDK support analyst. Analyze these {len(ticket_summaries)} support tickets to identify improvement opportunities.

For each ticket, return a JSON object with this EXACT structure:
{{
    "ticket_id": "string",
    "ticket_category": "string",
    "sdk_issues": ["issue1", "issue2"],
    "
