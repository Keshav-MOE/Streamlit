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
        self.analysis_count = 0
        
        # Try to configure API key
        if not self._configure_api_key():
            raise ValueError("❌ Gemini API key configuration failed")
        
        # Initialize model
        try:
            # Try different model names in order of preference
            model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            
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
        """Analyze a batch of tickets with focus on SDK improvements and debug info"""
        
        # Debug info
        st.info(f"🤖 Starting AI analysis of {len(batch_df)} tickets...")
        
        # Limit batch size to avoid token limits
        processing_df = batch_df.head(10)  # Process 10 tickets at a time
        
        if len(batch_df) > 10:
            st.warning(f"⚠️ Limiting batch to 10 tickets (from {len(batch_df)}) to avoid API limits")
        
        try:
            # Show what we're analyzing
            with st.expander("🔍 Tickets Being Analyzed"):
                for idx, row in processing_df.iterrows():
                    ticket_id = row.get('ticket_id', row.get('Ticket_ID', f'row_{idx}'))
                    subject = str(row.get('ticket_sub_name', row.get('Ticket subject', 'No subject')))[:50]
                    st.text(f"• {ticket_id}: {subject}...")
            
            # Create focused prompt for SDK analysis
            st.info("📝 Generating AI analysis prompt...")
            prompt = self._create_sdk_focused_prompt(processing_df)
            
            # Show prompt length for debugging
            st.info(f"📊 Prompt length: {len(prompt)} characters")
            
            # Call Gemini API with updated configuration
            st.info("🚀 Calling Gemini API...")
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Lower temperature for more consistent analysis
                    max_output_tokens=4096,
                    top_p=0.8,
                    top_k=40
                )
            )
            
            st.success("✅ Gemini API call completed")
            
            # Parse response
            st.info("📋 Parsing AI response...")
            analysis_results = self._parse_gemini_response(response.text, processing_df)
            
            # Debug: Show analysis results count
            st.success(f"🎉 Successfully analyzed {len(analysis_results)} tickets")
            
            # Update counter
            self.analysis_count += len(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            st.warning(f"⚠️ Gemini API Error: {e}")
            st.info("🔄 Using fallback analysis...")
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
                'sdk': str(row.get('sdk_name', row.get('SDK', 'Unknown')))[:50],
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
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "priority_score": 5,
    "resolution_time_hours": 0.0,
    "customer_satisfaction": 3,
    "sdk_impact": "Medium"
}}

Focus on identifying:
- SDK integration problems
- Documentation gaps
- API/SDK bugs
- Performance issues
- Developer experience problems

Tickets to analyze:
"""
        
        for i, ticket in enumerate(ticket_summaries):
            prompt += f"""
Ticket {i+1}:
ID: {ticket['id']}
Subject: {ticket['subject']}
Organization: {ticket['organization']}
Customer Tier: {ticket['customer_tier']}
SDK: {ticket['sdk']}
Issue Type: {ticket['sdk_issue_type']}
Resolution Time: {ticket['resolution_time']}
Call Made: {ticket['call_happened']}
Description: {ticket['conversation']}
---
"""
        
        prompt += f"""
Return a JSON array with exactly {len(ticket_summaries)} objects. Only return valid JSON, no other text or markdown.
"""
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str, df: pd.DataFrame) -> List[Dict]:
        """Parse and validate Gemini's JSON response with debug info"""
        
        st.info(f"📄 Response length: {len(response_text)} characters")
        
        # Show first 200 chars of response for debugging
        with st.expander("🔍 AI Response Preview"):
            st.text(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        
        try:
            # Clean the response text
            response_text = response_text.strip()
            
            # Remove markdown formatting if present
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                if end != -1:
                    response_text = response_text[start:end].strip()
                    st.info("🧹 Removed JSON markdown formatting")
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                if end != -1:
                    response_text = response_text[start:end].strip()
                    st.info("🧹 Removed generic markdown formatting")
            
            # Find JSON array bounds
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_content = response_text[start_idx:end_idx]
                st.info(f"📋 Extracted JSON content: {len(json_content)} characters")
                
                results = json.loads(json_content)
                st.success(f"✅ Successfully parsed JSON with {len(results)} items")
                
                # Validate results
                if isinstance(results, list):
                    validated_results = []
                    for i, result in enumerate(results):
                        # Ensure we have a ticket_id from the original data
                        if i < len(df):
                            original_row = df.iloc[i]
                            result['ticket_id'] = str(original_row.get('ticket_id', 
                                                                     original_row.get('Ticket_ID', f'ticket_{i}')))
                        
                        validated_result = self._validate_analysis_result(result)
                        validated_results.append(validated_result)
                    
                    st.success(f"✅ Validated {len(validated_results)} analysis results")
                    return validated_results
            
            # If parsing fails, create fallback
            raise ValueError("Could not parse JSON response")
                
        except Exception as e:
            st.warning(f"⚠️ Response parsing failed: {e}. Using fallback analysis.")
            return self._create_fallback_analysis(df)
    
    def _validate_analysis_result(self, result: Dict) -> Dict:
        """Validate and clean a single analysis result with array handling"""

        def safe_convert_float(value, default=0.0):
            try:
                if value is None or pd.isna(value):
                    return default
                # Handle arrays/series
                if isinstance(value, (pd.Series, np.ndarray)):
                    if len(value) == 0:
                        return default
                    elif len(value) == 1:
                        value = value.iloc[0] if isinstance(value, pd.Series) else value[0]
                    else:
                        return default  # Can't convert array to single float

                # Handle string formats like "2.5 hours", "3h", etc.
                if isinstance(value, str):
                    numbers = re.findall(r'[\d.]+', value)
                    if numbers:
                        return float(numbers[0])
                return float(value)
            except Exception:
                return default

        def safe_convert_int(value, default, min_val, max_val):
            try:
                if value is None or pd.isna(value):
                    return default
                # Handle arrays/series
                if isinstance(value, (pd.Series, np.ndarray)):
                    if len(value) == 0:
                        return default
                    elif len(value) == 1:
                        value = value.iloc[0] if isinstance(value, pd.Series) else value[0]
                    else:
                        return default
                return max(min_val, min(max_val, int(float(value))))
            except Exception:
                return default

        def ensure_list(value, default_list):
            if value is None or pd.isna(value):
                return default_list
            elif isinstance(value, str) and value.strip():
                return [value.strip()]
            elif isinstance(value, (list, tuple)):
                return [str(item) for item in value if item is not None and not pd.isna(item) and str(item).strip()]
            elif isinstance(value, (pd.Series, np.ndarray)):
                return [str(item) for item in value if item is not None and not pd.isna(item) and str(item).strip()]
            else:
                return [str(value)] if str(value).strip() else default_list

        # Apply safe conversions
        cleaned_result = {
            'ticket_id': str(result.get('ticket_id', 'unknown')),
            'ticket_category': str(result.get('ticket_category', 'General')),
            'sdk_issues': ensure_list(result.get('sdk_issues'), ['General SDK Issue']),
            'improvement_suggestions': ensure_list(result.get('improvement_suggestions'), ['Improve documentation']),
            'priority_score': safe_convert_int(result.get('priority_score'), 5, 1, 10),
            'resolution_time_hours': safe_convert_float(result.get('resolution_time_hours'), 0.0),
            'customer_satisfaction': safe_convert_int(result.get('customer_satisfaction'), 3, 1, 5),
            'sdk_impact': str(result.get('sdk_impact', 'Medium'))
        }

        return cleaned_result
    
    def _create_fallback_analysis(self, df: pd.DataFrame) -> List[Dict]:
        """Create basic analysis when Gemini fails"""
        
        st.info(f"🔄 Creating fallback analysis for {len(df)} tickets")
        
        results = []
        for idx, row in df.iterrows():
            
            # Get ticket ID from either column name format
            ticket_id = str(row.get('ticket_id', row.get('Ticket_ID', f'fallback_{idx}')))
            
            # Basic categorization based on available data
            subject = str(row.get('ticket_sub_name', row.get('Ticket subject', ''))).lower()
            sdk_name = str(row.get('sdk_name', row.get('SDK', 'Unknown')))
            sdk_issue_type = str(row.get('sdk_issue_category', row.get('SDK Issue Types', '')))
            
            # Determine category and issues based on available information
            if any(word in subject for word in ['api', 'integration', 'connect']):
                category = 'API Integration'
                sdk_issues = ['API Integration Issues']
                priority = 7
            elif any(word in subject for word in ['doc', 'guide', 'help', 'how']):
                category = 'Documentation'
                sdk_issues = ['Documentation Problems']
                priority = 5
            elif any(word in subject for word in ['error', 'bug', 'crash', 'fail']):
                category = 'Bug Report'
                sdk_issues = ['SDK Bug Reports']
                priority = 8
            elif any(word in subject for word in ['slow', 'performance', 'timeout']):
                category = 'Performance'
                sdk_issues = ['Performance Problems']
                priority = 7
            elif 'auth' in subject or 'login' in subject:
                category = 'Authentication'
                sdk_issues = ['Authentication Issues']
                priority = 8
            else:
                category = sdk_issue_type if sdk_issue_type and sdk_issue_type != 'nan' else 'General Support'
                sdk_issues = ['General SDK Questions']
                priority = 5
            
            # Extract resolution time
            resolution_time = 0.0
            resolution_field = row.get('resolution_time_hours', row.get('Full Resolution Time', 0))
            try:
                if pd.notna(resolution_field):
                    resolution_time = float(str(resolution_field).replace('h', '').replace('hours', '').strip())
            except:
                resolution_time = 0.0
            
            result = {
                'ticket_id': ticket_id,
                'ticket_category': category,
                'sdk_issues': sdk_issues,
                'improvement_suggestions': [
                    'Improve documentation',
                    'Add more code examples',
                    'Enhance error messages'
                ],
                'priority_score': priority,
                'resolution_time_hours': resolution_time,
                'customer_satisfaction': 3,  # Default neutral satisfaction
                'sdk_impact': 'Medium'
            }
            
            results.append(result)
        
        st.success(f"✅ Created {len(results)} fallback analysis results")
        return results
