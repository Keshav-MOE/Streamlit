import google.generativeai as genai
import pandas as pd
import json
import time
import streamlit as st
from typing import List, Dict

class GeminiTicketAnalyzer:
    def __init__(self, api_key=None):
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to get from Streamlit secrets
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            except:
                raise ValueError("Gemini API key not found")
        
        self.model = genai.GenerativeModel('gemini-pro')
        
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
    
    def analyze_batch(self, batch_df: pd.DataFrame) -> List[Dict]:
        """Analyze a batch of tickets with focus on SDK improvements"""
        
        # Limit batch size to avoid token limits
        processing_df = batch_df.head(20)
        
        try:
            # Create focused prompt for SDK analysis
            prompt = self._create_sdk_focused_prompt(processing_df)
            
            # Call Gemini API
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Lower temperature for more consistent analysis
                    max_output_tokens=4096,
                )
            )
            
            # Parse response
            analysis_results = self._parse_gemini_response(response.text, processing_df)
            
            return analysis_results
            
        except Exception as e:
            st.error(f"Gemini API Error: {e}")
            # Return fallback analysis
            return self._create_fallback_analysis(processing_df)
    
    def _create_sdk_focused_prompt(self, df: pd.DataFrame) -> str:
        """Create a comprehensive SDK-focused analysis prompt"""
        
        ticket_summaries = []
        for idx, row in df.iterrows():
            summary = {
                'id': str(row.get('ticket_id', f'ticket_{idx}')),
                'subject': str(row.get('ticket_sub_name', 'No Subject'))[:200],
                'resolution_time': str(row.get('resolution_time', 'Unknown')),
                'conversation': str(row.get('ticket_conversation', 'No conversation'))[:800],
                'category': str(row.get('category', 'General'))
            }
            ticket_summaries.append(summary)
        
        prompt = f"""
You are an expert SDK analyst reviewing support tickets to identify improvement opportunities.

Analyze these {len(ticket_summaries)} support tickets and return a JSON array with analysis for each ticket.

For each ticket, focus on:
1. SDK Integration challenges
2. Documentation gaps  
3. Developer experience issues
4. API/SDK bugs or limitations
5. Feature requests
6. Performance concerns

SDK Focus Areas to consider:
{', '.join(self.sdk_focus_areas)}

Return JSON array in this exact format:
[
  {{
    "ticket_id": "ticket_id_here",
    "ticket_category": "category_name",
    "sdk_issues": ["specific_issue_1", "specific_issue_2"],
    "improvement_suggestions": ["suggestion_1", "suggestion_2"], 
    "priority_score": 1-10,
    "resolution_time_hours": estimated_hours,
    "customer_satisfaction": 1-5,
    "sdk_impact": "High/Medium/Low"
  }}
]

Tickets to analyze:
"""
        
        for i, ticket in enumerate(ticket_summaries):
            prompt += f"""

Ticket {i+1}:
ID: {ticket['id']}
Subject: {ticket['subject']}
Resolution Time: {ticket['resolution_time']}
Category: {ticket['category']}
Conversation: {ticket['conversation']}
---
"""
        
        prompt += """

Return ONLY the JSON array. No markdown formatting, no explanations, just valid JSON.
"""
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str, df: pd.DataFrame) -> List[Dict]:
        """Parse and validate Gemini's JSON response"""
        
        try:
            # Clean the response text
            response_text = response_text.strip()
            
            # Remove markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Find JSON array bounds
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_content = response_text[start_idx:end_idx]
                results = json.loads(json_content)
                
                # Validate and clean results
                validated_results = []
                for result in results:
                    validated_result = self._validate_analysis_result(result)
                    validated_results.append(validated_result)
                
                return validated_results
            
            else:
                raise ValueError("No valid JSON array found in response")
                
        except Exception as e:
            st.warning(f"Failed to parse Gemini response: {e}. Using fallback analysis.")
            return self._create_fallback_analysis(df)
    
    def _validate_analysis_result(self, result: Dict) -> Dict:
        """Validate and clean a single analysis result"""
        
        cleaned_result = {
            'ticket_id': str(result.get('ticket_id', 'unknown')),
            'ticket_category': str(result.get('ticket_category', 'General')),
            'sdk_issues': result.get('sdk_issues', []) if isinstance(result.get('sdk_issues'), list) else ['API Integration'],
            'improvement_suggestions': result.get('improvement_suggestions', []) if isinstance(result.get('improvement_suggestions'), list) else ['Better documentation'],
            'priority_score': max(1, min(10, int(result.get('priority_score', 5)))),
            'resolution_time_hours': float(result.get('resolution_time_hours', 0)) if str(result.get('resolution_time_hours', 0)).replace('.','').isdigit() else 0.0,
            'customer_satisfaction': max(1, min(5, int(result.get('customer_satisfaction', 3)))),
            'sdk_impact': result.get('sdk_impact', 'Medium')
        }
        
        return cleaned_result
    
    def _create_fallback_analysis(self, df: pd.DataFrame) -> List[Dict]:
        """Create basic analysis when Gemini fails"""
        
        results = []
        for idx, row in df.iterrows():
            
            # Basic categorization based on subject
            subject = str(row.get('ticket_sub_name', '')).lower()
            
            if any(word in subject for word in ['api', 'integration', 'sdk']):
                category = 'SDK Integration'
                sdk_issues = ['API Integration Issues']
                priority = 7
            elif any(word in subject for word in ['doc', 'guide', 'help']):
                category = 'Documentation'
                sdk_issues = ['Documentation Problems']
                priority = 5
            elif any(word in subject for word in ['error', 'bug', 'issue']):
                category = 'Bug Report'
                sdk_issues = ['SDK Bug Reports']
                priority = 8
            else:
                category = 'General Support'
                sdk_issues = ['General SDK Questions']
                priority = 5
            
            result = {
                'ticket_id': str(row.get('ticket_id', f'fallback_{idx}')),
                'ticket_category': category,
                'sdk_issues': sdk_issues,
                'improvement_suggestions': ['Improve documentation', 'Add code examples'],
                'priority_score': priority,
                'resolution_time_hours': float(row.get('resolution_time', 0)) if str(row.get('resolution_time', 0)).replace('.','').isdigit() else 0.0,
                'customer_satisfaction': 3,
                'sdk_impact': 'Medium'
            }
            
            results.append(result)
        
        return results