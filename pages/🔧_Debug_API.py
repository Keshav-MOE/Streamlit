import streamlit as st
import google.generativeai as genai

st.title("🔧 API Key Debug")

if st.session_state.get('api_configured'):
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        
        # Test API
        models = genai.list_models()
        st.success(f"✅ API working! Found {len(list(models))} models")
        
        # Test generation
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say hello")
        st.write("Test response:", response.text)
        
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        st.code(str(e))
else:
    st.warning("⚠️ No API key configured")
