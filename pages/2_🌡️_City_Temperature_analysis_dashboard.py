import streamlit as st  
import pandas as pd 
import numpy as np 

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("City Temperature Analysis Dashboard", "🌡")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")
	
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('city_temperature.csv')

# Title
st.markdown(
    Components.page_header(
        "🌡 City Temperature Analysis Dashboard"
    ), unsafe_allow_html=True
)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>🌡 City Temperature Analysis Dashboard</strong></p>
    <p>City Temperature data analysis</p>
    <p style='font-size: 0.9rem;'>Navigate using the sidebar to explore different datasets</p>
</div>
""", unsafe_allow_html=True)
