import streamlit as st  
import pandas as pd 
import numpy as np 

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page(" World Stock Price Analysis Dashboard", "💱")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")
	
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('World-Stock-Prices-Dataset.csv')

# Title
st.markdown(
    Components.page_header(
        "💱  World Stock Price Analysis Dashboard"
    ), unsafe_allow_html=True
)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>💱 World Stock Price Analysis Dashboard</strong></p>
    <p> World Stock Price data analysis</p>
    <p style='font-size: 0.9rem;'>Navigate using the sidebar to explore different datasets</p>
</div>
""", unsafe_allow_html=True)
