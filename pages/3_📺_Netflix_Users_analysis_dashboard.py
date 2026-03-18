import streamlit as st  
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Netflix Users Analysis Dashboard", "🎬")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")
	
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('netflix_users.csv')
    df['Age'] = df['Age'].astype(int)
    df['Watch_Time_Hours'] = df['Watch_Time_Hours'].astype(float)
    df['Last_Login'] = pd.to_datetime(df['Last_Login'], errors='coerce')
    df['Watch_Time_Category'] = pd.cut(
        df['Watch_Time_Hours'], 
        bins=[0, 50, 150, 300, np.inf],
        labels=['Low (<50h)', 'Medium (50-150h)', 'High (150-300h)', 'Very High (>300h)'],
        right=False)
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 18, 25, 35, 50, 65, np.inf],
        labels=['<18', '18-24', '25-34', '35-49', '50-64', '65+'],
        right=False)
    df['Last_Login_Days_Ago'] = (datetime.today() - df['Last_Login']).dt.days
    for col in ['Country', 'Subscription_Type', 'Favorite_Genre']:
        df[col] = df[col].astype('category')
    return df


# Title
st.markdown(
    Components.page_header(
        "🎬 Netflix Users Analysis Dashboard"
    ), unsafe_allow_html=True
)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>📺 Netflix Users Analysis Dashboard</strong></p>
    <p>Netflix Users data analysis</p>
    <p style='font-size: 0.9rem;'>Navigate using the sidebar to explore different datasets</p>
</div>
""", unsafe_allow_html=True)
