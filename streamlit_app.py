import streamlit as st
from utils.theme import Components, Colors, apply_chart_theme, init_page

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    Components.page_header("📊 Multiple Analysis Dashboard"), unsafe_allow_html=True)

with st.container(height="content", width="stretch", horizontal_alignment="center"):    
    st.image("utils/image.svg")

col1, col2, col3 = st.columns(3)
with col1:
    st.link_button("OCD Patients Analysis Dashboard", 
    "https://analytics2.streamlit.app/OCD_patients_analysis_dashboard",
    icon="💊", icon_position="left", width="stretch"
    )

with col2:
    st.link_button("City Temperature Analysis Dashboard", 
    "https://analytics2.streamlit.app/City_Temperature_analysis_dashboard",
    icon="🌡", icon_position="left", width="stretch"
    )
with col3:
    st.link_button("Netflix Users Analysis Dashboard", 
    "https://analytics2.streamlit.app/Netflix_Users_analysis_dashboard", 
    icon="📺", icon_position="left", width="stretch"
    )

col4, col5 = st.columns(2)
with col4:
    st.link_button("Consumer Behavior Analysis Dashboard", 
    "https://analytics2.streamlit.app/Consumer_Behavior_analysis_dashboard",
    icon="🛒", icon_position="left", width="stretch"
    )

with col5:
    st.link_button("World Stock Prices Analysis Dashboard", 
    "https://analytics2.streamlit.app/World_Stock_Prices_analysis.dashboard",
    icon="💱", icon_position="left", width="stretch"
    )


# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>📊 Multiple Analysis Dashboard</strong></p>
    <p>Multiple Dashboards from several datasets analyzed</p>
    <p style='font-size: 0.9rem;'>Navigate using the sidebar to explore different datasets</p>
</div>
""", unsafe_allow_html=True)
