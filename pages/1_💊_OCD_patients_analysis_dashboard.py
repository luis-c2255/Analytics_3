import streamlit as st  
import pandas as pd  
import numpy as np  
import plotly.express as px  
import plotly.graph_objects as go  
from plotly.subplots import make_subplots  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.preprocessing import LabelEncoder, StandardScaler  
from sklearn.cluster import KMeans  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import r2_score, mean_absolute_error  
import warnings  
warnings.filterwarnings('ignore')  

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("OCD Patients Analysis Dashboard", "💊")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")
	
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('ocd_patient_dataset.csv')
    df['OCD Diagnosis Date'] = pd.to_datetime(df['OCD Diagnosis Date'])
    df['Diagnosis Year'] = df['OCD Diagnosis Date'].dt.year
    df['Age Group'] = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100],
    labels=['<18', '18-30', '31-45', '46-60', '60+'])
    df['Total Y-BOCS Score'] = df['Y-BOCS Score (Obsessions)'] + df['Y-BOCS Score (Compulsions)']
    df['Severity Category'] = pd.cut(df['Total Y-BOCS Score'], bins=[0, 7, 15, 23, 31, 40], labels=['Subclinical', 'Mild', 'Moderate', 'Severe', 'Extreme'])
    df['Comorbidity Profile'] = df.apply(lambda row: 'Both' if row['Depression Diagnosis'] == 'Yes' and row['Anxiety Diagnosis'] == 'Yes'
    else 'Depression Only' if row['Depression Diagnosis'] == 'Yes'
    else 'Anxiety Only' if row['Anxiety Diagnosis'] == 'Yes'
    else 'None', axis=1)
    return df

    df.load_data()

# Encode categorical variables
@st.cache_data
def encoded_data(df):
    df_encoded = df.copy()
    categorical_cols = ['Gender', 'Ethnicity', 'Marital Status', 'Education Level', 'Previous Diagnosis', 'Family History of OCD', 'Obsession Type',
    'Compulsion Type', 'Depression Diagnosis', 'Anxiety Diagnosis', 'Medications']
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
        return df_encoded
    
df_encoded = encoded_data(df)

# Title
st.markdown(
    Components.page_header(
        "💊 OCD Patients Analysis Dashboard"
    ), unsafe_allow_html=True
)
st.subheader("🏠 :orange[Overview]", divider="orange")
col1, col2, col3, col4 = st.columns(4)  
with col1:
    st.markdown(
        Components.metric_card(
            title="Total Patients",
            value=f"{len(df):,}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Avg Y-BOCS Score",
            value=f"{df['Total Y-BOCS Score'].mean():.1f}",
            delta=f"±{df['Total Y-BOCS Score'].std():.1f}",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Avg Age",
            value=f"{df['Age'].mean():.1f} yrs",
            delta=f"±{df['Age'].std():.1f}",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    comorbidity_rate = ((df['Depression Diagnosis']=='Yes') | (df['Anxiety Diagnosis']=='Yes')).sum() / len(df) * 100 
    st.markdown(
        Components.metric_card(
            title="Comorbidity Rate",
            value=f"{comorbidity_rate:.1f}%",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.markdown("   ")  
st.markdown("### 🎯 Key Statistics")
severity_counts = df['Severity Category'].value_counts().sort_index()

fig.add_trace(go.Bar(
    x=severity_counts.index,
    y=severity_counts.values,
    marker_color=['green', 'yellow', 'orange', 'red', 'darkred'],
    text=severity_counts.values,
    textposition='auto'
))
fig.update_layout(
    title="OCD Severity Distribution",
    xaxis_title='Severity',
    yaxis_title='Number of Patients',
    height=400
)
st.plotly_chart(fig, width="stretch")

st.markdown("### 👥 Demographics Overview") 
gender_counts = df['Gender'].value-counts()

fig2 = go.Figure(data=[go.Pie(
    labels=gender_counts.index,
    values=gender_counts.values,
    hole=0.4,
    marker_colors=['#FF6B6B', '#4ECDC4']
)])
fig2.update_layout(
    title='Gender Distribution',
    height=400
)
st.plotly_chart(fig2, width="stretch")
st.markdown("   ")

# Additional overview metrics  
col1, col2, col3 = st.columns(3) 
with col1:
    family_yes = (df['Family History of OCD'] == 'Yes').sum()  
    family_pct = family_yes / len(df) * 100  
    st.markdown(
        Components.metric_card(
            title="With Family History",
            value=f"{family_pct:.1f}%",
            delta=f"{family_yes}",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    depression_yes = (df['Depression Diagnosis'] == 'Yes').sum()  
    depression_pct = depression_yes / len(df) * 100  
    st.markdown(
        Components.metric_card(
            title="With Depression",
            value=f"{depression_pct:.1f}%",
            delta=f"{depression_yes}",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    anxiety_yes = (df['Anxiety Diagnosis'] == 'Yes').sum()  
    anxiety_pct = anxiety_yes / len(df) * 100 
    st.markdown(
        Components.metric_card(
            title="With Anxiety",
            value=f"{anxiety_pct:.1f}%",
            delta=f"{anxiety_yes}",
            card_type="info"
        ), unsafe_allow_html=True
    )

st.subheader("📊 :blue[Demographics]", divider="blue")

st.subheader("🔬 :violet[Clinical Analysis]", divider="violet")

st.subheader("🧬 :red[Comorbidities]", divider="red")

st.subheader("🤖 :yellow[Predictive Model]", divider="yellow")

st.subheader("👥 :violet[Patient Segmentation]", divider="violet")

st.subheader("📈 :blue[Trends & Insights]", divider="blue")

st.subheader("🔍 :green[Custom Explorer]", divider="green")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>💊 OCD Patients Analysis Dashboard</strong></p>
    <p>Patients data analysis</p>
    <p style='font-size: 0.9rem;'>Navigate using the sidebar to explore different datasets</p>
</div>
""", unsafe_allow_html=True)
