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

    df['Age Group'] = pd.cut(
        df['Age'], 
        bins=[0, 18, 30, 45, 60, 100],
        labels=['<18', '18-30', '31-45', '46-60', '60+']
    )

    df['Total Y-BOCS Score'] = (
        df['Y-BOCS Score (Obsessions)'] + 
        df['Y-BOCS Score (Compulsions)']
    )

    df['Severity Category'] = pd.cut(
        df['Total Y-BOCS Score'], 
        bins=[0, 7, 15, 23, 31, 40], 
        labels=['Subclinical', 'Mild', 'Moderate', 'Severe', 'Extreme']
    )

    df['Comorbidity Profile'] = df.apply(
        lambda row: 
            'Both' if row['Depression Diagnosis'] == 'Yes' and row['Anxiety Diagnosis'] == 'Yes'
            else 'Depression Only' if row['Depression Diagnosis'] == 'Yes'
            else 'Anxiety Only' if row['Anxiety Diagnosis'] == 'Yes'
            else 'None', 
        axis=1
    )

    return df

# Encode categorical variables
@st.cache_data
def encoded_data(df):
    df_encoded = df.copy()
    categorical_cols = [
        'Gender', 'Ethnicity', 'Marital Status', 'Education Level', 
        'Previous Diagnoses', 'Family History of OCD', 'Obsession Type',
        'Compulsion Type', 'Depression Diagnosis', 'Anxiety Diagnosis', 
        'Medications'
    ]
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
    return df_encoded
    
df = load_data()
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

fig = go.Figure()
    
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
st.markdown("   ") 
st.markdown("### 👥 Demographics Overview")
gender_counts = df['Gender'].value_counts()
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
st.markdown("   ")       
st.subheader("👥 :violet[Demographic Analysis]", divider="violet")
tab1, tab2, tab3 = st.tabs(["Age & Gender", "Ethnicity & Education", "Marital Status"]) 

with tab1:
    col1, col2 = st.columns(2)

with col1:
    fig3 = px.histogram(
        df,
        x='Age',
        nbins=30,
        title='Age Distribution',
        color_discrete_sequence=['skyblue'])
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, width="stretch")
with col2:
    age_group_counts = df['Age Group'].value_counts().sort_index()
    fig4 = px.bar(
        x=age_group_counts.index,
        y=age_group_counts.values,
        title='Patients by Age Group',
        labels={'x': 'Age Group', 'y': 'Count'},
        color=age_group_counts.values,
        color_continuous_scale='viridis')
    fig4.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig4, width="stretch")

st.markdown("### Y-BOCS Score by Gender")
fig5 = px.box(
    df,
    x='Gender',
    y='Total Y-BOCS Score',
    color='Gender',
    title='Y-BCOS Score Distribution by Gender',
    color_discrete_map={'Male': '#4ECDC4', 'Female': '#FF6B6B'})
fig5.update_layout(height=400)
st.plotly_chart(fig5, width="stretch")

with tab2:
    col1, col2 = st.columns(2)

with col1:
    ethnicity_counts = df['Ethnicity'].value_counts()
    fig6 = px.bar(
        x=ethnicity_counts.values,
        y=ethnicity_counts.index,
        orientation='h',
        title='Distribution by Ethnicity',
        labels={'x': 'Count', 'y': 'Ethnicity'},
        color=ethnicity_counts.values,
        color_continuous_scale='Teal')
    fig6.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig6, width="stretch")
with col2:
    edu_counts = df['Education Level'].value_counts()
    fig7 = go.Figure(data=[go.Pie(
        labels=edu_counts.index,
        values=edu_counts.values,
        hole=0.3
    )])
    fig7.update_layout(title='Education Level Distribution', height=400)
    st.plotly_chart(fig7, width="stretch")

st.markdown("### Y-BOCS Score by Ethnicity") 
fig8 = px.box(
    df,
    x='Ethnicity',
    y='Total Y-BOCS Score',
    color='Ethnicity',
    title="Y-BOCS Score by Ethnicity")
fig8.update_layout(height=400)
st.plotly_chart(fig8, width="stretch")

with tab3:
    col1, col2 = st.columns(2)

with col1:
    marital_counts = df['Marital Status'].value_counts()
    fig9 = px.pie(
        values=marital_counts.values,
        names=marital_counts.index,
        title='Marital Status Distribution',
        hole=0.4)
    fig9.update_layout(height=400)
    st.plotly_chart(fig9, width="stretch")
with col2:
    fig10 = px.box(
        df,
        x='Marital Status',
        y='Total Y-BOCS Score',
        color='Marital Status',
        title='Y-BOCS Score by Marital Status')
    fig10.update_layout(height=400)
    st.plotly_chart(fig10, width="stretch")

st.markdown("   ")  
st.subheader("🔬 :blue[Clinical Profile Analysis]", divider="blue")
col1, col2, col3 = st.columns(3) 
with col1:
    st.markdown(
        Components.metric_card(
            title="Mean Obsession Score",
            value=f"{df['Y-BOCS Score (Obsessions)'].mean():.2f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Mean Compulsion Score",
            value=f"{df['Y-BOCS Score (Compulsions)'].mean():.2f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Mean Total Score",
            value=f"{df['Total Y-BOCS Score'].mean():.2f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )

# Obsession scores distribution  
fig11 = px.histogram(
    df,
    x='Y-BOCS Score (Obsessions)',
    nbins=25,
    title="Y-BOCS Obsession Scores Distribution",
    color_discrete_sequence=['lightcoral'])
fig11.update_layout(height=400)
st.plotly_chart(fig11, width="stretch")

# Compulsion scores distribution
fig12 = px.histogram(
    df,
    x='Y-BOCS Score (Compulsions)',
    nbins=25,
    title="Y-BOCS Compulsion Scores Distribution",
    color_discrete_sequence=['lightskyblue'])
fig12.update_layout(height=400)
st.plotly_chart(fig12, width="stretch")

# Scatter plot: Obsessions vs Compulsions 
st.markdown("### Relationship: Obsessions vs Compulsions")
fig13 = px.scatter(
    df,
    x='Y-BOCS Score (Obsessions)',
    y='Y-BOCS Score (Compulsions)',
    color='Total Y-BOCS Score',
    size='Total Y-BOCS Score',
    hover_data=['Age', 'Gender', 'Severity Category'],
    title="Y-BOCS Obsessions vs Compulsions",
    color_continuous_scale='Viridis')
fig13.update_layout(height=500)
st.plotly_chart(fig13, width="stretch")

# Obsession types  
obsession_counts = df['Obsession Type'].value_counts()
fig14 = px.bar(
    x=obsession_counts.values,
    y=obsession_counts.index,
    orientation='h',
    title='Obsession Types Distribution',
    labels={'x': 'Count', 'y': 'Obsession Type'},
    color_continuous_scale='Reds')
fig14.update_layout(height=400, showlegend=False)
st.plotly_chart(fig14, width="stretch")

# Compulsion types
compulsion_counts = df['Compulsion Type'].value_counts()
fig15 = px.bar(
    x=compulsion_counts.values,
    y=compulsion_counts.index,
    orientation='h',
    title='Compulsion Types Distribution',
    labels={'x': 'Count', 'y': 'Compulsion Type'},
    color=compulsion_counts.values,
    color_continuous_scale='Blues')
fig15.update_layout(height=400, showlegend=False)
st.plotly_chart(fig15, width="stretch")

# Heatmap: Obsession vs Compulsion
st.markdown("### Cross-Tabulation: Obsession vs Compulsion Types")
crosstab = pd.crosstab(df['Obsession Type'], df['Compulsion Type'])
fig16 = px.imshow(
    crosstab,
    labels=dict(
        x='Compulsion Type',
        y='Obsession Type',
        color='Count'),
    title='Obsession-Compulsion Relationship Heatmap',
    color_continuous_scale='YlOrRd',
    text_auto=True)
fig16.update_layout(height=500)
st.plotly_chart(fig16, width="stretch")
st.markdown("   ")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        Components.metric_card(
            title="Mean Duration (months)",
            value=f"{df['Duration of Symptoms (months)'].mean():.1f}",
            delta="",
            metric_card="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Median Duration (months)",
            value=f"{df['Duration of Symptoms (months)'].median():.1f}",
            delta="",
            metric_card="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Min Duration (months)",
            value=f"{df['Duration of Symptoms (months)'].min():.1f}",
            delta="",
            metric_card="info"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.metric_card(
            title="Max Duration (months)",
            value=f"{df['Duration of Symptoms (months)'].max():.1f}",
            delta="",
            metric_card="info"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
# Duration distribution
fig17 = px.histogram(
    df,
    x='Duration of Symptoms (months)',
    nbins=40,
    title='Symptom Duration Distribution',
    color_discrete_sequence=['mediumpurple'])
fig17.update_layout(height=400)
st.plotly_chart(fig17, width="stretch")

# Duration vs Y-BOCS Score 
st.markdown("### Duration vs Y-BOCS Score") 
fig18 = px.scatter(
    df,
    x='Duration of Symptoms (months)',
    y='Total Y-BOCS Score',
    color='Severity Category',
    hover_data=['Age', 'Gender', 'Obsession Type'],
    title='Symptom Duration vs Y-BOCS Score',
    trendline='lowess')
fig18.update_layout(height=500)
st.plotly_chart(fig18, width="stretch")
st.markdown("   ")
st.subheader("🧬 :red[Comorbidity Analysis]", divider='red')

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
