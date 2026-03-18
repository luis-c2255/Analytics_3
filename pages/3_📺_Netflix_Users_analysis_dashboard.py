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
st.markdown("**Explore user demographics, subscription patterns, and viewing habits.**", text_alignment="center")

# Sidebar filters
st.sidebar.header("Global Filters")
# Country Filter
all_countries = ['All'] + sorted(df['Country'].unique().tolist())
selected_countries = st.sidebar.multiselect("Select Country(ies)", all_countries, default='All')
# Subscription Type filter
all_subscriptions = ['All'] + sorted(df['Subscription_Type'].unique().tolist())
selected_subscriptions = st.sidebar.multiselect("Select Subscription Type(s)", all_subscriptions, default='All')
# Age Range filter
min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))
# Apply filters
filtered_df = df.copy()
if 'All' not in selected_countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
if 'All' not in selected_subscriptions:
    filtered_df = filtered_df[filtered_df['Subscription_Type'].isin(selected_subscriptions)]
filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]

st.sidebar.markdown(f"Users in View: {len(filtered_df)} of {len(df)}")

st.subheader("📊 :green[Overview]", divider="green")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        Components.metric_card(
            title="Total Users",
            value=f"{len(filtered_df):,}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Average Watch Time (Hours)",
            value=f"{filtered_df['Watch_Time_Hours'].mean():.2f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Average Age",
            value=f"{filtered_df['Age'].mean():.1f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    most_popular_genre = filtered_df['Favorite_Genre'].mode()[0]
    st.markdown(
        Components.metric_card(
            title="Most Popular Genre",
            value=f"{most_popular_genre}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
st.subheader(":green[Distribution of Subscription Types]")
if not filtered_df.empty:
    sub_counts = filtered_df['Subscription_Type'].value_counts().reset_index()
    sub_counts.columns = ['Subscription_Type', 'Count']

fig = px.pie(
    sub_counts,
    names='Subscription_Type',
    values='Count',
    title='Subscription Type Distribution',
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig, width="stretch")
st.markdown("   ")
st.subheader(":green[Top 10 Countries by User Count]")
if not filtered_df.empty:
    country_counts = filtered_df['Country'].value_counts().head(10).reset_index()
    country_counts.columns = ['Country', 'Count']
fig2 = px.bar(
    country_counts,
    x='Country',
    y='Count',
    title='Top Countries',
    color='Country',
    color_discrete_sequence=px.colors.qualitative.Vivid
)
st.plotly_chart(fig2, width="stretch")
st.markdown("   ")

st.subheader("👤 :violet[Demographics]", divider="violet")
st.subheader("📈 :blue[Engagement Analysis]", divider="blue")
st.subheader("⭐ :yellow[Genre Insights]", divider="yellow")

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
