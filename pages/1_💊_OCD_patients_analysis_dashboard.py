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
    fig3.update_layout(height=700)
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
    fig4.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig4, width="stretch")

st.markdown("### Y-BOCS Score by Gender")
fig5 = px.box(
    df,
    x='Gender',
    y='Total Y-BOCS Score',
    color='Gender',
    title='Y-BCOS Score Distribution by Gender',
    color_discrete_map={'Male': '#4ECDC4', 'Female': '#FF6B6B'})
fig5.update_layout(height=700)
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
    fig6.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig6, width="stretch")
with col2:
    edu_counts = df['Education Level'].value_counts()
    fig7 = go.Figure(data=[go.Pie(
        labels=edu_counts.index,
        values=edu_counts.values,
        hole=0.3
    )])
    fig7.update_layout(title='Education Level Distribution', height=700)
    st.plotly_chart(fig7, width="stretch")

st.markdown("### Y-BOCS Score by Ethnicity") 
fig8 = px.box(
    df,
    x='Ethnicity',
    y='Total Y-BOCS Score',
    color='Ethnicity',
    title="Y-BOCS Score by Ethnicity")
fig8.update_layout(height=700)
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
    fig9.update_layout(height=700)
    st.plotly_chart(fig9, width="stretch")
with col2:
    fig10 = px.box(
        df,
        x='Marital Status',
        y='Total Y-BOCS Score',
        color='Marital Status',
        title='Y-BOCS Score by Marital Status')
    fig10.update_layout(height=700)
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
fig11.update_layout(height=700)
st.plotly_chart(fig11, width="stretch")

# Compulsion scores distribution
fig12 = px.histogram(
    df,
    x='Y-BOCS Score (Compulsions)',
    nbins=25,
    title="Y-BOCS Compulsion Scores Distribution",
    color_discrete_sequence=['lightskyblue'])
fig12.update_layout(height=700)
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
fig13.update_layout(height=700)
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
fig14.update_layout(height=700, showlegend=False)
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
fig15.update_layout(height=700, showlegend=False)
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
fig16.update_layout(height=700)
st.plotly_chart(fig16, width="stretch")
st.markdown("   ")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        Components.metric_card(
            title="Mean Duration (months)",
            value=f"{df['Duration of Symptoms (months)'].mean():.1f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Median Duration (months)",
            value=f"{df['Duration of Symptoms (months)'].median():.1f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Min Duration (months)",
            value=f"{df['Duration of Symptoms (months)'].min():.1f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.metric_card(
            title="Max Duration (months)",
            value=f"{df['Duration of Symptoms (months)'].max():.1f}",
            delta="",
            card_type="info"
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
fig17.update_layout(height=700)
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
fig18.update_layout(height=700)
st.plotly_chart(fig18, width="stretch")
st.markdown("   ")
st.subheader("🧬 :red[Comorbidity Analysis]", divider='red')
col1, col2, col3, col4 = st.columns(4) 
with col1:
    depression_count = (df['Depression Diagnosis'] == 'Yes').sum()
    depression_pct = depression_count / len(df) * 100
    st.markdown(
        Components.metric_card(
            title="Depression",
            value=f"{depression_pct:.1f}%",
            delta=f"{depression_count}",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    anxiety_count = (df['Anxiety Diagnosis'] == 'Yes').sum()  
    anxiety_pct = anxiety_count / len(df) * 100 
    st.markdown(
        Components.metric_card(
            title="Anxiety",
            value=f"{anxiety_pct:.1f}%",
            delta=f"{anxiety_count}",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    both_count = ((df['Depression Diagnosis'] == 'Yes') & (df['Anxiety Diagnosis'] == 'Yes')).sum()
    both_pct = both_count / len(df) * 100 
    st.markdown(
        Components.metric_card(
            title="Both",
            value=f"{both_pct:.1f}%",
            delta=f"{both_count}",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    family_count = (df['Family History of OCD'] == 'Yes').sum()  
    family_pct = family_count / len(df) * 100 
    st.markdown(
        Components.metric_card(
            title="Family History",
            value=f"{family_pct:.1f}%",
            delta=f"{family_count}",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.markdown("   ") 
# Comorbidity profile distribution 
comorbidity_counts = df['Comorbidity Profile'].value_counts() 
fig19 = px.bar(
    x=comorbidity_counts.index,
    y=comorbidity_counts.values,
    title='Comorbidity Profile Distribution',
    labels={'x': 'Profile', 'y': 'Count'},
    color=comorbidity_counts.index,
    color_discrete_map={
        'None': 'lightgreen',
        'Depression Only': 'lightblue',
        'Anxiety Only': 'lightyellow',
        'Both': 'lightcoral'
    })
fig19.update_layout(height=700)
st.plotly_chart(fig19, width="stretch")

# Y-BOCS by comorbidity profile 
fig20 = px.box(
    df,
    x='Comorbidity Profile',
    y='Total Y-BOCS Score',
    color='Comorbidity Profile',
    title='Y-BOCS Score by Comorbidity Profile',
    category_orders={'Comorbidity Profile': ['None', 'Depression Only', 'Anxiety Only', 'Both']})
fig20.update_layout(height=700)
st.plotly_chart(fig20, width="stretch")

# Family history impact  
st.markdown("### 🧬 Family History Impact on Severity")  
fig21 = px.box(
    df,
    x='Family History of OCD',
    y='Total Y-BOCS Score',
    color='Family History of OCD',
    title="Y-BOCS Score vs Family History",
    color_discrete_map={'Yes': '#FF6B6B', 'No': '#95E1D3'})
fig21.update_layout(height=700)
st.plotly_chart(fig21, width="stretch")

# Statistical comparison  
with_family = df[df['Family History of OCD'] == 'Yes']['Total Y-BOCS Score']
without_family = df[df['Family History of OCD'] == 'No']['Total Y-BOCS Score']

st.markdown("### Statistical Summary:")
st.write(f"**With Family History:** Mean = {with_family.mean():.2f}, Std = {with_family.std():.2f}")
st.write(f"**Without Family History:** Mean = {without_family.mean():.2f}, Std = {without_family.std():.2f}") 

# Medications distribution  
med_counts = df['Medications'].value_counts()
fig22 = px.pie(
    values=med_counts.values,
    names=med_counts.index,
    title='Medication Types Distribution',
    hole=0.4)
fig22.update_layout(height=700)
st.plotly_chart(fig22, width="stretch")

# Medication effectiveness  
st.markdown("### 💊 Medication Analysis")
col1, col2 = st.columns(2) 
with col1:
    fig23 = px.box(
        df,
        x='Medications',
        y='Total Y-BOCS Score',
        color='Medications',
        title="Y-BOCS Score by Medication Type")
    fig23.update_layout(height=400)
    st.plotly_chart(fig23, width="stretch")
with col2:
    fig24 = px.violin(
        df,
        x='Medications',
        y='Duration of Symptoms (months)',
        color='Medications',
        title="Symptom Duration by Medication Type")
    fig24.update_layout(height=400)
    st.plotly_chart(fig24, width="stretch")

st.markdown("   ")
st.subheader("🤖 :rainbow[Y-BOCS Score Prediction Model]", divider="rainbow")
st.info("📊 This model predicts Total Y-BOCS Score based on patient characteristics using Random Forest Regression.")

# Prepare features
feature_cols = [
    'Age', 'Duration of Symptoms (months)', 'Gender_Encoded',
    'Ethnicity_Encoded', 'Marital Status_Encoded',
    'Education Level_Encoded', 'Family History of OCD_Encoded',
    'Obsession Type_Encoded', 'Compulsion Type_Encoded',
    'Depression Diagnosis_Encoded', 'Anxiety Diagnosis_Encoded',
    'Medications_Encoded'
]
X = df_encoded[feature_cols]  
y = df_encoded['Total Y-BOCS Score'] 

# Split data  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model  
with st.spinner("Training Random Forest model..."):
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(((y_test - y_pred) **2).mean())

# Model performance  
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        Components.metric_card(
            title="R² Score",
            value=f"{r2:.4f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="MAE",
            value=f"{mae:.4f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="RMSE",
            value=f"{rmse:.4f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
# Actual vs Predicted  
fig25 = go.Figure()

fig25.add_trace(go.Scatter(
    x=y_test,
    y=y_pred,
    mode='markers',
    name='Predictions',
    marker=dict(color='steelblue', size=8, opacity=0.6)
))
# Perfect prediction line  
min_val = min(y_test.min(), y_pred.min())  
max_val = max(y_test.max(), y_pred.max())
fig25.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Perfect Prediction',
    line=dict(color='red', dash='dash', width=2)
))
fig25.update_layout(
    title='Actual vs Predicted Y-BOCS Scores',
    xaxis_title='Actual Y-BOCS Score',
    yaxis_title='Predicted Y-BOCS Score',
    height=500
)
st.plotly_chart(fig25, width="stretch")

# Residuals plot  
residuals = y_test - y_pred

fig26 = go.Figure()
fig26.add_trace(go.Scatter(
    x=y_pred,
    y=residuals,
    mode='markers',
    marker=dict(color='coral', size=8, opacity=0.6)
))
fig26.add_hline(
    y=0, 
    line_dash='dash',
    line_color='red',
    line_width=2
)
fig26.update_layout(
    title='Residual Plot',
    xaxis_title='Predicted Y-BOCS Score',
    yaxis_title='Residuals',
    height=500
)
st.plotly_chart(fig26, width="stretch")
# Feature importance  
st.markdown("### 📊 Feature Importance")

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Clean feature names
feature_importance['Feature'] = feature_importance['Feature'].str.replace('_Encoded', '')

fig27 = px.bar(
    feature_importance,
    x='Importance',
    y='Feature',
    orientation='h',
    title='Feature Importance for Y-BOCS Prediction',
    color='Importance',
    color_continuous_scale='Viridis')
fig27.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig27, width="stretch")

  
# Prediction tool  
st.markdown("   ")  
st.markdown("### 🎯 Make a Prediction")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        pred_age = st.number_input(
        'Age', 
        min_value=18,
        max_value=100,
        value=35)
        pred_duration = st.number_input(
        'Duration (months)',
        min_value=1,
        max_value=500,
        value=100
        )
        pred_gender = st.selectbox('Gender', df['Gender'].unique())
        pred_ethnicity = st.selectbox('Ethnicity', df['Ethnicity'].unique())

    with col2:
        pred_marital = st.selectbox('Marital Status', df['Marital Status'].unique())
        pred_education = st.selectbox('Education Level', df['Education Level'].unique())
        pred_family = st.selectbox('Family History of OCD', df['Family History of OCD'].unique())
        pred_obsession = st.selectbox('Obsession Type', df['Obsession Type'].unique())
    with col3:
        pred_compulsion = st.selectbox("Compulsion Type", df['Compulsion Type'].unique())
        pred_depression = st.selectbox("Depression Diagnosis", df['Depression Diagnosis'].unique())
        pred_anxiety = st.selectbox("Anxiety Diagnosis", df['Anxiety Diagnosis'].unique())
        pred_medication = st.selectbox("Medications", df['Medications'].unique())
    submit_button = st.form_submit_button("🔮 Predict Y-BOCS Score")
if submit_button:
    # Encode inputs
    from sklearn.preprocessing import LabelEncoder
    pred_data = {
        'Age': pred_age,
        'Duration of Symptoms (months)': pred_duration,  
        'Gender': pred_gender,  
        'Ethnicity': pred_ethnicity,  
        'Marital Status': pred_marital,  
        'Education Level': pred_education,  
        'Family History of OCD': pred_family,  
        'Obsession Type': pred_obsession,  
        'Compulsion Type': pred_compulsion,  
        'Depression Diagnosis': pred_depression,  
        'Anxiety Diagnosis': pred_anxiety,  
        'Medications': pred_medication  
    }
    # Encode categorical variables
    encoded_values = []
    encoded_values.append(pred_age)
    encoded_values.append(pred_duration)
    for col in [
        'Gender', 'Ethnicity', 'Marital Status',
        'Education Level', 'Family History of OCD', 'Obsession Type',
        'Compulsion Type', 'Depression Diagnosis', 'Anxiety Diagnosis',
        'Medications' ]:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoded_values.append(le.transform([pred_data[col]])[0])
    # Make prediction
    prediction = rf.model.predict([encoded_values])[0]
    # Determine severity
    if prediction <= 7:
        severity = "Subclinical"
        color='green'
    elif prediction <= 15:
        severity = "Mild"
        color='yellow'
    elif prediction <= 23:
        severity = "Moderate"
        color='orange'
    elif prediction <= 31:
        severity = "Severe"
        color='red'
    else:
        severity = "Extreme"
        color='darkred'
    st.success(f"### Predicted Y-BOCS Score: **{prediction:.2f}**")
    st.info(f"#### Severity Category: **{severity}**")

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
