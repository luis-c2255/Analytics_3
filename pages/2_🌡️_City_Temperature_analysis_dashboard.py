import streamlit as st  
import pandas as pd 
import numpy as np 
import plotly.express as px  
import plotly.graph_objects as go  
import numpy as np  
from datetime import datetime

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

# Data Cleaning
df = df[(df['AvgTemperature'] >= -100) & (df['AvgTemperature'] <= 150)]
df = df.drop_duplicates()
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce')
df = df.dropna(subset=['Date'])
df['AvgTemperature_C'] = (df['AvgTemperature'] - 32) * 5/9
df['Season'] = df['Month'].map({
	12: 'Winter', 1: 'Winter', 2: 'Winter',
	3: 'Spring', 4: 'Spring', 5: 'Spring',
	6: 'Summer', 7: 'Summer', 8: 'Summer',
	9: 'Fall', 10: 'Fall', 11: 'Fall' 
})
return df 
df = load_data() 

# Title
st.markdown(
    Components.page_header(
        "🌡 City Temperature Analysis Dashboard"
    ), unsafe_allow_html=True
)
st.markdown("""  
Explore **2.9M+ temperature records** across global cities from 1995-2020.  
Analyze climate trends, seasonal patterns, and geographic temperature variations.  
""") 
# Sidebar filters  
st.sidebar.header("🎛️ Filters") 
# Temperature unit selection  
temp_unit = st.sidebar.radio("Temperature Unit:", ["Fahrenheit", "Celsius"]) 
temp_col = 'AvgTemperature' if temp_unit == "Fahrenheit" else 'AvgTemperature_C' 
temp_symbol = '°F' if temp_unit == "Fahrenheit" else '°C'

# Region filter  
regions = ['All'] + sorted(df['Region'].unique().tolist())  
selected_region = st.sidebar.selectbox("Select Region:", regions)

# Country filter (dynamic based on region)  
if selected_region != 'All':
	countries = ['All'] + sorted(df[df['Region'] == selected_region]['Country'].unique().tolist())
else:
	countries = ['All'] + sorted(df['Country'].unique().tolist())
	selected_country = st.sidebar.selectbox("Select Country:", countries)
# Year range filter  
year_range = st.sidebar.slider(
	"Year Range:",
	int(df['Year'].min()),
	int(df['Year'].max()),
	(int(df['Year'].min()), int(df['Year'].max()))
)
# Apply filters 
df_filtered = df.copy()
if selected_region != 'All':
	df_filtered = df_filtered[df_filtered['Region'] == selected_region]
if selected_country != 'All':
	df_filtered = df_filtered[df_filtered['Country'] == selected_country]
	df_filtered = df_filtered[(df_filtered['Year'] >= year_range[0]) & (df_filtered['Year'] <= year_range[1])]

# Key Metrics  
st.header("📊 Key Metrics")  
col1, col2, col3, col4 = st.columns(4) 

with col1:
	avg_temp = df_filtered[temp_col].mean()
	st.metric("Average Temperature", f"{avg_temp:.1f}{temp_symbol}")
with col2:
	max_temp = df_filtered[temp_col].max()
	st.metric("Highest Temperature", f"{max_temp:.1f}{temp_symbol}")
with col3:
	min_temp = df_filtered[temp_col].min()
	st.metric("Lowest Temperature", f"{min_temp:.1f}{temp_symbol}")
with col4:
	unique_cities = df_filtered['City'].nunique()
	st.metric("Cities Analyzed", f"{unique_cities:,}")

# Tabs for different analyses  
tab1, tab2, tab3, tab4, tab5 = st.tabs([
	"🌍 Global Trends", "📈 Climate Change",
	"🗓️ Seasonal Patterns", "🏙️ City Comparison",
	"📍 Geographic Analysis"
])

# TAB 1: Global Trends  
with tab1:
	st.header("Global Temperature Trends Over Time")
# Yearly average trend  
yearly_data = df_filtered.groupby('Year')[temp_col].mean().reset_index()

fig1 = px.line(
	yearly_data,
	x='Year',
	y=temp_col,
	title=f'Average Global Temperature Trend ({temp_symbol})',
	labels={temp_col: f'Temperature ({temp_symbol})', 'Year': 'Year'}
)
fig1.add_scatter(  
	x=yearly_data['Year'],  
	y=yearly_data[temp_col].rolling(window=5, center=True).mean(),  
	mode='lines',  
	name='5-Year Moving Average',  
	line=dict(color='red', width=3, dash='dash')  
)
fig1.update_layout(height=500, hovermode='x unified')
st.plotly_chart(fig1, width="stretch")

# Regional trends comparison  
st.subheader("Temperature Trends by Region")  
regional_yearly = df_filtered.groupby(['Year', 'Region'])[temp_col].mean().reset_index() 

fig2 = px.line(  
regional_yearly,  
x='Year',  
y=temp_col,  
color='Region',  
title=f'Regional Temperature Trends ({temp_symbol})',  
labels={temp_col: f'Temperature ({temp_symbol})', 'Year': 'Year'}  
) 
fig2.update_layout(height=500, hovermode='x unified')  
st.plotly_chart(fig2, width="stretch")

# TAB 2: Climate Change Analysis  
with tab2:  
	st.header("Climate Change Indicators")  
  
col1, col2 = st.columns(2)
with col1:  
	# Calculate warming rates  
	st.subheader("Warming Rate by Region")

regional_trends = df_filtered.groupby(['Region', 'Year'])[temp_col].mean().reset_index()  
warming_rates = []
for region in df_filtered['Region'].unique():
	region_data = regional_trends[regional_trends['Region'] == region]
	if len(region_data) > 1:
		slope = np.polyfit(region_data['Year'], region_data[temp_col], 1)[0]
		warming_rates.append({'Region': region, 'Rate': slope})
		warming_df = pd.DataFrame(warming_rates).sort_values('Rate', ascending=False)

fig3 = px.bar(  
	warming_df,  
	x='Rate',  
	y='Region',  
	orientation='h',  
	title=f'Temperature Change Rate ({temp_symbol}/year)',  
	labels={'Rate': f'Change Rate ({temp_symbol}/year)', 'Region': 'Region'},  
	color='Rate',  
	color_continuous_scale='RdYlBu_r'  
	)  
fig3.update_layout(height=400)  
st.plotly_chart(fig3, width="stretch")

with col2:  
	# Decade comparison  
	st.subheader("Decade-over-Decade Change")
df_filtered['Decade'] = (df_filtered['Year'] // 10) * 10  
decade_avg = df_filtered.groupby('Decade')[temp_col].mean().reset_index()  
decade_avg['Change'] = decade_avg[temp_col].diff()

fig4 = px.bar(  
	decade_avg,  
	x='Decade',  
	y=temp_col,  
	title=f'Average Temperature by Decade ({temp_symbol})',  
	labels={temp_col: f'Temperature ({temp_symbol})', 'Decade': 'Decade'},  
	text=temp_col,  
	color=temp_col,  
	color_continuous_scale='thermal'  
	)  
fig4.update_traces(texttemplate='%{text:.1f}', textposition='outside')  
fig4.update_layout(height=400)  
st.plotly_chart(fig4, width="stretch")
# Temperature anomaly heatmap  
st.subheader("Temperature Anomaly Heatmap (Deviation from Average)")  
  
overall_avg = df_filtered[temp_col].mean()  
monthly_anomaly = df_filtered.groupby(['Year', 'Month'])[temp_col].mean().reset_index()  
monthly_anomaly['Anomaly'] = monthly_anomaly[temp_col] - overall_avg  
  
pivot_anomaly = monthly_anomaly.pivot(index='Month', columns='Year', values='Anomaly')

fig5 = px.imshow(  
pivot_anomaly,  
labels=dict(x="Year", y="Month", color=f"Anomaly ({temp_symbol})"),  
title=f"Temperature Anomaly: Deviation from Average ({overall_avg:.1f}{temp_symbol})",  
color_continuous_scale='RdBu_r',  
aspect='auto'  
)  
fig5.update_layout(height=500)  
st.plotly_chart(fig5, width="stretch")

# TAB 3: Seasonal Patterns  
with tab3:  
	st.header("Seasonal Temperature Patterns")  
  
# Seasonal averages by region  
col1, col2 = st.columns(2)
with col1:  
	st.subheader("Average Temperature by Season")

seasonal_data = df_filtered.groupby(['Season', 'Region'])[temp_col].mean().reset_index()  
season_order = ['Winter', 'Spring', 'Summer', 'Fall']  
seasonal_data['Season'] = pd.Categorical(seasonal_data['Season'], categories=season_order, ordered=True)  
seasonal_data = seasonal_data.sort_values('Season')

fig6 = px.bar(  
seasonal_data,  
x='Season',  
y=temp_col,  
color='Region',  
barmode='group',  
title=f'Seasonal Temperature Comparison ({temp_symbol})',  
labels={temp_col: f'Temperature ({temp_symbol})'}  
)  
fig6.update_layout(height=450)  
st.plotly_chart(fig6, width="stretch")

with col2:  
	st.subheader("Temperature Variability by Season")

seasonal_std = df_filtered.groupby('Season')[temp_col].std().reset_index()  
seasonal_std.columns = ['Season', 'Variability']  
seasonal_std['Season'] = pd.Categorical(seasonal_std['Season'], categories=season_order, ordered=True)  
seasonal_std = seasonal_std.sort_values('Season')
fig7 = px.bar(  
seasonal_std,  
x='Season',  
y='Variability',  
title=f'Temperature Variability by Season (Std Dev {temp_symbol})',  
labels={'Variability': f'Standard Deviation ({temp_symbol})'},  
color='Variability',  
color_continuous_scale='Oranges'  
)  
fig7.update_layout(height=450)  
st.plotly_chart(fig7, width="stretch")

# Monthly temperature distribution  
st.subheader("Monthly Temperature Distribution")  
  
monthly_data = df_filtered.groupby('Month')[temp_col].apply(list).reset_index()  
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 

fig8 = go.Figure()  
for month in range(1, 13):  
	month_temps = df_filtered[df_filtered['Month'] == month][temp_col]  
fig8.add_trace(go.Box(  
y=month_temps,  
name=month_names[month-1],  
boxmean='sd'  
))  
  
fig8.update_layout(  
title=f'Temperature Distribution by Month ({temp_symbol})',  
yaxis_title=f'Temperature ({temp_symbol})',  
xaxis_title='Month',  
height=500,  
showlegend=False  
)  
st.plotly_chart(fig8, width="stretch")

# TAB 4: City Comparison  
with tab4:  
	st.header("City-Level Temperature Analysis")

# Get top cities by data volume  
top_cities = df_  
filtered['City'].value_counts().head(20).index.tolist()
# City selector  
selected_cities = st.multiselect(  
	"Select Cities to Compare (Top 20 by data volume):",  
	top_cities,  
	default=top_cities[:5]  
)
if selected_cities:  
	city_filtered = df_filtered[df_filtered['City'].isin(selected_cities)]

col1, col2 = st.columns(2)
with col1:  
	st.subheader("Temperature Trends by City")

city_yearly = city_filtered.groupby(['Year', 'City'])[temp_col].mean().reset_index()
fig9 = px.line(  
city_yearly,  
x='Year',  
y=temp_col,  
color='City',  
title=f'City Temperature Trends ({temp_symbol})',  
labels={temp_col: f'Temperature ({temp_symbol})'}  
)  
fig9.update_layout(height=450, hovermode='x unified')  
st.plotly_chart(fig9, width="stretch")

with col2:  
	st.subheader("City Temperature Statistics") 
city_stats = city_filtered.groupby('City')[temp_col].agg(['mean', 'min', 'max', 'std']).reset_index()  
city_stats.columns = ['City', 'Average', 'Min', 'Max', 'Std Dev']  
city_stats = city_stats.sort_values('Average', ascending=False)  
  
fig10 = go.Figure()  
fig10.add_trace(go.Bar(  
x=city_stats['City'],  
y=city_stats['Average'],  
name='Average',  
error_y=dict(type='data', array=city_stats['Std Dev'])  
))
fig10.update_layout(  
title=f'Average Temperature by City with Variability ({temp_symbol})',  
xaxis_title='City',  
yaxis_title=f'Temperature ({temp_symbol})',  
height=450  
) 
st.plotly_chart(fig10, width="stretch")

# Seasonal comparison for selected cities  
st.subheader("Seasonal Patterns by City")  
  
city_seasonal = city_filtered.groupby(['City', 'Season'])[temp_col].mean().reset_index()  
season_order = ['Winter', 'Spring', 'Summer', 'Fall']  
city_seasonal['Season'] = pd.Categorical(city_seasonal['Season'], categories=season_order, ordered=True)  
city_seasonal = city_seasonal.sort_values('Season')
fig11 = px.line(  
city_seasonal,  
x='Season',  
y=temp_col,  
color='City',  
markers=True,  
title=f'Seasonal Temperature Patterns by City ({temp_symbol})',  
labels={temp_col: f'Temperature ({temp_symbol})'}  
)  
fig11.update_layout(height=450)
st.plotly_chart(fig11, width="stretch")

# Display city statistics table  
st.subheader("Detailed City Statistics")  
st.dataframe(  
city_stats.style.format({  
'Average': '{:.1f}',  
'Min': '{:.1f}',  
'Max': '{:.1f}',  
'Std Dev': '{:.2f}'  
}).background_gradient(cmap='RdYlBu_r', subset=['Average']),
	 width="stretch")

# TAB 5: Geographic Analysis  
with tab5:  
	st.header("Geographic Temperature Distribution")

col1, col2 = st.columns(2)  
  
with col1:  
	st.subheader("Temperature by Region")

regional_stats = df_filtered.groupby('Region')[temp_col].agg(['mean', 'min', 'max']).reset_index()  
regional_stats.columns = ['Region', 'Average', 'Min', 'Max']  
regional_stats = regional_stats.sort_values('Average', ascending=False)  
  
fig12 = go.Figure()  
fig12.add_trace(go.Bar(  
x=regional_stats['Region'],  
y=regional_stats['Average'],  
name='Average',  
marker_color='indianred'  
))  
fig12.add_trace(go.Scatter(  
x=regional_stats['Region'],  
y=regional_stats['Max'],  
mode='markers',  
name='Max',  
marker=dict(size=10, color='red', symbol='triangle-up')  
))  
fig12.add_trace(go.Scatter(  
x=regional_stats['Region'],  
y=regional_stats['Min'],  
mode='markers',  
name='Min',  
marker=dict(size=10, color='blue', symbol='triangle-down')  
))  
  
fig12.update_layout(  
title=f'Temperature Range by Region ({temp_symbol})',  
xaxis_title='Region',  
yaxis_title=f'Temperature ({temp_symbol})',  
height=450  
) 
st.plotly_chart(fig12, width="stretch")

with col2:  
	st.subheader("Top 10 Hottest Countries (Avg)")

ountry_avg = df_filtered.groupby('Country')[temp_col].mean().reset_index()  
country_avg.columns = ['Country', 'Average Temperature']  
top_hot = country_avg.nlargest(10, 'Average Temperature')  
  
fig13 = px.bar(  
top_hot,  
x='Average Temperature',  
y='Country',  
orientation='h',  
title=f'Hottest Countries by Average Temperature ({temp_symbol})',  
color='Average Temperature',  
color_continuous_scale='Reds'  
)  
fig13.update_layout(height=450)
st.plotly_chart(fig13, width="stretch")

# Coldest countries  
st.subheader("Top 10 Coldest Countries (Avg)")  
  
top_cold = country_avg.nsmallest(10, 'Average Temperature')  
  
fig14 = px.bar(  
top_cold,  
x='Average Temperature',  
y='Country',  
orientation='h',  
title=f'Coldest Countries by Average Temperature ({temp_symbol})',  
color='Average Temperature',  
color_continuous_scale='Blues_r'  
)  
fig14.update_layout(height=450)
st.plotly_chart(fig14, width="stretch")

# Temperature distribution map (country level)  
st.subheader("Global Temperature Distribution")  
  
country_map_data = df_filtered.groupby('Country')[temp_col].mean().reset_index()  
country_map_data.columns = ['Country', 'Average Temperature']  
  
fig15 = px.choropleth(  
country_map_data,  
locations='Country',  
locationmode='country names',  
color='Average Temperature',  
title=f'Global Average Temperature by Country ({temp_symbol})',  
color_continuous_scale='RdYlBu_r',  
labels={'Average Temperature': f'Avg Temp ({temp_symbol})'}  
)  
fig15.update_layout(height=600)
st.plotly_chart(fig15, width="stretch")

from prophet import Prophet  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  
import warnings  
warnings.filterwarnings('ignore')
# Cache forecasting models  
@st.cache_resource  
def train_prophet_model(train_data, region_filter=None, country_filter=None):  
	"""Train Prophet model on filtered data"""  
df = load_data()  

# Train-test split  
split_idx = int(len(daily_data) * 0.8)  
train = daily_data.iloc[:split_idx].copy()  
test = daily_data.iloc[split_idx:].copy()

# Train model  
model = Prophet(  
yearly_seasonality=True,  
weekly_seasonality=True,  
daily_seasonality=False,  
changepoint_prior_scale=0.05  
)  
model.fit(train) 
return model, train, test, daily_data 
  
df = load_data()
# Title  
st.title("🌡️ Global Temperature Analysis & Forecasting Dashboard")  
st.markdown("""  
Explore historical temperature data and predict future trends using machine learning.  
**Features:** Historical analysis, Prophet forecasting, and model performance metrics.  
""")  

# Sidebar  
st.sidebar.header("🎛️ Configuration")  
  
# Temperature unit  
temp_unit = st.sidebar.radio("Temperature Unit:", ["Fahrenheit", "Celsius"])  
temp_col = 'AvgTemperature' if temp_unit == "Fahrenheit" else 'AvgTemperature_C'  
temp_symbol = '°F' if temp_unit == "Fahrenheit" else '°C'  
  
# Region filter  
regions = ['All'] + sorted(df['Region'].unique().tolist())  
selected_region = st.sidebar.selectbox("Select Region:", regions)  
  
# Country filter 
if selected_region != 'All': 
	countries = ['All'] + sorted(df[df['Region'] == selected_region]['Country'].unique().tolist())
else:
	countries = ['All'] + sorted(df['Country'].unique().tolist())  
	selected_country = st.sidebar.selectbox("Select Country:", countries)
# Forecast period  
forecast_days = st.sidebar.slider("Forecast Period (days):", 30, 730, 365)
# Apply filters to main dataframe  
df_filtered = df.copy()  
if selected_region != 'All':  
	df_filtered[df_filtered['Region'] == selected_region]  
if selected_country != 'All':  
	df_filtered = df_filtered[df_filtered['Country'] == selected_country]

# Tabs  
tab1, tab2, tab3, tab4 = st.tabs([  
"📊 Historical Analysis",  
"🔮 Temperature Forecasting",  
"📈 Model Performance",  
"🎯 Insights & Recommendations"  
]) 

# TAB 1: Historical Analysis  
with tab1:  
	st.header("Historical Temperature Trends")

# Key metrics  
col1, col2, col3, col4 = st.columns(4)  
  
with col1:  
	avg_temp = df_filtered[temp_col].mean()  
	st.metric("Average Temperature", f"{avg_temp:.1f}{temp_symbol}")

with col2:  
	max_temp = df_filtered[temp_col].max()  
	st.metric("Highest Recorded", f"{max_temp:.1f}{temp_symbol}")

with col3:  
	min_temp = df_filtered[temp_col].min()  
	st.metric("Lowest Recorded", f"{min_temp:.1f}{temp_symbol}")

with col4:  
	unique_cities = df_filtered['City'].nunique()  
	st.metric("Cities", f"{unique_cities:,}")
# Yearly trend  
st.subheader("Temperature Trend Over Time")  
yearly_data = df_filtered.groupby('Year')[temp_col].mean().reset_index() 

# Calculate trend line  
z = np.polyfit(yearly_data['Year'], yearly_data[temp_col], 1)  
p = np.poly1d(z)  
yearly_data['Trend'] = p(yearly_data['Year'])

fig16 = go.Figure()  
fig16.add_trace(go.Scatter(  
x=yearly_data['Year'],  
y=yearly_data[temp_col],  
mode='lines+markers',  
name='Actual',  
line=dict(color='royalblue', width=2)  
))  
fig16.add_trace(go.Scatter(  
x=yearly_data['Year'],  
y=yearly_data['Trend'],  
mode='lines',  
name='Trend Line',  
line=dict(color='red', width=3, dash='dash')  
))
fig16.update_layout(  
title=f'Average Temperature Trend ({temp_symbol})',  
xaxis_title='Year',  
yaxis_title=f'Temperature ({temp_symbol})',  
height=500,  
hovermode='x unified'  
)  
st.plotly_chart(fig16, width="stretch") 

# Warming rate  
warming_rate = z[0]  
warming_per_decade = warming_rate * 10  
  
st.info(f"📊  **Warming Rate:**  {warming_rate:.4f}{temp_symbol}/year ({warming_per_decade:.3f}{temp_symbol}/decade)")

# Monthly patterns  
col1, col2 = st.columns(2)
with col1:  
	st.subheader("Monthly Temperature Patterns")

monthly_avg = df_filtered.groupby('Month')[temp_col].mean().reset_index()  
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',  
'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']  
monthly_avg['Month_Name'] = monthly_avg['Month'].apply(lambda x: month_names[x-1])

fig17 = px.line(  
monthly_avg,  
x='Month_Name',  
y=temp_col,  
markers=True,  
title=f'Average Temperature by Month ({temp_symbol})'  
)  
fig17.update_layout(height=400)  
st.plotly_chart(fig17, width="stretch")

with col2:  
	st.subheader("Temperature Variability")  
  
monthly_std = df_filtered.groupby('Month')[temp_col].std().reset_index()  
monthly_std['Month_Name'] = monthly_std['Month'].apply(lambda x: month_names[x-1])  
  
fig18 = px.bar(  
monthly_std,  
x='Month_Name',  
y=temp_col,  
title=f'Temperature Variability by Month (Std Dev)',  
color=temp_col,  
color_continuous_scale='Reds'  
)  
fig18.update_layout(height=400)  
st.plotly_chart(fig18, width="stretch")

# TAB 2: Forecasting  
with tab2:  
	st.header("🔮 Temperature Forecasting")  
  
st.markdown("""  
Using **Facebook Prophet** - a time series forecasting model that handles seasonality,  
trends, and holidays automatically.  
""")  

with st.spinner("Training forecasting model... This may take a moment."): 
	# Train model  
	model, train_data, test_data, full_data = train_prophet_model(  
	df_filtered,  
	selected_region if selected_region != 'All' else None,  
	selected_country if selected_country != 'All' else None  
	)

# Make predictions  
future = model.make_future_dataframe(periods=forecast_days, freq='D')  
forecast = model.predict(future) 

# Split into historical and future  
last_date = full_data['ds'].max()  
forecast_historical = forecast[forecast['ds'] <= last_date]  
forecast_future = forecast[forecast['ds'] > last_date]  

# Visualization  
st.subheader("Forecast Visualization") 
fig19 = go.Figure()  
  
# Historical actual data  
fig19.add_trace(go.Scatter(  
x=full_data['ds'],  
y=full_data['y'],  
mode='lines',  
name='Historical Data',  
line=dict(color='blue', width=1),  
opacity=0.6  
))  
  
# Forecast  
fig19.add_trace(go.Scatter(  
x=forecast['ds'],  
y=forecast['yhat'],  
mode='lines',  
name='Forecast',  
line=dict(color='red', width=2)  
))  
  
# Confidence interval  
fig19.add_trace(go.Scatter(  
x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],  
y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],  
fill='toself',  
fillcolor='rgba(255,0,0,0.1)',  
line=dict(color='rgba(255,255,255,0)'),  
name='Confidence Interval',  
showlegend=True  
))  
  
# Add vertical line at forecast start  
fig19.add_vline(  
x=last_date.timestamp() * 1000,  
line_dash="dash",  
line_color="green",  
annotation_text="Forecast Start"  
)  
  
fig19.update_layout(  
title=f'Temperature Forecast ({forecast_days} days ahead)',  
xaxis_title='Date',  
yaxis_title='Temperature (°C)',  
height=600,  
hovermode='x unified'  
)  
  
st.plotly_chart(fig19, width="stretch")

# Forecast statistics  
st.subheader("Forecast Summary")  
  
col1, col2, col3 = st.columns(3)
with col1:  
	current_avg = full_data['y'].tail(30).mean()  
	st.metric("Current Avg (Last 30 days)", f"{current_avg:.2f}°C")

with col2:  
	forecast_avg = forecast_future['yhat'].mean()  
	change = forecast_avg - current_avg  
	st.metric(  
	f"Predicted Avg (Next {forecast_days} days)",  
	f"{forecast_avg:.2f}°C",  
	f"{change:+.2f}°C"  
)  
  
with col3:  
	forecast_max = forecast_future['yhat'].max()  
	st.metric("Predicted Maximum", f"{forecast_max:.2f}°C")  
  
# Trend components  
st.subheader("Forecast Components Analysis")
# Create component plots  
fig_components = model.plot_components(forecast)  
st.pyplot(fig_components)  

# Download forecast data  
st.subheader("📥 Download Forecast Data")  
  
forecast_export = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()  
forecast_export.columns = ['Date', 'Predicted_Temp_C', 'Lower_Bound', 'Upper_Bound']  
forecast_export['Predicted_Temp_F'] = forecast_export['Predicted_Temp_C'] * 9/5 + 32 

csv_forecast = forecast_export.to_csv(index=False).encode('utf-8')

st.download_button(  
label="Download Forecast as CSV",  
data=csv_forecast,  
file_name=f'temperature_forecast_{datetime.now().strftime("%Y%m%d")}.csv',  
mime='text/csv'  
) 

# TAB 3: Model Performance  
with tab3:  
	st.header("📈 Model Performance Evaluation") 
	with st.spinner("Evaluating model performance..."):  
		# Get model and data  
		model, train_data, test_data, full_data = train_prophet_model(  
		df_filtered,  
		selected_region if selected_region != 'All' else None,  
		selected_country if selected_country != 'All' else None  
)
# Calculate metrics  
mae = mean_absolute_error(test_merged['y'], test_merged['yhat'])  
rmse = np.sqrt(mean_squared_error(test_merged['y'], test_merged['yhat']))  
r2 = r2_score(test_merged['y'], test_merged['yhat'])  
mape = np.mean(np.abs((test_merged['y'] - test_merged['yhat']) / test_merged['y'])) * 100 


# Generate predictions on test set  
test_forecast = model.predict(test_data[['ds']])  
test_merged = test_data.merge(test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')  

# Display metrics  
st.subheader("Performance Metrics") 

col1, col2, col3, col4 = st.columns(4)  
  
with col1:  
	st.metric("MAE (Mean Absolute Error)", f"{mae:.3f}°C")  
  
with col2:  
	st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.3f}°C")  
  
with col3:  
	st.metric("R² Score", f"{r2:.3f}")  
  
with col4:  
	st.metric("MAPE", f"{mape:.2f}%")  
  
# Interpretation  
st.info(f"""  
**Model Interpretation:**  
- **MAE**: On average, predictions are off by {mae:.2f}°C  
- **RMSE**: Root mean squared error of {rmse:.2f}°C (penalizes larger errors more)  
- **R² Score**: Model explains {r2*100:.1f}% of temperature variance  
- **MAPE**: Average percentage error of {mape:.2f}%  
  
{'✅ **Good performance!** R² > 0.7 indicates strong predictive power.' if r2 > 0.7 else '⚠️ **Moderate performance.** Consider using more data or different filters.'}  
""")  
  
# Actual vs Predicted plot  
st.subheader("Actual vs Predicted (Test Set)")  
  
col1, col2 = st.columns(2)  
  
with col1:  
	# Time series comparison  
	fig5 = go.Figure()  
  
	fig5.add_trace(go.Scatter(  
x=test_merged['ds'],  
y=test_merged['y'],  
mode='lines',  
name='Actual',  
line=dict(color='blue', width=2)  
))  
  
	fig5.add_trace(go.Scatter(  
x=test_merged['ds'],  
y=test_merged['yhat'],  
mode='lines',  
name='Predicted',  
line=dict(color='red', width=2)  
))  
  
	fig5.add_trace(go.Scatter(  
x=test_merged['ds'].tolist() + test_merged['ds'].tolist()[::-1],  
y=test_merged['yhat_upper'].tolist() + test_merged['yhat_lower'].tolist()[::-1],  
fill='toself',  
fillcolor='rgba(255,0,0,0.1)',  
line=dict(color='rgba(255,255,255,0)'),  
name='95% Confidence',  
showlegend=True  
))  
  
	fig5.update_layout(  
title='Actual vs Predicted Temperature (Test Period)',  
xaxis_title='Date',  
yaxis_title='Temperature (°C)',  
height=450,  
hovermode='x unified'  
)  
  
	st.plotly_chart(fig5, use_container_width=True)  
  
with col2:  
	# Scatter plot  
	fig6 = px.scatter(  
test_merged,  
x='y',  
y='yhat',  
labels={'y': 'Actual Temperature (°C)', 'yhat': 'Predicted Temperature (°C)'},  
title='Actual vs Predicted Scatter Plot',  
trendline='ols'  
)  
  
	# Add perfect prediction line  
	min_val = min(test_merged['y'].min(), test_merged['yhat'].min())  
	max_val = max(test_merged['y'].max(), test_merged['yhat'].max())  
  
	fig6.add_trace(go.Scatter(  
x=[min_val, max_val],  
y=[min_val, max_val],  
mode='lines',  
name='Perfect Prediction',  
line=dict(color='green', dash='dash')  
))  
  
	fig6.update_layout(height=450)  
	st.plotly_chart(fig6, use_container_width=True)  
  
# Residual analysis  
st.subheader("Residual Analysis")  
  
test_merged['residual'] = test_merged['y'] - test_merged['yhat']  
  
col1, col2 = st.columns(2)  
  
with col1:  
# Residual distribution  
fig7 = px.histogram(  
test_merged,  
x='residual',  
nbins  
=50,  
title='Distribution of Prediction Errors',  
labels={'residual': 'Residual (Actual - Predicted) °C'},  
color_discrete_sequence=['indianred']  
)  
fig7.add_vline(x=0, line_dash="dash", line_color="green")  
fig7.update_layout(height=400)  
st.plotly_chart(fig7, use_container_width=True)  
  
with col2:  
# Residuals over time  
fig8 = px.scatter(  
test_merged,  
x='ds',  
y='residual',  
title='Residuals Over Time',  
labels={'ds': 'Date', 'residual': 'Residual (°C)'}  
)  
fig8.add_hline(y=0, line_dash="dash", line_color="green")  
fig8.update_layout(height=400)  
st.plotly_chart(fig8, use_container_width=True)  
  
# Cross-validation results  
st.subheader("Model Stability: Cross-Validation")  
  
with st.expander("ℹ️ About Cross-Validation"):  
st.markdown("""  
Cross-validation tests how well the model performs across different time periods.  
We split the data into multiple train/test windows to ensure consistent performance.  
""")  
  
from prophet.diagnostics import cross_validation, performance_metrics  
  
with st.spinner("Running cross-validation (this may take a minute)..."):  
# Perform cross-validation  
initial_days = int(len(train_data) * 0.5)  
cv_results = cross_validation(  
model,  
initial=f'{initial_days} days',  
period='90 days',  
horizon='180 days'  
)  
  
# Calculate performance metrics  
cv_metrics = performance_metrics(cv_results)  
  
# Display CV metrics  
st.write("**Cross-Validation Performance:**")  
st.dataframe(  
cv_metrics[['horizon', 'mae', 'rmse', 'mape']].describe().style.format("{:.3f}"),  
use_container_width=True  
)  
  
# Plot CV metrics  
fig9 = px.line(  
cv_metrics,  
x='horizon',  
y=['mae', 'rmse'],  
title='Error Metrics vs Forecast Horizon',  
labels={'value': 'Error (°C)', 'horizon': 'Forecast Horizon'}  
)  
fig9.update_layout(height=400)  
st.plotly_chart(fig9, use_container_width=True)  
  
except Exception as e:  
st.error(f"Error in model evaluation: {str(e)}")  
st.info("Please ensure sufficient data is available for the selected filters.")  
  
# TAB 4: Insights & Recommendations  
with tab4:  
st.header("🎯 Key Insights & Recommendations")  
  
# Calculate insights  
yearly_data = df_filtered.groupby('Year')[temp_col].mean().reset_index()  
  
if len(yearly_data) > 1:  
# Trend analysis  
z = np.polyfit(yearly_data['Year'], yearly_data[temp_col], 1)  
warming_rate = z[0]  
  
# Convert to Celsius for consistent reporting  
if temp_unit == "Fahrenheit":  
warming_rate_c = warming_rate * 5/9  
else:  
warming_rate_c = warming_rate  
  
st.subheader("🌍 Climate Trends")  
  
col1, col2 = st.columns(2)  
  
with col1:  
st.markdown(f"""  
**Historical Trend Analysis:**  
  
- 📈 **Warming Rate:** {warming_rate_c:.4f}°C per year
- 📊 **Decade Change:** {warming_rate_c * 10:.3f}°C per decade  
- 📅 **Data Coverage:** {yearly_data['Year'].min()} - {yearly_data['Year'].max()}  
- 🌡️ **Total Change:** {warming_rate_c * len(yearly_data):.2f}°C over {len(yearly_data)} years  
  
{'🔴 **Alert:** Significant warming trend detected!' if warming_rate_c > 0.02 else '🟢 Temperature relatively stable.'}  
""")  
  
with col2:  
# Seasonal insights  
seasonal_avg = df_filtered.groupby('Season')[temp_col].mean().sort_values(ascending=False)  
  
st.markdown(f"""  
**Seasonal Patterns:**  
  
- 🔥 **Warmest Season:** {seasonal_avg.index[0]} ({seasonal_avg.values[0]:.1f}{temp_symbol})  
- ❄️ **Coldest Season:** {seasonal_avg.index[-1]} ({seasonal_avg.values[-1]:.1f}{temp_symbol})  
- 📏 **Seasonal Range:** {seasonal_avg.values[0] - seasonal_avg.values[-1]:.1f}{temp_symbol}  
- 🔄 **Variability:** {'High' if seasonal_avg.std() > 10 else 'Moderate' if seasonal_avg.std() > 5 else 'Low'}  
""")  
  
# Future projections  
st.subheader("🔮 Future Projections")  
  
st.markdown(f"""  
**Based on current trends, by 2030:**  
  
- Expected temperature increase: **{warming_rate_c * (2030 - yearly_data['Year'].max()):.2f}°C**  
- Projected average temperature: **{yearly_data[temp_col].iloc[-1] + warming_rate * (2030 - yearly_data['Year'].max()):.1f}{temp_symbol}**  
  
**By 2050:**  
  
- Expected temperature increase: **{warming_rate_c * (2050 - yearly_data['Year'].max()):.2f}°C**  
- Projected average temperature: **{yearly_data[temp_col].iloc[-1] + warming_rate * (2050 - yearly_data['Year'].max()):.1f}{temp_symbol}**  
  
⚠️ *Note: These are linear projections based on historical trends and do not account for policy changes or climate interventions.*  
""")  
  
# Recommendations  
st.subheader("💡 Actionable Recommendations")  
  
if warming_rate_c > 0.02:  
st.warning("""  
**High Warming Rate Detected - Urgent Actions Recommended:**  
  
1. **Monitoring:** Implement continuous temperature monitoring systems  
2. **Infrastructure:** Prepare infrastructure for higher temperature extremes  
3. **Planning:** Develop heat action plans for vulnerable populations  
4. **Green Spaces:** Increase urban greenery to mitigate heat island effects  
5. **Energy:** Transition to renewable energy sources to reduce emissions  
6. **Adaptation:** Develop climate adaptation strategies for agriculture and water resources  
""")  
elif warming_rate_c > 0.01:  
st.info("""  
**Moderate Warming Trend - Proactive Measures Suggested:**  
  
1. **Data Collection:** Continue detailed temperature monitoring  
2. **Public Awareness:** Educate communities about climate change impacts  
3. **Sustainable Practices:** Encourage energy efficiency and sustainable practices  
4. **Preparedness:** Develop contingency plans for extreme weather events  
5. **Research:** Support climate research and modeling initiatives  
""")  
else:  
st.success("""  
**Stable Temperature Trends - Maintain Current Practices:**  
  
1. **Vigilance:** Continue monitoring for any changes in patterns  
2. **Best Practices:** Maintain current environmental protection measures  
3. **Documentation:** Keep detailed records for long-term trend analysis  
4. **Prevention:** Implement preventive measures to avoid future warming  
5. **Collaboration:** Share data and insights with climate research communities  
""")  
  
# Statistical insights  
st.subheader("📊 Statistical Summary")  
  
# Extreme events analysis  
temp_mean = df_filtered[temp_col].mean()  
temp_std = df_filtered[temp_col].std()  
  
extreme_hot = df_filtered[df_filtered[temp_col] > temp_mean + 2*temp_std]  
extreme_cold = df_filtered[df_filtered[temp_col] < temp_mean - 2*temp_std]  
  
col1, col2, col3 = st.columns(3)  
  
with col1:  
st.metric(  
"Extreme Heat Events",  
f"{len(extreme_hot):,}",  
f"{len(extreme_hot)/len(df_filtered)*100:.2f}% of records"  
)  
  
with col2:  
st.metric(  
"Extreme Cold Events",  
f"{len(extreme_cold):,}",  
f"{len(extreme_cold)/len(df_filtered)*100:.2f}% of records"  
)  
  
with col3:  
recent_years = df_filtered[df_filtered['Year'] >= yearly_data['Year'].max() - 5]  
trend_direction = "↑ Warming" if warming_rate_c > 0 else "↓ Cooling" if warming_rate_c < 0 else "→ Stable"  
st.metric(  
"5-Year Trend",  
trend_direction,  
f"{warming_rate_c * 5:.3f}°C"  
)  
  
# Comparative analysis  
if selected_region != 'All' or selected_country != 'All':  
st.subheader("🌐 Comparative Context")  
  
# Global comparison  
global_avg = df[temp_col].mean()  
local_avg = df_filtered[temp_col].mean()  
difference = local_avg - global_avg  
  
st.markdown(f"""  
**How does your selection compare to global averages?**  
  
- **Global Average:** {global_avg:.2f}{temp_symbol}  
- **Selected Region/Country:** {local_avg:.2f}{temp_symbol}  
- **Difference:** {difference:+.2f}{temp_symbol} ({'warmer' if difference > 0 else 'cooler'} than global average)  
  
{'🔥 This region is significantly warmer than the global average.' if difference > 5 else '❄️ This region is significantly cooler than the global average.' if difference < -5 else '🌡️ This region is close to the global average.'}  
""")  
  
# Data quality assessment  
st.subheader("📋 Data Quality Assessment")  
  
total_possible_days = (df_filtered['Date'].max() - df_filtered['Date'].min()).days + 1  
actual_records = len(df_filtered)  
coverage = (actual_records / total_possible_days) * 100 if total_possible_days > 0 else 0  
  
missing_data = df_filtered.isnull().sum().sum()  
  
col1, col2 = st.columns(2)  
  
with col1:  
st.markdown(f"""  
**Data Coverage:**  
  
- Total Records: {actual_records:,}  
- Date Range: {total_possible_days:,} days  
- Coverage: {coverage:.1f}%  
- Missing Values: {missing_data}  
  
{'✅ Excellent data coverage' if coverage > 80 else '⚠️ Moderate data coverage - results may vary' if coverage > 50 else '❌ Low data coverage - use results with caution'}  
""")  
  
with col2:  
# Data freshnesslatest_date = df_filtered['Date'].max()  
days_old = (datetime.now() - latest_date).days  
  
st.markdown(f"""  
**Data Freshness:**  
  
- Latest Record: {latest_date.strftime('%Y-%m-%d')}  
- Days Since Update: {days_old}  
- Data Status: {'🟢 Recent' if days_old < 365 else '🟡 Moderately Old' if days_old < 730 else '🔴 Outdated'}  
  
{'✅ Data is current and reliable' if days_old < 365 else '⚠️ Consider updating with more recent data' if days_old < 1095 else '❌ Data may be too old for current analysis'}  
""")  
  
# Actionable business/research insights  
st.subheader("🎯 Domain-Specific Applications")  
  
tab_a, tab_b, tab_c, tab_d = st.tabs([  
"🏢 Business",  
"🔬 Research",  
"🏛️ Policy",  
"👥 Public Health"  
])  
  
with tab_a:  
st.markdown("""  
**Business Applications:**  
  
**Energy Sector:**  
- Forecast cooling/heating demand based on temperature predictions  
- Optimize energy production and distribution  
- Plan for renewable energy capacity (solar/wind correlates with temperature)  
  
**Agriculture:**  
- Plan crop cycles based on seasonal forecasts  
- Optimize irrigation schedules  
- Assess climate risk for crop insurance  
  
**Retail & Supply Chain:**  
- Forecast seasonal product demand (ice cream, heaters, etc.)  
- Optimize inventory management  
- Plan promotional campaigns around weather patterns  
  
**Real Estate & Construction:**  
- Assess climate risk for property investments  
- Plan construction schedules around weather patterns  
- Design climate-resilient buildings  
""")  
  
with tab_b:  
st.markdown("""  
**Research Applications:**  
  
**Climate Science:**  
- Validate climate models with historical data  
- Study regional climate change patterns  
- Identify climate feedback mechanisms  
  
**Ecology & Biology:**  
- Study impact on species migration patterns  
- Analyze ecosystem changes  
- Research phenological shifts (timing of biological events)  
  
**Urban Planning:**  
- Study urban heat island effects  
- Design climate-adaptive cities  
- Research heat mitigation strategies  
  
**Data Science:**  
- Benchmark forecasting algorithms  
- Develop improved prediction models  
- Test machine learning approaches on real-world data  
""")  
  
with tab_c:  
st.markdown(f"""  
**Policy Recommendations:**  
  
**Immediate Actions (0-2 years):**  
- Establish temperature monitoring networks in underserved areas  
- Develop heat emergency response protocols  
- Create public awareness campaigns on climate adaptation  
  
**Medium-term (2-5 years):**  
- Implement urban greening initiatives (target: {warming_rate_c * 5:.2f}°C reduction)  
- Update building codes for climate resilience  
- Invest in cooling infrastructure for vulnerable populations  
  
**Long-term (5+ years):**  
- Transition to 100% renewable energy  
- Develop comprehensive climate adaptation strategy  
- Establish climate monitoring and early warning systems  
  
**Funding Priorities:**  
- Allocate resources based on warming rate severity  
- Prioritize regions showing >0.03°C/year increase  
- Support climate research and monitoring infrastructure  
""")  
  
with tab_d:  
st.markdown("""  
**Public Health Implications:**  
  
**Heat-Related Risks:**  
- Heat exhaustion and heat stroke incidents may increase  
- Vulnerable populations: elderly, children, outdoor workers  
- Recommended: Establish cooling centers and heat alert systems  
  
**Vector-Borne Diseases:**  
- Warmer temperatures may expand mosquito habitats  
- Monitor for dengue, malaria, Zika virus spread  
- Implement disease surveillance programs  
  
**Air Quality:**  
- Higher temperatures increase ground-level ozone formation  
- Exacerbates respiratory conditions (asthma, COPD)  
- Recommended: Air quality monitoring and public alerts  
  
**Mental Health:**  
- Temperature extremes linked to increased anxiety and aggression  
- Climate anxiety affecting young populations  
- Recommended: Community support programs and mental health services  
  
**Healthcare Infrastructure:**  
- Prepare hospitals for heat-related emergency spikes  
- Stock heat illness medications and supplies  
- Train healthcare workers on climate-related health issues  
""")  
  
else:  
st.warning("Insufficient data for comprehensive insights. Please adjust filters or check data availability.")  
  
# Export comprehensive report  
st.subheader("📄 Generate Comprehensive Report")  
  
if st.button("Generate PDF Report Summary"):  
st.info("PDF generation feature coming soon! For now, you can:")  
st.markdown("""  
- Use your browser's Print to PDF function (Ctrl/Cmd + P)  
- Download data from the Historical Analysis tab  
- Download forecasts from the Forecasting tab  
- Screenshot visualizations for presentations  
""") 
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
