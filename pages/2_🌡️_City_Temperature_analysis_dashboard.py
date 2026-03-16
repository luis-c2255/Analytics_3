import streamlit as st  
import pandas as pd 
import numpy as np 
import plotly.express as px  
import plotly.graph_objects as go  
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("City Temperature Analysis Dashboard", "🌡")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")
    
# Load data
@st.cache_data(show_spinner="Loading & cleaning data...")
def load_and_clean():
    df = pd.read_csv("city_temperature.csv")

    # Basic cleaning
    df = df[(df["AvgTemperature"] >= -100) & (df["AvgTemperature"] <= 150)]
    df = df.drop_duplicates()

    # Date
    df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Celsius
    df["AvgTemperature_C"] = (df["AvgTemperature"] - 32) * 5 / 9

    # season
    df["Season"] = df["Month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    })
    return df

@st.cache_data
def precompute(df: pd.DataFrame):
    yearly = df.groupby("Year").agg(
        AvgTemperature=("AvgTemperature", "mean"),
        AvgTemperature_C=("AvgTemperature_C", "mean")
    ).reset_index()

    regional_yearly = df.groupby(["Year", "Region"]).agg(
        AvgTemperature=("AvgTemperature", "mean"),
        AvgTemperature_C=("AvgTemperature_C", "mean")
    ).reset_index()

    seasonal = df.groupby(["Season", "Region"]).agg(
        AvgTemperature=("AvgTemperature", "mean"),
        AvgTemperature_C=("AvgTemperature_C", "mean")
    ).reset_index()

    monthly = df.groupby("Month").agg(
        AvgTemperature=("AvgTemperature", "mean"),
        AvgTemperature_C=("AvgTemperature_C", "mean")
    ).reset_index()

    country_avg = df.groupby("Country").agg(
        AvgTemperature=("AvgTemperature", "mean"),
        AvgTemperature_C=("AvgTemperature_C", "mean")
    ).reset_index()

    regional_stats = df.groupby("Region").agg(
        AvgTemperature=("AvgTemperature", "mean"),
        AvgTemperature_C=("AvgTemperature_C", "mean"),
        Min_C=("AvgTemperature_C", "min"),
        Max_C=("AvgTemperature_C", "max")
    ).reset_index()

    city_counts = df["City"].value_counts().reset_index()
    city_counts.columns = ["City", "Count"]

    return {
        "yearly": yearly,
        "regional_yearly": regional_yearly,
        "seasonal": seasonal,
        "monthly": monthly,
        "country_avg": country_avg,
        "regional_stats": regional_stats,
        "city_counts": city_counts
    }
@st.cache_data
def filter_df(df, region, country, year_range):
    df2 = df
    if region != "All":
        df2 = df2[df2["Region"] == region]
    if country != "All":
        df2 = df2[df2["Country"] == country]
    df2 = df2[(df2["Year"] >= year_range[0]) & (df2["Year"] <= year_range[1])]
    return df2

@st.cache_resource
def train_prophet(df_filtered, temp_col):
    daily = df_filtered.groupby("Date")[temp_col].mean().reset_index()
    daily = daily.rename(columns={"Date": "ds", temp_col: "y"})
    daily = daily.sort_values("ds")

    if len(daily) < 50:
        return None, daily

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(daily)
    return model, daily

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
df = load_and_clean()
agg = precompute(df)


# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------
st.sidebar.header("🎛️ Filters")

temp_unit = st.sidebar.radio("Temperature Unit", ["Fahrenheit", "Celsius"])
temp_col = "AvgTemperature" if temp_unit == "Fahrenheit" else "AvgTemperature_C"
temp_symbol = "°F" if temp_unit == "Fahrenheit" else "°C"

regions = ["All"] + sorted(df["Region"].unique().tolist())
selected_region = st.sidebar.selectbox("Region", regions)

if selected_region != "All":
    countries = ["All"] + sorted(df[df["Region"] == selected_region]["Country"].unique().tolist())
else:
    countries = ["All"] + sorted(df["Country"].unique().tolist())
selected_country = st.sidebar.selectbox("Country", countries)

year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

forecast_days = st.sidebar.slider("Forecast Horizon (days)", 30, 730, 365)

df_filtered = filter_df(df, selected_region, selected_country, year_range)

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

# Key Metrics  
st.header("📊 :red[Key Metrics]", divider='red')  

col1, col2, col3, col4 = st.columns(4) 

with col1:
    avg_temp = df_filtered[temp_col].mean()
    st.markdown(
        Components.metric_card(
            title="Average Temperature",
            value=f"{avg_temp:.1f}{temp_symbol}",
            delta="🔥",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    yearly_filtered = df_filtered.groupby("Year")[temp_col].mean().reset_index()
    warming_rate = np.nan
    if len(yearly_filtered) > 1:
        z = np.polyfit(yearly_filtered["Year"], yearly_filtered[temp_col], 1)
        warming_rate = z[0]
    st.markdown(
        Components.metric_card(
            title="Warming Rate / Decade",
            value=f"{warming_rate*10:.3f}{temp_symbol}/decade",
            delta="📈",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    regions_count = df_filtered["Region"].nunique()
    st.markdown(
        Components.metric_card(
            title="Regions Analyzed",
            value=f"{regions_count}",
            delta="🌍",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    cities_count = df_filtered["City"].nunique()
    st.markdown(
        Components.metric_card(
            title="Cities Included",
            value=f"{cities_count}",
            delta="🏙️",
            card_type="info"
        ), unsafe_allow_html=True
    )

st.markdown("   ")

# -------------------------------------------------------------------
# GLOBAL TEMPERATURE OVERVIEW
# -------------------------------------------------------------------
st.subheader("🌍 :blue[Global Climate Trends]", divider="blue")

# Yearly average trend  
yearly = df_filtered.groupby("Year")[temp_col].mean().reset_index()
yearly = yearly.sort_values("Year")

fig_year = go.Figure()
fig_year.add_trace(go.Scatter(
        x=yearly["Year"],
        y=yearly[temp_col],
        mode="lines+markers",
        name="Yearly Avg",
        line=dict(color="#4c78a8", width=2)
    ))
if len(yearly) > 4:
        yearly["MA_5"] = yearly[temp_col].rolling(window=5, center=True).mean()
        fig_year.add_trace(go.Scatter(
            x=yearly["Year"],
            y=yearly["MA_5"],
            mode="lines",
            name="5-Year Moving Avg",
            line=dict(color="#f58518", width=3, dash="dash")
        ))
if len(yearly) > 1:
        z = np.polyfit(yearly["Year"], yearly[temp_col], 1)
        p = np.poly1d(z)
        yearly["Trend"] = p(yearly["Year"])
        fig_year.add_trace(go.Scatter(
            x=yearly["Year"],
            y=yearly["Trend"],
            mode="lines",
            name="Trend Line",
            line=dict(color="#e45756", width=2, dash="dot")
        ))
fig_year.update_layout(
        title=f"Global Average Temperature Over Time ({temp_symbol})",
        xaxis_title="Year",
        yaxis_title=f"Temperature ({temp_symbol})",
        hovermode="x unified",
        height=420
    )
st.plotly_chart(fig_year, width="stretch")

# Regional trends comparison  
regional_yearly = df_filtered.groupby(["Year", "Region"])[temp_col].mean().reset_index()

if not regional_yearly.empty:
        fig_reg = px.line(
            regional_yearly,
            x="Year",
            y=temp_col,
            color="Region",
            title=f"Regional Temperature Trends ({temp_symbol})"
        )
        fig_reg.update_layout(height=420, hovermode="x unified")
        st.plotly_chart(fig_reg, width="stretch")
else:
        st.info("No regional data available for current filters.")

# Decade summary & anomaly heatmap
df_dec = df_filtered.copy()
df_dec["Decade"] = (df_dec["Year"] // 10) * 10
decade_avg = df_dec.groupby("Decade")[temp_col].mean().reset_index()
decade_avg["Change"] = decade_avg[temp_col].diff()

fig_dec = px.bar(
        decade_avg,
        x="Decade",
        y=temp_col,
        text=temp_col,
        title=f"Average Temperature by Decade ({temp_symbol})",
        labels={temp_col: f"Temperature ({temp_symbol})"}
    )
fig_dec.update_traces(texttemplate="%{text:.1f}", textposition="outside")
fig_dec.update_layout(height=380)
st.plotly_chart(fig_dec, width="stretch")

overall_avg = df_filtered[temp_col].mean()
monthly_anomaly = df_filtered.groupby(["Year", "Month"])[temp_col].mean().reset_index()
monthly_anomaly["Anomaly"] = monthly_anomaly[temp_col] - overall_avg
if not monthly_anomaly.empty:
    pivot_anom = monthly_anomaly.pivot(index="Month", columns="Year", values="Anomaly")
    fig_anom = px.imshow(
        pivot_anom,
        labels=dict(x="Year", y="Month", color=f"Anomaly ({temp_symbol})"),
        title=f"Temperature Anomaly Heatmap (vs {overall_avg:.1f}{temp_symbol})",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    fig_anom.update_layout(height=380)
    st.plotly_chart(fig_anom, width="stretch")
else:
    st.info("Not enough data for anomaly heatmap.")
st.markdown("   ")

# -------------------------------------------------------------------
# SEASONAL & MONTHLY DYNAMICS
# -------------------------------------------------------------------
st.subheader("🗓️ :green[Seasonal & Monthly Patterns]", divider="green")
seasonal_data = df_filtered.groupby(["Season", "Region"])[temp_col].mean().reset_index()
season_order = ["Winter", "Spring", "Summer", "Fall"]
if not seasonal_data.empty:
    seasonal_data["Season"] = pd.Categorical(seasonal_data["Season"], categories=season_order, ordered=True)
    seasonal_data = seasonal_data.sort_values("Season")
    fig_season = px.bar(
            seasonal_data,
            x="Season",
            y=temp_col,
            color="Region",
            barmode="group",
            title=f"Seasonal Temperature by Region ({temp_symbol})",
            labels={temp_col: f"Temperature ({temp_symbol})"}
    )
    fig_season.update_layout(height=420)
    st.plotly_chart(fig_season, width="stretch")
else:
    st.info("No seasonal data available for current filters.")

st.markdown("   ")
seasonal_std = df_filtered.groupby("Season")[temp_col].std().reset_index()
if not seasonal_std.empty:
    seasonal_std["Season"] = pd.Categorical(seasonal_std["Season"], categories=season_order, ordered=True)
    seasonal_std = seasonal_std.sort_values("Season")
    fig_var = px.bar(
        seasonal_std,
        x="Season",
        y=temp_col,
        title=f"Temperature Variability by Season (Std Dev {temp_symbol})",
        labels={temp_col: f"Std Dev ({temp_symbol})"},
        color=temp_col,
        color_continuous_scale="Oranges"
    )
    fig_var.update_layout(height=420)
    st.plotly_chart(fig_var, width="stretch")
else:
    st.info("No variability data available for current filters.")

# Monthly distribution
st.markdown("#### :green[Monthly Temperature Distribution]")
monthly_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig_box = go.Figure()
for m in range(1, 13):
    month_temps = df_filtered[df_filtered["Month"] == m][temp_col]
    if not month_temps.empty:
        fig_box.add_trace(go.Box(
            y=month_temps,
            name=monthly_names[m - 1],
            boxmean="sd"
        ))
fig_box.update_layout(
    title=f"Temperature Distribution by Month ({temp_symbol})",
    yaxis_title=f"Temperature ({temp_symbol})",
    xaxis_title="Month",
    height=420,
    showlegend=False
)
st.plotly_chart(fig_box, width="stretch")
st.markdown("   ")

# -------------------------------------------------------------------
# CITY-LEVEL INTELLIGENCE
# -------------------------------------------------------------------
st.subheader("🏙️ :violet[City Temperature Intelligence]", divider="violet")
top_cities = agg["city_counts"]["City"].head(20).tolist()
selected_cities = st.multiselect(
    "Select Cities to Compare (Top 20 by data volume):",
    top_cities,
    default=top_cities[:5] if len(top_cities) >= 5 else top_cities
)
if selected_cities:
    city_filtered = df_filtered[df_filtered["City"].isin(selected_cities)]

city_yearly = city_filtered.groupby(["Year", "City"])[temp_col].mean().reset_index()
if not city_yearly.empty:
    fig_city_trend = px.line(
        city_yearly,
        x="Year",
        y=temp_col,
        color="City",
        title=f"City Temperature Trends ({temp_symbol})",
        labels={temp_col: f"Temperature ({temp_symbol})"}
    )
    fig_city_trend.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig_city_trend, width="stretch")
else:
    st.info("No city trend data for selected cities.")

city_stats = city_filtered.groupby("City")[temp_col].agg(["mean", "min", "max", "std"]).reset_index()
city_stats.columns = ["City", "Average", "Min", "Max", "Std Dev"]
if not city_stats.empty:
    fig_city_stats = go.Figure()
    fig_city_stats.add_trace(go.Bar(
        x=city_stats["City"],
        y=city_stats["Average"],
        error_y=dict(type="data", array=city_stats["Std Dev"]),
        name="Average"
    ))
    fig_city_stats.update_layout(
        title=f"Average Temperature by City with Variability ({temp_symbol})",
        xaxis_title="City",
        yaxis_title=f"Temperature ({temp_symbol})",
        height=420
    )
    st.plotly_chart(fig_city_stats, width="stretch")

st.markdown("#### :violet[Seasonal Patterns by City]")
city_seasonal = city_filtered.groupby(["City", "Season"])[temp_col].mean().reset_index()
if not city_seasonal.empty:
    city_seasonal["Season"] = pd.Categorical(city_seasonal["Season"], categories=season_order, ordered=True)
    city_seasonal = city_seasonal.sort_values("Season")
    fig_city_season = px.line(
        city_seasonal,
        x="Season",
        y=temp_col,
        color="City",
        markers=True,
        title=f"Seasonal Temperature Patterns by City ({temp_symbol})",
        labels={temp_col: f"Temperature ({temp_symbol})"}
    )
    fig_city_season.update_layout(height=420)
    st.plotly_chart(fig_city_season, width="stretch")

st.markdown("#### :violet[Detailed City Statistics]")
st.dataframe(
    city_stats.style.format({
        "Average": "{:.1f}",
        "Min": "{:.1f}",
        "Max": "{:.1f}",
        "Std Dev": "{:.2f}"
    }).background_gradient(cmap="RdYlBu_r", subset=["Average"]),
    width="stretch"
)


st.markdown("   ")
# -------------------------------------------------------------------
# GEOGRAPHIC TEMPERATURE DISTRIBUTION
# -------------------------------------------------------------------
st.subheader("📍 :rainbow[Geographic Analysis]", divider="rainbow")

regional_stats = df_filtered.groupby("Region")[temp_col].agg(["mean", "min", "max"]).reset_index()
regional_stats.columns = ["Region", "Average", "Min", "Max"]
if not regional_stats.empty:
    fig_reg_stats = go.Figure()
    fig_reg_stats.add_trace(go.Bar(
            x=regional_stats["Region"],
            y=regional_stats["Average"],
            name="Average",
            marker_color="indianred"
    ))
    fig_reg_stats.add_trace(go.Scatter(
            x=regional_stats["Region"],
            y=regional_stats["Max"],
            mode="markers",
            name="Max",
            marker=dict(size=10, color="red", symbol="triangle-up")
    ))
    fig_reg_stats.add_trace(go.Scatter(
            x=regional_stats["Region"],
            y=regional_stats["Min"],
            mode="markers",
            name="Min",
            marker=dict(size=10, color="blue", symbol="triangle-down")
    ))
    fig_reg_stats.update_layout(
            title=f"Temperature Range by Region ({temp_symbol})",
            xaxis_title="Region",
            yaxis_title=f"Temperature ({temp_symbol})",
            height=420
    )
    st.plotly_chart(fig_reg_stats, width="stretch")
else:
    st.info("No regional stats for current filters.")

country_avg = df_filtered.groupby("Country")[temp_col].mean().reset_index()
country_avg.columns = ["Country", "Average Temperature"]
if not country_avg.empty:
    top_hot = country_avg.nlargest(10, "Average Temperature")
    fig_hot = px.bar(
        top_hot,
        x="Average Temperature",
        y="Country",
        orientation="h",
        title=f"Top 10 Hottest Countries ({temp_symbol})",
        color="Average Temperature",
        color_continuous_scale="Reds"
    )
    fig_hot.update_layout(height=420)
    st.plotly_chart(fig_hot, width="stretch")

    top_cold = country_avg.nsmallest(10, "Average Temperature")
    fig_cold = px.bar(
        top_cold,
        x="Average Temperature",
        y="Country",
        orientation="h",
        title=f"Top 10 Coldest Countries ({temp_symbol})",
        color="Average Temperature",
        color_continuous_scale="Blues_r"
    )
    fig_cold.update_layout(height=420)
    st.plotly_chart(fig_cold, width="stretch")
else:
    st.info("No country-level data for current filters.")

st.markdown("   ")
st.markdown("#### :rainbow[Global Temperature Map]")
if not country_avg.empty:
    fig_map = px.choropleth(
        country_avg,
        locations="Country",
        locationmode="country names",
        color="Average Temperature",
        title=f"Global Average Temperature by Country ({temp_symbol})",
        color_continuous_scale="RdYlBu_r",
        labels={"Average Temperature": f"Avg Temp ({temp_symbol})"}
    )
    fig_map.update_layout(height=520)
    st.plotly_chart(fig_map, width="stretch")

st.markdown("   ")

# -------------------------------------------------------------------
# FORECASTING & MODEL PERFORMANCE
# -------------------------------------------------------------------
st.subheader("🔮 :yellow[Forecasting & Model Performance]", divider="yellow")
model, daily = train_prophet(df_filtered, temp_col)

if model is None or daily.empty:
    st.info("Not enough data to train a forecasting model for the current filters.")
else:
    # Convert all datetime columns to Python datetime (critical fix)
    daily["ds"] = pd.to_datetime(daily["ds"]).dt.to_pydatetime()

    # Forecast
    future = model.make_future_dataframe(periods=forecast_days, freq="D")
    future["ds"] = pd.to_datetime(future["ds"]).dt.to_pydatetime()

    forecast = model.predict(future)
    forecast["ds"] = pd.to_datetime(forecast["ds"]).dt.to_pydatetime()

    last_date = daily["ds"].max()
    last_date = last_date.to_pydatetime()

    forecast_future = forecast[forecast["ds"] > last_date]

    # -----------------------------
    # Forecast Visualization
    # -----------------------------
    st.markdown("### :yellow[Forecast Visualization]")

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
            x=daily["ds"],
            y=daily["y"],
            mode="lines",
            name="Historical",
            line=dict(color="blue", width=1),
            opacity=0.7
    ))
    fig_forecast.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color="red", width=2)
    ))
    fig_forecast.add_trace(go.Scatter(
            x=list(forecast["ds"]) + list(forecast["ds"])[::-1],
            y=list(forecast["yhat_upper"]) + list(forecast["yhat_lower"])[::-1],
            fill="toself",
            fillcolor="rgba(255,0,0,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval",
            showlegend=True
    ))
    fig_forecast.update_layout(
            title=f"Temperature Forecast ({forecast_days} days ahead)",
            xaxis_title="Date",
            yaxis_title=f"Temperature ({temp_symbol})",
            height=520,
            hovermode="x unified"
    )
    st.plotly_chart(fig_forecast, width="stretch")
st.markdown("   ")
st.markdown("#### :yellow[Forecast Summary]")
current_avg = daily["y"].tail(30).mean()
forecast_avg = forecast_future["yhat"].mean()
forecast_max = forecast_future["yhat"].max()
change = forecast_avg - current_avg

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        Components.metric_card(
            title="Current Avg (Last 30 days)",
            value=f"{current_avg:.2f}{temp_symbol}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title=f"Predicted Avg (Next {forecast_days} days)",
            value=f"{forecast_avg:.2f}{temp_symbol}",
            delta=f"{change:+.2f}{temp_symbol}",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Predicted Maximum",
            value=f"{forecast_max:.2f}{temp_symbol}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.metric_card(
            title="Data Points Used",
            value=f"{len(daily):,}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
# Model performance
st.markdown("#### :rainbow[Model Performance]")

# Train-test split on daily
split_idx = int(len(daily) * 0.8)
train_data = daily.iloc[:split_idx].copy()
test_data = daily.iloc[split_idx:].copy()

test_forecast = model.predict(test_data[["ds"]])
test_forecast["ds"] = pd.to_datetime(test_forecast["ds"]).dt.to_pydatetime()

test_merged = test_data.merge(
    test_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
    on="ds"
)

mae = mean_absolute_error(test_merged["y"], test_merged["yhat"])
rmse = np.sqrt(mean_squared_error(test_merged["y"], test_merged["yhat"]))
r2 = r2_score(test_merged["y"], test_merged["yhat"])
mape = np.mean(np.abs((test_merged["y"] - test_merged["yhat"]) / test_merged["y"])) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        Components.metric_card(
            title="MAE",
            value=f"{mae:.3f}{temp_symbol}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="RMSE",
            value=f"{rmse:.3f}{temp_symbol}",
            delta=f"{change:+.2f}{temp_symbol}",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="R² Score",
            value=f"{r2:.3f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.metric_card(
            title="MAPE",
            value=f"{mape:.2f}%",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.info(
    f"Model explains approximately {r2*100:.1f}% of the variance in daily temperature "
    f"for the selected filters."
)

fig_avp = go.Figure()
fig_avp.add_trace(go.Scatter(
    x=test_merged["ds"],
    y=test_merged["y"],
    mode="lines",
    name="Actual",
    line=dict(color="blue", width=2)
))
fig_avp.add_trace(go.Scatter(
    x=test_merged["ds"],
    y=test_merged["yhat"],
    mode="lines",
    name="Predicted",
    line=dict(color="red", width=2)
))
fig_avp.update_layout(
    title="Actual vs Predicted Temperature (Test Period)",
    xaxis_title="Date",
    yaxis_title=f"Temperature ({temp_symbol})",
    height=420,
    hovermode="x unified"
)
st.plotly_chart(fig_avp, width="stretch")

# Residuals
st.markdown("#### :rainbow[Residual Analysis]")
test_merged["residual"] = test_merged["y"] - test_merged["yhat"]

fig_res_hist = px.histogram(
    test_merged,
    x="residual",
    nbins=50,
    title="Distribution of Prediction Errors",
    labels={"residual": f"Residual (Actual - Predicted) {temp_symbol}"},
    color_discrete_sequence=["indianred"]
)
fig_res_hist.add_vline(x=0, line_dash="dash", line_color="green")
fig_res_hist.update_layout(height=380)
st.plotly_chart(fig_res_hist, width="stretch")

fig_res_time = px.scatter(
    test_merged,
    x="ds",
    y="residual",
    title="Residuals Over Time",
    labels={"ds": "Date", "residual": f"Residual ({temp_symbol})"}
)
fig_res_time.add_hline(y=0, line_dash="dash", line_color="green")
fig_res_time.update_layout(height=380)
st.plotly_chart(fig_res_time, width="stretch")

st.markdown("   ")

# -------------------------------------------------------------------
# INSIGHTS & RECOMMENDATIONS
# -------------------------------------------------------------------
st.subheader("🎯 :red[Key Insights & Recommendations]", divider="red")

with st.expander("View synthesized insights"):
    # Simple, data-driven narrative hooks
    if not df_filtered.empty:
        hottest_region = df_filtered.groupby("Region")[temp_col].mean().sort_values(ascending=False).head(1)
        coldest_region = df_filtered.groupby("Region")[temp_col].mean().sort_values(ascending=True).head(1)

        st.markdown("**Climate Signals**")
        st.markdown(
            f"- The warmest region in the current selection is **{hottest_region.index[0]}** "
            f"with an average of **{hottest_region.iloc[0]:.1f}{temp_symbol}**."
        )
        st.markdown(
            f"- The coldest region in the current selection is **{coldest_region.index[0]}** "
            f"with an average of **{coldest_region.iloc[0]:.1f}{temp_symbol}**."
        )
        if not np.isnan(warming_rate):
            st.markdown(
                f"- The estimated warming rate is **{warming_rate*10:.3f}{temp_symbol} per decade** "
                f"over the selected period."
            )

        st.markdown("**Recommendations**")
        st.markdown(
            "- Focus further analysis on regions with the highest warming rates and largest seasonal variability.\n"
            "- Use city-level volatility to identify locations most exposed to extreme temperature swings.\n"
            "- Extend the forecasting horizon cautiously where residuals remain centered and stable over time."
        )
    else:
        st.write("No data available for current filters to generate insights.")

# Future projections  
st.subheader("🔮 :violet[Future Projections]", divider="violet") 

st.markdown("**Based on current trends, by 2030:**")
st.warning(f"Expected temperature increase: **{warming_rate * (2030 - yearly['Year'].max()):.2f}°C**")
st.warning(f"Projected average temperature: **{yearly[temp_col].iloc[-1] + warming_rate * (2030 - yearly['Year'].max()):.1f}{temp_symbol}**")
st.markdown("   ")

st.markdown("**By 2050:**")
st.warning(f"Expected temperature increase: **{warming_rate * (2050 - yearly['Year'].max()):.2f}°C**")
st.warning(f"Projected average temperature: **{yearly[temp_col].iloc[-1] + warming_rate * (2050 - yearly['Year'].max()):.1f}{temp_symbol}**")
st.markdown("   ")
st.markdown("⚠️ *Note: These are linear projections based on historical trends and do not account for policy changes or climate interventions.*")

st.markdown("   ")

# Recommendations  
st.subheader("💡 :red[Actionable Recommendations]", divider="red")  
  
if warming_rate > 0.02:  
    with st.expander("High Warming Rate Detected - Urgent Actions Recommended"):
        st.markdown(
        """
        <ol style="margin: 0; padding-left: 20px;">
            <li><strong>Monitoring:</strong> Implement continuous temperature monitoring systems</li>
            <li><strong>Infrastructure:</strong> Prepare infrastructure for higher temperature extremes</li>
            <li><strong>Planning:</strong> Develop heat action plans for vulnerable populations</li>
            <li><strong>Green Spaces:</strong> Increase urban greenery to mitigate heat island effects</li>
            <li><strong>Energy:</strong> Transition to renewable energy sources to reduce emissions</li>
            <li><strong>Adaptation:</strong> Develop climate adaptation strategies for agriculture and water resources</li>
        </ol>
        """, unsafe_allow_html=True)
        

elif warming_rate > 0.01:  
    with st.expander("Moderate Warming Trend - Proactive Measures Suggested"):
        st.markdown(
        """
        <ol style="margin: 0; padding-left: 20px;">
            <li><strong>Data Collection:</strong> Continue detailed temperature monitoring</li>
            <li><strong>Public Awareness:</strong> Educate communities about climate change impacts</li>
            <li><strong>Sustainable Practices:</strong> Encourage energy efficiency and sustainable practices</li>
            <li><strong>Preparedness:</strong> Develop contingency plans for extreme weather events</li>
            <li><strong>Research:</strong> Support climate research and modeling initiatives</li>
         </ol>
        """, unsafe_allow_html=True)

else:  
    with st.expander("Stable Temperature Trends - Maintain Current Practices"):
        st.markdown(
        """
        <ol style="margin: 0; padding-left: 20px;">
            <li><strong>Vigilance:</strong> Continue monitoring for any changes in patterns</li>
            <li><strong>Best Practices:</strong> Maintain current environmental protection measures</li>
            <li><strong>Documentation:</strong> Keep detailed records for long-term trend analysis</li>
            <li><strong>Prevention:</strong> Implement preventive measures to avoid future warming</li>
            <li><strong>Collaboration:</strong> Share data and insights with climate research communities</li>
         </ol>
        """, unsafe_allow_html=True)
st.markdown("   ")
  
# Statistical insights  
st.subheader("📊 :red[Statistical Summary]", divider="red")  
  
# Extreme events analysis  
temp_mean = df_filtered[temp_col].mean()  
temp_std = df_filtered[temp_col].std()  
  
extreme_hot = df_filtered[df_filtered[temp_col] > temp_mean + 2*temp_std]  
extreme_cold = df_filtered[df_filtered[temp_col] < temp_mean - 2*temp_std]  
  
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        Components.metric_cards(
            title="Extreme Heat Events",
            value=f"{len(extreme_hot)/len(df_filtered)*100:.2f}% of records",
            delta=f"{len(extreme_hot):,}",
            card_type="error"
        ), unsafe_allow_html=True
    )
   with col2:
    st.markdown(
        Components.metric_cards(
            title="Extreme Cold Events",
            value=f"{len(extreme_cold)/len(df_filtered)*100:.2f}% of records",
            delta=f"{len(extreme_cold):,}",
            card_type="info"
        ), unsafe_allow_html=True
    )
   with col3:
    recent_years = df_filtered[df_filtered['Year'] >= yearly['Year'].max() - 5]  
	trend_direction = "↑ Warming" if warming_rate > 0 else "↓ Cooling" if warming_rate < 0 else "→ Stable"
    st.markdown(
        Components.metric_cards(
            title="5-Year Trend",
            value=f"{trend_direction}",
            delta=f"{warming_rate * 5:.3f}°C",
            card_type="success"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
# Actionable business/research insights  
st.subheader("🎯 :red[Domain-Specific Applications]", divider="red")  
  
 
with st.expander("🏢 Business Applications"):  
	st.markdown(
    """   
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
  
with st.expander("🔬 Research Applications"):  
	st.markdown(
    """
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
  
with st.expander("🏛 Policy Recommendations"):  
	st.markdown(
    """
    **Immediate Actions (0-2 years):**  
    - Establish temperature monitoring networks in underserved areas  
    - Develop heat emergency response protocols  
    - Create public awareness campaigns on climate adaptation  
  
    **Medium-term (2-5 years):**  
    - Implement urban greening initiatives (target: {warming_rate * 5:.2f}°C reduction)  
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
  
with st.expander("👥 Public Health Implications"):  
	st.markdown(
    """
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
st.markdown("   ")
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
