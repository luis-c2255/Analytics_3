import streamlit as st  
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    df = df.dropna(subset=['Date'])
    df['Capital Gains'] = df['Capital Gains'].fillna(0)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Capital Gains']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    df.drop_duplicates(inplace=True)
    df = df.sort_values(by=['Ticker', 'Date'])
    df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change() * 100
    df['Cumulative_Return'] = df.groupby('Ticker')['Close'].transform(lambda x: (x / x.iloc[0] - 1) * 100 if x.iloc[0] != 0 else 0)
    df['Price_Change'] = df['Close'] - df['Open']
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")

# Date Range Filter
min_date = df['Date'].min().to_pydatetime()
max_date = df['Date'].max().to_pydatetime()
date_range = st.sidebar.slider(
    "Select Date Range",
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)
filtered_df = df[(df['Date'] >= pd.to_datetime(date_range[0], utc=True)) & (df['Date'] <= pd.to_datetime(date_range[1], utc=True))]

# Select unique values for filters
all_tickers = sorted(filtered_df['Ticker'].unique().tolist())
all_industries = sorted(filtered_df['Industry_Tag'].unique().tolist())
all_countries = sorted(filtered_df['Country'].unique().tolist())
# Brand/Ticker Filter
selected_tickers = st.sidebar.multiselect(
    "Select Brands/Tickers",
    options=all_tickers,
    default=all_tickers[:5] # Default to top 5 for demo
)
if selected_tickers:
    filtered_df = filtered_df[filtered_df['Ticker'].isin(selected_tickers)]

# Industry Filter
selected_industries = st.sidebar.multiselect(
    "Select Industries",
    options=all_industries,
    default=all_industries[:3] # Default to top 3 for demo
)
if selected_industries:
    filtered_df = filtered_df[filtered_df['Industry_Tag'].isin(selected_industries)]

# Country Filter
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=all_countries,
    default=['usa', 'germany'] # Default to a few countries
)
if selected_countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

# Title
st.markdown(
    Components.page_header(
        "💱  World Stock Price Analysis Dashboard"
    ), unsafe_allow_html=True
)
st.markdown("   ")
st.subheader("📊 :red[Market Overview]", divider="red")

col1, col2, col3 = st.columns(3)
with col1:
    num_stocks = filtered_df['Ticker'].nunique()
    st.markdown(
        Components.metric_card(
            title="Total Unique Stocks",
            value=f"{num_stocks}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    total_volume = filtered_df['Volume'].sum()
    st.markdown(
        Components.metric_card(
            title="Total Trading Volume (B)",
            value=f"{total_volume/1_000_000_000:.2f}B",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    avg_daily_return = filtered_df['Daily_Return'].mean()
    st.markdown(
        Components.metric_card(
            title="Avg. Daily Return",
            value=f"{avg_daily_return:.2f}%",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.subheader(":red[Cumulative Returns Over Time]")
overview_df = filtered_df.copy()
fig_cum_return = px.line(
    overview_df,
    x='Date',
    y='Cumulative_Return',
    color='Ticker',
    title='Cumulative Returns (%) of Selected Stocks', 
    labels={'Cumulative_Return': 'Cumulative Return (%)'},
    hover_data={'Brand_Name': True, 'Industry_Tag': True, 'Country': True}
)
fig_cum_return.update_layout(hovermode='x unified')
st.plotly_chart(fig_cum_return, width="stretch")
st.markdown("   ")

st.subheader(":red[Daily Trading Volume]")
fig_volume = px.area(
    overview_df,
    x='Date',
    y='Volume',
    color='Ticker',
    title='Daily Trading Volume of Selected Stocks',
    labels={'Volume': 'Trading Volume'},
    hover_data={'Brand_Name': True}
)
fig_volume.update_layout(hovermode='x unified')
st.plotly_chart(fig_volume, width="stretch")

st.subheader("📈 :blue[Individual Stock Deep Dive]", divider="blue")
st.markdown("   ")
single_ticker_options = filtered_df['Ticker'].unique().tolist()
if not single_ticker_options:
    st.info("No stocks available for deep dive with current filters.")
else:
    selected_single_ticker = st.selectbox(
        "Select a Ticker for Detailed Analysis",
        options=single_ticker_options)
if selected_single_ticker:
    stock_df = filtered_df[filtered_df['Ticker'] == selected_single_ticker].sort_values('Date')

st.subheader(f"{selected_single_ticker} {stock_df['Brand_Name'].iloc[0]} Performance")
st.markdown("   ")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        Components.metric_card(
            title="Start Price",
            value=f"${stock_df['Open'].iloc[0]:.2f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="End Price",
            value=f"${stock_df['Close'].iloc[-1]:.2f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    price_change_total = stock_df['Close'].iloc[-1] - stock_df['Close'].iloc[0]
    st.markdown(
        Components.metric_card(
            title="Price Change",
            value=f"${price_change_total:.2f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    volatility = stock_df['Daily_Return'].std() if not stock_df['Daily_Return'].isnull().all() else 0
    st.markdown(
        Components.metric_card(
            title="Volatility (Daily Return Std Dev)",
            value=f"{volatility:.2f}%",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
fig_candlestick = go.Figure(data=[go.Candlestick(
    x=stock_df['Date'],
    open=stock_df['Open'],
    high=stock_df['High'],
    low=stock_df['Low'],
    close=stock_df['Close'],
    name='Candlestick'
)])
fig_candlestick.update_layout(
    title=f"{selected_single_ticker} Candlestick Chart",
    xaxis_rangeslider_visible=False,
    hovermode='x unified'
)
st.plotly_chart(fig_candlestick, width="stretch")
st.markdown("   ")
fig_daily_return = px.line(
    stock_df,
    x='Date',
    y='Daily_Return',
    title=f"{selected_single_ticker} Daily Returns (%)",
    labels={'Daily_Return': 'Daily Return (%)'},
    color_discrete_sequence=['red']
)
fig_daily_return.update_layout(hovermode='x unified')
st.plotly_chart(fig_daily_return, width="stretch")

st.subheader("🌎 :green[Industry/Country Comparison]", divider="green")
st.markdown("   ")
industry_performance = filtered_df.groupby(['Date', 'Industry_Tag'])['Cumulative_Return'].mean().reset_index()
fig_industry = px.line(
    industry_performance,
    x='Date',
    y='Cumulative_Return',
    color='Industry_Tag',
    title='Average Cumulative Return by Industry Tag',
    labels={'Cumulative_Return': 'Average Cumulative Return (%)'},
    hover_data={'Industry_Tag': True}
)
fig_industry.update_layout(hovermode='x unified')
st.plotly_chart(fig_industry, width="stretch")
st.markdown("   ")
country_performance = filtered_df.groupby(['Date', 'Country'])['Cumulative_Return'].mean().reset_index()
fig_country = px.line(
    country_performance,
    x='Date',
    y='Cumulative_Return',
    color='Country',
    title='Average Cumulative Return by Country',
    labels={'Cumulative_Return': 'Average Cumulative Return (%)'},
    hover_data={'Country': True}
)
fig_country.update_layout(hovermode="x unified")
st.plotly_chart(fig_country, width="stretch")
st.markdown("   ")
st.subheader(":green[Volatility Comparison]")
volatility_by_ticker = filtered_df.groupby('Ticker')['Daily_Return'].std().reset_index()
volatility_by_ticker.rename(columns={'Daily_Return': 'Volatility'}, inplace=True)
volatility_metrics = pd.merge(
        volatility_by_ticker,
        filtered_df[['Ticker', 'Industry_Tag', 'Country']].drop_duplicates(),
        on='Ticker',
        how='left'
    )
avg_industry_volatility = volatility_metrics.groupby('Industry_Tag')['Volatility'].mean().reset_index()
fig_ind_vol = px.bar(
    avg_industry_volatility,
    x='Industry_Tag',
    y='Volatility',
    title='Average Volatility by Industry',
    labels={'Volatility': 'Avg Daily Return Std Dev (%)'},
    color='Industry_Tag'
)
st.plotly_chart(fig_ind_vol, width="stretch")
st.markdown("   ")
avg_country_volatility = volatility_metrics.groupby('Country')['Volatility'].mean().reset_index()
fig_country_vol = px.bar(
    avg_country_volatility,
    x='Country',
    y='Volatility',
    title='Average Volatility by Country',
    labels={'Volatility': 'Avg Daily Return Std Dev (%)'},
    color='Country'
)
st.plotly_chart(fig_country_vol, width="stretch")

st.subheader("💰 :yellow[Corporate Actions Impact]", divider="yellow")
st.markdown("   ")
st.subheader(":yellow[Dividends Payouts Over Time]")
dividends_df = filtered_df[filtered_df['Dividends'] > 0]
if not dividends_df.empty:
    fig_dividends = px.bar(
        dividends_df,
        x='Date',
        y='Dividends',
        color='Ticker',
        title='Dividends Paid by Selected Stocks',
        labels={'Dividends': 'Dividend Amount ($)'},
        hover_data={'Brand_Name': True}
    )
    st.plotly_chart(fig_dividends, width="stretch")

st.markdown("   ")
st.subheader(":yellow[Stock Splits Over Time]")
splits_df = filtered_df[filtered_df['Stock Splits'] > 0]
if not splits_df.empty:
    fig_splits = px.bar(
        splits_df,
        x='Date',
        y='Stock Splits',
        color='Ticker',
        title='Stock Splits by Selected Stocks',
        labels={'Stock Splits': 'Split Ratio'},
        hover_data={'Brand_Name': True}
    )
    st.plotly_chart(fig_splits, width="stretch")

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
