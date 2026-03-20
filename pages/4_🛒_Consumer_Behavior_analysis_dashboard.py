import streamlit as st  
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Consumer Behavior Analysis Dashboard", "🛒")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")
	
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('consumer_behaviour.csv')
    df['days_since_prior_order'] = df['days_since_prior_order'].fillna(-1)

    df['order_id'] = df['order_id'].astype('category')
    df['user_id'] = df['user_id'].astype('category')
    df['product_id'] = df['product_id'].astype('category')
    df['department_id'] = df['department_id'].astype('category')

    day_mapping = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    df['order_dow_name'] = df['order_dow'].map(day_mapping).astype('category')
    df['order_hour_of_day'] = df['order_hour_of_day'].astype('category')
    return df

# Title
st.markdown(
    Components.page_header(
        "🛒 Consumer Behavior Analysis Dashboard"
    ), unsafe_allow_html=True
)
st.markdown("### Explore purchasing patterns, product popularity, and user habits.", text_alignment="center")
df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Options")

# Department flter
all_departments = ['All'] + sorted(df['department'].unique().tolist())
selected_department = st.sidebar.selectbox("Select Department", all_departments)

# Reordered status filter
reorder_options = {'All': -1, 'Reordered': 1, 'Not Reordered': 0}
selected_reorder_status_label = st.sidebar.selectbox("Reorder Status", list(reorder_options.keys()))
selected_reorder_status = reorder_options[selected_reorder_status_label]

# Apply filters
filtered_df = df.copy()
if selected_department != 'All':
    filtered_df = filtered_df[filtered_df['department'] == selected_department]
if selected_reorder_status != -1:
    filtered_df = filtered_df[filtered_df['reordered'] == selected_reorder_status]

st.sidebar.markdown(f"Data displayed: {filtered_df.shape[0]:,} rows")

st.subheader("📊 :red[Overview]", divider="red")

col1, col2, col3, col4 = st.columns(4)
total_orders = filtered_df['order_id'].nunique()
total_products = filtered_df['product_id'].nunique()
total_users = filtered_df['user_id'].nunique()
avg_items_per_order = filtered_df.groupby('order_id').size().mean()
with col1:
    st.markdown(
        Components.metric_card(
            title="Total Orders",
            value=f"{total_orders:,}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Unique Products Purchased",
            value=f"{total_products:,}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Unique Users",
            value=f"{total_users:,}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.metric_card(
            title="Avg Items per Order",
            value=f"{total_products:,}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.subheader(":red[Overall Purchase Trends]")
orders_by_dow = filtered_df['order_dow_name'].value_counts().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
]).reset_index()
orders_by_dow.columns = ['Day of Week', 'Number of Orders']
fig_dow = px.bar(
    orders_by_dow,
    x='Day of Week',
    y='Number of Orders',
    title='Distribution of Orders by Day of Week',
    labels={'Number of Orders': 'Total Orders'},
    color_discrete_sequence=px.colors.qualitative.Pastel
)
st.plotly_chart(fig_dow, width="stretch")

st.markdown("   ")
orders_by_hour = filtered_df['order_hour_of_day'].value_counts().sort_index().reset_index()
orders_by_hour.columns = ['order_hour_of_day', 'Number of Orders']
fig_hour = px.line(
    orders_by_hour,
    x='order_hour_of_day',
    y='Number of Orders',
    title='Distribution of Orders by Hour of Day',
    labels={'Number of Orders': 'Total Orders'},
    markers=True, line_shape='spline',
    color_discrete_sequence=px.colors.qualitative.Dark24
)
st.plotly_chart(fig_hour, width="stretch")
st.subheader("🛒 :green[Product Analysis]", divider="green")
st.markdown("   ")
col1, col2, col3, col4 = st.columns(4)
with col1:
    top_product_name = filtered_df['product_name'].value_counts().idxmax()
    top_product_count = filtered_df['product_name'].value_counts().max()
    st.markdown(
        Components.metric_card(
            title="Top Product Sold",
            value=f"{top_product_name}",
            delta=f"{top_product_count}",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    reorder_rate = filtered_df['reordered'].mean()
    st.markdown(
        Components.metric_card(
            title="Reorder Rate",
            value=f"{reorder_rate:.1%}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    dept_stats = df.groupby('department')['reordered'].agg(['mean', 'count'])
    top_dept_name = dept_stats[dept_stats['count'] >= 25]['mean'].idxmax()
    top_dept_count = dept_stats[dept_stats['count'] >= 25]['mean'].max()
    st.markdown(
        Components.metric_card(
            title="Highest Loyalty Department",
            value=f"{top_dept_name}",
            delta=f"{top_dept_count:.1%}",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    most_reordered_item = df[df['reordered'] == 1]['product_name'].value_counts().idxmax()
    most_reordered = df[df['reordered'] == 1]['product_name'].value_counts().max()
    st.markdown(
        Components.metric_card(
            title="Highest Reordered Item",
            value=f"{most_reordered_item}",
            delta=f"{most_reordered}",
            card_type="info"
        ), unsafe_allow_html=True
    )

st.subheader(":green[Top 10 Most Popular Products]")
st.markdown("   ")

top_products = filtered_df['product_name'].value_counts().head(10).reset_index()
top_products.columns = ['Product Name', 'Number of Orders']

fig_top_products = px.bar(
    top_products,
    x='Number of Orders',
    y='Product Name',
    orientation='h',
    title='Top 10 Products by Order Count',
    labels={'Number of Orders': 'Times Ordered'},
    color_discrete_sequence=px.colors.qualitative.Vivid,
    height=400
)
fig_top_products.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig_top_products, width="stretch")
st.markdown("   ")
st.subheader(":green[Top 10 Most Popular Departments]")

top_departments = filtered_df['department'].value_counts().head(10).reset_index()
top_departments.columns = ['Department', 'Number of Orders']
fig_top_departments = px.bar(
    top_departments,
    x='Number of Orders',
    y='Department', 
    orientation='h',
    title='Top 10 Department by Order Count',
    labels={'Number of Orders': 'Times Ordered'},
    color_discrete_sequence=px.colors.qualitative.Prism,
    height=400
)
fig_top_departments.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_top_departments, width="stretch")
st.markdown("   ")

st.subheader(":green[Reorder Rate Analysis]")

reorder_counts = filtered_df['reordered'].value_counts().reset_index()
reorder_counts.columns = ['Reordered Status', 'Count']
reorder_counts['Reordered Status'] = reorder_counts['Reordered Status'].map({1: 'Reordered', 0: 'Not Reordered'})

fig_reorder = px.pie(
    reorder_counts,
    values='Count',
    names='Reordered Status',
    title='Proportion of Reordered Items',
    color_discrete_sequence=px.colors.qualitative.Set3
)
st.plotly_chart(fig_reorder, width="stretch")
st.markdown("   ")

st.subheader(":green[Reorder Rate by Product (Top 10 Products)]")
product_reorder_stats = filtered_df.groupby('product_name')['reordered'].agg(
        total_items='count',
        reordered_items='sum'
    ).reset_index()
product_reorder_stats['reorder_rate'] = (product_reorder_stats['reordered_items'] / product_reorder_stats['total_items']) * 100
product_reorder_stats = product_reorder_stats[product_reorder_stats['total_items'] >= 50].sort_values(by='total_items', ascending=False).head(10) 
product_reorder_stats = product_reorder_stats.sort_values(by='reorder_rate', ascending=False)

fig_prod_reorder_rate = px.bar(
    product_reorder_stats,
    x='reorder_rate',
    y='product_name',
    orientation='h',
    title='Top 10 Products by Reorder Rate (among popular items)',
    labels={'reorder_rate': 'Reorder Rate (%)', 'product_name': 'Product Name'},
    color_discrete_sequence=px.colors.qualitative.Plotly,
    height=400
)
fig_prod_reorder_rate.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig_prod_reorder_rate, width="stretch")

st.subheader("📈 :blue[User Order Patterns]", divider="blue")
st.markdown("   ")
col1, col2, col3, col4 = st.columns(4)
with col1:
    day_mapping = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    top_day = day_mapping[df['order_dow'].mode()[0]]
    st.markdown(
        Components.metric_card(
            title="Busiest day",
            value=f"{top_day}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    peak_hour = df.groupby('order_id')['order_hour_of_day'].first().mode()[0]
    st.markdown(
        Components.metric_card(
            title="Peak Hour",
            value=f"{peak_hour} A.M",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    avg_basket = df.groupby('order_id')['product_id'].count().mean()
    st.markdown(
        Components.metric_card(
            title="Average Basket Size",
            value=f"{avg_basket:.2f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    avg_wait_time = df['days_since_prior_order'].mean()
    st.markdown(
        Components.metric_card(
            title="Avg Days between orders",
            value=f"{avg_wait_time:.2f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
st.subheader(":blue[Days Since Prior Order Distribution]")
days_since_prior_df = filtered_df[filtered_df['days_since_prior_order'] != -1]

fig_days_since = px.histogram(
        days_since_prior_df,
        x='days_since_prior_order',
        nbins=30,
        title='Distribution of Days Since Prior Order',
        labels={'days_since_prior_order': 'Days Since Prior Order'},
        color_discrete_sequence=px.colors.qualitative.D3
    )
st.plotly_chart(fig_days_since, width="stretch")
st.markdown("   ")
st.subheader(":blue[Average Add-to-Cart Order Position]")
avg_add_to_cart = filtered_df.groupby('product_name')['add_to_cart_order'].mean().sort_values().head(10).reset_index()
avg_add_to_cart.columns = ['Product Name', 'Avg. Add-to-Cart Position']
fig_avg_cart = px.bar(
    avg_add_to_cart,
    x='Avg. Add-to-Cart Position', 
    y='Product Name', 
    orientation='h',
    title='Top 10 Products by Lowest Average Add-to-Cart Position (Early in Cart)',
    labels={'Avg. Add-to-Cart Position': 'Average Position in Cart'},
    color_discrete_sequence=px.colors.qualitative.Bold,
    height=400
)
fig_avg_cart.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_avg_cart, width="stretch")

st.markdown("   ")
st.subheader(":rainbow[Raw Data]", divider="rainbow")

st.dataframe(filtered_df.head(1000))
st.download_button(
        label="Download Filtered Data",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_consumer_behaviour.csv',
        mime='text/csv',
    )
# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>🛒 Consumer Behavior Analysis Dashboard</strong></p>
    <p>Consumer Behavior data analysis</p>
    <p style='font-size: 0.9rem;'>Navigate using the sidebar to explore different datasets</p>
</div>
""", unsafe_allow_html=True)
