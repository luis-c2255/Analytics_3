import streamlit as st  
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

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

@st.cache_data
def get_transaction_data(df_input):
    """Prepare data for Market Basket Analysis."""
    # Group products by order_id to create transaction lists
    transactions = df_input.groupby('order_id')['product_name'].apply(list).tolist()
    # Use TransactionEncoder for one-hot encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    return df_encoded

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
st.markdown("   ")

st.subheader(":green[Top 10 Most Popular Products]")

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

heatmap_data = filtered_df.groupby(['order_dow', 'order_hour_of_day'])['order_id'].nunique().unstack(fill_value=0)
days_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6:'Saturday'}
heatmap_data.index = heatmap_data.index.map(days_map)

fig = go.Figure(
    data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlGnBu'
    )
)
fig.update_layout(
    title_text='Order Density: Day vs. Hour',
    xaxis_title='Hour of Day',
    yaxis_title='Day of Week'
)
st.plotly_chart(fig, width="stretch")
st.markdown("   ")
max_cell = heatmap_data.stack().idxmax()
max_val = heatmap_data.stack().max()

st.success(f"Peak period: {max_cell[0]} a {max_cell[1]}:00 with {max_val} orders.")
st.markdown("   ")

# 1. Calculate metrics per user
user_stats = filtered_df.groupby('user_id').agg(
    total_orders=('order_id', 'nunique'),
    total_items=('product_id', 'count'),
    avg_basket_size=('order_id', lambda x: len(x) / x.nunique()),
    reorder_rate=('reordered', 'mean')
)
# 2. Identify Top Values for Metric Cards
top_order_user_id = user_stats['total_orders'].idxmax()
max_orders = user_stats['total_orders'].max()

top_basket_user_id = user_stats['avg_basket_size'].idxmax()
max_basket = user_stats['avg_basket_size'].max()

# Use reorder_rate across all users or top_items if total_orders <= 1
top_items_user_id = user_stats['total_items'].idxmax()
max_items = user_stats['total_items'].max()

# Find users with high loyalty, filtering for any repeat buyers if they exist
repeat_buyers = user_stats[user_stats['total_orders'] > 1]
if not repeat_buyers.empty:
    top_loyalty_user_id = repeat_buyers['reorder_rate'].idxmax()
    max_loyalty = repeat_buyers['reorder_rate'].max()
else:
    top_loyalty_user_id = "N/A (No Repeat Buyers)"
    max_loyalty = 0.0

col1, col2 = st.columns(2, border=True, gap="medium", vertical_alignment="center")
with col1:
    st.info(f"User with most orders: ID {top_order_user_id} ({max_orders} orders)")
with col2:
    st.success(f"User with largest single/avg basket: ID {top_basket_user_id} ({max_basket:.1f} items)")

col3, col4 = st.columns(2, border=True, gap="medium", vertical_alignment="center")
with col3:
    st.success(f"Most loyal user (repeat): {top_loyalty_user_id} ({max_loyalty:.1%})")
with col4:
    st.info(f"Top purchaser by volume: ID {top_items_user_id} ({max_items} total items)")
st.markdown("   ")
fig2= px.scatter(
    user_stats,
    x='total_items',
    y='reorder_rate',
    opacity=0.5,
    color_discrete_sequence=['blue']
)
fig2.update_layout(
    title_text='User Clusters: Total Items vs Reorder Rate',
    xaxis_title_text='Total Items Purchased',
    yaxis_title_text='Reorder Rate (Loyalty)',
    width=1000,
    height=600
)
st.plotly_chart(fig2, width="stretch")
st.markdown("   ")
import plotly.figure_factory as ff

basket_sizes = filtered_df.groupby('order_id')['product_id'].count().tolist()

group_labels = ['Basket Size Distribution']


fig3 = ff.create_distplot(
    [basket_sizes],
    group_labels,
    bin_size=2,
    curve_type='kde',
    colors=['#800080']
)
fig3.update_layout(
    title='Basket Size Distribution with KDE Curve',
    xaxis_title_text='Number of Items per Order',
    yaxis_title_text='Probability Density',
    showlegend=False
)
st.plotly_chart(fig3, width="stretch")
st.markdown("   ")
# 1. Create a per-order dataset
order_stats = filtered_df.groupby('order_id').agg(
    basket_size=('product_id', 'count'),
    order_dow=('order_dow', 'first'),
    order_hour_of_day=('order_hour_of_day', 'first')
).reset_index()

# 2. Avg Basket Size by Day of the Week
days_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
basket_by_day = order_stats.groupby('order_dow')['basket_size'].mean().reset_index()
basket_by_day['day_name'] = basket_by_day['order_dow'].map(days_map)
# Ensure correct order
days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
basket_by_day['day_name'] = pd.Categorical(basket_by_day['day_name'], categories=days_order, ordered=True)
basket_by_day = basket_by_day.sort_values('day_name')
# 3. Avg Basket Size by Hour of Day
basket_by_hour = order_stats.groupby('order_hour_of_day')['basket_size'].mean()

fig4 = px.bar(
    basket_by_day,
    x='day_name',
    y='basket_size',
    title='Average Basket Size by Day of the Week',
    labels={'day_name': 'Day of the Week', 'basket_size': 'Avg Items per Order'},
    color='basket_size',
    color_continuous_scale='Blues'
)
fig4.update_layout(
    xaxis_title="Day of the Week",
    yaxis_title="Avg Number of Items",
    coloraxis_showscale=False,  # Hide scale if not needed
    hoverlabel=dict(bgcolor="white", font_size=16)
)
st.plotly_chart(fig4, width="stretch")
st.markdown("   ")

fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=basket_by_hour.index,
    y=basket_by_hour.values,
    mode='lines+markers',  # Combines 'line' kind and 'o' marker
    line=dict(color='purple'), # Line color
    marker=dict(color='purple') # Marker color (default symbol is circle)
))
fig5.update_layout(
    title_text='Average Basket Size by Hour of Day',
    xaxis_title='Hour of Day (24h)',
    yaxis_title='Avg Number of Items',
    width=1000, # Matplotlib figsize=(10, 5) is approximated as 1000x500 pixels
    height=500,
    xaxis=dict(
        tickvals=list(range(0, 24)), # Equivalent to plt.xticks(range(0, 24))
        showgrid=True,
        gridcolor='rgba(128,128,128,0.6)', # Approximates alpha=0.6 for grid lines
        griddash='dash' # Equivalent to linestyle='--'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(128,128,128,0.6)',
        griddash='dash'
    )
)
st.plotly_chart(fig5, width="stretch")
st.markdown("   ")

st.subheader("🛒 :orange[Market Basket Analysis (Association Rules)]", divider='orange')
st.markdown("""
    Market Basket Analysis helps discover produts frequently purchased together.
    Adjust the parameters below to find interesting association rules.
    Note: This analysis can be computationally intensive for large datasets.
    Start with higher min_support and min_confidence to get initial results faster.
    """, text_alignment="center")

col1, col2, col3 = st.columns(3)
with col1:
    min_support = st.slider("Minimum Support", min_value=0.001, max_value=0.1, value=0.005, step=0.001, help='Frequency of itemsets (e.g., 0.01 means in 1% of transactions)')

with col2: 
    min_confidence = st.slider("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.05, help='Likelihood of buying Y given X (P(Y|X))')

with col3:
    min_lift = st.slider("Minimum Lift", min_value=1.0, max_value=5.0, value=1.2, step=0.1, help='How much more likely Y is bought when X is bought, vs. independently (Lift > 1 implies positive correlation)')

if st.button("Run Market Basket Analysis"):
    with st.spinner("Preparing transaction data and computing rules... This may take a few minutes for large datasets."):
        # Filter the main DataFrame to only include relevant columns for MBA to save memory
        df_mba_prep = df[['order_id', 'product_name']]

        # If a department filter is applied globally, it impacts the scope of MBA
        if selected_department != 'All':
            # Need to rejoin department info to filter product_name effectively
            # Simpler: just pass filtered_df to get_transaction_data, if filtered_df is not too small
            # For MBA, it's often better to run on a broader dataset first, then filter rules.
            # However, if the user explicitly filtered by department, they want MBA within that department.
            df_mba_prep = filtered_df[['order_id', 'product_name']]
            if df_mba_prep.empty:
                st.warning("No data for Market Basket Analysis with current filters. Adjust department/reorder filters.")
                st.stop()
        
# ensure unique order_id and product_name combinations for MBA
df_mba_prep = df_mba_prep.drop_duplicates(subset=['order_id', 'product_name'])

# GEt one-hot encoded transaction DataFrame
df_encoded_mba = get_transaction_data(df_mba_prep)

# Ensure there are enough unique orders after filtering
if df_encoded_mba.shape[0] < 2:
    st.warning("Not enough transactions (orders) top perform Market Basket Analysis with current filters. Please broaden your selection.")
    st.stop()
        
# Find frequent itemsets
frequent_itemsets = apriori(df_encoded_mba, min_support=min_support, use_colnames=True)

if frequent_itemsets.empty:
    st.warning(f"No frequent itemsets found with minimum support of {min_support}. Try lowering the support threshold.")
else:
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift]

if rules.empty:
    st.warning(f"No association rules found with minimum confidence of {min_confidence} and minimum lift of {min_lift}. Try lowering the thresholds.")
else:
    # Clean up and display rules
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False)
st.subheader(":orange[Discovered Association Rules]")
st.dataframe(rules_display)

st.markdown("   ")
st.subheader(":orange[Top Antecedents & Consequents]")
col1, col2 = st.columns(2)
with col1:
    st.write("Most Frequent Antecedents (What customers *start* with)")
    top_antecedents = rules['antecedents'].value_counts().head(10).reset_index()
    top_antecedents.columns = ['Antecedent', 'Count']
    fig_ant = px.bar(
        top_antecedents,
        x='Count',
        y='Antecedent',
        orientation='h',
        title='Top 10 Antecedents in Rules',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_ant.update_layout(
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig_ant, width="stretch")

with col2:
    st.write("Most Frequent Consequents (What customers *end up* buying)")
    top_consequents = rules['consequents'].value_counts().head(10).reset_index()
    top_consequents.columns = ['Consequent', 'Count']
    fig_con = px.bar(
        top_consequents,
        x='Count',
        y='Consequent',
        orientation='h',
        title='Top 10 Consequents in Rules',
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig_con.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_con, width="stretch")


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
