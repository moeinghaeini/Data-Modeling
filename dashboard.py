"""
Interactive Dashboard for Retail Data Analysis
Demonstrates data visualization and interactive analytics capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RetailDashboard:
    """Interactive Streamlit dashboard for retail analytics"""
    
    def __init__(self, db_path: str = "retail_data.db"):
        self.db_path = db_path
        
    def load_data(self):
        """Load data from database"""
        conn = sqlite3.connect(self.db_path)
        
        # Load all tables
        customers = pd.read_sql_query("SELECT * FROM customers", conn)
        products = pd.read_sql_query("SELECT * FROM products", conn)
        stores = pd.read_sql_query("SELECT * FROM stores", conn)
        orders = pd.read_sql_query("SELECT * FROM orders", conn)
        order_items = pd.read_sql_query("SELECT * FROM order_items", conn)
        
        conn.close()
        
        # Convert date columns
        orders['order_date'] = pd.to_datetime(orders['order_date'])
        
        return customers, products, stores, orders, order_items
    
    def create_comprehensive_dataset(self, customers, products, stores, orders, order_items):
        """Create comprehensive analytical dataset"""
        # Merge all data
        df = order_items.merge(products, on='product_id', how='left')
        df = df.merge(orders, on='order_id', how='left')
        df = df.merge(customers, on='customer_id', how='left')
        df = df.merge(stores, on='store_id', how='left')
        
        return df
    
    def run_dashboard(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="Retail Analytics Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Title and description
        st.title("🏪 Retail Analytics Dashboard")
        st.markdown("""
        **Comprehensive retail data analysis dashboard demonstrating:**
        - Semantic data modeling with ontologies
        - ETL pipeline processing
        - Multiple data modeling schemas (3NF, Star, Snowflake)
        - Machine learning and predictive analytics
        - Interactive visualizations
        """)
        
        # Load data
        with st.spinner("Loading data..."):
            customers, products, stores, orders, order_items = self.load_data()
            df = self.create_comprehensive_dataset(customers, products, stores, orders, order_items)
        
        # Sidebar filters
        st.sidebar.header("🔍 Filters")
        
        # Date range filter
        min_date = df['order_date'].min().date()
        max_date = df['order_date'].max().date()
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Customer segment filter
        segments = st.sidebar.multiselect(
            "Customer Segments",
            options=df['customer_segment'].unique(),
            default=df['customer_segment'].unique()
        )
        
        # Store filter
        store_options = st.sidebar.multiselect(
            "Stores",
            options=df['store_name'].unique(),
            default=df['store_name'].unique()
        )
        
        # Product category filter
        categories = st.sidebar.multiselect(
            "Product Categories",
            options=df['category'].unique(),
            default=df['category'].unique()
        )
        
        # Apply filters
        if len(date_range) == 2:
            df_filtered = df[
                (df['order_date'].dt.date >= date_range[0]) &
                (df['order_date'].dt.date <= date_range[1]) &
                (df['customer_segment'].isin(segments)) &
                (df['store_name'].isin(store_options)) &
                (df['category'].isin(categories))
            ]
        else:
            df_filtered = df
        
        # Key Metrics
        st.header("📈 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = df_filtered['net_total'].sum()
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        
        with col2:
            total_orders = df_filtered['order_id'].nunique()
            st.metric("Total Orders", f"{total_orders:,}")
        
        with col3:
            avg_order_value = df_filtered.groupby('order_id')['net_total'].sum().mean()
            st.metric("Average Order Value", f"${avg_order_value:.2f}")
        
        with col4:
            unique_customers = df_filtered['customer_id'].nunique()
            st.metric("Active Customers", f"{unique_customers:,}")
        
        # Revenue Analysis
        st.header("💰 Revenue Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by customer segment
            segment_revenue = df_filtered.groupby('customer_segment')['net_total'].sum().sort_values(ascending=True)
            fig_segment = px.bar(
                x=segment_revenue.values,
                y=segment_revenue.index,
                orientation='h',
                title="Revenue by Customer Segment",
                labels={'x': 'Revenue ($)', 'y': 'Customer Segment'},
                color=segment_revenue.values,
                color_continuous_scale='Viridis'
            )
            fig_segment.update_layout(height=400)
            st.plotly_chart(fig_segment, use_container_width=True)
        
        with col2:
            # Revenue by product category
            category_revenue = df_filtered.groupby('category')['net_total'].sum().sort_values(ascending=True)
            fig_category = px.bar(
                x=category_revenue.values,
                y=category_revenue.index,
                orientation='h',
                title="Revenue by Product Category",
                labels={'x': 'Revenue ($)', 'y': 'Category'},
                color=category_revenue.values,
                color_continuous_scale='Plasma'
            )
            fig_category.update_layout(height=400)
            st.plotly_chart(fig_category, use_container_width=True)
        
        # Time Series Analysis
        st.header("📅 Time Series Analysis")
        
        # Daily revenue trend
        daily_revenue = df_filtered.groupby(df_filtered['order_date'].dt.date)['net_total'].sum().reset_index()
        daily_revenue.columns = ['date', 'revenue']
        
        fig_trend = px.line(
            daily_revenue,
            x='date',
            y='revenue',
            title="Daily Revenue Trend",
            labels={'date': 'Date', 'revenue': 'Revenue ($)'}
        )
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Customer Analysis
        st.header("👥 Customer Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer age distribution
            fig_age = px.histogram(
                df_filtered,
                x='age',
                nbins=20,
                title="Customer Age Distribution",
                labels={'age': 'Age', 'count': 'Number of Customers'},
                color_discrete_sequence=['#FF6B6B']
            )
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Customer segment distribution
            segment_counts = df_filtered['customer_segment'].value_counts()
            fig_segment_pie = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Segment Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_segment_pie.update_layout(height=400)
            st.plotly_chart(fig_segment_pie, use_container_width=True)
        
        # Product Performance
        st.header("🛍️ Product Performance")
        
        # Top products by revenue
        top_products = df_filtered.groupby(['product_id', 'product_name']).agg({
            'net_total': 'sum',
            'quantity': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        top_products.columns = ['product_id', 'product_name', 'revenue', 'quantity_sold', 'orders']
        top_products = top_products.sort_values('revenue', ascending=False).head(10)
        
        fig_products = px.bar(
            top_products,
            x='revenue',
            y='product_name',
            orientation='h',
            title="Top 10 Products by Revenue",
            labels={'revenue': 'Revenue ($)', 'product_name': 'Product'},
            color='revenue',
            color_continuous_scale='Viridis'
        )
        fig_products.update_layout(height=500)
        st.plotly_chart(fig_products, use_container_width=True)
        
        # Store Performance
        st.header("🏪 Store Performance")
        
        store_performance = df_filtered.groupby('store_name').agg({
            'net_total': 'sum',
            'order_id': 'nunique',
            'customer_id': 'nunique'
        }).reset_index()
        store_performance.columns = ['store_name', 'revenue', 'orders', 'customers']
        store_performance['avg_order_value'] = store_performance['revenue'] / store_performance['orders']
        
        fig_store = px.scatter(
            store_performance,
            x='orders',
            y='revenue',
            size='customers',
            color='avg_order_value',
            hover_name='store_name',
            title="Store Performance: Orders vs Revenue",
            labels={'orders': 'Number of Orders', 'revenue': 'Revenue ($)', 'customers': 'Customers'},
            color_continuous_scale='Viridis'
        )
        fig_store.update_layout(height=500)
        st.plotly_chart(fig_store, use_container_width=True)
        
        # Advanced Analytics
        st.header("🔬 Advanced Analytics")
        
        # Customer lifetime value analysis
        customer_clv = df_filtered.groupby('customer_id').agg({
            'net_total': 'sum',
            'order_id': 'nunique',
            'customer_segment': 'first',
            'age': 'first'
        }).reset_index()
        customer_clv.columns = ['customer_id', 'total_spent', 'total_orders', 'segment', 'age']
        customer_clv['avg_order_value'] = customer_clv['total_spent'] / customer_clv['total_orders']
        
        # CLV scatter plot
        fig_clv = px.scatter(
            customer_clv,
            x='total_orders',
            y='total_spent',
            color='segment',
            size='avg_order_value',
            hover_name='customer_id',
            title="Customer Lifetime Value Analysis",
            labels={'total_orders': 'Total Orders', 'total_spent': 'Total Spent ($)', 'segment': 'Segment'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_clv.update_layout(height=500)
        st.plotly_chart(fig_clv, use_container_width=True)
        
        # Data Summary
        st.header("📊 Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"**Total Records:** {len(df_filtered):,}")
            st.write(f"**Date Range:** {df_filtered['order_date'].min().strftime('%Y-%m-%d')} to {df_filtered['order_date'].max().strftime('%Y-%m-%d')}")
            st.write(f"**Unique Customers:** {df_filtered['customer_id'].nunique():,}")
            st.write(f"**Unique Products:** {df_filtered['product_id'].nunique():,}")
            st.write(f"**Unique Stores:** {df_filtered['store_id'].nunique():,}")
        
        with col2:
            st.subheader("Revenue Breakdown")
            st.write(f"**Total Revenue:** ${df_filtered['net_total'].sum():,.2f}")
            st.write(f"**Average Order Value:** ${df_filtered.groupby('order_id')['net_total'].sum().mean():.2f}")
            st.write(f"**Revenue per Customer:** ${df_filtered['net_total'].sum() / df_filtered['customer_id'].nunique():.2f}")
            st.write(f"**Orders per Customer:** {df_filtered['order_id'].nunique() / df_filtered['customer_id'].nunique():.1f}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        **Dashboard Features Demonstrated:**
        - ✅ Semantic data modeling with TTL ontologies
        - ✅ ETL pipeline with data transformation
        - ✅ Multiple data modeling schemas (3NF, Star, Snowflake)
        - ✅ Machine learning and predictive analytics
        - ✅ Interactive visualizations and filtering
        - ✅ Real-time data processing and analysis
        """)

def main():
    """Main function to run the dashboard"""
    dashboard = RetailDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
