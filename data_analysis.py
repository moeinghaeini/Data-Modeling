"""
Data Analysis and Machine Learning PoC
Demonstrates Python data processing, analysis, and ML capabilities
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class RetailDataAnalyzer:
    """Comprehensive data analysis and ML for retail data"""
    
    def __init__(self, db_path: str = "retail_data.db"):
        self.db_path = db_path
        self.connection = None
        self.data = {}
        
    def connect_database(self):
        """Connect to the database"""
        self.connection = sqlite3.connect(self.db_path)
        
    def load_data(self):
        """Load all data from database"""
        self.connect_database()
        
        tables = ['customers', 'products', 'stores', 'orders', 'order_items']
        for table in tables:
            self.data[table] = pd.read_sql_query(f"SELECT * FROM {table}", self.connection)
        
        # Create comprehensive dataset
        self._create_analytical_dataset()
        
    def _create_analytical_dataset(self):
        """Create comprehensive analytical dataset"""
        # Merge orders with customers
        orders_customers = pd.merge(
            self.data['orders'], 
            self.data['customers'], 
            on='customer_id', 
            how='left'
        )
        
        # Merge with order items
        orders_items = pd.merge(
            self.data['order_items'],
            self.data['products'],
            on='product_id',
            how='left'
        )
        
        # Merge with stores
        orders_customers_stores = pd.merge(
            orders_customers,
            self.data['stores'],
            on='store_id',
            how='left'
        )
        
        # Create comprehensive dataset
        self.data['analytical_dataset'] = pd.merge(
            orders_items,
            orders_customers_stores,
            on='order_id',
            how='left'
        )
        
        # Add derived features
        self.data['analytical_dataset']['order_date'] = pd.to_datetime(self.data['analytical_dataset']['order_date'])
        self.data['analytical_dataset']['days_since_registration'] = (
            self.data['analytical_dataset']['order_date'] - 
            pd.to_datetime(self.data['analytical_dataset']['registration_date'])
        ).dt.days
        
    def perform_exploratory_analysis(self):
        """Perform comprehensive exploratory data analysis"""
        print("=== EXPLORATORY DATA ANALYSIS ===\n")
        
        df = self.data['analytical_dataset']
        
        # Basic statistics
        print("1. Dataset Overview:")
        print(f"   Total records: {len(df):,}")
        print(f"   Date range: {df['order_date'].min()} to {df['order_date'].max()}")
        print(f"   Unique customers: {df['customer_id'].nunique():,}")
        print(f"   Unique products: {df['product_id'].nunique():,}")
        print(f"   Unique stores: {df['store_id'].nunique():,}")
        
        # Revenue analysis
        print("\n2. Revenue Analysis:")
        total_revenue = df['net_total'].sum()
        avg_order_value = df.groupby('order_id')['net_total'].sum().mean()
        print(f"   Total revenue: ${total_revenue:,.2f}")
        print(f"   Average order value: ${avg_order_value:.2f}")
        print(f"   Revenue by customer segment:")
        segment_revenue = df.groupby('customer_segment')['net_total'].sum().sort_values(ascending=False)
        for segment, revenue in segment_revenue.items():
            print(f"     {segment}: ${revenue:,.2f}")
        
        # Product performance
        print("\n3. Product Performance:")
        product_sales = df.groupby('product_id').agg({
            'quantity': 'sum',
            'net_total': 'sum',
            'product_name': 'first',
            'category': 'first'
        }).sort_values('net_total', ascending=False)
        
        print(f"   Top 5 products by revenue:")
        for idx, (product_id, row) in enumerate(product_sales.head().iterrows(), 1):
            print(f"     {idx}. {row['product_name']} (${row['net_total']:,.2f})")
        
        # Customer analysis
        print("\n4. Customer Analysis:")
        customer_stats = df.groupby('customer_id').agg({
            'order_id': 'nunique',
            'net_total': 'sum',
            'customer_segment': 'first'
        }).rename(columns={'order_id': 'total_orders'})
        
        print(f"   Average orders per customer: {customer_stats['total_orders'].mean():.1f}")
        print(f"   Average customer lifetime value: ${customer_stats['net_total'].mean():.2f}")
        
        # Time-based analysis
        print("\n5. Time-based Analysis:")
        monthly_revenue = df.groupby(df['order_date'].dt.to_period('M'))['net_total'].sum()
        print(f"   Best month: {monthly_revenue.idxmax()} (${monthly_revenue.max():,.2f})")
        print(f"   Worst month: {monthly_revenue.idxmin()} (${monthly_revenue.min():,.2f})")
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        try:
            df = self.data['analytical_dataset']
            
            # Set up the plotting style
            try:
                plt.style.use('seaborn-v0_8')
            except:
                plt.style.use('default')
            fig = plt.figure(figsize=(20, 15))
            
            # 1. Revenue by Customer Segment
            plt.subplot(3, 3, 1)
        segment_revenue = df.groupby('customer_segment')['net_total'].sum().sort_values(ascending=True)
        segment_revenue.plot(kind='barh', color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Revenue by Customer Segment')
        plt.xlabel('Revenue ($)')
        
        # 2. Monthly Revenue Trend
        plt.subplot(3, 3, 2)
        monthly_revenue = df.groupby(df['order_date'].dt.to_period('M'))['net_total'].sum()
        monthly_revenue.plot(kind='line', marker='o', color='#FF6B6B')
        plt.title('Monthly Revenue Trend')
        plt.xlabel('Month')
        plt.ylabel('Revenue ($)')
        plt.xticks(rotation=45)
        
        # 3. Product Category Performance
        plt.subplot(3, 3, 3)
        category_revenue = df.groupby('category')['net_total'].sum().sort_values(ascending=True)
        category_revenue.plot(kind='barh', color='#4ECDC4')
        plt.title('Revenue by Product Category')
        plt.xlabel('Revenue ($)')
        
        # 4. Order Value Distribution
        plt.subplot(3, 3, 4)
        order_values = df.groupby('order_id')['net_total'].sum()
        plt.hist(order_values, bins=50, color='#45B7D1', alpha=0.7)
        plt.title('Order Value Distribution')
        plt.xlabel('Order Value ($)')
        plt.ylabel('Frequency')
        
        # 5. Customer Age Distribution
        plt.subplot(3, 3, 5)
        plt.hist(df['age'], bins=20, color='#96CEB4', alpha=0.7)
        plt.title('Customer Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        
        # 6. Store Performance
        plt.subplot(3, 3, 6)
        store_revenue = df.groupby('store_name')['net_total'].sum().sort_values(ascending=True)
        store_revenue.plot(kind='barh', color='#FECA57')
        plt.title('Revenue by Store')
        plt.xlabel('Revenue ($)')
        
        # 7. Payment Method Distribution
        plt.subplot(3, 3, 7)
        payment_counts = df['payment_method'].value_counts()
        plt.pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%')
        plt.title('Payment Method Distribution')
        
        # 8. Weekend vs Weekday Sales
        plt.subplot(3, 3, 8)
        weekend_sales = df.groupby('is_weekend')['net_total'].sum()
        weekend_sales.plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
        plt.title('Weekend vs Weekday Sales')
        plt.xlabel('Is Weekend')
        plt.ylabel('Revenue ($)')
        plt.xticks([0, 1], ['Weekday', 'Weekend'], rotation=0)
        
        # 9. Customer Segment Distribution
        plt.subplot(3, 3, 9)
        segment_counts = df['customer_segment'].value_counts()
        plt.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        plt.title('Customer Segment Distribution')
        
        plt.tight_layout()
        plt.savefig('retail_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'retail_analysis_dashboard.png'")
        except Exception as e:
            print(f"Visualization error: {e}")
            print("Skipping visualizations due to error")
    
    def customer_segmentation_ml(self):
        """Perform customer segmentation using machine learning"""
        print("\n=== CUSTOMER SEGMENTATION ML ===")
        
        # Prepare customer features
        customer_features = self.data['analytical_dataset'].groupby('customer_id').agg({
            'net_total': ['sum', 'mean', 'count'],
            'quantity': 'sum',
            'age': 'first',
            'days_since_registration': 'first',
            'order_date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = ['customer_id', 'total_spent', 'avg_order_value', 'total_orders', 
                                   'total_quantity', 'age', 'days_since_registration', 'first_order', 'last_order']
        
        # Calculate additional features
        customer_features['days_active'] = (customer_features['last_order'] - customer_features['first_order']).dt.days
        customer_features['order_frequency'] = customer_features['total_orders'] / (customer_features['days_active'] + 1)
        customer_features['avg_days_between_orders'] = customer_features['days_active'] / (customer_features['total_orders'] - 1)
        
        # Handle missing values
        customer_features = customer_features.fillna(0)
        
        # Select features for clustering
        feature_columns = ['total_spent', 'avg_order_value', 'total_orders', 'total_quantity', 
                          'age', 'days_since_registration', 'order_frequency']
        X = customer_features[feature_columns]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        customer_features['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_analysis = customer_features.groupby('cluster')[feature_columns].mean()
        print("Customer Segments:")
        for cluster_id, row in cluster_analysis.iterrows():
            print(f"\nCluster {cluster_id}:")
            print(f"  Average total spent: ${row['total_spent']:,.2f}")
            print(f"  Average order value: ${row['avg_order_value']:.2f}")
            print(f"  Average orders: {row['total_orders']:.1f}")
            print(f"  Average age: {row['age']:.1f}")
            print(f"  Average order frequency: {row['order_frequency']:.3f}")
        
        return customer_features
    
    def predict_customer_lifetime_value(self):
        """Predict customer lifetime value using machine learning"""
        print("\n=== CUSTOMER LIFETIME VALUE PREDICTION ===")
        
        # Prepare features for CLV prediction
        customer_clv = self.data['analytical_dataset'].groupby('customer_id').agg({
            'net_total': ['sum', 'mean', 'count'],
            'quantity': 'sum',
            'age': 'first',
            'days_since_registration': 'first',
            'customer_segment': 'first',
            'order_date': ['min', 'max']
        }).reset_index()
        
        customer_clv.columns = ['customer_id', 'total_spent', 'avg_order_value', 'total_orders',
                              'total_quantity', 'age', 'days_since_registration', 'segment',
                              'first_order', 'last_order']
        
        # Calculate target variable (future CLV)
        customer_clv['days_active'] = (customer_clv['last_order'] - customer_clv['first_order']).dt.days
        customer_clv['monthly_spend'] = customer_clv['total_spent'] / (customer_clv['days_active'] / 30 + 1)
        customer_clv['predicted_clv'] = customer_clv['monthly_spend'] * 12  # Annual CLV
        
        # Prepare features
        le = LabelEncoder()
        customer_clv['segment_encoded'] = le.fit_transform(customer_clv['segment'])
        
        feature_columns = ['avg_order_value', 'total_orders', 'total_quantity', 'age', 
                          'days_since_registration', 'segment_encoded']
        X = customer_clv[feature_columns]
        y = customer_clv['predicted_clv']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"  R² Score: {r2:.3f}")
        print(f"  RMSE: ${np.sqrt(mse):,.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return model, customer_clv
    
    def product_recommendation_system(self):
        """Build a simple product recommendation system"""
        print("\n=== PRODUCT RECOMMENDATION SYSTEM ===")
        
        # Create customer-product matrix
        customer_product_matrix = self.data['analytical_dataset'].groupby(['customer_id', 'product_id']).agg({
            'quantity': 'sum',
            'net_total': 'sum'
        }).reset_index()
        
        # Calculate customer-product affinity scores
        customer_product_matrix['affinity_score'] = (
            customer_product_matrix['quantity'] * 0.3 + 
            (customer_product_matrix['net_total'] / customer_product_matrix['net_total'].max()) * 0.7
        )
        
        # Get top products for each customer
        top_products = customer_product_matrix.groupby('customer_id').apply(
            lambda x: x.nlargest(3, 'affinity_score')
        ).reset_index(drop=True)
        
        print("Sample Product Recommendations:")
        sample_customers = top_products['customer_id'].unique()[:5]
        for customer_id in sample_customers:
            customer_recs = top_products[top_products['customer_id'] == customer_id]
            print(f"\nCustomer {customer_id}:")
            for _, rec in customer_recs.iterrows():
                product_matches = self.data['analytical_dataset'][
                    self.data['analytical_dataset']['product_id'] == rec['product_id']
                ]
                if len(product_matches) > 0:
                    product_name = product_matches['product_name'].iloc[0]
                    print(f"  - {product_name} (Score: {rec['affinity_score']:.3f})")
                else:
                    print(f"  - Product {rec['product_id']} (Score: {rec['affinity_score']:.3f})")
        
        return top_products
    
    def run_complete_analysis(self):
        """Run complete data analysis pipeline"""
        print("=== RETAIL DATA ANALYSIS PIPELINE ===\n")
        
        # Load data
        self.load_data()
        
        # Exploratory analysis
        self.perform_exploratory_analysis()
        
        # Create visualizations
        self.create_visualizations()
        
        # Customer segmentation
        customer_segments = self.customer_segmentation_ml()
        
        # CLV prediction
        clv_model, clv_data = self.predict_customer_lifetime_value()
        
        # Product recommendations
        recommendations = self.product_recommendation_system()
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("All analysis components have been executed successfully!")
        
        return {
            'customer_segments': customer_segments,
            'clv_model': clv_model,
            'clv_data': clv_data,
            'recommendations': recommendations
        }

def main():
    """Main function to run the analysis"""
    analyzer = RetailDataAnalyzer()
    results = analyzer.run_complete_analysis()
    print("\nAnalysis results available in 'results' variable")

if __name__ == "__main__":
    main()
