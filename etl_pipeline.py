"""
ETL Pipeline for Retail Data Modeling
Demonstrates data extraction, transformation, and loading processes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetailETLPipeline:
    """ETL Pipeline for retail data processing and modeling"""
    
    def __init__(self, db_path: str = "retail_data.db"):
        self.db_path = db_path
        self.connection = None
        
    def connect_database(self):
        """Establish database connection"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def close_database(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def extract_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Extract sample retail data (simulating data source)"""
        logger.info("Extracting sample retail data...")
        
        # Generate sample customers
        customers_data = {
            'customer_id': [f"CUST_{i:04d}" for i in range(1, 1001)],
            'customer_name': [f"Customer {i}" for i in range(1, 1001)],
            'email': [f"customer{i}@email.com" for i in range(1, 1001)],
            'age': np.random.randint(18, 80, 1000),
            'customer_segment': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 1000, p=[0.4, 0.3, 0.2, 0.1]),
            'registration_date': pd.date_range('2020-01-01', periods=1000, freq='D').tolist()
        }
        customers_df = pd.DataFrame(customers_data)
        
        # Generate sample products
        categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Beauty']
        products_data = {
            'product_id': [f"PROD_{i:04d}" for i in range(1, 501)],
            'product_name': [f"Product {i}" for i in range(1, 501)],
            'price': np.random.uniform(10, 500, 500),
            'cost': np.random.uniform(5, 250, 500),
            'weight_kg': np.random.uniform(0.1, 10, 500),
            'category': np.random.choice(categories, 500)
        }
        products_df = pd.DataFrame(products_data)
        
        # Generate sample stores
        stores_data = {
            'store_id': [f"STORE_{i:03d}" for i in range(1, 21)],
            'store_name': [f"Store {i}" for i in range(1, 21)],
            'location': [f"Location {i}" for i in range(1, 21)],
            'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 20),
            'state': np.random.choice(['NY', 'CA', 'IL', 'TX', 'AZ'], 20),
            'country': ['USA'] * 20,
            'store_type': np.random.choice(['Physical', 'Online', 'Hybrid'], 20)
        }
        stores_df = pd.DataFrame(stores_data)
        
        # Generate sample orders
        orders_data = {
            'order_id': [f"ORDER_{i:06d}" for i in range(1, 5001)],
            'customer_id': np.random.choice(customers_df['customer_id'], 5000),
            'store_id': np.random.choice(stores_df['store_id'], 5000),
            'order_date': pd.date_range('2023-01-01', periods=5000, freq='H').tolist(),
            'order_total': np.random.uniform(20, 1000, 5000),
            'order_status': np.random.choice(['Completed', 'Pending', 'Cancelled', 'Refunded'], 5000, p=[0.8, 0.1, 0.05, 0.05]),
            'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash'], 5000)
        }
        orders_df = pd.DataFrame(orders_data)
        
        # Generate sample order items
        order_items_data = []
        for order in orders_df.itertuples():
            num_items = np.random.randint(1, 6)
            for _ in range(num_items):
                product = products_df.sample(1).iloc[0]
                quantity = np.random.randint(1, 5)
                unit_price = product['price']
                line_total = unit_price * quantity
                
                order_items_data.append({
                    'order_item_id': str(uuid.uuid4()),
                    'order_id': order.order_id,
                    'product_id': product['product_id'],
                    'quantity': quantity,
                    'unit_price': unit_price,
                    'line_total': line_total
                })
        
        order_items_df = pd.DataFrame(order_items_data)
        
        return {
            'customers': customers_df,
            'products': products_df,
            'stores': stores_df,
            'orders': orders_df,
            'order_items': order_items_df
        }
    
    def transform_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Transform raw data according to business rules"""
        logger.info("Transforming data...")
        
        transformed_data = {}
        
        # Transform customers
        customers_df = raw_data['customers'].copy()
        customers_df['age_group'] = pd.cut(customers_df['age'], 
                                         bins=[0, 25, 35, 50, 65, 100], 
                                         labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        customers_df['is_premium'] = customers_df['customer_segment'].isin(['Gold', 'Platinum'])
        customers_df['registration_year'] = pd.to_datetime(customers_df['registration_date']).dt.year
        transformed_data['customers'] = customers_df
        
        # Transform products
        products_df = raw_data['products'].copy()
        products_df['price_range'] = pd.cut(products_df['price'], 
                                          bins=[0, 50, 100, 200, 500, float('inf')], 
                                          labels=['Budget', 'Mid-range', 'Premium', 'Luxury', 'Ultra-luxury'])
        products_df['is_premium'] = products_df['price'] > 100
        products_df['gross_margin'] = products_df['price'] - products_df['cost']
        products_df['margin_percentage'] = (products_df['gross_margin'] / products_df['price']) * 100
        transformed_data['products'] = products_df
        
        # Transform orders
        orders_df = raw_data['orders'].copy()
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
        orders_df['order_year'] = orders_df['order_date'].dt.year
        orders_df['order_quarter'] = orders_df['order_date'].dt.quarter
        orders_df['order_month'] = orders_df['order_date'].dt.month
        orders_df['order_day_of_week'] = orders_df['order_date'].dt.day_name()
        orders_df['is_weekend'] = orders_df['order_date'].dt.dayofweek.isin([5, 6])
        transformed_data['orders'] = orders_df
        
        # Transform order items
        order_items_df = raw_data['order_items'].copy()
        order_items_df['discount_amount'] = np.random.uniform(0, order_items_df['line_total'] * 0.2, len(order_items_df))
        order_items_df['tax_amount'] = order_items_df['line_total'] * 0.08  # 8% tax
        order_items_df['net_total'] = order_items_df['line_total'] - order_items_df['discount_amount'] + order_items_df['tax_amount']
        transformed_data['order_items'] = order_items_df
        
        return transformed_data
    
    def load_to_database(self, transformed_data: Dict[str, pd.DataFrame]):
        """Load transformed data into database"""
        logger.info("Loading data to database...")
        
        if not self.connection:
            self.connect_database()
        
        try:
            # Create tables
            self._create_tables()
            
            # Load data
            for table_name, df in transformed_data.items():
                df.to_sql(table_name, self.connection, if_exists='replace', index=False)
                logger.info(f"Loaded {len(df)} records into {table_name}")
            
            self.connection.commit()
            logger.info("Data loading completed successfully")
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            self.connection.rollback()
            raise
    
    def _create_tables(self):
        """Create database tables"""
        cursor = self.connection.cursor()
        
        # Create customers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                customer_name TEXT,
                email TEXT,
                age INTEGER,
                customer_segment TEXT,
                registration_date TEXT,
                age_group TEXT,
                is_premium BOOLEAN,
                registration_year INTEGER
            )
        """)
        
        # Create products table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                product_name TEXT,
                price REAL,
                cost REAL,
                weight_kg REAL,
                category TEXT,
                price_range TEXT,
                is_premium BOOLEAN,
                gross_margin REAL,
                margin_percentage REAL
            )
        """)
        
        # Create stores table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stores (
                store_id TEXT PRIMARY KEY,
                store_name TEXT,
                location TEXT,
                city TEXT,
                state TEXT,
                country TEXT,
                store_type TEXT
            )
        """)
        
        # Create orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT,
                store_id TEXT,
                order_date TEXT,
                order_total REAL,
                order_status TEXT,
                payment_method TEXT,
                order_year INTEGER,
                order_quarter INTEGER,
                order_month INTEGER,
                order_day_of_week TEXT,
                is_weekend BOOLEAN,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
                FOREIGN KEY (store_id) REFERENCES stores(store_id)
            )
        """)
        
        # Create order_items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_items (
                order_item_id TEXT PRIMARY KEY,
                order_id TEXT,
                product_id TEXT,
                quantity INTEGER,
                unit_price REAL,
                line_total REAL,
                discount_amount REAL,
                tax_amount REAL,
                net_total REAL,
                FOREIGN KEY (order_id) REFERENCES orders(order_id),
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        """)
        
        self.connection.commit()
    
    def run_pipeline(self):
        """Execute the complete ETL pipeline"""
        logger.info("Starting ETL pipeline...")
        
        try:
            # Extract
            raw_data = self.extract_sample_data()
            logger.info(f"Extracted {sum(len(df) for df in raw_data.values())} total records")
            
            # Transform
            transformed_data = self.transform_data(raw_data)
            logger.info("Data transformation completed")
            
            # Load
            self.load_to_database(transformed_data)
            logger.info("ETL pipeline completed successfully")
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            raise
        finally:
            self.close_database()

def main():
    """Main function to run the ETL pipeline"""
    pipeline = RetailETLPipeline()
    transformed_data = pipeline.run_pipeline()
    
    # Display summary
    print("\n=== ETL Pipeline Summary ===")
    for table_name, df in transformed_data.items():
        print(f"{table_name}: {len(df)} records")
    
    print(f"\nDatabase created: retail_data.db")

if __name__ == "__main__":
    main()
