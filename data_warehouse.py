"""
Data Warehouse Implementation
Demonstrates data warehousing principles and dimensional modeling
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import json

class DataWarehouse:
    """Data warehouse implementation with dimensional modeling"""
    
    def __init__(self, db_path: str = "data_warehouse.db"):
        self.db_path = db_path
        self.connection = None
        
    def connect_database(self):
        """Establish database connection"""
        self.connection = sqlite3.connect(self.db_path)
        
    def close_database(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
    
    def create_dimension_tables(self):
        """Create dimension tables for the data warehouse"""
        cursor = self.connection.cursor()
        
        # Date Dimension
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dim_date (
                date_sk INTEGER PRIMARY KEY,
                full_date DATE,
                year INTEGER,
                quarter INTEGER,
                month INTEGER,
                month_name VARCHAR(20),
                day_of_year INTEGER,
                day_of_week INTEGER,
                day_name VARCHAR(20),
                is_weekend BOOLEAN,
                is_holiday BOOLEAN,
                fiscal_year INTEGER,
                fiscal_quarter INTEGER,
                week_of_year INTEGER,
                quarter_name VARCHAR(10)
            )
        """)
        
        # Customer Dimension
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dim_customer (
                customer_sk INTEGER PRIMARY KEY,
                customer_id VARCHAR(50),
                customer_name VARCHAR(100),
                email VARCHAR(100),
                age INTEGER,
                age_group VARCHAR(20),
                customer_segment VARCHAR(50),
                registration_date DATE,
                registration_year INTEGER,
                is_premium BOOLEAN,
                customer_lifetime_value DECIMAL(12,2),
                total_orders INTEGER,
                effective_date DATE,
                expiry_date DATE,
                is_current BOOLEAN
            )
        """)
        
        # Product Dimension
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dim_product (
                product_sk INTEGER PRIMARY KEY,
                product_id VARCHAR(50),
                product_name VARCHAR(200),
                category VARCHAR(100),
                subcategory VARCHAR(100),
                brand VARCHAR(100),
                price DECIMAL(10,2),
                cost DECIMAL(10,2),
                price_range VARCHAR(20),
                is_premium BOOLEAN,
                weight_kg DECIMAL(8,3),
                dimensions VARCHAR(50),
                effective_date DATE,
                expiry_date DATE,
                is_current BOOLEAN
            )
        """)
        
        # Store Dimension
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dim_store (
                store_sk INTEGER PRIMARY KEY,
                store_id VARCHAR(50),
                store_name VARCHAR(100),
                location VARCHAR(200),
                city VARCHAR(100),
                state VARCHAR(50),
                country VARCHAR(50),
                store_type VARCHAR(50),
                region VARCHAR(50),
                effective_date DATE,
                expiry_date DATE,
                is_current BOOLEAN
            )
        """)
        
        # Geography Dimension
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dim_geography (
                geography_sk INTEGER PRIMARY KEY,
                country VARCHAR(50),
                state VARCHAR(50),
                city VARCHAR(100),
                postal_code VARCHAR(20),
                region VARCHAR(50),
                timezone VARCHAR(50),
                population INTEGER,
                effective_date DATE,
                expiry_date DATE,
                is_current BOOLEAN
            )
        """)
        
        self.connection.commit()
    
    def create_fact_tables(self):
        """Create fact tables for the data warehouse"""
        cursor = self.connection.cursor()
        
        # Sales Fact Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_sales (
                sales_sk INTEGER PRIMARY KEY,
                customer_sk INTEGER,
                product_sk INTEGER,
                store_sk INTEGER,
                date_sk INTEGER,
                geography_sk INTEGER,
                order_id VARCHAR(50),
                order_item_id VARCHAR(50),
                quantity INTEGER,
                unit_price DECIMAL(10,2),
                line_total DECIMAL(12,2),
                discount_amount DECIMAL(10,2),
                tax_amount DECIMAL(10,2),
                net_total DECIMAL(12,2),
                gross_profit DECIMAL(12,2),
                margin_percentage DECIMAL(5,2),
                FOREIGN KEY (customer_sk) REFERENCES dim_customer(customer_sk),
                FOREIGN KEY (product_sk) REFERENCES dim_product(product_sk),
                FOREIGN KEY (store_sk) REFERENCES dim_store(store_sk),
                FOREIGN KEY (date_sk) REFERENCES dim_date(date_sk),
                FOREIGN KEY (geography_sk) REFERENCES dim_geography(geography_sk)
            )
        """)
        
        # Customer Metrics Fact Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_customer_metrics (
                customer_sk INTEGER,
                date_sk INTEGER,
                total_orders INTEGER,
                total_revenue DECIMAL(12,2),
                avg_order_value DECIMAL(10,2),
                days_since_last_order INTEGER,
                order_frequency DECIMAL(8,4),
                customer_lifetime_value DECIMAL(12,2),
                churn_risk_score DECIMAL(5,4),
                PRIMARY KEY (customer_sk, date_sk),
                FOREIGN KEY (customer_sk) REFERENCES dim_customer(customer_sk),
                FOREIGN KEY (date_sk) REFERENCES dim_date(date_sk)
            )
        """)
        
        # Product Performance Fact Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_product_performance (
                product_sk INTEGER,
                date_sk INTEGER,
                total_quantity_sold INTEGER,
                total_revenue DECIMAL(12,2),
                total_orders INTEGER,
                avg_selling_price DECIMAL(10,2),
                inventory_turnover DECIMAL(8,4),
                profit_margin DECIMAL(5,2),
                PRIMARY KEY (product_sk, date_sk),
                FOREIGN KEY (product_sk) REFERENCES dim_product(product_sk),
                FOREIGN KEY (date_sk) REFERENCES dim_date(date_sk)
            )
        """)
        
        self.connection.commit()
    
    def populate_date_dimension(self, start_date: str, end_date: str):
        """Populate the date dimension table"""
        cursor = self.connection.cursor()
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        date_data = []
        current_date = start
        
        while current_date <= end:
            date_sk = int(current_date.strftime('%Y%m%d'))
            
            # Calculate fiscal year (assuming fiscal year starts in April)
            fiscal_year = current_date.year
            if current_date.month < 4:
                fiscal_year -= 1
            
            fiscal_quarter = ((current_date.month - 1) // 3) + 1
            if current_date.month < 4:
                fiscal_quarter = ((current_date.month + 9) // 3)
            
            date_data.append({
                'date_sk': date_sk,
                'full_date': current_date.date(),
                'year': current_date.year,
                'quarter': ((current_date.month - 1) // 3) + 1,
                'month': current_date.month,
                'month_name': current_date.strftime('%B'),
                'day_of_year': current_date.timetuple().tm_yday,
                'day_of_week': current_date.weekday() + 1,
                'day_name': current_date.strftime('%A'),
                'is_weekend': current_date.weekday() >= 5,
                'is_holiday': self._is_holiday(current_date),
                'fiscal_year': fiscal_year,
                'fiscal_quarter': fiscal_quarter,
                'week_of_year': current_date.isocalendar()[1],
                'quarter_name': f"Q{((current_date.month - 1) // 3) + 1}"
            })
            
            current_date += timedelta(days=1)
        
        # Insert data
        for date_info in date_data:
            cursor.execute("""
                INSERT OR REPLACE INTO dim_date VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(date_info.values()))
        
        self.connection.commit()
        print(f"Populated date dimension with {len(date_data)} records")
    
    def _is_holiday(self, date: datetime) -> bool:
        """Simple holiday detection (can be enhanced)"""
        # Major US holidays
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (12, 25), # Christmas
            (11, 24), # Thanksgiving (approximate)
        ]
        
        return (date.month, date.day) in holidays
    
    def populate_customer_dimension(self, source_data: pd.DataFrame):
        """Populate customer dimension from source data"""
        cursor = self.connection.cursor()
        
        # Calculate customer metrics
        customer_metrics = source_data.groupby('customer_id').agg({
            'net_total': 'sum',
            'order_id': 'nunique',
            'customer_name': 'first',
            'email': 'first',
            'age': 'first',
            'customer_segment': 'first',
            'registration_date': 'first'
        }).reset_index()
        
        customer_metrics.columns = ['customer_id', 'total_spent', 'total_orders', 'customer_name', 
                                  'email', 'age', 'segment', 'registration_date']
        
        customer_metrics['avg_order_value'] = customer_metrics['total_spent'] / customer_metrics['total_orders']
        customer_metrics['age_group'] = pd.cut(customer_metrics['age'], 
                                             bins=[0, 25, 35, 50, 65, 100], 
                                             labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        customer_metrics['is_premium'] = customer_metrics['segment'].isin(['Gold', 'Platinum'])
        customer_metrics['registration_year'] = pd.to_datetime(customer_metrics['registration_date']).dt.year
        
        # Insert data
        for idx, row in customer_metrics.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO dim_customer VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                idx + 1,  # customer_sk
                row['customer_id'],
                row['customer_name'],
                row['email'],
                row['age'],
                row['age_group'],
                row['segment'],
                row['registration_date'],
                row['registration_year'],
                row['is_premium'],
                row['total_spent'],  # customer_lifetime_value
                row['total_orders'],
                '2020-01-01',  # effective_date
                '2099-12-31',  # expiry_date
                True  # is_current
            ))
        
        self.connection.commit()
        print(f"Populated customer dimension with {len(customer_metrics)} records")
    
    def populate_product_dimension(self, source_data: pd.DataFrame):
        """Populate product dimension from source data"""
        cursor = self.connection.cursor()
        
        # Get unique products
        products = source_data[['product_id', 'product_name', 'price', 'cost', 'weight_kg', 'category']].drop_duplicates()
        
        products['price_range'] = pd.cut(products['price'], 
                                       bins=[0, 50, 100, 200, 500, float('inf')], 
                                       labels=['Budget', 'Mid-range', 'Premium', 'Luxury', 'Ultra-luxury'])
        products['is_premium'] = products['price'] > 100
        products['gross_margin'] = products['price'] - products['cost']
        products['margin_percentage'] = (products['gross_margin'] / products['price']) * 100
        
        # Insert data
        for idx, row in products.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO dim_product VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                idx + 1,  # product_sk
                row['product_id'],
                row['product_name'],
                row['category'],
                None,  # subcategory
                None,  # brand
                row['price'],
                row['cost'],
                row['price_range'],
                row['is_premium'],
                row['weight_kg'],
                None,  # dimensions
                '2020-01-01',  # effective_date
                '2099-12-31',  # expiry_date
                True  # is_current
            ))
        
        self.connection.commit()
        print(f"Populated product dimension with {len(products)} records")
    
    def populate_sales_fact(self, source_data: pd.DataFrame):
        """Populate sales fact table"""
        cursor = self.connection.cursor()
        
        # Add surrogate keys to source data
        source_data['date_sk'] = pd.to_datetime(source_data['order_date']).dt.strftime('%Y%m%d').astype(int)
        
        # Get customer SKs
        customer_sk_map = {}
        cursor.execute("SELECT customer_sk, customer_id FROM dim_customer")
        for row in cursor.fetchall():
            customer_sk_map[row[1]] = row[0]
        
        # Get product SKs
        product_sk_map = {}
        cursor.execute("SELECT product_sk, product_id FROM dim_product")
        for row in cursor.fetchall():
            product_sk_map[row[1]] = row[0]
        
        # Get store SKs (assuming store_sk = store_id for simplicity)
        store_sk_map = {}
        cursor.execute("SELECT store_sk, store_id FROM dim_store")
        for row in cursor.fetchall():
            store_sk_map[row[1]] = row[0]
        
        # Calculate metrics
        source_data['gross_profit'] = source_data['net_total'] - source_data['discount_amount']
        source_data['margin_percentage'] = (source_data['gross_profit'] / source_data['net_total']) * 100
        
        # Insert fact records
        for idx, row in source_data.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO fact_sales VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                idx + 1,  # sales_sk
                customer_sk_map.get(row['customer_id'], 1),
                product_sk_map.get(row['product_id'], 1),
                store_sk_map.get(row['store_id'], 1),
                row['date_sk'],
                1,  # geography_sk (default)
                row['order_id'],
                row['order_item_id'],
                row['quantity'],
                row['unit_price'],
                row['line_total'],
                row['discount_amount'],
                row['tax_amount'],
                row['net_total'],
                row['gross_profit'],
                row['margin_percentage']
            ))
        
        self.connection.commit()
        print(f"Populated sales fact table with {len(source_data)} records")
    
    def create_aggregated_tables(self):
        """Create aggregated fact tables for better performance"""
        cursor = self.connection.cursor()
        
        # Daily sales aggregation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_daily_sales AS
            SELECT 
                date_sk,
                customer_sk,
                product_sk,
                store_sk,
                SUM(quantity) as total_quantity,
                SUM(net_total) as total_revenue,
                COUNT(DISTINCT order_id) as total_orders,
                AVG(unit_price) as avg_unit_price,
                SUM(gross_profit) as total_profit
            FROM fact_sales
            GROUP BY date_sk, customer_sk, product_sk, store_sk
        """)
        
        # Monthly sales aggregation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_monthly_sales AS
            SELECT 
                (date_sk / 100) as month_sk,
                customer_sk,
                product_sk,
                store_sk,
                SUM(quantity) as total_quantity,
                SUM(net_total) as total_revenue,
                COUNT(DISTINCT order_id) as total_orders,
                AVG(unit_price) as avg_unit_price,
                SUM(gross_profit) as total_profit
            FROM fact_sales
            GROUP BY month_sk, customer_sk, product_sk, store_sk
        """)
        
        self.connection.commit()
        print("Created aggregated fact tables")
    
    def create_indexes(self):
        """Create indexes for better query performance"""
        cursor = self.connection.cursor()
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_fact_sales_date ON fact_sales(date_sk)",
            "CREATE INDEX IF NOT EXISTS idx_fact_sales_customer ON fact_sales(customer_sk)",
            "CREATE INDEX IF NOT EXISTS idx_fact_sales_product ON fact_sales(product_sk)",
            "CREATE INDEX IF NOT EXISTS idx_fact_sales_store ON fact_sales(store_sk)",
            "CREATE INDEX IF NOT EXISTS idx_dim_date_full_date ON dim_date(full_date)",
            "CREATE INDEX IF NOT EXISTS idx_dim_customer_id ON dim_customer(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_dim_product_id ON dim_product(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_dim_store_id ON dim_store(store_id)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        self.connection.commit()
        print("Created performance indexes")
    
    def run_data_warehouse_setup(self, source_data: pd.DataFrame):
        """Run complete data warehouse setup"""
        print("=== DATA WAREHOUSE SETUP ===")
        
        self.connect_database()
        
        # Create tables
        self.create_dimension_tables()
        self.create_fact_tables()
        
        # Populate dimensions
        self.populate_date_dimension('2020-01-01', '2024-12-31')
        self.populate_customer_dimension(source_data)
        self.populate_product_dimension(source_data)
        
        # Populate facts
        self.populate_sales_fact(source_data)
        
        # Create aggregations and indexes
        self.create_aggregated_tables()
        self.create_indexes()
        
        print("Data warehouse setup completed successfully!")
        
        self.close_database()

def main():
    """Main function to demonstrate data warehouse setup"""
    # This would typically be called with source data
    print("Data warehouse implementation ready!")
    print("Use run_data_warehouse_setup() method with source data to populate the warehouse.")

if __name__ == "__main__":
    main()
