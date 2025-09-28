#!/bin/bash

# Enterprise Data Modeling Application Entrypoint
set -e

echo "🚀 Starting Enterprise Data Modeling Application..."

# Wait for database to be ready
echo "⏳ Waiting for database connection..."
python -c "
import time
import psycopg2
import os

max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=os.getenv('POSTGRES_PORT', '5432'),
            database=os.getenv('POSTGRES_DB', 'retail_analytics'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'password')
        )
        conn.close()
        print('✅ Database connection established')
        break
    except psycopg2.OperationalError:
        retry_count += 1
        print(f'⏳ Waiting for database... ({retry_count}/{max_retries})')
        time.sleep(2)
else:
    print('❌ Failed to connect to database')
    exit(1)
"

# Initialize database schema
echo "📊 Initializing database schema..."
python -c "
import sqlalchemy
import os

# Create database engine
engine = sqlalchemy.create_engine(
    f'postgresql://{os.getenv(\"POSTGRES_USER\", \"postgres\")}:{os.getenv(\"POSTGRES_PASSWORD\", \"password\")}@{os.getenv(\"POSTGRES_HOST\", \"postgres\")}:{os.getenv(\"POSTGRES_PORT\", \"5432\")}/{os.getenv(\"POSTGRES_DB\", \"retail_analytics\")}'
)

# Create tables
from sqlalchemy import text
with engine.connect() as conn:
    # Create customers table
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS customers (
            customer_id VARCHAR(50) PRIMARY KEY,
            customer_name VARCHAR(100),
            email VARCHAR(100),
            age INTEGER,
            customer_segment VARCHAR(50),
            registration_date DATE,
            city VARCHAR(100),
            state VARCHAR(50),
            country VARCHAR(50),
            phone VARCHAR(20),
            loyalty_points INTEGER,
            preferred_payment_method VARCHAR(50),
            marketing_consent BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    '''))
    
    # Create products table
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS products (
            product_id VARCHAR(50) PRIMARY KEY,
            product_name VARCHAR(200),
            category VARCHAR(100),
            subcategory VARCHAR(100),
            brand VARCHAR(100),
            price DECIMAL(10,2),
            cost DECIMAL(10,2),
            weight_kg DECIMAL(8,3),
            dimensions VARCHAR(50),
            color VARCHAR(50),
            size VARCHAR(20),
            material VARCHAR(50),
            warranty_months INTEGER,
            in_stock BOOLEAN,
            rating DECIMAL(3,1),
            review_count INTEGER,
            launch_date DATE,
            discontinued BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    '''))
    
    # Create stores table
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS stores (
            store_id VARCHAR(50) PRIMARY KEY,
            store_name VARCHAR(100),
            store_type VARCHAR(50),
            city VARCHAR(100),
            state VARCHAR(50),
            country VARCHAR(50),
            address VARCHAR(200),
            postal_code VARCHAR(20),
            phone VARCHAR(20),
            email VARCHAR(100),
            manager_name VARCHAR(100),
            opening_date DATE,
            square_footage INTEGER,
            employee_count INTEGER,
            parking_spaces INTEGER,
            wifi_available BOOLEAN,
            restaurant_available BOOLEAN,
            pharmacy_available BOOLEAN,
            grocery_available BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    '''))
    
    # Create orders table
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS orders (
            order_id VARCHAR(50) PRIMARY KEY,
            customer_id VARCHAR(50),
            store_id VARCHAR(50),
            order_date TIMESTAMP,
            order_total DECIMAL(12,2),
            order_status VARCHAR(50),
            payment_method VARCHAR(50),
            shipping_address VARCHAR(200),
            shipping_city VARCHAR(100),
            shipping_state VARCHAR(50),
            shipping_postal_code VARCHAR(20),
            shipping_method VARCHAR(50),
            shipping_cost DECIMAL(10,2),
            tax_amount DECIMAL(10,2),
            discount_amount DECIMAL(10,2),
            loyalty_points_earned INTEGER,
            loyalty_points_used INTEGER,
            coupon_code VARCHAR(50),
            gift_wrap BOOLEAN,
            gift_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (store_id) REFERENCES stores(store_id)
        )
    '''))
    
    # Create order_items table
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS order_items (
            order_item_id VARCHAR(50) PRIMARY KEY,
            order_id VARCHAR(50),
            product_id VARCHAR(50),
            quantity INTEGER,
            unit_price DECIMAL(10,2),
            line_total DECIMAL(12,2),
            discount_percent DECIMAL(5,2),
            discount_amount DECIMAL(10,2),
            final_total DECIMAL(12,2),
            tax_amount DECIMAL(10,2),
            net_total DECIMAL(12,2),
            returned BOOLEAN,
            return_date DATE,
            return_reason VARCHAR(100),
            refund_amount DECIMAL(12,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
    '''))
    
    conn.commit()
    print('✅ Database schema initialized')
"

# Run data generation if needed
echo "📊 Checking for existing data..."
python -c "
import sqlalchemy
import os
import pandas as pd

engine = sqlalchemy.create_engine(
    f'postgresql://{os.getenv(\"POSTGRES_USER\", \"postgres\")}:{os.getenv(\"POSTGRES_PASSWORD\", \"password\")}@{os.getenv(\"POSTGRES_HOST\", \"postgres\")}:{os.getenv(\"POSTGRES_PORT\", \"5432\")}/{os.getenv(\"POSTGRES_DB\", \"retail_analytics\")}'
)

# Check if data exists
with engine.connect() as conn:
    result = conn.execute(sqlalchemy.text('SELECT COUNT(*) FROM customers'))
    customer_count = result.scalar()
    
    if customer_count == 0:
        print('📊 No data found, generating sample data...')
        # Import and run data generation
        from comprehensive_demo import ComprehensiveDataModelingDemo
        demo = ComprehensiveDataModelingDemo()
        demo._generate_comprehensive_dataset()
        print('✅ Sample data generated')
    else:
        print(f'✅ Found {customer_count} customers in database')
"

# Start the application
echo "🚀 Starting Streamlit application..."
exec streamlit run enterprise_dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
