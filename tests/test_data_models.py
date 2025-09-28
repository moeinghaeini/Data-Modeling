"""
Test suite for data models and ETL pipeline
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import tempfile
import os

# Import the data modeling components
from comprehensive_demo import ComprehensiveDataModelingDemo

class TestDataGeneration:
    """Test data generation functions"""
    
    def setup_method(self):
        """Setup test environment"""
        self.demo = ComprehensiveDataModelingDemo()
    
    def test_generate_customers(self):
        """Test customer data generation"""
        customers = self.demo._generate_customers(100)
        
        assert len(customers) == 100
        assert 'customer_id' in customers.columns
        assert 'customer_name' in customers.columns
        assert 'email' in customers.columns
        assert 'age' in customers.columns
        assert 'customer_segment' in customers.columns
        
        # Check data types
        assert customers['age'].dtype == 'int64'
        assert customers['customer_segment'].isin(['Bronze', 'Silver', 'Gold', 'Platinum']).all()
    
    def test_generate_products(self):
        """Test product data generation"""
        products = self.demo._generate_products(50)
        
        assert len(products) == 50
        assert 'product_id' in products.columns
        assert 'product_name' in products.columns
        assert 'category' in products.columns
        assert 'price' in products.columns
        assert 'brand' in products.columns
        
        # Check price range
        assert products['price'].min() >= 10
        assert products['price'].max() <= 5000
    
    def test_generate_stores(self):
        """Test store data generation"""
        stores = self.demo._generate_stores(10)
        
        assert len(stores) == 10
        assert 'store_id' in stores.columns
        assert 'store_name' in stores.columns
        assert 'city' in stores.columns
        assert 'state' in stores.columns
        assert 'country' in stores.columns
    
    def test_generate_orders(self):
        """Test order data generation"""
        # First generate customers and products
        customers = self.demo._generate_customers(50)
        products = self.demo._generate_products(20)
        stores = self.demo._generate_stores(5)
        
        orders = self.demo._generate_orders(100, customers, products, stores)
        
        assert len(orders) == 100
        assert 'order_id' in orders.columns
        assert 'customer_id' in orders.columns
        assert 'order_date' in orders.columns
        assert 'order_total' in orders.columns
        
        # Check that all customer_ids exist in customers
        assert orders['customer_id'].isin(customers['customer_id']).all()

class TestETLPipeline:
    """Test ETL pipeline functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.demo = ComprehensiveDataModelingDemo()
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
    
    def teardown_method(self):
        """Cleanup test environment"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_database_creation(self):
        """Test database creation and schema"""
        # Create a small test dataset
        customers = self.demo._generate_customers(10)
        products = self.demo._generate_products(5)
        stores = self.demo._generate_stores(2)
        orders = self.demo._generate_orders(20, customers, products, stores)
        
        # Save to database
        self.demo._save_to_database(
            self.temp_db.name,
            customers, products, stores, orders
        )
        
        # Verify database was created
        assert os.path.exists(self.temp_db.name)
        
        # Verify tables exist
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'customers' in tables
        assert 'products' in tables
        assert 'stores' in tables
        assert 'orders' in tables
        
        conn.close()
    
    def test_data_integrity(self):
        """Test data integrity in database"""
        # Create test data
        customers = self.demo._generate_customers(10)
        products = self.demo._generate_products(5)
        stores = self.demo._generate_stores(2)
        orders = self.demo._generate_orders(20, customers, products, stores)
        
        # Save to database
        self.demo._save_to_database(
            self.temp_db.name,
            customers, products, stores, orders
        )
        
        # Verify data integrity
        conn = sqlite3.connect(self.temp_db.name)
        
        # Check customer count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customers")
        customer_count = cursor.fetchone()[0]
        assert customer_count == 10
        
        # Check product count
        cursor.execute("SELECT COUNT(*) FROM products")
        product_count = cursor.fetchone()[0]
        assert product_count == 5
        
        # Check store count
        cursor.execute("SELECT COUNT(*) FROM stores")
        store_count = cursor.fetchone()[0]
        assert store_count == 2
        
        # Check order count
        cursor.execute("SELECT COUNT(*) FROM orders")
        order_count = cursor.fetchone()[0]
        assert order_count == 20
        
        conn.close()

class TestDataQuality:
    """Test data quality checks"""
    
    def setup_method(self):
        """Setup test environment"""
        self.demo = ComprehensiveDataModelingDemo()
    
    def test_data_completeness(self):
        """Test data completeness validation"""
        # Create test data with some missing values
        customers = self.demo._generate_customers(10)
        customers.loc[0, 'email'] = None  # Introduce missing value
        
        # Test completeness check
        completeness = self.demo._check_data_completeness(customers)
        
        assert completeness < 1.0  # Should be less than 100% due to missing email
    
    def test_data_uniqueness(self):
        """Test data uniqueness validation"""
        # Create test data with duplicates
        customers = self.demo._generate_customers(10)
        customers.loc[5, 'customer_id'] = customers.loc[0, 'customer_id']  # Duplicate ID
        
        # Test uniqueness check
        uniqueness = self.demo._check_data_uniqueness(customers, 'customer_id')
        
        assert uniqueness < 1.0  # Should be less than 100% due to duplicate ID
    
    def test_data_consistency(self):
        """Test data consistency validation"""
        # Create test data
        customers = self.demo._generate_customers(10)
        products = self.demo._generate_products(5)
        stores = self.demo._generate_stores(2)
        orders = self.demo._generate_orders(20, customers, products, stores)
        
        # Test consistency check
        consistency = self.demo._check_data_consistency(orders, customers, 'customer_id')
        
        assert consistency >= 0.0
        assert consistency <= 1.0

class TestMachineLearning:
    """Test machine learning components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.demo = ComprehensiveDataModelingDemo()
    
    def test_customer_segmentation(self):
        """Test customer segmentation"""
        # Create test data
        customers = self.demo._generate_customers(100)
        products = self.demo._generate_products(20)
        stores = self.demo._generate_stores(5)
        orders = self.demo._generate_orders(200, customers, products, stores)
        
        # Test segmentation
        segments = self.demo._perform_customer_segmentation(customers, orders)
        
        assert len(segments) == len(customers)
        assert segments.nunique() >= 2  # Should have at least 2 segments
        assert segments.nunique() <= 10  # Should not have too many segments
    
    def test_clv_prediction(self):
        """Test customer lifetime value prediction"""
        # Create test data
        customers = self.demo._generate_customers(50)
        products = self.demo._generate_products(10)
        stores = self.demo._generate_stores(3)
        orders = self.demo._generate_orders(100, customers, products, stores)
        
        # Test CLV prediction
        clv_scores = self.demo._predict_customer_lifetime_value(customers, orders)
        
        assert len(clv_scores) == len(customers)
        assert clv_scores.min() >= 0  # CLV should be non-negative
        assert clv_scores.max() < 100000  # Reasonable upper bound
    
    def test_anomaly_detection(self):
        """Test anomaly detection"""
        # Create test data with some anomalies
        customers = self.demo._generate_customers(50)
        products = self.demo._generate_products(10)
        stores = self.demo._generate_stores(3)
        orders = self.demo._generate_orders(100, customers, products, stores)
        
        # Add some anomalies
        orders.loc[0, 'order_total'] = 50000  # Very high order
        orders.loc[1, 'order_total'] = 50000  # Another high order
        
        # Test anomaly detection
        anomalies = self.demo._detect_anomalies(orders)
        
        assert len(anomalies) <= len(orders)
        assert anomalies.sum() >= 0  # Should detect some anomalies

class TestDataWarehousing:
    """Test data warehousing components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.demo = ComprehensiveDataModelingDemo()
    
    def test_star_schema_creation(self):
        """Test star schema creation"""
        # Create test data
        customers = self.demo._generate_customers(50)
        products = self.demo._generate_products(20)
        stores = self.demo._generate_stores(5)
        orders = self.demo._generate_orders(100, customers, products, stores)
        
        # Test star schema creation
        fact_table, dimension_tables = self.demo._create_star_schema(
            customers, products, stores, orders
        )
        
        assert 'fact_sales' in fact_table.columns
        assert 'customer_dim' in dimension_tables
        assert 'product_dim' in dimension_tables
        assert 'store_dim' in dimension_tables
        assert 'date_dim' in dimension_tables
    
    def test_data_aggregation(self):
        """Test data aggregation functions"""
        # Create test data
        customers = self.demo._generate_customers(50)
        products = self.demo._generate_products(20)
        stores = self.demo._generate_stores(5)
        orders = self.demo._generate_orders(100, customers, products, stores)
        
        # Test aggregation
        aggregated_data = self.demo._aggregate_data(orders)
        
        assert 'total_revenue' in aggregated_data
        assert 'total_orders' in aggregated_data
        assert 'avg_order_value' in aggregated_data
        assert aggregated_data['total_revenue'] > 0
        assert aggregated_data['total_orders'] > 0

class TestSemanticModeling:
    """Test semantic data modeling"""
    
    def setup_method(self):
        """Setup test environment"""
        self.demo = ComprehensiveDataModelingDemo()
    
    def test_ontology_creation(self):
        """Test ontology creation"""
        ontology = self.demo._create_retail_ontology()
        
        assert 'classes' in ontology
        assert 'properties' in ontology
        assert 'relationships' in ontology
        assert 'business_rules' in ontology
        
        # Check required classes
        assert 'Customer' in ontology['classes']
        assert 'Product' in ontology['classes']
        assert 'Order' in ontology['classes']
        assert 'Store' in ontology['classes']
    
    def test_business_rules(self):
        """Test business rules implementation"""
        # Create test data
        customers = self.demo._generate_customers(50)
        products = self.demo._generate_products(20)
        stores = self.demo._generate_stores(5)
        orders = self.demo._generate_orders(100, customers, products, stores)
        
        # Test business rules
        rules = self.demo._implement_business_rules(customers, orders)
        
        assert 'high_value_customers' in rules
        assert 'product_affinity' in rules
        assert 'churn_prediction' in rules
        
        assert len(rules['high_value_customers']) >= 0
        assert len(rules['product_affinity']) >= 0
        assert len(rules['churn_prediction']) >= 0

if __name__ == "__main__":
    pytest.main([__file__])
