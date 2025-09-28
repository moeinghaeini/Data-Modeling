-- Star Schema for Data Warehouse
-- Demonstrates dimensional modeling for analytics

-- Customer Dimension
CREATE TABLE dim_customer (
    customer_sk INT IDENTITY(1,1) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    customer_name VARCHAR(100),
    email VARCHAR(100),
    age_group VARCHAR(20),
    customer_segment VARCHAR(50),
    registration_year INT,
    is_premium BOOLEAN DEFAULT FALSE,
    effective_date DATE,
    expiry_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Product Dimension
CREATE TABLE dim_product (
    product_sk INT IDENTITY(1,1) PRIMARY KEY,
    product_id VARCHAR(50) NOT NULL,
    product_name VARCHAR(200),
    category_name VARCHAR(100),
    subcategory_name VARCHAR(100),
    price_range VARCHAR(20),
    is_premium BOOLEAN DEFAULT FALSE,
    effective_date DATE,
    expiry_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Store Dimension
CREATE TABLE dim_store (
    store_sk INT IDENTITY(1,1) PRIMARY KEY,
    store_id VARCHAR(50) NOT NULL,
    store_name VARCHAR(100),
    location VARCHAR(200),
    city VARCHAR(100),
    state VARCHAR(50),
    country VARCHAR(50),
    store_type VARCHAR(50),
    effective_date DATE,
    expiry_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Date Dimension
CREATE TABLE dim_date (
    date_sk INT PRIMARY KEY,
    full_date DATE,
    year INT,
    quarter INT,
    month INT,
    month_name VARCHAR(20),
    day_of_year INT,
    day_of_week INT,
    day_name VARCHAR(20),
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    fiscal_year INT,
    fiscal_quarter INT
);

-- Sales Fact Table (Star Schema)
CREATE TABLE fact_sales (
    sales_sk INT IDENTITY(1,1) PRIMARY KEY,
    customer_sk INT,
    product_sk INT,
    store_sk INT,
    date_sk INT,
    order_id VARCHAR(50),
    quantity INT,
    unit_price DECIMAL(10,2),
    line_total DECIMAL(12,2),
    discount_amount DECIMAL(10,2),
    tax_amount DECIMAL(10,2),
    gross_profit DECIMAL(12,2),
    FOREIGN KEY (customer_sk) REFERENCES dim_customer(customer_sk),
    FOREIGN KEY (product_sk) REFERENCES dim_product(product_sk),
    FOREIGN KEY (store_sk) REFERENCES dim_store(store_sk),
    FOREIGN KEY (date_sk) REFERENCES dim_date(date_sk)
);

-- Aggregated Sales Fact Table
CREATE TABLE fact_sales_aggregated (
    customer_sk INT,
    product_sk INT,
    store_sk INT,
    date_sk INT,
    total_quantity INT,
    total_revenue DECIMAL(15,2),
    total_orders INT,
    avg_order_value DECIMAL(10,2),
    FOREIGN KEY (customer_sk) REFERENCES dim_customer(customer_sk),
    FOREIGN KEY (product_sk) REFERENCES dim_product(product_sk),
    FOREIGN KEY (store_sk) REFERENCES dim_store(store_sk),
    FOREIGN KEY (date_sk) REFERENCES dim_date(date_sk)
);

-- Indexes for Star Schema
CREATE INDEX idx_fact_sales_customer ON fact_sales(customer_sk);
CREATE INDEX idx_fact_sales_product ON fact_sales(product_sk);
CREATE INDEX idx_fact_sales_store ON fact_sales(store_sk);
CREATE INDEX idx_fact_sales_date ON fact_sales(date_sk);
CREATE INDEX idx_fact_sales_order ON fact_sales(order_id);
