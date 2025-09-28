-- Snowflake Schema for Advanced Analytics
-- Demonstrates hierarchical dimensional modeling

-- Customer Dimension (Level 1)
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

-- Customer Segment Dimension (Level 2 - Snowflake)
CREATE TABLE dim_customer_segment (
    segment_sk INT IDENTITY(1,1) PRIMARY KEY,
    segment_id VARCHAR(50) NOT NULL,
    segment_name VARCHAR(100),
    segment_description TEXT,
    min_order_value DECIMAL(10,2),
    max_order_value DECIMAL(10,2),
    loyalty_tier VARCHAR(50),
    discount_percentage DECIMAL(5,2),
    effective_date DATE,
    expiry_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Customer Geography Dimension (Level 2 - Snowflake)
CREATE TABLE dim_customer_geography (
    geography_sk INT IDENTITY(1,1) PRIMARY KEY,
    country VARCHAR(50),
    state VARCHAR(50),
    city VARCHAR(100),
    postal_code VARCHAR(20),
    region VARCHAR(50),
    timezone VARCHAR(50),
    effective_date DATE,
    expiry_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Product Dimension (Level 1)
CREATE TABLE dim_product (
    product_sk INT IDENTITY(1,1) PRIMARY KEY,
    product_id VARCHAR(50) NOT NULL,
    product_name VARCHAR(200),
    brand VARCHAR(100),
    price_range VARCHAR(20),
    is_premium BOOLEAN DEFAULT FALSE,
    effective_date DATE,
    expiry_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Product Category Dimension (Level 2 - Snowflake)
CREATE TABLE dim_product_category (
    category_sk INT IDENTITY(1,1) PRIMARY KEY,
    category_id VARCHAR(50) NOT NULL,
    category_name VARCHAR(100),
    parent_category_id VARCHAR(50),
    category_level INT,
    category_path VARCHAR(500),
    is_leaf_category BOOLEAN,
    effective_date DATE,
    expiry_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Product Subcategory Dimension (Level 3 - Snowflake)
CREATE TABLE dim_product_subcategory (
    subcategory_sk INT IDENTITY(1,1) PRIMARY KEY,
    subcategory_id VARCHAR(50) NOT NULL,
    subcategory_name VARCHAR(100),
    category_sk INT,
    subcategory_description TEXT,
    target_demographic VARCHAR(100),
    effective_date DATE,
    expiry_date DATE,
    is_current BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (category_sk) REFERENCES dim_product_category(category_sk)
);

-- Store Dimension (Level 1)
CREATE TABLE dim_store (
    store_sk INT IDENTITY(1,1) PRIMARY KEY,
    store_id VARCHAR(50) NOT NULL,
    store_name VARCHAR(100),
    store_type VARCHAR(50),
    effective_date DATE,
    expiry_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Store Geography Dimension (Level 2 - Snowflake)
CREATE TABLE dim_store_geography (
    geography_sk INT IDENTITY(1,1) PRIMARY KEY,
    country VARCHAR(50),
    state VARCHAR(50),
    city VARCHAR(100),
    postal_code VARCHAR(20),
    region VARCHAR(50),
    timezone VARCHAR(50),
    effective_date DATE,
    expiry_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Store Management Dimension (Level 2 - Snowflake)
CREATE TABLE dim_store_management (
    management_sk INT IDENTITY(1,1) PRIMARY KEY,
    manager_id VARCHAR(50),
    manager_name VARCHAR(100),
    department VARCHAR(50),
    management_level VARCHAR(50),
    hire_date DATE,
    effective_date DATE,
    expiry_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Date Dimension (Level 1)
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

-- Time Dimension (Level 2 - Snowflake)
CREATE TABLE dim_time (
    time_sk INT IDENTITY(1,1) PRIMARY KEY,
    hour_24 INT,
    hour_12 INT,
    minute INT,
    second INT,
    am_pm VARCHAR(2),
    time_period VARCHAR(20),
    shift VARCHAR(20),
    effective_date DATE,
    expiry_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Sales Fact Table (Snowflake Schema)
CREATE TABLE fact_sales (
    sales_sk INT IDENTITY(1,1) PRIMARY KEY,
    customer_sk INT,
    product_sk INT,
    store_sk INT,
    date_sk INT,
    time_sk INT,
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
    FOREIGN KEY (date_sk) REFERENCES dim_date(date_sk),
    FOREIGN KEY (time_sk) REFERENCES dim_time(time_sk)
);

-- Customer-Product Relationship Fact
CREATE TABLE fact_customer_product_affinity (
    customer_sk INT,
    product_sk INT,
    affinity_score DECIMAL(5,4),
    purchase_frequency INT,
    last_purchase_date DATE,
    total_spent DECIMAL(12,2),
    FOREIGN KEY (customer_sk) REFERENCES dim_customer(customer_sk),
    FOREIGN KEY (product_sk) REFERENCES dim_product(product_sk)
);

-- Indexes for Snowflake Schema
CREATE INDEX idx_fact_sales_customer ON fact_sales(customer_sk);
CREATE INDEX idx_fact_sales_product ON fact_sales(product_sk);
CREATE INDEX idx_fact_sales_store ON fact_sales(store_sk);
CREATE INDEX idx_fact_sales_date ON fact_sales(date_sk);
CREATE INDEX idx_fact_sales_time ON fact_sales(time_sk);
CREATE INDEX idx_customer_affinity ON fact_customer_product_affinity(customer_sk, product_sk);
