-- 3rd Normal Form (3NF) Relational Schema
-- Demonstrates normalized data modeling for retail analytics

-- Customer Dimension (3NF)
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    age INTEGER,
    customer_segment VARCHAR(50),
    registration_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Product Dimension (3NF)
CREATE TABLE products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    price DECIMAL(10,2),
    cost DECIMAL(10,2),
    weight_kg DECIMAL(8,3),
    dimensions VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Category Dimension (3NF)
CREATE TABLE categories (
    category_id VARCHAR(50) PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    parent_category_id VARCHAR(50),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_category_id) REFERENCES categories(category_id)
);

-- Store Dimension (3NF)
CREATE TABLE stores (
    store_id VARCHAR(50) PRIMARY KEY,
    store_name VARCHAR(100) NOT NULL,
    location VARCHAR(200),
    city VARCHAR(100),
    state VARCHAR(50),
    country VARCHAR(50),
    store_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Product-Category Relationship (3NF)
CREATE TABLE product_categories (
    product_id VARCHAR(50),
    category_id VARCHAR(50),
    PRIMARY KEY (product_id, category_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

-- Orders Fact Table (3NF)
CREATE TABLE orders (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    store_id VARCHAR(50) NOT NULL,
    order_date TIMESTAMP NOT NULL,
    order_total DECIMAL(12,2),
    order_status VARCHAR(50),
    payment_method VARCHAR(50),
    shipping_address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id)
);

-- Order Items (3NF)
CREATE TABLE order_items (
    order_item_id VARCHAR(50) PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL,
    product_id VARCHAR(50) NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    line_total DECIMAL(12,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Indexes for performance
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
CREATE INDEX idx_orders_store_date ON orders(store_id, order_date);
CREATE INDEX idx_order_items_product ON order_items(product_id);
CREATE INDEX idx_products_category ON product_categories(category_id);
