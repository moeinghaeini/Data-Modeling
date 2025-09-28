"""
Test suite for the Enterprise Data Modeling API
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

# Import the main application
from production_app import app

client = TestClient(app)

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test health check returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data

class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns application info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Enterprise Data Modeling API"
        assert data["version"] == "2.0.0"
        assert data["status"] == "operational"

class TestMetricsEndpoint:
    """Test metrics endpoint"""
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint returns Prometheus metrics"""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Check if response contains Prometheus metrics format
        assert "http_requests_total" in response.text

class TestDataEndpoints:
    """Test data access endpoints"""
    
    @patch('production_app.get_database')
    def test_get_customers_without_auth(self, mock_db):
        """Test customers endpoint requires authentication"""
        response = client.get("/api/data/customers")
        assert response.status_code == 401
    
    @patch('production_app.get_database')
    def test_get_customers_with_auth(self, mock_db):
        """Test customers endpoint with authentication"""
        # Mock database connection
        mock_engine = Mock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result._mapping = {"customer_id": "CUST_001", "customer_name": "Test Customer"}
        mock_conn.execute.return_value = [mock_result]
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_db.return_value = mock_engine
        
        # Mock authentication
        with patch('production_app.verify_token', return_value="valid-token"):
            response = client.get("/api/data/customers")
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "count" in data
            assert "timestamp" in data

    @patch('production_app.get_database')
    def test_get_products_with_auth(self, mock_db):
        """Test products endpoint with authentication"""
        # Mock database connection
        mock_engine = Mock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result._mapping = {"product_id": "PROD_001", "product_name": "Test Product"}
        mock_conn.execute.return_value = [mock_result]
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_db.return_value = mock_engine
        
        # Mock authentication
        with patch('production_app.verify_token', return_value="valid-token"):
            response = client.get("/api/data/products")
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "count" in data

class TestAnalyticsEndpoints:
    """Test analytics endpoints"""
    
    @patch('production_app.get_database')
    def test_revenue_analytics(self, mock_db):
        """Test revenue analytics endpoint"""
        # Mock database connection
        mock_engine = Mock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result._mapping = {
            "date": "2023-01-01",
            "daily_revenue": 1000.0,
            "daily_orders": 10,
            "daily_customers": 5
        }
        mock_conn.execute.return_value = [mock_result]
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_db.return_value = mock_engine
        
        # Mock authentication
        with patch('production_app.verify_token', return_value="valid-token"):
            response = client.get("/api/analytics/revenue")
            assert response.status_code == 200
            data = response.json()
            assert "analytics" in data
            assert "summary" in data

    @patch('production_app.get_database')
    def test_customer_segments(self, mock_db):
        """Test customer segments endpoint"""
        # Mock database connection
        mock_engine = Mock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result._mapping = {
            "customer_segment": "Gold",
            "customer_count": 100,
            "avg_age": 35.5,
            "total_revenue": 50000.0
        }
        mock_conn.execute.return_value = [mock_result]
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_db.return_value = mock_engine
        
        # Mock authentication
        with patch('production_app.verify_token', return_value="valid-token"):
            response = client.get("/api/analytics/customer-segments")
            assert response.status_code == 200
            data = response.json()
            assert "segments" in data

    @patch('production_app.get_database')
    def test_anomalies(self, mock_db):
        """Test anomalies endpoint"""
        # Mock database connection
        mock_engine = Mock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result._mapping = {
            "order_id": "ORDER_001",
            "customer_id": "CUST_001",
            "order_total": 10000.0,
            "is_anomaly": True
        }
        mock_conn.execute.return_value = [mock_result]
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_db.return_value = mock_engine
        
        # Mock authentication
        with patch('production_app.verify_token', return_value="valid-token"):
            response = client.get("/api/analytics/anomalies")
            assert response.status_code == 200
            data = response.json()
            assert "anomalies" in data
            assert "count" in data

class TestMLEndpoints:
    """Test ML prediction endpoints"""
    
    def test_predict_clv_without_auth(self):
        """Test CLV prediction requires authentication"""
        response = client.post("/api/ml/predict/clv", json={
            "customer_id": "CUST_001",
            "features": {}
        })
        assert response.status_code == 401
    
    def test_predict_clv_with_auth(self):
        """Test CLV prediction with authentication"""
        with patch('production_app.verify_token', return_value="valid-token"):
            response = client.post("/api/ml/predict/clv", json={
                "customer_id": "CUST_001",
                "features": {}
            })
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "model_version" in data
            assert "timestamp" in data

class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_endpoint(self):
        """Test invalid endpoint returns 404"""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test method not allowed returns 405"""
        response = client.post("/health")
        assert response.status_code == 405

class TestDataValidation:
    """Test data validation"""
    
    def test_invalid_data_request(self):
        """Test invalid data request parameters"""
        with patch('production_app.verify_token', return_value="valid-token"):
            response = client.get("/api/data/customers?limit=50000")  # Exceeds max limit
            # Should still work but limit should be capped
            assert response.status_code == 200

class TestPerformance:
    """Test performance characteristics"""
    
    def test_response_time(self):
        """Test response time is reasonable"""
        import time
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second

class TestConcurrency:
    """Test concurrent requests"""
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get("/health")
            results.append(response.status_code)
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 10

if __name__ == "__main__":
    pytest.main([__file__])
