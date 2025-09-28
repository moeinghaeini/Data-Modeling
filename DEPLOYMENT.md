# 🚀 Enterprise Data Modeling Application - Deployment Guide

This guide provides comprehensive instructions for deploying the Enterprise Data Modeling Application in various environments.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Production Deployment](#production-deployment)
6. [Monitoring & Observability](#monitoring--observability)
7. [Troubleshooting](#troubleshooting)
8. [Security Considerations](#security-considerations)

## 🔧 Prerequisites

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended (16GB+ for production)
- **Storage**: 50GB+ available space
- **OS**: Linux, macOS, or Windows with WSL2

### Required Software
- **Docker**: 20.10+ with Docker Compose 2.0+
- **Python**: 3.9+ (for local development)
- **Git**: Latest version
- **kubectl**: Latest version (for Kubernetes deployment)
- **Helm**: 3.0+ (optional, for advanced Kubernetes features)

## 🏠 Local Development

### 1. Clone Repository
```bash
git clone <repository-url>
cd data-modeling
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp env.example .env
# Edit .env with your configuration
```

### 5. Run Application
```bash
# Run comprehensive demo
python comprehensive_demo.py

# Run production API
python production_app.py

# Run Streamlit dashboard
streamlit run enterprise_dashboard.py
```

## 🐳 Docker Deployment

### Quick Start
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment
```bash
# Deploy production stack
docker-compose -f docker-compose.prod.yml up -d

# Scale application
docker-compose up -d --scale app=3

# Monitor services
docker-compose ps
```

### Custom Configuration
```bash
# Override environment variables
POSTGRES_PASSWORD=your-secure-password docker-compose up -d

# Use custom configuration
docker-compose -f docker-compose.override.yml up -d
```

## ☸️ Kubernetes Deployment

### 1. Prerequisites
```bash
# Ensure kubectl is configured
kubectl cluster-info

# Create namespace
kubectl create namespace retail-analytics
```

### 2. Deploy Application
```bash
# Deploy all components
kubectl apply -f k8s/

# Check deployment status
kubectl get all -n retail-analytics
```

### 3. Access Application
```bash
# Port forward to access locally
kubectl port-forward service/retail-analytics-service 8501:80 -n retail-analytics

# Get external IP (if LoadBalancer)
kubectl get ingress -n retail-analytics
```

### 4. Scale Application
```bash
# Scale application replicas
kubectl scale deployment retail-analytics-app --replicas=5 -n retail-analytics

# Auto-scaling (requires metrics-server)
kubectl autoscale deployment retail-analytics-app --min=2 --max=10 -n retail-analytics
```

## 🏭 Production Deployment

### 1. Environment Setup
```bash
# Set production environment variables
export ENVIRONMENT=production
export POSTGRES_PASSWORD=your-secure-password
export SECRET_KEY=your-secret-key-here
export JWT_SECRET_KEY=your-jwt-secret-key-here
```

### 2. Deploy with Production Script
```bash
# Run production deployment script
./scripts/production-deploy.sh deploy

# Check deployment status
./scripts/production-deploy.sh status

# View application logs
./scripts/production-deploy.sh logs
```

### 3. Configure Load Balancer
```yaml
# nginx-ingress configuration
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: retail-analytics-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - retail-analytics.yourdomain.com
    secretName: retail-analytics-tls
  rules:
  - host: retail-analytics.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: retail-analytics-service
            port:
              number: 80
```

## 📊 Monitoring & Observability

### Prometheus Metrics
```bash
# Access Prometheus
http://localhost:9090

# View metrics
curl http://localhost:9090/metrics
```

### Grafana Dashboards
```bash
# Access Grafana
http://localhost:3000
# Username: admin
# Password: admin

# Import dashboards
# - Retail Analytics Dashboard
# - System Metrics Dashboard
# - Application Performance Dashboard
```

### Log Aggregation
```bash
# View application logs
docker-compose logs -f app

# View database logs
docker-compose logs -f postgres

# View all logs
docker-compose logs -f
```

### Health Checks
```bash
# Application health
curl http://localhost:8501/_stcore/health

# API health
curl http://localhost:8000/health

# Database health
docker-compose exec postgres pg_isready -U postgres
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Database Connection Issues
```bash
# Check database status
docker-compose ps postgres

# View database logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U postgres -d retail_analytics -c "SELECT 1;"
```

#### 2. Application Not Starting
```bash
# Check application logs
docker-compose logs app

# Check resource usage
docker stats

# Restart application
docker-compose restart app
```

#### 3. Memory Issues
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Check memory usage
docker stats

# Restart with more memory
docker-compose down
docker-compose up -d
```

#### 4. Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8501
netstat -tulpn | grep :8000

# Use different ports
docker-compose -f docker-compose.override.yml up -d
```

### Performance Optimization

#### 1. Database Optimization
```sql
-- Create indexes for better performance
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_products_category ON products(category);
```

#### 2. Application Optimization
```bash
# Scale application
docker-compose up -d --scale app=3

# Use production configuration
docker-compose -f docker-compose.prod.yml up -d
```

#### 3. Caching
```bash
# Redis cache configuration
# Increase Redis memory
docker-compose exec redis redis-cli CONFIG SET maxmemory 1gb
```

## 🔒 Security Considerations

### 1. Environment Variables
```bash
# Use strong passwords
export POSTGRES_PASSWORD=$(openssl rand -base64 32)
export SECRET_KEY=$(openssl rand -base64 32)
export JWT_SECRET_KEY=$(openssl rand -base64 32)
```

### 2. Network Security
```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: retail-analytics-netpol
spec:
  podSelector:
    matchLabels:
      app: retail-analytics
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: retail-analytics
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: retail-analytics
```

### 3. SSL/TLS Configuration
```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem -days 365 -nodes

# Configure HTTPS
# Update nginx configuration for SSL
```

### 4. Access Control
```bash
# Implement authentication
# Use JWT tokens for API access
# Configure role-based access control
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale application replicas
docker-compose up -d --scale app=5

# Kubernetes auto-scaling
kubectl autoscale deployment retail-analytics-app --min=2 --max=10 -n retail-analytics
```

### Vertical Scaling
```yaml
# Increase resource limits
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### Database Scaling
```bash
# Use read replicas
# Implement connection pooling
# Optimize queries
```

## 🔄 Backup & Recovery

### Database Backup
```bash
# Create backup
docker-compose exec postgres pg_dump -U postgres retail_analytics > backup.sql

# Restore backup
docker-compose exec -T postgres psql -U postgres retail_analytics < backup.sql
```

### Application Backup
```bash
# Backup application data
tar -czf app-backup.tar.gz data/ logs/ reports/ models/

# Restore application data
tar -xzf app-backup.tar.gz
```

## 📞 Support

### Getting Help
- **Documentation**: Check this guide and README.md
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Logs**: Check application logs for errors

### Monitoring
- **Health Checks**: Regular health check endpoints
- **Metrics**: Prometheus metrics and Grafana dashboards
- **Logs**: Centralized logging with ELK stack
- **Alerts**: Configure alerting for critical issues

---

**For additional support, please refer to the main README.md file or create an issue in the repository.**
