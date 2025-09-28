#!/bin/bash

# Production Deployment Script for Enterprise Data Modeling Application
set -e

echo "🚀 Starting Production Deployment..."

# Configuration
NAMESPACE="retail-analytics"
APP_NAME="retail-analytics"
REGISTRY="ghcr.io"
IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        print_warning "Helm is not installed. Some features may not be available."
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    print_success "Prerequisites check completed"
}

# Create namespace
create_namespace() {
    print_status "Creating namespace..."
    
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    print_success "Namespace created/verified"
}

# Deploy secrets
deploy_secrets() {
    print_status "Deploying secrets..."
    
    # Create secret for database password
    kubectl create secret generic retail-analytics-secrets \
        --from-literal=POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-password}" \
        --from-literal=SECRET_KEY="${SECRET_KEY:-your-secret-key-here}" \
        --from-literal=JWT_SECRET_KEY="${JWT_SECRET_KEY:-your-jwt-secret-key-here}" \
        --from-literal=ENCRYPTION_KEY="${ENCRYPTION_KEY:-your-encryption-key-here}" \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    print_success "Secrets deployed"
}

# Deploy configmaps
deploy_configmaps() {
    print_status "Deploying configmaps..."
    
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secret.yaml
    
    print_success "Configmaps deployed"
}

# Deploy database
deploy_database() {
    print_status "Deploying PostgreSQL database..."
    
    # Create PostgreSQL deployment
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: retail_analytics
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: retail-analytics-secrets
              key: POSTGRES_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: $NAMESPACE
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
EOF
    
    print_success "Database deployed"
}

# Deploy Redis
deploy_redis() {
    print_status "Deploying Redis cache..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: $NAMESPACE
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
EOF
    
    print_success "Redis deployed"
}

# Deploy MLflow
deploy_mlflow() {
    print_status "Deploying MLflow tracking server..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: python:3.11-slim
        command: ["bash", "-c"]
        args:
        - |
          pip install mlflow psycopg2-binary &&
          mlflow server 
          --backend-store-uri postgresql://postgres:password@postgres-service:5432/retail_analytics
          --default-artifact-root /mlflow
          --host 0.0.0.0
          --port 5000
        ports:
        - containerPort: 5000
        volumeMounts:
        - name: mlflow-storage
          mountPath: /mlflow
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: mlflow-storage
        persistentVolumeClaim:
          claimName: mlflow-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
  namespace: $NAMESPACE
spec:
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
EOF
    
    print_success "MLflow deployed"
}

# Deploy main application
deploy_application() {
    print_status "Deploying main application..."
    
    # Update image tag in deployment
    sed "s|retail-analytics:latest|$REGISTRY/$APP_NAME:$IMAGE_TAG|g" k8s/app-deployment.yaml | kubectl apply -f -
    
    print_success "Application deployed"
}

# Deploy services
deploy_services() {
    print_status "Deploying services..."
    
    kubectl apply -f k8s/services.yaml
    
    print_success "Services deployed"
}

# Deploy ingress
deploy_ingress() {
    print_status "Deploying ingress..."
    
    kubectl apply -f k8s/ingress.yaml
    
    print_success "Ingress deployed"
}

# Wait for deployment
wait_for_deployment() {
    print_status "Waiting for deployment to be ready..."
    
    # Wait for database
    kubectl wait --for=condition=available --timeout=300s deployment/postgres -n $NAMESPACE
    
    # Wait for Redis
    kubectl wait --for=condition=available --timeout=300s deployment/redis -n $NAMESPACE
    
    # Wait for MLflow
    kubectl wait --for=condition=available --timeout=300s deployment/mlflow -n $NAMESPACE
    
    # Wait for main application
    kubectl wait --for=condition=available --timeout=600s deployment/retail-analytics-app -n $NAMESPACE
    
    print_success "All deployments are ready"
}

# Check deployment status
check_status() {
    print_status "Checking deployment status..."
    
    echo "Deployment Status:"
    kubectl get deployments -n $NAMESPACE
    
    echo ""
    echo "Service Status:"
    kubectl get services -n $NAMESPACE
    
    echo ""
    echo "Pod Status:"
    kubectl get pods -n $NAMESPACE
    
    echo ""
    echo "Ingress Status:"
    kubectl get ingress -n $NAMESPACE
}

# Display access information
display_access_info() {
    print_success "🎉 Production deployment completed successfully!"
    echo ""
    echo "📊 Access Information:"
    echo "  • Namespace: $NAMESPACE"
    echo "  • Application: $APP_NAME"
    echo "  • Registry: $REGISTRY"
    echo ""
    echo "🔧 Management Commands:"
    echo "  • View logs: kubectl logs -f deployment/retail-analytics-app -n $NAMESPACE"
    echo "  • Scale app: kubectl scale deployment retail-analytics-app --replicas=3 -n $NAMESPACE"
    echo "  • Port forward: kubectl port-forward service/retail-analytics-service 8501:80 -n $NAMESPACE"
    echo ""
    echo "📈 Monitoring:"
    echo "  • Check status: kubectl get all -n $NAMESPACE"
    echo "  • View events: kubectl get events -n $NAMESPACE"
    echo "  • Resource usage: kubectl top pods -n $NAMESPACE"
}

# Main deployment function
main() {
    echo "🏢 Enterprise Data Modeling Application - Production Deployment"
    echo "=============================================================="
    echo ""
    
    check_prerequisites
    create_namespace
    deploy_secrets
    deploy_configmaps
    deploy_database
    deploy_redis
    deploy_mlflow
    deploy_application
    deploy_services
    deploy_ingress
    wait_for_deployment
    check_status
    display_access_info
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "status")
        check_status
        ;;
    "logs")
        kubectl logs -f deployment/retail-analytics-app -n $NAMESPACE
        ;;
    "scale")
        kubectl scale deployment retail-analytics-app --replicas=${2:-3} -n $NAMESPACE
        ;;
    "delete")
        print_status "Deleting deployment..."
        kubectl delete namespace $NAMESPACE
        print_success "Deployment deleted"
        ;;
    *)
        echo "Usage: $0 {deploy|status|logs|scale|delete}"
        exit 1
        ;;
esac
