#!/bin/bash

# Enterprise Data Modeling Application Deployment Script
set -e

echo "🚀 Starting Enterprise Data Modeling Application Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        cp env.example .env
        print_warning "Please edit .env file with your configuration before continuing."
        read -p "Press Enter to continue after editing .env file..."
    fi
    
    print_success "Prerequisites check completed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data logs reports models
    mkdir -p nginx/ssl
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/datasources
    mkdir -p redis
    mkdir -p sql
    
    print_success "Directories created"
}

# Generate SSL certificates (self-signed for development)
generate_ssl_certificates() {
    print_status "Generating SSL certificates..."
    
    if [ ! -f "nginx/ssl/cert.pem" ]; then
        openssl req -x509 -newkey rsa:4096 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        print_success "SSL certificates generated"
    else
        print_status "SSL certificates already exist"
    fi
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build main application
    docker-compose build --no-cache app
    
    print_success "Docker images built"
}

# Start services
start_services() {
    print_status "Starting services..."
    
    # Start core services first
    docker-compose up -d postgres redis
    
    # Wait for database to be ready
    print_status "Waiting for database to be ready..."
    sleep 30
    
    # Start remaining services
    docker-compose up -d
    
    print_success "Services started"
}

# Check service health
check_health() {
    print_status "Checking service health..."
    
    # Wait for services to start
    sleep 60
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        print_success "Services are running"
    else
        print_error "Some services failed to start"
        docker-compose logs
        exit 1
    fi
}

# Display access information
display_access_info() {
    print_success "🎉 Deployment completed successfully!"
    echo ""
    echo "📊 Access Information:"
    echo "  • Main Dashboard: http://localhost:8501"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • MLflow Tracking: http://localhost:5000"
    echo "  • Grafana Monitoring: http://localhost:3000 (admin/admin)"
    echo "  • Prometheus Metrics: http://localhost:9090"
    echo ""
    echo "🔧 Management Commands:"
    echo "  • View logs: docker-compose logs -f"
    echo "  • Stop services: docker-compose down"
    echo "  • Restart services: docker-compose restart"
    echo "  • Scale app: docker-compose up -d --scale app=3"
    echo ""
    echo "📈 Monitoring:"
    echo "  • Check status: docker-compose ps"
    echo "  • Resource usage: docker stats"
    echo "  • Service health: curl http://localhost:8501/_stcore/health"
}

# Main deployment function
main() {
    echo "🏢 Enterprise Data Modeling Application Deployment"
    echo "=================================================="
    echo ""
    
    check_prerequisites
    create_directories
    generate_ssl_certificates
    build_images
    start_services
    check_health
    display_access_info
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        print_status "Stopping services..."
        docker-compose down
        print_success "Services stopped"
        ;;
    "restart")
        print_status "Restarting services..."
        docker-compose restart
        print_success "Services restarted"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "clean")
        print_status "Cleaning up..."
        docker-compose down -v
        docker system prune -f
        print_success "Cleanup completed"
        ;;
    "production")
        print_status "Deploying to production..."
        docker-compose -f docker-compose.prod.yml up -d
        print_success "Production deployment completed"
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|clean|production}"
        exit 1
        ;;
esac
