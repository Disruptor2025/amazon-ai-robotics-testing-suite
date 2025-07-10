#!/bin/bash

# Amazon AI & Robotics Testing Suite - Environment Setup Script
# This script sets up the complete testing environment for Amazon AI and robotics applications

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
CONFIG_DIR="$PROJECT_ROOT/config"
LOGS_DIR="$PROJECT_ROOT/logs"
DATA_DIR="$PROJECT_ROOT/data"

# Create necessary directories
log "Creating project directories..."
mkdir -p "$CONFIG_DIR" "$LOGS_DIR" "$DATA_DIR" "$PROJECT_ROOT/tests/results"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Python dependencies
install_python_dependencies() {
    log "Installing Python dependencies..."
    
    if ! command_exists python3; then
        error "Python 3 is not installed. Please install Python 3.9+ first."
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log "Python version: $PYTHON_VERSION"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        log "Creating Python virtual environment..."
        python3 -m venv "$PROJECT_ROOT/venv"
    fi
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
        log "Python dependencies installed successfully"
    else
        warn "requirements.txt not found, skipping Python dependency installation"
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    log "Installing system dependencies..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command_exists apt-get; then
            # Ubuntu/Debian
            sudo apt-get update
            sudo apt-get install -y \
                curl \
                wget \
                git \
                docker.io \
                docker-compose \
                jq \
                tree \
                htop \
                unzip \
                build-essential \
                python3-dev \
                python3-pip \
                python3-venv
        elif command_exists yum; then
            # CentOS/RHEL
            sudo yum update -y
            sudo yum install -y \
                curl \
                wget \
                git \
                docker \
                docker-compose \
                jq \
                tree \
                htop \
                unzip \
                gcc \
                python3-devel \
                python3-pip
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command_exists brew; then
            brew update
            brew install \
                curl \
                wget \
                git \
                docker \
                docker-compose \
                jq \
                tree \
                htop \
                python3
        else
            warn "Homebrew not found. Please install Homebrew first: https://brew.sh/"
        fi
    else
        warn "Unsupported OS: $OSTYPE"
    fi
}

# Function to setup Docker
setup_docker() {
    log "Setting up Docker..."
    
    if ! command_exists docker; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Start Docker service
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker "$USER"
        log "Docker service started and enabled"
    fi
    
    # Test Docker
    if docker --version >/dev/null 2>&1; then
        log "Docker is working correctly"
    else
        error "Docker is not working correctly"
    fi
}

# Function to setup AWS CLI
setup_aws_cli() {
    log "Setting up AWS CLI..."
    
    if ! command_exists aws; then
        log "Installing AWS CLI..."
        
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
            unzip awscliv2.zip
            sudo ./aws/install
            rm -rf aws awscliv2.zip
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
            sudo installer -pkg AWSCLIV2.pkg -target /
            rm AWSCLIV2.pkg
        fi
    fi
    
    # Check AWS CLI version
    AWS_VERSION=$(aws --version 2>&1)
    log "AWS CLI version: $AWS_VERSION"
    
    # Create AWS config template if it doesn't exist
    if [ ! -f "$CONFIG_DIR/aws_config.template" ]; then
        cat > "$CONFIG_DIR/aws_config.template" << EOF
# AWS Configuration Template
# Copy this file to aws_config.env and fill in your AWS credentials

AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1
AWS_SESSION_TOKEN=your_session_token_here  # Optional

# SageMaker Configuration
SAGEMAKER_ROLE_ARN=your_sagemaker_role_arn_here
SAGEMAKER_BUCKET=your_sagemaker_bucket_here

# S3 Configuration
S3_BUCKET=your_s3_bucket_here
S3_PREFIX=amazon-ai-testing

# CloudWatch Configuration
CLOUDWATCH_LOG_GROUP=/aws/sagemaker/amazon-ai-testing
CLOUDWATCH_METRIC_NAMESPACE=AmazonAITesting
EOF
        log "AWS config template created at $CONFIG_DIR/aws_config.template"
    fi
}

# Function to setup ROS (Robot Operating System)
setup_ros() {
    log "Setting up ROS (Robot Operating System)..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Add ROS repository
        if ! grep -q "packages.ros.org" /etc/apt/sources.list.d/ros-latest.list 2>/dev/null; then
            sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
            sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
            sudo apt-get update
        fi
        
        # Install ROS Noetic (for Ubuntu 20.04)
        if ! command_exists roscore; then
            sudo apt-get install -y ros-noetic-desktop-full
            echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
            source ~/.bashrc
            log "ROS Noetic installed successfully"
        fi
        
        # Install Gazebo
        if ! command_exists gazebo; then
            sudo apt-get install -y gazebo11
            log "Gazebo installed successfully"
        fi
    else
        warn "ROS setup is primarily for Linux systems. Skipping ROS installation on $OSTYPE"
    fi
}

# Function to create configuration files
create_config_files() {
    log "Creating configuration files..."
    
    # Create main config file
    cat > "$CONFIG_DIR/config.env" << EOF
# Amazon AI & Robotics Testing Suite Configuration

# Project paths
PROJECT_ROOT=$PROJECT_ROOT
CONFIG_DIR=$CONFIG_DIR
LOGS_DIR=$LOGS_DIR
DATA_DIR=$DATA_DIR

# Testing configuration
TEST_TIMEOUT=300
MAX_CONCURRENT_TESTS=10
TEST_RETRY_COUNT=3

# AWS configuration
AWS_REGION=us-east-1
SAGEMAKER_ROLE_ARN=your_role_arn_here
S3_BUCKET=your_bucket_here

# Database configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=amazon_ai_testing
DB_USER=test_user
DB_PASSWORD=test_password

# Monitoring configuration
ENABLE_MONITORING=true
METRICS_INTERVAL=60
ALERT_EMAIL=admin@example.com

# Logging configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_ROTATION=daily
LOG_RETENTION=30
EOF

    # Create Docker Compose file
    cat > "$PROJECT_ROOT/docker-compose.yml" << EOF
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: amazon_ai_testing
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test_user -d amazon_ai_testing"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - monitoring

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
EOF

    # Create Prometheus configuration
    cat > "$CONFIG_DIR/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'amazon-ai-testing'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
EOF

    log "Configuration files created successfully"
}

# Function to setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Create monitoring directories
    mkdir -p "$PROJECT_ROOT/monitoring/prometheus" "$PROJECT_ROOT/monitoring/grafana"
    
    # Set permissions
    chmod 755 "$PROJECT_ROOT/monitoring"
    
    log "Monitoring setup completed"
}

# Function to setup security
setup_security() {
    log "Setting up security..."
    
    # Create .env file for sensitive data
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        cat > "$PROJECT_ROOT/.env" << EOF
# Environment variables for sensitive data
# DO NOT commit this file to version control

AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1

DB_PASSWORD=your_db_password_here
API_SECRET_KEY=your_api_secret_here
EOF
        chmod 600 "$PROJECT_ROOT/.env"
        log "Environment file created at $PROJECT_ROOT/.env"
    fi
    
    # Create .gitignore if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/.gitignore" ]; then
        cat > "$PROJECT_ROOT/.gitignore" << EOF
# Environment files
.env
*.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# Data
data/
*.csv
*.json
*.parquet

# Test results
tests/results/
coverage.xml
.coverage

# AWS
.aws/
aws_config.env

# Docker
.dockerignore

# OS
.DS_Store
Thumbs.db
EOF
        log ".gitignore file created"
    fi
}

# Function to run health checks
run_health_checks() {
    log "Running health checks..."
    
    # Check Python
    if command_exists python3; then
        log "✓ Python 3 is installed"
    else
        error "✗ Python 3 is not installed"
    fi
    
    # Check pip
    if command_exists pip3; then
        log "✓ pip3 is installed"
    else
        error "✗ pip3 is not installed"
    fi
    
    # Check Docker
    if command_exists docker; then
        log "✓ Docker is installed"
        if docker --version >/dev/null 2>&1; then
            log "✓ Docker is working"
        else
            error "✗ Docker is not working"
        fi
    else
        error "✗ Docker is not installed"
    fi
    
    # Check AWS CLI
    if command_exists aws; then
        log "✓ AWS CLI is installed"
    else
        error "✗ AWS CLI is not installed"
    fi
    
    # Check Git
    if command_exists git; then
        log "✓ Git is installed"
    else
        error "✗ Git is not installed"
    fi
    
    log "All health checks passed!"
}

# Main setup function
main() {
    log "Starting Amazon AI & Robotics Testing Suite environment setup..."
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        error "Please do not run this script as root"
    fi
    
    # Install system dependencies
    install_system_dependencies
    
    # Setup Docker
    setup_docker
    
    # Setup AWS CLI
    setup_aws_cli
    
    # Setup ROS (optional)
    setup_ros
    
    # Install Python dependencies
    install_python_dependencies
    
    # Create configuration files
    create_config_files
    
    # Setup monitoring
    setup_monitoring
    
    # Setup security
    setup_security
    
    # Run health checks
    run_health_checks
    
    log "Environment setup completed successfully!"
    log ""
    log "Next steps:"
    log "1. Configure AWS credentials: cp $CONFIG_DIR/aws_config.template $CONFIG_DIR/aws_config.env"
    log "2. Edit $CONFIG_DIR/aws_config.env with your AWS credentials"
    log "3. Start services: docker-compose up -d"
    log "4. Run tests: python -m pytest tests/ -v"
    log ""
    log "For more information, see the documentation in the docs/ directory"
}

# Run main function
main "$@" 