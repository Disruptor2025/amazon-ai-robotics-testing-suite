# Amazon AI & Robotics Testing Suite - Setup Guide

This guide provides step-by-step instructions for setting up the Amazon AI & Robotics Testing Suite on your local machine or cloud environment.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+ with WSL2
- **Python**: 3.9 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: Minimum 20GB free space
- **Network**: Stable internet connection for AWS services

### Required Software

- **Git**: Version control system
- **Docker**: Containerization platform
- **Docker Compose**: Multi-container orchestration
- **AWS CLI**: AWS command-line interface
- **Python pip**: Python package manager

## Installation Steps

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/your-username/amazon-ai-robotics-testing.git
cd amazon-ai-robotics-testing

# Verify the structure
ls -la
```

### Step 2: Run Automated Setup

The project includes an automated setup script that handles most of the installation:

```bash
# Make the setup script executable
chmod +x src/bash/system_setup/setup_environment.sh

# Run the setup script
./src/bash/system_setup/setup_environment.sh
```

This script will:
- Install system dependencies
- Set up Python virtual environment
- Install Python packages
- Configure Docker
- Set up AWS CLI
- Create configuration files
- Set up monitoring

### Step 3: Manual Setup (Alternative)

If you prefer manual setup or the automated script fails, follow these steps:

#### 3.1 Install System Dependencies

**Ubuntu/Debian:**
```bash
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
```

**macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
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
```

**CentOS/RHEL:**
```bash
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
```

#### 3.2 Setup Docker

```bash
# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in for group changes to take effect
# Or run: newgrp docker

# Test Docker
docker --version
docker run hello-world
```

#### 3.3 Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
```

#### 3.4 Setup AWS CLI

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Configure AWS CLI
aws configure
```

### Step 4: Configure AWS Credentials

#### 4.1 Create AWS Configuration File

```bash
# Copy the template
cp config/aws_config.template config/aws_config.env

# Edit the configuration file
nano config/aws_config.env
```

Fill in your AWS credentials:

```bash
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
```

#### 4.2 Create IAM Role for SageMaker

Create an IAM role with the following policies:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`
- `CloudWatchFullAccess`

```bash
# Create IAM role (requires AWS CLI access)
aws iam create-role \
    --role-name AmazonAITestingRole \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }'

# Attach policies
aws iam attach-role-policy \
    --role-name AmazonAITestingRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
    --role-name AmazonAITestingRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
    --role-name AmazonAITestingRole \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchFullAccess
```

### Step 5: Create S3 Bucket

```bash
# Create S3 bucket for testing data
aws s3 mb s3://your-testing-bucket-name

# Create SageMaker bucket
aws s3 mb s3://your-sagemaker-bucket-name
```

### Step 6: Start Services

```bash
# Start all services using Docker Compose
docker-compose up -d

# Verify services are running
docker-compose ps
```

### Step 7: Verify Installation

#### 7.1 Run Health Checks

```bash
# Run system health checks
./src/bash/monitoring/system_monitor.sh status
```

#### 7.2 Run Test Suite

```bash
# Activate virtual environment
source venv/bin/activate

# Run unit tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_ai_testing.py -v
python -m pytest tests/test_aws_integration.py -v
```

#### 7.3 Test AWS Integration

```bash
# Test AWS connectivity
python -c "
import boto3
s3 = boto3.client('s3')
print('AWS S3 connection successful')
print('Available buckets:', [b['Name'] for b in s3.list_buckets()['Buckets']])
"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Environment variables for sensitive data
# DO NOT commit this file to version control

AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1

DB_PASSWORD=your_db_password_here
API_SECRET_KEY=your_api_secret_here

# Testing configuration
TEST_TIMEOUT=300
MAX_CONCURRENT_TESTS=10
TEST_RETRY_COUNT=3

# Monitoring configuration
ENABLE_MONITORING=true
METRICS_INTERVAL=60
ALERT_EMAIL=admin@example.com
```

### Docker Configuration

The `docker-compose.yml` file is automatically created during setup. It includes:

- **PostgreSQL**: Database for test results
- **Redis**: Caching and session storage
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboard

### Monitoring Setup

Access monitoring dashboards:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Usage Examples

### Basic Testing

```bash
# Run a simple model validation test
python -c "
from src.python.ai_testing import ModelValidator
import asyncio

async def test_model():
    validator = ModelValidator()
    # Add your test code here
    
asyncio.run(test_model())
"
```

### Performance Testing

```bash
# Run performance benchmarks
python -c "
from src.python.ai_testing import PerformanceBenchmark
import asyncio

async def benchmark():
    benchmark = PerformanceBenchmark()
    # Add your benchmark code here
    
asyncio.run(benchmark())
"
```

### Drift Detection

```bash
# Run drift detection
python -c "
from src.python.ai_testing import DriftDetector
import pandas as pd

detector = DriftDetector()
# Add your drift detection code here
"
```

### CI/CD Pipeline

```bash
# Run the complete CI/CD pipeline
./src/shell/ci_cd/pipeline.sh run

# Run specific pipeline stages
./src/shell/ci_cd/pipeline.sh test
./src/shell/ci_cd/pipeline.sh build
./src/shell/ci_cd/pipeline.sh deploy
```

## Troubleshooting

### Common Issues

#### 1. Docker Permission Issues

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Restart Docker service
sudo systemctl restart docker

# Log out and back in
```

#### 2. Python Virtual Environment Issues

```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. AWS Credentials Issues

```bash
# Verify AWS credentials
aws sts get-caller-identity

# Reconfigure AWS CLI
aws configure
```

#### 4. Port Conflicts

```bash
# Check what's using the ports
sudo netstat -tulpn | grep :5432
sudo netstat -tulpn | grep :6379
sudo netstat -tulpn | grep :9090
sudo netstat -tulpn | grep :3000

# Stop conflicting services
sudo systemctl stop postgresql  # If using system PostgreSQL
```

#### 5. Memory Issues

```bash
# Check available memory
free -h

# Increase swap space if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Log Files

Check log files for detailed error information:

```bash
# Application logs
tail -f logs/pipeline.log
tail -f logs/system_monitor.log

# Docker logs
docker-compose logs -f

# Service-specific logs
docker-compose logs postgres
docker-compose logs redis
docker-compose logs prometheus
docker-compose logs grafana
```

### Getting Help

1. **Check the documentation**: Review the docs/ directory
2. **Run health checks**: `./src/bash/monitoring/system_monitor.sh status`
3. **Check system resources**: `htop` or `top`
4. **Verify network connectivity**: `ping google.com`
5. **Check AWS connectivity**: `aws sts get-caller-identity`

## Next Steps

After successful setup:

1. **Configure your AWS resources**: Set up SageMaker endpoints, S3 buckets, etc.
2. **Create test data**: Prepare sample datasets for testing
3. **Set up monitoring**: Configure alerts and dashboards
4. **Run initial tests**: Verify everything works correctly
5. **Customize for your needs**: Modify configurations for your specific use case

## Security Considerations

1. **Never commit credentials**: Keep `.env` and `aws_config.env` out of version control
2. **Use IAM roles**: Prefer IAM roles over access keys when possible
3. **Regular updates**: Keep dependencies updated
4. **Network security**: Use VPC and security groups in production
5. **Audit logging**: Enable CloudTrail for AWS API calls

## Performance Optimization

1. **Resource allocation**: Adjust Docker resource limits
2. **Caching**: Use Redis for frequently accessed data
3. **Database optimization**: Tune PostgreSQL settings
4. **Monitoring**: Set up performance alerts
5. **Scaling**: Use auto-scaling for production workloads

This setup guide provides everything needed to get started with the Amazon AI & Robotics Testing Suite. For additional help, refer to the troubleshooting section or create an issue in the project repository. 