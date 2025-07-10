# Day 1 Quick Start Guide

## 🚀 **Immediate Action Items**

### Step 1: Environment Check (15 minutes)
First, let's check what you already have installed. Run these commands in your terminal:

```bash
python3 --version
docker --version
aws --version
```

**What you need:**
- Python 3.9+ ✅ (if not, install from python.org)
- Docker Desktop ✅ (if not, install from docker.com)
- AWS CLI ✅ (if not, run: `pip install awscli`)

### Step 2: Project Setup (30 minutes)
1. **Navigate to your project directory:**
   ```bash
   cd "/Users/idreeskhan/Desktop/Amazon AI"
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   pip install -r requirements.txt
   ```

### Step 3: AWS Configuration (15 minutes)
1. **Configure AWS CLI:**
   ```bash
   aws configure
   ```
   - Enter your AWS Access Key ID
   - Enter your AWS Secret Access Key
   - Enter your default region (e.g., us-east-1)
   - Enter your output format (json)

2. **Create AWS config file:**
   ```bash
   cp config/aws_config.template config/aws_config.env
   # Edit the file with your credentials
   ```

### Step 4: First Test Run (30 minutes)
1. **Run the setup script:**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Run your first test:**
   ```bash
   python -m pytest tests/ -v
   ```

3. **Check test coverage:**
   ```bash
   python -m pytest tests/ --cov=src --cov-report=html
   ```

## 📚 **What You'll Learn Today**

### Morning Focus: Understanding the Architecture
- **Project Structure**: How all the components fit together
- **Testing Framework**: Understanding pytest and test organization
- **AWS Integration**: How the project connects to AWS services

### Afternoon Focus: Hands-on Practice
- **Writing Tests**: Create your first custom test
- **Running Tests**: Understand test output and debugging
- **Project Navigation**: Find your way around the codebase

## 🎯 **Success Metrics for Day 1**
- [ ] All environment tools installed and working
- [ ] Project dependencies installed successfully
- [ ] AWS credentials configured
- [ ] Setup script runs without errors
- [ ] All existing tests pass
- [ ] You can run and understand test output
- [ ] You understand the basic project structure

## 🔍 **Key Files to Explore Today**

### Core Project Files:
- `README.md` - Project overview and setup instructions
- `requirements.txt` - Python dependencies
- `scripts/setup.sh` - Main setup script

### Source Code Structure:
- `src/python/ai_testing/` - AI model testing framework
- `src/python/aws_integration/` - AWS service integration
- `src/python/robotics_interface/` - Robotics simulation interface

### Test Files:
- `tests/` - All test suites
- `tests/test_ai_models.py` - AI model tests
- `tests/test_aws_integration.py` - AWS integration tests

## 🚨 **Common Issues & Solutions**

### Issue 1: Python version too old
**Solution**: Install Python 3.9+ from python.org

### Issue 2: Docker not running
**Solution**: Start Docker Desktop application

### Issue 3: AWS credentials not found
**Solution**: Run `aws configure` and enter your credentials

### Issue 4: Permission denied on scripts
**Solution**: Run `chmod +x scripts/*.sh`

## 📝 **Evening Assignment**
1. Read through the project documentation in `docs/`
2. Run all existing tests and understand what each one does
3. Try modifying a test and see how it affects the results
4. Prepare 3 questions about the project for tomorrow

## 🎯 **Ready to Start?**
Let's begin! Start with Step 1 above and let me know if you encounter any issues. I'm here to help you through each step of the learning process.

**Next**: Once you've completed the environment setup, we'll dive into understanding the project architecture and writing your first custom test. 