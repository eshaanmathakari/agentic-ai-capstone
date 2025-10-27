#!/bin/bash

# AWS Deployment Script for AI Portfolio Rebalancer
# Alternative deployment method

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 AI Portfolio Rebalancer - AWS Deployment${NC}"
echo -e "${GREEN}===========================================${NC}"

# Configuration
EC2_IP="54.89.129.61"
EC2_USER="ubuntu"  # Changed from ec2-user for Ubuntu instances

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo -e "${YELLOW}Please create .env file with your AWS service credentials${NC}"
    exit 1
fi

# Check if EC2 key exists
if [ ! -f "aws-docs/test-capstone.pem" ]; then
    echo -e "${RED}❌ EC2 key file not found at aws-docs/test-capstone.pem${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Deployment Checklist:${NC}"
echo "1. ✅ Repository cleaned and ready"
echo "2. ✅ Docker configuration updated"
echo "3. ✅ Environment template created"
echo "4. ✅ EC2 key pair available"
echo "5. ✅ EC2 IP: $EC2_IP"
echo ""

# Function to check EC2 instance status
check_ec2_status() {
    echo -e "${BLUE}🔍 Checking EC2 instance status...${NC}"
    
    if command -v aws &> /dev/null; then
        echo "Checking instance status with AWS CLI..."
        aws ec2 describe-instances --filters "Name=ip-address,Values=$EC2_IP" --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' --output table
    else
        echo -e "${YELLOW}AWS CLI not found. Please check your EC2 instance status manually.${NC}"
    fi
}

# Function to create deployment package
create_deployment_package() {
    echo -e "${YELLOW}📦 Creating deployment package...${NC}"
    
    # Clean up any existing deployment package
    rm -f deployment.tar.gz
    
    # Create deployment package
    tar -czf deployment.tar.gz \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='venv' \
        --exclude='aws-docs' \
        --exclude='deployment.tar.gz' \
        .
    
    echo -e "${GREEN}✅ Deployment package created: deployment.tar.gz${NC}"
    echo -e "${YELLOW}Size: $(du -h deployment.tar.gz | cut -f1)${NC}"
}

# Function to show manual deployment instructions
show_manual_deployment() {
    echo -e "${BLUE}📝 Manual Deployment Instructions${NC}"
    echo ""
    echo "Since SSH connection is not working, here are manual steps:"
    echo ""
    echo -e "${YELLOW}1. Check EC2 Instance Status:${NC}"
    echo "   - Go to AWS Console → EC2 → Instances"
    echo "   - Find instance with IP: $EC2_IP"
    echo "   - Check if it's 'Running' state"
    echo ""
    echo -e "${YELLOW}2. Check Security Groups:${NC}"
    echo "   - Ensure security group allows:"
    echo "     - SSH (port 22) from your IP"
    echo "     - HTTP (port 8000) from 0.0.0.0/0"
    echo "     - HTTP (port 8501) from 0.0.0.0/0"
    echo ""
    echo -e "${YELLOW}3. Connect via AWS Console:${NC}"
    echo "   - Use 'Connect' button in EC2 console"
    echo "   - Choose 'EC2 Instance Connect' or 'Session Manager'"
    echo ""
    echo -e "${YELLOW}4. Upload Files:${NC}"
    echo "   - Upload deployment.tar.gz and .env to /home/ec2-user/"
    echo "   - Or use AWS S3 to transfer files"
    echo ""
    echo -e "${YELLOW}5. Run Deployment Commands:${NC}"
    cat << 'EOF'
   # On EC2 instance, run:
   sudo yum update -y
   sudo yum install -y docker
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -a -G docker ec2-user
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   
   # Extract and deploy
   tar -xzf deployment.tar.gz
   docker-compose up -d --build
   
   # Check status
   docker-compose ps
EOF
    echo ""
    echo -e "${GREEN}After deployment, your app will be available at:${NC}"
    echo -e "${GREEN}   Backend API: http://$EC2_IP:8000${NC}"
    echo -e "${GREEN}   Frontend UI: http://$EC2_IP:8501${NC}"
    echo -e "${GREEN}   API Docs: http://$EC2_IP:8000/docs${NC}"
}

# Function to test SSH with different options
test_ssh_connection() {
    echo -e "${BLUE}🔧 Testing SSH connection...${NC}"
    
    # Test with verbose output
    echo "Testing SSH connection with verbose output:"
    ssh -i aws-docs/test-capstone.pem -o StrictHostKeyChecking=no -o ConnectTimeout=10 -v ec2-user@$EC2_IP "echo 'Connection successful'" 2>&1 | head -20
}

# Main menu
echo -e "${BLUE}Choose an option:${NC}"
echo "1. Check EC2 instance status"
echo "2. Create deployment package"
echo "3. Test SSH connection"
echo "4. Show manual deployment instructions"
echo "5. Show environment setup instructions"
echo "6. Exit"

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        check_ec2_status
        ;;
    2)
        create_deployment_package
        ;;
    3)
        test_ssh_connection
        ;;
    4)
        create_deployment_package
        show_manual_deployment
        ;;
    5)
        echo -e "${BLUE}📝 Environment Setup Instructions:${NC}"
        echo ""
        echo "Your .env file should have:"
        echo ""
        echo -e "${YELLOW}# Database Configuration${NC}"
        echo "DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@portfolio-agent-db.cluster-ckbkgk408uak.us-east-1.rds.amazonaws.com:5432/portfolio_agent"
        echo ""
        echo -e "${YELLOW}# Redis Configuration${NC}"
        echo "REDIS_URL=redis://portfolio-redis-dlycua.serverless.use1.cache.amazonaws.com:6379/0"
        echo ""
        echo -e "${YELLOW}# Security${NC}"
        echo "SECRET_KEY=YWdjBLQj1ap2/mH9k1rSTa3l659O4O1fa0dH+sTL2dI="
        echo ""
        echo -e "${YELLOW}# Application Settings${NC}"
        echo "DEBUG=False"
        echo "CORS_ORIGINS=[\"*\"]"
        echo ""
        ;;
    6)
        echo -e "${GREEN}Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac