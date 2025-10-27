#!/bin/bash

# Simple AWS Deployment Script for AI Portfolio Rebalancer

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
EC2_KEY_PATH="aws-docs/test-capstone.pem"
EC2_USER="ubuntu"
EC2_IP="54.89.129.61"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo -e "${YELLOW}Please create .env file with your AWS service credentials${NC}"
    exit 1
fi

# Check if EC2 key exists
if [ ! -f "$EC2_KEY_PATH" ]; then
    echo -e "${RED}❌ EC2 key file not found at $EC2_KEY_PATH${NC}"
    exit 1
fi

# Set proper permissions for EC2 key
chmod 400 "$EC2_KEY_PATH"

echo -e "${BLUE}🔧 Deploying to EC2 instance: $EC2_IP${NC}"

# Create deployment package
echo -e "${YELLOW}📦 Creating deployment package...${NC}"
tar -czf deployment.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='venv' \
    --exclude='aws-docs' \
    --exclude='deployment.tar.gz' \
    .

# Copy files to EC2
echo -e "${YELLOW}📤 Copying files to EC2...${NC}"
scp -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=no \
    deployment.tar.gz \
    .env \
    "$EC2_USER@$EC2_IP:/home/$EC2_USER/"

# Execute deployment commands on EC2
echo -e "${YELLOW}🚀 Executing deployment on EC2...${NC}"
ssh -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_IP" << 'EOF'
    # Update system
    sudo apt update -y
    
    # Install Docker
    sudo apt install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -a -G docker ubuntu
    
    # Install Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Extract deployment files
    tar -xzf deployment.tar.gz
    rm deployment.tar.gz
    
    # Build and start services
    docker-compose up -d --build
    
    # Show status
    docker-compose ps
    
    echo "🎉 Deployment completed!"
    echo "Backend API: http://54.89.129.61:8000"
    echo "Frontend UI: http://54.89.129.61:8501"
EOF

# Clean up local files
rm deployment.tar.gz

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${GREEN}🌐 Your application should be available at:${NC}"
echo -e "${GREEN}   Backend API: http://$EC2_IP:8000${NC}"
echo -e "${GREEN}   Frontend UI: http://$EC2_IP:8501${NC}"
echo -e "${GREEN}   API Docs: http://$EC2_IP:8000/docs${NC}"
