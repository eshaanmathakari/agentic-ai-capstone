#!/bin/bash

# AWS Deployment Script for AI Portfolio Rebalancer
# This script deploys the application to AWS ECS using Fargate

set -e

# Configuration
AWS_REGION="us-east-1"
ECR_REPOSITORY="portfolio-rebalancer"
ECS_CLUSTER="portfolio-cluster"
ECS_SERVICE="portfolio-service"
TASK_DEFINITION="portfolio-task"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting AWS deployment...${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo -e "${RED}❌ Failed to get AWS account ID. Please check your AWS credentials.${NC}"
    exit 1
fi

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"

echo -e "${YELLOW}📦 Building Docker image...${NC}"
docker build -t ${ECR_REPOSITORY}:latest .

echo -e "${YELLOW}🏷️ Tagging image for ECR...${NC}"
docker tag ${ECR_REPOSITORY}:latest ${ECR_URI}:latest

echo -e "${YELLOW}🔐 Logging in to ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URI}

echo -e "${YELLOW}📤 Pushing image to ECR...${NC}"
docker push ${ECR_URI}:latest

echo -e "${YELLOW}🔄 Updating ECS service...${NC}"
aws ecs update-service \
    --cluster ${ECS_CLUSTER} \
    --service ${ECS_SERVICE} \
    --force-new-deployment \
    --region ${AWS_REGION}

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${GREEN}🌐 Your application should be available at:${NC}"
echo -e "${GREEN}   Backend API: https://your-alb-endpoint.amazonaws.com${NC}"
echo -e "${GREEN}   Streamlit UI: https://your-alb-endpoint.amazonaws.com:8501${NC}"

# Wait for deployment to complete
echo -e "${YELLOW}⏳ Waiting for deployment to complete...${NC}"
aws ecs wait services-stable \
    --cluster ${ECS_CLUSTER} \
    --services ${ECS_SERVICE} \
    --region ${AWS_REGION}

echo -e "${GREEN}🎉 Deployment is now stable and ready!${NC}"
