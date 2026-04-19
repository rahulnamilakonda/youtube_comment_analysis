#!/bin/bash
# SentiTube v3 - Application Start Script
exec > /home/ubuntu/deploy.log 2>&1

# Configuration
DOCKER_IMAGE="rahulnamilakonda/youtube_comment_analysis:latest"
CONTAINER_NAME="youtube-comment-analysis"

echo "Step 1: Pulling latest image from Docker Hub..."
docker pull $DOCKER_IMAGE

echo "Step 2: Checking for existing container..."
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping existing container..."
    docker stop $CONTAINER_NAME
fi

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Removing existing container..."
    docker rm $CONTAINER_NAME
fi

echo "Step 3: Starting new container..."
# We use the .env file that was injected into the deployment zip by CI/CD
docker run -d \
    -p 80:8000 \
    --name $CONTAINER_NAME \
    --restart always \
    --env-file /home/ubuntu/app/.env \
    $DOCKER_IMAGE

echo "Step 4: Cleanup..."
# Remove dangling images to save space on EC2
docker image prune -f

echo "Deployment finished successfully."
