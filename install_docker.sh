#!/bin/bash

# Docker Installation Script for Ubuntu WSL
# This script installs Docker Engine and Docker Compose

set -e

echo "=========================================="
echo "Docker Installation for Ubuntu WSL"
echo "=========================================="
echo ""

# Check if Docker is already installed
if command -v docker &> /dev/null; then
    echo "Docker is already installed!"
    docker --version
    exit 0
fi

echo "Step 1: Updating package index..."
sudo apt-get update

echo ""
echo "Step 2: Installing prerequisites..."
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

echo ""
echo "Step 3: Adding Docker's official GPG key..."
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo ""
echo "Step 4: Setting up Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo ""
echo "Step 5: Installing Docker Engine..."
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo ""
echo "Step 6: Starting Docker service..."
sudo service docker start

echo ""
echo "Step 7: Adding user to docker group (to run without sudo)..."
sudo usermod -aG docker $USER

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: You need to log out and log back in for group changes to take effect."
echo ""
echo "After logging back in, verify installation:"
echo "  docker --version"
echo "  docker compose version"
echo "  docker run hello-world"
echo ""
echo "Then you can run the simulation:"
echo "  cd /home/nansi/Work/open-federated-trainer"
echo "  docker compose up --scale client=5"
echo ""

