# Installing Docker on Ubuntu WSL

## Quick Installation

### Option 1: Install Docker Desktop for Windows (Recommended for WSL)

1. **Download Docker Desktop:**
   - Go to: https://www.docker.com/products/docker-desktop/
   - Download Docker Desktop for Windows
   - Install and follow the setup wizard

2. **Enable WSL Integration:**
   - Open Docker Desktop
   - Go to Settings → Resources → WSL Integration
   - Enable integration with your WSL distro (Ubuntu)
   - Click "Apply & Restart"

3. **Verify Installation:**
   ```bash
   docker --version
   docker compose version
   ```

### Option 2: Install Docker Engine in WSL (Native)

**Step 1: Update packages**
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
```

**Step 2: Add Docker's official GPG key**
```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

**Step 3: Set up repository**
```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

**Step 4: Install Docker Engine**
```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

**Step 5: Start Docker service**
```bash
sudo service docker start
```

**Step 6: Add your user to docker group (optional, to run without sudo)**
```bash
sudo usermod -aG docker $USER
# Log out and log back in for this to take effect
```

**Step 7: Verify installation**
```bash
docker --version
docker compose version
sudo docker run hello-world
```

## Verify Docker is Working

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker compose version

# Test Docker
docker run hello-world
```

## Troubleshooting

### Docker service not running
```bash
sudo service docker start
sudo service docker status
```

### Permission denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Then log out and log back in
```

### WSL Integration Issues (Docker Desktop)
- Ensure WSL 2 is enabled in Windows
- Restart Docker Desktop
- Check WSL integration settings in Docker Desktop

## After Installation

Once Docker is installed, you can proceed with simulation:

```bash
cd /home/nansi/Work/open-federated-trainer
docker compose up --scale client=5
```

