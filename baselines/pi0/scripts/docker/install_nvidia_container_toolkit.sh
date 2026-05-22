#!/bin/bash

# Installs the NVIDIA Container Toolkit, which allows Docker containers to access NVIDIA GPUs.
# NVIDIA's official documentation: [REDACTED_URL]

curl -fsSL [REDACTED_URL] | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg &&
	curl -s -L [REDACTED_URL] |
	sed 's#deb [REDACTED_URL] [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] [REDACTED_URL]' |
		sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# NVIDIA's documentation omits 'sudo' in the following command, but it is required.
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
