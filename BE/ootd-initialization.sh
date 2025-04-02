#!/bin/bash

# Clone the main repo
git clone https://github.com/levihsu/OOTDiffusion
cd OOTDiffusion

# Install Homebrew (Linux)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

echo 'eval "$($(brew --prefix)/bin/brew shellenv)"' >> ~/.bashrc

eval "$($(brew --prefix)/bin/brew shellenv)"

# Install git-lfs and set it up
brew install git-lfs
git lfs install

# Clone model weights using git-lfs
cd checkpoints

git clone https://huggingface.co/levihsu/OOTDiffusion

git clone https://huggingface.co/openai/clip-vit-large-patch14

cd ..

# Set up Python virtual environment and install requirements
python3 -m venv ootd
source ootd/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Navigate to the run directory
cd run
