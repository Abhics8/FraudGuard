#!/bin/bash

# Ask for username if not provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide your Hugging Face username."
    echo "Usage: ./deploy.sh YOUR_USERNAME"
    exit 1
fi

USERNAME=$1
SPACE_URL="https://huggingface.co/spaces/$USERNAME/FraudGuard"

echo "🚀 Preparing to deploy to: $SPACE_URL"

# 1. Create a temporary directory
mkdir -p deploy_temp
cd deploy_temp

# 2. Clone the Space
echo "📥 Cloning repository..."
git clone $SPACE_URL .

# Check if clone was successful
if [ ! -d ".git" ]; then
    echo "❌ Error: Could not clone repository."
    echo "Make sure you have created the Space 'FraudGuard' on Hugging Face first!"
    echo "Create it here: https://huggingface.co/new-space"
    exit 1
fi

# 3. Copy files from project
echo "📋 Copying files..."
cp ../app.py ./app.py
cp ../README_HF.md ./README.md
cp ../requirements_hf.txt ./requirements.txt

# 4. Push to Hugging Face
echo "fw Pushing to Hugging Face..."
git add .
git commit -m "Deploy FraudGuard via script"
git push

echo "✅ Deployment complete!"
echo "View your app at: $SPACE_URL"

# Cleanup
cd ..
rm -rf deploy_temp
