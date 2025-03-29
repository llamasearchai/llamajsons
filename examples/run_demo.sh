#!/bin/bash
# Simple script to run the llama-explorer demo

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}     Llama Explorer Demo Script       ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv ../venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source ../venv/bin/activate

# Install llama-explorer in development mode if not already installed
if ! pip show llama-explorer > /dev/null 2>&1; then
    echo -e "${YELLOW}Installing llama-explorer in development mode...${NC}"
    pip install -e ..
fi

# Create output directory if it doesn't exist
if [ ! -d "./demo_output" ]; then
    echo -e "${YELLOW}Creating demo_output directory...${NC}"
    mkdir -p ./demo_output
fi

# Run the demo script
echo -e "${GREEN}Running the demo script...${NC}"
python simple_demo.py

# Deactivate virtual environment
deactivate

echo -e "${GREEN}Demo completed!${NC}"
echo -e "${YELLOW}Check the ./demo_output directory for results${NC}" 