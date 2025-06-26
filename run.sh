#!/bin/bash

# Create tmp/prompts directory if it doesn't exist
mkdir -p tmp/prompts

# Start the application
uvicorn main:app --reload --port 8000
