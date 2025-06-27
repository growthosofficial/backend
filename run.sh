#!/bin/bash

rm -rf tmp/prompts
mkdir -p tmp/prompts

# Start the application
uvicorn main:app --reload --port 8000
