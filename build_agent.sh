#!/bin/bash

# This script builds a Docker image for a specified agent.
# It takes one argument: the name of the agent to build.

# --- Configuration ---
# Set the directories for the build environment.
export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent

# --- Argument Check ---
# Check if an agent name was provided as an argument.
# If not, print a usage message and exit with an error.
if [ -z "$1" ]; then
  echo "Usage: $0 <agent_name>"
  exit 1
fi

# Assign the first script argument to the AGENT variable.
AGENT=$1

# --- Build Command ---
# Announce which agent is being built.
echo "Building Docker image for agent: $AGENT"

# Run the Docker build command.
# --platform=linux/amd64 ensures the image is built for that specific architecture.
# -t "$AGENT" tags the image with the provided agent name.
# "agents/$AGENT/" is the build context (the path to the Dockerfile and related files).
# --build-arg passes environment variables into the Docker build process.
docker build --platform=linux/amd64 -t "$AGENT" "agents/$AGENT/" \
  --build-arg SUBMISSION_DIR="$SUBMISSION_DIR" \
  --build-arg LOGS_DIR="$LOGS_DIR" \
  --build-arg CODE_DIR="$CODE_DIR" \
  --build-arg AGENT_DIR="$AGENT_DIR"

# --- Completion Message ---
# Check the exit code of the docker build command to see if it was successful.
if [ $? -eq 0 ]; then
  echo "Successfully built image for agent: $AGENT"
else
  echo "Error building image for agent: $AGENT"
fi

