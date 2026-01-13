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

# Resolve repo root (directory containing this script).
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR_HOST="${REPO_DIR}/agents/${AGENT}"
DOCKERFILE_PATH="${AGENT_DIR_HOST}/Dockerfile"

if [ ! -f "${DOCKERFILE_PATH}" ]; then
  echo "Error: Dockerfile not found at ${DOCKERFILE_PATH}"
  exit 1
fi

# Some agent Dockerfiles (local forks) expect build context to be the *parent* directory
# containing `mle-bench/` plus sibling repos like `RD-Agent/`.
BUILD_CONTEXT="${AGENT_DIR_HOST}"
if grep -q "COPY mle-bench/" "${DOCKERFILE_PATH}"; then
  BUILD_CONTEXT="$(dirname "${REPO_DIR}")"
fi

# Fail fast if the Dockerfile expects sibling repos that aren't present in the build context.
if grep -qE '^COPY[[:space:]]+RD-Agent[[:space:]]' "${DOCKERFILE_PATH}" && [ ! -d "${BUILD_CONTEXT}/RD-Agent" ]; then
  echo "Error: ${DOCKERFILE_PATH} expects ${BUILD_CONTEXT}/RD-Agent but it does not exist."
  exit 1
fi
if grep -qE '^COPY[[:space:]]+ML-Master_submit[[:space:]]' "${DOCKERFILE_PATH}" && [ ! -d "${BUILD_CONTEXT}/ML-Master_submit" ]; then
  echo "Error: ${DOCKERFILE_PATH} expects ${BUILD_CONTEXT}/ML-Master_submit but it does not exist."
  exit 1
fi
if grep -qE '^COPY[[:space:]]+aideml_submit[[:space:]]' "${DOCKERFILE_PATH}" && [ ! -d "${BUILD_CONTEXT}/aideml_submit" ]; then
  echo "Error: ${DOCKERFILE_PATH} expects ${BUILD_CONTEXT}/aideml_submit but it does not exist."
  exit 1
fi

# --- Build Command ---
# Announce which agent is being built.
echo "Building Docker image for agent: $AGENT"

# Run the Docker build command.
# --platform=linux/amd64 ensures the image is built for that specific architecture.
# -t "$AGENT" tags the image with the provided agent name.
# "agents/$AGENT/" is the build context (the path to the Dockerfile and related files).
# --build-arg passes environment variables into the Docker build process.
docker build --platform=linux/amd64 -t "$AGENT" -f "${DOCKERFILE_PATH}" "${BUILD_CONTEXT}" \
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
