FROM mlebench-env

# where to put submission.csv, will be extracted
ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

RUN mkdir ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent
ARG REQUIREMENTS=${AGENT_DIR}/requirements.txt

# copy just the requirements file, so that we can cache conda separately from the agent files
COPY mle-bench/agents/rdagent_ds/requirements.txt ${AGENT_DIR}/requirements.txt

# Requirements for RD-Agent
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y

# Copy RD-Agent codebase from local directory
COPY RD-Agent ${AGENT_DIR}/RD-Agent

# Install RD-Agent from local directory
RUN conda run -n ${CONDA_ENV_NAME} bash -c "cd ${AGENT_DIR}/RD-Agent && pip install -e . && pip install --upgrade openai httpx anthropic" && \
    conda clean -afy

# Create a 'kaggle' conda environment (RD-Agent expects this name for DS execution)
# Clone the 'agent' environment to create 'kaggle' env
RUN conda create -n kaggle --clone ${CONDA_ENV_NAME} && \
    conda clean -afy

# put all the agent files in the expected location
COPY mle-bench/agents/rdagent_ds ${AGENT_DIR}/wrapper

# MLEBench always runs `/home/agent/start.sh` inside the container.
# Our wrapper lives under `${AGENT_DIR}/wrapper`, so create a stable entrypoint path.
RUN ln -sf ${AGENT_DIR}/wrapper/start.sh ${AGENT_DIR}/start.sh
