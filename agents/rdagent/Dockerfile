FROM mlebench-env

# MLE-bench standard environment variables
ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

# Install RD-Agent
WORKDIR /opt
COPY RD-Agent /opt/rdagent

# Install RD-Agent and dependencies in the agent conda environment
ARG CONDA_ENV_NAME=agent
WORKDIR /opt/rdagent
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
RUN conda run -n ${CONDA_ENV_NAME} pip install -e . && \
    conda run -n ${CONDA_ENV_NAME} pip install python-dotenv pandas numpy scikit-learn lightgbm xgboost && \
    conda clean -afy

# Set up working directory
WORKDIR /home/agent

# Copy agent files
COPY mle-bench/agents/rdagent/start.sh /home/agent/

# Make start script executable
RUN chmod +x /home/agent/start.sh

CMD ["/bin/bash"]
