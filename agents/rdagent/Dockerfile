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

# Install RD-Agent source
WORKDIR /opt
COPY RD-Agent /opt/rdagent

ARG CONDA_ENV_NAME=agent
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
COPY mle-bench/agents/rdagent/requirements.txt /tmp/rdagent-requirements.txt
RUN conda run -n ${CONDA_ENV_NAME} pip install -e /opt/rdagent && \
    conda run -n ${CONDA_ENV_NAME} pip install -r /tmp/rdagent-requirements.txt && \
    conda clean -afy

# Set up working directory
WORKDIR /home/agent

# Copy agent files
COPY mle-bench/agents/rdagent/ ${AGENT_DIR}/
RUN chmod +x ${AGENT_DIR}/start.sh

WORKDIR ${AGENT_DIR}

CMD ["/bin/bash"]
