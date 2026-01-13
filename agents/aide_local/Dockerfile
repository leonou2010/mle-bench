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
COPY mle-bench/agents/aide_local/requirements.txt ${AGENT_DIR}/requirements.txt

# Requirements for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY aideml_submit ${AGENT_DIR}/aideml_submit
# install AIDE from the local fork (editable) and ensure up-to-date client libs
RUN conda run -n ${CONDA_ENV_NAME} bash -c "pip install -e ${AGENT_DIR}/aideml_submit && pip install --upgrade openai httpx" && \
    conda clean -afy

# put all the wrapper files in the expected location
COPY mle-bench/agents/aide_local ${AGENT_DIR}
