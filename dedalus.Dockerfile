FROM continuumio/miniconda3

WORKDIR /workspace

RUN wget https://raw.githubusercontent.com/DedalusProject/dedalus_conda/master/conda_install_dedalus3.sh
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN source conda_install_dedalus3.sh
SHELL ["conda", "run", "-n", "dedalus3", "/bin/bash", "-c"]

COPY src/ ./src/

RUN pip install -e src/

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

COPY config/ ./config/
COPY generate.py .