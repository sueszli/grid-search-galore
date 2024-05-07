FROM ubuntu:latest

# get system dependencies
RUN apt-get update && apt-get install -y git python3 python3-pip

# mount workspace
COPY . /workspace
WORKDIR /workspace
VOLUME [ "/workspace" ]

# install python dependencies
RUN pip install --no-cache-dir numpy pandas --break-system-packages
RUN pip install --no-cache-dir torch torchvision torchaudio --break-system-packages

# stay alive so we can exec into the container
CMD ["tail", "-f", "/dev/null"]

# alternatively: run jupyter notebook server
# RUN pip install --no-cache-dir jupyter jupyterlab jupyter_contrib_nbextensions --break-system-packages
# ENV JUPYTER_ENABLE_LAB=yes
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''", "--ServerApp.allow_origin='*'", "--ServerApp.disable_check_xsrf=True"]
# EXPOSE 8888
