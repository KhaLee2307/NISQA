# Select Image
FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update
RUN apt-get install -y libicu-dev git wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 mercurial subversion g++ gcc && \
        apt-get clean && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
ENV PATH=/root/miniconda3/bin:$PATH

# init conda and update
RUN echo "Running $(conda --version)" && \
    conda init bash && . /root/.bashrc && \
    conda update conda

# set up conda environment
RUN conda create -n nisqa python=3.8
RUN echo "source activate nisqa" > ~/.bashrc
ENV PATH /root/miniconda3/envs/nisqa/bin:$PATH

# install dependencies
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install -U scikit-learn, seaborn, torchinfo, librosa, pyyaml, pesq, natsort, einops, tqdm

# Install git credential
RUN wget https://github.com/git-ecosystem/git-credential-manager/releases/download/v2.2.2/gcm-linux_amd64.2.2.2.tar.gz
RUN tar -xvf gcm-linux_amd64.2.2.2.tar.gz -C /usr/local/bin
RUN git config --global credential.credentialStore plaintext
RUN git-credential-manager configure

# Get repository
WORKDIR /home