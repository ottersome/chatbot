ARG HOME
ARG USER
ARG TZ
#FROM pytorch/manylinux-cuda116
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
ARG USER
ARG U="trainer"
ARG P="trainer"
ARG TZ=""
#RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list && mkdir /run/sshd

RUN apt-get update && apt-get install -y --no-install-recommends  \
        build-essential \
        git \
        curl \
        ca-certificates \
        zsh \
        sudo \
        locales \
        openssh-server \
        screen \
        rsync \
        #openssh-clients \
        vim && \
    # Remove the effect of `apt-get update`
    rm -rf /var/lib/apt/lists/* && \
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* 
ENV LANG en_US.utf8
#RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY requirements.txt .
#RUN python3.8 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    #python3.8 -m pip install --no-cache-dir -r requirements.txt && \
#RUN conda init bash && \
    #conda activate base && \
ENV TORCH_CUDA_ARCH_LIST=Volta;Turing
RUN conda run python -m pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    git clone https://github.com/NVIDIA/apex /tmp/apex && \
    conda install transformers pandas tensorboardX tqdm ipywidgets scikit-learn && \
    pip install bitsandbytes accelerate
WORKDIR /tmp/apex
RUN git checkout -b 22.04  remotes/origin/22.04-dev && \
    python -m pip install -v --disable-pip-version-check --no-cache-dir ./

RUN useradd --create-home --shell /bin/zsh --groups sudo -p $P $U  && \
    echo "${U}:${P}" | chpasswd && \
    ssh-keygen -A && \
    mkdir -p $HOME/.ssh && \
    mkdir -p /run/sshd

ENV PATH /opt/conda/bin:$PATH
ENV HOME /home/$USER
WORKDIR $HOME
#COPY id_rsa.pub $HOME/.ssh/authorized_keys
#RUN chown $USER:$USER $HOME/.ssh/authorized_keys && chmod 600 $HOME/.ssh/authorized_keys

CMD ["/usr/bin/sudo", "/usr/sbin/sshd", "-D", "-o", "ListenAddress=0.0.0.0"]
