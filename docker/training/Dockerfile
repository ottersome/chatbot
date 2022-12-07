ARG HOME
ARG USER
ARG TZ
#FROM pytorch/manylinux-cuda116
FROM  pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
ARG USER
ARG U="trainer"
ARG P="trainer"
ARG TZ=""
RUN echo uname -a
#RUN yum -y install \
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN mkdir /run/sshd

RUN apt-get update && apt-get install -y --no-install-recommends  \
        python3.8 \
        python3-pip \
        python3.8-dev \
        build-essential \
        # git \
        curl \
        ca-certificates \
        zsh \
        sudo \
        locales \
        openssh-server \
        #openssh-clients \
        vim && \
    # Remove the effect of `apt-get update`
    rm -rf /var/lib/apt/lists/* && \
    # Make the "en_US.UTF-8" locale
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
ENV LANG en_US.utf8
#RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY requirements.txt .
RUN python3.8 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3.8 -m pip install --no-cache-dir -r requirements.txt

#RUN useradd --create-home --shell /bin/zsh --groups wheel -p $P $U 
RUN useradd --create-home --shell /bin/zsh --groups sudo -p $P $U 
RUN echo "${U}:${P}" | chpasswd
RUN ssh-keygen -A

ENV PATH /opt/conda/bin:$PATH

ENV HOME /home/$USER
WORKDIR $HOME
RUN mkdir -p $HOME/.ssh
RUN ls -lah $HOME/.ssh
#COPY id_rsa.pub $HOME/.ssh/authorized_keys
#RUN chown $USER:$USER $HOME/.ssh/authorized_keys && chmod 600 $HOME/.ssh/authorized_keys

CMD ["/usr/bin/sudo", "/usr/sbin/sshd", "-D", "-o", "ListenAddress=0.0.0.0"]
