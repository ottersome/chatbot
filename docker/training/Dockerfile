ARG HOME
ARG USER
ARG TZ
FROM pytorch/manylinux-cuda116
ARG USER
ENV U=$USER
ENV TZ=$TZ
RUN echo uname -a
RUN yum -y install \
        build-essential \
        # git \
        curl \
        ca-certificates \
        zsh \
        sudo \
        locales \
        openssh-server \
        openssh-clients \
        vim && \
    # Remove the effect of `apt-get update`
    rm -rf /var/lib/apt/lists/* && \
    # Make the "en_US.UTF-8" locale
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN useradd --create-home --shell /bin/zsh --groups wheel $USER
RUN ssh-keygen -A

ENV HOME /home/$USER
WORKDIR $HOME
RUN mkdir -p $HOME/.ssh
RUN ls -lah $HOME/.ssh
COPY id_rsa.pub $HOME/.ssh/authorized_keys
RUN chown $USER:$USER $HOME/.ssh/authorized_keys && chmod 600 $HOME/.ssh/authorized_keys

CMD ["/usr/bin/sudo", "/usr/sbin/sshd", "-D", "-o", "ListenAddress=0.0.0.0"]
