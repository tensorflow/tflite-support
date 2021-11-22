ARG IMAGE
FROM ${IMAGE}
ARG PYTHON_VERSION

COPY update_sources.sh /
RUN /update_sources.sh

RUN apt-get update && \
    apt-get install -y \
      build-essential \
      software-properties-common \
      zlib1g-dev  \
      curl \
      unzip \
      git && \
    apt-get clean

# Install Python packages.
RUN dpkg --add-architecture armhf
RUN dpkg --add-architecture arm64
RUN yes | add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y \
      python$PYTHON_VERSION \
      python$PYTHON_VERSION-dev \
      python$PYTHON_VERSION-distutils \
      libpython$PYTHON_VERSION-dev \
      libpython$PYTHON_VERSION-dev:armhf \
      libpython$PYTHON_VERSION-dev:arm64
RUN ln -sf /usr/bin/python$PYTHON_VERSION /usr/bin/python3
RUN curl -OL https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN rm get-pip.py
RUN pip3 install --upgrade pip
RUN pip3 install numpy~=1.19.2 setuptools pybind11
RUN ln -sf /usr/include/python$PYTHON_VERSION /usr/include/python3
RUN ln -sf /usr/local/lib/python$PYTHON_VERSION/dist-packages/numpy/core/include/numpy /usr/include/python3/numpy
RUN ln -sf /usr/bin/python3 /usr/bin/python

ENV CI_BUILD_PYTHON=python$PYTHON_VERSION
ENV CROSSTOOL_PYTHON_INCLUDE_PATH=/usr/include/python$PYTHON_VERSION

COPY install_bazel.sh /
RUN /install_bazel.sh

COPY with_the_same_user /
