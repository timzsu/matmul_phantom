# Use an official CUDA image as the base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install necessary packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget=1.21.2-2ubuntu1.1 \
    cmake=3.22.1-1ubuntu1.22.04.2 \
    libgmp-dev=2:6.2.1+dfsg-3ubuntu1 \
    libeigen3-dev=3.4.0-2ubuntu2 \
    git=1:2.34.1-1ubuntu1.12 \
    openssh-server=1:8.9p1-3ubuntu0.10 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install NTL library
RUN wget https://libntl.org/ntl-11.5.1.tar.gz \
    && tar xzvf ntl-11.5.1.tar.gz \
    && cd ntl-11.5.1/src \
    && ./configure DEF_PREFIX=/usr/local \
    && make -j8 \
    && make install \
    && cd ../.. \
    && rm -rf ntl-11.5.1

# Install Catch2
RUN git clone https://github.com/catchorg/Catch2.git \
    && cd Catch2 \
    && git checkout v3.8.0 \
    && cmake -S. -Bbuild -DCATCH_INSTALL_DOCS=OFF \
    && cmake --build build --target install --parallel \
    && cd .. \
    && rm -rf Catch2

# Set up the working directory
WORKDIR /workspace

# Copy the project files into the container
COPY . /workspace

RUN mkdir -p /run/sshd; chmod 0755 /run/sshd