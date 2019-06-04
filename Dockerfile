# TODO: We should have a job that creates a S4TF base image so that
#we don't have to duplicate the installation everywhere.
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Allows the caller to specify the toolchain to use.
ARG swift_tf_url=https://storage.googleapis.com/s4tf-kokoro-artifact-testing/latest/swift-tensorflow-DEVELOPMENT-cuda10.0-cudnn7-ubuntu18.04.tar.gz

# Install Swift deps.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        python \
        clang \
        libbsd-dev \
        libcurl4-openssl-dev \
        libicu-dev \
        libncurses5-dev \
        libxml2 \
        libblocksruntime-dev

# Download and extract S4TF
WORKDIR /swift-tensorflow-toolchain
RUN curl -fSsL $swift_tf_url -o swift.tar.gz \
    && mkdir usr \
    && tar -xzf swift.tar.gz --directory=usr --strip-components=1 \
    && rm swift.tar.gz

# Copy the kernel into the container
WORKDIR /swift-apis
COPY . .

# Configure cuda
RUN echo "/usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs" > /etc/ld.so.conf.d/cuda-10.0-stubs.conf && \
    ldconfig

# Print out swift version for better debugging for toolchain problems
RUN /swift-tensorflow-toolchain/usr/bin/swift --version

# Run SwiftPM tests
RUN /swift-tensorflow-toolchain/usr/bin/swift test
