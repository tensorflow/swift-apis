# TODO: We should have a job that creates a S4TF base image so that
#we don't have to duplicate the installation everywhere.
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Allows the caller to specify the toolchain to use.
ARG swift_tf_url=https://storage.googleapis.com/swift-tensorflow-artifacts/nightlies/latest/swift-tensorflow-DEVELOPMENT-cuda10.0-cudnn7-ubuntu18.04.tar.gz

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
        libblocksruntime-dev \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-dev

RUN pip3 install psutil junit-xml

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

# Clean out existing artifacts.
# TODO: move into bash scripts...
RUN rm /swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/TensorFlow.swiftinterface
RUN rm /swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/TensorFlow.swiftdoc
RUN rm /swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/TensorFlow.swiftmodule
RUN rm /swift-tensorflow-toolchain/usr/lib/swift/linux/libswiftTensorFlow.so

# Benchmark compile times
RUN python3 Tools/benchmark_compile.py /swift-tensorflow-toolchain/usr/bin/swift benchmark_results.xml

# Run SwiftPM tests
RUN /swift-tensorflow-toolchain/usr/bin/swift test

# Install into toolchain
# TODO: Unify this with testing. (currently there is a demangling bug).
RUN /swift-tensorflow-toolchain/usr/bin/swift build -Xswiftc -module-link-name -Xswiftc TensorFlow
RUN cp /swift-apis/.build/debug/TensorFlow.swiftmodule /swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/
RUN cp /swift-apis/.build/debug/libTensorFlow.so /swift-tensorflow-toolchain/usr/lib/swift/linux/

WORKDIR /
RUN git clone https://github.com/tensorflow/swift-models.git
RUN git clone https://github.com/fastai/fastai_dev.git
RUN git clone https://github.com/deepmind/open_spiel.git

WORKDIR /swift-models

RUN /swift-tensorflow-toolchain/usr/bin/swift build

WORKDIR /fastai_dev/swift/FastaiNotebook_11_imagenette

RUN /swift-tensorflow-toolchain/usr/bin/swift build

WORKDIR /open_spiel
RUN /swift-tensorflow-toolchain/usr/bin/swift test
