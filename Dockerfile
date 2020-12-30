FROM gcr.io/swift-tensorflow/base-deps-cuda10.2-cudnn7-ubuntu18.04

# Allows the caller to specify the toolchain to use.
ARG swift_tf_url=https://storage.googleapis.com/swift-tensorflow-artifacts/nightlies/latest/swift-tensorflow-DEVELOPMENT-notf-ubuntu18.04.tar.gz
ARG bazel_version=3.1.0
ARG TENSORFLOW_USE_STANDARD_TOOLCHAIN=NO

RUN if test -d /swift-apis/google-cloud-sdk; then \
  mv /swift-apis/google-cloud-sdk /opt/google-cloud-sdk; \
  /opt/google-cloud-sdk/bin/gcloud auth list; \
  echo "build --remote_cache=grpcs://remotebuildexecution.googleapis.com \
    --auth_enabled=true \
    --remote_instance_name=projects/tensorflow-swift/instances/s4tf-remote-bazel-caching \
    --host_platform_remote_properties_override='properties:{name:\"cache-silo-key\" value:\"s4tf-basic-cache-key-cuda-10.2\"}'" >> ~/.bazelrc; \
  cat ~/.bazelrc; \
fi

# Download and extract S4TF
WORKDIR /swift-tensorflow-toolchain
RUN if ! curl -fSsL --retry 5 $swift_tf_url -o swift.tar.gz; \
    then sleep 30 && curl -fSsL --retry 5 $swift_tf_url -o swift.tar.gz; \
    fi;

RUN mkdir usr \
    && tar -xzf swift.tar.gz --directory=usr --strip-components=1 \
    && rm swift.tar.gz

# Add bazel and cmake repositories.
RUN curl -qL https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN echo 'deb https://apt.kitware.com/ubuntu/ bionic main' >> /etc/apt/sources.list

RUN curl -qL https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN echo 'deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8' >> /etc/apt/sources.list.d/bazel.list

# Install bazel, cmake, ninja, python, and python dependencies
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NONINTERACTIVE_SEEN=true
RUN apt-get -yq update                                                          \
 && apt-get -yq install --no-install-recommends cmake ninja-build               \
 && apt-get -yq install --no-install-recommends python-dev python-pip python-setuptools \
 && apt-get -yq install --no-install-recommends bazel-$bazel_version            \
 && apt-get clean                                                               \
 && rm -rf /tmp/* /var/tmp/* /var/lib/apt/archive/* /var/lib/apt/lists/*
RUN ln -s /usr/bin/bazel-$bazel_version /usr/bin/bazel
RUN pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1'         \
 && pip install -U --no-deps keras_applications keras_preprocessing

# Print out swift version for better debugging for toolchain problems
RUN /swift-tensorflow-toolchain/usr/bin/swift --version

WORKDIR /swift-apis

# Copy the kernel into the container
COPY . /swift-apis

# Perform CMake based build
ENV TF_NEED_CUDA=1
ENV CTEST_OUTPUT_ON_FAILURE=1
RUN cmake                                                                       \
      -B /BinaryCache/tensorflow-swift-apis                                     \
      -D BUILD_SHARED_LIBS=YES                                                  \
      -D CMAKE_BUILD_TYPE=Release                                               \
      -D CMAKE_INSTALL_PREFIX=/swift-tensorflow-toolchain/usr                   \
      -D CMAKE_Swift_COMPILER=/swift-tensorflow-toolchain/usr/bin/swiftc        \
      -D TENSORFLOW_USE_STANDARD_TOOLCHAIN="$TENSORFLOW_USE_STANDARD_TOOLCHAIN" \
      -G Ninja                                                                  \
      -S /swift-apis
RUN cmake --build /BinaryCache/tensorflow-swift-apis --verbose
RUN cmake --build /BinaryCache/tensorflow-swift-apis --target install
RUN cmake --build /BinaryCache/tensorflow-swift-apis --target test

WORKDIR /
RUN git clone https://github.com/tensorflow/swift-models.git
RUN git clone https://github.com/fastai/fastai_dev.git
RUN git clone https://github.com/deepmind/open_spiel.git

RUN if test x"$TENSORFLOW_USE_STANDARD_TOOLCHAIN" = x"YES" ; then \
      echo "-Xswiftc -DTENSORFLOW_USE_STANDARD_TOOLCHAIN -Xcc -I/swift-tensorflow-toolchain/usr/lib/swift -Xlinker -L/swift-tensorflow-toolchain/usr/lib/swift/linux" > /.swiftflags ; \
    else \
      echo "-Xcc -I/swift-tensorflow-toolchain/usr/lib/swift -Xlinker -L/swift-tensorflow-toolchain/usr/lib/swift/linux" > /.swiftflags ; \
    fi

WORKDIR /swift-models

RUN /swift-tensorflow-toolchain/usr/bin/swift build $(cat /.swiftflags)
# Swift Numerics in release mode does not currently build, which prevents the use of swift-models
# RUN /swift-tensorflow-toolchain/usr/bin/swift build -c release $(cat /.swiftflags)

WORKDIR /fastai_dev/swift/FastaiNotebook_11_imagenette

RUN /swift-tensorflow-toolchain/usr/bin/swift build $(cat /.swiftflags)
RUN /swift-tensorflow-toolchain/usr/bin/swift build -c release $(cat /.swiftflags)

WORKDIR /open_spiel
RUN rm -f Package.resolved
RUN /swift-tensorflow-toolchain/usr/bin/swift test $(cat /.swiftflags)

WORKDIR /swift-apis
# TODO: move into bash scripts...
RUN rm -f /swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/TensorFlow.swiftinterface
RUN rm -f /swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/TensorFlow.swiftdoc
RUN rm -f /swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/TensorFlow.swiftmodule
RUN rm -f /swift-tensorflow-toolchain/usr/lib/swift/linux/libswiftTensorFlow.so

# Benchmark compile times
RUN python3 Utilities/benchmark_compile.py /swift-tensorflow-toolchain/usr/bin/swift benchmark_results.xml

# Run SwiftPM tests
RUN rm -f /swift-tensorflow-toolchain/usr/lib/swift/tensorflow/module.modulemap
RUN /swift-tensorflow-toolchain/usr/bin/swift test $(cat /.swiftflags)
