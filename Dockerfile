FROM gcr.io/swift-tensorflow/base-deps-cuda10.1-cudnn7-ubuntu18.04

# Allows the caller to specify the toolchain to use.
ARG swift_tf_url=https://storage.googleapis.com/swift-tensorflow-artifacts/nightlies/latest/swift-tensorflow-DEVELOPMENT-x10-cuda10.1-cudnn7-ubuntu18.04.tar.gz

# Add bazel and cmake repositories.
RUN curl -qL https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN echo 'deb https://apt.kitware.com/ubuntu/ bionic main' >> /etc/apt/sources.list
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get update

# Copy the kernel into the container
WORKDIR /swift-apis
COPY . .

RUN if test -d google-cloud-sdk; then \
  mv google-cloud-sdk /opt/google-cloud-sdk; \
  /opt/google-cloud-sdk/bin/gcloud auth list; \
  echo "build --remote_cache=grpcs://remotebuildexecution.googleapis.com \
    --auth_enabled=true \
    --remote_instance_name=projects/tensorflow-swift/instances/s4tf-remote-bazel-caching \
    --host_platform_remote_properties_override='properties:{name:\"cache-silo-key\" value:\"s4tf-basic-cache-key\"}'" >> ~/.bazelrc; \
  cat ~/.bazelrc; \
fi

# Build C++ x10 parts.
WORKDIR /
RUN /swift-apis/Sources/x10/docker_scripts/build_x10_cpp.sh

# Download and extract S4TF
WORKDIR /swift-tensorflow-toolchain
RUN curl -fSsL $swift_tf_url -o swift.tar.gz \
    && mkdir usr \
    && tar -xzf swift.tar.gz --directory=usr --strip-components=1 \
    && rm swift.tar.gz

# Copy over x10 parts.
RUN /swift-apis/Sources/x10/docker_scripts/copy_x10_cpp.sh

WORKDIR /swift-apis

# Print out swift version for better debugging for toolchain problems
RUN /swift-tensorflow-toolchain/usr/bin/swift --version

# Perform CMake based build
RUN apt-get -yq install --no-install-recommends cmake ninja-build
RUN cmake                                                                       \
      -B /BinaryCache/tensorflow-swift-apis                                     \
      -D CMAKE_BUILD_TYPE=Release                                               \
      -D CMAKE_Swift_COMPILER=/swift-tensorflow-toolchain/usr/bin/swiftc        \
      -D TensorFlow_INCLUDE_DIR=/swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/modulemaps/CTensorFlow \
      -D TensorFlow_LIBRARY=/swift-tensorflow-toolchain/usr/lib/swift/linux/libtensorflow.so \
      -D USE_BUNDLED_CTENSORFLOW=YES                                            \
      -D BUILD_X10=YES                                                          \
      -D X10_INCLUDE_DIR='/x10_include'                                         \
      -G Ninja                                                                  \
      -S /swift-apis
RUN cmake --build /BinaryCache/tensorflow-swift-apis --verbose
RUN cmake --build /BinaryCache/tensorflow-swift-apis --target test

# Clean out existing artifacts.
# TODO: move into bash scripts...
RUN rm -f /swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/TensorFlow.swiftinterface
RUN rm -f /swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/TensorFlow.swiftdoc
RUN rm -f /swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/TensorFlow.swiftmodule
RUN rm -f /swift-tensorflow-toolchain/usr/lib/swift/linux/libswiftTensorFlow.so

# Benchmark compile times
RUN python3 Utilities/benchmark_compile.py /swift-tensorflow-toolchain/usr/bin/swift benchmark_results.xml

# Run SwiftPM tests
RUN /swift-tensorflow-toolchain/usr/bin/swift test

# Install into toolchain
# TODO: Unify this with testing. (currently there is a demangling bug).
RUN /swift-tensorflow-toolchain/usr/bin/swift build -Xswiftc -module-link-name -Xswiftc TensorFlow
RUN cp /swift-apis/.build/debug/TensorFlow.swiftmodule /swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/
RUN cp /swift-apis/.build/debug/Tensor.swiftmodule /swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/
RUN cp /swift-apis/.build/debug/libTensorFlow.so /swift-tensorflow-toolchain/usr/lib/swift/linux/
RUN cp /swift-apis/.build/debug/libTensor.so /swift-tensorflow-toolchain/usr/lib/swift/linux/
RUN /swift-apis/Sources/x10/docker_scripts/copy_x10_swift.sh

# Run x10 tests
RUN XRT_WORKERS='localservice:0;grpc://localhost:40935' /BinaryCache/tensorflow-swift-apis/Sources/x10/ops_test

WORKDIR /
RUN git clone https://github.com/tensorflow/swift-models.git
RUN git clone https://github.com/fastai/fastai_dev.git
RUN git clone https://github.com/deepmind/open_spiel.git

WORKDIR /swift-models

RUN /swift-tensorflow-toolchain/usr/bin/swift build
RUN /swift-tensorflow-toolchain/usr/bin/swift build -c release

WORKDIR /fastai_dev/swift/FastaiNotebook_11_imagenette

RUN /swift-tensorflow-toolchain/usr/bin/swift build
RUN /swift-tensorflow-toolchain/usr/bin/swift build -c release

WORKDIR /open_spiel
RUN /swift-tensorflow-toolchain/usr/bin/swift test
