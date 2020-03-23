FROM gcr.io/swift-tensorflow/base-deps-cuda10.1-cudnn7-ubuntu18.04

# Allows the caller to specify the toolchain to use.
ARG swift_tf_url=https://storage.googleapis.com/swift-tensorflow-artifacts/nightlies/latest/swift-tensorflow-DEVELOPMENT-x10-ubuntu18.04.tar.gz

# Download and extract S4TF
WORKDIR /swift-tensorflow-toolchain
RUN curl -fSsL $swift_tf_url -o swift.tar.gz \
    && mkdir usr \
    && tar -xzf swift.tar.gz --directory=usr --strip-components=1 \
    && rm swift.tar.gz

# Copy the kernel into the container
WORKDIR /swift-apis
COPY . .

# Print out swift version for better debugging for toolchain problems
RUN /swift-tensorflow-toolchain/usr/bin/swift --version

# Perform CMake based build
RUN curl -qL https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN echo 'deb https://apt.kitware.com/ubuntu/ bionic main' >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get -yq install --no-install-recommends cmake ninja-build
RUN cmake                                                                       \
      -B /BinaryCache/tensorflow-swift-apis                                     \
      -D BUILD_X10=TRUE                                                         \
      -D CMAKE_BUILD_TYPE=Release                                               \
      -D CMAKE_Swift_COMPILER=/swift-tensorflow-toolchain/usr/bin/swiftc        \
      -D TensorFlow_INCLUDE_DIR=/swift-tensorflow-toolchain/usr/lib/swift/linux/x86_64/modulemaps/CTensorFlow \
      -D TensorFlow_LIBRARY=/swift-tensorflow-toolchain/usr/lib/swift/linux/libtensorflow.so \
      -D USE_BUNDLED_CTENSORFLOW=YES                                            \
      -D X10_INCLUDE_DIR=/swift-tensorflow-toolchain/usr/lib/swift/x10/include  \
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
