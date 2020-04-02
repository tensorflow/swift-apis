#!/bin/sh

# Copy over C++ library.
cp /tensorflow/bazel-bin/tensorflow/compiler/tf2xla/xla_tensor/libx10.so /swift-tensorflow-toolchain/usr/lib/swift/linux/
# Prepare C++ headers and copy over C headers.
mkdir /x10_include
python3 /swift-apis/Sources/x10/docker_scripts/collect_headers.py /tensorflow /x10_include
cp /swift-apis/Sources/x10/swift_bindings/device_wrapper.h /swift-tensorflow-toolchain/usr/lib/swift/tensorflow/x10
cp /swift-apis/Sources/x10/swift_bindings/xla_tensor_wrapper.h /swift-tensorflow-toolchain/usr/lib/swift/tensorflow/x10
cp /swift-apis/Sources/x10/swift_bindings/xla_tensor_tf_ops.h /swift-tensorflow-toolchain/usr/lib/swift/tensorflow/x10
