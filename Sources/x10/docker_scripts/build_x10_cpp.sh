#!/bin/sh

apt-get -yq install bazel-2.0.0
ln -s /usr/bin/bazel-2.0.0 /usr/bin/bazel

git clone -b v2.2.0-rc0 --recursive https://github.com/tensorflow/tensorflow.git
ln -s /swift-apis/Sources/x10/xla_client /tensorflow/tensorflow/compiler/xla/
ln -s /swift-apis/Sources/x10/xla_tensor /tensorflow/tensorflow/compiler/tf2xla/
ls -l /tensorflow/tensorflow/compiler/tf2xla/xla_tensor/
apt-get -yq install python-dev python-pip
pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1'
pip install -U keras_applications --no-deps
pip install -U keras_preprocessing --no-deps
cd /tensorflow
yes "" | ./configure
export TF_NEED_CUDA=1
export CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
bazel build -c opt --config=cuda --define framework_shared_object=false //tensorflow/compiler/tf2xla/xla_tensor:libx10.so \
 --remote_cache=grpcs://remotebuildexecution.googleapis.com \
 --auth_enabled=true \
 --remote_instance_name=projects/tensorflow-swift/instances/s4tf-remote-bazel-caching \
 --host_platform_remote_properties_override='properties:{name:"cache-silo-key" value:"s4tf-basic-cache-key"}'
