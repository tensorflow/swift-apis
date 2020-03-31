#!/bin/bash

set -exuo pipefail

gcloud auth list

UBUNTU_VERSION=$(lsb_release -a | grep Release | awk '{print $2}')
IMAGE_VERSION=$(cat /VERSION)
CACHE_SILO_VAL="gpu-ubuntu-16-${UBUNTU_VERSION}-${IMAGE_VERSION}-s4tf"

use_bazel.sh 2.0.0 --quiet
bazel version

git clone -b v2.2.0-rc0 --recursive https://github.com/tensorflow/tensorflow.git

cp -R github/swift-apis/Sources/x10/xla_tensor tensorflow/tensorflow/compiler/tf2xla/
cp -R github/swift-apis/Sources/x10/xla_client tensorflow/tensorflow/compiler/xla/

# find tensorflow
pushd tensorflow
sudo -H apt-get -yq install python-dev python-pip
sudo -H pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1'
sudo -H pip install -U keras_applications --no-deps
sudo -H pip install -U keras_preprocessing --no-deps
bazel --batch --bazelrc=/dev/null version
yes "" | ./configure || true
bazel build -c opt --define framework_shared_object=false //tensorflow/compiler/tf2xla/xla_tensor:libx10.so \
 --remote_cache=grpcs://remotebuildexecution.googleapis.com \
 --auth_enabled=true \
 --remote_instance_name=projects/tensorflow-swift/instances/s4tf-remote-bazel-caching \
 --host_platform_remote_properties_override='properties:{name:"cache-silo-key" value:"${CACHE_SILO_VAL}"}'
popd

sudo apt-get install -y docker.io

# Sets 'swift_tf_url' to the public url corresponding to
# 'swift_tf_bigstore_gfile', if it exists.
if [[ ! -z ${swift_tf_bigstore_gfile+x} ]]; then
  export swift_tf_url="${swift_tf_bigstore_gfile/\/bigstore/https://storage.googleapis.com}"
fi

cd github/swift-apis
sudo -E docker build -t build-img -f Dockerfile --build-arg swift_tf_url .

sudo docker create --name build-container build-img
mkdir -p "$KOKORO_ARTIFACTS_DIR/swift_apis_benchmarks"
sudo docker cp build-container:/swift-apis/benchmark_results.xml "$KOKORO_ARTIFACTS_DIR/swift_apis_benchmarks/sponge_log.xml"
