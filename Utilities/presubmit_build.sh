#!/bin/bash

set -exuo pipefail

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
