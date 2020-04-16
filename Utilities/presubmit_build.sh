#!/bin/bash

set -exuo pipefail

sudo apt-get install -y docker.io
gcloud auth list
gcloud beta auth configure-docker

# Sets 'swift_tf_url' to the public url corresponding to
# 'swift_tf_bigstore_gfile', if it exists.
if [[ ! -z ${swift_tf_bigstore_gfile+x} ]]; then
  export swift_tf_url="${swift_tf_bigstore_gfile/\/bigstore/https://storage.googleapis.com}"
fi

# Help debug the job's disk space.
df -h

# Move docker images into /tmpfs, where there is more space.
sudo /etc/init.d/docker stop
sudo mv /var/lib/docker /tmpfs/
sudo ln -s /tmpfs/docker /var/lib/docker
sudo /etc/init.d/docker start

# Help debug the job's disk space.
df -h

cd github/swift-apis
cp -R /opt/google-cloud-sdk .
sudo -E docker build -t build-img -f Dockerfile --build-arg swift_tf_url .

sudo docker create --name build-container build-img
mkdir -p "$KOKORO_ARTIFACTS_DIR/swift_apis_benchmarks"
sudo docker cp build-container:/swift-apis/benchmark_results.xml "$KOKORO_ARTIFACTS_DIR/swift_apis_benchmarks/sponge_log.xml"
