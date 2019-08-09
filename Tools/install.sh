#!/usr/bin/env bash

TENSORFLOW_DIRECTORY='../../tensorflow'
TENSORFLOW_BIN_DIRECTORY="$TENSORFLOW_DIRECTORY/bazel-bin/tensorflow"
USR_DIRECTORY='../usr'

# set -x

function copy_file() {
  if [[ -L "$1/$2" ]]; then
    local target=`readlink "$1/$2"`
    copy_file $1 $target $3
    (cd $3; ln -s $target -f -r $2)
  else
    cp "$1/$2" $3
  fi
}

function fix_tf_header() {
  cp "$1" "$2"
  sed -i -e 's#include "'"$3"'tensorflow/c/c_api.h"#include "c_api.h"#g' "$2"
  sed -i -e 's#include "'"$3"'tensorflow/c/tf_attrtype.h"#include "tf_attrtype.h"#g' "$2"
  sed -i -e 's#include "'"$3"'tensorflow/c/tf_status.h"#include "tf_status.h"#g' "$2"
  sed -i -e 's#include "'"$3"'tensorflow/c/c_api_experimental.h"#include "c_api_experimental.h"#g' "$2"
  sed -i -e 's#include "'"$3"'tensorflow/c/eager/c_api.h"#include "c_api_eager.h"#g' "$2"
}

function install_header() {
  echo "Install header: " $1 $2
  fix_tf_header $1 "$USR_DIRECTORY/lib/swift/linux/x86_64/modulemaps/CTensorFlow/$2" ""
}

mkdir -p $USR_DIRECTORY/lib/swift/linux
copy_file $TENSORFLOW_BIN_DIRECTORY libtensorflow.so $USR_DIRECTORY/lib/swift/linux
copy_file $TENSORFLOW_BIN_DIRECTORY libtensorflow_framework.so $USR_DIRECTORY/lib/swift/linux

mkdir -p $USR_DIRECTORY/lib/swift/linux/x86_64/modulemaps/CTensorFlow
install_header "$TENSORFLOW_DIRECTORY/tensorflow/c/c_api.h" c_api.h
install_header "$TENSORFLOW_DIRECTORY/tensorflow/c/c_api_experimental.h" c_api_experimental.h
install_header "$TENSORFLOW_DIRECTORY/tensorflow/c/tf_attrtype.h" tf_attrtype.h
install_header "$TENSORFLOW_DIRECTORY/tensorflow/c/tf_status.h" tf_status.h
install_header "$TENSORFLOW_DIRECTORY/tensorflow/c/eager/c_api.h" c_api_eager.h
cp tools/module.modulemap "$USR_DIRECTORY/lib/swift/linux/x86_64/modulemaps/CTensorFlow/"

$USR_DIRECTORY/bin/swift build -Xswiftc -module-link-name -Xswiftc TensorFlow

BIN_DIR='.build/x86_64-unknown-linux/debug'

cp $BIN_DIR/libTensorFlow.so $USR_DIRECTORY/lib/swift/linux/
cp $BIN_DIR/TensorFlow.swiftdoc $USR_DIRECTORY/lib/swift/linux/x86_64/
cp $BIN_DIR/TensorFlow.swiftmodule $USR_DIRECTORY/lib/swift/linux/x86_64/
