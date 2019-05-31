# Swift for TensorFlow Ops Bindings

This repository contains TensorFlow ops bindings for
[Swift for TensorFlow](https://github.com/tensorflow/swift).

These bindings are automatically generated from TensorFlow ops
specified either using ops registered to the TensorFlow runtime
or via a protobuf file similar to
[ops.pbtxt](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/ops.pbtxt)
in the main TensorFlow repo.

## How to regenerate the bindings

To regenerate the swift ops bindings, run the following command. Note
that this will use the TensorFlow (1.9 or above) python package.

``` shell
python generate_wrappers.py --output_path=RawOpsGenerated.swift
```

Documentation gets automatically generated when adding a path to the
`api_def` proto directory. This directory should contain per operator
`api_def` protos with names like `api_def_OpName.pbtxt`.

```shell
python generate_wrappers.py --output_path=RawOpsGenerated.swift --api_def_path=/path/to/tensorflow/core/api_def/base_api
```
