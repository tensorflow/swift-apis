cc_library(
    name = "device_wrapper",
    srcs = ["device_wrapper.cc"],
    hdrs = ["device_wrapper.h"],
    deps = [
        "//tensorflow/compiler/xla/xla_client:xrt_computation_client",
        "//tensorflow/compiler/tf2xla/xla_tensor:tensor",
    ],
)

cc_library(
    name = "xla_tensor_wrapper",
    srcs = ["xla_tensor_wrapper.cc"],
    hdrs = ["xla_tensor_wrapper.h"],
    deps = [
        ":device_wrapper",
        "//tensorflow/compiler/tf2xla/xla_tensor:tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "xla_tensor_tf_ops",
    srcs = ["xla_tensor_tf_ops.cc"],
    hdrs = ["xla_tensor_tf_ops.h"],
    deps = [
        ":xla_tensor_wrapper",
        "//tensorflow/compiler/tf2xla/xla_tensor:tensor",
        "//tensorflow/compiler/xla/client/lib:pooling",
        "//tensorflow/core:framework",
    ],
)
