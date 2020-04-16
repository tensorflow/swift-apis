licenses(["notice"])  # Apache 2.0

load(
    "//utils:tensorflow.bzl",
    "s4tf_cc_shared_object",
)

cc_library(
    name = "tensor",
    srcs = glob(
        [
            "*.cpp",
            "ops/*.cpp",
        ],
        exclude = ["test.cpp"],
    ),
    hdrs = glob([
        "*.h",
        "ops/*.h",
    ]),
    deps = [
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/c:c_api_experimental",
        "@org_tensorflow//tensorflow/c/eager:c_api",
        "@org_tensorflow//tensorflow/c/eager:c_api_experimental",
        "@org_tensorflow//tensorflow/compiler/tf2xla:common",
        "@org_tensorflow//tensorflow/compiler/tf2xla/kernels:conv_op_helpers",
        "@org_tensorflow//tensorflow/compiler/tf2xla/lib:random",
        "@org_tensorflow//tensorflow/compiler/tf2xla/lib:scatter",
        "@org_tensorflow//tensorflow/compiler/xla:literal",
        "@org_tensorflow//tensorflow/compiler/xla:literal_util",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:status",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/compiler/xla:util",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/client:padding",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:arithmetic",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:comparators",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:constants",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:logdet",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:math",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:matrix",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:pooling",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:prng",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:qr",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:self_adjoint_eig",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:slicing",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:svd",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "//xla_client:xrt_computation_client",
        "@org_tensorflow//tensorflow/core:core_cpu_lib",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/kernels:conv_ops",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

s4tf_cc_shared_object(
    name = "libx10.so",
    linkopts = [
        "-z defs",
        "-s",
        "-Wl,--version-script,$(location :tf_version_script.lds)",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor",
        ":tf_exported_symbols.lds",
        ":tf_version_script.lds",
    ],
)

