licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_shared_object",
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
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla/kernels:conv_op_helpers",
        "//tensorflow/compiler/tf2xla/lib:random",
        "//tensorflow/compiler/tf2xla/lib:scatter",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:comparators",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/compiler/xla/client/lib:logdet",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/client/lib:pooling",
        "//tensorflow/compiler/xla/client/lib:prng",
        "//tensorflow/compiler/xla/client/lib:qr",
        "//tensorflow/compiler/xla/client/lib:self_adjoint_eig",
        "//tensorflow/compiler/xla/client/lib:slicing",
        "//tensorflow/compiler/xla/client/lib:svd",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/xla_client:xrt_computation_client",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:conv_ops",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/debugging:stacktrace",
        "@com_google_absl//absl/debugging:symbolize",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

filegroup(
    name = "get_x10_dll_import_lib",
    srcs = [":x10.dll"],
    output_group = "interface_library",
    visibility = ["//visibility:public"],
)

genrule(
    name = "x10_dll_import_lib",
    srcs = [":get_x10_dll_import_lib"],
    outs = ["x10.lib"],
    cmd = select({
        "//tensorflow:windows": "cp -f $< $@",
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    visibility = ["//visibility:public"],
)

tf_cc_shared_object(
    name = "x10",
    linkopts = select({
        "//tensorflow:macos": [
            # TODO
        ],
        "//tensorflow:windows": [
        ],
        "//conditions:default": [
            "-z defs",
            "-s",
            "-Wl,--version-script,$(location :tf_version_script.lds)",
        ],
    }),
    per_os_targets = True,
    visibility = ["//visibility:public"],
    deps = [
        "//swift_bindings:device_wrapper",
        "//swift_bindings:xla_tensor_wrapper",
        "//swift_bindings:xla_tensor_tf_ops",
        ":tensor",
        ":tf_exported_symbols.lds",
        ":tf_version_script.lds",
    ],
)
