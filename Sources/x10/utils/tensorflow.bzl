# Taken in part from tensorflow/tensorflow.bzl.

load(
    "@org_tensorflow//tensorflow/core/platform:rules_cc.bzl",
    "cc_binary",
    "cc_library",
    "cc_test",
)

load(
    "@org_tensorflow//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "VERSION",
    "VERSION_MAJOR",
    "SHARED_LIBRARY_NAME_PATTERNS",
)

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def _make_search_paths(prefix, levels_to_root):
    return ",".join(
        [
            "-rpath,%s/%s" % (prefix, "/".join([".."] * search_level))
            for search_level in range(levels_to_root + 1)
        ],
    )

def _rpath_linkopts(name):
    # Search parent directories up to the TensorFlow root directory for shared
    # object dependencies, even if this op shared object is deeply nested
    # (e.g. tensorflow/contrib/package:python/ops/_op_lib.so). tensorflow/ is then
    # the root and tensorflow/libtensorflow_framework.so should exist when
    # deployed. Other shared object dependencies (e.g. shared between contrib/
    # ops) are picked up as long as they are in either the same or a parent
    # directory in the tensorflow/ tree.
    levels_to_root = native.package_name().count("/") + name.count("/")
    return select({
        clean_dep("@org_tensorflow//tensorflow:macos"): [
            "-Wl,%s" % (_make_search_paths("@loader_path", levels_to_root),),
        ],
        clean_dep("@org_tensorflow//tensorflow:windows"): [],
        "//conditions:default": [
            "-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),),
        ],
    })

def s4tf_cc_shared_object(
        name,
        srcs = [],
        deps = [],
        data = [],
        linkopts = [],
        soversion = None,
        kernels = [],
        per_os_targets = False,  # Generate targets with SHARED_LIBRARY_NAME_PATTERNS
        visibility = None,
        **kwargs):
    """Configure the shared object (.so) file for TensorFlow."""
    if soversion != None:
        suffix = "." + str(soversion).split(".")[0]
        longsuffix = "." + str(soversion)
    else:
        suffix = ""
        longsuffix = ""

    if per_os_targets:
        names = [
            (
                pattern % (name, ""),
                pattern % (name, suffix),
                pattern % (name, longsuffix),
            )
            for pattern in SHARED_LIBRARY_NAME_PATTERNS
        ]
    else:
        names = [(
            name,
            name + suffix,
            name + longsuffix,
        )]

    for name_os, name_os_major, name_os_full in names:
        # Windows DLLs cant be versioned
        if name_os.endswith(".dll"):
            name_os_major = name_os
            name_os_full = name_os

        if name_os != name_os_major:
            native.genrule(
                name = name_os + "_sym",
                outs = [name_os],
                srcs = [name_os_major],
                output_to_bindir = 1,
                cmd = "ln -sf $$(basename $<) $@",
            )
            native.genrule(
                name = name_os_major + "_sym",
                outs = [name_os_major],
                srcs = [name_os_full],
                output_to_bindir = 1,
                cmd = "ln -sf $$(basename $<) $@",
            )

        soname = name_os_major.split("/")[-1]

        data_extra = []

        cc_binary(
            name = name_os_full,
            srcs = srcs,
            deps = deps,
            linkshared = 1,
            data = data + data_extra,
            linkopts = linkopts + _rpath_linkopts(name_os_full) + select({
                clean_dep("@org_tensorflow//tensorflow:macos"): [
                    "-Wl,-install_name,@rpath/" + soname,
                ],
                clean_dep("@org_tensorflow//tensorflow:windows"): [],
                "//conditions:default": [
                    "-Wl,-soname," + soname,
                ],
            }),
            visibility = visibility,
            **kwargs
        )

    flat_names = [item for sublist in names for item in sublist]
    if name not in flat_names:
        native.filegroup(
            name = name,
            srcs = select({
                "@org_tensorflow//tensorflow:windows": [":%s.dll" % (name)],
                "@org_tensorflow//tensorflow:macos": [":lib%s%s.dylib" % (name, longsuffix)],
                "//conditions:default": [":lib%s.so%s" % (name, longsuffix)],
            }),
            visibility = visibility,
        )
