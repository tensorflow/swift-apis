# Swift for TensorFlow Developer Guide

## Requirements

* The latest Swift toolchain snapshot from [swift.org][swift]

Swift for TensorFlow APIs can be built with either:

* CMake 3.16 or later from [cmake.org][cmake]
* Swift Package Manager (included in the above toolchain)

## Components

Building Swift for TensorFlow APIs involves two distinct components:

1. X10
2. Swift APIs

<details>
    <summary>What is X10?</summary>

> X10 provides the underlying lazy tensor implementation that is used for
> tensor computations in the Swift for TensorFlow APIs. This library builds on
> top of the TensorFlow library, using the XLA compiler to perform
> optimizations.
>
> The X10 implementation consists of two halves:
>
> 1. [XLA Client](Sources/x10/xla_client): provides an abstract interface for
>    dispatching XLA computations on devices
> 2. [X10](Sources/x10/xla_tensor): Lowering rules and tracing support for
>    Tensors


</details>


The two components can be built together (CMake only) or separately.

Follow the instructions below based on your preferred build tool (CMake or
SwiftPM).

## Building With CMake

With CMake, X10 and Swift APIs can be built either together or separately.

To determine the appropriate build path, first consider your platform and
whether you are modifying X10.

* If you are modifying files in any subdirectory within **Sources/CX10** or
  **Sources/x10**, you are modifying X10 and must follow
  [**Option 1**](#option-1-build-x10-and-swift-apis-together).
* If you are not modifying X10 and are on Windows or macOS or have previously
  built X10, use [**Option 2**](#option-2-use-a-prebuilt-version-of-x10).
* Otherwise, use [**Option 1**](#option-1-build-x10-and-swift-apis-together).

> **Note:** In-tree builds are not supported.

> **Note:** To enable CUDA support, run `export TF_NEED_CUDA=1` before the
  steps below.

> **Note:** If `swiftc` is not in your `PATH`, you must specify the path to it
  using `-D CMAKE_Swift_COMPILER=`.

### Option 1: Build X10 and Swift APIs together

This command builds both X10 and Swift APIs in sequence. Ensure that you
do not have the X10 modules in the toolchain that you are using to develop here
(more explicitly, use a stock toolchain rather than a TensorFlow toolchain).

```shell
cmake -B out -G Ninja -S swift-apis -D CMAKE_BUILD_TYPE=Release
cmake --build out
```

### Option 2: Use a prebuilt version of X10

If you are not modifying X10 or have already built it yourself using Option 1
above, you can inform CMake where to find the build time components of the
library (SDK contents).

There are prebuilt versions of the X10 library for certain platforms. If a
prebuilt library is unavailable for your desired platform, you can build X10
from source following the instructions in [Option 1](#option-2-build-x10).

- [Windows 10 (x64)][windows10]
- [macOS (x64)][macOS]

Note the location where you extract the prebuilt library. The path to these
libraries is not fixed and depends on your machine setup.
You should substitute the paths with the appropriate values. In the example
commands below, we assume that the library is packaged in a traditional Unix
style layout and placed in `/Library/tensorflow-2.4.0`.

Because the library name differs based on the platform, the following examples
may help identify what the flags should look like for the target that you are
building for.

macOS:

```shell
cmake -B out -G Ninja -S swift-apis -D CMAKE_BUILD_TYPE=Release \
  -D X10_LIBRARY=/Library/tensorflow-2.4.0/usr/lib/libx10.dylib \
  -D X10_INCLUDE_DIRS=/Library/tensorflow-2.4.0/usr/include
cmake --build out
```

Windows:

```shell
cmake -B out -G Ninja -S swift-apis -D CMAKE_BUILD_TYPE=Release \
  -D X10_LIBRARY=/Library/tensorflow-2.4.0/usr/lib/x10.lib \
  -D X10_INCLUDE_DIRS=/Library/tensorflow-2.4.0/usr/include
cmake --build out
```

Other Unix systems (e.g. Linux, BSD, Android, etc):

```shell
cmake -B out -G Ninja -S swift-apis -D CMAKE_BUILD_TYPE=Release \
  -D X10_LIBRARY=/Library/tensorflow-2.4.0/usr/lib/libx10.so \
  -D X10_INCLUDE_DIRS=/Library/tensorflow-2.4.0/usr/include
cmake --build out
```

### Running tests

To run tests:

> **Note:** To view failure output, run `export CTEST_OUTPUT_ON_FAILURE=1`
  before running tests.

```shell
cmake --build out --target test
```

#### macOS

On macOS, passing `-D BUILD_TESTING=NO` is currently necessary to skip building
tests. This avoids an error: `cannot load underlying module for 'XCTest'`.

```shell
cmake -B out -G Ninja -S swift-apis -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_TESTING=NO
cmake --build out
```

## Building With Swift Package Manager

Building with SwiftPM involves building X10 and Swift APIs separately.

### Building X10

To determine the appropriate build path, first consider your platform and
whether you are modifying X10.

* If you are modifying files in any subdirectory within **Sources/CX10** or
  **Sources/x10**, you are modifying X10 and must follow
  [**Option 1**](#option-1-build-x10).
* If you are on Windows or macOS and are not modifying X10, use
  [**Option 2**](#option-2-use-a-prebuilt-version-of-x10-1).
* Otherwise, use [**Option 1**](#option-1-build-x10).

#### Option 1: Build X10

Although using the prebuilt libraries is more convenient, there may be some
situations where you may need to build X10 yourself.  These include:

1. You are targeting a platform where we do not have prebuilt binaries for X10
2. You are attempting to make changes to the X10 implementation itself, and must
therefore build a new version.

##### Requirements

* Bazel. This can be installed [manually][bazel] or with
[Bazelisk][bazelisk]. You will need a version supported by TensorFlow
(between `_TF_MIN_BAZEL_VERSION` and `_TF_MAX_BAZEL_VERSION` as specified in
[tensorflow/configure.py][configure.py]).
* Python3 with [numpy][numpy].

##### Building

The library is designed to be built as part of the
[tensorflow](https://github.com/tensorflow/tensorflow) build. As such, in
order to build X10, you must build tensorflow.

Currently X10 is developed against TensorFlow 2.4.0. The following build
scripts provide commands to build on common platforms. They largely replicate
the build instructions for TensorFlow. The instructions diverge in that we
must copy the additional X10 library sources into the tensorflow repository.
The following table identifies the copied locations:

| swift-apis source | tensorflow destination |
|-------------------|------------------------|
| Sources/CX10      | swift_bindings         |
| Sources/x10/xla_client | tensorflow/compiler/xla/xla_client |
| Sources/x10/xla_tensor | tensorflow/compiler/tf2xla/xla_tensor |

We build two specific targets:
1. `//tensorflow:tensorflow`
2. `//tensorflow/compiler/tf2xla/xla_tensor:x10`

On Windows, we build the additional targets to allow us to link against the
libraries:
1. `//tensorflow:tensorflow_dll_import_lib`
2. `//tensorflow/compiler/tf2xla/xla_tensor:x10_dll_import_lib `

We must pass the `--nocheck_visibility` flag to bazel to accomodate the new
libraries.

<details>
    <summary>Windows Build Script</summary>


*Note: This must be executed in the x64 Native Developer Command Prompt.*

*Note: You will either need to be running with elevated privilleges, have
rights to create symbolic links and junctions, or have enabled developer mode
to successfully create the directory junctions. You may alternatively copy the
sources instead of creating a junction.*

```cmd
:: clone swift-apis
git clone git://github.com/tensorflow/swift-apis
:: checkout tensorflow
git clone --depth 1 --no-tags git://github.com/tensorflow/tensorflow
git -C tensorflow checkout refs/heads/r2.4

:: Link X10 into the source tree
mklink /J %CD%\tensorflow\swift_bindings %CD%\swift-apis\Sources\CX10
mklink /J %CD%\tensorflow\tensorflow\compiler\xla\xla_client %CD%\swift-apis\Sources\x10\xla_client
mklink /J %CD%\tensorflow\tensorflow\compiler\tf2xla\xla_tensor %CD%\swift-apis\Sources\x10\xla_tensor

:: configure path - we need the Git tools in the path
path %ProgramFiles%\Git\usr\bin;%PATH%
:: ensure that python dependencies are available
python -m pip install --user numpy six
:: configure X10/TensorFlow
set TF_ENABLE_XLA=1
set TF_NEED_ROCM=0
set TF_NEED_CUDA=0
set TF_CUDA_COMPUTE_CAPABILITIES=7.5
set CC_OPT_FLAGS="/arch:AVX /D_USE_MATH_DEFINES"
set TF_OVERRIDE_EIGEN_STRONG_INLINE=1
.\tensorflow\configure.py
:: build
set BAZEL_SH=%ProgramFiles%\Git\usr\bin\bash.exe
set BAZEL_VC=%VCINSTALLDIR%
bazel --output_user_root %CD%/caches/bazel/tensorflow build -c opt --copt /D_USE_MATH_DEFINES --define framework_shared_object=false --config short_logs --nocheck_visibility //tensorflow:tensorflow //tensorflow:tensorflow_dll_import_lib //tensorflow/compiler/tf2xla/xla_tensor:x10 //tensorflow/compiler/tf2xla/xla_tensor:x10_dll_import_lib
:: terminate bazel daemon
bazel --output_user_root %CD%/caches/bazel/tensorflow shutdown

:: package
set DESTDIR=%CD%\Library\tensorflow-windows-%VSCMD_ARG_TGT_ARCH%\tensorflow-2.4.0

md %DESTDIR\usr\bin
copy tensorflow\bazel-bin\tensorflow\tensorflow.dll %DESTDIR%\usr\bin\
copy tensorflow\bazel-bin\tensorflow\compiler\tf2xla\xla_tensor\x10.dll %DESTDIR%\usr\bin\

md %DESTDIR%\usr\lib
copy tensorflow\bazel-out\%VSCMD_ARG_TGT_ARCH%_windows-opt\bin\tensorflow\tensorflow.lib %DESTDIR%\usr\lib\
copy tensorflow\bazel-out\%VSCMD_ARG_TGT_ARCH%_windows-opt\bin\tensorflow\compiler\tf2xla\xla_tensor\x10.lib %DESTDIR%\usr\lib\

md %DESTDIR%\usr\include\tensorflow\c
copy tensorflow\tensorflow\c\c_api.h %DESTDIR%\usr\include\tensorflow\c\
copy tensorflow\tensorflow\c\c_api_experimental.h %DESTDIR%\usr\include\tensorflow\c\
copy tensorflow\tensorflow\c\tf_attrtype.h %DESTDIR%\usr\include\tensorflow\c\
copy tensorflow\tensorflow\c\tf_datatype.h %DESTDIR%\usr\include\tensorflow\c\
copy tensorflow\tensorflow\c\tf_file_statistics.h %DESTDIR%\usr\include\tensorflow\c\
copy tensorflow\tensorflow\c\tf_status.h %DESTDIR%\usr\include\tensorflow\c\
copy tensorflow\tensorflow\c\tf_tensor.h %DESTDIR%\usr\include\tensorflow\c\

md %DESTDIR%\usr\include\tensorflow\c\eager
cp tensorflow\tensorflow\c\eager\c_api.h %DESTDIR%\usr\include\tensorflow\c\eager\

md %DESTDIR%\usr\include\x10
copy swift-apis\Sources\x10\swift_bindings\device_wrapper.h %DESTDIR%\usr\include\x10\
copy swift-apis\Sources\x10\swift_bindings\xla_tensor_tf_ops.h %DESTDIR%\usr\include\x10\
copy swift-apis\Sources\x10\swift_bindings\xla_tensor_wrapper.h %DESTDIR%\usr\include\x10\

md %DESTDIR%\usr\share
copy tensorflow\bazel-out\%VSCMD_ARG_TGT_ARCH%_windows-opt\bin\tensorflow\tensorflow_filtered_def_file.def %DESTDIR%\usr\share
```
</details>

<details>
    <summary>macOS/Linux Build Script</summary>
    
> **Note:** If you are unable to run bazel on macOS due to an error about an
> unverified developer due to System Integrity Protection (SIP), you can use
> `xattr -dr com.apple.quarantine bazel` to designate it as trusted.

```bash
# clone swift-apis
git clone git://github.com/tensorflow/swift-apis
# checkout tensorflow
git clone --depth 1 --no-tags git://github.com/tensorflow/tensorflow
git -C tensorflow checkout refs/heads/r2.4

# Link X10 into the source tree
ln -sf ${PWD}/swift-apis/Sources/CX10 ${PWD}/tensorflow/swift_bindings
ln -sf ${PWD}/swift-apis/Sources/x10/xla_client ${PWD}/tensorflow/tensorflow/compiler/xla/xla_client
ln -sf ${PWD}/swift-apis/Sources/x10/xla_tensor ${PWD}/tensorflow/tensorflow/compiler/tf2xla/xla_tensor

# ensure that python dependencies are available
python3 -m pip install --user numpy six
# configure X10/TensorFlow
export PYTHON_BIN_PATH=$(which python3)
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_NEED_OPENCL_SYCL=0
export TF_DOWNLOAD_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_CONFIGURE_IOS=0 
export TF_ENABLE_XLA=1
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_CUDA_COMPUTE_CAPABILITIES=7.5
export CC_OPT_FLAGS="-march=native"
python3 ./tensorflow/configure.py
bazel --output_user_root ${PWD}/caches/bazel/tensorflow build -c opt --define framework_shared_object=false --config short_logs --nocheck_visibility //tensorflow:tensorflow //tensorflow/compiler/tf2xla/xla_tensor:x10
# terminate bazel daemon
bazel --output_user_root ${PWD}/caches/bazel/tensorflow shutdown

# package
DESTDIR=${PWD}/Library/tensorflow-$(echo $(uname -s) | tr 'A-Z' 'a-z')-$(uname -m)/tensorflow-2.4.0

mkdir -p ${DESTDIR}/usr/lib
cp tensorflow/bazel-bin/tensorflow/libtensorflow-2.4.0.(dylib|so) ${DESTDIR}/usr/lib/
cp tensorflow/bazel-bin/tensorflow/compiler/tf2xla/xla_tensor/libx10.(dylib|so) ${DESTDIR}/usr/lib/

mkdir -p ${DESTDIR}/usr/include/tensorflow/c
cp tensorflow/tensorflow/c/c_api.h ${DESTDIR}/usr/include/tensorflow/c/
cp tensorflow/tensorflow/c/c_api_experimental.h ${DESTDIR}/usr/include/tensorflow/c/
cp tensorflow/tensorflow/c/tf_attrtype.h ${DESTDIR}/usr/include/tensorflow/c/
cp tensorflow/tensorflow/c/tf_datatype.h ${DESTDIR}/usr/include/tensorflow/c/
cp tensorflow/tensorflow/c/tf_file_statistics.h ${DESTDIR}/usr/include/tensorflow/c/
cp tensorflow/tensorflow/c/tf_status.h ${DESTDIR}/usr/include/tensorflow/c/
cp tensorflow/tensorflow/c/tf_tensor.h ${DESTDIR}/usr/include/tensorflow/c/

mkdir -p ${DESTDIR}/usr/include/tensorflow/c/eager
cp tensorflow/tensorflow/c/eager/c_api.h ${DESTDIR}/usr/include/tensorflow/c/eager/

mkdir -p ${DESTDIR}/usr/include/x10
cp swift-apis/Sources/x10/swift_bindings/device_wrapper.h ${DESTDIR}/usr/include/x10/
cp swift-apis/Sources/x10/swift_bindings/xla_tensor_tf_ops.h ${DESTDIR}/usr/include/x10/
cp swift-apis/Sources/x10/swift_bindings/xla_tensor_wrapper.h ${DESTDIR}/usr/include/x10/
```
</details>

#### Option 2: Use a prebuilt version of X10

You can use a prebuilt version of the X10 library for building the Swift for
TensorFlow APIs package if you are on a supported platform and do not need to
make changes to the X10 library implementation.

There are prebuilt versions of the X10 library for certain platforms. If a
prebuilt library is unavailable for your desired platform, you can build X10
from source following the instructions in [Option 1](#option-1-build-x10).

- [Windows 10 (x64)][windows10]
- [macOS (x64)][macOS]

Note the location where you extract the prebuilt library, since it is required
for building Swift for TensorFlow APIs.

### Building Swift APIs

> **Note:** This step requires pre-built X10 binaries. Building with SwiftPM
> does not include changes to X10.

SwiftPM requires two items:

1. The location of the X10 & TensorFlow headers.
2. The location of the X10 & TensorFlow libraries.

The path to these libraries is not fixed and depends on your machine setup.
You should substitute the paths with the appropriate values. In the example
commands below, we assume that the library is packaged in a traditional Unix
style layout and placed in `/Library/tensorflow-2.4.0`.

```shell
$ swift build -Xcc -I/Library/tensorflow-2.4.0/usr/include -Xlinker -L/Library/tensorflow-2.4.0/usr/lib
```

#### macOS

On macOS, in order to select the proper toolchain, the `TOOLCHAINS` environment
variable can be used to modify the selected Xcode toolchain temporarily.  The
macOS (Xcode) toolchain distributed from [swift.org][swift] has a
bundle identifier which can uniquely identify the toolchain to the system.  The
following attempts to determine the latest toolchain snapshot and extract the
identifier for it.

```shell
xpath 2>/dev/null $(find /Library/Developer/Toolchains ~/Library/Developer/Toolchains -type d -depth 1 -regex '.*/swift-DEVELOPMENT-SNAPSHOT-.*.xctoolchain | sort -u | tail -n 1)/Info.plist "/plist/dict/key[. = 'CFBundleIdentifier']/following-sibling::string[1]//text()"
```
This allows one to build the package as:
```shell
TOOLCHAINS=$(xpath 2>/dev/null $(find /Library/Developer/Toolchains ~/Library/Developer/Toolchains -type d -depth 1 -regex '.*/swift-DEVELOPMENT-SNAPSHOT-.*.xctoolchain | sort -u | tail -n 1)/Info.plist "/plist/dict/key[. = 'CFBundleIdentifier']/following-sibling::string[1]//text()") swift build -Xswiftc -DTENSORFLOW_USE_STANDARD_TOOLCHAIN -Xcc -I/Library/tensorflow-2.4.0/usr/include -Xlinker -L/Library/tensorflow-2.4.0/usr/lib
```

### Running tests

To run tests:

```shell
$ swift test -Xcc -I/Library/tensorflow-2.4.0/usr/include -Xlinker -L/Library/tensorflow-2.4.0/usr/lib
```

[swift]: https://swift.org/download/#snapshots
[cmake]: https://www.cmake.org/download
[windows10]: https://artprodeus21.artifacts.visualstudio.com/A8fd008a0-56bc-482c-ba46-67f9425510be/3133d6ab-80a8-4996-ac4f-03df25cd3224/_apis/artifact/cGlwZWxpbmVhcnRpZmFjdDovL2NvbXBuZXJkL3Byb2plY3RJZC8zMTMzZDZhYi04MGE4LTQ5OTYtYWM0Zi0wM2RmMjVjZDMyMjQvYnVpbGRJZC80Mzc2OC9hcnRpZmFjdE5hbWUvdGVuc29yZmxvdy13aW5kb3dzLXg2NA2/content?format=zip
[macOS]: https://artprodeus21.artifacts.visualstudio.com/A8fd008a0-56bc-482c-ba46-67f9425510be/3133d6ab-80a8-4996-ac4f-03df25cd3224/_apis/artifact/cGlwZWxpbmVhcnRpZmFjdDovL2NvbXBuZXJkL3Byb2plY3RJZC8zMTMzZDZhYi04MGE4LTQ5OTYtYWM0Zi0wM2RmMjVjZDMyMjQvYnVpbGRJZC80Mzc2OC9hcnRpZmFjdE5hbWUvdGVuc29yZmxvdy1kYXJ3aW4teDY00/content?format=zip
[bazel]: https://docs.bazel.build/versions/master/install.html
[bazelisk]: https://github.com/bazelbuild/bazelisk
[configure.py]: https://github.com/tensorflow/tensorflow/blob/master/configure.py
[numpy]: https://numpy.org/
