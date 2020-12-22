# Swift for TensorFlow Deep Learning Library

## Development

### Requirements

* The latest Swift toolchain snapshot from [swift.org](https://swift.org/download/#sanpshots).

### Building

### X10

X10 provides the underlying lazy tensor implementation that is used for the
tensor computation in the TensorFlow Swift APIs. This library builds atop of
the TensorFlow library, using the XLA compiler to perform optimizations over
the tensor computation.

In the case that a prebuilt option is not available for your platform, note that
by default the CMake based build will build a copy of X10 if it does not find
one.

#### (Option 1) Use a prebuilt version of X10

You can use a prebuilt version of the X10 library for building the TensorFlow
Swift APIs package if you do not need to make changes to the X10 library
implementation.

There are prebuilt versions of the X10 library for certain platforms. If a
prebuilt library is unavailable for your desired platform, you can build X10
from source following the instructions below.

- [Windows 10 (x64)](https://artprodeus21.artifacts.visualstudio.com/A8fd008a0-56bc-482c-ba46-67f9425510be/3133d6ab-80a8-4996-ac4f-03df25cd3224/_apis/artifact/cGlwZWxpbmVhcnRpZmFjdDovL2NvbXBuZXJkL3Byb2plY3RJZC8zMTMzZDZhYi04MGE4LTQ5OTYtYWM0Zi0wM2RmMjVjZDMyMjQvYnVpbGRJZC80Mzc2OC9hcnRpZmFjdE5hbWUvdGVuc29yZmxvdy13aW5kb3dzLXg2NA2/content?format=zip)
- [macOS (x64)](https://artprodeus21.artifacts.visualstudio.com/A8fd008a0-56bc-482c-ba46-67f9425510be/3133d6ab-80a8-4996-ac4f-03df25cd3224/_apis/artifact/cGlwZWxpbmVhcnRpZmFjdDovL2NvbXBuZXJkL3Byb2plY3RJZC8zMTMzZDZhYi04MGE4LTQ5OTYtYWM0Zi0wM2RmMjVjZDMyMjQvYnVpbGRJZC80Mzc2OC9hcnRpZmFjdE5hbWUvdGVuc29yZmxvdy1kYXJ3aW4teDY00/content?format=zip)

The location where you extract the prebuilt library is important to remember
as it will be required in building the TensorFlow Swift APIs.

#### (Option 2) Building X10

Although using the prebuilt libraries is more convenient, there may be some
situations where you may need to build X10 yourself.  These include:

1. You are targeting a platform where we do not have prebuilt binaries for X10
2. You are attempting to make changes to the X10 implementation itself, and must
therefore build a new version.

#### Requirements

* Bazel. This can be installed [manually][bazel] or with
[Bazelisk][bazelisk]. You will need a version supported by TensorFlow
(between `_TF_MIN_BAZEL_VERSION` and `_TF_MAX_BAZEL_VERSION` as specified in
[tensorflow/configure.py][configure.py]).
* Python3 with [numpy][numpy].

#### Building

The X10 implementation is distributed as part of the Swift for TensorFlow APIs
repository. It consists of two halves:

1. [XLA Client](Sources/x10/xla_client): provides an abstract interface for dispatching XLA computations on devices
2. [X10](Sources/x10/xla_tensor): Lowering rules and tracing support for Tensors

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


*Note: This must be executed in the x64 Native Developer Command Prompt*

*Note: You will either need to be running with elevated privilleges, have rights to create symbolic links and junctions, or have enabled developer mode to successfully create the directory junctions.  You may alternatively copy the sources instead of creating a junction.*

In an empty directory:

```cmd
:: clone swift-apis
git clone git://github.com/tensorflow/swift-apis
:: checkout tensorflow
git clone --depth 1 --no-tags git://github.com/tensorflow/tensorflow
git -C tensorflow checkout -B refs/heads/r2.4

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
cd tensorflow
bazel --output_user_root %CD%/../caches/bazel/tensorflow build -c opt --copt /D_USE_MATH_DEFINES --define framework_shared_object=false --config short_logs --nocheck_visibility //tensorflow:tensorflow //tensorflow:tensorflow_dll_import_lib //tensorflow/compiler/tf2xla/xla_tensor:x10 //tensorflow/compiler/tf2xla/xla_tensor:x10_dll_import_lib
:: terminate bazel daemon
bazel --output_user_root %CD%/../caches/bazel/tensorflow shutdown
cd ..

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
    
*Note: If you are unable to run bazel on macOS due to an error about an unverified developer due to System Integrity Protection (SIP), you can use `xattr -dr com.apple.quarantine bazel`*

In an empty directory:

```bash
# clone swift-apis
git clone git://github.com/tensorflow/swift-apis
# checkout tensorflow
git clone --depth 1 --no-tags git://github.com/tensorflow/tensorflow
git -C tensorflow checkout -B refs/heads/r2.4

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
cd tensorflow
bazel --output_user_root ${PWD}/../caches/bazel/tensorflow build -c opt --define framework_shared_object=false --config short_logs --nocheck_visibility //tensorflow:tensorflow //tensorflow/compiler/tf2xla/xla_tensor:x10
# terminate bazel daemon
bazel --output_user_root ${PWD}/../caches/bazel/tensorflow shutdown
cd ..

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

### (TensorFlow) Swift APIs

#### SwiftPM

*Note: Building with SwiftPM does not include changes to X10 modules.*

Because X10 is a required component for building the TensorFlow Swift APIs,
and the project does not build with Swift Package Manager, we must pass along
the necessary information to the build. We need to pass two items to the
build:

1. the location of the X10 & TensorFlow headers
2. the location of the X10 & TensorFlow libraries

The path to these libraries is not fixed and depends on your machine setup.
You should substitute the paths with the appropriate values. In the example
commands below, we assume that the library is packaged in a traditional Unix
style layout and placed in `/Library/tensorflow-2.4.0`.

```shell
$ swift build -Xcc -I/Library/tensorflow-2.4.0/usr/include -Xlinker -L/Library/tensorflow-2.4.0/usr/lib
```

```shell
$ swift test -Xcc -I/Library/tensorflow-2.4.0/usr/include -Xlinker -L/Library/tensorflow-2.4.0/usr/lib
```

#### CMake

*Note: In-tree builds are not supported.*

*Note: To enable CUDA support, run `export TF_NEED_CUDA=1` before the steps below.*

*Note: If `swiftc` is not in your `PATH`, you must specify the path to it using
`-D CMAKE_Swift_COMPILER=`.*

By default, CMake will build X10 if it is not informed of where the library is
available.  If you wish to build X10 as part of the CMake build, ensure that you
do not have the X10 modules in the toolchain that you are using to develop here
(more explicitly, use a stock toolchain rather than a TensorFlow toolchain).

```shell
cmake -B out -G Ninja -S swift-apis -D CMAKE_BUILD_TYPE=Release
cmake --build out
```

If you have a prebuilt version of the X10 library that you wish to use (i.e. you
are using one of the prebuilt libraries referenced above or have built it
yourself above), you can inform CMake where to find the build time components of
the library (SDK contents) and avoid building X10.

The path to these libraries is not fixed and depends on your machine setup.
You should substitute the paths with the appropriate values. In the example
commands below, we assume that the library is packaged in a traditional Unix
style layout and placed in `/Library/tensorflow-2.4.0`.

Because the library name differs based on the platform, the following examples
may help identify what the flags should look like for the target that you are
building for.

macOS:

```shell
-D X10_LIBRARY=/Library/tensorflow-2.4.0/usr/lib/libx10.dylib -D X10_INCLUDE_DIRS=/Library/tensorflow-2.4.0/usr/include
```

Windows:

```shell
-D X10_LIBRARY=/Library/tensorflow-2.4.0/usr/lib/x10.lib -D X10_INCLUDE_DIRS=/Library/tensorflow-2.4.0/usr/include
```

Other Unix systems (e.g. Linux, BSD, Android, etc):

```shell
-D X10_LIBRARY=/Library/tensorflow-2.4.0/usr/lib/libx10.so -D X10_INCLUDE_DIRS=/Library/tensorflow-2.4.0/usr/include
```

To run tests:

*Note: To view failure output, run `export CTEST_OUTPUT_ON_FAILURE=1` before
running tests.*

```shell
cmake --build out --target test
```

If you are not intending to develop X10, you can reduce the build times by
using the bundled X10 in the toolchain using
`-D USE_BUNDLED_X10=YES -D USE_BUNDLED_CTENSORFLOW=YES`:

```shell
cmake -B out -D USE_BUNDLED_CTENSORFLOW=YES -D USE_BUNDLED_X10=YES -G Ninja -S swift-apis
cmake --build out
cmake --build out --target test
```

#### macOS

On macOS, passing `-D BUILD_TESTING=NO` is currently necessary to skip building
tests. This avoids an error: `cannot load underlying module for 'XCTest'`.

```shell
cmake -B out -D USE_BUNDLED_CTENSORFLOW=YES -D USE_BUNDLED_X10=YES -D BUILD_TESTING=NO -G Ninja -S swift-apis
cmake --build out
```

[bazel]: https://docs.bazel.build/versions/master/install.html
[bazelisk]: https://github.com/bazelbuild/bazelisk
[configure.py]: https://github.com/tensorflow/tensorflow/blob/master/configure.py
[numpy]: https://numpy.org/
