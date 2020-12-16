# Swift for TensorFlow Deep Learning Library

## Development

### Requirements

* [Swift for TensorFlow toolchain][toolchain].
* An environment that can run the Swift for TensorFlow toolchains: Ubuntu 18.04, macOS with Xcode 10, or Windows 10.
* Bazel. This can be installed [manually][bazel] or with
[Bazelisk][bazelisk]. You will need a version supported by TensorFlow
(between `_TF_MIN_BAZEL_VERSION` and `_TF_MAX_BAZEL_VERSION` as specified in
[tensorflow/configure.py][configure.py]).
* Python3 with [numpy][numpy].
* CMake.  CMake 3.16 or newer is required to build with CMake.

### Building and testing

#### SwiftPM

*Note: Building with SwiftPM does not include changes to X10 modules.*

```shell
$ swift build
```

*Note: Testing with SwiftPM does not run X10 tests.*

```shell
$ swift test
```

#### CMake

*Note: CMake is required for building X10 modules.*

In-tree builds are not supported.  

*Note: To enable CUDA support, run `export TF_NEED_CUDA=1` before the steps below.*

*Note: If `swiftc` is not in your `PATH`, you must specify the path to it using
`-D CMAKE_Swift_COMPILER=`.*

This will build X10 as part of the build.  Ensure that you do not have the
x10 modules in the toolchain that you are using to develop here.

```shell
cmake -B out -G Ninja -S swift-apis
cmake --build out
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
