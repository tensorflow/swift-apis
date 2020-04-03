cmake_minimum_required(VERSION 3.15.1)
project(TensorFlow
  LANGUAGES CXX Swift)

if(CMAKE_VERSION VERSION_LESS 3.17)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
    set(CMAKE_EXECUTABLE_RUNTIME_Swift_FLAG "-Xlinker -rpath -Xlinker ")
    set(CMAKE_SHARED_LIBRARY_RUNTIME_Swift_FLAG "-Xlinker -rpath -Xlinker ")
    if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
      set(CMAKE_EXECUTABLE_RUNTIME_Swift_FLAG_SEP "")
      set(CMAKE_SHARED_LIBRARY_RUNTIME_Swift_FLAG_SEP "")
    else()
      set(CMAKE_EXECUTABLE_RUNTIME_Swift_FLAG_SEP ":")
      set(CMAKE_SHARED_LIBRARY_RUNTIME_Swift_FLAG_SEP ":")
    endif()
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)

option(BUILD_X10 "enable the x10 tensor library" OFF)
option(USE_BUNDLED_CTENSORFLOW
  "Use the CTensorFlow module bundled in the active Swift toolchain" OFF)

if(BUILD_X10)
  include(ExternalProject)

  ExternalProject_Add(tensorflow
    GIT_REPOSITORY
      git://github.com/tensorflow/tensorflow
    GIT_TAG
      r2.2
    UPDATE_DISCONNECTED
      TRUE
    CONFIGURE_COMMAND
      COMMAND
        ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/Sources/x10/xla_client <SOURCE_DIR>/tensorflow/compiler/xla/xla_client
      COMMAND
        ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/Sources/x10/xla_tensor <SOURCE_DIR>/tensorflow/compiler/tf2xla/xla_tensor
      COMMAND
        TF_NEED_CUDA=1 CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu <SOURCE_DIR>/configure
    BUILD_COMMAND
      COMMAND
        ${CMAKE_COMMAND} -E rm -Rrf <SOURCE_DIR>/bazel-bin
      COMMAND
        bazel build -c opt --define framework_shared_object=false //tensorflow/compiler/tf2xla/xla_tensor:libx10.so
      COMMAND
        bazel shutdown
    INSTALL_COMMAND
      "" # we do not do any installation, we fetch out the artifacts manually
    BUILD_IN_SOURCE
      TRUE
    BUILD_BYPRODUCTS
      <SOURCE_DIR>/bazel-bin/tensorflow/compiler/tf2xla/xla_tensor/libx10.so
    USES_TERMINAL_BUILD
      TRUE
    STEP_TARGETS
      build)
  ExternalProject_Get_Property(tensorflow SOURCE_DIR)

  add_library(x10 IMPORTED UNKNOWN)
  set_target_properties(x10 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SOURCE_DIR};${SOURCE_DIR}/bazel-bin;${SOURCE_DIR}/bazel-tensorflow/external/com_google_absl;${SOURCE_DIR}/bazel-tensorflow/external/com_google_protobuf/src;${SOURCE_DIR}/bazel-tensorflow/external/eigen_archive"
    IMPORTED_LOCATION ${SOURCE_DIR}/bazel-bin/tensorflow/compiler/tf2xla/xla_tensor/libx10.so)
  add_dependencies(x10
    tensorflow-build)

  get_target_property(DIRECTORIES x10 INTERFACE_INCLUDE_DIRECTORIES)
  foreach(directory ${DIRECTORIES})
    file(MAKE_DIRECTORY ${directory})
  endforeach()
else()
  find_package(TensorFlow REQUIRED)
endif()

include(CTest)
include(SwiftSupport)

add_compile_options("$<$<COMPILE_LANGUAGE:Swift>:SHELL:-Xllvm -sil-inline-generics>")
add_compile_options("$<$<COMPILE_LANGUAGE:Swift>:SHELL:-Xllvm -sil-partial-specialization>")

add_subdirectory(Sources)
if(BUILD_TESTING)
  add_subdirectory(Tests)
endif()

if(BUILD_X10)
  get_swift_host_os(host_os)
  install(FILES ${SOURCE_DIR}/bazel-bin/tensorflow/compiler/tf2xla/xla_tensor/libx10.so
    DESTINATION lib/swift/${host_os})
endif()
