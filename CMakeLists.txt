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
option(USE_BUNDLED_X10
  "Use the x10 library bundled in the active Swift toolchain" OFF)
option(USE_BUNDLED_CTENSORFLOW
  "Use the CTensorFlow module bundled in the active Swift toolchain" OFF)

if(BUILD_X10)
  find_package(X10)
  if(NOT X10_FOUND AND NOT USE_BUNDLED_X10)
    include(ExternalProject)

    ExternalProject_Add(libtensorflow
      GIT_REPOSITORY
        git://github.com/tensorflow/tensorflow
      GIT_TAG
        r2.2
      UPDATE_DISCONNECTED
        TRUE
      CONFIGURE_COMMAND
        COMMAND
          ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/Sources/CX10 <SOURCE_DIR>/swift_bindings
        COMMAND
          ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/Sources/x10/xla_client <SOURCE_DIR>/tensorflow/compiler/xla/xla_client
        COMMAND
          ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/Sources/x10/xla_tensor <SOURCE_DIR>/tensorflow/compiler/tf2xla/xla_tensor
        COMMAND
          # Note that the configure script is affected by the following environment variables:
          #   TF_CUDA_VERSION
          #   TF_NEED_CUDA
          #   TF_CUDNN_VERSION
          #   CUDA_TOOLKIT_PATH
          #   CUDNN_INSTALL_PATH
          sh -c "yes '' | <SOURCE_DIR>/configure"
      BUILD_COMMAND
        COMMAND
          rm -rf <SOURCE_DIR>/bazel-bin # ${CMAKE_COMMAND} -E rm -Rrf <SOURCE_DIR>/bazel-bin
        COMMAND
          bazel build -c opt --define framework_shared_object=false //tensorflow/compiler/tf2xla/xla_tensor:x10 --nocheck_visibility
        COMMAND
          bazel shutdown
      INSTALL_COMMAND
        "" # we do not do any installation, we fetch out the artifacts manually
      BUILD_IN_SOURCE
        TRUE
      BUILD_BYPRODUCTS
        <SOURCE_DIR>/bazel-bin/tensorflow/compiler/tf2xla/xla_tensor/${CMAKE_SHARED_LIBRARY_PREFIX}x10${CMAKE_SHARED_LIBRARY_SUFFIX}
      USES_TERMINAL_BUILD
        TRUE
      STEP_TARGETS
        build)
    ExternalProject_Get_Property(libtensorflow SOURCE_DIR)

    set(X10_LIBRARY ${SOURCE_DIR}/bazel-bin/tensorflow/compiler/tf2xla/xla_tensor/${CMAKE_SHARED_LIBRARY_PREFIX}x10${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(X10_INCLUDE_DIRS "${SOURCE_DIR};${SOURCE_DIR}/bazel-bin;${SOURCE_DIR}/bazel-libtensorflow/external/com_google_absl;${SOURCE_DIR}/bazel-libtensorflow/external/com_google_protobuf/src;${SOURCE_DIR}/bazel-libtensorflow/external/eigen_archive")
    add_library(x10 IMPORTED UNKNOWN)
    set_target_properties(x10 PROPERTIES
      IMPORTED_LOCATION ${X10_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES "${X10_INCLUDE_DIRS}")
    add_dependencies(x10
      libtensorflow-build)

    get_target_property(DIRECTORIES x10 INTERFACE_INCLUDE_DIRECTORIES)
    foreach(directory ${DIRECTORIES})
      file(MAKE_DIRECTORY ${directory})
    endforeach()
  endif()
endif()

if(BUILD_X10)
  if(NOT X10_FOUND AND NOT USE_BUNDLED_X10)
    add_library(tensorflow UNKNOWN IMPORTED)
    set_target_properties(tensorflow PROPERTIES
      IMPORTED_LOCATION ${X10_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES "${SOURCE_DIR}")
  else()
    add_library(tensorflow UNKNOWN IMPORTED)
    set_target_properties(tensorflow PROPERTIES
      IMPORTED_LOCATION ${X10_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES "${X10_INCLUDE_DIRS}")
  endif()
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

if(BUILD_X10 AND NOT X10_FOUND AND NOT USE_BUNDLED_X10)
  get_swift_host_os(host_os)
  install(FILES ${SOURCE_DIR}/bazel-bin/tensorflow/compiler/tf2xla/xla_tensor/${CMAKE_SHARED_LIBRARY_PREFIX}x10${CMAKE_SHARED_LIBRARY_SUFFIX}
    DESTINATION lib/swift/${host_os})
endif()
