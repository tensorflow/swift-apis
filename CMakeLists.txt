cmake_minimum_required(VERSION 3.15.1)
project(TensorFlow
  LANGUAGES Swift)

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

set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)

option(USE_BUNDLED_X10
  "Use the x10 library bundled in the active Swift toolchain" OFF)
option(USE_BUNDLED_CTENSORFLOW
  "Use the CTensorFlow module bundled in the active Swift toolchain" OFF)
option(ENABLE_PYTHON_SUPPORT
  "Enable Python interop using PythonKit" ON)

if(ENABLE_PYTHON_SUPPORT)
  include(ExternalProject)

  ExternalProject_Add(python-kit
    GIT_REPOSITORY
      git://github.com/pvieito/PythonKit
    GIT_TAG
      master
    CMAKE_ARGS
      -D BUILD_SHARED_LIBS=YES
      -D CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
      -D CMAKE_Swift_COMPILER=${CMAKE_Swift_COMPILER}
      -D CMAKE_Swift_FLAGS=${CMAKE_Swift_FLAGS}
    INSTALL_COMMAND
      COMMAND
        ${CMAKE_COMMAND} -E copy_if_different <BINARY_DIR>/swift/PythonKit.swiftmodule ${CMAKE_Swift_MODULE_DIRECTORY}/
      COMMAND
        ${CMAKE_COMMAND} -E copy_if_different <BINARY_DIR>/swift/PythonKit.swiftdoc ${CMAKE_Swift_MODULE_DIRECTORY}/
    UPDATE_DISCONNECTED
      TRUE
    STEP_TARGETS
      install)
  ExternalProject_Get_Property(python-kit BINARY_DIR)
  ExternalProject_Get_Property(python-kit SOURCE_DIR)

  add_library(PythonKit IMPORTED UNKNOWN)
  set_target_properties(PythonKit PROPERTIES
    IMPORTED_LOCATION ${BINARY_DIR}/PythonKit/${CMAKE_SHARED_LIBRARY_PREFIX}PythonKit${CMAKE_SHARED_LIBRARY_SUFFIX}
    IMPORTED_IMPLIB ${BINARY_DIR}/PythonKit/${CMAKE_IMPORT_LIBRARY_PREFIX}PythonKit${CMAKE_IMPORT_LIBRARY_SUFFIX}
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_Swift_MODULE_DIRECTORY}")
  add_dependencies(PythonKit
    python-kit-install)

  file(MAKE_DIRECTORY ${CMAKE_Swift_MODULE_DIRECTORY})
endif()

find_package(X10)
if(NOT X10_FOUND AND NOT USE_BUNDLED_X10)
  include(ExternalProject)

  if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    set(VISIBILITY_FLAGS "")
  else()
    set(VISIBILITY_FLAGS --cxxopt=-fvisibility=hidden --cxxopt=-fvisibility-inlines-hidden)
  endif()

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
        bazel build ${VISIBILITY_FLAGS} -c opt --define framework_shared_object=false //tensorflow/compiler/tf2xla/xla_tensor:x10 --nocheck_visibility
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
  set(X10_INCLUDE_DIRS
    ${SOURCE_DIR}
    ${SOURCE_DIR}/bazel-bin
    ${SOURCE_DIR}/bazel-libtensorflow/external/com_google_absl
    ${SOURCE_DIR}/bazel-libtensorflow/external/com_google_protobuf/src
    ${SOURCE_DIR}/bazel-libtensorflow/external/eigen_archive)
  add_library(x10 IMPORTED UNKNOWN)
  set_target_properties(x10 PROPERTIES
    IMPORTED_LOCATION ${X10_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES "${X10_INCLUDE_DIRS}")
  add_dependencies(x10
    libtensorflow-build)

  foreach(directory ${X10_INCLUDE_DIRS})
    file(MAKE_DIRECTORY ${directory})
  endforeach()
endif()

find_package(dispatch CONFIG QUIET)
find_package(Foundation CONFIG QUIET)

include(CTest)
include(SwiftSupport)

add_compile_options("$<$<COMPILE_LANGUAGE:Swift>:SHELL:-Xllvm -sil-inline-generics>")
add_compile_options("$<$<COMPILE_LANGUAGE:Swift>:SHELL:-Xllvm -sil-partial-specialization>")

add_subdirectory(Sources)
if(BUILD_TESTING)
  add_subdirectory(Tests)
endif()

if(NOT X10_FOUND AND NOT USE_BUNDLED_X10)
  get_swift_host_os(host_os)
  install(FILES $<TARGET_PROPERTY:x10,IMPORTED_LOCATION>
    DESTINATION lib/swift/${host_os})
endif()
if(ENABLE_PYTHON_SUPPORT)
  get_swift_host_os(swift_os)
  get_swift_host_arch(swift_arch)

  install(FILES $<TARGET_PROPERTY:PythonKit,IMPORTED_LOCATION>
    DESTINATION lib/swift/${swift_os})
  if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    install(FILES $<TARGET_PROPERTY:PythonKit,IMPORTED_IMPLIB>
      DESTINATION lib/swift/${swift_os})
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    install(FILES ${CMAKE_Swift_MODULE_DIRECTORY}/PythonKit.swiftdoc
      DESTINATION lib/swift/${swift_os}/PythonKit.swiftmodule
      RENAME ${swift_arch}.swiftdoc)
    install(FILES ${CMAKE_Swift_MODULE_DIRECTORY}/PythonKit.swiftmodule
      DESTINATION lib/swift/${swift_os}/PythonKit.swiftmodule
      RENAME ${swift_arch}.swiftmodule)
  else()
    install(FILES
      ${CMAKE_Swift_MODULE_DIRECTORY}/PythonKit.swiftdoc
      ${CMAKE_Swift_MODULE_DIRECTORY}/PythonKit.swiftmodule
      DESTINATION lib/swift/${swift_os}/${swift_arch})
  endif()
endif()
