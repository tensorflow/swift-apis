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
  find_package(X10 REQUIRED)
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
