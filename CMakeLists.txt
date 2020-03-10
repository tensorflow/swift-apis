cmake_minimum_required(VERSION 3.15.1)
project(TensorFlow
  LANGUAGES CXX Swift)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)

option(USE_BUNDLED_CTENSORFLOW
  "Use the CTensorFlow module bundled in the active Swift toolchain" OFF)

find_package(TensorFlow REQUIRED)

include(SwiftSupport)

add_compile_options("$<$<COMPILE_LANGUAGE:Swift>:SHELL:-Xllvm -sil-inline-generics>")
add_compile_options("$<$<COMPILE_LANGUAGE:Swift>:SHELL:-Xllvm -sil-partial-specialization>")

add_subdirectory(Sources)
