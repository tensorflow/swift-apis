cmake_minimum_required(VERSION 3.15.1)
project(TensorFlow
  LANGUAGES CXX Swift)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)

find_package(TensorFlow REQUIRED)

add_compile_options($<$<COMPILE_LANGUAGE:Swift>:-Xllvm$<SEMICOLON>-sil-inline-generics>)
add_compile_options($<$<COMPILE_LANGUAGE:Swift>:-Xllvm$<SEMICOLON>-sil-partial-specialization>)

add_subdirectory(Sources)
