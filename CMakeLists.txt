
cmake_minimum_required(VERSION 3.15.1)
project(TensorFlow
  LANGUAGES CXX Swift)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)

find_package(TensorFlow REQUIRED)

add_subdirectory(Sources)
