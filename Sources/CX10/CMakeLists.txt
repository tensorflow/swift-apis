# TODO(abdulras) switch to `find_package(X10)`
add_library(x10 IMPORTED UNKNOWN)
set_target_properties(x10 PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${X10_INCLUDE_DIR}"
  IMPORTED_LOCATION ${TensorFlow_LIBRARY})

add_library(CX10 STATIC
  device_wrapper.cc
  xla_tensor_tf_ops.cc
  xla_tensor_wrapper.cc)
set_target_properties(CX10 PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(CX10 PUBLIC
  include)
target_link_libraries(CX10 PUBLIC
  x10)

