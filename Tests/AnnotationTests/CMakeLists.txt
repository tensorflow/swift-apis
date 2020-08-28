add_executable(TFEagerTests
  TFEagerTests.swift)
target_link_libraries(TFEagerTests PRIVATE
  x10
  TensorFlow)
add_test(NAME Annotations.TFEager
  COMMAND TFEagerTests)

add_executable(XLATests
  XLATests.swift)
target_link_libraries(XLATests PRIVATE
  x10
  TensorFlow)
add_test(NAME Annotations.XLA
  COMMAND XLATests)

