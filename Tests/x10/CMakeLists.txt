add_executable(ops_test
  ops_test.swift)
target_link_libraries(ops_test PRIVATE
  x10
  x10_tensor
  TensorFlow)
add_test(NAME ops_test
  COMMAND ops_test)

add_executable(xla_tensor_test
  xla_tensor_test.swift)
target_link_libraries(xla_tensor_test PRIVATE
  x10
  x10_tensor)
add_test(NAME xla_tensor_test
  COMMAND xla_tensor_test)

add_executable(keypathiterable_test
  keypathiterable_test.swift)
target_link_libraries(keypathiterable_test PRIVATE
  x10
  x10_tensor)
add_test(NAME keypathiterable_test
  COMMAND keypathiterable_test)

add_executable(tensor_visitor_plan_test
  TensorVisitorPlanTest.swift)
target_link_libraries(tensor_visitor_plan_test PRIVATE
  x10
  x10_optimizers_tensor_visitor_plan)
add_test(NAME tensor_visitor_plan_test
  COMMAND tensor_visitor_plan_test)

