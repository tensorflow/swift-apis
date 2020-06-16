include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(x10_optimizers_tensor_visitor_plan SHARED
  swift_bindings/optimizers/TensorVisitorPlan.swift)
set_target_properties(x10_optimizers_tensor_visitor_plan PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(x10_optimizers_tensor_visitor_plan PUBLIC
  TensorFlow)

add_library(x10_optimizers_optimizer SHARED
  swift_bindings/optimizers/Optimizer.swift
  swift_bindings/optimizers/Optimizers.swift)
set_target_properties(x10_optimizers_optimizer PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(x10_optimizers_optimizer PUBLIC
  x10_optimizers_tensor_visitor_plan
  TensorFlow)

add_library(x10_training_loop SHARED
  swift_bindings/training_loop.swift)
set_target_properties(x10_training_loop PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(x10_training_loop PUBLIC
  TensorFlow)
target_link_libraries(x10_training_loop PRIVATE
  CX10)


_install_target(x10_optimizers_tensor_visitor_plan)
_install_target(x10_optimizers_optimizer)
_install_target(x10_training_loop)
