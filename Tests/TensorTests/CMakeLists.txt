add_library(TensorTests
  PRNGTests.swift
  UtilitiesTests.swift
  XCTestManifests.swift)
target_link_libraries(TensorTests PUBLIC
  Tensor
  XCTest)
