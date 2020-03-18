add_library(ExperimentalTests
  ComplexTests.swift
  XCTestManifests.swift)
target_link_libraries(ExperimentalTests PUBLIC
  _Experimental
  XCTest)
