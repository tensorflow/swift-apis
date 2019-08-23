import XCTest

import TensorFlowTests
import ExperimentalTests

var tests = [XCTestCaseEntry]()
tests += TensorFlowTests.allTests()
tests += ExperimentalTests.allTests()
XCTMain(tests)
