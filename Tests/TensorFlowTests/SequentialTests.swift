// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 1)
// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 16)

import TensorFlow
import XCTest

final class SequentialTests: XCTestCase {
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 22)
  func testSequential2() {
    let input = Float(1)
    let model = Sequential {
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 1)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 2)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 28)
    }

    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(2)), output)
    XCTAssertEqual(Float(factorial(2)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel1 = gModel.layer1
    XCTAssertEqual(Float(factorial(2) / 1), gModel1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel2 = gModel.layer2
    XCTAssertEqual(Float(factorial(2) / 2), gModel2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 40)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 22)
  func testSequential3() {
    let input = Float(1)
    let model = Sequential {
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 1)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 2)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 3)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 28)
    }

    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(3)), output)
    XCTAssertEqual(Float(factorial(3)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel1 = gModel.layer1
    XCTAssertEqual(Float(factorial(3) / 1), gModel1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel2 = gModel.layer2.layer1
    XCTAssertEqual(Float(factorial(3) / 2), gModel2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel3 = gModel.layer2.layer2
    XCTAssertEqual(Float(factorial(3) / 3), gModel3.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 40)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 22)
  func testSequential4() {
    let input = Float(1)
    let model = Sequential {
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 1)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 2)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 3)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 4)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 28)
    }

    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(4)), output)
    XCTAssertEqual(Float(factorial(4)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel1 = gModel.layer1
    XCTAssertEqual(Float(factorial(4) / 1), gModel1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel2 = gModel.layer2.layer1
    XCTAssertEqual(Float(factorial(4) / 2), gModel2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel3 = gModel.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(4) / 3), gModel3.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel4 = gModel.layer2.layer2.layer2
    XCTAssertEqual(Float(factorial(4) / 4), gModel4.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 40)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 22)
  func testSequential5() {
    let input = Float(1)
    let model = Sequential {
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 1)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 2)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 3)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 4)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 5)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 28)
    }

    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(5)), output)
    XCTAssertEqual(Float(factorial(5)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel1 = gModel.layer1
    XCTAssertEqual(Float(factorial(5) / 1), gModel1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel2 = gModel.layer2.layer1
    XCTAssertEqual(Float(factorial(5) / 2), gModel2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel3 = gModel.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(5) / 3), gModel3.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel4 = gModel.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(5) / 4), gModel4.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel5 = gModel.layer2.layer2.layer2.layer2
    XCTAssertEqual(Float(factorial(5) / 5), gModel5.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 40)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 22)
  func testSequential6() {
    let input = Float(1)
    let model = Sequential {
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 1)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 2)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 3)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 4)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 5)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 6)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 28)
    }

    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(6)), output)
    XCTAssertEqual(Float(factorial(6)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel1 = gModel.layer1
    XCTAssertEqual(Float(factorial(6) / 1), gModel1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel2 = gModel.layer2.layer1
    XCTAssertEqual(Float(factorial(6) / 2), gModel2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel3 = gModel.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(6) / 3), gModel3.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel4 = gModel.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(6) / 4), gModel4.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel5 = gModel.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(6) / 5), gModel5.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel6 = gModel.layer2.layer2.layer2.layer2.layer2
    XCTAssertEqual(Float(factorial(6) / 6), gModel6.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 40)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 22)
  func testSequential7() {
    let input = Float(1)
    let model = Sequential {
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 1)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 2)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 3)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 4)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 5)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 6)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 7)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 28)
    }

    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(7)), output)
    XCTAssertEqual(Float(factorial(7)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel1 = gModel.layer1
    XCTAssertEqual(Float(factorial(7) / 1), gModel1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel2 = gModel.layer2.layer1
    XCTAssertEqual(Float(factorial(7) / 2), gModel2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel3 = gModel.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(7) / 3), gModel3.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel4 = gModel.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(7) / 4), gModel4.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel5 = gModel.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(7) / 5), gModel5.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel6 = gModel.layer2.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(7) / 6), gModel6.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel7 = gModel.layer2.layer2.layer2.layer2.layer2.layer2
    XCTAssertEqual(Float(factorial(7) / 7), gModel7.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 40)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 22)
  func testSequential8() {
    let input = Float(1)
    let model = Sequential {
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 1)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 2)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 3)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 4)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 5)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 6)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 7)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 8)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 28)
    }

    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(8)), output)
    XCTAssertEqual(Float(factorial(8)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel1 = gModel.layer1
    XCTAssertEqual(Float(factorial(8) / 1), gModel1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel2 = gModel.layer2.layer1
    XCTAssertEqual(Float(factorial(8) / 2), gModel2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel3 = gModel.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(8) / 3), gModel3.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel4 = gModel.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(8) / 4), gModel4.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel5 = gModel.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(8) / 5), gModel5.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel6 = gModel.layer2.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(8) / 6), gModel6.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel7 = gModel.layer2.layer2.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(8) / 7), gModel7.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel8 = gModel.layer2.layer2.layer2.layer2.layer2.layer2.layer2
    XCTAssertEqual(Float(factorial(8) / 8), gModel8.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 40)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 22)
  func testSequential9() {
    let input = Float(1)
    let model = Sequential {
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 1)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 2)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 3)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 4)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 5)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 6)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 7)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 8)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 9)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 28)
    }

    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(9)), output)
    XCTAssertEqual(Float(factorial(9)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel1 = gModel.layer1
    XCTAssertEqual(Float(factorial(9) / 1), gModel1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel2 = gModel.layer2.layer1
    XCTAssertEqual(Float(factorial(9) / 2), gModel2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel3 = gModel.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(9) / 3), gModel3.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel4 = gModel.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(9) / 4), gModel4.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel5 = gModel.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(9) / 5), gModel5.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel6 = gModel.layer2.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(9) / 6), gModel6.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel7 = gModel.layer2.layer2.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(9) / 7), gModel7.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel8 = gModel.layer2.layer2.layer2.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(9) / 8), gModel8.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel9 = gModel.layer2.layer2.layer2.layer2.layer2.layer2.layer2.layer2
    XCTAssertEqual(Float(factorial(9) / 9), gModel9.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 40)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 22)
  func testSequential10() {
    let input = Float(1)
    let model = Sequential {
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 1)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 2)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 3)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 4)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 5)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 6)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 7)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 8)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 9)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 26)
      Multiply(coefficient: 10)
      // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 28)
    }

    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(10)), output)
    XCTAssertEqual(Float(factorial(10)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel1 = gModel.layer1
    XCTAssertEqual(Float(factorial(10) / 1), gModel1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel2 = gModel.layer2.layer1
    XCTAssertEqual(Float(factorial(10) / 2), gModel2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel3 = gModel.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(10) / 3), gModel3.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel4 = gModel.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(10) / 4), gModel4.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel5 = gModel.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(10) / 5), gModel5.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel6 = gModel.layer2.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(10) / 6), gModel6.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel7 = gModel.layer2.layer2.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(10) / 7), gModel7.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel8 = gModel.layer2.layer2.layer2.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(10) / 8), gModel8.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel9 = gModel.layer2.layer2.layer2.layer2.layer2.layer2.layer2.layer2.layer1
    XCTAssertEqual(Float(factorial(10) / 9), gModel9.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 37)
    let gModel10 = gModel.layer2.layer2.layer2.layer2.layer2.layer2.layer2.layer2.layer2
    XCTAssertEqual(Float(factorial(10) / 10), gModel10.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 40)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 42)

  static var allTests = [
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 45)
    ("testSequential2", testSequential2),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 45)
    ("testSequential3", testSequential3),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 45)
    ("testSequential4", testSequential4),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 45)
    ("testSequential5", testSequential5),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 45)
    ("testSequential6", testSequential6),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 45)
    ("testSequential7", testSequential7),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 45)
    ("testSequential8", testSequential8),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 45)
    ("testSequential9", testSequential9),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 45)
    ("testSequential10", testSequential10),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequentialTests.swift.gyb", line: 47)
  ]
}
