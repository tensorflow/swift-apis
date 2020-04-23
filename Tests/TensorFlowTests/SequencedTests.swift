// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 1)
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

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 16)

import TensorFlow
import XCTest

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 21)
struct Model2: Layer {
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply1: Multiply = Multiply(coefficient: 1)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply2: Multiply = Multiply(coefficient: 2)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 25)

  @differentiable
  func callAsFunction(_ input: Float) -> Float {
    input.sequenced(
      through: multiply1, multiply2
    )
  }
}
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 21)
struct Model3: Layer {
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply1: Multiply = Multiply(coefficient: 1)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply2: Multiply = Multiply(coefficient: 2)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply3: Multiply = Multiply(coefficient: 3)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 25)

  @differentiable
  func callAsFunction(_ input: Float) -> Float {
    input.sequenced(
      through: multiply1, multiply2, multiply3
    )
  }
}
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 21)
struct Model4: Layer {
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply1: Multiply = Multiply(coefficient: 1)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply2: Multiply = Multiply(coefficient: 2)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply3: Multiply = Multiply(coefficient: 3)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply4: Multiply = Multiply(coefficient: 4)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 25)

  @differentiable
  func callAsFunction(_ input: Float) -> Float {
    input.sequenced(
      through: multiply1, multiply2, multiply3, multiply4
    )
  }
}
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 21)
struct Model5: Layer {
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply1: Multiply = Multiply(coefficient: 1)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply2: Multiply = Multiply(coefficient: 2)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply3: Multiply = Multiply(coefficient: 3)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply4: Multiply = Multiply(coefficient: 4)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply5: Multiply = Multiply(coefficient: 5)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 25)

  @differentiable
  func callAsFunction(_ input: Float) -> Float {
    input.sequenced(
      through: multiply1, multiply2, multiply3, multiply4, multiply5
    )
  }
}
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 21)
struct Model6: Layer {
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply1: Multiply = Multiply(coefficient: 1)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply2: Multiply = Multiply(coefficient: 2)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply3: Multiply = Multiply(coefficient: 3)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply4: Multiply = Multiply(coefficient: 4)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply5: Multiply = Multiply(coefficient: 5)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 23)
  var multiply6: Multiply = Multiply(coefficient: 6)
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 25)

  @differentiable
  func callAsFunction(_ input: Float) -> Float {
    input.sequenced(
      through: multiply1, multiply2, multiply3, multiply4, multiply5, multiply6
    )
  }
}
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 34)

final class SequencedTests: XCTestCase {
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 37)
  func testSequenced2() {
    let input = Float(1)
    let model = Model2()
    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(2)), output)
    XCTAssertEqual(Float(factorial(2)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(2) / 1), gModel.multiply1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(2) / 2), gModel.multiply2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 49)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 37)
  func testSequenced3() {
    let input = Float(1)
    let model = Model3()
    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(3)), output)
    XCTAssertEqual(Float(factorial(3)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(3) / 1), gModel.multiply1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(3) / 2), gModel.multiply2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(3) / 3), gModel.multiply3.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 49)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 37)
  func testSequenced4() {
    let input = Float(1)
    let model = Model4()
    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(4)), output)
    XCTAssertEqual(Float(factorial(4)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(4) / 1), gModel.multiply1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(4) / 2), gModel.multiply2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(4) / 3), gModel.multiply3.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(4) / 4), gModel.multiply4.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 49)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 37)
  func testSequenced5() {
    let input = Float(1)
    let model = Model5()
    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(5)), output)
    XCTAssertEqual(Float(factorial(5)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(5) / 1), gModel.multiply1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(5) / 2), gModel.multiply2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(5) / 3), gModel.multiply3.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(5) / 4), gModel.multiply4.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(5) / 5), gModel.multiply5.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 49)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 37)
  func testSequenced6() {
    let input = Float(1)
    let model = Model6()
    let (output, (gInput, gModel)) = valueWithGradient(at: input, model) {
      $1($0)
    }

    XCTAssertEqual(Float(factorial(6)), output)
    XCTAssertEqual(Float(factorial(6)), gInput)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(6) / 1), gModel.multiply1.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(6) / 2), gModel.multiply2.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(6) / 3), gModel.multiply3.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(6) / 4), gModel.multiply4.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(6) / 5), gModel.multiply5.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 47)
    XCTAssertEqual(Float(factorial(6) / 6), gModel.multiply6.coefficient)
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 49)
  }
  // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 51)

  static var allTests = [
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 54)
    ("testSequenced2", testSequenced2),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 54)
    ("testSequenced3", testSequenced3),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 54)
    ("testSequenced4", testSequenced4),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 54)
    ("testSequenced5", testSequenced5),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 54)
    ("testSequenced6", testSequenced6),
    // ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/swift-apis/Tests/TensorFlowTests/SequencedTests.swift.gyb", line: 56)
  ]
}
