// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import XCTest

@testable import TensorFlow

final class TensorTests: XCTestCase {
  func testSimpleCond() {
    func selectValue(_ pred: Bool) -> Tensor<Int32> {
      let a = Tensor<Int32>(0)
      let b = Tensor<Int32>(1)
      if pred {
        return a
      }
      return b
    }
    XCTAssertEqual(selectValue(true).scalar, 0)
  }

  func testRankGetter() {
    let vector = Tensor<Int32>([1])
    let matrix = Tensor<Float>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    let ones = Tensor<Int32>(ones: [1, 2, 2, 2, 2, 2, 1])
    let tensor = Tensor<Int32>(shape: [3, 4, 5], scalars: Array(0..<60))
    XCTAssertEqual(vector.rank, 1)
    XCTAssertEqual(matrix.rank, 2)
    XCTAssertEqual(ones.rank, 7)
    XCTAssertEqual(tensor.rank, 3)
  }

  func testShapeGetter() {
    let vector = Tensor<Int32>([1])
    let matrix = Tensor<Float>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    let ones = Tensor<Int32>(ones: [1, 2, 2, 2, 2, 2, 1])
    let tensor = Tensor<Int32>(shape: [3, 4, 5], scalars: Array(0..<60))
    XCTAssertEqual(vector.shape, [1])
    XCTAssertEqual(matrix.shape, [2, 3])
    XCTAssertEqual(ones.shape, [1, 2, 2, 2, 2, 2, 1])
    XCTAssertEqual(tensor.shape, [3, 4, 5])
  }

  func testTensorShapeDescription() {
    XCTAssertEqual(Tensor<Int32>(ones: [2, 2]).shape.description, "[2, 2]")
    XCTAssertEqual(Tensor(1).shape.description, "[]")
  }

  func testEquality() {
    let tensor = Tensor<Float>([0, 1, 2, 3, 4, 5])
    let zeros = Tensor<Float>(zeros: [6])

    XCTAssertTrue(tensor == tensor)
    XCTAssertFalse(tensor != tensor)
    XCTAssertFalse(tensor == zeros)
    XCTAssertTrue(tensor != zeros)
    XCTAssertFalse(tensor == tensor.reshaped(to: [2, 3]))
    XCTAssertTrue(tensor != tensor.reshaped(to: [2, 3]))
  }

  func testTensorShapeCollectionOperations() {
    let dims1 = [Int](1...3)
    let dims2 = [Int](1...2)
    let dims3: [Int] = [4, 5]
    let shape1 = TensorShape(dims1)
    let shape2 = TensorShape(dims2)
    XCTAssertEqual((shape1 + shape2), TensorShape(dims1 + dims2))
    XCTAssertTrue((shape1 + shape2).count == shape1.count + shape2.count)
    XCTAssertTrue((shape1 + dims3).count == shape1.count + dims3.count)

    var shape3: TensorShape = shape2
    let firstValue: Int! = shape3.popFirst()
    XCTAssertTrue(firstValue == shape2[0])
    XCTAssertTrue(shape3 == shape2[1..<shape2.count])

    shape3.insert(firstValue, at: 0)
    XCTAssertTrue(shape3 == shape2)

    shape3.append(contentsOf: shape2)
    XCTAssertTrue(shape3 == (shape2 + shape2))
  }

  func testInitShapeScalars() {
    XCTAssertEqual(
      Tensor<Float>(shape: [2, 2], scalars: [1, 2, 3, 4]),
      Tensor<Float>([[1, 2], [3, 4]])
    )
  }

  func testInitShapeScalarsDerivative() {
    let (value, pullback) = valueWithPullback(at: [1, 2, 3, 4]) {
      Tensor<Float>(shape: [2, 2], scalars: $0)
    }
    XCTAssertEqual(value, Tensor<Float>([[1, 2], [3, 4]]))
    XCTAssertEqual(
      pullback(Tensor([[1, 0], [0, 0]])),
      Array.DifferentiableView([1, 0, 0, 0])
    )
    XCTAssertEqual(
      pullback(Tensor([[0, 1], [0, 0]])),
      Array.DifferentiableView([0, 1, 0, 0])
    )
    XCTAssertEqual(
      pullback(Tensor([[0, 0], [1, 0]])),
      Array.DifferentiableView([0, 0, 1, 0])
    )
    XCTAssertEqual(
      pullback(Tensor([[0, 0], [0, 1]])),
      Array.DifferentiableView([0, 0, 0, 1])
    )
  }

  static var allTests = [
    ("testSimpleCond", testSimpleCond),
    ("testRankGetter", testRankGetter),
    ("testShapeGetter", testShapeGetter),
    ("testTensorShapeDescription", testTensorShapeDescription),
    ("testEquality", testEquality),
    ("testTensorShapeCollectionOperations", testTensorShapeCollectionOperations),
    ("testInitShapeScalars", testInitShapeScalars),
    ("testInitShapeScalarsDerivative", testInitShapeScalarsDerivative),
  ]
}
