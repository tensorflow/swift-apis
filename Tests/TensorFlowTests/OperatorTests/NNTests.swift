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

import TensorFlow
import XCTest

class NNTests: XCTestCase {
  func testDepthToSpace() {
    let input = Tensor<Float>([
      [
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
      ]
    ])
    let expected = Tensor<Float>([
      [
        [[0, 1], [2, 3], [8, 9], [10, 11]],
        [[4, 5], [6, 7], [12, 13], [14, 15]],
      ]
    ])
    XCTAssertEqual(depthToSpace(input, blockSize: 2), expected)

    let emptyInput = Tensor<Float>(zeros: [0, 1, 2, 8])
    XCTAssertEqual(depthToSpace(emptyInput, blockSize: 2), Tensor(zeros: [0, 2, 4, 2]))
  }

  func testDepthToSpaceGrad() {
    let input = Tensor<Float>([
      [
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
      ]
    ])
    let grad = Tensor<Float>([
      [
        [[0, 1], [2, 3], [8, 9], [10, 11]],
        [[4, 5], [6, 7], [12, 13], [14, 15]],
      ]
    ])
    let depthToSpacePullback = pullback(at: input) { depthToSpace($0, blockSize: 2) }
    XCTAssertEqual(depthToSpacePullback(grad), input)

    let emptyGrad = Tensor<Float>(zeros: [0, 2, 4, 2])
    XCTAssertEqual(depthToSpacePullback(emptyGrad), Tensor(zeros: [0, 1, 2, 8]))
  }

  func testSpaceToDepth() {
    let input = Tensor<Float>([
      [
        [[0, 1], [2, 3], [8, 9], [10, 11]],
        [[4, 5], [6, 7], [12, 13], [14, 15]],
      ]
    ])
    let expected = Tensor<Float>([
      [
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
      ]
    ])
    XCTAssertEqual(spaceToDepth(input, blockSize: 2), expected)

    let emptyInput = Tensor<Float>(zeros: [0, 2, 4, 2])
    XCTAssertEqual(spaceToDepth(emptyInput, blockSize: 2), Tensor(zeros: [0, 1, 2, 8]))
  }

  func testSpaceToDepthGrad() {
    let input = Tensor<Float>([
      [
        [[0, 1], [2, 3], [8, 9], [10, 11]],
        [[4, 5], [6, 7], [12, 13], [14, 15]],
      ]
    ])
    let grad = Tensor<Float>([
      [
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
      ]
    ])
    let spaceToDepthPullback = pullback(at: input) { spaceToDepth($0, blockSize: 2) }
    XCTAssertEqual(spaceToDepthPullback(grad), input)

    let emptyGrad = Tensor<Float>(zeros: [0, 1, 2, 8])
    XCTAssertEqual(spaceToDepthPullback(emptyGrad), Tensor(zeros: [0, 2, 4, 2]))
  }

  static let allTests = [
    ("testDepthToSpace", testDepthToSpace),
    ("testDepthToSpaceGrad", testDepthToSpaceGrad),
    ("testSpaceToDepth", testSpaceToDepth),
    ("testSpaceToDepthGrad", testSpaceToDepthGrad),
  ]
}
