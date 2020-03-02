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
import TensorFlow

class NNTests: XCTestCase {
    func testDepthToSpace() {
        let input = Tensor<Float>([[
            [[0, 1, 2, 3], [4, 5, 6, 7]]
        ]])
        let expected = Tensor<Float>([[
            [[0], [1], [4], [5]],
            [[2], [3], [6], [7]]
        ]])
        XCTAssertEqual(depthToSpace(input, size: 2), expected)
    }
    
    func testDepthToSpaceGrad() {
        let input = Tensor<Float>([[
            [[0, 1, 2, 3], [4, 5, 6, 7]]
        ]])
        let grad = Tensor<Float>([[
            [[0], [1], [4], [5]],
            [[2], [3], [6], [7]]
        ]])
        let depthToSpacePullback = pullback(at: input) { depthToSpace($0, size: 2) }
        XCTAssertEqual(depthToSpacePullback(grad), input)
    }
    
    func testSpaceToDepth() {
        let input = Tensor<Float>([[
            [[0], [1], [4], [5]],
            [[2], [3], [6], [7]]
        ]])
        let expected = Tensor<Float>([[
            [[0, 1, 2, 3], [4, 5, 6, 7]]
        ]])
        XCTAssertEqual(spaceToDepth(input, size: 2), expected)
    }
    
    func testSpaceToDepthGrad() {
        let input = Tensor<Float>([[
            [[0], [1], [4], [5]],
            [[2], [3], [6], [7]]
        ]])
        let grad = Tensor<Float>([[
            [[0, 1, 2, 3], [4, 5, 6, 7]]
        ]])
        let spaceToDepthPullback = pullback(at: input) { spaceToDepth($0, size: 2) }
        XCTAssertEqual(spaceToDepthPullback(grad), input)
    }
    
    static let allTests = [
        ("testDepthToSpace", testDepthToSpace),
        ("testDepthToSpaceGrad", testDepthToSpaceGrad),
        ("testSpaceToDepth", testSpaceToDepth),
        ("testSpaceToDepthGrad", testSpaceToDepthGrad),
    ]
}