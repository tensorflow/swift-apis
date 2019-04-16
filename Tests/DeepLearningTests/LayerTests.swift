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
@testable import DeepLearning

final class LayerTests: XCTestCase {
    func testConv1D() {
        let filter = Tensor<Float>(ones: [3, 1, 2]) * Tensor<Float>([[[0.33333333, 1]]])
        let bias = Tensor<Float>([0, 1])
        let layer = Conv1D<Float>(filter: filter, bias: bias, activation: identity, stride: 1, padding: .valid)
        let input = Tensor<Float>([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]).expandingShape(at: 2)
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[[1, 4], [2, 7], [3, 10]], [[11, 34], [12, 37], [13, 40]]])
        XCTAssertEqual(round(output), expected)
    }

    func testMaxPool1D() {
        let layer = MaxPool1D<Float>(poolSize: 3, stride: 1, padding: .valid)
        let input = Tensor<Float>([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]).expandingShape(at: 2)
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[[2], [3], [4]], [[12], [13], [14]]])
        XCTAssertEqual(round(output), expected)
    }

    func testAvgPool1D() {
        let layer = AvgPool1D<Float>(poolSize: 3, stride: 1, padding: .valid)
        let input = Tensor<Float>([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]).expandingShape(at: 2)
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[[1], [2], [3]], [[11], [12], [13]]])
        XCTAssertEqual(round(output), expected)
    }

    func testGlobalAvgPool1D() {
        let layer = GlobalAveragePooling1D<Float>()
        let input = Tensor<Float>([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]).expandingShape(at: 2)
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[2.0], [12.0]])
        XCTAssertEqual(round(output), expected)
    }

    func testGlobalAvgPool2D() {
        let layer = GlobalAveragePooling2D<Float>()
        let input = Tensor<Float>([[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15], [15, 16], [17, 18], [19, 20], [21, 22]]]).expandingShape(at: 3)
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[5.5], [16.833334]])
        XCTAssertEqual(round(output), round(expected))
    }

    func testGlobalAvgPool3D() {
        let layer = GlobalAveragePooling3D<Float>()
        let input = Tensor<Float>([[[[0.5], [1.0]], [[1.5], [2.0]], [[2.5], [3.0]], [[3.5], [4.0]], [[4.5], [5.0]], [[5.5], [6.0]]], [[[6.5], [7.0]], [[7.5], [8.0]], [[8.5], [9.0]], [[9.5], [10.0]], [[10.5], [11.0]], [[11.5], [12.0]]]]).expandingShape(at: 4)
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[3.25], [9.25]])
        XCTAssertEqual(round(output), round(expected))
    }

    static var allTests = [
        ("testConv1D", testConv1D),
        ("testMaxPool1D", testMaxPool1D),
        ("testAvgPool1D", testAvgPool1D),
        ("testGlobalAvgPool1D", testGlobalAvgPool1D),
        ("testGlobalAvgPool2D", testGlobalAvgPool2D),
        ("testGlobalAvgPool3D", testGlobalAvgPool3D)
    ]
}
