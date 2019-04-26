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
        let layer = GlobalAvgPool1D<Float>()
        let input = Tensor(shape: [2, 5, 1], scalars: (0..<10).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[2], [7]])
        XCTAssertEqual(output, expected)
    }

    func testGlobalAvgPool2D() {
        let layer = GlobalAvgPool2D<Float>()
        let input = Tensor(shape: [2, 6, 2, 1], scalars: (0..<24).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[5.5], [17.5]])
        XCTAssertEqual(output, expected)
    }

    func testGlobalAvgPool3D() {
        let layer = GlobalAvgPool3D<Float>()
        let input = Tensor<Float>(shape: [2, 6, 2, 1, 1], scalars: (0..<24).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[5.5], [17.5]])
        XCTAssertEqual(output, expected)
    }

    func testUpSampling1D() {
      let size = 6
      let layer = UpSampling1D<Float>(size: size)
      let input = Tensor<Float>(shape: [1, 10, 1], scalars: (0..<10).map(Float.init))
      let output = layer.inferring(from: input)
      let expected = TensorShape([1, input.shape[1] * size, 1])
      XCTAssertEqual(output.shape, expected)
    }

    func testUpSampling2D() {
      let size = 6
      let layer = UpSampling2D<Float>(size: size)
      let input = Tensor<Float>(shape: [1, 3, 5, 1], scalars: (0..<15).map(Float.init))
      let output = layer.inferring(from: input)
      let expected = TensorShape([1, input.shape[1] * size, input.shape[2] * size, 1])
      XCTAssertEqual(output.shape, expected)
    }

    func testUpSampling3D() {
      let size = 6
      let layer = UpSampling3D<Float>(size: size)
      let input = Tensor<Float>(shape: [1, 4, 3, 2, 1], scalars: (0..<24).map(Float.init))
      let output = layer.inferring(from: input)
      let expected = TensorShape([1, input.shape[1] * size, input.shape[2] * size, input.shape[3] * size, 1])
      XCTAssertEqual(output.shape, expected)
    }

    func testReshape() {
        let layer = Reshape<Float>(shape: [10, 2, 1])
        let input = Tensor(shape: [20, 1], scalars: (0..<20).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = TensorShape([10, 2, 1])
        XCTAssertEqual(output.shape, expected)
    }

    func testFlatten() {
        let layer = Flatten<Float>()
        let input = Tensor(shape: [10, 2, 2], scalars: (0..<40).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = TensorShape([10, 4])
        XCTAssertEqual(output.shape, expected)
    }

    func testSimpleRNNCell() {
        let weight = Tensor<Float>(ones: [7, 5]) * Tensor<Float>([0.3333, 1, 0.3333, 1, 0.3333])
        let bias = Tensor<Float>(ones: [5])
        var cell = SimpleRNNCell<Float>(inputSize: 2, hiddenSize: 5)
        cell.W_h = weigth
        cell.b_h = bias
        let state = Tensor<Float>(ones: [1, 5]) * Tensor<Float>([1, 0.2, 0.5, 2, 0.6])
        let input = Tensor<Float>(ones: [1, 2]) * Tensor<Float>([0.3, 0.7])
        let output = cell(input: input, state: state).state
        let expected = Tensor<Float>([[2.76649, 6.2999997, 2.76649, 6.2999997, 2.76649]])
        XCTAssertEqual(output, expected)
    }

    func testRNN() {
        let x = Tensor<Float>(rangeFrom: 0.0, to: 0.4, stride: 0.1).rankLifted()
        let inputs: [Tensor<Float>] = Array(repeating: x, count: 4)
        let rnn = RNN(SimpleRNNCell<Float>(inputSize: 4, hiddenSize: 4,
                                           seed: (0xFeedBeef, 0xDeadBeef)))
        let (outputs, pullback) = rnn.valueWithPullback(at: inputs) { rnn, inputs in
            return rnn(inputs)
        }
        XCTAssertEqual(outputs, [[[-0.0026294366, -0.0058668107,  0.04495003,  0.20311214]],
                                 [[ 0.06788494,    0.050665878,   0.02415526,  0.09249911]],
                                 [[ 0.06621192,    0.009049267,   0.065047316, 0.11534518]],
                                 [[ 0.05612204,    0.00022032857, 0.05407162,  0.09784105]]])
        let (𝛁rnn, 𝛁inputs) = pullback(.init(inputs))
        XCTAssertEqual(𝛁rnn.cell.W_h,
                       [[          0.0,           0.0,           0.0,           0.0],
                        [-0.0051278225,  0.0013102926,    0.00740262,   0.018119661],
                        [ -0.010255645,  0.0026205853,    0.01480524,   0.036239322],
                        [ -0.015383467,   0.003930878,    0.02220786,   0.054358985],
                        [          0.0,           0.0,           0.0,           0.0],
                        [          0.0,           0.0,           0.0,           0.0],
                        [          0.0,           0.0,           0.0,           0.0],
                        [          0.0,           0.0,           0.0,           0.0]])
        XCTAssertEqual(𝛁rnn.cell.b_h, [-0.051278222,  0.013102926,    0.0740262,   0.18119662])
    }

    static var allTests = [
        ("testConv1D", testConv1D),
        ("testMaxPool1D", testMaxPool1D),
        ("testAvgPool1D", testAvgPool1D),
        ("testGlobalAvgPool1D", testGlobalAvgPool1D),
        ("testGlobalAvgPool2D", testGlobalAvgPool2D),
        ("testGlobalAvgPool3D", testGlobalAvgPool3D),
        ("testUpSampling1D", testUpSampling1D),
        ("testUpSampling2D", testUpSampling2D),
        ("testUpSampling3D", testUpSampling3D),
        ("testReshape", testReshape),
        ("testFlatten", testFlatten),
        ("testSimpleRNNCell", testSimpleRNNCell),
        ("testRNN", testRNN)
    ]
}
