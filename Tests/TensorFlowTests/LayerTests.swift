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

final class LayerTests: XCTestCase {
    func testConv1D() {
        let filter = Tensor<Float>(ones: [3, 1, 2]) * Tensor<Float>([[[0.5, 1]]])
        let bias = Tensor<Float>([0, 1])
        let layer = Conv1D<Float>(filter: filter, bias: bias, activation: identity, stride: 1, padding: .valid)
        let input = Tensor<Float>([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]).expandingShape(at: 2)
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>(
          shape: [2, 3, 2],
          scalars: [1.5, 4, 3, 7, 4.5, 10, 16.5, 34, 18, 37, 19.5, 40])
        XCTAssertEqual(output, expected)
    }

    func testConv2D() {
        let filter =  Tensor(shape: [1, 2, 2, 1], scalars: (0..<4).map(Float.init))
        let bias = Tensor<Float>([1, 2])
        let layer = Conv2D<Float>(filter: filter, bias: bias, activation: identity,
                                  strides: (2, 2), padding: .valid)
        let input = Tensor(shape: [2, 2, 2, 2], scalars: (0..<16).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>(shape: [2, 1, 1, 2],
                                     scalars: [15, 16, 63, 64])
        XCTAssertEqual(output, expected)
    }

    func testConv3D() {
        let filter =  Tensor(shape: [1, 2, 2, 2, 1], scalars: (0..<8).map(Float.init))
        let bias = Tensor<Float>([-1, 1])
        let layer = Conv3D<Float>(filter: filter, bias: bias, activation: identity,
                                  strides: (1, 2, 1), padding: .valid)
        let input = Tensor(shape: [2, 2, 2, 2, 2], scalars: (0..<32).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>(shape: [2, 2, 1, 1, 2],
                                     scalars: [139, 141, 363, 365, 587, 589, 811, 813])
        XCTAssertEqual(output, expected)
    }

    func testMaxPool1D() {
        let layer = MaxPool1D<Float>(poolSize: 3, stride: 1, padding: .valid)
        let input = Tensor<Float>([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]).expandingShape(at: 2)
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[[2], [3], [4]], [[12], [13], [14]]])
        XCTAssertEqual(output, expected)
    }

    func testMaxPool2D() {
        let layer = MaxPool2D<Float>(poolSize: (2, 2), strides: (1, 1), padding: .valid)
        let input = Tensor(shape: [1, 2, 2, 1], scalars: (0..<4).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[[[3]]]])
        XCTAssertEqual(output, expected)
    }

    func testMaxPool3D() {
        let layer = MaxPool3D<Float>(poolSize: (2, 2, 2), strides: (1, 1, 1), padding: .valid)
        let input = Tensor(shape: [1, 2, 2, 2, 1], scalars: (0..<8).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[[[[7]]]]])
        XCTAssertEqual(output, expected)
    }

    func testAvgPool1D() {
        let layer = AvgPool1D<Float>(poolSize: 3, stride: 1, padding: .valid)
        let input = Tensor<Float>([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]).expandingShape(at: 2)
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[[1], [2], [3]], [[11], [12], [13]]])
        XCTAssertEqual(output, expected)
    }

    func testAvgPool2D() {
        let layer = AvgPool2D<Float>(poolSize: (2, 5), strides: (1, 1), padding: .valid)
        let input = Tensor(shape: [1, 2, 5, 1], scalars: (0..<10).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[[[4.5]]]])
        XCTAssertEqual(output, expected)
    }

    func testAvgPool3D() {
        let layer = AvgPool3D<Float>(poolSize: (2, 4, 5), strides: (1, 1, 1), padding: .valid)
        let input = Tensor(shape: [1, 2, 4, 5, 1], scalars: (0..<40).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[[[[19.5]]]]])
        XCTAssertEqual(output, expected)
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
        cell.weight = weight
        cell.bias = bias
        let state = SimpleRNNCell.State(
            Tensor<Float>(ones: [1, 5]) * Tensor<Float>([1, 0.2, 0.5, 2, 0.6])
        )
        let input = Tensor<Float>(ones: [1, 2]) * Tensor<Float>([0.3, 0.7])
        let output = cell(input: input, state: state).state
        let expected = SimpleRNNCell.State(
            Tensor<Float>([[0.9921227, 0.9999934, 0.9921227, 0.9999934, 0.9921227]])
        )
        XCTAssertEqual(output, expected)
    }

    // TODO(TF-507): Remove references to `SimpleRNNCell.State` after SR-10697 is fixed.
    func testRNN() {
        let x = Tensor<Float>(rangeFrom: 0.0, to: 0.4, stride: 0.1).rankLifted()
        let inputs: [Tensor<Float>] = Array(repeating: x, count: 4)
        let rnn = RNN(SimpleRNNCell<Float>(inputSize: 4, hiddenSize: 4,
                                           seed: (0xFeedBeef, 0xDeadBeef)))
        let (outputs, pullback) = rnn.valueWithPullback(at: inputs) { rnn, inputs in
            return rnn(inputs)
        }
        XCTAssertEqual(outputs.map { $0.value },
                       [[[ -0.00262943,  -0.005866742, 0.044919778,  0.20036437]],
                        [[ 0.066890605,   0.049586136, 0.024610005,  0.09341654]],
                        [[ 0.065792546,   0.009325638, 0.06439907,  0.114802904]],
                        [[ 0.055909205, 0.00035158166, 0.054020774,  0.09812111]]])
        let (𝛁rnn, 𝛁inputs) = pullback(.init(inputs.map { SimpleRNNCell<Float>.State($0) }))
        XCTAssertEqual(𝛁rnn.cell.weight,
                       [[          0.0,           0.0,           0.0,           0.0],
                        [-0.0051169936,  0.0014167001,  0.0074189613,   0.017496519],
                        [ -0.010233987,  0.0028334002,  0.0148379225,   0.034993038],
                        [ -0.015350982,  0.0042501003,   0.022256885,    0.05248956],
                        [          0.0,           0.0,           0.0,           0.0],
                        [          0.0,           0.0,           0.0,           0.0],
                        [          0.0,           0.0,           0.0,           0.0],
                        [          0.0,           0.0,           0.0,           0.0]])
        XCTAssertEqual(𝛁rnn.cell.bias, [-0.051169936,  0.014167001,   0.07418961,   0.17496519])
    }

    static var allTests = [
        ("testConv1D", testConv1D),
        ("testConv2D", testConv2D),
        ("testConv3D", testConv3D),
        ("testMaxPool1D", testMaxPool1D),
        ("testMaxPool2D", testMaxPool2D),
        ("testMaxPool3D", testMaxPool3D),
        ("testAvgPool1D", testAvgPool1D),
        ("testAvgPool2D", testAvgPool2D),
        ("testAvgPool3D", testAvgPool3D),
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
