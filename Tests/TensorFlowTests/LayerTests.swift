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

fileprivate struct Sigmoid<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  typealias TangentVector = EmptyTangentVector

  @differentiable
  func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    sigmoid(input)
  }
}

final class LayerTests: XCTestCase {
  func testConv1D() {
    let filter = Tensor<Float>(ones: [3, 1, 2]) * Tensor<Float>([[[0.5, 1]]])
    let bias = Tensor<Float>([0, 1])
    let layer = Conv1D<Float>(
      filter: filter, bias: bias, activation: identity, stride: 1,
      padding: .valid)
    let input = Tensor<Float>([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]).expandingShape(at: 2)
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(
      shape: [2, 3, 2],
      scalars: [1.5, 4, 3, 7, 4.5, 10, 16.5, 34, 18, 37, 19.5, 40])
    XCTAssertEqual(output, expected)

    let layerNoBias = Conv1D<Float>(
      filter: filter, bias: nil, activation: identity, stride: 1,
      padding: .valid)
    let outputNoBias = layerNoBias.inferring(from: input)
    let expectedNoBias = Tensor<Float>(
      shape: [2, 3, 2],
      scalars: [1.5, 3, 3, 6, 4.5, 9, 16.5, 33, 18, 36, 19.5, 39])
    XCTAssertEqual(outputNoBias, expectedNoBias)
  }

  func testConv1DDilation() {
    // Filter shapes.
    let width = 3
    let inputChannels = 1
    let outputChannels = 2

    // Input shapes.
    let inputHeight = 2
    let inputWidth = 5

    let filter = Tensor<Float>(
      shape: [width, inputChannels, outputChannels],
      scalars: [2, 3, 4, 1, 2, 3])
    let bias = Tensor<Float>([0])
    let layer = Conv1D<Float>(
      filter: filter, bias: bias, activation: identity, stride: 1,
      padding: .valid, dilation: 2)
    let input = Tensor<Float>(
      shape: [inputHeight, inputWidth, 1],
      scalars: [0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(shape: [2, 1, 2], scalars: [16, 14, 96, 84])
    XCTAssertEqual(output, expected)
  }

  func testConv2D() {
    let filter = Tensor(shape: [1, 2, 2, 1], scalars: (0..<4).map(Float.init))
    let bias = Tensor<Float>([1, 2])
    let layer = Conv2D<Float>(
      filter: filter, bias: bias, activation: identity,
      strides: (2, 2), padding: .valid)
    let input = Tensor(shape: [2, 2, 2, 2], scalars: (0..<16).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(
      shape: [2, 1, 1, 2],
      scalars: [15, 16, 63, 64])
    XCTAssertEqual(output, expected)

    let layerNoBias = Conv2D<Float>(
      filter: filter, bias: nil, activation: identity,
      strides: (2, 2), padding: .valid)
    let outputNoBias = layerNoBias.inferring(from: input)
    let expectedNoBias = Tensor<Float>(
      shape: [2, 1, 1, 1],
      scalars: [14, 62])
    XCTAssertEqual(outputNoBias, expectedNoBias)
  }

  func testConv2DGradient() {
    let filter = Tensor(shape: [3, 3, 2, 4], scalars: (0..<72).map(Float.init))
    let bias = Tensor<Float>(zeros: [4])
    let layer = Conv2D<Float>(
      filter: filter,
      bias: bias,
      activation: identity,
      strides: (2, 2),
      padding: .valid)
    let input = Tensor(shape: [2, 4, 4, 2], scalars: (0..<64).map(Float.init))
    let grads = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(64, dtype=tf.float32), [2, 4, 4, 2])
    // filter = tf.reshape(tf.range(72, dtype=tf.float32), [3, 3, 2, 4])
    // bias = tf.zeros([4])
    // with tf.GradientTape() as tape:
    //     tape.watch([x, filter, bias])
    //     y = tf.math.reduce_sum(tf.nn.conv2d(input=x,
    //                                         filters=filter,
    //                                         strides=[1, 2, 2, 1],
    //                                         data_format="NHWC",
    //                                         padding="VALID") + bias)
    // print(tape.gradient(y, [x, filter, bias]))
    // ```
    XCTAssertEqual(
      grads.0,
      [
        [
          [[6, 22], [38, 54], [70, 86], [0, 0]],
          [[102, 118], [134, 150], [166, 182], [0, 0]],
          [[198, 214], [230, 246], [262, 278], [0, 0]],
          [[0, 0], [0, 0], [0, 0], [0, 0]],
        ],
        [
          [[6, 22], [38, 54], [70, 86], [0, 0]],
          [[102, 118], [134, 150], [166, 182], [0, 0]],
          [[198, 214], [230, 246], [262, 278], [0, 0]],
          [[0, 0], [0, 0], [0, 0], [0, 0]],
        ],
      ])
    XCTAssertEqual(
      grads.1.filter,
      [
        [
          [[32, 32, 32, 32], [34, 34, 34, 34]],
          [[36, 36, 36, 36], [38, 38, 38, 38]],
          [[40, 40, 40, 40], [42, 42, 42, 42]],
        ],
        [
          [[48, 48, 48, 48], [50, 50, 50, 50]],
          [[52, 52, 52, 52], [54, 54, 54, 54]],
          [[56, 56, 56, 56], [58, 58, 58, 58]],
        ],
        [
          [[64, 64, 64, 64], [66, 66, 66, 66]],
          [[68, 68, 68, 68], [70, 70, 70, 70]],
          [[72, 72, 72, 72], [74, 74, 74, 74]],
        ],
      ])
    XCTAssertEqual(grads.1.bias, [2, 2, 2, 2])
  }

  func testConv2DDilation() {
    // Input shapes. (Data format = NHWC)
    let batchSize = 2
    let inputHeight = 4
    let inputWidth = 4
    let inputChannels = 1
    let inputSize = batchSize * inputHeight * inputWidth * inputChannels

    // Filter shapes.
    let filterHeight = 2
    let filterWidth = 2
    let outputChannels = 1
    let filterSize = filterHeight * filterWidth * inputChannels * outputChannels

    // Testing.
    let filter = Tensor<Float>(
      shape: [filterHeight, filterWidth, inputChannels, outputChannels],
      scalars: (0..<filterSize).map(Float.init))
    let bias = Tensor<Float>([0])
    let layer = Conv2D<Float>(
      filter: filter, bias: bias, activation: identity, strides: (1, 1),
      padding: .valid, dilations: (2, 2))
    let input = Tensor<Float>(
      shape: [batchSize, inputHeight, inputWidth, inputChannels],
      scalars: (0..<inputSize).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(
      shape: [2, 2, 2, 1],
      scalars: [48, 54, 72, 78, 144, 150, 168, 174])
    XCTAssertEqual(output, expected)
  }

  func testConv3D() {
    let filter = Tensor(shape: [1, 2, 2, 2, 1], scalars: (0..<8).map(Float.init))
    let bias = Tensor<Float>([-1, 1])
    let layer = Conv3D<Float>(
      filter: filter, bias: bias, activation: identity,
      strides: (1, 2, 1), padding: .valid, dilations: (1, 1, 1))
    let input = Tensor(shape: [2, 2, 2, 2, 2], scalars: (0..<32).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(
      shape: [2, 2, 1, 1, 2],
      scalars: [139, 141, 363, 365, 587, 589, 811, 813])
    XCTAssertEqual(output, expected)

    let layerNoBias = Conv3D<Float>(
      filter: filter, bias: nil, activation: identity,
      strides: (1, 2, 1), padding: .valid, dilations: (1, 1, 1))
    let outputNoBias = layerNoBias.inferring(from: input)
    let expectedNoBias = Tensor<Float>(
      shape: [2, 2, 1, 1, 1],
      scalars: [140, 364, 588, 812])
    XCTAssertEqual(outputNoBias, expectedNoBias)
  }

  func testConv3DGradient() {
    let filter = Tensor(shape: [1, 4, 4, 1, 1], scalars: (0..<16).map(Float.init))
    let bias = Tensor<Float>(ones: [2])
    let layer = Conv3D(
      filter: filter,
      bias: bias,
      activation: identity,
      strides: (2, 2, 2),
      padding: .same)
    let input = Tensor(shape: [1, 4, 4, 4, 1], scalars: (0..<64).map(Float.init))
    let grads = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(64, dtype=tf.float32), [1, 4, 4, 4, 1])
    // filter = tf.reshape(tf.range(72, dtype=tf.float32), [1, 4, 4, 1, 1])
    // bias = tf.ones([2])
    // with tf.GradientTape() as tape:
    //     tape.watch([x, filter, bias])
    //     y = tf.math.reduce_sum(tf.nn.conv3d(input=x,
    //                                         filters=filter,
    //                                         strides=[1, 2, 2, 2, 1],
    //                                         padding="SAME") + bias)
    // print(tape.gradient(y, [x, filter, bias]))
    // ```
    XCTAssertEqual(
      grads.0,
      [
        [
          [
            [[10.0], [20.0], [24.0], [12.0]],
            [[20.0], [40.0], [48.0], [24.0]],
            [[36.0], [72.0], [80.0], [40.0]],
            [[18.0], [36.0], [40.0], [20.0]],
          ],
          [
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
          ],
          [
            [[10.0], [20.0], [24.0], [12.0]],
            [[20.0], [40.0], [48.0], [24.0]],
            [[36.0], [72.0], [80.0], [40.0]],
            [[18.0], [36.0], [40.0], [20.0]],
          ],
          [
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
          ],
        ]
      ])
    XCTAssertEqual(
      grads.1.filter,
      [
        [
          [[[84.0]], [[168.0]], [[176.0]], [[88.0]]],
          [[[168.0]], [[336.0]], [[352.0]], [[176.0]]],
          [[[200.0]], [[400.0]], [[416.0]], [[208.0]]],
          [[[100.0]], [[200.0]], [[208.0]], [[104.0]]],
        ]
      ])
    XCTAssertEqual(grads.1.bias, [8.0, 8.0])
  }

  func testDepthwiseConv2D() {
    let filter = Tensor(shape: [2, 2, 2, 2], scalars: (0..<16).map(Float.init))
    let bias = Tensor<Float>([1, 2, 3, 4])
    let layer = DepthwiseConv2D<Float>(
      filter: filter, bias: bias, activation: identity,
      strides: (2, 2), padding: .same)
    let input = Tensor(shape: [1, 1, 8, 2], scalars: (0..<16).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(
      shape: [1, 1, 4, 4],
      scalars: [
        9, 12, 23, 28, 25, 36, 55, 68, 41, 60, 87, 108,
        57, 84, 119, 148,
      ])
    XCTAssertEqual(output, expected)

    let channelMultiplier = 4
    let multiplierLayer = DepthwiseConv2D<Float>(
      filterShape: (2, 2, input.shape[3], channelMultiplier),
      filterInitializer: glorotUniform(),
      biasInitializer: zeros())
    let multiplierOutput = multiplierLayer.inferring(from: input)
    XCTAssertEqual(multiplierOutput.shape[3], input.shape[3] * channelMultiplier)
  }

  func testDepthwiseConv2DGradient() {
    let filter = Tensor(shape: [2, 1, 2, 2], scalars: (0..<8).map(Float.init))
    let bias = Tensor<Float>(ones: [4])
    let layer = DepthwiseConv2D<Float>(
      filter: filter,
      bias: bias,
      activation: identity,
      strides: (1, 1),
      padding: .same)
    let input = Tensor(shape: [2, 1, 2, 2], scalars: (0..<8).map(Float.init))
    let grads = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // input = tf.reshape(tf.range(8, dtype=tf.float32), [2, 1, 2, 2])
    // filter = tf.reshape(tf.range(8, dtype=tf.float32), [2, 1, 2, 2])
    // bias = tf.ones([4])
    // with tf.GradientTape() as tape:
    //     tape.watch([x, filter, bias])
    //     y = tf.math.reduce_sum(tf.nn.depthwise_conv2d(input=x,
    //                                                   filters=filter,
    //                                                   strides=[1, 1, 1, 1],
    //                                                   data_format="NHWC",
    //                                                   padding="SAME") + bias)
    // print(tape.gradient(y, [x, filter, bias]))
    // ```
    XCTAssertEqual(
      grads.0,
      [
        [[[1, 5], [1, 5]]],
        [[[1, 5], [1, 5]]],
      ])
    XCTAssertEqual(
      grads.1.filter,
      [
        [[[12, 12], [16, 16]]],
        [[[0, 0], [0, 0]]],
      ])
    XCTAssertEqual(grads.1.bias, [4, 4, 4, 4])
  }

  func testTransposedConv1D() {
    let filter = Tensor(shape: [4, 1, 1], scalars: (0..<4).map(Float.init))
    let bias = Tensor<Float>([8])
    let layer = TransposedConv1D(
      filter: filter, bias: bias, activation: identity,
      stride: 1, padding: .same)
    let input = Tensor(shape: [1, 4, 1], scalars: (0..<4).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(
      shape: [1, 1, 4, 1],
      scalars: [8, 9, 12, 18])
    XCTAssertEqual(output, expected)

    let layerNoBias = TransposedConv1D(
      filter: filter, bias: nil, activation: identity,
      stride: 1, padding: .same)
    let outputNoBias = layerNoBias.inferring(from: input)
    let expectedNoBias = Tensor<Float>(
      shape: [1, 1, 4, 1],
      scalars: [0, 1, 4, 10])
    XCTAssertEqual(outputNoBias, expectedNoBias)
  }

  func testTransposedConv2D() {
    let filter = Tensor(shape: [4, 2, 1, 1], scalars: (0..<8).map(Float.init))
    let bias = Tensor<Float>([8])
    let layer = TransposedConv2D(
      filter: filter, bias: bias, activation: identity,
      strides: (1, 1), padding: .same)
    let input = Tensor(shape: [1, 4, 2, 1], scalars: (0..<8).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(
      shape: [1, 4, 2, 1],
      scalars: [8, 12, 12, 28, 24, 64, 48, 112])
    XCTAssertEqual(output, expected)

    let layerNoBias = TransposedConv2D(
      filter: filter, bias: nil, activation: identity,
      strides: (1, 1), padding: .same)
    let outputNoBias = layerNoBias.inferring(from: input)
    let expectedNoBias = Tensor<Float>(
      shape: [1, 4, 2, 1],
      scalars: [0, 4, 4, 20, 16, 56, 40, 104])
    XCTAssertEqual(outputNoBias, expectedNoBias)
  }

  func testTransposedConv2DGradient() {
    let filter = Tensor(shape: [3, 3, 2, 4], scalars: (0..<72).map(Float.init))
    let bias = Tensor<Float>(zeros: [2])
    let layer = TransposedConv2D<Float>(
      filter: filter,
      bias: bias,
      activation: identity,
      strides: (2, 2),
      padding: .same)
    let input = Tensor(shape: [2, 2, 2, 4], scalars: (0..<32).map(Float.init))
    let grads = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(32, dtype=tf.float32), [2, 2, 2, 4])
    // filter = tf.reshape(tf.range(72, dtype=tf.float32), [3, 3, 2, 4])
    // bias = tf.zeros([2])
    // with tf.GradientTape() as tape:
    //     tape.watch([x, filter, bias])
    //     y = tf.math.reduce_sum(tf.nn.conv2d_transpose(input=x,
    //                                                   filters=filter,
    //                                                   output_shape=[2, 4, 4, 2],
    //                                                   strides=[1, 2, 2, 1],
    //                                                   data_format="NHWC",
    //                                                   padding="SAME") + bias)
    // print(tape.gradient(y, [x, filter, bias]))
    // ```
    XCTAssertEqual(
      grads.0,
      [
        [
          [[612, 630, 648, 666], [360, 372, 384, 396]],
          [[264, 276, 288, 300], [144, 152, 160, 168]],
        ],
        [
          [[612, 630, 648, 666], [360, 372, 384, 396]],
          [[264, 276, 288, 300], [144, 152, 160, 168]],
        ],
      ])
    XCTAssertEqual(
      grads.1.filter,
      [
        [
          [[112, 120, 128, 136], [112, 120, 128, 136]],
          [[112, 120, 128, 136], [112, 120, 128, 136]],
          [[48, 52, 56, 60], [48, 52, 56, 60]],
        ],
        [
          [[112, 120, 128, 136], [112, 120, 128, 136]],
          [[112, 120, 128, 136], [112, 120, 128, 136]],
          [[48, 52, 56, 60], [48, 52, 56, 60]],
        ],
        [
          [[40, 44, 48, 52], [40, 44, 48, 52]],
          [[40, 44, 48, 52], [40, 44, 48, 52]],
          [[16, 18, 20, 22], [16, 18, 20, 22]],
        ],
      ])
    XCTAssertEqual(grads.1.bias, [32, 32])
  }

  func testTransposedConv3D() {
    let filter = Tensor(shape: [2, 2, 2, 1, 1], scalars: (0..<8).map(Float.init))
    let bias = Tensor<Float>([8])
    let layer = TransposedConv3D(
      filter: filter, bias: bias, activation: identity,
      strides: (1, 1, 1), padding: .same)
    let input = Tensor(shape: [1, 2, 2, 2, 1], scalars: (0..<8).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(
      shape: [1, 2, 2, 2, 1],
      scalars: [8, 8, 8, 12, 8, 16, 24, 64])
    XCTAssertEqual(output, expected)

    let layerNoBias = TransposedConv3D(
      filter: filter, bias: nil, activation: identity,
      strides: (1, 1, 1), padding: .same)
    let outputNoBias = layerNoBias.inferring(from: input)
    let expectedNoBias = Tensor<Float>(
      shape: [1, 2, 2, 2, 1],
      scalars: [0, 0, 0, 4, 0, 8, 16, 56])
    XCTAssertEqual(outputNoBias, expectedNoBias)
  }

  func testSeparableConv1D() {
    let depthwiseFilter = Tensor(shape: [2, 2, 2], scalars: (0..<8).map(Float.init))
    let pointwiseFilter = Tensor(shape: [1, 4, 1], scalars: (0..<4).map(Float.init))
    let bias = Tensor<Float>([4])
    let layer = SeparableConv1D<Float>(
      depthwiseFilter: depthwiseFilter,
      pointwiseFilter: pointwiseFilter,
      bias: bias,
      activation: identity,
      stride: 1,
      padding: .same)
    let input = Tensor(shape: [2, 2, 2], scalars: (0..<8).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(shape: [2, 2, 1], scalars: [17, 45, 73, 101])
    XCTAssertEqual(output, expected)

    let layerNoBias = SeparableConv1D<Float>(
      depthwiseFilter: depthwiseFilter,
      pointwiseFilter: pointwiseFilter,
      bias: nil,
      activation: identity,
      stride: 1,
      padding: .same)
    let outputNoBias = layerNoBias.inferring(from: input)
    let expectedNoBias = Tensor<Float>(shape: [2, 2, 1], scalars: [13, 41, 69, 97])
    XCTAssertEqual(outputNoBias, expectedNoBias)
  }

  func testSeparableConv2D() {
    let depthwiseFilter = Tensor(shape: [2, 2, 2, 2], scalars: (0..<16).map(Float.init))
    let pointwiseFilter = Tensor(shape: [1, 1, 4, 1], scalars: (0..<4).map(Float.init))
    let bias = Tensor<Float>([4])
    let layer = SeparableConv2D<Float>(
      depthwiseFilter: depthwiseFilter,
      pointwiseFilter: pointwiseFilter,
      bias: bias,
      activation: identity,
      strides: (2, 2),
      padding: .valid)
    let input = Tensor(shape: [2, 2, 2, 2], scalars: (0..<16).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(
      shape: [2, 1, 1, 1],
      scalars: [1016, 2616])
    XCTAssertEqual(output, expected)

    let layerNoBias = SeparableConv2D<Float>(
      depthwiseFilter: depthwiseFilter,
      pointwiseFilter: pointwiseFilter,
      bias: nil,
      activation: identity,
      strides: (2, 2),
      padding: .valid)
    let outputNoBias = layerNoBias.inferring(from: input)
    let expectedNoBias = Tensor<Float>(
      shape: [2, 1, 1, 1],
      scalars: [1012, 2612])
    XCTAssertEqual(outputNoBias, expectedNoBias)
  }

  func testSeparableConv2DGradient() {
    let depthwiseFilter = Tensor(shape: [2, 1, 2, 2], scalars: (0..<8).map(Float.init))
    let pointwiseFilter = Tensor(shape: [1, 1, 4, 1], scalars: (0..<4).map(Float.init))
    let bias = Tensor<Float>([1, 1])
    let layer = SeparableConv2D<Float>(
      depthwiseFilter: depthwiseFilter,
      pointwiseFilter: pointwiseFilter,
      bias: bias,
      activation: identity,
      strides: (1, 1),
      padding: .same)
    let input = Tensor(shape: [2, 1, 2, 2], scalars: (0..<8).map(Float.init))
    let grads = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(8, dtype=tf.float32), [2, 1, 2, 2])
    // depthwiseFilter = tf.reshape(tf.range(8, dtype=tf.float32), [2, 1, 2, 2])
    // pointwiseFilter = tf.reshape(tf.range(4, dtype=tf.float32), [1, 1, 4, 1])
    // bias = tf.ones([2])
    // with tf.GradientTape() as tape:
    //     tape.watch([x, depthwiseFilter, pointwiseFilter, bias])
    //     y = tf.math.reduce_sum(tf.nn.separable_conv2D(input,
    //                                                   depthwiseFilter,
    //                                                   pointwiseFilter
    //                                                   strides=[1, 1, 1, 1],
    //                                                   padding="SAME") + bias)
    // print(tape.gradient(y, [x, depthwiseFilter, pointwiseFilter, bias])
    // ```
    XCTAssertEqual(
      grads.0,
      [
        [[[2.0, 26.0], [2.0, 26.0]]],
        [[[2.0, 26.0], [2.0, 26.0]]],
      ])
    XCTAssertEqual(
      grads.1.depthwiseFilter,
      [
        [[[0.0, 24.0], [64.0, 96.0]]],
        [[[0.0, 0.0], [0.0, 0.0]]],
      ])
    XCTAssertEqual(grads.1.bias, [4.0, 4.0])
  }

  func testZeroPadding1D() {
    let input = Tensor<Float>(shape: [1, 3, 1], scalars: [0.0, 1.0, 2.0])
    let layer = ZeroPadding1D<Float>(padding: 2)
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(
      shape: [1, 7, 1],
      scalars: [0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0])
    XCTAssertEqual(output, expected)
  }

  func testZeroPadding1DGradient() {
    let x = Tensor<Float>(shape: [1, 3, 1], scalars: [0.0, 1.0, 2.0])
    let layer = ZeroPadding1D<Float>(padding: 2)
    let computedGradient = gradient(at: x, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.constant([0.0, 1.0, 2.0]), [1, 3, 1])
    // layer = tf.keras.layers.ZeroPadding1D(2)
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.reduce_sum(layer(x))
    // print(tape.gradient(y, x))
    // ```
    let expectedGradient = Tensor<Float>(onesLike: x)
    XCTAssertEqual(computedGradient.0, expectedGradient)
  }

  func testZeroPadding2D() {
    let input = Tensor<Float>(shape: [1, 3, 1, 1], scalars: [0.0, 1.0, 2.0])
    let layer = ZeroPadding2D<Float>(padding: ((0, 0), (0, 1)))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(
      shape: [1, 3, 2, 1],
      scalars: [0.0, 0.0, 1.0, 0.0, 2.0, 0.0])
    XCTAssertEqual(output, expected)
  }

  func testZeroPadding2DGradient() {
    let x = Tensor<Float>(shape: [1, 3, 1, 1], scalars: [0.0, 1.0, 2.0])
    let layer = ZeroPadding2D<Float>(padding: ((0, 0), (0, 1)))
    let computedGradient = gradient(at: x, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.constant([0.0, 1.0, 2.0]), [1, 3, 1, 1])
    // layer = tf.keras.layers.ZeroPadding2D(((0, 0), (0, 1)))
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.reduce_sum(layer(x))
    // print(tape.gradient(y, x))
    // ```
    let expectedGradient = Tensor<Float>(onesLike: x)
    XCTAssertEqual(computedGradient.0, expectedGradient)
  }

  func testZeroPadding3D() {
    let input = Tensor<Float>(shape: [1, 3, 1, 1, 1], scalars: [0.0, 1.0, 2.0])
    let layer = ZeroPadding3D<Float>(padding: ((0, 0), (0, 1), (0, 0)))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(shape: [1, 3, 2, 1, 1], scalars: [0, 0, 1, 0, 2, 0])
    XCTAssertEqual(output, expected)
  }

  func testZeroPadding3DGradient() {
    let x = Tensor<Float>(shape: [1, 3, 1, 1, 1], scalars: [0.0, 1.0, 2.0])
    let layer = ZeroPadding3D<Float>(padding: ((0, 0), (0, 1), (0, 0)))
    let computedGradient = gradient(at: x, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.constant([0.0, 1.0, 2.0]), [1, 3, 1, 1, 1])
    // layer = tf.keras.layers.ZeroPadding3D(((0, 0), (0, 1), (0, 0)))
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.reduce_sum(layer(x))
    // print(tape.gradient(y, x))
    // ```
    let expectedGradient = Tensor<Float>(onesLike: x)
    XCTAssertEqual(computedGradient.0, expectedGradient)
  }

  func testMaxPool1D() {
    let layer = MaxPool1D<Float>(poolSize: 3, stride: 1, padding: .valid)
    let input = Tensor<Float>([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]).expandingShape(at: 2)
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[[2], [3], [4]], [[12], [13], [14]]])
    XCTAssertEqual(output, expected)
  }

  func testMaxPool1DGradient() {
    let layer = MaxPool1D<Float>(poolSize: 2, stride: 1, padding: .valid)
    let x = Tensor<Float>(shape: [1, 4, 4], scalars: (0..<16).map(Float.init))
    let computedGradient = gradient(at: x, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // maxpool1D = tf.keras.layers.MaxPool1D()
    // x = tf.reshape(tf.range(16, dtype=tf.float32), [1, 4, 4])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(maxpool1D(x))
    // print(tape.gradient(y, x))
    // ```
    let expectedGradient = Tensor<Float>([
      [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
      ]
    ])
    XCTAssertEqual(computedGradient.0, expectedGradient)
  }

  func testMaxPool2D() {
    let layer = MaxPool2D<Float>(poolSize: (2, 2), strides: (1, 1), padding: .valid)
    let input = Tensor(shape: [1, 2, 2, 1], scalars: (0..<4).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[[[3]]]])
    XCTAssertEqual(output, expected)
  }

  func testMaxPool2DGradient() {
    let layer = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2), padding: .valid)
    let x = Tensor(shape: [1, 4, 4, 1], scalars: (0..<16).map(Float.init))
    let computedGradient = gradient(at: x, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // maxpool2D = tf.keras.layers.MaxPool2D(strides=(2, 2))
    // x = tf.reshape(tf.range(16, dtype=tf.float32), [1, 4, 4, 1])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(maxpool2D(x))
    // print(tape.gradient(y, x))
    // ```
    let expectedGradient = Tensor<Float>([
      [
        [[0], [0], [0], [0]],
        [[0], [1], [0], [1]],
        [[0], [0], [0], [0]],
        [[0], [1], [0], [1]],
      ]
    ])
    XCTAssertEqual(computedGradient.0, expectedGradient)
  }

  func testMaxPool3D() {
    let layer = MaxPool3D<Float>(poolSize: (2, 2, 2), strides: (1, 1, 1), padding: .valid)
    let input = Tensor(shape: [1, 2, 2, 2, 1], scalars: (0..<8).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[[[[7]]]]])
    XCTAssertEqual(output, expected)
  }

  func testMaxPool3DGradient() {
    let layer = MaxPool3D<Float>(poolSize: (2, 2, 2), strides: (1, 1, 1), padding: .valid)
    let x = Tensor(shape: [1, 2, 2, 2, 1], scalars: (0..<8).map(Float.init))
    let computedGradient = gradient(at: x, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // maxpool3D = tf.keras.layers.MaxPool3D(strides=(1, 1, 1))
    // x = tf.reshape(tf.range(8, dtype=tf.float32), [1, 2, 2, 2, 1])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(maxpool3D(x))
    // print(tape.gradient(y, x))
    // ```
    let expectedGradient = Tensor<Float>([
      [
        [
          [[0], [0]],
          [[0], [0]],
        ],
        [
          [[0], [0]],
          [[0], [1]],
        ],
      ]
    ])
    XCTAssertEqual(computedGradient.0, expectedGradient)
  }

  func testAvgPool1D() {
    let layer = AvgPool1D<Float>(poolSize: 3, stride: 1, padding: .valid)
    let input = Tensor<Float>([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]).expandingShape(at: 2)
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[[1], [2], [3]], [[11], [12], [13]]])
    XCTAssertEqual(output, expected)
  }

  func testAvgPool1DGradient() {
    let layer = AvgPool1D<Float>(poolSize: 2, stride: 1, padding: .valid)
    let x = Tensor(shape: [1, 4, 4], scalars: (0..<16).map(Float.init))
    let computedGradient = gradient(at: x, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // avgpool1D = tf.keras.layers.AvgPool1D(strides=1)
    // x = tf.reshape(tf.range(16, dtype=tf.float32), [1, 4, 4])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(avgpool1D(x))
    // print(tape.gradient(y, x))
    // ```
    let expectedGradient = Tensor<Float>([
      [
        [0.5, 0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
      ]
    ])
    XCTAssertEqual(computedGradient.0, expectedGradient)
  }

  func testAvgPool2D() {
    let layer = AvgPool2D<Float>(poolSize: (2, 5), strides: (1, 1), padding: .valid)
    let input = Tensor(shape: [1, 2, 5, 1], scalars: (0..<10).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[[[4.5]]]])
    XCTAssertEqual(output, expected)
  }

  func testAvgPool2DGradient() {
    let layer = AvgPool2D<Float>(poolSize: (2, 2), strides: (1, 1), padding: .valid)
    let x = Tensor(shape: [1, 4, 4, 2], scalars: (0..<32).map(Float.init))
    let computedGradient = gradient(at: x, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // avgpool2D = tf.keras.layers.AvgPool2D(strides=(1, 1))
    // x = tf.reshape(tf.range(32, dtype=tf.float32), [1, 4, 4, 2])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(avgpool2D(x))
    // print(tape.gradient(y, x))
    // ```
    let expectedGradient = Tensor<Float>([
      [
        [[0.25, 0.25], [0.50, 0.50], [0.50, 0.50], [0.25, 0.25]],
        [[0.50, 0.50], [1.00, 1.00], [1.00, 1.00], [0.50, 0.50]],
        [[0.50, 0.50], [1.00, 1.00], [1.00, 1.00], [0.50, 0.50]],
        [[0.25, 0.25], [0.50, 0.50], [0.50, 0.50], [0.25, 0.25]],
      ]
    ])
    XCTAssertEqual(computedGradient.0, expectedGradient)
  }

  func testAvgPool3D() {
    let layer = AvgPool3D<Float>(poolSize: (2, 4, 5), strides: (1, 1, 1), padding: .valid)
    let input = Tensor(shape: [1, 2, 4, 5, 1], scalars: (0..<40).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[[[[19.5]]]]])
    XCTAssertEqual(output, expected)
  }

  func testAvgPool3DGradient() {
    let layer = AvgPool3D<Float>(poolSize: (2, 2, 2), strides: (1, 1, 1), padding: .valid)
    let x = Tensor(shape: [1, 2, 2, 2, 1], scalars: (0..<8).map(Float.init))
    let computedGradient = gradient(at: x, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // avgpool3D = tf.keras.layers.AvgPool3D(strides=(1, 1, 1))
    // x = tf.reshape(tf.range(8, dtype=tf.float32), [1, 2, 2, 2, 1])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(avgpool3D(x))
    // print(tape.gradient(y, x))
    // ```
    let expectedGradient = Tensor<Float>(repeating: 0.125, shape: [1, 2, 2, 2, 1])
    XCTAssertEqual(computedGradient.0, expectedGradient)
  }

  func testGlobalAvgPool1D() {
    let layer = GlobalAvgPool1D<Float>()
    let input = Tensor(shape: [2, 5, 1], scalars: (0..<10).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[2], [7]])
    XCTAssertEqual(output, expected)
  }

  func testGlobalAvgPool1DGradient() {
    let layer = GlobalAvgPool1D<Float>()
    let input = Tensor(shape: [2, 2, 2], scalars: (0..<8).map(Float.init))
    let computedGradient = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // globalAvgPool1D = tf.keras.layers.GlobalAveragePooling1D()
    // x = tf.reshape(tf.range(8, dtype=tf.float32), [2, 2, 2])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(globalAvgPool1D(x))
    // print(tape.gradient(y, x))
    // ```
    XCTAssertEqual(
      computedGradient.0,
      [
        [
          [0.5, 0.5],
          [0.5, 0.5],
        ],
        [
          [0.5, 0.5],
          [0.5, 0.5],
        ],
      ])
  }

  func testGlobalAvgPool2D() {
    let layer = GlobalAvgPool2D<Float>()
    let input = Tensor(shape: [2, 6, 2, 1], scalars: (0..<24).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[5.5], [17.5]])
    XCTAssertEqual(output, expected)
  }

  func testGlobalAvgPool2DGradient() {
    let layer = GlobalAvgPool2D<Float>()
    let input = Tensor(shape: [2, 2, 2, 2], scalars: (0..<16).map(Float.init))
    let computedGradient = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // globalAvgPool2D = tf.keras.layers.GlobalAveragePooling2D()
    // x = tf.reshape(tf.range(16, dtype=tf.float32), [2, 2, 2, 2])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(globalAvgPool2D(x))
    // print(tape.gradient(y, x))
    // ```
    XCTAssertEqual(
      computedGradient.0,
      [
        [
          [[0.25, 0.25], [0.25, 0.25]],
          [[0.25, 0.25], [0.25, 0.25]],
        ],
        [
          [[0.25, 0.25], [0.25, 0.25]],
          [[0.25, 0.25], [0.25, 0.25]],
        ],
      ])
  }

  func testGlobalAvgPool3D() {
    let layer = GlobalAvgPool3D<Float>()
    let input = Tensor<Float>(shape: [2, 6, 2, 1, 1], scalars: (0..<24).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[5.5], [17.5]])
    XCTAssertEqual(output, expected)
  }

  func testGlobalAvgPool3DGradient() {
    let layer = GlobalAvgPool3D<Float>()
    let input = Tensor(shape: [1, 3, 2, 3, 1], scalars: (0..<18).map(Float.init))
    let computedGradient = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // globalAvgPool3D = tf.keras.layers.GlobalAveragePooling3D()
    // x = tf.reshape(tf.range(18, dtype=tf.float32), [1, 3, 2, 2, 1])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(globalAvgPool3D(x))
    // print(tape.gradient(y, x))
    // ```
    XCTAssertEqual(
      computedGradient.0,
      [
        [
          [
            [[0.055555556], [0.055555556], [0.055555556]],
            [[0.055555556], [0.055555556], [0.055555556]],
          ],
          [
            [[0.055555556], [0.055555556], [0.055555556]],
            [[0.055555556], [0.055555556], [0.055555556]],
          ],
          [
            [[0.055555556], [0.055555556], [0.055555556]],
            [[0.055555556], [0.055555556], [0.055555556]],
          ],
        ]
      ])
  }

  func testGlobalMaxPool1D() {
    let layer = GlobalMaxPool1D<Float>()
    let input = Tensor(shape: [1, 10, 1], scalars: (0..<10).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[9]])
    XCTAssertEqual(output, expected)
  }

  func testGlobalMaxPool1DGradient() {
    let layer = GlobalMaxPool1D<Float>()
    let input = Tensor(shape: [2, 2, 2], scalars: (0..<8).map(Float.init))
    let computedGradient = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // globalMaxPool1D = tf.keras.layers.GlobalMaxPooling1D()
    // x = tf.reshape(tf.range(8, dtype=tf.float32), [2, 2, 2])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(globalMaxPool1D(x))
    // print(tape.gradient(y, x))
    // ```
    XCTAssertEqual(
      computedGradient.0,
      [
        [
          [0.0, 0.0],
          [1.0, 1.0],
        ],
        [
          [0.0, 0.0],
          [1.0, 1.0],
        ],
      ])
  }

  func testGlobalMaxPool2D() {
    let layer = GlobalMaxPool2D<Float>()
    let input = Tensor(shape: [1, 2, 10, 1], scalars: (0..<20).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[19]])
    XCTAssertEqual(output, expected)
  }

  func testGlobalMaxPool2DGradient() {
    let layer = GlobalMaxPool2D<Float>()
    let input = Tensor(shape: [2, 3, 3, 2], scalars: (0..<36).map(Float.init))
    let computedGradient = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    //import tensorflow as tf
    // globalMaxPool2D = tf.keras.layers.GlobalMaxPooling2D()
    // x = tf.reshape(tf.range(36, dtype=tf.float32), [2, 3, 3, 2])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(globalMaxPool2D(x))
    // print(tape.gradient(y, x))
    // ```
    XCTAssertEqual(
      computedGradient.0,
      [
        [
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
          [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        ],
        [
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
          [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        ],
      ])
  }

  func testGlobalMaxPool3D() {
    let layer = GlobalMaxPool3D<Float>()
    let input = Tensor<Float>(shape: [1, 2, 3, 5, 1], scalars: (0..<30).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[29]])
    XCTAssertEqual(output, expected)
  }

  func testGlobalMaxPool3DGradient() {
    let layer = GlobalMaxPool3D<Float>()
    let input = Tensor(shape: [2, 2, 2, 2, 2], scalars: (0..<32).map(Float.init))
    let computedGradient = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // globalMaxPool3D = tf.keras.layers.GlobalMaxPooling3D()
    // x = tf.reshape(tf.range(32, dtype=tf.float32), [2, 2, 2, 2, 2])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(globalMaxPool3D(x))
    // print(tape.gradient(y, x))
    // ```
    XCTAssertEqual(
      computedGradient.0,
      [
        [
          [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
          ],
          [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
          ],
        ],
        [
          [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
          ],
          [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
          ],
        ],
      ])
  }

  func testFractionalMaxPool2D() {
    let layer = FractionalMaxPool2D<Float>(poolingRatio: (1.4, 1.5), pseudoRandom: false)
    let input = Tensor(shape: [1, 2, 2, 1], scalars: (0..<4).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[[[3]]]])
    XCTAssertEqual(output, expected)
  }

  func testFractionalMaxPool2DGradient() {
    let layer = FractionalMaxPool2D<Float>(poolingRatio: (1.4, 1.5), pseudoRandom: false)
    let x = Tensor(shape: [1, 2, 2, 1], scalars: (0..<4).map(Float.init))
    let computedGradient = gradient(at: x, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(4, dtype=tf.float32), [1, 2, 2, 1])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(tf.nn.fractional_max_pool(x, pooling_ratio=[1, 1.4, 1.5, 1], pseudo_random=False)[0])
    // print(tape.gradient(y, x))
    // ```
    let expectedGradient = Tensor<Float>([
      [
        [
          [0.0],
          [0.0],
        ],
        [
          [0.0],
          [1.0],
        ],
      ]
    ])
    XCTAssertEqual(computedGradient.0, expectedGradient)
  }

  func testUpSampling1D() {
    let size = 6
    let layer = UpSampling1D<Float>(size: size)
    let input = Tensor<Float>(shape: [1, 10, 1], scalars: (0..<10).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = TensorShape([1, input.shape[1] * size, 1])
    XCTAssertEqual(output.shape, expected)
  }

  func testUpSampling1DGradient() {
    let layer = UpSampling1D<Float>(size: 3)
    let input = Tensor(shape: [2, 2, 2], scalars: (0..<8).map(Float.init))
    let computedGradient = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // upSampling1D = tf.keras.layers.UpSampling1D(size = 3)
    // x = tf.reshape(tf.range(8, dtype=tf.float32), [2, 2, 2])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(upSampling1D(x))
    // print(tape.gradient(y, x))
    // ```
    XCTAssertEqual(
      computedGradient.0,
      [
        [
          [3.0, 3.0],
          [3.0, 3.0],
        ],
        [
          [3.0, 3.0],
          [3.0, 3.0],
        ],
      ])
  }

  func testUpSampling2D() {
    let size = 6
    let layer = UpSampling2D<Float>(size: size)
    let input = Tensor<Float>(shape: [1, 3, 5, 1], scalars: (0..<15).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = TensorShape([1, input.shape[1] * size, input.shape[2] * size, 1])
    XCTAssertEqual(output.shape, expected)
  }

  func testUpSampling2DGradient() {
    let layer = UpSampling2D<Float>(size: 3)
    let input = Tensor(shape: [1, 3, 4, 2], scalars: (0..<24).map(Float.init))
    let computedGradient = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // upSampling2D = tf.keras.layers.UpSampling2D(size = 3)
    // x = tf.reshape(tf.range(24, dtype=tf.float32), [1, 3, 4, 2])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(upSampling2D(x))
    // print(tape.gradient(y, x))
    // ```
    XCTAssertEqual(
      computedGradient.0,
      [
        [
          [[9.0, 9.0], [9.0, 9.0], [9.0, 9.0], [9.0, 9.0]],
          [[9.0, 9.0], [9.0, 9.0], [9.0, 9.0], [9.0, 9.0]],
          [[9.0, 9.0], [9.0, 9.0], [9.0, 9.0], [9.0, 9.0]],
        ]
      ])
  }

  func testUpSampling3D() {
    let size = 6
    let layer = UpSampling3D<Float>(size: size)
    let input = Tensor<Float>(shape: [1, 4, 3, 2, 1], scalars: (0..<24).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = TensorShape(
      [1, input.shape[1] * size, input.shape[2] * size, input.shape[3] * size, 1])
    XCTAssertEqual(output.shape, expected)
    XCTAssertEqual(output.shape, expected)
  }

  func testUpSampling3DGradient() {
    let layer = UpSampling3D<Float>(size: 3)
    let input = Tensor(shape: [1, 2, 2, 2, 4], scalars: (0..<32).map(Float.init))
    let computedGradient = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // upSampling3D = tf.keras.layers.UpSampling3D(size = 3)
    // x = tf.reshape(tf.range(32, dtype=tf.float32), [1, 2, 2, 2, 4])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(upSampling3D(x))
    // print(tape.gradient(y, x))
    // ```
    XCTAssertEqual(
      computedGradient.0,
      [
        [
          [
            [[27.0, 27.0, 27.0, 27.0], [27.0, 27.0, 27.0, 27.0]],
            [[27.0, 27.0, 27.0, 27.0], [27.0, 27.0, 27.0, 27.0]],
          ],
          [
            [[27.0, 27.0, 27.0, 27.0], [27.0, 27.0, 27.0, 27.0]],
            [[27.0, 27.0, 27.0, 27.0], [27.0, 27.0, 27.0, 27.0]],
          ],
        ]
      ])
  }

  func testReshape() {
    let layer = Reshape<Float>(shape: [10, 2, 1])
    let input = Tensor(shape: [20, 1], scalars: (0..<20).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = TensorShape([10, 2, 1])
    XCTAssertEqual(output.shape, expected)
  }

  func testReshapeGradient() {
    let layer = Reshape<Float>(shape: [10, 2, 1])
    let input = Tensor(shape: [1, 5, 4], scalars: (0..<20).map(Float.init))
    let computedGradient = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // reshape = tf.keras.layers.Reshape(target_shape = (10, 2, 1))
    // x = tf.reshape(tf.range(20, dtype=tf.float32), [1, 5, 4])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(reshape(x))
    // print(tape.gradient(y, x))
    // ```
    XCTAssertEqual(
      computedGradient.0,
      [
        [
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
        ]
      ])
  }

  func testFlatten() {
    let layer = Flatten<Float>()
    let input = Tensor(shape: [10, 2, 2], scalars: (0..<40).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = TensorShape([10, 4])
    XCTAssertEqual(output.shape, expected)
  }

  func testFlattenGradient() {
    let layer = Flatten<Float>()
    let input = Tensor(shape: [1, 4, 4], scalars: (0..<16).map(Float.init))
    let computedGradient = gradient(at: input, layer) { $1($0).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // flatten = tf.keras.layers.Flatten()
    // x = tf.reshape(tf.range(16, dtype=tf.float32), [1, 4, 4])
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = tf.math.reduce_sum(flatten(x))
    // print(tape.gradient(y, x))
    // ```
    XCTAssertEqual(
      computedGradient.0,
      [
        [
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
        ]
      ])
  }

  func testEmbedding() {
    var layer = Embedding<Float>(vocabularySize: 3, embeddingSize: 5)
    var input = Tensor<Int32>(shape: [2, 3], scalars: [0, 1, 2, 1, 2, 2])
    var output = layer(input)
    let expectedShape = TensorShape([2, 3, 5])
    XCTAssertEqual(output.shape, expectedShape)

    let pretrained = Tensor<Float>(shape: [2, 2], scalars: [0.4, 0.3, 0.2, 0.1])
    layer = Embedding<Float>(embeddings: pretrained)
    input = Tensor<Int32>(shape: [2, 2], scalars: [0, 1, 1, 1])
    output = layer(input)
    let expected = Tensor<Float>([[[0.4, 0.3], [0.2, 0.1]], [[0.2, 0.1], [0.2, 0.1]]])
    XCTAssertEqual(output, expected)
  }

  func testEmbeddingGradient() {
    let embeddings = Tensor<Float>([
      [0.0, 0.2, 0.1],
      [0.1, 0.7, 0.5],
      [0.2, 0.4, 0.6],
      [0.3, 0.2, 0.3],
    ])
    let layer = Embedding<Float>(embeddings: embeddings)
    let indices = Tensor<Int32>(shape: [2, 3], scalars: [0, 1, 2, 1, 2, 2])
    let grad = gradient(at: layer) { $0(indices).sum() }
    // The expected value of the gradient was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // indices = tf.constant([0, 1, 2, 1, 2, 2], dtype=tf.int32)
    // embeddings = tf.constant([
    //      [0.0, 0.2, 0.1],
    //      [0.1, 0.7, 0.5],
    //      [0.2, 0.4, 0.6],
    //      [0.3, 0.2, 0.3]])
    // layer = tf.keras.layers.Embedding(4, 3, weights=[embeddings])
    // with tf.GradientTape() as tape:
    //     tape.watch(layer.weights)
    //     y = tf.reduce_sum(layer(indices))
    // grad_slice = tape.gradient(y, layer.weights)[0]  # IndexedSlice
    // grad = tf.zeros_like(embeddings).numpy()
    // for index in grad_slice.indices:
    //     grad[index] += grad_slice.values[index].numpy()
    // print(grad)
    // ```
    let expected = Tensor<Float>([
      [1, 1, 1],
      [2, 2, 2],
      [3, 3, 3],
      [0, 0, 0],
    ])
    XCTAssertEqual(grad.embeddings, expected)
  }

  func testBasicRNNCell() {
    let weight = Tensor<Float>(ones: [7, 5]) * Tensor<Float>([0.3333, 1, 0.3333, 1, 0.3333])
    let bias = Tensor<Float>(ones: [5])
    var cell = BasicRNNCell<Float>(inputSize: 2, hiddenSize: 5)
    cell.weight = weight
    cell.bias = bias
    let state = Tensor<Float>(ones: [1, 5]) * Tensor<Float>([1, 0.2, 0.5, 2, 0.6])
    let input = Tensor<Float>(ones: [1, 2]) * Tensor<Float>([0.3, 0.7])
    let output = cell(input: input, state: state).state
    let expected = Tensor<Float>([[0.9921227, 0.9999934, 0.9921227, 0.9999934, 0.9921227]])
    assertEqual(output, expected, accuracy: 1e-6)
  }

  func testDense() {
    let weight = Tensor<Float>(shape: [3, 2], scalars: (0..<6).map(Float.init))
    let bias = Tensor<Float>([[1.0, 2.0]])
    let layer = Dense<Float>(weight: weight, bias: bias, activation: identity)
    let input = Tensor<Float>(shape: [1, 3], scalars: (0..<3).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>([[11.0, 15.0]])
    XCTAssertEqual(output, expected)
    XCTAssertFalse(layer.batched)

    let layerNoBias = Dense<Float>(weight: weight, bias: nil, activation: identity)
    let outputNoBias = layerNoBias.inferring(from: input)
    let expectedNoBias = Tensor<Float>([[10.0, 13.0]])
    XCTAssertEqual(outputNoBias, expectedNoBias)
    XCTAssertFalse(layerNoBias.batched)

    let weightBatched = Tensor<Float>(shape: [2, 2, 3], scalars: (0..<12).map(Float.init))
    let biasBatched = Tensor<Float>([[1.0, 2.0, 3.0]])
    let layerBatched = Dense<Float>(
      weight: weightBatched,
      bias: biasBatched,
      activation: identity)
    let inputBatched = Tensor<Float>(shape: [2, 2], scalars: (0..<4).map(Float.init))
    let outputBatched = layerBatched.inferring(from: inputBatched)
    let expectedBatched = Tensor<Float>([[4.0, 6.0, 8.0], [40.0, 46.0, 52.0]])
    XCTAssertEqual(outputBatched, expectedBatched)
    XCTAssertTrue(layerBatched.batched)

    let layerBatchedNoBias = Dense<Float>(
      weight: weightBatched,
      bias: nil,
      activation: identity)
    let outputBatchedNoBias = layerBatchedNoBias.inferring(from: inputBatched)
    let expectedBatchedNoBias = Tensor<Float>([[3.0, 4.0, 5.0], [39.0, 44.0, 49.0]])
    XCTAssertEqual(outputBatchedNoBias, expectedBatchedNoBias)
    XCTAssertTrue(layerBatchedNoBias.batched)
  }

  func testDenseGradient() {
    let weight = Tensor<Float>(shape: [4, 8], scalars: (0..<32).map(Float.init))
    let bias = Tensor<Float>(shape: [1, 8], scalars: (0..<8).map(Float.init))

    // Test `Dense.init` derivative.
    let denseInitPullback = pullback(at: weight) { weight in
      Dense(weight: weight, bias: bias, activation: identity)
    }
    let weightGrad = denseInitPullback(.init(weight: Tensor(100), bias: Tensor(1)))
    XCTAssertEqual(Tensor(100), weightGrad)

    // Test `Dense.callAsFunction` derivative.
    let layer = Dense<Float>(weight: weight, bias: bias, activation: identity)
    let x = Tensor<Float>(shape: [2, 4], scalars: (0..<8).map(Float.init))
    let grad = gradient(at: x, layer) { $1($0).squared().sum() }
    let value = layer(x)
    // The expected values and gradients were computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(8, dtype=tf.float32), [2, 4])
    // kernel_value = np.arange(32.0).reshape([4, 8])
    // bias_value = np.arange(8.0)
    // kernel_initializer = tf.compat.v2.constant_initializer(kernel_value)
    // bias_initializer = tf.compat.v2.constant_initializer(bias_value)
    // layer = tf.keras.layers.Dense(8,
    //                               kernel_initializer=kernel_initializer,
    //                               bias_initializer=bias_initializer)
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = layer(x)
    //     z = tf.math.reduce_sum(tf.math.square(y))
    // print(y, tape.gradient(z, [x] + layer.trainable_variables))
    // ```
    assertEqual(
      value,
      [
        [112.0, 119.0, 126.0, 133.0, 140.0, 147.0, 154.0, 161.0],
        [304.0, 327.0, 350.0, 373.0, 396.0, 419.0, 442.0, 465.0],
      ],
      accuracy: 1e-5)
    assertEqual(
      grad.0,
      [
        [8232.0, 25704.0, 43176.0, 60648.0],
        [23464.0, 72680.0, 121896.0, 171112.0],
      ],
      accuracy: 1e-5)
    assertEqual(
      grad.1.weight,
      [
        [2432.0, 2616.0, 2800.0, 2984.0, 3168.0, 3352.0, 3536.0, 3720.0],
        [3264.0, 3508.0, 3752.0, 3996.0, 4240.0, 4484.0, 4728.0, 4972.0],
        [4096.0, 4400.0, 4704.0, 5008.0, 5312.0, 5616.0, 5920.0, 6224.0],
        [4928.0, 5292.0, 5656.0, 6020.0, 6384.0, 6748.0, 7112.0, 7476.0],
      ],
      accuracy: 1e-5)
    assertEqual(
      grad.1.bias,
      [[832.0, 892.0, 952.0, 1012.0, 1072.0, 1132.0, 1192.0, 1252.0]],
      accuracy: 1e-5)
  }

  // TODO(TF-507): Remove references to `BasicRNNCell.State` after SR-10697 is fixed.
  func testRNN() {
    // Data generated by "Utilities/ReferenceImplementations/rnn.py".
    // BEGIN GENERATED CODE
    let kernel = Tensor<Float>(
        [[-0.3603096,-0.5082418,0.06129855,0.106101096],
        [-0.144324,0.5331736,-0.011734486,0.86278516],
        [0.34075528,-0.64887196,0.36341316,0.28131205],
        [0.12515205,-0.23425382,-0.1376664,0.22526532]]
    )
    let recurrentKernel = Tensor<Float>(
        [[0.34720516,0.6280322,-0.6963875,-0.008289718],
        [-0.057777617,-0.7034996,-0.6661205,0.240902],
        [0.2612954,-0.2549268,-0.08859509,-0.9267621],
        [-0.89879644,0.21372059,-0.25195017,-0.2881138]]
    )
    let bias = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.5279052,0.4159922,-0.02112794,0.6500738],
          [-0.16088444,-0.8408272,0.36293703,-0.23294312],
          [0.14004618,0.69427365,-0.637702,0.077886045],
          [0.3410867,-0.85888004,-0.23541403,0.1133005]]]
    )
    let initialState = Tensor<Float>(
        [[-0.71174276,-0.39879853,0.7789655,-0.1812765]]
    )
    let expectedSum = Tensor<Float>(
        -2.2336872
    )
    let expectedStates = Tensor<Float>(
        [[[0.33346513,-0.052190434,0.53956664,-0.30630472],
          [0.6689981,-0.4655246,-0.004116005,-0.8075811],
          [0.5551112,0.8539695,-0.19158833,0.5159598],
          [-0.40981215,-0.53569716,-0.813944,-0.47423384]]]
    )
    let expectedFinalState = Tensor<Float>(
        [[-0.40981215,-0.53569716,-0.813944,-0.47423384]]
    )
    let expectedGradKernel = Tensor<Float>(
        [[0.41519257,0.5748159,-0.07927984,0.35657609],
        [-0.8111099,-1.2970712,-1.4684451,-1.2794279],
        [-0.30848962,0.060518686,0.65190136,0.2506955],
        [-0.10142271,-0.49896276,-0.24145404,-0.19505012]]
    )
    let expectedGradRecurrentKernel = Tensor<Float>(
        [[0.956356,0.9344623,0.03898424,0.2403388],
        [0.63330317,0.9714962,0.5893948,1.1109966],
        [-0.20000206,-0.56593496,0.32656404,-0.41654968],
        [0.11153888,0.5413084,0.492511,0.9971381]]
    )
    let expectedGradBias = Tensor<Float>(
        [1.2097166,0.09570068,0.3972284,-0.23335484]
    )
    let expectedGradX = Tensor<Float>(
        [[[0.4343156,-0.67581093,0.29182205,0.0771346],
          [-0.20495477,0.11982066,0.19146681,-0.14591387],
          [-0.15502127,-0.69844127,-0.21601494,0.025262251],
          [-0.5592608,0.92486906,0.16155839,0.06524551]]]
    )
    let expectedGradInitialState = Tensor<Float>(
        [[-0.5194068,0.50558126,0.46355626,0.17153597]]
    )
    // END GENERATED CODE
    withTensorLeakChecking {
      var rnn = BasicRNN<Float>(
        BasicRNNCell(inputSize: 4, hiddenSize: 4, seed: (0xFeed, 0xBeef))
      )

      rnn.cell.weight = Tensor(concatenating: [kernel, recurrentKernel])
      rnn.cell.bias = bias

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }

      let initialState = BasicRNNCell<Float>.State(initialState)

      let outputs = rnn(
        inputs,
        initialState: initialState
      )
      assertEqual(
        Tensor(concatenating: outputs.map { $0 }),
        expectedStates,
        accuracy: 1e-6)
      assertEqual(
        outputs.last!,
        expectedFinalState,
        accuracy: 1e-6)

      let (outputSum, (gradRNN, gradInputs, gradInitialState)) =
        valueWithGradient(at: rnn, inputs, initialState) { rnn, inputs, initialState in
          rnn.lastOutput(
            from: inputs,
            initialState: initialState
          ).sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradRNN.cell.weight,
        Tensor(concatenating: [expectedGradKernel, expectedGradRecurrentKernel]),
        accuracy: 1e-6)
      assertEqual(
        gradRNN.cell.bias,
        expectedGradBias,
        accuracy: 1e-6)
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialState,
        expectedGradInitialState,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalBasicRNNSumMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_rnn.py --merge-mode sum".
    // BEGIN GENERATED CODE
    let kernelForward = Tensor<Float>(
        [[-0.3603096,-0.5082418,0.06129855,0.106101096],
        [-0.144324,0.5331736,-0.011734486,0.86278516],
        [0.34075528,-0.64887196,0.36341316,0.28131205],
        [0.12515205,-0.23425382,-0.1376664,0.22526532]]
    )
    let recurrentKernelForward = Tensor<Float>(
        [[0.34720516,0.6280322,-0.6963875,-0.008289718],
        [-0.057777617,-0.7034996,-0.6661205,0.240902],
        [0.2612954,-0.2549268,-0.08859509,-0.9267621],
        [-0.89879644,0.21372059,-0.25195017,-0.2881138]]
    )
    let biasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let kernelBackward = Tensor<Float>(
        [[-0.5279052,0.4159922,-0.02112794,0.6500738],
        [-0.16088444,-0.8408272,0.36293703,-0.23294312],
        [0.14004618,0.69427365,-0.637702,0.077886045],
        [0.3410867,-0.85888004,-0.23541403,0.1133005]]
    )
    let recurrentKernelBackward = Tensor<Float>(
        [[0.67490876,-0.05608359,0.44851846,0.58325285],
        [0.6672334,0.4354152,-0.21524309,-0.56469786],
        [0.027053772,-0.4727077,0.6594652,-0.5838845],
        [-0.31395176,0.7640783,0.5635665,0.003379479]]
    )
    let biasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.8425845,-0.7054219,-0.023461938,0.35520858],
          [-0.24800867,0.09574294,0.3316267,-0.13671225],
          [-0.5058244,-0.6568735,-0.31837505,-0.7905574],
          [-0.50511885,0.50999504,0.74102694,0.12891835]]]
    )
    let initialForwardLayerState = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialBackwardLayerState = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let expectedSum = Tensor<Float>(
      -1.3748394
    )
    let expectedStates = Tensor<Float>(
        [[[0.039731443,0.30118462,-1.1362883,-0.6404417],
          [-0.36877188,0.23013103,0.053439736,0.29040766],
          [-0.7231241,0.4473409,-0.46913218,-0.64710104],
          [0.41570857,-1.500107,-0.20221666,-0.08822429]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those two and test it.
    // let expectedFinalStateForward = Tensor<Float>(
    //     [[0.8125853,-0.5553411,0.27422628,0.73577887]]
    // )
    // let expectedFinalStateBackward = Tensor<Float>(
    //     [[0.7070243,0.29404274,-0.76932997,-0.51012546]]
    // )
    let expectedGradKernelForward = Tensor<Float>(
        [[-0.25060007,0.44844362,0.017417982,-0.39773455],
        [0.08897984,1.4653344,1.0778612,0.13088357],
        [0.23334727,1.066688,0.9485493,0.37866586],
        [0.20101692,0.36850378,0.45986187,0.18827611]]
    )
    let expectedGradRecurrentKernelForward = Tensor<Float>(
        [[-0.016110066,-0.50083566,-0.22681743,-0.17596911],
        [0.24926353,0.33903763,0.51411784,0.3097041],
        [-0.12677985,-0.19154882,-0.1418991,-0.20690772],
        [-0.14268747,-1.8729569,-1.4547974,-0.23180081]]
    )
    let expectedGradBiasForward = Tensor<Float>(
        [0.3045485,0.011022985,0.25799057,0.6699498]
    )
    let expectedGradKernelBackward = Tensor<Float>(
        [[-0.42555702,-0.054258525,-0.39045796,-0.16215265],
        [0.42966515,0.054782312,0.39422727,0.163718],
        [0.624307,0.07959915,0.5728154,0.23788358],
        [0.10861228,0.013848067,0.09965416,0.041385215]]
    )
    let expectedGradRecurrentKernelBackward = Tensor<Float>(
        [[-0.82486856,-0.10517075,-0.7568351,-0.31430483],
        [-0.5736132,-0.07313569,-0.52630275,-0.21856743],
        [0.87171334,0.11114346,0.79981625,0.3321544],
        [-0.8984336,-0.1145503,-0.82433265,-0.3423358]]
    )
    let expectedGradBiasBackward = Tensor<Float>(
        [0.8424889,0.107417345,0.77300215,0.32101882]
    )
    let expectedGradX = Tensor<Float>(
        [[[0.29702687,-0.20684643,0.48203292,0.27731684],
          [-0.36015642,0.51716775,-0.49674046,-0.22718005],
          [0.40323955,-0.47796234,0.2614523,0.24429059],
          [-0.5762627,0.6844691,-0.14327642,-0.093995675]]]
    )
    let expectedGradInitialStateForward = Tensor<Float>(
        [[-0.18421142,0.7250765,0.04656464,-0.31281057]]
    )
    let expectedGradInitialStateBackward = Tensor<Float>(
        [[1.0965196,0.26124585,0.29434562,0.2542974]]
    )
    // END GENERATED CODE
    withTensorLeakChecking {
      var rnn = BidirectionalBasicRNN<Float>(
        BasicRNNCell(inputSize: 4, hiddenSize: 4),
        mergeFunction: BasicRNNCell<Float>.State.sum
      )

      rnn.forward.cell.weight = Tensor(concatenating: [kernelForward, recurrentKernelForward])
      rnn.forward.cell.bias = biasForward
      rnn.backward.cell.weight = Tensor(concatenating: [kernelBackward, recurrentKernelBackward])
      rnn.backward.cell.bias = biasBackward

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }

      let initialForwardLayerState = BasicRNNCell<Float>.State(initialForwardLayerState)
      let initialBackwardLayerState = BasicRNNCell<Float>.State(initialBackwardLayerState)

      let outputs = rnn(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      )
      assertEqual(
        Tensor(concatenating: outputs),
        expectedStates,
        accuracy: 1e-6)

      let (outputSum, (gradRNN)) =
        valueWithGradient(at: rnn) { rnn in
          rnn.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradRNN.forward.cell.weight,
        Tensor(concatenating: [expectedGradKernelForward, expectedGradRecurrentKernelForward]),
        accuracy: 1e-6)
      assertEqual(
        gradRNN.forward.cell.bias,
        expectedGradBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradRNN.backward.cell.weight,
        Tensor(concatenating: [expectedGradKernelBackward, expectedGradRecurrentKernelBackward]),
        accuracy: 1e-6)
      assertEqual(
        gradRNN.backward.cell.bias,
        expectedGradBiasBackward,
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          rnn.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward,
        expectedGradInitialStateForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward,
        expectedGradInitialStateBackward,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalBasicRNNConcatenateMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_rnn.py --merge-mode concat".
    // BEGIN GENERATED CODE
    let kernelForward = Tensor<Float>(
        [[-0.3603096,-0.5082418,0.06129855,0.106101096],
        [-0.144324,0.5331736,-0.011734486,0.86278516],
        [0.34075528,-0.64887196,0.36341316,0.28131205],
        [0.12515205,-0.23425382,-0.1376664,0.22526532]]
    )
    let recurrentKernelForward = Tensor<Float>(
        [[0.34720516,0.6280322,-0.6963875,-0.008289718],
        [-0.057777617,-0.7034996,-0.6661205,0.240902],
        [0.2612954,-0.2549268,-0.08859509,-0.9267621],
        [-0.89879644,0.21372059,-0.25195017,-0.2881138]]
    )
    let biasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let kernelBackward = Tensor<Float>(
        [[-0.5279052,0.4159922,-0.02112794,0.6500738],
        [-0.16088444,-0.8408272,0.36293703,-0.23294312],
        [0.14004618,0.69427365,-0.637702,0.077886045],
        [0.3410867,-0.85888004,-0.23541403,0.1133005]]
    )
    let recurrentKernelBackward = Tensor<Float>(
        [[0.67490876,-0.05608359,0.44851846,0.58325285],
        [0.6672334,0.4354152,-0.21524309,-0.56469786],
        [0.027053772,-0.4727077,0.6594652,-0.5838845],
        [-0.31395176,0.7640783,0.5635665,0.003379479]]
    )
    let biasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.8425845,-0.7054219,-0.023461938,0.35520858],
          [-0.24800867,0.09574294,0.3316267,-0.13671225],
          [-0.5058244,-0.6568735,-0.31837505,-0.7905574],
          [-0.50511885,0.50999504,0.74102694,0.12891835]]]
    )
    let initialForwardLayerState = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialBackwardLayerState = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let expectedSum = Tensor<Float>(
        -1.3748394
    )
    let expectedStates = Tensor<Float>(
        [[[-0.66729283,0.007141886,-0.36695832,-0.13031624,0.7070243,0.29404274,
          -0.76932997,-0.51012546],
          [-0.03940088,-0.34914643,0.57051975,0.46499598,-0.329371,0.57927746,
          -0.51708,-0.17458834],
          [-0.19038993,0.44098023,0.062064223,-0.9267066,-0.53273416,
          0.0063606827,-0.5311964,0.27960557],
          [0.8125853,-0.5553411,0.27422628,0.73577887,-0.39687672,-0.9447659,
          -0.47644293,-0.82400316]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those two and test it.
    // let expectedFinalStateForward = Tensor<Float>(
    //     [[0.8125853,-0.5553411,0.27422628,0.73577887]]
    // )
    // let expectedFinalStateBackward = Tensor<Float>(
    //     [[0.7070243,0.29404274,-0.76932997,-0.51012546]]
    // )
    let expectedGradKernelForward = Tensor<Float>(
        [[-0.25060007,0.44844362,0.017417982,-0.39773455],
        [0.08897984,1.4653344,1.0778612,0.13088357],
        [0.23334727,1.066688,0.9485493,0.37866586],
        [0.20101692,0.36850378,0.45986187,0.18827611]]
    )
    let expectedGradRecurrentKernelForward = Tensor<Float>(
        [[-0.016110066,-0.50083566,-0.22681743,-0.17596911],
        [0.24926353,0.33903763,0.51411784,0.3097041],
        [-0.12677985,-0.19154882,-0.1418991,-0.20690772],
        [-0.14268747,-1.8729569,-1.4547974,-0.23180081]]
    )
    let expectedGradBiasForward = Tensor<Float>(
        [0.3045485,0.011022985,0.25799057,0.6699498]
    )
    let expectedGradKernelBackward = Tensor<Float>(
        [[-0.42555702,-0.054258525,-0.39045796,-0.16215265],
        [0.42966515,0.054782312,0.39422727,0.163718],
        [0.624307,0.07959915,0.5728154,0.23788358],
        [0.10861228,0.013848067,0.09965416,0.041385215]]
    )
    let expectedGradRecurrentKernelBackward = Tensor<Float>(
        [[-0.82486856,-0.10517075,-0.7568351,-0.31430483],
        [-0.5736132,-0.07313569,-0.52630275,-0.21856743],
        [0.87171334,0.11114346,0.79981625,0.3321544],
        [-0.8984336,-0.1145503,-0.82433265,-0.3423358]]
    )
    let expectedGradBiasBackward = Tensor<Float>(
        [0.8424889,0.107417345,0.77300215,0.32101882]
    )
    let expectedGradX = Tensor<Float>(
        [[[0.29702687,-0.20684643,0.48203292,0.27731684],
          [-0.36015642,0.51716775,-0.49674046,-0.22718005],
          [0.40323955,-0.47796234,0.2614523,0.24429059],
          [-0.5762627,0.6844691,-0.14327642,-0.093995675]]]
    )
    let expectedGradInitialStateForward = Tensor<Float>(
        [[-0.18421142,0.7250765,0.04656464,-0.31281057]]
    )
    let expectedGradInitialStateBackward = Tensor<Float>(
        [[1.0965196,0.26124585,0.29434562,0.2542974]]
    )
    // END GENERATED CODE
    withTensorLeakChecking {
      var rnn = BidirectionalBasicRNN<Float>(
        BasicRNNCell(inputSize: 4, hiddenSize: 4)
      )

      rnn.forward.cell.weight = Tensor(concatenating: [kernelForward, recurrentKernelForward])
      rnn.forward.cell.bias = biasForward
      rnn.backward.cell.weight = Tensor(concatenating: [kernelBackward, recurrentKernelBackward])
      rnn.backward.cell.bias = biasBackward

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }

      let initialForwardLayerState = BasicRNNCell<Float>.State(initialForwardLayerState)
      let initialBackwardLayerState = BasicRNNCell<Float>.State(initialBackwardLayerState)

      let outputs = rnn(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      )
      assertEqual(
        Tensor(concatenating: outputs),
        expectedStates,
        accuracy: 1e-6)

      let (outputSum, (gradRNN)) =
        valueWithGradient(at: rnn) { rnn in
          rnn.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradRNN.forward.cell.weight,
        Tensor(concatenating: [expectedGradKernelForward, expectedGradRecurrentKernelForward]),
        accuracy: 1e-6)
      assertEqual(
        gradRNN.forward.cell.bias,
        expectedGradBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradRNN.backward.cell.weight,
        Tensor(concatenating: [expectedGradKernelBackward, expectedGradRecurrentKernelBackward]),
        accuracy: 1e-6)
      assertEqual(
        gradRNN.backward.cell.bias,
        expectedGradBiasBackward,
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          rnn.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward,
        expectedGradInitialStateForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward,
        expectedGradInitialStateBackward,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalBasicRNNAverageMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_rnn.py --merge-mode ave".
    // BEGIN GENERATED CODE
    let kernelForward = Tensor<Float>(
        [[-0.3603096,-0.5082418,0.06129855,0.106101096],
        [-0.144324,0.5331736,-0.011734486,0.86278516],
        [0.34075528,-0.64887196,0.36341316,0.28131205],
        [0.12515205,-0.23425382,-0.1376664,0.22526532]]
    )
    let recurrentKernelForward = Tensor<Float>(
        [[0.34720516,0.6280322,-0.6963875,-0.008289718],
        [-0.057777617,-0.7034996,-0.6661205,0.240902],
        [0.2612954,-0.2549268,-0.08859509,-0.9267621],
        [-0.89879644,0.21372059,-0.25195017,-0.2881138]]
    )
    let biasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let kernelBackward = Tensor<Float>(
        [[-0.5279052,0.4159922,-0.02112794,0.6500738],
        [-0.16088444,-0.8408272,0.36293703,-0.23294312],
        [0.14004618,0.69427365,-0.637702,0.077886045],
        [0.3410867,-0.85888004,-0.23541403,0.1133005]]
    )
    let recurrentKernelBackward = Tensor<Float>(
        [[0.67490876,-0.05608359,0.44851846,0.58325285],
        [0.6672334,0.4354152,-0.21524309,-0.56469786],
        [0.027053772,-0.4727077,0.6594652,-0.5838845],
        [-0.31395176,0.7640783,0.5635665,0.003379479]]
    )
    let biasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.8425845,-0.7054219,-0.023461938,0.35520858],
          [-0.24800867,0.09574294,0.3316267,-0.13671225],
          [-0.5058244,-0.6568735,-0.31837505,-0.7905574],
          [-0.50511885,0.50999504,0.74102694,0.12891835]]]
    )
    let initialForwardLayerState = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialBackwardLayerState = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let expectedSum = Tensor<Float>(
        -0.6874197
    )
    let expectedStates = Tensor<Float>(
        [[[0.019865721,0.15059231,-0.56814414,-0.32022086],
          [-0.18438594,0.115065515,0.026719868,0.14520383],
          [-0.36156204,0.22367045,-0.23456609,-0.32355052],
          [0.20785429,-0.7500535,-0.10110833,-0.044112146]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those two and test it.
    // let expectedFinalStateForward = Tensor<Float>(
    //     [[0.8125853,-0.5553411,0.27422628,0.73577887]]
    // )
    // let expectedFinalStateBackward = Tensor<Float>(
    //     [[0.7070243,0.29404274,-0.76932997,-0.51012546]]
    // )
    let expectedGradKernelForward = Tensor<Float>(
        [[-0.12530003,0.22422181,0.008708991,-0.19886728],
        [0.04448992,0.7326672,0.5389306,0.06544179],
        [0.11667363,0.533344,0.47427464,0.18933293],
        [0.10050846,0.18425189,0.22993094,0.094138056]]
    )
    let expectedGradRecurrentKernelForward = Tensor<Float>(
        [[-0.008055033,-0.25041783,-0.113408715,-0.087984554],
        [0.12463176,0.16951881,0.25705892,0.15485205],
        [-0.06338993,-0.09577441,-0.07094955,-0.10345386],
        [-0.071343735,-0.93647844,-0.7273987,-0.115900405]]
    )
    let expectedGradBiasForward = Tensor<Float>(
        [0.15227425,0.0055114925,0.12899528,0.3349749]
    )
    let expectedGradKernelBackward = Tensor<Float>(
        [[-0.21277851,-0.027129263,-0.19522898,-0.081076324],
        [0.21483257,0.027391156,0.19711363,0.081859],
        [0.3121535,0.039799575,0.2864077,0.11894179],
        [0.05430614,0.0069240336,0.04982708,0.020692607]]
    )
    let expectedGradRecurrentKernelBackward = Tensor<Float>(
        [[-0.41243428,-0.052585375,-0.37841755,-0.15715241],
        [-0.2868066,-0.036567844,-0.26315138,-0.109283715],
        [0.43585667,0.05557173,0.39990813,0.1660772],
        [-0.4492168,-0.05727515,-0.41216633,-0.1711679]]
    )
    let expectedGradBiasBackward = Tensor<Float>(
        [0.42124444,0.053708673,0.38650107,0.16050941]
    )
    let expectedGradX = Tensor<Float>(
        [[[0.14851344,-0.103423215,0.24101646,0.13865842],
          [-0.18007821,0.25858387,-0.24837023,-0.113590024],
          [0.20161977,-0.23898117,0.13072614,0.122145295],
          [-0.28813136,0.34223455,-0.07163821,-0.046997838]]]
    )
    let expectedGradInitialStateForward = Tensor<Float>(
        [[-0.09210571,0.36253825,0.02328232,-0.15640529]]
    )
    let expectedGradInitialStateBackward = Tensor<Float>(
        [[0.5482598,0.13062292,0.14717281,0.1271487]]
    )
    // END GENERATED CODE
    withTensorLeakChecking {
      var rnn = BidirectionalBasicRNN<Float>(
        BasicRNNCell(inputSize: 4, hiddenSize: 4),
        mergeFunction: BasicRNNCell<Float>.State.average
      )

      rnn.forward.cell.weight = Tensor(concatenating: [kernelForward, recurrentKernelForward])
      rnn.forward.cell.bias = biasForward
      rnn.backward.cell.weight = Tensor(concatenating: [kernelBackward, recurrentKernelBackward])
      rnn.backward.cell.bias = biasBackward

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }

      let initialForwardLayerState = BasicRNNCell<Float>.State(initialForwardLayerState)
      let initialBackwardLayerState = BasicRNNCell<Float>.State(initialBackwardLayerState)

      let outputs = rnn(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      )
      assertEqual(
        Tensor(concatenating: outputs),
        expectedStates,
        accuracy: 1e-6)

      let (outputSum, (gradRNN)) =
        valueWithGradient(at: rnn) { rnn in
          rnn.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradRNN.forward.cell.weight,
        Tensor(concatenating: [expectedGradKernelForward, expectedGradRecurrentKernelForward]),
        accuracy: 1e-6)
      assertEqual(
        gradRNN.forward.cell.bias,
        expectedGradBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradRNN.backward.cell.weight,
        Tensor(concatenating: [expectedGradKernelBackward, expectedGradRecurrentKernelBackward]),
        accuracy: 1e-6)
      assertEqual(
        gradRNN.backward.cell.bias,
        expectedGradBiasBackward,
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          rnn.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward,
        expectedGradInitialStateForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward,
        expectedGradInitialStateBackward,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalBasicRNNMultiplyMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_rnn.py --merge-mode mul".
    // BEGIN GENERATED CODE
    let kernelForward = Tensor<Float>(
        [[-0.3603096,-0.5082418,0.06129855,0.106101096],
        [-0.144324,0.5331736,-0.011734486,0.86278516],
        [0.34075528,-0.64887196,0.36341316,0.28131205],
        [0.12515205,-0.23425382,-0.1376664,0.22526532]]
    )
    let recurrentKernelForward = Tensor<Float>(
        [[0.34720516,0.6280322,-0.6963875,-0.008289718],
        [-0.057777617,-0.7034996,-0.6661205,0.240902],
        [0.2612954,-0.2549268,-0.08859509,-0.9267621],
        [-0.89879644,0.21372059,-0.25195017,-0.2881138]]
    )
    let biasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let kernelBackward = Tensor<Float>(
        [[-0.5279052,0.4159922,-0.02112794,0.6500738],
        [-0.16088444,-0.8408272,0.36293703,-0.23294312],
        [0.14004618,0.69427365,-0.637702,0.077886045],
        [0.3410867,-0.85888004,-0.23541403,0.1133005]]
    )
    let recurrentKernelBackward = Tensor<Float>(
        [[0.67490876,-0.05608359,0.44851846,0.58325285],
        [0.6672334,0.4354152,-0.21524309,-0.56469786],
        [0.027053772,-0.4727077,0.6594652,-0.5838845],
        [-0.31395176,0.7640783,0.5635665,0.003379479]]
    )
    let biasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.8425845,-0.7054219,-0.023461938,0.35520858],
          [-0.24800867,0.09574294,0.3316267,-0.13671225],
          [-0.5058244,-0.6568735,-0.31837505,-0.7905574],
          [-0.50511885,0.50999504,0.74102694,0.12891835]]]
    )
    let initialForwardLayerState = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialBackwardLayerState = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let expectedSum = Tensor<Float>(
        -0.5347661
    )
    let expectedStates = Tensor<Float>(
        [[[-0.47179222,0.0021000197,0.28231204,0.066477634],
          [0.012977507,-0.20225266,-0.29500437,-0.081182875],
          [0.10142722,0.0028049352,-0.032968294,-0.25911233],
          [-0.32249618,0.5246674,-0.13065317,-0.60628414]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those two and test it.
    // let expectedFinalStateForward = Tensor<Float>(
    //     [[0.8125853,-0.5553411,0.27422628,0.73577887]]
    // )
    // let expectedFinalStateBackward = Tensor<Float>(
    //     [[0.7070243,0.29404274,-0.76932997,-0.51012546]]
    // )
    let expectedGradKernelForward = Tensor<Float>(
        [[0.29835904,-0.271358,-0.056388527,0.19803399],
        [0.13539863,-1.1518296,-0.62982273,-0.16866776],
        [-0.07457287,-0.876209,-0.54846895,-0.26091105],
        [0.045224264,-0.22196792,-0.41959473,-0.09994551]]
    )
    let expectedGradRecurrentKernelForward = Tensor<Float>(
        [[0.11663125,0.39018548,0.1591308,0.027596578],
        [-0.071245454,-0.27228498,-0.3515284,-0.19403662],
        [0.06461461,0.05803746,0.27710906,0.00072359666],
        [-0.108036324,1.5429575,0.7437697,0.30051786]]
    )
    let expectedGradBiasForward = Tensor<Float>(
        [-0.51533103,-0.16392553,-0.019812264,-0.31751463]
    )
    let expectedGradKernelBackward = Tensor<Float>(
        [[-0.34580138,0.030131992,-0.10707383,-0.119308494],
        [0.3491396,-0.030422872,0.10810747,0.12046025],
        [0.50730264,-0.044204682,0.15708104,0.17502971],
        [0.08825674,-0.0076904013,0.027327789,0.030450366]]
    )
    let expectedGradRecurrentKernelBackward = Tensor<Float>(
        [[-0.67027605,0.058405645,-0.20754407,-0.23125885],
        [-0.46610966,0.040615257,-0.14432605,-0.1608173],
        [0.7083415,-0.06172254,0.21933062,0.24439217],
        [-0.73005396,0.063614495,-0.22605367,-0.25188342]]
    )
    let expectedGradBiasBackward = Tensor<Float>(
        [0.6845941,-0.05965327,0.2119775,0.23619886]
    )
    let expectedGradX = Tensor<Float>(
        [[[-0.23223981,0.2862338,-0.41787457,-0.17583233],
          [0.34112248,-0.2498242,0.34255287,0.17837948],
          [-0.18834808,0.32665232,-0.20202757,-0.20915407],
          [0.07640687,-0.68787134,0.049270652,0.37331474]]]
    )
    let expectedGradInitialStateForward = Tensor<Float>(
        [[0.25459114,-0.45141447,-0.14997824,0.26960957]]
    )
    let expectedGradInitialStateBackward = Tensor<Float>(
        [[0.6982236,0.25180238,0.048598364,-0.14024764]]
    )
    // END GENERATED CODE
    withTensorLeakChecking {
      var rnn = BidirectionalBasicRNN<Float>(
        BasicRNNCell(inputSize: 4, hiddenSize: 4),
        mergeFunction: BasicRNNCell<Float>.State.multiply
      )

      rnn.forward.cell.weight = Tensor(concatenating: [kernelForward, recurrentKernelForward])
      rnn.forward.cell.bias = biasForward
      rnn.backward.cell.weight = Tensor(concatenating: [kernelBackward, recurrentKernelBackward])
      rnn.backward.cell.bias = biasBackward

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }

      let initialForwardLayerState = BasicRNNCell<Float>.State(initialForwardLayerState)
      let initialBackwardLayerState = BasicRNNCell<Float>.State(initialBackwardLayerState)

      let outputs = rnn(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      )
      assertEqual(
        Tensor(concatenating: outputs),
        expectedStates,
        accuracy: 1e-6)

      let (outputSum, (gradRNN)) =
        valueWithGradient(at: rnn) { rnn in
          rnn.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradRNN.forward.cell.weight,
        Tensor(concatenating: [expectedGradKernelForward, expectedGradRecurrentKernelForward]),
        accuracy: 1e-6)
      assertEqual(
        gradRNN.forward.cell.bias,
        expectedGradBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradRNN.backward.cell.weight,
        Tensor(concatenating: [expectedGradKernelBackward, expectedGradRecurrentKernelBackward]),
        accuracy: 1e-6)
      assertEqual(
        gradRNN.backward.cell.bias,
        expectedGradBiasBackward,
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          rnn.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward,
        expectedGradInitialStateForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward,
        expectedGradInitialStateBackward,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalBasicRNNStackMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_rnn.py --merge-mode none".
    // BEGIN GENERATED CODE
    let kernelForward = Tensor<Float>(
        [[-0.3603096,-0.5082418,0.06129855,0.106101096],
        [-0.144324,0.5331736,-0.011734486,0.86278516],
        [0.34075528,-0.64887196,0.36341316,0.28131205],
        [0.12515205,-0.23425382,-0.1376664,0.22526532]]
    )
    let recurrentKernelForward = Tensor<Float>(
        [[0.34720516,0.6280322,-0.6963875,-0.008289718],
        [-0.057777617,-0.7034996,-0.6661205,0.240902],
        [0.2612954,-0.2549268,-0.08859509,-0.9267621],
        [-0.89879644,0.21372059,-0.25195017,-0.2881138]]
    )
    let biasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let kernelBackward = Tensor<Float>(
        [[-0.5279052,0.4159922,-0.02112794,0.6500738],
        [-0.16088444,-0.8408272,0.36293703,-0.23294312],
        [0.14004618,0.69427365,-0.637702,0.077886045],
        [0.3410867,-0.85888004,-0.23541403,0.1133005]]
    )
    let recurrentKernelBackward = Tensor<Float>(
        [[0.67490876,-0.05608359,0.44851846,0.58325285],
        [0.6672334,0.4354152,-0.21524309,-0.56469786],
        [0.027053772,-0.4727077,0.6594652,-0.5838845],
        [-0.31395176,0.7640783,0.5635665,0.003379479]]
    )
    let biasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.8425845,-0.7054219,-0.023461938,0.35520858],
          [-0.24800867,0.09574294,0.3316267,-0.13671225],
          [-0.5058244,-0.6568735,-0.31837505,-0.7905574],
          [-0.50511885,0.50999504,0.74102694,0.12891835]]]
    )
    let initialForwardLayerState = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialBackwardLayerState = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let expectedSum = Tensor<Float>(
        -1.3748394
    )
    let expectedStatesForward = Tensor<Float>(
        [[[-0.66729283,0.007141886,-0.36695832,-0.13031624],
          [-0.03940088,-0.34914643,0.57051975,0.46499598],
          [-0.19038993,0.44098023,0.062064223,-0.9267066],
          [0.8125853,-0.5553411,0.27422628,0.73577887]]]
    )
    let expectedStatesBackward = Tensor<Float>(
        [[[0.7070243,0.29404274,-0.76932997,-0.51012546],
          [-0.329371,0.57927746,-0.51708,-0.17458834],
          [-0.53273416,0.0063606827,-0.5311964,0.27960557],
          [-0.39687672,-0.9447659,-0.47644293,-0.82400316]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those two and test it.
    // let expectedFinalStateForward = Tensor<Float>(
    //     [[0.8125853,-0.5553411,0.27422628,0.73577887]]
    // )
    // let expectedFinalStateBackward = Tensor<Float>(
    //     [[0.7070243,0.29404274,-0.76932997,-0.51012546]]
    // )
    let expectedGradKernelForward = Tensor<Float>(
        [[-0.25060007,0.44844362,0.017417982,-0.39773455],
        [0.08897984,1.4653344,1.0778612,0.13088357],
        [0.23334727,1.066688,0.9485493,0.37866586],
        [0.20101692,0.36850378,0.45986187,0.18827611]]
    )
    let expectedGradRecurrentKernelForward = Tensor<Float>(
        [[-0.016110066,-0.50083566,-0.22681743,-0.17596911],
        [0.24926353,0.33903763,0.51411784,0.3097041],
        [-0.12677985,-0.19154882,-0.1418991,-0.20690772],
        [-0.14268747,-1.8729569,-1.4547974,-0.23180081]]
    )
    let expectedGradBiasForward = Tensor<Float>(
        [0.3045485,0.011022985,0.25799057,0.6699498]
    )
    let expectedGradKernelBackward = Tensor<Float>(
        [[-0.42555702,-0.054258525,-0.39045796,-0.16215265],
        [0.42966515,0.054782312,0.39422727,0.163718],
        [0.624307,0.07959915,0.5728154,0.23788358],
        [0.10861228,0.013848067,0.09965416,0.041385215]]
    )
    let expectedGradRecurrentKernelBackward = Tensor<Float>(
        [[-0.82486856,-0.10517075,-0.7568351,-0.31430483],
        [-0.5736132,-0.07313569,-0.52630275,-0.21856743],
        [0.87171334,0.11114346,0.79981625,0.3321544],
        [-0.8984336,-0.1145503,-0.82433265,-0.3423358]]
    )
    let expectedGradBiasBackward = Tensor<Float>(
        [0.8424889,0.107417345,0.77300215,0.32101882]
    )
    let expectedGradX = Tensor<Float>(
        [[[0.29702687,-0.20684643,0.48203292,0.27731684],
          [-0.36015642,0.51716775,-0.49674046,-0.22718005],
          [0.40323955,-0.47796234,0.2614523,0.24429059],
          [-0.5762627,0.6844691,-0.14327642,-0.093995675]]]
    )
    let expectedGradInitialStateForward = Tensor<Float>(
        [[-0.18421142,0.7250765,0.04656464,-0.31281057]]
    )
    let expectedGradInitialStateBackward = Tensor<Float>(
        [[1.0965196,0.26124585,0.29434562,0.2542974]]
    )
    // END GENERATED CODE
    withTensorLeakChecking {
      var rnn = BidirectionalBasicRNN<Float>(
        BasicRNNCell(inputSize: 4, hiddenSize: 4),
        mergeFunction: BasicRNNCell<Float>.State.stack
      )

      rnn.forward.cell.weight = Tensor(concatenating: [kernelForward, recurrentKernelForward])
      rnn.forward.cell.bias = biasForward
      rnn.backward.cell.weight = Tensor(concatenating: [kernelBackward, recurrentKernelBackward])
      rnn.backward.cell.bias = biasBackward

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }

      let initialForwardLayerState = BasicRNNCell<Float>.State(initialForwardLayerState)
      let initialBackwardLayerState = BasicRNNCell<Float>.State(initialBackwardLayerState)

      let outputs = rnn(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      ).map { $0.unstacked(alongAxis: 0) }
      assertEqual(
        Tensor(concatenating: outputs.map { $0[0] }),
        expectedStatesForward,
        accuracy: 1e-6)
      assertEqual(
        Tensor(concatenating: outputs.map { $0[1] }),
        expectedStatesBackward,
        accuracy: 1e-6)

      let (outputSum, (gradRNN)) =
        valueWithGradient(at: rnn) { rnn in
          rnn.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradRNN.forward.cell.weight,
        Tensor(concatenating: [expectedGradKernelForward, expectedGradRecurrentKernelForward]),
        accuracy: 1e-6)
      assertEqual(
        gradRNN.forward.cell.bias,
        expectedGradBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradRNN.backward.cell.weight,
        Tensor(concatenating: [expectedGradKernelBackward, expectedGradRecurrentKernelBackward]),
        accuracy: 1e-6)
      assertEqual(
        gradRNN.backward.cell.bias,
        expectedGradBiasBackward,
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          rnn.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward,
        expectedGradInitialStateForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward,
        expectedGradInitialStateBackward,
        accuracy: 1e-6)
    }
  }

  func testLSTM() {
    // Data generated by "Utilities/ReferenceImplementations/lstm.py".
    // BEGIN GENERATED CODE
    let kernel = Tensor<Float>(
      [
        [
          0.4545238, 0.17410219, 0.07151973, -0.043370485, -0.08141422, -0.23986903,
          0.2063713, 0.17585361, 0.4534099, -0.5096329, 0.2199418, -0.2149244,
          0.23440659, 0.43826634, -0.13891399, -0.17842606,
        ],
        [
          -0.3797379, 0.079282284, -0.10015741, -0.3239718, -0.27132228, 0.39218837,
          -0.43364745, -0.34852988, 0.2132551, -0.5461597, -0.048002183, 0.26265675,
          -0.18781787, -0.41067505, -0.051611483, 0.4885484,
        ],
        [
          -0.06933203, 0.54473567, -0.21874839, -0.49106207, -0.071650594, -0.08455315,
          0.5346174, 0.057993293, -0.13344201, -0.48908734, -0.058430463, -0.15182033,
          0.03391558, 0.009597003, 0.273346, -0.49635035,
        ],
        [
          -0.07164055, 0.37893647, -0.108646095, 0.07596207, 0.31213146, -0.3623113,
          -0.20532963, -0.302337, -0.26968825, 0.4479969, -0.085390985, 0.2579115,
          0.2743805, -0.21505526, -0.31751555, -0.44200268,
        ],
      ]
    )
    let recurrentKernel = Tensor<Float>(
      [
        [
          0.001444459, -0.44615534, 0.17120321, 0.028076064, -0.36675194, -0.37769908,
          -0.038331404, -0.3308463, 0.18919824, 0.21540813, 0.21875387, -0.17696984,
          0.24726436, -0.2989273, 0.26229933, 0.045673575,
        ],
        [
          -0.0060651056, -0.010614913, -0.41048288, -0.16720156, -0.042991478,
          -0.034523666, -0.22591552, -0.46458426, -0.15950868, 0.0032770582,
          0.02271657, 0.26582226, -0.38004795, 0.21254668, 0.35087258, 0.35231543,
        ],
        [
          -0.2352461, -0.12130469, -0.19648746, 0.29172993, 0.10191456, -0.2919168,
          -0.1870388, 0.2183716, 0.34960067, -0.24624759, 0.3270614, 0.4467347,
          0.11563631, 0.2551108, 0.06388308, -0.2366966,
        ],
        [
          -0.36797202, 0.21800567, -0.06400205, 0.43761855, 0.1086247, 0.3239568,
          -0.23723324, -0.07836083, -0.10523385, 0.34244883, 0.11385839, -0.15765572,
          0.42096126, 0.08826415, 0.10015068, 0.28478572,
        ],
      ]
    )
    let bias = Tensor<Float>(
      [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    let x = Tensor<Float>(
      [
        [
          [-0.6346513, -0.43788078, -0.40050998, -0.35219777],
          [-0.59232813, -0.728384, -0.22974151, 0.3288936],
          [0.5766824, -0.5468713, 0.16438323, 0.6741958],
          [-0.47952235, -0.24399251, 0.8553043, 0.8588342],
        ]
      ]
    )
    let initialStateHidden = Tensor<Float>(
      [[0.9577427, -0.7377565, 0.35953724, -0.5784317]]
    )
    let initialStateCell = Tensor<Float>(
      [[-0.905442, -0.95177335, -0.05345559, -1.0015926]]
    )
    let expectedStates = Tensor<Float>(
      [
        [
          [-0.24107721, -0.17449945, 0.037506927, -0.30990827],
          [-0.26881245, -0.00634476, -0.019400658, -0.2071909],
          [-0.2842662, 0.032890536, -0.015112571, -0.13682269],
          [-0.31182146, 0.08775673, -0.072591506, -0.07358349],
        ]
      ]
    )
    let expectedOutput = Tensor<Float>(
      [[-0.7036462, 0.19863679, -0.14904207, -0.25543863]]
    )
    let expectedGradKernel = Tensor<Float>(
      [
        [
          0.0805003, -0.08275342, 0.03522955, -0.030312482, 0.11720966, 0.07540316,
          -0.0023639323, 0.10740622, -0.2532557, -0.16903767, -0.5353516, -0.3493193,
          0.016285714, 0.008904671, -0.003577555, -0.006157878,
        ],
        [
          0.08940101, -0.095668815, 0.032129567, -0.013392258, 0.17761931, 0.071601115,
          -0.000029361108, 0.17159097, -0.78527546, -0.5366851, -0.898436, -0.68411684,
          0.05064467, 0.0044404613, 0.0010539321, 0.022477672,
        ],
        [
          -0.124524735, 0.014374042, -0.065310925, 0.04207304, -0.018112179,
          0.054847445, -0.0053864224, -0.044751726, 0.12626463, 0.57195514, 0.0789956,
          0.21590737, 0.010621901, 0.0052103405, -0.0023645863, -0.00021189498,
        ],
        [
          -0.1812453, 0.07054679, -0.08742681, 0.046577554, -0.10540785, 0.021182634,
          -0.002338124, -0.14174302, 0.6233163, 0.88522494, 0.5829264, 0.6225763,
          -0.026014743, 0.0054547004, -0.0054682456, -0.018445207,
        ],
      ]
    )
    let expectedGradRecurrentKernel = Tensor<Float>(
      [
        [
          0.109804794, -0.006917909, 0.06479619, -0.02927, -0.07282017, -0.06949664,
          -0.007851637, -0.0064141527, 0.01585731, -0.27055892, 0.18191284,
          -0.036890596, 0.00081066974, -0.012041213, 0.008538903, 0.004745292,
        ],
        [
          -0.022427287, -0.037062988, -0.021642044, 0.009100201, 0.113654904,
          0.07015832, 0.00417687, 0.073882714, -0.31279865, -0.057589047, -0.4670751,
          -0.2540243, 0.019137694, 0.008957855, -0.004623856, 0.005064313,
        ],
        [
          0.015348908, 0.013859384, 0.012477149, -0.0050131185, -0.050523866,
          -0.031698443, -0.0024695867, -0.0299116, 0.12353469, 0.012106655, 0.19668634,
          0.0989304, -0.0068828184, -0.0043967483, 0.0024977797, -0.001459891,
        ],
        [
          0.024811182, -0.05493941, 0.0018467754, -0.001980858, 0.1320867, 0.06232968,
          0.0034385736, 0.10646886, -0.46687102, -0.2663457, -0.5953469, -0.4029197,
          0.02738668, 0.0066831633, -0.0024504992, 0.010666027,
        ],
      ]
    )
    let expectedGradBias = Tensor<Float>(
      [
        -0.2057409, 0.18275242, -0.07681093, 0.042995583, -0.3810982, -0.12361911,
        -0.008955841, -0.3550135, 1.5545796, 1.3119538, 1.8124579, 1.4128002,
        -0.08343395, -0.010454425, 0.0008502216, -0.038896702,
      ]
    )
    let expectedGradX = Tensor<Float>(
      [
        [
          [0.21591075, 0.11293747, -0.13014226, -0.022586256],
          [0.08480768, 0.13718912, -0.12702335, 0.077232406],
          [0.01666388, 0.0068455637, -0.35854548, 0.1287557],
          [-0.2761167, -0.088074416, -0.415294, 0.32159615],
        ]
      ]
    )
    let expectedGradInitialStateHidden = Tensor<Float>(
      [[0.25400645, 0.07120966, 0.36865664, -0.05423181]]
    )
    let expectedGradInitialStateCell = Tensor<Float>(
      [[0.48873883, 0.23135301, 0.6608742, 0.37556332]]
    )
    // END GENERATED CODE

    // Swaps the columns corresponding to "forget" and "update" gates.
    //
    // This translates between the keras ordering
    //   [input | forget | update | output]
    // and the swift-apis ordering
    //   [input | update | forget | output]
    func swapForgetUpdate(_ t: Tensor<Float>) -> Tensor<Float> {
      var split = t.split(count: 4, alongAxis: -1)
      split.swapAt(1, 2)
      return Tensor(concatenating: split, alongAxis: -1)
    }

    withTensorLeakChecking {
      var lstm = LSTM<Float>(LSTMCell(inputSize: 4, hiddenSize: 4))
      lstm.cell.fusedWeight =
        swapForgetUpdate(Tensor(concatenating: [kernel, recurrentKernel]))
      lstm.cell.fusedBias = swapForgetUpdate(bias)

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }
      let initialState = LSTMCell<Float>.State(
        cell: initialStateCell,
        hidden: initialStateHidden)

      let outputs = lstm(inputs, initialState: initialState)
      assertEqual(
        Tensor(concatenating: outputs.map { $0.hidden }),
        expectedStates,
        accuracy: 1e-6)
      assertEqual(
        outputs.last!.cell,
        expectedOutput,
        accuracy: 1e-6)

      let (gradLSTM, gradInputs, gradInitialState) =
        gradient(at: lstm, inputs, initialState) { lstm, inputs, initialState in
          lstm.lastOutput(from: inputs, initialState: initialState).cell.sum()
        }
      assertEqual(
        gradLSTM.cell.fusedWeight,
        swapForgetUpdate(
          Tensor(concatenating: [expectedGradKernel, expectedGradRecurrentKernel])),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.cell.fusedBias,
        swapForgetUpdate(expectedGradBias),
        accuracy: 1e-6)
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialState.cell,
        expectedGradInitialStateCell,
        accuracy: 1e-6)
      assertEqual(
        gradInitialState.hidden,
        expectedGradInitialStateHidden,
        accuracy: 1e-6)

    }
  }

  func testBidirectionalLSTMSumMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_lstm.py --merge-mode sum".
    // BEGIN GENERATED CODE
    let kernelForward = Tensor<Float>(
        [[-0.23379976,-0.32979083,0.03977573,0.06884748,-0.093649775,0.34596872,
          -0.0076143146,0.55984896,0.22111118,-0.4210437,0.23581362,0.1825394,
          0.08120936,-0.15200394,-0.08932975,0.14617139],
        [0.4650857,0.18167573,0.37479192,-0.46759078,-0.24752906,-0.5445049,
          0.2544219,0.29843968,0.20215267,0.03678304,0.28829694,-0.5086538,
          -0.5053389,0.2829352,-0.36783904,-0.2113666],
        [-0.23447555,-0.44898528,-0.3757536,0.3030774,0.096295714,0.541732,
          0.46623164,-0.4027334,-0.45229727,0.119446635,-0.36198467,0.020288646,
          0.5418573,-0.3643924,-0.5106659,0.10962713]]
    )
    let recurrentKernelForward = Tensor<Float>(
        [[0.22804272,-0.03794806,0.17161758,-0.5903252,-0.32132986,-0.028507836,
          -0.02729048,0.15502541,-0.29579595,0.041225288,-0.04219827,-0.22448958,
          0.51597697,-0.103077516,0.005849608,0.14667785],
        [0.09507164,-0.25853005,-0.22325921,0.499246,-0.3109755,0.01945003,
          0.33349884,-0.030728217,-0.2997637,0.4836203,0.035094753,-0.1599233,
          0.16274704,-0.0034662904,-0.18398553,-0.06985295],
        [-0.13635698,-0.51163936,-0.09270489,-0.12567669,0.41930002,-0.48658654,
          -0.12225373,-0.28107005,-0.004303295,0.16249746,0.024185713,-0.24801093,
          0.019665923,0.040028445,0.038016353,0.31107086],
        [0.14521983,0.63589036,0.04985115,0.17409262,0.18315184,-0.341375,
          -0.048293047,-0.037892636,0.08461128,0.19565175,-0.18345755,-0.12093313,
          0.27880925,0.39949074,-0.1981584,0.13265577]]
    )
    let biasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    )
    let kernelBackward = Tensor<Float>(
        [[-0.3425501,0.26993138,-0.013709605,0.42182362,-0.10439563,-0.5456007,
          0.23550463,-0.15115345,0.09087402,0.45050424,-0.41379565,0.050539136,
          0.22132623,-0.55731493,-0.15275678,0.07351905],
        [-0.20979714,0.1675657,0.5581859,-0.50259715,0.4392975,-0.26118547,
          -0.06672484,0.26642138,-0.37113252,-0.29193833,0.5320031,0.54281443,
          -0.5196755,-0.013125479,0.006757021,0.5087437],
        [0.41886348,0.16300112,0.15921205,0.3881111,0.34322268,0.0014763474,
          0.2432934,0.32898188,0.26853347,-0.48425156,-0.3587765,-0.19107765,
          0.5605115,-0.18283507,0.13686264,0.008577526]]
    )
    let recurrentKernelBackward = Tensor<Float>(
        [[0.332492,0.3287107,0.013327966,-0.15466748,0.18726069,-0.18434268,
          -0.13676699,-0.06803291,-0.55743134,0.2967973,-0.09094952,-0.12427574,
          -0.27146038,-0.37563083,0.15853615,0.049586363],
        [-0.23724489,0.016642023,-0.2502587,0.48801288,0.4526761,0.012872117,
          -0.0565288,-0.113935634,-0.08418312,0.2147764,0.16762011,0.058674254,
          0.16494237,0.1398105,0.25171497,0.4759469],
        [-0.0038104642,-0.2387862,0.22545648,-0.28967497,0.5137987,-0.12194693,
          -0.17891966,0.33095407,-0.003036025,-0.4760634,0.06607926,-0.2880439,
          0.09167388,-0.10248731,-0.025532419,0.23298186],
        [0.06349304,0.13959168,-0.2960539,-0.20912763,0.18433903,-0.2516703,
          -0.007072272,-0.09261524,0.37721395,0.09471353,-0.3878575,0.20997167,
          -0.18772967,0.004912289,-0.48590505,0.3492866]]
    )
    let biasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.90076077,-0.7541278,-0.025081873],
          [0.37973404,-0.26513243,0.10235357],
          [0.3545239,-0.14615154,-0.540749],
          [-0.70222735,-0.34035724,-0.8451414]]]
    )
    let initialStateHiddenForward = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialStateCellForward = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let initialStateHiddenBackward = Tensor<Float>(
        [[0.60601985,0.77093506,0.7210845,-0.012122989]]
    )
    let initialStateCellBackward = Tensor<Float>(
        [[0.06269443,-0.80621815,0.47859335,0.13418436]]
    )
    let expectedSum = Tensor<Float>(
        -0.045482513
    )
    let expectedStates = Tensor<Float>(
        [[[-0.3856787,-0.18899523,0.28026745,-0.3131796],
          [-0.2729215,-0.13601428,0.23670897,-0.14869213],
          [-0.18623777,-0.24313766,0.3612848,-0.042357195],
          [-0.12827212,-0.32416072,0.43230215,-0.025351819]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those four and test it.
    // let expectedFinalStateHiddenForward = Tensor<Float>(
    //     [[-0.048278596,-0.028744427,0.17434421,-0.011268792]]
    // )
    // let expectedFinalStateCellForward = Tensor<Float>(
    //     [[-0.123089634,-0.049444646,0.27142566,-0.023337575]]
    // )
    // let expectedFinalStateHiddenBackward = Tensor<Float>(
    //     [[0.05788272,-0.12734284,0.03379057,-0.11494892]]
    // )
    // let expectedFinalStateCellBackward = Tensor<Float>(
    //     [[0.10647693,-0.207313,0.0632082,-0.31257194]]
    // )
    let expectedGradKernelForward = Tensor<Float>(
        [[0.002949718,-0.05191034,0.024074148,0.00030408613,0.01421415,
          0.046752453,-0.050661147,0.03154892,-0.12825678,-0.26617724,-0.16949388,
          -0.14972451,0.018881105,0.006620474,-0.040960394,0.0029444664],
        [-0.005592436,-0.019203003,0.010193658,-0.0143793905,0.036039345,
          0.04006479,-0.05743648,0.042255975,-0.14608386,-0.28398836,-0.20509121,
          -0.21936977,0.0006449912,0.0065013315,-0.018848399,0.00052668864],
        [-0.030478455,-0.0063452446,-0.01916258,-0.008174095,0.03241433,
          0.035240535,-0.06443699,0.008317959,-0.21671411,-0.38373917,-0.36592808,
          -0.2629513,0.020803409,0.012763705,-0.05019115,0.0049239537]]
    )
    let expectedGradRecurrentKernelForward = Tensor<Float>(
        [[-0.009446432,0.0046872147,-0.0034881886,-0.013322812,0.028175268,
          0.017745323,-0.035205625,0.023002235,-0.097775,-0.1761905,-0.15024684,
          -0.15604898,-0.0037074601,0.0044266693,-0.0065767397,-0.000016547507],
        [-0.007838752,0.00966959,-0.011096001,0.0011524474,-0.001799419,
          -0.0070223105,0.0018404042,-0.0133392075,-0.013871465,-0.009875648,
          -0.04120832,-0.0055879317,0.0054747947,0.0015693348,-0.0077096755,
          0.0012488244],
        [0.017221574,-0.023692459,0.024221376,-0.00008906564,0.00012817048,
          0.014727241,-0.0016071051,0.025199406,0.034627024,0.029167444,
          0.09496009,0.025977321,-0.009273197,-0.0034023263,0.013966143,
          -0.0023199678],
        [-0.012845716,0.03203996,-0.026622673,0.0055777295,-0.015383496,
          -0.032603078,0.027565092,-0.04110182,0.035235237,0.10470742,
          0.0075816773,0.07301342,0.007193738,0.000036919606,-0.0029361092,
          0.0017134398]]
    )
    let expectedGradBiasForward = Tensor<Float>(
        [0.04171711,0.0077266786,0.011825588,0.041857343,-0.10179798,-0.08539298,
        0.15041551,-0.08090454,0.4313099,0.7878589,0.66939783,0.62814295,
        -0.005506222,-0.02124701,0.05680367,-0.0033201296]
    )
    let expectedGradKernelBackward = Tensor<Float>(
        [[0.034572557,-0.014308241,-0.03773727,0.028604513,-0.0025521838,
          0.04228866,-0.036268644,-0.011494972,-0.09629307,-0.12330009,-0.0996568,
          -0.15201326,0.033916347,0.08542741,-0.07855631,0.004139293],
        [0.01675671,-0.0069349525,-0.01829059,0.013864105,-0.0012369986,
          0.02049657,-0.017578775,-0.0055714105,-0.04667156,-0.059761383,
          -0.048301898,-0.07367815,0.016438656,0.041405164,-0.03807486,
          0.0020062423],
        [0.0416086,-0.017220188,-0.045417383,0.03442597,-0.0030715924,
          0.050895054,-0.04364987,-0.013834375,-0.11589019,-0.14839356,
          -0.11993849,-0.18295029,0.040818844,0.1028132,-0.09454372,0.0049817027]]
    )
    let expectedGradRecurrentKernelBackward = Tensor<Float>(
        [[-0.029835999,0.012347964,0.032567136,-0.024685599,0.002202526,
          -0.036494974,0.031299718,0.009920122,0.0831006,0.10640756,0.08600348,
          0.13118693,-0.029269692,-0.07372357,0.06779383,-0.0035721958],
        [-0.03795522,0.015708195,0.04142958,-0.03140325,0.0028018958,
          -0.046426293,0.039817262,0.012619669,0.105714634,0.13536409,0.10940748,
          0.16688661,-0.037234806,-0.093785845,0.08624245,-0.0045442916],
        [-0.035500944,0.014692465,0.038750637,-0.02937264,0.0026207182,
          -0.043424252,0.037242576,0.011803649,0.09887886,0.1266111,0.10233291,
          0.15609531,-0.03482711,-0.087721415,0.080665804,-0.0042504463],
        [0.0005968476,-0.0002470121,-0.00065148197,0.00049381756,-0.00004405994,
          0.00073005556,-0.0006261282,-0.00019844486,-0.0016623674,-0.0021286064,
          -0.0017204374,-0.0026242996,0.000585519,0.0014747867,-0.0013561664,
          0.000071459195]]
    )
    let expectedGradBiasBackward = Tensor<Float>(
        [-0.04923271,0.02037551,0.053739388,-0.04073398,0.0036344123,-0.060220756,
        0.05164801,0.016369302,0.13712521,0.17558429,0.14191529,0.21647298,
        -0.04829824,-0.12165207,0.11186734,-0.0058945194]
    )
    let expectedGradX = Tensor<Float>(
        [[[-0.054938864,-0.022083992,-0.004919176],
          [0.011244737,-0.020291347,-0.024337105],
          [0.017743915,0.023487188,-0.07303475],
          [0.11390987,0.2743578,-0.34671983]]]
    )
    let expectedGradInitialStateHiddenForward = Tensor<Float>(
        [[-0.04673153,0.058627672,0.00071731675,0.035859868]]
    )
    let expectedGradInitialStateCellForward = Tensor<Float>(
        [[0.09301593,0.20037422,0.123320214,0.10345342]]
    )
    let expectedGradInitialStateHiddenBackward = Tensor<Float>(
        [[0.013064697,0.037812762,-0.108545646,0.01709126]]
    )
    let expectedGradInitialStateCellBackward = Tensor<Float>(
        [[0.30869982,0.34445402,0.26831728,0.4068746]]
    )
    // END GENERATED CODE

    // Swaps the columns corresponding to "forget" and "update" gates.
    //
    // This translates between the keras ordering
    //   [input | forget | update | output]
    // and the swift-apis ordering
    //   [input | update | forget | output]
    func swapForgetUpdate(_ t: Tensor<Float>) -> Tensor<Float> {
      var split = t.split(count: 4, alongAxis: -1)
      split.swapAt(1, 2)
      return Tensor(concatenating: split, alongAxis: -1)
    }

    withTensorLeakChecking {
      var lstm = BidirectionalLSTM<Float>(
        LSTMCell(inputSize: 3, hiddenSize: 4),
        mergeFunction: LSTMCell<Float>.State.sum
      )

      lstm.forward.cell.fusedWeight =
        swapForgetUpdate(Tensor(concatenating: [kernelForward, recurrentKernelForward]))
      lstm.forward.cell.fusedBias = swapForgetUpdate(biasForward)

      lstm.backward.cell.fusedWeight =
        swapForgetUpdate(Tensor(concatenating: [kernelBackward, recurrentKernelBackward]))
      lstm.backward.cell.fusedBias = swapForgetUpdate(biasBackward)

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }

      let initialForwardLayerState = LSTMCell<Float>.State(
        cell: initialStateCellForward,
        hidden: initialStateHiddenForward)
      let initialBackwardLayerState = LSTMCell<Float>.State(
        cell: initialStateCellBackward,
        hidden: initialStateHiddenBackward)

      let outputs = lstm(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      )
      assertEqual(
        Tensor(concatenating: outputs.map { $0.hidden }),
        expectedStates,
        accuracy: 1e-6)

      let (outputSum, (gradLSTM)) =
        valueWithGradient(at: lstm) { lstm in
          lstm.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).hidden.sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.forward.cell.fusedWeight,
        swapForgetUpdate(Tensor(concatenating: [expectedGradKernelForward, expectedGradRecurrentKernelForward])),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.forward.cell.fusedBias,
        swapForgetUpdate(expectedGradBiasForward),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.backward.cell.fusedWeight,
        swapForgetUpdate(Tensor(concatenating: [expectedGradKernelBackward, expectedGradRecurrentKernelBackward])),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.backward.cell.fusedBias,
        swapForgetUpdate(expectedGradBiasBackward),
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          lstm.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).hidden.sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward.hidden,
        expectedGradInitialStateHiddenForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward.cell,
        expectedGradInitialStateCellForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward.hidden,
        expectedGradInitialStateHiddenBackward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward.cell,
        expectedGradInitialStateCellBackward,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalLSTMConcatenateMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_lstm.py --merge-mode concat".
    // BEGIN GENERATED CODE
    let kernelForward = Tensor<Float>(
        [[-0.23379976,-0.32979083,0.03977573,0.06884748,-0.093649775,0.34596872,
          -0.0076143146,0.55984896,0.22111118,-0.4210437,0.23581362,0.1825394,
          0.08120936,-0.15200394,-0.08932975,0.14617139],
        [0.4650857,0.18167573,0.37479192,-0.46759078,-0.24752906,-0.5445049,
          0.2544219,0.29843968,0.20215267,0.03678304,0.28829694,-0.5086538,
          -0.5053389,0.2829352,-0.36783904,-0.2113666],
        [-0.23447555,-0.44898528,-0.3757536,0.3030774,0.096295714,0.541732,
          0.46623164,-0.4027334,-0.45229727,0.119446635,-0.36198467,0.020288646,
          0.5418573,-0.3643924,-0.5106659,0.10962713]]
    )
    let recurrentKernelForward = Tensor<Float>(
        [[0.22804272,-0.03794806,0.17161758,-0.5903252,-0.32132986,-0.028507836,
          -0.02729048,0.15502541,-0.29579595,0.041225288,-0.04219827,-0.22448958,
          0.51597697,-0.103077516,0.005849608,0.14667785],
        [0.09507164,-0.25853005,-0.22325921,0.499246,-0.3109755,0.01945003,
          0.33349884,-0.030728217,-0.2997637,0.4836203,0.035094753,-0.1599233,
          0.16274704,-0.0034662904,-0.18398553,-0.06985295],
        [-0.13635698,-0.51163936,-0.09270489,-0.12567669,0.41930002,-0.48658654,
          -0.12225373,-0.28107005,-0.004303295,0.16249746,0.024185713,-0.24801093,
          0.019665923,0.040028445,0.038016353,0.31107086],
        [0.14521983,0.63589036,0.04985115,0.17409262,0.18315184,-0.341375,
          -0.048293047,-0.037892636,0.08461128,0.19565175,-0.18345755,-0.12093313,
          0.27880925,0.39949074,-0.1981584,0.13265577]]
    )
    let biasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    )
    let kernelBackward = Tensor<Float>(
        [[-0.3425501,0.26993138,-0.013709605,0.42182362,-0.10439563,-0.5456007,
          0.23550463,-0.15115345,0.09087402,0.45050424,-0.41379565,0.050539136,
          0.22132623,-0.55731493,-0.15275678,0.07351905],
        [-0.20979714,0.1675657,0.5581859,-0.50259715,0.4392975,-0.26118547,
          -0.06672484,0.26642138,-0.37113252,-0.29193833,0.5320031,0.54281443,
          -0.5196755,-0.013125479,0.006757021,0.5087437],
        [0.41886348,0.16300112,0.15921205,0.3881111,0.34322268,0.0014763474,
          0.2432934,0.32898188,0.26853347,-0.48425156,-0.3587765,-0.19107765,
          0.5605115,-0.18283507,0.13686264,0.008577526]]
    )
    let recurrentKernelBackward = Tensor<Float>(
        [[0.332492,0.3287107,0.013327966,-0.15466748,0.18726069,-0.18434268,
          -0.13676699,-0.06803291,-0.55743134,0.2967973,-0.09094952,-0.12427574,
          -0.27146038,-0.37563083,0.15853615,0.049586363],
        [-0.23724489,0.016642023,-0.2502587,0.48801288,0.4526761,0.012872117,
          -0.0565288,-0.113935634,-0.08418312,0.2147764,0.16762011,0.058674254,
          0.16494237,0.1398105,0.25171497,0.4759469],
        [-0.0038104642,-0.2387862,0.22545648,-0.28967497,0.5137987,-0.12194693,
          -0.17891966,0.33095407,-0.003036025,-0.4760634,0.06607926,-0.2880439,
          0.09167388,-0.10248731,-0.025532419,0.23298186],
        [0.06349304,0.13959168,-0.2960539,-0.20912763,0.18433903,-0.2516703,
          -0.007072272,-0.09261524,0.37721395,0.09471353,-0.3878575,0.20997167,
          -0.18772967,0.004912289,-0.48590505,0.3492866]]
    )
    let biasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.90076077,-0.7541278,-0.025081873],
          [0.37973404,-0.26513243,0.10235357],
          [0.3545239,-0.14615154,-0.540749],
          [-0.70222735,-0.34035724,-0.8451414]]]
    )
    let initialStateHiddenForward = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialStateCellForward = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let initialStateHiddenBackward = Tensor<Float>(
        [[0.60601985,0.77093506,0.7210845,-0.012122989]]
    )
    let initialStateCellBackward = Tensor<Float>(
        [[0.06269443,-0.80621815,0.47859335,0.13418436]]
    )
    let expectedSum = Tensor<Float>(
        -0.045482457
    )
    let expectedStates = Tensor<Float>(
        [[[-0.44356138,-0.061652392,0.24647687,-0.19823068,0.05788271,
          -0.12734284,0.033790566,-0.11494893],
          [-0.2737949,-0.07248981,0.19220428,-0.08825301,0.0008733621,
          -0.06352448,0.044504695,-0.060439102],
          [-0.124358594,-0.1213435,0.21946031,-0.025490183,-0.061879173,
          -0.12179416,0.1418245,-0.016867002],
          [-0.048278596,-0.028744407,0.1743442,-0.011268781,-0.07999351,
          -0.29541627,0.25795794,-0.014083028]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those four and test it.
    // let expectedFinalStateHiddenForward = Tensor<Float>(
    //     [[-0.048278596,-0.028744407,0.1743442,-0.011268781]]
    // )
    // let expectedFinalStateCellForward = Tensor<Float>(
    //     [[-0.123089634,-0.049444616,0.27142566,-0.023337554]]
    // )
    // let expectedFinalStateHiddenBackward = Tensor<Float>(
    //     [[0.05788271,-0.12734284,0.033790566,-0.11494893]]
    // )
    // let expectedFinalStateCellBackward = Tensor<Float>(
    //     [[0.106476925,-0.20731302,0.0632082,-0.31257194]]
    // )
    let expectedGradKernelForward = Tensor<Float>(
        [[0.0029497184,-0.05191034,0.02407415,0.000304088,0.014214157,0.046752453,
          -0.050661147,0.031548906,-0.12825678,-0.26617724,-0.16949388,
          -0.14972453,0.018881105,0.0066204695,-0.040960398,0.002944462],
        [-0.005592435,-0.019203003,0.010193659,-0.014379387,0.03603935,
          0.04006479,-0.05743648,0.04225596,-0.14608386,-0.28398833,-0.20509121,
          -0.21936974,0.00064499304,0.0065013287,-0.0188484,0.0005266868],
        [-0.030478455,-0.0063452474,-0.019162579,-0.008174095,0.03241433,
          0.035240535,-0.06443699,0.008317955,-0.21671411,-0.38373917,-0.36592808,
          -0.26295128,0.020803409,0.012763699,-0.050191153,0.0049239486]]
    )
    let expectedGradRecurrentKernelForward = Tensor<Float>(
        [[-0.009446431,0.0046872147,-0.0034881886,-0.013322809,0.028175268,
          0.017745323,-0.035205625,0.02300223,-0.097775005,-0.17619048,
          -0.15024684,-0.15604895,-0.0037074587,0.0044266684,-0.006576741,
          -0.000016547943],
        [-0.007838752,0.00966959,-0.011096001,0.0011524465,-0.0017994214,
          -0.0070223105,0.0018404042,-0.013339203,-0.013871463,-0.009875648,
          -0.04120832,-0.005587943,0.0054747937,0.0015693338,-0.007709676,
          0.0012488235],
        [0.017221576,-0.023692459,0.024221376,-0.00008906424,0.00012817606,
          0.014727241,-0.0016071051,0.025199397,0.034627017,0.029167429,
          0.09496009,0.025977343,-0.009273196,-0.0034023244,0.013966144,
          -0.0023199664],
        [-0.012845716,0.032039955,-0.026622673,0.0055777263,-0.0153835025,
          -0.032603078,0.027565094,-0.041101806,0.03523525,0.104707435,
          0.0075816847,0.07301338,0.0071937367,0.000036919257,-0.0029361085,
          0.0017134396]]
    )
    let expectedGradBiasForward = Tensor<Float>(
        [0.041717105,0.0077266824,0.011825588,0.041857336,-0.10179798,-0.08539298,
        0.15041551,-0.08090452,0.43130994,0.7878589,0.66939783,0.6281428,
        -0.005506225,-0.021247003,0.056803677,-0.0033201245]
    )
    let expectedGradKernelBackward = Tensor<Float>(
        [[0.034572553,-0.014308241,-0.037737265,0.028604513,-0.002552184,
          0.042288657,-0.036268644,-0.011494972,-0.09629306,-0.12330007,
          -0.09965681,-0.15201326,0.033916343,0.08542741,-0.07855631,0.004139293],
        [0.016756708,-0.0069349525,-0.018290589,0.013864105,-0.0012369987,
          0.020496568,-0.017578775,-0.0055714105,-0.046671554,-0.059761375,
          -0.0483019,-0.07367815,0.016438654,0.041405164,-0.03807486,0.0020062423],
        [0.041608598,-0.017220188,-0.04541738,0.03442597,-0.0030715927,
          0.050895046,-0.04364987,-0.013834375,-0.11589018,-0.14839353,-0.1199385,
          -0.18295029,0.04081884,0.10281321,-0.09454372,0.0049817027]]
    )
    let expectedGradRecurrentKernelBackward = Tensor<Float>(
        [[-0.029835997,0.012347964,0.032567132,-0.024685599,0.0022025262,
          -0.03649497,0.031299718,0.009920122,0.08310059,0.106407546,0.08600349,
          0.13118693,-0.02926969,-0.07372357,0.06779383,-0.0035721958],
        [-0.03795522,0.015708195,0.041429576,-0.03140325,0.0028018963,
          -0.046426285,0.039817262,0.012619669,0.10571462,0.13536406,0.109407485,
          0.16688661,-0.0372348,-0.09378585,0.08624245,-0.0045442916],
        [-0.03550094,0.014692465,0.038750634,-0.02937264,0.0026207187,
          -0.043424245,0.037242576,0.011803649,0.098878846,0.12661108,0.10233293,
          0.15609531,-0.03482711,-0.08772142,0.080665804,-0.0042504463],
        [0.00059684756,-0.0002470121,-0.00065148197,0.00049381756,
          -0.000044059943,0.00073005544,-0.0006261282,-0.00019844486,
          -0.0016623671,-0.002128606,-0.0017204377,-0.0026242996,0.00058551895,
          0.0014747867,-0.0013561664,0.000071459195]]
    )
    let expectedGradBiasBackward = Tensor<Float>(
        [-0.049232706,0.02037551,0.053739384,-0.04073398,0.0036344128,-0.06022075,
        0.05164801,0.016369302,0.1371252,0.17558426,0.1419153,0.21647298,
        -0.048298236,-0.121652074,0.11186734,-0.0058945194]
    )
    let expectedGradX = Tensor<Float>(
        [[[-0.054938864,-0.022083957,-0.004919187],
          [0.01124474,-0.020291314,-0.02433711],
          [0.017743908,0.02348721,-0.07303476],
          [0.113909885,0.27435783,-0.3467198]]]
    )
    let expectedGradInitialStateHiddenForward = Tensor<Float>(
        [[-0.046731517,0.05862768,0.0007173242,0.035859868]]
    )
    let expectedGradInitialStateCellForward = Tensor<Float>(
        [[0.09301593,0.20037422,0.123320214,0.10345338]]
    )
    let expectedGradInitialStateHiddenBackward = Tensor<Float>(
        [[0.013064705,0.037812762,-0.10854563,0.017091252]]
    )
    let expectedGradInitialStateCellBackward = Tensor<Float>(
        [[0.3086998,0.34445396,0.26831728,0.4068746]]
    )
    // END GENERATED CODE

    // Swaps the columns corresponding to "forget" and "update" gates.
    //
    // This translates between the keras ordering
    //   [input | forget | update | output]
    // and the swift-apis ordering
    //   [input | update | forget | output]
    func swapForgetUpdate(_ t: Tensor<Float>) -> Tensor<Float> {
      var split = t.split(count: 4, alongAxis: -1)
      split.swapAt(1, 2)
      return Tensor(concatenating: split, alongAxis: -1)
    }

    withTensorLeakChecking {
      var lstm = BidirectionalLSTM<Float>(
        LSTMCell(inputSize: 3, hiddenSize: 4)
      )

      lstm.forward.cell.fusedWeight =
        swapForgetUpdate(Tensor(concatenating: [kernelForward, recurrentKernelForward]))
      lstm.forward.cell.fusedBias = swapForgetUpdate(biasForward)

      lstm.backward.cell.fusedWeight =
        swapForgetUpdate(Tensor(concatenating: [kernelBackward, recurrentKernelBackward]))
      lstm.backward.cell.fusedBias = swapForgetUpdate(biasBackward)

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }

      let initialForwardLayerState = LSTMCell<Float>.State(
        cell: initialStateCellForward,
        hidden: initialStateHiddenForward)
      let initialBackwardLayerState = LSTMCell<Float>.State(
        cell: initialStateCellBackward,
        hidden: initialStateHiddenBackward)

      let outputs = lstm(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      )
      assertEqual(
        Tensor(concatenating: outputs.map { $0.hidden }),
        expectedStates,
        accuracy: 1e-6)

      let (outputSum, (gradLSTM)) =
        valueWithGradient(at: lstm) { lstm in
          lstm.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).hidden.sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.forward.cell.fusedWeight,
        swapForgetUpdate(Tensor(concatenating: [expectedGradKernelForward, expectedGradRecurrentKernelForward])),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.forward.cell.fusedBias,
        swapForgetUpdate(expectedGradBiasForward),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.backward.cell.fusedWeight,
        swapForgetUpdate(Tensor(concatenating: [expectedGradKernelBackward, expectedGradRecurrentKernelBackward])),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.backward.cell.fusedBias,
        swapForgetUpdate(expectedGradBiasBackward),
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          lstm.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).hidden.sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward.hidden,
        expectedGradInitialStateHiddenForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward.cell,
        expectedGradInitialStateCellForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward.hidden,
        expectedGradInitialStateHiddenBackward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward.cell,
        expectedGradInitialStateCellBackward,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalLSTMAverageMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_lstm.py --merge-mode ave".
    // BEGIN GENERATED CODE
    let kernelForward = Tensor<Float>(
        [[-0.23379976,-0.32979083,0.03977573,0.06884748,-0.093649775,0.34596872,
          -0.0076143146,0.55984896,0.22111118,-0.4210437,0.23581362,0.1825394,
          0.08120936,-0.15200394,-0.08932975,0.14617139],
        [0.4650857,0.18167573,0.37479192,-0.46759078,-0.24752906,-0.5445049,
          0.2544219,0.29843968,0.20215267,0.03678304,0.28829694,-0.5086538,
          -0.5053389,0.2829352,-0.36783904,-0.2113666],
        [-0.23447555,-0.44898528,-0.3757536,0.3030774,0.096295714,0.541732,
          0.46623164,-0.4027334,-0.45229727,0.119446635,-0.36198467,0.020288646,
          0.5418573,-0.3643924,-0.5106659,0.10962713]]
    )
    let recurrentKernelForward = Tensor<Float>(
        [[0.22804272,-0.03794806,0.17161758,-0.5903252,-0.32132986,-0.028507836,
          -0.02729048,0.15502541,-0.29579595,0.041225288,-0.04219827,-0.22448958,
          0.51597697,-0.103077516,0.005849608,0.14667785],
        [0.09507164,-0.25853005,-0.22325921,0.499246,-0.3109755,0.01945003,
          0.33349884,-0.030728217,-0.2997637,0.4836203,0.035094753,-0.1599233,
          0.16274704,-0.0034662904,-0.18398553,-0.06985295],
        [-0.13635698,-0.51163936,-0.09270489,-0.12567669,0.41930002,-0.48658654,
          -0.12225373,-0.28107005,-0.004303295,0.16249746,0.024185713,-0.24801093,
          0.019665923,0.040028445,0.038016353,0.31107086],
        [0.14521983,0.63589036,0.04985115,0.17409262,0.18315184,-0.341375,
          -0.048293047,-0.037892636,0.08461128,0.19565175,-0.18345755,-0.12093313,
          0.27880925,0.39949074,-0.1981584,0.13265577]]
    )
    let biasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    )
    let kernelBackward = Tensor<Float>(
        [[-0.3425501,0.26993138,-0.013709605,0.42182362,-0.10439563,-0.5456007,
          0.23550463,-0.15115345,0.09087402,0.45050424,-0.41379565,0.050539136,
          0.22132623,-0.55731493,-0.15275678,0.07351905],
        [-0.20979714,0.1675657,0.5581859,-0.50259715,0.4392975,-0.26118547,
          -0.06672484,0.26642138,-0.37113252,-0.29193833,0.5320031,0.54281443,
          -0.5196755,-0.013125479,0.006757021,0.5087437],
        [0.41886348,0.16300112,0.15921205,0.3881111,0.34322268,0.0014763474,
          0.2432934,0.32898188,0.26853347,-0.48425156,-0.3587765,-0.19107765,
          0.5605115,-0.18283507,0.13686264,0.008577526]]
    )
    let recurrentKernelBackward = Tensor<Float>(
        [[0.332492,0.3287107,0.013327966,-0.15466748,0.18726069,-0.18434268,
          -0.13676699,-0.06803291,-0.55743134,0.2967973,-0.09094952,-0.12427574,
          -0.27146038,-0.37563083,0.15853615,0.049586363],
        [-0.23724489,0.016642023,-0.2502587,0.48801288,0.4526761,0.012872117,
          -0.0565288,-0.113935634,-0.08418312,0.2147764,0.16762011,0.058674254,
          0.16494237,0.1398105,0.25171497,0.4759469],
        [-0.0038104642,-0.2387862,0.22545648,-0.28967497,0.5137987,-0.12194693,
          -0.17891966,0.33095407,-0.003036025,-0.4760634,0.06607926,-0.2880439,
          0.09167388,-0.10248731,-0.025532419,0.23298186],
        [0.06349304,0.13959168,-0.2960539,-0.20912763,0.18433903,-0.2516703,
          -0.007072272,-0.09261524,0.37721395,0.09471353,-0.3878575,0.20997167,
          -0.18772967,0.004912289,-0.48590505,0.3492866]]
    )
    let biasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.90076077,-0.7541278,-0.025081873],
          [0.37973404,-0.26513243,0.10235357],
          [0.3545239,-0.14615154,-0.540749],
          [-0.70222735,-0.34035724,-0.8451414]]]
    )
    let initialStateHiddenForward = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialStateCellForward = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let initialStateHiddenBackward = Tensor<Float>(
        [[0.60601985,0.77093506,0.7210845,-0.012122989]]
    )
    let initialStateCellBackward = Tensor<Float>(
        [[0.06269443,-0.80621815,0.47859335,0.13418436]]
    )
    let expectedSum = Tensor<Float>(
        -0.02274122
    )
    let expectedStates = Tensor<Float>(
        [[[-0.19283932,-0.09449761,0.14013372,-0.1565898],
          [-0.13646077,-0.06800714,0.118354484,-0.07434605],
          [-0.09311888,-0.12156883,0.1806424,-0.021178592],
          [-0.06413605,-0.16208033,0.21615106,-0.012675904]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those four and test it.
    // let expectedFinalStateHiddenForward = Tensor<Float>(
    //     [[-0.048278596,-0.028744407,0.1743442,-0.011268781]]
    // )
    // let expectedFinalStateCellForward = Tensor<Float>(
    //     [[-0.123089634,-0.049444616,0.27142566,-0.023337554]]
    // )
    // let expectedFinalStateHiddenBackward = Tensor<Float>(
    //     [[0.05788271,-0.12734284,0.033790566,-0.11494893]]
    // )
    // let expectedFinalStateCellBackward = Tensor<Float>(
    //     [[0.106476925,-0.20731302,0.0632082,-0.31257194]]
    // )
    let expectedGradKernelForward = Tensor<Float>(
        [[0.0014748592,-0.02595517,0.012037075,0.000152044,0.0071070786,
          0.023376226,-0.025330573,0.015774453,-0.06412839,-0.13308862,
          -0.08474694,-0.074862264,0.009440552,0.0033102348,-0.020480199,
          0.001472231],
        [-0.0027962176,-0.009601502,0.0050968295,-0.0071896934,0.018019674,
          0.020032395,-0.02871824,0.02112798,-0.07304193,-0.14199416,-0.102545604,
          -0.10968487,0.00032249652,0.0032506643,-0.0094242,0.0002633434],
        [-0.015239228,-0.0031726237,-0.009581289,-0.0040870477,0.016207164,
          0.017620267,-0.032218494,0.0041589774,-0.10835706,-0.19186959,
          -0.18296404,-0.13147564,0.010401704,0.0063818493,-0.025095576,
          0.0024619743]]
    )
    let expectedGradRecurrentKernelForward = Tensor<Float>(
        [[-0.0047232155,0.0023436074,-0.0017440943,-0.0066614044,0.014087634,
          0.008872662,-0.017602812,0.011501115,-0.048887502,-0.08809524,
          -0.07512342,-0.07802448,-0.0018537294,0.0022133342,-0.0032883706,
          -0.000008273972],
        [-0.003919376,0.004834795,-0.0055480003,0.00057622325,-0.0008997107,
          -0.0035111553,0.0009202021,-0.0066696014,-0.0069357315,-0.004937824,
          -0.02060416,-0.0027939714,0.0027373969,0.0007846669,-0.003854838,
          0.00062441174],
        [0.008610788,-0.011846229,0.012110688,-0.00004453212,0.00006408803,
          0.0073636207,-0.00080355257,0.012599698,0.017313508,0.014583714,
          0.047480047,0.012988672,-0.004636598,-0.0017011622,0.006983072,
          -0.0011599832],
        [-0.006422858,0.016019978,-0.013311337,0.0027888631,-0.0076917512,
          -0.016301539,0.013782547,-0.020550903,0.017617624,0.052353717,
          0.0037908424,0.03650669,0.0035968684,0.000018459628,-0.0014680543,
          0.0008567198]]
    )
    let expectedGradBiasForward = Tensor<Float>(
        [0.020858552,0.0038633412,0.005912794,0.020928668,-0.05089899,-0.04269649,
        0.075207755,-0.04045226,0.21565497,0.39392945,0.33469892,0.3140714,
        -0.0027531125,-0.010623502,0.028401839,-0.0016600622]
    )
    let expectedGradKernelBackward = Tensor<Float>(
        [[0.017286276,-0.0071541206,-0.018868633,0.0143022565,-0.001276092,
          0.021144329,-0.018134322,-0.005747486,-0.04814653,-0.061650034,
          -0.049828406,-0.07600663,0.016958172,0.042713705,-0.039278153,
          0.0020696465],
        [0.008378354,-0.0034674762,-0.009145294,0.0069320523,-0.00061849935,
          0.010248284,-0.0087893875,-0.0027857053,-0.023335777,-0.029880688,
          -0.02415095,-0.036839075,0.008219327,0.020702582,-0.01903743,
          0.0010031211],
        [0.020804299,-0.008610094,-0.02270869,0.017212985,-0.0015357963,
          0.025447523,-0.021824935,-0.0069171875,-0.05794509,-0.07419676,
          -0.05996925,-0.091475144,0.02040942,0.051406603,-0.04727186,
          0.0024908513]]
    )
    let expectedGradRecurrentKernelBackward = Tensor<Float>(
        [[-0.014917999,0.006173982,0.016283566,-0.012342799,0.0011012631,
          -0.018247485,0.015649859,0.004960061,0.041550294,0.053203773,
          0.043001745,0.065593466,-0.014634845,-0.036861785,0.033896916,
          -0.0017860979],
        [-0.01897761,0.0078540975,0.020714788,-0.015701625,0.0014009482,
          -0.023213143,0.019908631,0.0063098343,0.05285731,0.06768203,0.054703742,
          0.08344331,-0.0186174,-0.046892926,0.043121226,-0.0022721458],
        [-0.01775047,0.0073462324,0.019375317,-0.01468632,0.0013103593,
          -0.021712122,0.018621288,0.0059018247,0.049439423,0.06330554,
          0.051166464,0.078047656,-0.017413555,-0.04386071,0.040332902,
          -0.0021252232],
        [0.00029842378,-0.00012350605,-0.00032574098,0.00024690878,
          -0.000022029972,0.00036502772,-0.0003130641,-0.00009922243,
          -0.00083118357,-0.001064303,-0.00086021883,-0.0013121498,0.00029275948,
          0.00073739333,-0.0006780832,0.000035729598]]
    )
    let expectedGradBiasBackward = Tensor<Float>(
        [-0.024616353,0.010187755,0.026869692,-0.02036699,0.0018172064,
        -0.030110374,0.025824005,0.008184651,0.0685626,0.08779213,0.07095765,
        0.10823649,-0.024149118,-0.060826037,0.05593367,-0.0029472597]
    )
    let expectedGradX = Tensor<Float>(
        [[[-0.027469432,-0.011041978,-0.0024595936],
          [0.00562237,-0.010145657,-0.012168555],
          [0.008871954,0.011743605,-0.03651738],
          [0.056954943,0.13717891,-0.1733599]]]
    )
    let expectedGradInitialStateHiddenForward = Tensor<Float>(
        [[-0.023365758,0.02931384,0.0003586621,0.017929934]]
    )
    let expectedGradInitialStateCellForward = Tensor<Float>(
        [[0.046507966,0.10018711,0.061660107,0.05172669]]
    )
    let expectedGradInitialStateHiddenBackward = Tensor<Float>(
        [[0.0065323524,0.018906381,-0.054272816,0.008545626]]
    )
    let expectedGradInitialStateCellBackward = Tensor<Float>(
        [[0.1543499,0.17222698,0.13415864,0.2034373]]
    )
    // END GENERATED CODE

    // Swaps the columns corresponding to "forget" and "update" gates.
    //
    // This translates between the keras ordering
    //   [input | forget | update | output]
    // and the swift-apis ordering
    //   [input | update | forget | output]
    func swapForgetUpdate(_ t: Tensor<Float>) -> Tensor<Float> {
      var split = t.split(count: 4, alongAxis: -1)
      split.swapAt(1, 2)
      return Tensor(concatenating: split, alongAxis: -1)
    }

    withTensorLeakChecking {
      var lstm = BidirectionalLSTM<Float>(
        LSTMCell(inputSize: 3, hiddenSize: 4),
        mergeFunction: LSTMCell<Float>.State.average
      )

      lstm.forward.cell.fusedWeight =
        swapForgetUpdate(Tensor(concatenating: [kernelForward, recurrentKernelForward]))
      lstm.forward.cell.fusedBias = swapForgetUpdate(biasForward)

      lstm.backward.cell.fusedWeight =
        swapForgetUpdate(Tensor(concatenating: [kernelBackward, recurrentKernelBackward]))
      lstm.backward.cell.fusedBias = swapForgetUpdate(biasBackward)

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }

      let initialForwardLayerState = LSTMCell<Float>.State(
        cell: initialStateCellForward,
        hidden: initialStateHiddenForward)
      let initialBackwardLayerState = LSTMCell<Float>.State(
        cell: initialStateCellBackward,
        hidden: initialStateHiddenBackward)

      let outputs = lstm(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      )
      assertEqual(
        Tensor(concatenating: outputs.map { $0.hidden }),
        expectedStates,
        accuracy: 1e-6)

      let (outputSum, (gradLSTM)) =
        valueWithGradient(at: lstm) { lstm in
          lstm.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).hidden.sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.forward.cell.fusedWeight,
        swapForgetUpdate(Tensor(concatenating: [expectedGradKernelForward, expectedGradRecurrentKernelForward])),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.forward.cell.fusedBias,
        swapForgetUpdate(expectedGradBiasForward),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.backward.cell.fusedWeight,
        swapForgetUpdate(Tensor(concatenating: [expectedGradKernelBackward, expectedGradRecurrentKernelBackward])),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.backward.cell.fusedBias,
        swapForgetUpdate(expectedGradBiasBackward),
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          lstm.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).hidden.sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward.hidden,
        expectedGradInitialStateHiddenForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward.cell,
        expectedGradInitialStateCellForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward.hidden,
        expectedGradInitialStateHiddenBackward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward.cell,
        expectedGradInitialStateCellBackward,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalLSTMMultiplyMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_lstm.py --merge-mode mul".
    // BEGIN GENERATED CODE
    let kernelForward = Tensor<Float>(
        [[-0.23379976,-0.32979083,0.03977573,0.06884748,-0.093649775,0.34596872,
          -0.0076143146,0.55984896,0.22111118,-0.4210437,0.23581362,0.1825394,
          0.08120936,-0.15200394,-0.08932975,0.14617139],
        [0.4650857,0.18167573,0.37479192,-0.46759078,-0.24752906,-0.5445049,
          0.2544219,0.29843968,0.20215267,0.03678304,0.28829694,-0.5086538,
          -0.5053389,0.2829352,-0.36783904,-0.2113666],
        [-0.23447555,-0.44898528,-0.3757536,0.3030774,0.096295714,0.541732,
          0.46623164,-0.4027334,-0.45229727,0.119446635,-0.36198467,0.020288646,
          0.5418573,-0.3643924,-0.5106659,0.10962713]]
    )
    let recurrentKernelForward = Tensor<Float>(
        [[0.22804272,-0.03794806,0.17161758,-0.5903252,-0.32132986,-0.028507836,
          -0.02729048,0.15502541,-0.29579595,0.041225288,-0.04219827,-0.22448958,
          0.51597697,-0.103077516,0.005849608,0.14667785],
        [0.09507164,-0.25853005,-0.22325921,0.499246,-0.3109755,0.01945003,
          0.33349884,-0.030728217,-0.2997637,0.4836203,0.035094753,-0.1599233,
          0.16274704,-0.0034662904,-0.18398553,-0.06985295],
        [-0.13635698,-0.51163936,-0.09270489,-0.12567669,0.41930002,-0.48658654,
          -0.12225373,-0.28107005,-0.004303295,0.16249746,0.024185713,-0.24801093,
          0.019665923,0.040028445,0.038016353,0.31107086],
        [0.14521983,0.63589036,0.04985115,0.17409262,0.18315184,-0.341375,
          -0.048293047,-0.037892636,0.08461128,0.19565175,-0.18345755,-0.12093313,
          0.27880925,0.39949074,-0.1981584,0.13265577]]
    )
    let biasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    )
    let kernelBackward = Tensor<Float>(
        [[-0.3425501,0.26993138,-0.013709605,0.42182362,-0.10439563,-0.5456007,
          0.23550463,-0.15115345,0.09087402,0.45050424,-0.41379565,0.050539136,
          0.22132623,-0.55731493,-0.15275678,0.07351905],
        [-0.20979714,0.1675657,0.5581859,-0.50259715,0.4392975,-0.26118547,
          -0.06672484,0.26642138,-0.37113252,-0.29193833,0.5320031,0.54281443,
          -0.5196755,-0.013125479,0.006757021,0.5087437],
        [0.41886348,0.16300112,0.15921205,0.3881111,0.34322268,0.0014763474,
          0.2432934,0.32898188,0.26853347,-0.48425156,-0.3587765,-0.19107765,
          0.5605115,-0.18283507,0.13686264,0.008577526]]
    )
    let recurrentKernelBackward = Tensor<Float>(
        [[0.332492,0.3287107,0.013327966,-0.15466748,0.18726069,-0.18434268,
          -0.13676699,-0.06803291,-0.55743134,0.2967973,-0.09094952,-0.12427574,
          -0.27146038,-0.37563083,0.15853615,0.049586363],
        [-0.23724489,0.016642023,-0.2502587,0.48801288,0.4526761,0.012872117,
          -0.0565288,-0.113935634,-0.08418312,0.2147764,0.16762011,0.058674254,
          0.16494237,0.1398105,0.25171497,0.4759469],
        [-0.0038104642,-0.2387862,0.22545648,-0.28967497,0.5137987,-0.12194693,
          -0.17891966,0.33095407,-0.003036025,-0.4760634,0.06607926,-0.2880439,
          0.09167388,-0.10248731,-0.025532419,0.23298186],
        [0.06349304,0.13959168,-0.2960539,-0.20912763,0.18433903,-0.2516703,
          -0.007072272,-0.09261524,0.37721395,0.09471353,-0.3878575,0.20997167,
          -0.18772967,0.004912289,-0.48590505,0.3492866]]
    )
    let biasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.90076077,-0.7541278,-0.025081873],
          [0.37973404,-0.26513243,0.10235357],
          [0.3545239,-0.14615154,-0.540749],
          [-0.70222735,-0.34035724,-0.8451414]]]
    )
    let initialStateHiddenForward = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialStateCellForward = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let initialStateHiddenBackward = Tensor<Float>(
        [[0.60601985,0.77093506,0.7210845,-0.012122989]]
    )
    let initialStateCellBackward = Tensor<Float>(
        [[0.06269443,-0.80621815,0.47859335,0.13418436]]
    )
    let expectedSum = Tensor<Float>(
        0.057485707
    )
    let expectedStates = Tensor<Float>(
        [[[-0.025674535,0.007850991,0.008328593,0.022786405],
          [-0.00023912209,0.0046048774,0.008553993,0.0053339326],
          [0.007695207,0.01477893,0.031124849,0.000429943],
          [0.0038619742,0.0084915655,0.04497347,0.00015869855]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those four and test it.
    // let expectedFinalStateHiddenForward = Tensor<Float>(
    //     [[-0.048278596,-0.028744407,0.1743442,-0.011268781]]
    // )
    // let expectedFinalStateCellForward = Tensor<Float>(
    //     [[-0.123089634,-0.049444616,0.27142566,-0.023337554]]
    // )
    // let expectedFinalStateHiddenBackward = Tensor<Float>(
    //     [[0.05788271,-0.12734284,0.033790566,-0.11494893]]
    // )
    // let expectedFinalStateCellBackward = Tensor<Float>(
    //     [[0.106476925,-0.20731302,0.0632082,-0.31257194]]
    // )
    let expectedGradKernelForward = Tensor<Float>(
        [[-0.0006356439,0.015833925,0.005248367,0.000088096946,-0.001286473,
          -0.014080829,-0.012435054,-0.0042246087,0.010528095,0.0785853,
          -0.04466948,0.005445134,-0.0011754832,-0.0016675827,-0.011209748,
          -0.00035536994],
        [0.00035326666,0.005798911,0.0020744896,0.0016610449,-0.0035952562,
          -0.012197385,-0.013497299,-0.0053806542,0.013618026,0.08609101,
          -0.04995886,0.018841323,-0.0003639532,-0.0020185974,-0.0043832674,
          -0.0008603267],
        [0.002633221,0.00171838,-0.004717764,0.00048049382,-0.0028204727,
          -0.010515593,-0.016336953,-0.00037660176,0.018144649,0.114633985,
          -0.09280567,0.008752153,-0.0020848382,-0.0041145375,-0.01230508,
          -0.00033673478]]
    )
    let expectedGradRecurrentKernelForward = Tensor<Float>(
        [[0.00085955136,-0.0015130613,-0.0009309999,0.0014424899,-0.0028096742,
          -0.005426242,-0.008224582,-0.0027340245,0.009384694,0.053980358,
          -0.035707016,0.014470998,-0.00012773072,-0.0015185195,-0.0009792406,
          -0.00059664843],
        [0.00075029954,-0.0030118634,-0.0025134794,-0.00027205143,0.00029734446,
          0.0022050086,0.00009390758,0.0019716173,0.0007304535,0.0025818497,
          -0.010839711,-0.0033222511,-0.0004790762,-0.0005259809,-0.0018902641,
          0.0002027992],
        [-0.0016653631,0.007366645,0.0054886946,0.0003065939,-0.00023920857,
          -0.004606634,0.00025729695,-0.0038056956,-0.0022326056,-0.008233182,
          0.024358716,0.0047218287,0.0009058922,0.0011722009,0.0032562078,
          -0.00033911638],
        [0.001339203,-0.009877648,-0.0059118806,-0.00090225495,0.0017254684,
          0.0100316,0.005946205,0.005787832,-0.004097135,-0.032188937,0.000986672,
          -0.01200073,-0.0005991369,-0.000113752816,-0.00061858393,0.0006727869]]
    )
    let expectedGradBiasForward = Tensor<Float>(
        [-0.0036032305,-0.0020163525,0.0033609606,-0.004352904,0.009900483,
        0.02591736,0.035977,0.009494458,-0.039504252,-0.23896675,0.1634979,
        -0.0476815,0.0017783717,0.006988898,0.012471562,0.0020117098]
    )
    let expectedGradKernelBackward = Tensor<Float>(
        [[-0.0016691145,0.00041128194,-0.0065792724,-0.00032233805,0.00012321587,
          -0.0012155625,-0.0063232277,0.00012953431,0.004648894,0.0035441876,
          -0.017374588,0.0017130041,-0.0016374333,-0.0024555605,-0.013695836,
          -0.000046644785],
        [-0.0008089904,0.00019934113,-0.0031888576,-0.00015623159,0.00005972056,
          -0.0005891617,-0.0030647572,0.000062783,0.0022532372,0.0017178054,
          -0.008421157,0.0008302629,-0.00079363515,-0.0011901669,-0.006638131,
          -0.000022607906],
        [-0.0020088048,0.00049498415,-0.007918255,-0.0003879388,0.00014829218,
          -0.001462948,-0.0076101017,0.00015589653,0.005595016,0.0042654844,
          -0.020910583,0.0020616266,-0.0019706762,-0.0029553047,-0.01648315,
          -0.000056137716]]
    )
    let expectedGradRecurrentKernelBackward = Tensor<Float>(
        [[0.0014404402,-0.00035493495,0.00567789,0.00027817665,-0.000106334875,
          0.0010490263,0.0054569244,-0.000111787675,-0.0040119803,-0.003058622,
          0.014994211,-0.0014783167,0.0014130995,0.0021191405,0.011819461,
          0.000040254294],
        [0.0018324248,-0.00045152282,0.0072230054,0.0003538764,-0.00013527162,
          0.0013344962,0.006941908,-0.00014220827,-0.005103754,-0.0038909598,
          0.019074561,-0.0018806087,0.001797644,0.0026958187,0.015035871,
          0.00005120863],
        [0.0017139356,-0.00042232624,0.0067559476,0.0003309939,-0.00012652461,
          0.0012482043,0.0064930273,-0.00013301274,-0.0047737323,-0.0036393607,
          0.017841153,-0.0017590038,0.0016814038,0.0025215005,0.014063614,
          0.000047897352],
        [-0.000028814964,0.000007100217,-0.000113582086,-0.000005564723,
          0.0000021271524,-0.000020985013,-0.000109161825,0.0000022362317,
          0.00008025676,0.00006118552,-0.00029994833,0.000029572657,
          -0.000028268032,-0.000042391872,-0.00023643977,-0.000000805258]]
    )
    let expectedGradBiasBackward = Tensor<Float>(
        [0.002376886,-0.00058568205,0.009369149,0.00045902235,-0.00017546435,
        0.0017310098,0.0090045305,-0.00018446207,-0.0066202125,-0.0050470657,
        0.024742112,-0.0024393867,0.002331771,0.0034968168,0.019503422,
        0.00006642405]
    )
    let expectedGradX = Tensor<Float>(
        [[[0.025954716,0.0037635816,0.005093654],
          [0.016548203,0.014938684,-0.0098211765],
          [0.028925406,0.018546272,-0.017679248],
          [0.045659292,0.028008066,-0.028804896]]]
    )
    let expectedGradInitialStateHiddenForward = Tensor<Float>(
        [[0.002969412,-0.012251236,-0.0049122106,-0.018355861]]
    )
    let expectedGradInitialStateCellForward = Tensor<Float>(
        [[-0.010009506,-0.061632644,0.027160529,-0.014186442]]
    )
    let expectedGradInitialStateHiddenBackward = Tensor<Float>(
        [[0.00047474913,0.006052384,0.0042700856,-0.026273249]]
    )
    let expectedGradInitialStateCellBackward = Tensor<Float>(
        [[-0.014903592,-0.009901126,0.046779558,-0.0045849807]]
    )
    // END GENERATED CODE

    // Swaps the columns corresponding to "forget" and "update" gates.
    //
    // This translates between the keras ordering
    //   [input | forget | update | output]
    // and the swift-apis ordering
    //   [input | update | forget | output]
    func swapForgetUpdate(_ t: Tensor<Float>) -> Tensor<Float> {
      var split = t.split(count: 4, alongAxis: -1)
      split.swapAt(1, 2)
      return Tensor(concatenating: split, alongAxis: -1)
    }

    withTensorLeakChecking {
      var lstm = BidirectionalLSTM<Float>(
        LSTMCell(inputSize: 3, hiddenSize: 4),
        mergeFunction: LSTMCell<Float>.State.multiply
      )

      lstm.forward.cell.fusedWeight =
        swapForgetUpdate(Tensor(concatenating: [kernelForward, recurrentKernelForward]))
      lstm.forward.cell.fusedBias = swapForgetUpdate(biasForward)

      lstm.backward.cell.fusedWeight =
        swapForgetUpdate(Tensor(concatenating: [kernelBackward, recurrentKernelBackward]))
      lstm.backward.cell.fusedBias = swapForgetUpdate(biasBackward)

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }

      let initialForwardLayerState = LSTMCell<Float>.State(
        cell: initialStateCellForward,
        hidden: initialStateHiddenForward)
      let initialBackwardLayerState = LSTMCell<Float>.State(
        cell: initialStateCellBackward,
        hidden: initialStateHiddenBackward)

      let outputs = lstm(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      )
      assertEqual(
        Tensor(concatenating: outputs.map { $0.hidden }),
        expectedStates,
        accuracy: 1e-6)

      let (outputSum, (gradLSTM)) =
        valueWithGradient(at: lstm) { lstm in
          lstm.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).hidden.sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.forward.cell.fusedWeight,
        swapForgetUpdate(Tensor(concatenating: [expectedGradKernelForward, expectedGradRecurrentKernelForward])),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.forward.cell.fusedBias,
        swapForgetUpdate(expectedGradBiasForward),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.backward.cell.fusedWeight,
        swapForgetUpdate(Tensor(concatenating: [expectedGradKernelBackward, expectedGradRecurrentKernelBackward])),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.backward.cell.fusedBias,
        swapForgetUpdate(expectedGradBiasBackward),
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          lstm.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).hidden.sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward.hidden,
        expectedGradInitialStateHiddenForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward.cell,
        expectedGradInitialStateCellForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward.hidden,
        expectedGradInitialStateHiddenBackward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward.cell,
        expectedGradInitialStateCellBackward,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalLSTMStackMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_lstm.py --merge-mode none".
    // BEGIN GENERATED CODE
    let kernelForward = Tensor<Float>(
        [[-0.23379976,-0.32979083,0.03977573,0.06884748,-0.093649775,0.34596872,
          -0.0076143146,0.55984896,0.22111118,-0.4210437,0.23581362,0.1825394,
          0.08120936,-0.15200394,-0.08932975,0.14617139],
        [0.4650857,0.18167573,0.37479192,-0.46759078,-0.24752906,-0.5445049,
          0.2544219,0.29843968,0.20215267,0.03678304,0.28829694,-0.5086538,
          -0.5053389,0.2829352,-0.36783904,-0.2113666],
        [-0.23447555,-0.44898528,-0.3757536,0.3030774,0.096295714,0.541732,
          0.46623164,-0.4027334,-0.45229727,0.119446635,-0.36198467,0.020288646,
          0.5418573,-0.3643924,-0.5106659,0.10962713]]
    )
    let recurrentKernelForward = Tensor<Float>(
        [[0.22804272,-0.03794806,0.17161758,-0.5903252,-0.32132986,-0.028507836,
          -0.02729048,0.15502541,-0.29579595,0.041225288,-0.04219827,-0.22448958,
          0.51597697,-0.103077516,0.005849608,0.14667785],
        [0.09507164,-0.25853005,-0.22325921,0.499246,-0.3109755,0.01945003,
          0.33349884,-0.030728217,-0.2997637,0.4836203,0.035094753,-0.1599233,
          0.16274704,-0.0034662904,-0.18398553,-0.06985295],
        [-0.13635698,-0.51163936,-0.09270489,-0.12567669,0.41930002,-0.48658654,
          -0.12225373,-0.28107005,-0.004303295,0.16249746,0.024185713,-0.24801093,
          0.019665923,0.040028445,0.038016353,0.31107086],
        [0.14521983,0.63589036,0.04985115,0.17409262,0.18315184,-0.341375,
          -0.048293047,-0.037892636,0.08461128,0.19565175,-0.18345755,-0.12093313,
          0.27880925,0.39949074,-0.1981584,0.13265577]]
    )
    let biasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    )
    let kernelBackward = Tensor<Float>(
        [[-0.3425501,0.26993138,-0.013709605,0.42182362,-0.10439563,-0.5456007,
          0.23550463,-0.15115345,0.09087402,0.45050424,-0.41379565,0.050539136,
          0.22132623,-0.55731493,-0.15275678,0.07351905],
        [-0.20979714,0.1675657,0.5581859,-0.50259715,0.4392975,-0.26118547,
          -0.06672484,0.26642138,-0.37113252,-0.29193833,0.5320031,0.54281443,
          -0.5196755,-0.013125479,0.006757021,0.5087437],
        [0.41886348,0.16300112,0.15921205,0.3881111,0.34322268,0.0014763474,
          0.2432934,0.32898188,0.26853347,-0.48425156,-0.3587765,-0.19107765,
          0.5605115,-0.18283507,0.13686264,0.008577526]]
    )
    let recurrentKernelBackward = Tensor<Float>(
        [[0.332492,0.3287107,0.013327966,-0.15466748,0.18726069,-0.18434268,
          -0.13676699,-0.06803291,-0.55743134,0.2967973,-0.09094952,-0.12427574,
          -0.27146038,-0.37563083,0.15853615,0.049586363],
        [-0.23724489,0.016642023,-0.2502587,0.48801288,0.4526761,0.012872117,
          -0.0565288,-0.113935634,-0.08418312,0.2147764,0.16762011,0.058674254,
          0.16494237,0.1398105,0.25171497,0.4759469],
        [-0.0038104642,-0.2387862,0.22545648,-0.28967497,0.5137987,-0.12194693,
          -0.17891966,0.33095407,-0.003036025,-0.4760634,0.06607926,-0.2880439,
          0.09167388,-0.10248731,-0.025532419,0.23298186],
        [0.06349304,0.13959168,-0.2960539,-0.20912763,0.18433903,-0.2516703,
          -0.007072272,-0.09261524,0.37721395,0.09471353,-0.3878575,0.20997167,
          -0.18772967,0.004912289,-0.48590505,0.3492866]]
    )
    let biasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.90076077,-0.7541278,-0.025081873],
          [0.37973404,-0.26513243,0.10235357],
          [0.3545239,-0.14615154,-0.540749],
          [-0.70222735,-0.34035724,-0.8451414]]]
    )
    let initialStateHiddenForward = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialStateCellForward = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let initialStateHiddenBackward = Tensor<Float>(
        [[0.60601985,0.77093506,0.7210845,-0.012122989]]
    )
    let initialStateCellBackward = Tensor<Float>(
        [[0.06269443,-0.80621815,0.47859335,0.13418436]]
    )
    let expectedSum = Tensor<Float>(
        -0.045482457
    )
    let expectedStatesForward = Tensor<Float>(
        [[[-0.44356138,-0.061652392,0.24647687,-0.19823068],
          [-0.2737949,-0.07248981,0.19220428,-0.08825301],
          [-0.124358594,-0.1213435,0.21946031,-0.025490183],
          [-0.048278596,-0.028744407,0.1743442,-0.011268781]]]
    )
    let expectedStatesBackward = Tensor<Float>(
        [[[0.05788271,-0.12734284,0.033790566,-0.11494893],
          [0.0008733621,-0.06352448,0.044504695,-0.060439102],
          [-0.061879173,-0.12179416,0.1418245,-0.016867002],
          [-0.07999351,-0.29541627,0.25795794,-0.014083028]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those four and test it.
    // let expectedFinalStateHiddenForward = Tensor<Float>(
    //     [[-0.048278596,-0.028744407,0.1743442,-0.011268781]]
    // )
    // let expectedFinalStateCellForward = Tensor<Float>(
    //     [[-0.123089634,-0.049444616,0.27142566,-0.023337554]]
    // )
    // let expectedFinalStateHiddenBackward = Tensor<Float>(
    //     [[0.05788271,-0.12734284,0.033790566,-0.11494893]]
    // )
    // let expectedFinalStateCellBackward = Tensor<Float>(
    //     [[0.106476925,-0.20731302,0.0632082,-0.31257194]]
    // )
    let expectedGradKernelForward = Tensor<Float>(
        [[0.0029497184,-0.05191034,0.02407415,0.000304088,0.014214157,0.046752453,
          -0.050661147,0.031548906,-0.12825678,-0.26617724,-0.16949388,
          -0.14972453,0.018881105,0.0066204695,-0.040960398,0.002944462],
        [-0.005592435,-0.019203003,0.010193659,-0.014379387,0.03603935,
          0.04006479,-0.05743648,0.04225596,-0.14608386,-0.28398833,-0.20509121,
          -0.21936974,0.00064499304,0.0065013287,-0.0188484,0.0005266868],
        [-0.030478455,-0.0063452474,-0.019162579,-0.008174095,0.03241433,
          0.035240535,-0.06443699,0.008317955,-0.21671411,-0.38373917,-0.36592808,
          -0.26295128,0.020803409,0.012763699,-0.050191153,0.0049239486]]
    )
    let expectedGradRecurrentKernelForward = Tensor<Float>(
        [[-0.009446431,0.0046872147,-0.0034881886,-0.013322809,0.028175268,
          0.017745323,-0.035205625,0.02300223,-0.097775005,-0.17619048,
          -0.15024684,-0.15604895,-0.0037074587,0.0044266684,-0.006576741,
          -0.000016547943],
        [-0.007838752,0.00966959,-0.011096001,0.0011524465,-0.0017994214,
          -0.0070223105,0.0018404042,-0.013339203,-0.013871463,-0.009875648,
          -0.04120832,-0.005587943,0.0054747937,0.0015693338,-0.007709676,
          0.0012488235],
        [0.017221576,-0.023692459,0.024221376,-0.00008906424,0.00012817606,
          0.014727241,-0.0016071051,0.025199397,0.034627017,0.029167429,
          0.09496009,0.025977343,-0.009273196,-0.0034023244,0.013966144,
          -0.0023199664],
        [-0.012845716,0.032039955,-0.026622673,0.0055777263,-0.0153835025,
          -0.032603078,0.027565094,-0.041101806,0.03523525,0.104707435,
          0.0075816847,0.07301338,0.0071937367,0.000036919257,-0.0029361085,
          0.0017134396]]
    )
    let expectedGradBiasForward = Tensor<Float>(
        [0.041717105,0.0077266824,0.011825588,0.041857336,-0.10179798,-0.08539298,
        0.15041551,-0.08090452,0.43130994,0.7878589,0.66939783,0.6281428,
        -0.005506225,-0.021247003,0.056803677,-0.0033201245]
    )
    let expectedGradKernelBackward = Tensor<Float>(
        [[0.034572553,-0.014308241,-0.037737265,0.028604513,-0.002552184,
          0.042288657,-0.036268644,-0.011494972,-0.09629306,-0.12330007,
          -0.09965681,-0.15201326,0.033916343,0.08542741,-0.07855631,0.004139293],
        [0.016756708,-0.0069349525,-0.018290589,0.013864105,-0.0012369987,
          0.020496568,-0.017578775,-0.0055714105,-0.046671554,-0.059761375,
          -0.0483019,-0.07367815,0.016438654,0.041405164,-0.03807486,0.0020062423],
        [0.041608598,-0.017220188,-0.04541738,0.03442597,-0.0030715927,
          0.050895046,-0.04364987,-0.013834375,-0.11589018,-0.14839353,-0.1199385,
          -0.18295029,0.04081884,0.10281321,-0.09454372,0.0049817027]]
    )
    let expectedGradRecurrentKernelBackward = Tensor<Float>(
        [[-0.029835997,0.012347964,0.032567132,-0.024685599,0.0022025262,
          -0.03649497,0.031299718,0.009920122,0.08310059,0.106407546,0.08600349,
          0.13118693,-0.02926969,-0.07372357,0.06779383,-0.0035721958],
        [-0.03795522,0.015708195,0.041429576,-0.03140325,0.0028018963,
          -0.046426285,0.039817262,0.012619669,0.10571462,0.13536406,0.109407485,
          0.16688661,-0.0372348,-0.09378585,0.08624245,-0.0045442916],
        [-0.03550094,0.014692465,0.038750634,-0.02937264,0.0026207187,
          -0.043424245,0.037242576,0.011803649,0.098878846,0.12661108,0.10233293,
          0.15609531,-0.03482711,-0.08772142,0.080665804,-0.0042504463],
        [0.00059684756,-0.0002470121,-0.00065148197,0.00049381756,
          -0.000044059943,0.00073005544,-0.0006261282,-0.00019844486,
          -0.0016623671,-0.002128606,-0.0017204377,-0.0026242996,0.00058551895,
          0.0014747867,-0.0013561664,0.000071459195]]
    )
    let expectedGradBiasBackward = Tensor<Float>(
        [-0.049232706,0.02037551,0.053739384,-0.04073398,0.0036344128,-0.06022075,
        0.05164801,0.016369302,0.1371252,0.17558426,0.1419153,0.21647298,
        -0.048298236,-0.121652074,0.11186734,-0.0058945194]
    )
    let expectedGradX = Tensor<Float>(
        [[[-0.054938864,-0.022083957,-0.004919187],
          [0.01124474,-0.020291314,-0.02433711],
          [0.017743908,0.02348721,-0.07303476],
          [0.113909885,0.27435783,-0.3467198]]]
    )
    let expectedGradInitialStateHiddenForward = Tensor<Float>(
        [[-0.046731517,0.05862768,0.0007173242,0.035859868]]
    )
    let expectedGradInitialStateCellForward = Tensor<Float>(
        [[0.09301593,0.20037422,0.123320214,0.10345338]]
    )
    let expectedGradInitialStateHiddenBackward = Tensor<Float>(
        [[0.013064705,0.037812762,-0.10854563,0.017091252]]
    )
    let expectedGradInitialStateCellBackward = Tensor<Float>(
        [[0.3086998,0.34445396,0.26831728,0.4068746]]
    )
    // END GENERATED CODE

    // Swaps the columns corresponding to "forget" and "update" gates.
    //
    // This translates between the keras ordering
    //   [input | forget | update | output]
    // and the swift-apis ordering
    //   [input | update | forget | output]
    func swapForgetUpdate(_ t: Tensor<Float>) -> Tensor<Float> {
      var split = t.split(count: 4, alongAxis: -1)
      split.swapAt(1, 2)
      return Tensor(concatenating: split, alongAxis: -1)
    }

    withTensorLeakChecking {
      var lstm = BidirectionalLSTM<Float>(
        LSTMCell(inputSize: 3, hiddenSize: 4),
        mergeFunction: LSTMCell<Float>.State.stack
      )

      lstm.forward.cell.fusedWeight =
        swapForgetUpdate(Tensor(concatenating: [kernelForward, recurrentKernelForward]))
      lstm.forward.cell.fusedBias = swapForgetUpdate(biasForward)

      lstm.backward.cell.fusedWeight =
        swapForgetUpdate(Tensor(concatenating: [kernelBackward, recurrentKernelBackward]))
      lstm.backward.cell.fusedBias = swapForgetUpdate(biasBackward)

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }

      let initialForwardLayerState = LSTMCell<Float>.State(
        cell: initialStateCellForward,
        hidden: initialStateHiddenForward)
      let initialBackwardLayerState = LSTMCell<Float>.State(
        cell: initialStateCellBackward,
        hidden: initialStateHiddenBackward)

      let outputs = lstm(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      ).map { $0.hidden.unstacked(alongAxis: 0) }
      assertEqual(
        Tensor(concatenating: outputs.map { $0[0] }),
        expectedStatesForward,
        accuracy: 1e-6)
      assertEqual(
        Tensor(concatenating: outputs.map { $0[1] }),
        expectedStatesBackward,
        accuracy: 1e-6)

      let (outputSum, (gradLSTM)) =
        valueWithGradient(at: lstm) { lstm in
          lstm.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).hidden.sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.forward.cell.fusedWeight,
        swapForgetUpdate(Tensor(concatenating: [expectedGradKernelForward, expectedGradRecurrentKernelForward])),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.forward.cell.fusedBias,
        swapForgetUpdate(expectedGradBiasForward),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.backward.cell.fusedWeight,
        swapForgetUpdate(Tensor(concatenating: [expectedGradKernelBackward, expectedGradRecurrentKernelBackward])),
        accuracy: 1e-6)
      assertEqual(
        gradLSTM.backward.cell.fusedBias,
        swapForgetUpdate(expectedGradBiasBackward),
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          lstm.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).hidden.sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward.hidden,
        expectedGradInitialStateHiddenForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward.cell,
        expectedGradInitialStateCellForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward.hidden,
        expectedGradInitialStateHiddenBackward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward.cell,
        expectedGradInitialStateCellBackward,
        accuracy: 1e-6)
    }
  }

  func testGRU() {
    // Data generated by "Utilities/ReferenceImplementations/gru.py".
    // BEGIN GENERATED CODE
    let updateKernel = Tensor<Float>(
        [[-0.26313293,-0.37116733,0.044766128,0.07748526],
        [0.09139812,-0.17107481,-0.1005373,0.16451049],
        [0.22751534,0.04139793,0.32446748,-0.572471]]
    )
    let resetKernel = Tensor<Float>(
        [[-0.10539937,0.38937503,-0.008569658,0.6300892],
        [0.5234366,0.20446932,0.42181438,-0.5262561],
        [-0.5687402,0.31843305,-0.4139892,-0.23788527]]
    )
    let outputKernel = Tensor<Float>(
        [[0.24885243,-0.4738691,0.26539946,0.2054413],
        [-0.27858475,-0.6128201,0.28634238,0.33588284],
        [-0.2638935,-0.5053163,-0.42289674,0.34110242]]
    )
    let updateRecurrentKernel = Tensor<Float>(
        [[0.27225327,-0.04530505,0.20488906,-0.70477146],
        [0.1233796,-0.2725925,-0.21331471,0.45468533],
        [-0.12901938,-0.5375645,-0.07981183,-0.18903717],
        [0.22767149,0.6950712,0.06780873,0.12117213]]
    )
    let resetRecurrentKernel = Tensor<Float>(
        [[-0.38362607,-0.03403465,-0.032581285,0.18508019],
        [-0.35768405,0.017118583,0.34339952,-0.015158445],
        [0.42593452,-0.51679564,-0.13895734,-0.2845649],
        [0.15196994,-0.4617733,-0.03934008,-0.038054474]]
    )
    let outputRecurrentKernel = Tensor<Float>(
        [[-0.3531419,0.04921763,-0.050379246,-0.26801136],
        [-0.34328082,0.5067098,0.031886574,-0.19034609],
        [-0.020022772,0.16366726,0.021551719,-0.27538475],
        [0.0029521734,0.31402355,-0.22738007,-0.23688483]]
    )
    let updateBias = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetBias = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputBias = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateRecurrentBias = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetRecurrentBias = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputRecurrentBias = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.5643544,0.44471443,-0.022586703],
          [0.6949582,-0.17199272,-0.8988821],
          [0.38799608,-0.24902666,0.14971566],
          [0.7422098,-0.6817321,0.083263636]]]
    )
    let initialState = Tensor<Float>(
        [[-0.71174276,-0.39879853,0.7789655,-0.1812765]]
    )
    let expectedSum = Tensor<Float>(
        0.2622946
    )
    let expectedStates = Tensor<Float>(
        [[[-0.31175402,-0.20596652,0.36340874,-0.075575344],
          [0.21953434,0.03525635,0.43513912,-0.11752244],
          [0.1467207,-0.031999923,0.21731833,-0.0734479],
          [0.24567646,0.0015162053,0.10337216,-0.08827023]]]
    )
    let expectedFinalState = Tensor<Float>(
        [[0.24567646,0.0015162053,0.10337216,-0.08827023]]
    )
    let expectedGradX = Tensor<Float>(
        [[[0.0044808714,-0.0021028928,-0.030270718],
          [0.008932473,-0.025464162,-0.09908461],
          [0.011008073,-0.06111902,-0.16229588],
          [0.1040091,-0.177594,-0.41953978]]]
    )
    let expectedGradInitialState = Tensor<Float>(
        [[0.005508149,0.03993896,0.053400937,0.037300646]]
    )
    let expectedGradUpdateKernel = Tensor<Float>(
        [[-0.040157042,-0.0136785135,0.050436385,0.007616069],
        [0.029918717,0.006519991,-0.045159567,-0.005434308],
        [0.013227239,0.01569719,0.019206645,-0.004196606]]
    )
    let expectedGradResetKernel = Tensor<Float>(
        [[-0.004324538,0.0015231472,0.0015622675,-0.010165867],
        [0.00435052,-0.0018293709,-0.00096766773,0.008604034],
        [-0.0024347848,0.0026376988,-0.00044454157,-0.0025067823]]
    )
    let expectedGradOutputKernel = Tensor<Float>(
        [[0.4559958,0.5759123,0.4702625,0.51912045],
        [-0.38423252,-0.45619243,-0.36462966,-0.4307824],
        [0.025001988,-0.0074325786,-0.018035406,0.030591251]]
    )
    let expectedGradUpdateRecurrentKernel = Tensor<Float>(
        [[0.0015186802,0.010627928,0.008849502,0.0014116366],
        [0.0055325227,0.0070856367,-0.0066436627,0.0005385389],
        [-0.012196659,-0.00962431,0.050553232,-0.0037394736],
        [0.003526025,0.0022264342,-0.014122595,0.0008748602]]
    )
    let expectedGradResetRecurrentKernel = Tensor<Float>(
        [[-0.0021709674,0.0029862812,-0.0005916139,-0.0038731585],
        [-0.0003447486,0.0013247593,-0.0006560448,-0.00038965122],
        [-0.0019658464,-0.00036801025,0.0022314903,-0.004976805],
        [0.00065238454,-0.000011753844,-0.0005693683,0.0015112129]]
    )
    let expectedGradOutputRecurrentKernel = Tensor<Float>(
        [[0.03295932,0.04461184,0.012124008,0.06561776],
        [-0.010434465,-0.022276893,-0.025632525,-0.019136468],
        [0.08186121,0.15295859,0.12944305,0.1582405],
        [-0.024327286,-0.04418066,-0.035645414,-0.047330596]]
    )
    let expectedGradUpdateBias = Tensor<Float>(
        [-0.052024383,-0.026385367,0.12956381,-0.0010043234]
    )
    let expectedGradResetBias = Tensor<Float>(
        [-0.0070920927,0.0005423294,0.0050295107,-0.01650119]
    )
    let expectedGradOutputBias = Tensor<Float>(
        [0.6879913,0.98991126,0.85718745,0.871734]
    )
    let expectedGradUpdateRecurrentBias = Tensor<Float>(
        [-0.052024383,-0.026385367,0.12956381,-0.0010043234]
    )
    let expectedGradResetRecurrentBias = Tensor<Float>(
        [-0.0070920927,0.0005423294,0.0050295107,-0.01650119]
    )
    let expectedGradOutputRecurrentBias = Tensor<Float>(
        [0.28977823,0.49616843,0.3776239,0.5536645]
    )
    // END GENERATED CODE
    withTensorLeakChecking {
      var gru = GRU<Float>(
        GRUCell(inputSize: 3, hiddenSize: 4)
      )

      XCTAssertEqual(gru.cell.updateKernel.shape, updateKernel.shape)
      XCTAssertEqual(gru.cell.updateRecurrentKernel.shape, updateRecurrentKernel.shape)
      XCTAssertEqual(gru.cell.resetKernel.shape, resetKernel.shape)
      XCTAssertEqual(gru.cell.resetRecurrentKernel.shape, resetRecurrentKernel.shape)
      XCTAssertEqual(gru.cell.outputKernel.shape, outputKernel.shape)
      XCTAssertEqual(gru.cell.outputRecurrentKernel.shape, outputRecurrentKernel.shape)
      XCTAssertEqual(gru.cell.updateBias.shape, updateBias.shape)
      XCTAssertEqual(gru.cell.updateRecurrentBias.shape, updateRecurrentBias.shape)
      XCTAssertEqual(gru.cell.resetBias.shape, resetBias.shape)
      XCTAssertEqual(gru.cell.resetRecurrentBias.shape, resetRecurrentBias.shape)
      XCTAssertEqual(gru.cell.outputBias.shape, outputBias.shape)
      XCTAssertEqual(gru.cell.outputRecurrentBias.shape, outputRecurrentBias.shape)

      gru.cell.updateKernel = updateKernel
      gru.cell.updateRecurrentKernel = updateRecurrentKernel
      gru.cell.resetKernel = resetKernel
      gru.cell.resetRecurrentKernel = resetRecurrentKernel
      gru.cell.outputKernel = outputKernel
      gru.cell.outputRecurrentKernel = outputRecurrentKernel

      gru.cell.updateBias = updateBias
      gru.cell.updateRecurrentBias = updateRecurrentBias
      gru.cell.resetBias = resetBias
      gru.cell.resetRecurrentBias = resetRecurrentBias
      gru.cell.outputBias = outputBias
      gru.cell.outputRecurrentBias = outputRecurrentBias

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }
      let initialState = GRUCell<Float>.State(initialState)

      let outputs = gru(inputs, initialState: initialState)
      assertEqual(
        Tensor(concatenating: outputs.map { $0 }),
        expectedStates,
        accuracy: 1e-6)
      assertEqual(
        outputs.last!,
        expectedFinalState,
        accuracy: 1e-6)

      let (outputSum, (gradGRU, gradInputs, gradInitialState)) =
        valueWithGradient(at: gru, inputs, initialState) { gru, inputs, initialState in
          gru.lastOutput(
            from: inputs,
            initialState: initialState
          ).sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialState,
        expectedGradInitialState,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.cell.updateKernel,
        expectedGradUpdateKernel,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.cell.resetKernel,
        expectedGradResetKernel,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.cell.outputKernel,
        expectedGradOutputKernel,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.cell.updateRecurrentKernel,
        expectedGradUpdateRecurrentKernel,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.cell.resetRecurrentKernel,
        expectedGradResetRecurrentKernel,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.cell.outputRecurrentKernel,
        expectedGradOutputRecurrentKernel,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.cell.updateBias,
        expectedGradUpdateBias,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.cell.resetBias,
        expectedGradResetBias,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.cell.outputBias,
        expectedGradOutputBias,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.cell.updateRecurrentBias,
        expectedGradUpdateRecurrentBias,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.cell.resetRecurrentBias,
        expectedGradResetRecurrentBias,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.cell.outputRecurrentBias,
        expectedGradOutputRecurrentBias,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalGRUSumMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_gru.py --merge-mode sum".
    // BEGIN GENERATED CODE
    let updateKernelForward = Tensor<Float>(
        [[-0.26313293,-0.37116733,0.044766128,0.07748526],
        [0.09139812,-0.17107481,-0.1005373,0.16451049],
        [0.22751534,0.04139793,0.32446748,-0.572471]]
    )
    let resetKernelForward = Tensor<Float>(
        [[-0.10539937,0.38937503,-0.008569658,0.6300892],
        [0.5234366,0.20446932,0.42181438,-0.5262561],
        [-0.5687402,0.31843305,-0.4139892,-0.23788527]]
    )
    let outputKernelForward = Tensor<Float>(
        [[0.24885243,-0.4738691,0.26539946,0.2054413],
        [-0.27858475,-0.6128201,0.28634238,0.33588284],
        [-0.2638935,-0.5053163,-0.42289674,0.34110242]]
    )
    let updateRecurrentKernelForward = Tensor<Float>(
        [[0.27225327,-0.04530505,0.20488906,-0.70477146],
        [0.1233796,-0.2725925,-0.21331471,0.45468533],
        [-0.12901938,-0.5375645,-0.07981183,-0.18903717],
        [0.22767149,0.6950712,0.06780873,0.12117213]]
    )
    let resetRecurrentKernelForward = Tensor<Float>(
        [[-0.38362607,-0.03403465,-0.032581285,0.18508019],
        [-0.35768405,0.017118583,0.34339952,-0.015158445],
        [0.42593452,-0.51679564,-0.13895734,-0.2845649],
        [0.15196994,-0.4617733,-0.03934008,-0.038054474]]
    )
    let outputRecurrentKernelForward = Tensor<Float>(
        [[-0.3531419,0.04921763,-0.050379246,-0.26801136],
        [-0.34328082,0.5067098,0.031886574,-0.19034609],
        [-0.020022772,0.16366726,0.021551719,-0.27538475],
        [0.0029521734,0.31402355,-0.22738007,-0.23688483]]
    )
    let updateBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateKernelBackward = Tensor<Float>(
        [[-0.38552743,0.30379778,-0.015429676,0.47474676],
        [0.24909449,-0.6272373,-0.17192209,0.08274299],
        [-0.41769588,-0.32856578,0.5987498,0.6109175]]
    )
    let resetKernelBackward = Tensor<Float>(
        [[-0.11749339,-0.61405337,0.26505172,-0.17011762],
        [-0.23611891,0.18858898,0.6282174,-0.5656544],
        [-0.5848755,-0.014772236,0.007604778,0.5725722]]
    )
    let outputKernelBackward = Tensor<Float>(
        [[0.10227531,0.5070258,-0.46571168,0.056879938],
        [0.49441308,-0.29395455,-0.07509631,0.29984736],
        [0.47141522,0.18345171,0.1791873,0.4368046]]
    )
    let updateRecurrentKernelBackward = Tensor<Float>(
        [[0.3819934,0.37764925,0.015312237,-0.17769441],
        [-0.30984464,0.0024376288,-0.3083459,0.6083385],
        [-0.035942614,-0.24049878,0.19538112,-0.22652327],
        [0.06525053,0.14899497,-0.36888042,-0.2775943]]
    )
    let resetRecurrentKernelBackward = Tensor<Float>(
        [[0.2151401,-0.21178766,-0.15712889,-0.078161664],
        [0.54615974,0.025927158,-0.061987586,-0.13631916],
        [0.61778206,-0.12979838,-0.19998191,0.32677993],
        [0.3122218,-0.3348464,-0.029183073,-0.08691845]]
    )
    let outputRecurrentKernelBackward = Tensor<Float>(
        [[-0.64042187,0.34098455,-0.10449011,-0.14277793],
        [-0.07292092,0.24773169,0.21102233,0.07893403],
        [-0.029284794,-0.45729926,0.09502104,-0.29511455],
        [0.49253088,0.07420389,-0.48452082,0.2440229]]
    )
    let updateBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.90076077,-0.7541278,-0.025081873],
          [0.37973404,-0.26513243,0.10235357],
          [0.3545239,-0.14615154,-0.540749],
          [-0.70222735,-0.34035724,-0.8451414]]]
    )
    let initialForwardLayerState = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialBackwardLayerState = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let expectedSum = Tensor<Float>(
        -1.0636934
    )
    let expectedStates = Tensor<Float>(
        [[[-0.47425687,0.13057935,-0.28099662,0.09199959],
          [-0.23789163,0.07386574,-0.09076065,-0.07140374],
          [-0.37719586,-0.22458386,0.3951493,-0.43688574],
          [-0.6900477,-0.26849592,0.795504,-0.9006538]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those two and test it.
    // let expectedFinalStateForward = Tensor<Float>(
    //     [[0.08708767,0.49491018,0.019275323,-0.13219076]]
    // )
    // let expectedFinalStateBackward = Tensor<Float>(
    //     [[-0.328244,-0.30765843,0.35560328,-0.3700244]]
    // )
    let expectedGradX = Tensor<Float>(
        [[[0.016884286,0.031954587,-0.034630362],
          [-0.030145949,-0.032830622,-0.09927184],
          [0.010677528,-0.06555416,-0.24680781],
          [0.36794654,0.22227536,0.035613686]]]
    )
    let expectedGradInitialStateForward = Tensor<Float>(
        [[-0.07189826,0.26491642,0.021249572,0.09973241]]
    )
    let expectedGradUpdateKernelForward = Tensor<Float>(
        [[-0.014633494,0.11262925,-0.0124673275,-0.14178306],
        [-0.0015309703,0.043804236,0.026814371,-0.117686585],
        [0.0006633089,0.10263121,0.05096156,-0.15763283]]
    )
    let expectedGradResetKernelForward = Tensor<Float>(
        [[0.010167458,0.0012375548,0.0016463948,0.0056149918],
        [0.0059102885,-0.007958793,0.0032499032,0.0029476206],
        [0.01513045,-0.011363849,0.0046043447,0.0065481835]]
    )
    let expectedGradOutputKernelForward = Tensor<Float>(
        [[-0.27720484,0.01121016,-0.3119189,-0.15031564],
        [-0.17816868,-0.1504507,-0.26702198,-0.19132887],
        [-0.47820407,-0.2667625,-0.5929085,-0.3111684]]
    )
    let expectedGradUpdateRecurrentKernelForward = Tensor<Float>(
        [[0.0013306543,-0.01375161,0.0037764863,-0.0035531288],
        [-0.0023589546,-0.024710568,-0.038062952,0.096005075],
        [0.0055330433,-0.0055813976,0.042736366,-0.09101869],
        [-0.002846387,-0.016188195,-0.037352126,0.11634412]]
    )
    let expectedGradResetRecurrentKernelForward = Tensor<Float>(
        [[-0.0016176911,-0.0013255397,0.0003207235,-0.0006573718],
        [-0.0055763996,0.010337391,-0.0034352662,-0.0023184759],
        [0.0016612073,-0.011832552,0.003696582,0.000721002],
        [-0.0028628546,0.010542515,-0.0038019938,-0.0016213771]]
    )
    let expectedGradOutputRecurrentKernelForward = Tensor<Float>(
        [[0.027280934,-0.011064235,0.02276091,-0.004477814],
        [0.09789748,0.09476159,0.15628886,0.10902606],
        [-0.032271817,-0.10743487,-0.093189105,-0.10548531],
        [0.05016677,0.09146461,0.10986484,0.10788684]]
    )
    let expectedGradUpdateBiasForward = Tensor<Float>(
        [-0.007191318,-0.0988105,-0.11562109,0.29936516]
    )
    let expectedGradResetBiasForward = Tensor<Float>(
        [-0.019118859,0.030049099,-0.010341871,-0.0081044715]
    )
    let expectedGradOutputBiasForward = Tensor<Float>(
        [0.61583984,0.5973234,0.9004895,0.605246]
    )
    let expectedGradUpdateRecurrentBiasForward = Tensor<Float>(
        [-0.007191318,-0.0988105,-0.11562109,0.29936516]
    )
    let expectedGradResetRecurrentBiasForward = Tensor<Float>(
        [-0.019118859,0.030049099,-0.010341871,-0.0081044715]
    )
    let expectedGradOutputRecurrentBiasForward = Tensor<Float>(
        [0.33583188,0.2782093,0.5110848,0.33206433]
    )
    let expectedGradInitialStateBackward = Tensor<Float>(
        [[0.436919,0.39781284,0.48556933,0.34003386]]
    )
    let expectedGradUpdateKernelBackward = Tensor<Float>(
        [[0.080392286,-0.02186797,-0.106636874,0.05397816],
        [0.038964726,-0.010599021,-0.051685017,0.026162265],
        [0.09675335,-0.026318438,-0.12833911,0.06496355]]
    )
    let expectedGradResetKernelBackward = Tensor<Float>(
        [[-0.006548254,0.033489596,-0.03336197,0.033302337],
        [-0.0031738235,0.016231818,-0.016169962,0.016141057],
        [-0.007880924,0.040305246,-0.04015165,0.040079877]]
    )
    let expectedGradOutputKernelBackward = Tensor<Float>(
        [[-0.2241628,-0.14797445,-0.24141216,-0.2907229],
        [-0.108647764,-0.071720615,-0.11700823,-0.14090827],
        [-0.26978338,-0.17808953,-0.29054326,-0.3498895]]
    )
    let expectedGradUpdateRecurrentKernelBackward = Tensor<Float>(
        [[0.11208751,-0.03048957,-0.14867921,0.07525943],
        [0.0779456,-0.021202432,-0.10339145,0.05233537],
        [-0.118453026,0.03222109,0.15712279,-0.07953346],
        [0.12208392,-0.03320875,-0.16193901,0.08197136]]
    )
    let expectedGradResetRecurrentKernelBackward = Tensor<Float>(
        [[-0.009129949,0.046693105,-0.046515163,0.04643202],
        [-0.006348962,0.032470364,-0.03234662,0.032288805],
        [0.009648444,-0.049344834,0.04915679,-0.04906892],
        [-0.009944194,0.05085738,-0.050663568,0.05057301]]
    )
    let expectedGradOutputRecurrentKernelBackward = Tensor<Float>(
        [[-0.18567991,-0.14199756,-0.13620174,-0.24541472],
        [-0.12912172,-0.09874504,-0.09471462,-0.17066127],
        [0.19622478,0.1500617,0.14393671,0.25935197],
        [-0.20223959,-0.15466146,-0.14834875,-0.26730177]]
    )
    let expectedGradUpdateBiasBackward = Tensor<Float>(
        [-0.11448185,0.03114087,0.1518552,-0.076867074]
    )
    let expectedGradResetBiasBackward = Tensor<Float>(
        [0.009324977,-0.047690533,0.04750879,-0.04742387]
    )
    let expectedGradOutputBiasBackward = Tensor<Float>(
        [0.31921685,0.21072157,0.34378064,0.4140011]
    )
    let expectedGradUpdateRecurrentBiasBackward = Tensor<Float>(
        [-0.11448185,0.03114087,0.1518552,-0.076867074]
    )
    let expectedGradResetRecurrentBiasBackward = Tensor<Float>(
        [0.009324977,-0.047690533,0.04750879,-0.04742387]
    )
    let expectedGradOutputRecurrentBiasBackward = Tensor<Float>(
        [0.18964629,0.14503083,0.13911119,0.2506571]
    )
    // END GENERATED CODE
    withTensorLeakChecking {
      var gru = BidirectionalGRU<Float>(
        GRUCell(inputSize: 3, hiddenSize: 4),
        mergeFunction: GRUCell<Float>.State.sum
      )

      XCTAssertEqual(gru.forward.cell.updateKernel.shape, updateKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.updateRecurrentKernel.shape, updateRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.resetKernel.shape, resetKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.resetRecurrentKernel.shape, resetRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.outputKernel.shape, outputKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.outputRecurrentKernel.shape, outputRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.updateBias.shape, updateBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.updateRecurrentBias.shape, updateRecurrentBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.resetBias.shape, resetBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.resetRecurrentBias.shape, resetRecurrentBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.outputBias.shape, outputBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.outputRecurrentBias.shape, outputRecurrentBiasForward.shape)

      XCTAssertEqual(gru.backward.cell.updateKernel.shape, updateKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateRecurrentKernel.shape, updateRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetKernel.shape, resetKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetRecurrentKernel.shape, resetRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputKernel.shape, outputKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputRecurrentKernel.shape, outputRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateBias.shape, updateBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateRecurrentBias.shape, updateRecurrentBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetBias.shape, resetBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetRecurrentBias.shape, resetRecurrentBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputBias.shape, outputBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputRecurrentBias.shape, outputRecurrentBiasBackward.shape)

      gru.forward.cell.updateKernel = updateKernelForward
      gru.forward.cell.updateRecurrentKernel = updateRecurrentKernelForward
      gru.forward.cell.resetKernel = resetKernelForward
      gru.forward.cell.resetRecurrentKernel = resetRecurrentKernelForward
      gru.forward.cell.outputKernel = outputKernelForward
      gru.forward.cell.outputRecurrentKernel = outputRecurrentKernelForward

      gru.forward.cell.updateBias = updateBiasForward
      gru.forward.cell.updateRecurrentBias = updateRecurrentBiasForward
      gru.forward.cell.resetBias = resetBiasForward
      gru.forward.cell.resetRecurrentBias = resetRecurrentBiasForward
      gru.forward.cell.outputBias = outputBiasForward
      gru.forward.cell.outputRecurrentBias = outputRecurrentBiasForward

      gru.backward.cell.updateKernel = updateKernelBackward
      gru.backward.cell.updateRecurrentKernel = updateRecurrentKernelBackward
      gru.backward.cell.resetKernel = resetKernelBackward
      gru.backward.cell.resetRecurrentKernel = resetRecurrentKernelBackward
      gru.backward.cell.outputKernel = outputKernelBackward
      gru.backward.cell.outputRecurrentKernel = outputRecurrentKernelBackward

      gru.backward.cell.updateBias = updateBiasBackward
      gru.backward.cell.updateRecurrentBias = updateRecurrentBiasBackward
      gru.backward.cell.resetBias = resetBiasBackward
      gru.backward.cell.resetRecurrentBias = resetRecurrentBiasBackward
      gru.backward.cell.outputBias = outputBiasBackward
      gru.backward.cell.outputRecurrentBias = outputRecurrentBiasBackward

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }
      let initialForwardLayerState = GRUCell<Float>.State(initialForwardLayerState)
      let initialBackwardLayerState = GRUCell<Float>.State(initialBackwardLayerState)

      let outputs = gru(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      )
      assertEqual(
        Tensor(concatenating: outputs),
        expectedStates,
        accuracy: 1e-6)

      let (outputSum, (gradGRU)) =
        valueWithGradient(at: gru) { gru in
          gru.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateKernel,
        expectedGradUpdateKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetKernel,
        expectedGradResetKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputKernel,
        expectedGradOutputKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateRecurrentKernel,
        expectedGradUpdateRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetRecurrentKernel,
        expectedGradResetRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputRecurrentKernel,
        expectedGradOutputRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateBias,
        expectedGradUpdateBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetBias,
        expectedGradResetBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputBias,
        expectedGradOutputBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateRecurrentBias,
        expectedGradUpdateRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetRecurrentBias,
        expectedGradResetRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputRecurrentBias,
        expectedGradOutputRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateKernel,
        expectedGradUpdateKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetKernel,
        expectedGradResetKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputKernel,
        expectedGradOutputKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateRecurrentKernel,
        expectedGradUpdateRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetRecurrentKernel,
        expectedGradResetRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputRecurrentKernel,
        expectedGradOutputRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateBias,
        expectedGradUpdateBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetBias,
        expectedGradResetBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputBias,
        expectedGradOutputBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateRecurrentBias,
        expectedGradUpdateRecurrentBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetRecurrentBias,
        expectedGradResetRecurrentBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputRecurrentBias,
        expectedGradOutputRecurrentBiasBackward,
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          gru.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward,
        expectedGradInitialStateForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward,
        expectedGradInitialStateBackward,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalGRUConcatenateMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_gru.py --merge-mode concat".
    // BEGIN GENERATED CODE
    let updateKernelForward = Tensor<Float>(
        [[-0.26313293,-0.37116733,0.044766128,0.07748526],
        [0.09139812,-0.17107481,-0.1005373,0.16451049],
        [0.22751534,0.04139793,0.32446748,-0.572471]]
    )
    let resetKernelForward = Tensor<Float>(
        [[-0.10539937,0.38937503,-0.008569658,0.6300892],
        [0.5234366,0.20446932,0.42181438,-0.5262561],
        [-0.5687402,0.31843305,-0.4139892,-0.23788527]]
    )
    let outputKernelForward = Tensor<Float>(
        [[0.24885243,-0.4738691,0.26539946,0.2054413],
        [-0.27858475,-0.6128201,0.28634238,0.33588284],
        [-0.2638935,-0.5053163,-0.42289674,0.34110242]]
    )
    let updateRecurrentKernelForward = Tensor<Float>(
        [[0.27225327,-0.04530505,0.20488906,-0.70477146],
        [0.1233796,-0.2725925,-0.21331471,0.45468533],
        [-0.12901938,-0.5375645,-0.07981183,-0.18903717],
        [0.22767149,0.6950712,0.06780873,0.12117213]]
    )
    let resetRecurrentKernelForward = Tensor<Float>(
        [[-0.38362607,-0.03403465,-0.032581285,0.18508019],
        [-0.35768405,0.017118583,0.34339952,-0.015158445],
        [0.42593452,-0.51679564,-0.13895734,-0.2845649],
        [0.15196994,-0.4617733,-0.03934008,-0.038054474]]
    )
    let outputRecurrentKernelForward = Tensor<Float>(
        [[-0.3531419,0.04921763,-0.050379246,-0.26801136],
        [-0.34328082,0.5067098,0.031886574,-0.19034609],
        [-0.020022772,0.16366726,0.021551719,-0.27538475],
        [0.0029521734,0.31402355,-0.22738007,-0.23688483]]
    )
    let updateBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateKernelBackward = Tensor<Float>(
        [[-0.38552743,0.30379778,-0.015429676,0.47474676],
        [0.24909449,-0.6272373,-0.17192209,0.08274299],
        [-0.41769588,-0.32856578,0.5987498,0.6109175]]
    )
    let resetKernelBackward = Tensor<Float>(
        [[-0.11749339,-0.61405337,0.26505172,-0.17011762],
        [-0.23611891,0.18858898,0.6282174,-0.5656544],
        [-0.5848755,-0.014772236,0.007604778,0.5725722]]
    )
    let outputKernelBackward = Tensor<Float>(
        [[0.10227531,0.5070258,-0.46571168,0.056879938],
        [0.49441308,-0.29395455,-0.07509631,0.29984736],
        [0.47141522,0.18345171,0.1791873,0.4368046]]
    )
    let updateRecurrentKernelBackward = Tensor<Float>(
        [[0.3819934,0.37764925,0.015312237,-0.17769441],
        [-0.30984464,0.0024376288,-0.3083459,0.6083385],
        [-0.035942614,-0.24049878,0.19538112,-0.22652327],
        [0.06525053,0.14899497,-0.36888042,-0.2775943]]
    )
    let resetRecurrentKernelBackward = Tensor<Float>(
        [[0.2151401,-0.21178766,-0.15712889,-0.078161664],
        [0.54615974,0.025927158,-0.061987586,-0.13631916],
        [0.61778206,-0.12979838,-0.19998191,0.32677993],
        [0.3122218,-0.3348464,-0.029183073,-0.08691845]]
    )
    let outputRecurrentKernelBackward = Tensor<Float>(
        [[-0.64042187,0.34098455,-0.10449011,-0.14277793],
        [-0.07292092,0.24773169,0.21102233,0.07893403],
        [-0.029284794,-0.45729926,0.09502104,-0.29511455],
        [0.49253088,0.07420389,-0.48452082,0.2440229]]
    )
    let updateBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.90076077,-0.7541278,-0.025081873],
          [0.37973404,-0.26513243,0.10235357],
          [0.3545239,-0.14615154,-0.540749],
          [-0.70222735,-0.34035724,-0.8451414]]]
    )
    let initialForwardLayerState = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialBackwardLayerState = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let expectedSum = Tensor<Float>(
        -1.0636933
    )
    let expectedStates = Tensor<Float>(
        [[[-0.14601289,0.4382378,-0.6365999,0.462024,-0.328244,-0.30765843,
          0.35560328,-0.37002444],
          [-0.024607085,0.29749468,-0.35770136,0.2921221,-0.21328454,-0.22362898,
          0.2669407,-0.36352584],
          [0.10575075,0.28877518,-0.03438914,0.12433955,-0.48294663,-0.51335907,
          0.4295384,-0.5612253],
          [0.087087676,0.49491018,0.019275311,-0.13219076,-0.7771353,-0.7634061,
          0.77622867,-0.768463]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those two and test it.
    // let expectedFinalStateForward = Tensor<Float>(
    //     [[0.087087676,0.49491018,0.019275311,-0.13219076]]
    // )
    // let expectedFinalStateBackward = Tensor<Float>(
    //     [[-0.328244,-0.30765843,0.35560328,-0.37002444]]
    // )
    let expectedGradX = Tensor<Float>(
        [[[0.016884288,0.031954587,-0.03463035],
          [-0.030145962,-0.03283064,-0.09927187],
          [0.010677524,-0.065554164,-0.24680781],
          [0.36794654,0.22227536,0.035613716]]]
    )
    let expectedGradInitialStateForward = Tensor<Float>(
        [[-0.07189825,0.26491642,0.021249587,0.0997324]]
    )
    let expectedGradUpdateKernelForward = Tensor<Float>(
        [[-0.014633494,0.11262925,-0.012467325,-0.14178306],
        [-0.0015309711,0.04380424,0.026814379,-0.11768658],
        [0.0006633061,0.10263122,0.050961565,-0.15763283]]
    )
    let expectedGradResetKernelForward = Tensor<Float>(
        [[0.010167458,0.001237556,0.0016463955,0.0056149918],
        [0.0059102885,-0.007958793,0.0032499037,0.00294762],
        [0.01513045,-0.011363849,0.0046043447,0.006548183]]
    )
    let expectedGradOutputKernelForward = Tensor<Float>(
        [[-0.27720484,0.011210177,-0.3119189,-0.15031563],
        [-0.17816868,-0.1504507,-0.267022,-0.19132887],
        [-0.47820407,-0.2667625,-0.59290844,-0.3111684]]
    )
    let expectedGradUpdateRecurrentKernelForward = Tensor<Float>(
        [[0.0013306541,-0.013751614,0.0037764856,-0.0035531214],
        [-0.0023589535,-0.02471057,-0.03806296,0.09600505],
        [0.0055330424,-0.0055813985,0.042736374,-0.09101869],
        [-0.0028463858,-0.01618819,-0.03735213,0.1163441]]
    )
    let expectedGradResetRecurrentKernelForward = Tensor<Float>(
        [[-0.0016176912,-0.0013255394,0.00032072357,-0.0006573719],
        [-0.005576399,0.010337391,-0.0034352664,-0.002318475],
        [0.0016612078,-0.011832552,0.0036965827,0.000721002],
        [-0.0028628546,0.010542516,-0.0038019943,-0.0016213768]]
    )
    let expectedGradOutputRecurrentKernelForward = Tensor<Float>(
        [[0.027280936,-0.011064235,0.022760917,-0.0044778096],
        [0.09789747,0.09476159,0.15628885,0.109026045],
        [-0.03227183,-0.10743488,-0.09318911,-0.10548531],
        [0.05016677,0.09146462,0.10986483,0.107886836]]
    )
    let expectedGradUpdateBiasForward = Tensor<Float>(
        [-0.0071913144,-0.09881051,-0.11562111,0.29936513]
    )
    let expectedGradResetBiasForward = Tensor<Float>(
        [-0.019118859,0.0300491,-0.010341872,-0.0081044715]
    )
    let expectedGradOutputBiasForward = Tensor<Float>(
        [0.61583984,0.5973235,0.9004895,0.605246]
    )
    let expectedGradUpdateRecurrentBiasForward = Tensor<Float>(
        [-0.0071913144,-0.09881051,-0.11562111,0.29936513]
    )
    let expectedGradResetRecurrentBiasForward = Tensor<Float>(
        [-0.019118859,0.0300491,-0.010341872,-0.0081044715]
    )
    let expectedGradOutputRecurrentBiasForward = Tensor<Float>(
        [0.33583188,0.27820933,0.5110848,0.33206433]
    )
    let expectedGradInitialStateBackward = Tensor<Float>(
        [[0.43691894,0.39781284,0.48556933,0.3400339]]
    )
    let expectedGradUpdateKernelBackward = Tensor<Float>(
        [[0.080392286,-0.02186797,-0.106636874,0.05397816],
        [0.038964726,-0.010599021,-0.051685017,0.026162265],
        [0.09675335,-0.026318438,-0.12833911,0.06496355]]
    )
    let expectedGradResetKernelBackward = Tensor<Float>(
        [[-0.006548254,0.033489596,-0.03336197,0.033302337],
        [-0.0031738235,0.016231818,-0.016169962,0.016141057],
        [-0.007880924,0.040305246,-0.04015165,0.040079877]]
    )
    let expectedGradOutputKernelBackward = Tensor<Float>(
        [[-0.22416282,-0.14797445,-0.24141216,-0.29072288],
        [-0.10864778,-0.071720615,-0.11700823,-0.14090827],
        [-0.2697834,-0.17808953,-0.29054326,-0.34988946]]
    )
    let expectedGradUpdateRecurrentKernelBackward = Tensor<Float>(
        [[0.11208751,-0.03048957,-0.14867921,0.07525943],
        [0.0779456,-0.021202432,-0.10339145,0.05233537],
        [-0.118453026,0.03222109,0.15712279,-0.07953346],
        [0.12208392,-0.03320875,-0.16193901,0.08197136]]
    )
    let expectedGradResetRecurrentKernelBackward = Tensor<Float>(
        [[-0.009129949,0.046693105,-0.046515163,0.04643202],
        [-0.006348962,0.032470364,-0.03234662,0.032288805],
        [0.009648444,-0.049344834,0.04915679,-0.04906892],
        [-0.009944194,0.05085738,-0.050663568,0.05057301]]
    )
    let expectedGradOutputRecurrentKernelBackward = Tensor<Float>(
        [[-0.18567993,-0.14199756,-0.13620174,-0.24541469],
        [-0.12912174,-0.09874504,-0.09471462,-0.17066126],
        [0.1962248,0.1500617,0.14393671,0.25935194],
        [-0.2022396,-0.15466146,-0.14834875,-0.26730174]]
    )
    let expectedGradUpdateBiasBackward = Tensor<Float>(
        [-0.11448185,0.03114087,0.1518552,-0.076867074]
    )
    let expectedGradResetBiasBackward = Tensor<Float>(
        [0.009324977,-0.047690533,0.04750879,-0.04742387]
    )
    let expectedGradOutputBiasBackward = Tensor<Float>(
        [0.31921688,0.21072157,0.34378064,0.41400108]
    )
    let expectedGradUpdateRecurrentBiasBackward = Tensor<Float>(
        [-0.11448185,0.03114087,0.1518552,-0.076867074]
    )
    let expectedGradResetRecurrentBiasBackward = Tensor<Float>(
        [0.009324977,-0.047690533,0.04750879,-0.04742387]
    )
    let expectedGradOutputRecurrentBiasBackward = Tensor<Float>(
        [0.1896463,0.14503083,0.13911119,0.25065708]
    )
    // END GENERATED CODE
    withTensorLeakChecking {
      var gru = BidirectionalGRU<Float>(
        GRUCell(inputSize: 3, hiddenSize: 4)
      )

      XCTAssertEqual(gru.forward.cell.updateKernel.shape, updateKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.updateRecurrentKernel.shape, updateRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.resetKernel.shape, resetKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.resetRecurrentKernel.shape, resetRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.outputKernel.shape, outputKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.outputRecurrentKernel.shape, outputRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.updateBias.shape, updateBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.updateRecurrentBias.shape, updateRecurrentBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.resetBias.shape, resetBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.resetRecurrentBias.shape, resetRecurrentBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.outputBias.shape, outputBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.outputRecurrentBias.shape, outputRecurrentBiasForward.shape)

      XCTAssertEqual(gru.backward.cell.updateKernel.shape, updateKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateRecurrentKernel.shape, updateRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetKernel.shape, resetKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetRecurrentKernel.shape, resetRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputKernel.shape, outputKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputRecurrentKernel.shape, outputRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateBias.shape, updateBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateRecurrentBias.shape, updateRecurrentBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetBias.shape, resetBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetRecurrentBias.shape, resetRecurrentBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputBias.shape, outputBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputRecurrentBias.shape, outputRecurrentBiasBackward.shape)

      gru.forward.cell.updateKernel = updateKernelForward
      gru.forward.cell.updateRecurrentKernel = updateRecurrentKernelForward
      gru.forward.cell.resetKernel = resetKernelForward
      gru.forward.cell.resetRecurrentKernel = resetRecurrentKernelForward
      gru.forward.cell.outputKernel = outputKernelForward
      gru.forward.cell.outputRecurrentKernel = outputRecurrentKernelForward

      gru.forward.cell.updateBias = updateBiasForward
      gru.forward.cell.updateRecurrentBias = updateRecurrentBiasForward
      gru.forward.cell.resetBias = resetBiasForward
      gru.forward.cell.resetRecurrentBias = resetRecurrentBiasForward
      gru.forward.cell.outputBias = outputBiasForward
      gru.forward.cell.outputRecurrentBias = outputRecurrentBiasForward

      gru.backward.cell.updateKernel = updateKernelBackward
      gru.backward.cell.updateRecurrentKernel = updateRecurrentKernelBackward
      gru.backward.cell.resetKernel = resetKernelBackward
      gru.backward.cell.resetRecurrentKernel = resetRecurrentKernelBackward
      gru.backward.cell.outputKernel = outputKernelBackward
      gru.backward.cell.outputRecurrentKernel = outputRecurrentKernelBackward

      gru.backward.cell.updateBias = updateBiasBackward
      gru.backward.cell.updateRecurrentBias = updateRecurrentBiasBackward
      gru.backward.cell.resetBias = resetBiasBackward
      gru.backward.cell.resetRecurrentBias = resetRecurrentBiasBackward
      gru.backward.cell.outputBias = outputBiasBackward
      gru.backward.cell.outputRecurrentBias = outputRecurrentBiasBackward

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }
      let initialForwardLayerState = GRUCell<Float>.State(initialForwardLayerState)
      let initialBackwardLayerState = GRUCell<Float>.State(initialBackwardLayerState)

      let outputs = gru(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      )
      assertEqual(
        Tensor(concatenating: outputs),
        expectedStates,
        accuracy: 1e-6)

      let (outputSum, (gradGRU)) =
        valueWithGradient(at: gru) { gru in
          gru.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateKernel,
        expectedGradUpdateKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetKernel,
        expectedGradResetKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputKernel,
        expectedGradOutputKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateRecurrentKernel,
        expectedGradUpdateRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetRecurrentKernel,
        expectedGradResetRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputRecurrentKernel,
        expectedGradOutputRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateBias,
        expectedGradUpdateBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetBias,
        expectedGradResetBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputBias,
        expectedGradOutputBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateRecurrentBias,
        expectedGradUpdateRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetRecurrentBias,
        expectedGradResetRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputRecurrentBias,
        expectedGradOutputRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateKernel,
        expectedGradUpdateKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetKernel,
        expectedGradResetKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputKernel,
        expectedGradOutputKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateRecurrentKernel,
        expectedGradUpdateRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetRecurrentKernel,
        expectedGradResetRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputRecurrentKernel,
        expectedGradOutputRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateBias,
        expectedGradUpdateBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetBias,
        expectedGradResetBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputBias,
        expectedGradOutputBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateRecurrentBias,
        expectedGradUpdateRecurrentBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetRecurrentBias,
        expectedGradResetRecurrentBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputRecurrentBias,
        expectedGradOutputRecurrentBiasBackward,
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          gru.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward,
        expectedGradInitialStateForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward,
        expectedGradInitialStateBackward,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalGRUAverageMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_gru.py --merge-mode ave".
    // BEGIN GENERATED CODE
    let updateKernelForward = Tensor<Float>(
        [[-0.26313293,-0.37116733,0.044766128,0.07748526],
        [0.09139812,-0.17107481,-0.1005373,0.16451049],
        [0.22751534,0.04139793,0.32446748,-0.572471]]
    )
    let resetKernelForward = Tensor<Float>(
        [[-0.10539937,0.38937503,-0.008569658,0.6300892],
        [0.5234366,0.20446932,0.42181438,-0.5262561],
        [-0.5687402,0.31843305,-0.4139892,-0.23788527]]
    )
    let outputKernelForward = Tensor<Float>(
        [[0.24885243,-0.4738691,0.26539946,0.2054413],
        [-0.27858475,-0.6128201,0.28634238,0.33588284],
        [-0.2638935,-0.5053163,-0.42289674,0.34110242]]
    )
    let updateRecurrentKernelForward = Tensor<Float>(
        [[0.27225327,-0.04530505,0.20488906,-0.70477146],
        [0.1233796,-0.2725925,-0.21331471,0.45468533],
        [-0.12901938,-0.5375645,-0.07981183,-0.18903717],
        [0.2276715,0.6950712,0.06780873,0.121172115]]
    )
    let resetRecurrentKernelForward = Tensor<Float>(
        [[-0.38362607,-0.03403465,-0.032581285,0.18508019],
        [-0.35768405,0.017118583,0.34339952,-0.015158445],
        [0.42593452,-0.51679564,-0.13895734,-0.2845649],
        [0.15196995,-0.4617733,-0.03934008,-0.038054474]]
    )
    let outputRecurrentKernelForward = Tensor<Float>(
        [[-0.3531419,0.04921763,-0.050379246,-0.26801136],
        [-0.34328082,0.5067098,0.031886574,-0.19034609],
        [-0.020022772,0.16366726,0.021551719,-0.27538475],
        [0.002952166,0.31402355,-0.22738007,-0.23688485]]
    )
    let updateBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateKernelBackward = Tensor<Float>(
        [[-0.38552743,0.30379778,-0.015429676,0.47474676],
        [0.24909449,-0.6272373,-0.17192209,0.08274299],
        [-0.41769588,-0.32856578,0.5987498,0.6109175]]
    )
    let resetKernelBackward = Tensor<Float>(
        [[-0.11749339,-0.61405337,0.26505172,-0.17011762],
        [-0.23611891,0.18858898,0.6282174,-0.5656544],
        [-0.5848755,-0.014772236,0.007604778,0.5725722]]
    )
    let outputKernelBackward = Tensor<Float>(
        [[0.10227531,0.5070258,-0.46571168,0.056879938],
        [0.49441308,-0.29395455,-0.07509631,0.29984736],
        [0.47141522,0.18345171,0.1791873,0.4368046]]
    )
    let updateRecurrentKernelBackward = Tensor<Float>(
        [[0.3819934,0.37764925,0.015312237,-0.17769441],
        [-0.30984464,0.0024376288,-0.3083459,0.6083385],
        [-0.035942614,-0.24049878,0.19538112,-0.22652327],
        [0.06525053,0.14899497,-0.36888042,-0.2775943]]
    )
    let resetRecurrentKernelBackward = Tensor<Float>(
        [[0.2151401,-0.21178766,-0.15712889,-0.078161664],
        [0.54615974,0.025927158,-0.061987586,-0.13631916],
        [0.61778206,-0.12979838,-0.19998191,0.32677993],
        [0.3122218,-0.3348464,-0.029183073,-0.08691845]]
    )
    let outputRecurrentKernelBackward = Tensor<Float>(
        [[-0.64042187,0.34098455,-0.10449011,-0.14277793],
        [-0.07292092,0.24773169,0.21102233,0.07893403],
        [-0.029284794,-0.45729926,0.09502104,-0.29511455],
        [0.49253088,0.07420389,-0.48452082,0.2440229]]
    )
    let updateBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.90076077,-0.7541278,-0.025081873],
          [0.37973404,-0.26513243,0.10235357],
          [0.3545239,-0.14615154,-0.540749],
          [-0.70222735,-0.34035724,-0.8451414]]]
    )
    let initialForwardLayerState = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialBackwardLayerState = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let expectedSum = Tensor<Float>(
        -0.53184664
    )
    let expectedStates = Tensor<Float>(
        [[[-0.23712845,0.065289676,-0.14049831,0.045999765],
          [-0.11894582,0.03693285,-0.045380324,-0.03570187],
          [-0.18859795,-0.11229195,0.19757463,-0.21844287],
          [-0.3450238,-0.13424796,0.397752,-0.4503269]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those two and test it.
    // let expectedFinalStateForward = Tensor<Float>(
    //     [[0.087087676,0.49491018,0.019275311,-0.13219076]]
    // )
    // let expectedFinalStateBackward = Tensor<Float>(
    //     [[-0.328244,-0.30765843,0.35560328,-0.37002444]]
    // )
    let expectedGradX = Tensor<Float>(
        [[[0.008442146,0.015977293,-0.017315175],
          [-0.015072985,-0.01641532,-0.049635936],
          [0.005338762,-0.032777082,-0.12340391],
          [0.18397327,0.11113768,0.017806858]]]
    )
    let expectedGradInitialStateForward = Tensor<Float>(
        [[-0.035949126,0.1324582,0.010624796,0.0498662]]
    )
    let expectedGradUpdateKernelForward = Tensor<Float>(
        [[-0.0073167454,0.056314625,-0.0062336624,-0.07089153],
        [-0.00076548447,0.02190212,0.013407189,-0.05884329],
        [0.00033165538,0.05131561,0.025480783,-0.07881642]]
    )
    let expectedGradResetKernelForward = Tensor<Float>(
        [[0.0050837277,0.000618778,0.00082319777,0.002807496],
        [0.0029551436,-0.003979396,0.0016249518,0.0014738103],
        [0.0075652236,-0.0056819245,0.0023021724,0.0032740915]]
    )
    let expectedGradOutputKernelForward = Tensor<Float>(
        [[-0.13860242,0.0056050895,-0.15595947,-0.075157814],
        [-0.08908434,-0.07522535,-0.133511,-0.095664434],
        [-0.23910204,-0.13338125,-0.29645422,-0.1555842]]
    )
    let expectedGradUpdateRecurrentKernelForward = Tensor<Float>(
        [[0.0006653269,-0.006875807,0.0018882437,-0.0017765616],
        [-0.0011794777,-0.012355285,-0.01903148,0.048002526],
        [0.0027665214,-0.0027906992,0.021368187,-0.045509342],
        [-0.0014231934,-0.008094095,-0.018676067,0.058172047]]
    )
    let expectedGradResetRecurrentKernelForward = Tensor<Float>(
        [[-0.0008088454,-0.0006627698,0.00016036181,-0.0003286859],
        [-0.0027881989,0.0051686955,-0.0017176332,-0.0011592377],
        [0.0008306038,-0.0059162755,0.0018482914,0.00036050117],
        [-0.0014314271,0.005271257,-0.0019009972,-0.00081068865]]
    )
    let expectedGradOutputRecurrentKernelForward = Tensor<Float>(
        [[0.013640466,-0.005532119,0.011380454,-0.0022389072],
        [0.048948735,0.047380794,0.07814442,0.054513022],
        [-0.016135914,-0.05371744,-0.046594556,-0.052742656],
        [0.025083385,0.045732304,0.054932415,0.053943418]]
    )
    let expectedGradUpdateBiasForward = Tensor<Float>(
        [-0.0035956602,-0.049405254,-0.057810556,0.14968255]
    )
    let expectedGradResetBiasForward = Tensor<Float>(
        [-0.009559427,0.01502455,-0.005170936,-0.004052236]
    )
    let expectedGradOutputBiasForward = Tensor<Float>(
        [0.30791992,0.29866174,0.45024475,0.302623]
    )
    let expectedGradUpdateRecurrentBiasForward = Tensor<Float>(
        [-0.0035956602,-0.049405254,-0.057810556,0.14968255]
    )
    let expectedGradResetRecurrentBiasForward = Tensor<Float>(
        [-0.009559427,0.01502455,-0.005170936,-0.004052236]
    )
    let expectedGradOutputRecurrentBiasForward = Tensor<Float>(
        [0.16791594,0.13910466,0.2555424,0.16603217]
    )
    let expectedGradInitialStateBackward = Tensor<Float>(
        [[0.21845947,0.19890642,0.24278466,0.17001694]]
    )
    let expectedGradUpdateKernelBackward = Tensor<Float>(
        [[0.040196143,-0.010933985,-0.053318437,0.02698908],
        [0.019482363,-0.0052995104,-0.025842508,0.013081132],
        [0.048376676,-0.013159219,-0.064169556,0.032481775]]
    )
    let expectedGradResetKernelBackward = Tensor<Float>(
        [[-0.003274127,0.016744798,-0.016680986,0.016651168],
        [-0.0015869117,0.008115909,-0.008084981,0.0080705285],
        [-0.003940462,0.020152623,-0.020075824,0.020039938]]
    )
    let expectedGradOutputKernelBackward = Tensor<Float>(
        [[-0.11208141,-0.07398722,-0.12070608,-0.14536144],
        [-0.05432389,-0.035860308,-0.058504116,-0.070454136],
        [-0.1348917,-0.089044765,-0.14527163,-0.17494473]]
    )
    let expectedGradUpdateRecurrentKernelBackward = Tensor<Float>(
        [[0.056043755,-0.015244785,-0.074339606,0.037629716],
        [0.0389728,-0.010601216,-0.051695727,0.026167685],
        [-0.059226513,0.016110545,0.078561395,-0.03976673],
        [0.06104196,-0.016604375,-0.080969505,0.04098568]]
    )
    let expectedGradResetRecurrentKernelBackward = Tensor<Float>(
        [[-0.0045649745,0.023346553,-0.023257582,0.02321601],
        [-0.003174481,0.016235182,-0.01617331,0.016144402],
        [0.004824222,-0.024672417,0.024578394,-0.02453446],
        [-0.004972097,0.02542869,-0.025331784,0.025286505]]
    )
    let expectedGradOutputRecurrentKernelBackward = Tensor<Float>(
        [[-0.092839964,-0.07099878,-0.06810087,-0.122707345],
        [-0.06456087,-0.04937252,-0.04735731,-0.08533063],
        [0.0981124,0.07503085,0.071968354,0.12967597],
        [-0.1011198,-0.07733073,-0.074174374,-0.13365087]]
    )
    let expectedGradUpdateBiasBackward = Tensor<Float>(
        [-0.057240926,0.015570435,0.0759276,-0.038433537]
    )
    let expectedGradResetBiasBackward = Tensor<Float>(
        [0.0046624886,-0.023845267,0.023754396,-0.023711935]
    )
    let expectedGradOutputBiasBackward = Tensor<Float>(
        [0.15960844,0.10536078,0.17189032,0.20700054]
    )
    let expectedGradUpdateRecurrentBiasBackward = Tensor<Float>(
        [-0.057240926,0.015570435,0.0759276,-0.038433537]
    )
    let expectedGradResetRecurrentBiasBackward = Tensor<Float>(
        [0.0046624886,-0.023845267,0.023754396,-0.023711935]
    )
    let expectedGradOutputRecurrentBiasBackward = Tensor<Float>(
        [0.09482315,0.07251541,0.069555596,0.12532854]
    )
    // END GENERATED CODE
    withTensorLeakChecking {
      var gru = BidirectionalGRU<Float>(
        GRUCell(inputSize: 3, hiddenSize: 4),
        mergeFunction: GRUCell<Float>.State.average
      )

      XCTAssertEqual(gru.forward.cell.updateKernel.shape, updateKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.updateRecurrentKernel.shape, updateRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.resetKernel.shape, resetKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.resetRecurrentKernel.shape, resetRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.outputKernel.shape, outputKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.outputRecurrentKernel.shape, outputRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.updateBias.shape, updateBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.updateRecurrentBias.shape, updateRecurrentBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.resetBias.shape, resetBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.resetRecurrentBias.shape, resetRecurrentBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.outputBias.shape, outputBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.outputRecurrentBias.shape, outputRecurrentBiasForward.shape)

      XCTAssertEqual(gru.backward.cell.updateKernel.shape, updateKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateRecurrentKernel.shape, updateRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetKernel.shape, resetKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetRecurrentKernel.shape, resetRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputKernel.shape, outputKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputRecurrentKernel.shape, outputRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateBias.shape, updateBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateRecurrentBias.shape, updateRecurrentBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetBias.shape, resetBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetRecurrentBias.shape, resetRecurrentBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputBias.shape, outputBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputRecurrentBias.shape, outputRecurrentBiasBackward.shape)

      gru.forward.cell.updateKernel = updateKernelForward
      gru.forward.cell.updateRecurrentKernel = updateRecurrentKernelForward
      gru.forward.cell.resetKernel = resetKernelForward
      gru.forward.cell.resetRecurrentKernel = resetRecurrentKernelForward
      gru.forward.cell.outputKernel = outputKernelForward
      gru.forward.cell.outputRecurrentKernel = outputRecurrentKernelForward

      gru.forward.cell.updateBias = updateBiasForward
      gru.forward.cell.updateRecurrentBias = updateRecurrentBiasForward
      gru.forward.cell.resetBias = resetBiasForward
      gru.forward.cell.resetRecurrentBias = resetRecurrentBiasForward
      gru.forward.cell.outputBias = outputBiasForward
      gru.forward.cell.outputRecurrentBias = outputRecurrentBiasForward

      gru.backward.cell.updateKernel = updateKernelBackward
      gru.backward.cell.updateRecurrentKernel = updateRecurrentKernelBackward
      gru.backward.cell.resetKernel = resetKernelBackward
      gru.backward.cell.resetRecurrentKernel = resetRecurrentKernelBackward
      gru.backward.cell.outputKernel = outputKernelBackward
      gru.backward.cell.outputRecurrentKernel = outputRecurrentKernelBackward

      gru.backward.cell.updateBias = updateBiasBackward
      gru.backward.cell.updateRecurrentBias = updateRecurrentBiasBackward
      gru.backward.cell.resetBias = resetBiasBackward
      gru.backward.cell.resetRecurrentBias = resetRecurrentBiasBackward
      gru.backward.cell.outputBias = outputBiasBackward
      gru.backward.cell.outputRecurrentBias = outputRecurrentBiasBackward

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }
      let initialForwardLayerState = GRUCell<Float>.State(initialForwardLayerState)
      let initialBackwardLayerState = GRUCell<Float>.State(initialBackwardLayerState)

      let outputs = gru(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      )
      assertEqual(
        Tensor(concatenating: outputs),
        expectedStates,
        accuracy: 1e-6)

      let (outputSum, (gradGRU)) =
        valueWithGradient(at: gru) { gru in
          gru.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateKernel,
        expectedGradUpdateKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetKernel,
        expectedGradResetKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputKernel,
        expectedGradOutputKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateRecurrentKernel,
        expectedGradUpdateRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetRecurrentKernel,
        expectedGradResetRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputRecurrentKernel,
        expectedGradOutputRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateBias,
        expectedGradUpdateBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetBias,
        expectedGradResetBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputBias,
        expectedGradOutputBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateRecurrentBias,
        expectedGradUpdateRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetRecurrentBias,
        expectedGradResetRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputRecurrentBias,
        expectedGradOutputRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateKernel,
        expectedGradUpdateKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetKernel,
        expectedGradResetKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputKernel,
        expectedGradOutputKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateRecurrentKernel,
        expectedGradUpdateRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetRecurrentKernel,
        expectedGradResetRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputRecurrentKernel,
        expectedGradOutputRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateBias,
        expectedGradUpdateBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetBias,
        expectedGradResetBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputBias,
        expectedGradOutputBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateRecurrentBias,
        expectedGradUpdateRecurrentBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetRecurrentBias,
        expectedGradResetRecurrentBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputRecurrentBias,
        expectedGradOutputRecurrentBiasBackward,
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          gru.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward,
        expectedGradInitialStateForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward,
        expectedGradInitialStateBackward,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalGRUMultiplyMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_gru.py --merge-mode mul".
    // BEGIN GENERATED CODE
    let updateKernelForward = Tensor<Float>(
        [[-0.26313293,-0.37116733,0.044766128,0.07748526],
        [0.09139812,-0.17107481,-0.1005373,0.16451049],
        [0.22751534,0.04139793,0.32446748,-0.572471]]
    )
    let resetKernelForward = Tensor<Float>(
        [[-0.10539937,0.38937503,-0.008569658,0.6300892],
        [0.5234366,0.20446932,0.42181438,-0.5262561],
        [-0.5687402,0.31843305,-0.4139892,-0.23788527]]
    )
    let outputKernelForward = Tensor<Float>(
        [[0.24885243,-0.4738691,0.26539946,0.2054413],
        [-0.27858475,-0.6128201,0.28634238,0.33588284],
        [-0.2638935,-0.5053163,-0.42289674,0.34110242]]
    )
    let updateRecurrentKernelForward = Tensor<Float>(
        [[0.27225327,-0.04530505,0.20488906,-0.70477146],
        [0.1233796,-0.2725925,-0.21331471,0.45468533],
        [-0.12901938,-0.5375645,-0.07981183,-0.18903717],
        [0.2276715,0.6950712,0.06780873,0.121172115]]
    )
    let resetRecurrentKernelForward = Tensor<Float>(
        [[-0.38362607,-0.03403465,-0.032581285,0.18508019],
        [-0.35768405,0.017118583,0.34339952,-0.015158445],
        [0.42593452,-0.51679564,-0.13895734,-0.2845649],
        [0.15196995,-0.4617733,-0.03934008,-0.038054474]]
    )
    let outputRecurrentKernelForward = Tensor<Float>(
        [[-0.3531419,0.04921763,-0.050379246,-0.26801136],
        [-0.34328082,0.5067098,0.031886574,-0.19034609],
        [-0.020022772,0.16366726,0.021551719,-0.27538475],
        [0.002952166,0.31402355,-0.22738007,-0.23688485]]
    )
    let updateBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateKernelBackward = Tensor<Float>(
        [[-0.38552743,0.30379778,-0.015429676,0.47474676],
        [0.24909449,-0.6272373,-0.17192209,0.08274299],
        [-0.41769588,-0.32856578,0.5987498,0.6109175]]
    )
    let resetKernelBackward = Tensor<Float>(
        [[-0.11749339,-0.61405337,0.26505172,-0.17011762],
        [-0.23611891,0.18858898,0.6282174,-0.5656544],
        [-0.5848755,-0.014772236,0.007604778,0.5725722]]
    )
    let outputKernelBackward = Tensor<Float>(
        [[0.10227531,0.5070258,-0.46571168,0.056879938],
        [0.49441308,-0.29395455,-0.07509631,0.29984736],
        [0.47141522,0.18345171,0.1791873,0.4368046]]
    )
    let updateRecurrentKernelBackward = Tensor<Float>(
        [[0.3819934,0.37764925,0.015312237,-0.17769441],
        [-0.30984464,0.0024376288,-0.3083459,0.6083385],
        [-0.035942614,-0.24049878,0.19538112,-0.22652327],
        [0.06525053,0.14899497,-0.36888042,-0.2775943]]
    )
    let resetRecurrentKernelBackward = Tensor<Float>(
        [[0.2151401,-0.21178766,-0.15712889,-0.078161664],
        [0.54615974,0.025927158,-0.061987586,-0.13631916],
        [0.61778206,-0.12979838,-0.19998191,0.32677993],
        [0.3122218,-0.3348464,-0.029183073,-0.08691845]]
    )
    let outputRecurrentKernelBackward = Tensor<Float>(
        [[-0.64042187,0.34098455,-0.10449011,-0.14277793],
        [-0.07292092,0.24773169,0.21102233,0.07893403],
        [-0.029284794,-0.45729926,0.09502104,-0.29511455],
        [0.49253088,0.07420389,-0.48452082,0.2440229]]
    )
    let updateBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.90076077,-0.7541278,-0.025081873],
          [0.37973404,-0.26513243,0.10235357],
          [0.3545239,-0.14615154,-0.540749],
          [-0.70222735,-0.34035724,-0.8451414]]]
    )
    let initialForwardLayerState = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialBackwardLayerState = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let expectedSum = Tensor<Float>(
        -0.32895058
    )
    let expectedStates = Tensor<Float>(
        [[[0.04792786,-0.13482755,-0.22637701,-0.17096016],
          [0.005248314,-0.06652843,-0.095485054,-0.10619393],
          [-0.051071964,-0.14824535,-0.014771455,-0.0697825],
          [-0.067678906,-0.37781745,0.014962049,0.10158371]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those two and test it.
    // let expectedFinalStateForward = Tensor<Float>(
    //     [[0.087087676,0.49491018,0.019275311,-0.13219076]]
    // )
    // let expectedFinalStateBackward = Tensor<Float>(
    //     [[-0.328244,-0.30765843,0.35560328,-0.37002444]]
    // )
    let expectedGradX = Tensor<Float>(
        [[[-0.0045455527,-0.020066032,0.0030737333],
          [0.040171646,0.049551535,-0.0197279],
          [0.05888499,0.1390787,0.007269824],
          [0.0767434,0.15354612,-0.019848084]]]
    )
    let expectedGradInitialStateForward = Tensor<Float>(
        [[0.07075158,-0.173892,0.09029196,-0.13666949]]
    )
    let expectedGradUpdateKernelForward = Tensor<Float>(
        [[0.012366264,-0.08216441,-0.008099017,0.12655589],
        [0.00005636626,-0.0324539,0.02747383,0.117940515],
        [-0.0014967625,-0.07796255,0.040063504,0.12748465]]
    )
    let expectedGradResetKernelForward = Tensor<Float>(
        [[-0.0076980656,-0.00063710264,0.0027035254,-0.0046277456],
        [-0.004731113,0.0055333427,0.0041351225,-0.0024739115],
        [-0.011927827,0.0085395295,0.0036171544,-0.0051743267]]
    )
    let expectedGradOutputKernelForward = Tensor<Float>(
        [[0.20785503,-0.001673013,-0.25896353,0.10652162],
        [0.14713746,0.10674674,-0.23975003,0.18590957],
        [0.37952816,0.2001083,-0.46228534,0.25734156]]
    )
    let expectedGradUpdateRecurrentKernelForward = Tensor<Float>(
        [[-0.0014494247,0.010546453,0.005384476,0.011515614],
        [0.0034076856,0.019562693,-0.035582643,-0.09373548],
        [-0.0065402063,0.0038736751,0.043143254,0.10498153],
        [0.004144326,0.011323398,-0.039187685,-0.13013235]]
    )
    let expectedGradResetRecurrentKernelForward = Tensor<Float>(
        [[0.0012057752,0.0007886884,0.00077032676,0.00046306645],
        [0.00456115,-0.0072078817,-0.0036669155,0.001852406],
        [-0.0016002035,0.008008988,0.0047597894,-0.00071795564],
        [0.0024727562,-0.0071561057,-0.0052759107,0.0015336934]]
    )
    let expectedGradOutputRecurrentKernelForward = Tensor<Float>(
        [[-0.019984528,0.0067213336,0.012130001,0.011362088],
        [-0.08117275,-0.06644401,0.1340633,-0.10929223],
        [0.032181997,0.07354428,-0.09417313,0.11926204],
        [-0.045114394,-0.06297048,0.10899238,-0.1180308]]
    )
    let expectedGradUpdateBiasForward = Tensor<Float>(
        [0.009944006,0.076615654,-0.10517438,-0.28469688]
    )
    let expectedGradResetBiasForward = Tensor<Float>(
        [0.015496891,-0.021194058,-0.010666848,0.0065090316]
    )
    let expectedGradOutputBiasForward = Tensor<Float>(
        [-0.5120879,-0.4268308,0.7647575,-0.58130544]
    )
    let expectedGradUpdateRecurrentBiasForward = Tensor<Float>(
        [0.009944006,0.076615654,-0.10517438,-0.28469688]
    )
    let expectedGradResetRecurrentBiasForward = Tensor<Float>(
        [0.015496891,-0.021194058,-0.010666848,0.0065090316]
    )
    let expectedGradOutputRecurrentBiasForward = Tensor<Float>(
        [-0.27554107,-0.19721822,0.42974865,-0.3239422]
    )
    let expectedGradInitialStateBackward = Tensor<Float>(
        [[0.072507605,0.20855014,-0.011596381,-0.024698015]]
    )
    let expectedGradUpdateKernelBackward = Tensor<Float>(
        [[0.007001178,-0.010822684,-0.002055459,-0.007135414],
        [0.0033933478,-0.0052455645,-0.0009962448,-0.0034584096],
        [0.0084260255,-0.013025266,-0.0024737765,-0.00858758]]
    )
    let expectedGradResetKernelBackward = Tensor<Float>(
        [[-0.00057027227,0.016574344,-0.0006430624,-0.0044022617],
        [-0.00027640094,0.008033292,-0.00031168104,-0.0021336987],
        [-0.0006863314,0.019947477,-0.00077393546,-0.0052981894]]
    )
    let expectedGradOutputKernelBackward = Tensor<Float>(
        [[-0.019521821,-0.07323406,-0.0046532946,0.03843088],
        [-0.0094618825,-0.035495263,-0.0022553701,0.018626772],
        [-0.023494812,-0.08813832,-0.005600312,0.046252154]]
    )
    let expectedGradUpdateRecurrentKernelBackward = Tensor<Float>(
        [[0.0097614415,-0.015089602,-0.0028658381,-0.009948601],
        [0.0067881015,-0.010493302,-0.0019929025,-0.0069182524],
        [-0.010315799,0.01594655,0.0030285907,0.010513588],
        [0.0106320055,-0.016435353,-0.003121425,-0.010835856]]
    )
    let expectedGradResetRecurrentKernelBackward = Tensor<Float>(
        [[-0.00079510605,0.023108894,-0.00089659425,-0.006137884],
        [-0.0005529164,0.016069913,-0.0006234912,-0.0042682816],
        [0.00084026065,-0.024421262,0.00094751234,0.0064864582],
        [-0.0008660168,0.025169836,-0.0009765561,-0.006685285]]
    )
    let expectedGradOutputRecurrentKernelBackward = Tensor<Float>(
        [[-0.016170435,-0.07027604,-0.0026253306,0.032441556],
        [-0.011244912,-0.048869923,-0.0018256537,0.022559844],
        [0.017088763,0.07426706,0.0027744249,-0.034283932],
        [-0.017612576,-0.07654353,-0.002859468,0.035334826]]
    )
    let expectedGradUpdateBiasBackward = Tensor<Float>(
        [-0.009969959,0.015411937,0.0029270563,0.010161117]
    )
    let expectedGradResetBiasBackward = Tensor<Float>(
        [0.00081209064,-0.023602532,0.0009157467,0.0062689977]
    )
    let expectedGradOutputBiasBackward = Tensor<Float>(
        [0.027799858,0.10428825,0.006626479,-0.05472712]
    )
    let expectedGradUpdateRecurrentBiasBackward = Tensor<Float>(
        [-0.009969959,0.015411937,0.0029270563,0.010161117]
    )
    let expectedGradResetRecurrentBiasBackward = Tensor<Float>(
        [0.00081209064,-0.023602532,0.0009157467,0.0062689977]
    )
    let expectedGradOutputRecurrentBiasBackward = Tensor<Float>(
        [0.016515857,0.07177723,0.0026814113,-0.033134554]
    )
    // END GENERATED CODE
    withTensorLeakChecking {
      var gru = BidirectionalGRU<Float>(
        GRUCell(inputSize: 3, hiddenSize: 4),
        mergeFunction: GRUCell<Float>.State.multiply
      )

      XCTAssertEqual(gru.forward.cell.updateKernel.shape, updateKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.updateRecurrentKernel.shape, updateRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.resetKernel.shape, resetKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.resetRecurrentKernel.shape, resetRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.outputKernel.shape, outputKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.outputRecurrentKernel.shape, outputRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.updateBias.shape, updateBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.updateRecurrentBias.shape, updateRecurrentBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.resetBias.shape, resetBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.resetRecurrentBias.shape, resetRecurrentBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.outputBias.shape, outputBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.outputRecurrentBias.shape, outputRecurrentBiasForward.shape)

      XCTAssertEqual(gru.backward.cell.updateKernel.shape, updateKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateRecurrentKernel.shape, updateRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetKernel.shape, resetKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetRecurrentKernel.shape, resetRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputKernel.shape, outputKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputRecurrentKernel.shape, outputRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateBias.shape, updateBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateRecurrentBias.shape, updateRecurrentBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetBias.shape, resetBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetRecurrentBias.shape, resetRecurrentBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputBias.shape, outputBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputRecurrentBias.shape, outputRecurrentBiasBackward.shape)

      gru.forward.cell.updateKernel = updateKernelForward
      gru.forward.cell.updateRecurrentKernel = updateRecurrentKernelForward
      gru.forward.cell.resetKernel = resetKernelForward
      gru.forward.cell.resetRecurrentKernel = resetRecurrentKernelForward
      gru.forward.cell.outputKernel = outputKernelForward
      gru.forward.cell.outputRecurrentKernel = outputRecurrentKernelForward

      gru.forward.cell.updateBias = updateBiasForward
      gru.forward.cell.updateRecurrentBias = updateRecurrentBiasForward
      gru.forward.cell.resetBias = resetBiasForward
      gru.forward.cell.resetRecurrentBias = resetRecurrentBiasForward
      gru.forward.cell.outputBias = outputBiasForward
      gru.forward.cell.outputRecurrentBias = outputRecurrentBiasForward

      gru.backward.cell.updateKernel = updateKernelBackward
      gru.backward.cell.updateRecurrentKernel = updateRecurrentKernelBackward
      gru.backward.cell.resetKernel = resetKernelBackward
      gru.backward.cell.resetRecurrentKernel = resetRecurrentKernelBackward
      gru.backward.cell.outputKernel = outputKernelBackward
      gru.backward.cell.outputRecurrentKernel = outputRecurrentKernelBackward

      gru.backward.cell.updateBias = updateBiasBackward
      gru.backward.cell.updateRecurrentBias = updateRecurrentBiasBackward
      gru.backward.cell.resetBias = resetBiasBackward
      gru.backward.cell.resetRecurrentBias = resetRecurrentBiasBackward
      gru.backward.cell.outputBias = outputBiasBackward
      gru.backward.cell.outputRecurrentBias = outputRecurrentBiasBackward

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }
      let initialForwardLayerState = GRUCell<Float>.State(initialForwardLayerState)
      let initialBackwardLayerState = GRUCell<Float>.State(initialBackwardLayerState)

      let outputs = gru(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      )
      assertEqual(
        Tensor(concatenating: outputs),
        expectedStates,
        accuracy: 1e-6)

      let (outputSum, (gradGRU)) =
        valueWithGradient(at: gru) { gru in
          gru.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateKernel,
        expectedGradUpdateKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetKernel,
        expectedGradResetKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputKernel,
        expectedGradOutputKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateRecurrentKernel,
        expectedGradUpdateRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetRecurrentKernel,
        expectedGradResetRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputRecurrentKernel,
        expectedGradOutputRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateBias,
        expectedGradUpdateBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetBias,
        expectedGradResetBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputBias,
        expectedGradOutputBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateRecurrentBias,
        expectedGradUpdateRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetRecurrentBias,
        expectedGradResetRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputRecurrentBias,
        expectedGradOutputRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateKernel,
        expectedGradUpdateKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetKernel,
        expectedGradResetKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputKernel,
        expectedGradOutputKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateRecurrentKernel,
        expectedGradUpdateRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetRecurrentKernel,
        expectedGradResetRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputRecurrentKernel,
        expectedGradOutputRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateBias,
        expectedGradUpdateBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetBias,
        expectedGradResetBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputBias,
        expectedGradOutputBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateRecurrentBias,
        expectedGradUpdateRecurrentBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetRecurrentBias,
        expectedGradResetRecurrentBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputRecurrentBias,
        expectedGradOutputRecurrentBiasBackward,
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          gru.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward,
        expectedGradInitialStateForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward,
        expectedGradInitialStateBackward,
        accuracy: 1e-6)
    }
  }

  func testBidirectionalGRUStackMerge() {
    // Data generated by "Utilities/ReferenceImplementations/bidirectional_gru.py --merge-mode none".
    // BEGIN GENERATED CODE
    let updateKernelForward = Tensor<Float>(
        [[-0.26313293,-0.37116733,0.044766128,0.07748526],
        [0.09139812,-0.17107481,-0.1005373,0.16451049],
        [0.22751534,0.04139793,0.32446748,-0.572471]]
    )
    let resetKernelForward = Tensor<Float>(
        [[-0.10539937,0.38937503,-0.008569658,0.6300892],
        [0.5234366,0.20446932,0.42181438,-0.5262561],
        [-0.5687402,0.31843305,-0.4139892,-0.23788527]]
    )
    let outputKernelForward = Tensor<Float>(
        [[0.24885243,-0.4738691,0.26539946,0.2054413],
        [-0.27858475,-0.6128201,0.28634238,0.33588284],
        [-0.2638935,-0.5053163,-0.42289674,0.34110242]]
    )
    let updateRecurrentKernelForward = Tensor<Float>(
        [[0.27225327,-0.04530505,0.20488906,-0.70477146],
        [0.1233796,-0.2725925,-0.21331471,0.45468533],
        [-0.12901938,-0.5375645,-0.07981183,-0.18903717],
        [0.2276715,0.6950712,0.06780873,0.121172115]]
    )
    let resetRecurrentKernelForward = Tensor<Float>(
        [[-0.38362607,-0.03403465,-0.032581285,0.18508019],
        [-0.35768405,0.017118583,0.34339952,-0.015158445],
        [0.42593452,-0.51679564,-0.13895734,-0.2845649],
        [0.15196995,-0.4617733,-0.03934008,-0.038054474]]
    )
    let outputRecurrentKernelForward = Tensor<Float>(
        [[-0.3531419,0.04921763,-0.050379246,-0.26801136],
        [-0.34328082,0.5067098,0.031886574,-0.19034609],
        [-0.020022772,0.16366726,0.021551719,-0.27538475],
        [0.002952166,0.31402355,-0.22738007,-0.23688485]]
    )
    let updateBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputRecurrentBiasForward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateKernelBackward = Tensor<Float>(
        [[-0.38552743,0.30379778,-0.015429676,0.47474676],
        [0.24909449,-0.6272373,-0.17192209,0.08274299],
        [-0.41769588,-0.32856578,0.5987498,0.6109175]]
    )
    let resetKernelBackward = Tensor<Float>(
        [[-0.11749339,-0.61405337,0.26505172,-0.17011762],
        [-0.23611891,0.18858898,0.6282174,-0.5656544],
        [-0.5848755,-0.014772236,0.007604778,0.5725722]]
    )
    let outputKernelBackward = Tensor<Float>(
        [[0.10227531,0.5070258,-0.46571168,0.056879938],
        [0.49441308,-0.29395455,-0.07509631,0.29984736],
        [0.47141522,0.18345171,0.1791873,0.4368046]]
    )
    let updateRecurrentKernelBackward = Tensor<Float>(
        [[0.3819934,0.37764925,0.015312237,-0.17769441],
        [-0.30984464,0.0024376288,-0.3083459,0.6083385],
        [-0.035942614,-0.24049878,0.19538112,-0.22652327],
        [0.06525053,0.14899497,-0.36888042,-0.2775943]]
    )
    let resetRecurrentKernelBackward = Tensor<Float>(
        [[0.2151401,-0.21178766,-0.15712889,-0.078161664],
        [0.54615974,0.025927158,-0.061987586,-0.13631916],
        [0.61778206,-0.12979838,-0.19998191,0.32677993],
        [0.3122218,-0.3348464,-0.029183073,-0.08691845]]
    )
    let outputRecurrentKernelBackward = Tensor<Float>(
        [[-0.64042187,0.34098455,-0.10449011,-0.14277793],
        [-0.07292092,0.24773169,0.21102233,0.07893403],
        [-0.029284794,-0.45729926,0.09502104,-0.29511455],
        [0.49253088,0.07420389,-0.48452082,0.2440229]]
    )
    let updateBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let updateRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let resetRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let outputRecurrentBiasBackward = Tensor<Float>(
        [0.0,0.0,0.0,0.0]
    )
    let x = Tensor<Float>(
        [[[-0.90076077,-0.7541278,-0.025081873],
          [0.37973404,-0.26513243,0.10235357],
          [0.3545239,-0.14615154,-0.540749],
          [-0.70222735,-0.34035724,-0.8451414]]]
    )
    let initialForwardLayerState = Tensor<Float>(
        [[-0.22500783,0.3640524,-0.76411664,1.0557337]]
    )
    let initialBackwardLayerState = Tensor<Float>(
        [[-0.9790854,-0.6808555,1.0346882,-1.0664041]]
    )
    let expectedSum = Tensor<Float>(
        -1.0636933
    )
    let expectedStatesForward = Tensor<Float>(
        [[[-0.1460129,0.4382378,-0.6365999,0.46202397],
          [-0.0246071,0.29749468,-0.35770136,0.2921221],
          [0.10575074,0.28877518,-0.03438914,0.12433955],
          [0.087087676,0.49491018,0.019275311,-0.13219076]]]
    )
    let expectedStatesBackward = Tensor<Float>(
        [[[-0.328244,-0.30765843,0.35560328,-0.37002444],
          [-0.21328454,-0.22362898,0.2669407,-0.36352584],
          [-0.48294663,-0.51335907,0.4295384,-0.5612253],
          [-0.7771353,-0.7634061,0.77622867,-0.768463]]]
    )
    // TODO: Modify BidirectionalRecurrentLayer to return those two and test it.
    // let expectedFinalStateForward = Tensor<Float>(
    //     [[0.087087676,0.49491018,0.019275311,-0.13219076]]
    // )
    // let expectedFinalStateBackward = Tensor<Float>(
    //     [[-0.328244,-0.30765843,0.35560328,-0.37002444]]
    // )
    let expectedGradX = Tensor<Float>(
        [[[0.016884292,0.031954587,-0.03463035],
          [-0.03014597,-0.03283064,-0.09927187],
          [0.010677524,-0.065554164,-0.24680781],
          [0.36794654,0.22227536,0.035613716]]]
    )
    let expectedGradInitialStateForward = Tensor<Float>(
        [[-0.07189825,0.2649164,0.021249592,0.0997324]]
    )
    let expectedGradUpdateKernelForward = Tensor<Float>(
        [[-0.014633491,0.11262925,-0.012467325,-0.14178306],
        [-0.0015309689,0.04380424,0.026814379,-0.11768658],
        [0.00066331076,0.10263122,0.050961565,-0.15763284]]
    )
    let expectedGradResetKernelForward = Tensor<Float>(
        [[0.010167455,0.001237556,0.0016463955,0.005614992],
        [0.005910287,-0.007958792,0.0032499037,0.0029476206],
        [0.015130447,-0.011363849,0.0046043447,0.006548183]]
    )
    let expectedGradOutputKernelForward = Tensor<Float>(
        [[-0.27720484,0.011210179,-0.31191894,-0.15031563],
        [-0.17816868,-0.1504507,-0.267022,-0.19132887],
        [-0.47820407,-0.2667625,-0.59290844,-0.3111684]]
    )
    let expectedGradUpdateRecurrentKernelForward = Tensor<Float>(
        [[0.0013306538,-0.013751614,0.0037764874,-0.0035531232],
        [-0.0023589553,-0.02471057,-0.03806296,0.09600505],
        [0.005533043,-0.0055813985,0.042736374,-0.091018684],
        [-0.0028463867,-0.01618819,-0.037352134,0.116344094]]
    )
    let expectedGradResetRecurrentKernelForward = Tensor<Float>(
        [[-0.0016176908,-0.0013255395,0.00032072363,-0.0006573718],
        [-0.0055763978,0.010337391,-0.0034352664,-0.0023184754],
        [0.0016612076,-0.011832551,0.0036965827,0.00072100235],
        [-0.0028628542,0.010542514,-0.0038019945,-0.0016213773]]
    )
    let expectedGradOutputRecurrentKernelForward = Tensor<Float>(
        [[0.027280932,-0.011064238,0.022760907,-0.0044778143],
        [0.09789747,0.09476159,0.15628885,0.109026045],
        [-0.03227183,-0.10743488,-0.09318911,-0.10548531],
        [0.05016677,0.09146461,0.10986483,0.107886836]]
    )
    let expectedGradUpdateBiasForward = Tensor<Float>(
        [-0.0071913204,-0.09881051,-0.11562111,0.2993651]
    )
    let expectedGradResetBiasForward = Tensor<Float>(
        [-0.019118855,0.0300491,-0.010341872,-0.008104472]
    )
    let expectedGradOutputBiasForward = Tensor<Float>(
        [0.61583984,0.5973235,0.9004895,0.605246]
    )
    let expectedGradUpdateRecurrentBiasForward = Tensor<Float>(
        [-0.0071913204,-0.09881051,-0.11562111,0.2993651]
    )
    let expectedGradResetRecurrentBiasForward = Tensor<Float>(
        [-0.019118855,0.0300491,-0.010341872,-0.008104472]
    )
    let expectedGradOutputRecurrentBiasForward = Tensor<Float>(
        [0.33583188,0.27820933,0.5110848,0.33206433]
    )
    let expectedGradInitialStateBackward = Tensor<Float>(
        [[0.43691894,0.39781284,0.48556933,0.3400339]]
    )
    let expectedGradUpdateKernelBackward = Tensor<Float>(
        [[0.080392286,-0.02186797,-0.106636874,0.05397816],
        [0.038964726,-0.010599021,-0.051685017,0.026162265],
        [0.09675335,-0.026318438,-0.12833911,0.06496355]]
    )
    let expectedGradResetKernelBackward = Tensor<Float>(
        [[-0.006548254,0.033489596,-0.03336197,0.033302337],
        [-0.0031738235,0.016231818,-0.016169962,0.016141057],
        [-0.007880924,0.040305246,-0.04015165,0.040079877]]
    )
    let expectedGradOutputKernelBackward = Tensor<Float>(
        [[-0.22416282,-0.14797445,-0.24141216,-0.29072288],
        [-0.10864778,-0.071720615,-0.11700823,-0.14090827],
        [-0.2697834,-0.17808953,-0.29054326,-0.34988946]]
    )
    let expectedGradUpdateRecurrentKernelBackward = Tensor<Float>(
        [[0.11208751,-0.03048957,-0.14867921,0.07525943],
        [0.0779456,-0.021202432,-0.10339145,0.05233537],
        [-0.118453026,0.03222109,0.15712279,-0.07953346],
        [0.12208392,-0.03320875,-0.16193901,0.08197136]]
    )
    let expectedGradResetRecurrentKernelBackward = Tensor<Float>(
        [[-0.009129949,0.046693105,-0.046515163,0.04643202],
        [-0.006348962,0.032470364,-0.03234662,0.032288805],
        [0.009648444,-0.049344834,0.04915679,-0.04906892],
        [-0.009944194,0.05085738,-0.050663568,0.05057301]]
    )
    let expectedGradOutputRecurrentKernelBackward = Tensor<Float>(
        [[-0.18567993,-0.14199756,-0.13620174,-0.24541469],
        [-0.12912174,-0.09874504,-0.09471462,-0.17066126],
        [0.1962248,0.1500617,0.14393671,0.25935194],
        [-0.2022396,-0.15466146,-0.14834875,-0.26730174]]
    )
    let expectedGradUpdateBiasBackward = Tensor<Float>(
        [-0.11448185,0.03114087,0.1518552,-0.076867074]
    )
    let expectedGradResetBiasBackward = Tensor<Float>(
        [0.009324977,-0.047690533,0.04750879,-0.04742387]
    )
    let expectedGradOutputBiasBackward = Tensor<Float>(
        [0.31921688,0.21072157,0.34378064,0.41400108]
    )
    let expectedGradUpdateRecurrentBiasBackward = Tensor<Float>(
        [-0.11448185,0.03114087,0.1518552,-0.076867074]
    )
    let expectedGradResetRecurrentBiasBackward = Tensor<Float>(
        [0.009324977,-0.047690533,0.04750879,-0.04742387]
    )
    let expectedGradOutputRecurrentBiasBackward = Tensor<Float>(
        [0.1896463,0.14503083,0.13911119,0.25065708]
    )
    // END GENERATED CODE
    withTensorLeakChecking {
      var gru = BidirectionalGRU<Float>(
        GRUCell(inputSize: 3, hiddenSize: 4),
        mergeFunction: GRUCell<Float>.State.stack
      )

      XCTAssertEqual(gru.forward.cell.updateKernel.shape, updateKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.updateRecurrentKernel.shape, updateRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.resetKernel.shape, resetKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.resetRecurrentKernel.shape, resetRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.outputKernel.shape, outputKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.outputRecurrentKernel.shape, outputRecurrentKernelForward.shape)
      XCTAssertEqual(gru.forward.cell.updateBias.shape, updateBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.updateRecurrentBias.shape, updateRecurrentBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.resetBias.shape, resetBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.resetRecurrentBias.shape, resetRecurrentBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.outputBias.shape, outputBiasForward.shape)
      XCTAssertEqual(gru.forward.cell.outputRecurrentBias.shape, outputRecurrentBiasForward.shape)

      XCTAssertEqual(gru.backward.cell.updateKernel.shape, updateKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateRecurrentKernel.shape, updateRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetKernel.shape, resetKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetRecurrentKernel.shape, resetRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputKernel.shape, outputKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputRecurrentKernel.shape, outputRecurrentKernelBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateBias.shape, updateBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.updateRecurrentBias.shape, updateRecurrentBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetBias.shape, resetBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.resetRecurrentBias.shape, resetRecurrentBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputBias.shape, outputBiasBackward.shape)
      XCTAssertEqual(gru.backward.cell.outputRecurrentBias.shape, outputRecurrentBiasBackward.shape)

      gru.forward.cell.updateKernel = updateKernelForward
      gru.forward.cell.updateRecurrentKernel = updateRecurrentKernelForward
      gru.forward.cell.resetKernel = resetKernelForward
      gru.forward.cell.resetRecurrentKernel = resetRecurrentKernelForward
      gru.forward.cell.outputKernel = outputKernelForward
      gru.forward.cell.outputRecurrentKernel = outputRecurrentKernelForward

      gru.forward.cell.updateBias = updateBiasForward
      gru.forward.cell.updateRecurrentBias = updateRecurrentBiasForward
      gru.forward.cell.resetBias = resetBiasForward
      gru.forward.cell.resetRecurrentBias = resetRecurrentBiasForward
      gru.forward.cell.outputBias = outputBiasForward
      gru.forward.cell.outputRecurrentBias = outputRecurrentBiasForward

      gru.backward.cell.updateKernel = updateKernelBackward
      gru.backward.cell.updateRecurrentKernel = updateRecurrentKernelBackward
      gru.backward.cell.resetKernel = resetKernelBackward
      gru.backward.cell.resetRecurrentKernel = resetRecurrentKernelBackward
      gru.backward.cell.outputKernel = outputKernelBackward
      gru.backward.cell.outputRecurrentKernel = outputRecurrentKernelBackward

      gru.backward.cell.updateBias = updateBiasBackward
      gru.backward.cell.updateRecurrentBias = updateRecurrentBiasBackward
      gru.backward.cell.resetBias = resetBiasBackward
      gru.backward.cell.resetRecurrentBias = resetRecurrentBiasBackward
      gru.backward.cell.outputBias = outputBiasBackward
      gru.backward.cell.outputRecurrentBias = outputRecurrentBiasBackward

      let inputs = x.squeezingShape(at: 0).unstacked().map { $0.rankLifted() }
      let initialForwardLayerState = GRUCell<Float>.State(initialForwardLayerState)
      let initialBackwardLayerState = GRUCell<Float>.State(initialBackwardLayerState)

      let outputs = gru(
        inputs,
        initialForwardLayerState: initialForwardLayerState,
        initialBackwardLayerState: initialBackwardLayerState
      ).map { $0.unstacked(alongAxis: 0) }
      assertEqual(
        Tensor(concatenating: outputs.map { $0[0] }),
        expectedStatesForward,
        accuracy: 1e-6)
      assertEqual(
        Tensor(concatenating: outputs.map { $0[1] }),
        expectedStatesBackward,
        accuracy: 1e-6)

      let (outputSum, (gradGRU)) =
        valueWithGradient(at: gru) { gru in
          gru.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        outputSum,
        expectedSum,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateKernel,
        expectedGradUpdateKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetKernel,
        expectedGradResetKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputKernel,
        expectedGradOutputKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateRecurrentKernel,
        expectedGradUpdateRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetRecurrentKernel,
        expectedGradResetRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputRecurrentKernel,
        expectedGradOutputRecurrentKernelForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateBias,
        expectedGradUpdateBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetBias,
        expectedGradResetBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputBias,
        expectedGradOutputBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.updateRecurrentBias,
        expectedGradUpdateRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.resetRecurrentBias,
        expectedGradResetRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.forward.cell.outputRecurrentBias,
        expectedGradOutputRecurrentBiasForward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateKernel,
        expectedGradUpdateKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetKernel,
        expectedGradResetKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputKernel,
        expectedGradOutputKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateRecurrentKernel,
        expectedGradUpdateRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetRecurrentKernel,
        expectedGradResetRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputRecurrentKernel,
        expectedGradOutputRecurrentKernelBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateBias,
        expectedGradUpdateBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetBias,
        expectedGradResetBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputBias,
        expectedGradOutputBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.updateRecurrentBias,
        expectedGradUpdateRecurrentBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.resetRecurrentBias,
        expectedGradResetRecurrentBiasBackward,
        accuracy: 1e-6)
      assertEqual(
        gradGRU.backward.cell.outputRecurrentBias,
        expectedGradOutputRecurrentBiasBackward,
        accuracy: 1e-6)

      let (gradInputs, gradInitialStateForward, gradInitialStateBackward) =
        gradient(at: inputs, initialForwardLayerState, initialBackwardLayerState) {
          inputs, initialForwardLayerState, initialBackwardLayerState in
          gru.lastOutput(
            from: inputs,
            initialForwardLayerState: initialForwardLayerState,
            initialBackwardLayerState: initialBackwardLayerState
          ).sum()
        }
      assertEqual(
        Tensor(concatenating: gradInputs.map { $0 }),
        expectedGradX,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateForward,
        expectedGradInitialStateForward,
        accuracy: 1e-6)
      assertEqual(
        gradInitialStateBackward,
        expectedGradInitialStateBackward,
        accuracy: 1e-6)
    }
  }

  func testFunction() {
    let tanhLayer = Function<Tensor<Float>, Tensor<Float>>(tanh)
    let input = Tensor(shape: [5, 1], scalars: (0..<5).map(Float.init))
    let output = tanhLayer.inferring(from: input)
    let expected = Tensor<Float>([[0.0], [0.7615942], [0.9640276], [0.9950547], [0.9993292]])
    assertEqual(output, expected, accuracy: 1e-6)
  }

  func testBatchNorm() {
    let x = Tensor<Float>([
      [-1.0474433, -0.11914538, -0.08634827, 0.15446888, 1.0572497],
      [1.5165012, 0.3753972, -0.30856386, -0.3100725, -1.9584457],
      [0.006384419, 1.4424847, 0.91568077, 0.66328526, -1.0794537],
      [1.056803, 0.14263044, -1.8308276, 0.4189805, 0.6933893],
      [0.30175626, -0.16121633, -0.4191958, -0.53092813, -0.029484272],
    ])
    let bnLayer = BatchNorm<Float>(featureCount: 5, axis: 0)
    Context.local.learningPhase = .training
    withTensorLeakChecking {
      let output = bnLayer(x)
      let grad = gradient(at: x, bnLayer) { $1($0).squared().sum() }
      // The expected values and gradients were computed using the following Python code:
      // ```
      // import tensorflow as tf
      // x = tf.constant(
      //        [[  -1.0474433,  -0.11914538,  -0.08634827,   0.15446888,    1.0572497],
      //         [   1.5165012,    0.3753972,  -0.30856386,   -0.3100725,   -1.9584457],
      //         [ 0.006384419,    1.4424847,   0.91568077,   0.66328526,   -1.0794537],
      //         [    1.056803,   0.14263044,   -1.8308276,    0.4189805,    0.6933893],
      //         [  0.30175626,  -0.16121633,   -0.4191958,  -0.53092813, -0.029484272]])
      // scale = tf.reshape(tf.constant([1., 1., 1., 1., 1.]), [5, 1])
      // offset = tf.reshape(tf.constant([0., 0., 0., 0., 0.]), [5, 1])
      // (mean, var) = tf.nn.moments(x, axes=1, keepdims=True)
      // bn = tf.nn.batch_normalization(x, mean, var, offset=offset, scale=scale,
      //                                variance_epsilon=0.001)
      // scaled = tf.reduce_sum(tf.square(bn))
      // g = tf.gradients(scaled, [x, offset, scale])
      // init = tf.initialize_all_variables()
      // with tf.Session() as sess:
      //     sess.run(init)
      //     print(sess.run([bn, g]))
      // ```
      assertEqual(
        output,
        [
          [-1.5439795, -0.16477099, -0.11604305, 0.24174842, 1.5830451],
          [1.4639764, 0.45368853, -0.15186328, -0.15319899, -1.6126028],
          [-0.44139984, 1.2124169, 0.60574806, 0.3150888, -1.6918538],
          [0.9507547, 0.04595902, -1.9072568, 0.31947452, 0.5910686],
          [1.5834246, 0.02224666, -0.8476793, -1.2244489, 0.46645695],
        ],
        accuracy: 1e-5)
      assertEqual(
        grad.0,
        [
          [-1.0127544e-02, -1.0807812e-03, -7.6115131e-04, 1.5857220e-03, 1.0383606e-02],
          [2.0323221e-03, 6.2976527e-04, -2.1077941e-04, -2.1265696e-04, -2.2384699e-03],
          [-1.3483668e-03, 3.7030075e-03, 1.8500184e-03, 9.6232636e-04, -5.1673558e-03],
          [1.8438101e-03, 8.9146197e-05, -3.6990643e-03, 6.1964989e-04, 1.1463165e-03],
          [1.2142579e-01, 1.7060755e-03, -6.5005139e-02, -9.3897656e-02, 3.5770576e-02],
        ],
        accuracy: 1e-5)
      assertEqual(grad.1.offset, [0.0, 0.0, 0.0, 0.0, 0.0], accuracy: 1e-5)
      assertEqual(
        grad.1.scale,
        [9.977925, 9.992161, 9.986738, 9.990202, 9.886292],
        accuracy: 1e-5)
    }
  }

  func testBatchNormInference() {
    Context.local.learningPhase = .inference
    // This tests for a specific failure that had impacted the MiniGo model.
    let miniGoTensor = Tensor<Float>(randomUniform: [2, 19, 19, 256])
    let miniGoBatchNorm = BatchNorm<Float>(featureCount: 256, momentum: 0.95, epsilon: 1e-5)
    let miniGoResult = miniGoBatchNorm(miniGoTensor)
    XCTAssertEqual(miniGoTensor.shape, miniGoResult.shape)

    let x = Tensor<Float>(rangeFrom: 0, to: 20, stride: 1).reshaped(to: [4, 5])
    let epsilon: Float = 0.001
    let bnLayer = BatchNorm<Float>(featureCount: 5, axis: 1, epsilon: epsilon)
    // Test inference before any training.
    assertEqual(bnLayer.inferring(from: x), x / TensorFlow.sqrt(1 + epsilon), accuracy: 1e-5)
    // Perform one training step, updating the running mean and variance.
    Context.local.learningPhase = .training
    _ = bnLayer(x)  // This line is important and cannot be removed.
    // Test inference after training step.
    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(20, dtype=tf.float32), [4,5])
    // bn = tf.nn.batch_normalization(x, mean, var, offset=offset, scale=scale,
    //                                variance_epsilon=0.001)
    // y_train = bnLayer(x, training=True)
    // y = bnLayer(x, training=False)
    // print(y)
    // ```
    assertEqual(
      bnLayer.inferring(from: x),
      [
        [-0.06569097, 0.8014299, 1.6685508, 2.5356717, 3.4027927],
        [4.3137074, 5.180828, 6.0479493, 6.91507, 7.7821913],
        [8.693106, 9.560227, 10.427347, 11.294469, 12.16159],
        [13.072505, 13.939626, 14.806746, 15.673867, 16.540987],
      ],
      accuracy: 1e-5)
  }

  func testLayerNorm() {
    let x = Tensor<Float>([
      [2.736876, -0.8932728, -0.11240143, 1.252899, -0.35648823],
      [-0.43356904, -0.5147881, 0.8055815, 0.97228354, 1.4561518],
      [0.56300443, -0.87069905, -0.20677163, 1.1823419, 1.0455104],
      [-0.8246169, 1.4249208, 1.2131604, 1.1445689, -0.94032115],
    ])
    let lnLayer = LayerNorm<Float>(featureCount: 5, axis: 1)
    let value = lnLayer(x)
    let grad = gradient(at: x, lnLayer) { $1($0).squared().sum() }
    // The expected values and gradients were computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.constant([[ 2.736876  , -0.8932728 , -0.11240143,  1.252899  , -0.35648823],
    //                  [-0.43356904, -0.5147881 ,  0.8055815 ,  0.97228354,  1.4561518 ],
    //                  [ 0.56300443, -0.87069905, -0.20677163,  1.1823419 ,  1.0455104 ],
    //                  [-0.8246169 ,  1.4249208 ,  1.2131604 ,  1.1445689 , -0.94032115]])
    // lnLayer = tf.keras.layers.LayerNormalization(axis=1, epsilon=0.001)
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = lnLayer(x)
    //     z = tf.math.reduce_sum(tf.math.square(y))
    // print(y, tape.gradient(z, [x] + lnLayer.trainable_variables))
    // ```
    assertEqual(
      value,
      [
        [1.6839857, -1.0804383, -0.4857906, 0.5539104, -0.6716671],
        [-1.1261504, -1.228839, 0.44055927, 0.6513276, 1.2631025],
        [0.28318238, -1.5595294, -0.70619607, 1.079205, 0.9033381],
        [-1.1639192, 0.96795416, 0.7672701, 0.70226634, -1.2735714],
      ],
      accuracy: 1e-5)
    assertEqual(
      grad.0,
      [
        [0.00148721, -0.00095408, -0.00042902, 0.00048913, -0.00059323],
        [-0.00455132, -0.00496664, 0.00178061, 0.00263247, 0.00510535],
        [0.0012024, -0.00662184, -0.00299847, 0.00458241, 0.00383568],
        [-0.0019815, 0.00164783, 0.00130618, 0.00119543, -0.00216818],
      ],
      accuracy: 1e-5)
    assertEqual(
      grad.1.offset,
      [-0.645803, -5.8017054, 0.03168535, 5.973418, 0.44240427],
      accuracy: 1e-5)
    assertEqual(
      grad.1.scale,
      [11.077844, 12.092919, 3.0350027, 4.7778125, 8.969137],
      accuracy: 1e-5)
  }

  func testLayerNormInference() {
    Context.local.learningPhase = .inference
    // This tests for a specific failure that had impacted the Transformer model.
    let transformerTensor = Tensor<Float>(randomUniform: [1, 1, 768])
    let transformerLayerNorm = LayerNorm<Float>(featureCount: 768, axis: -1, epsilon: 1e-5)
    let transformerResult = transformerLayerNorm(transformerTensor)
    XCTAssertEqual(transformerTensor.shape, transformerResult.shape)
  }

  func testGroupNorm() {
    // The expected values were computed using the following Python code:
    // ```
    // import tensorflow as tf
    // import tensorflow_addons as tfa
    // x = tf.reshape(tf.range(24, dtype=tf.float32), [2, 2, 1, 6])
    // layer = tfa.layers.GroupNormalization(groups=2)
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = layer(x)
    //     z = tf.math.reduce_sum(tf.math.square(y))
    // print(y, tape.gradient(z, [x] + layer.trainable_variables))
    // ```
    let tensor = Tensor<Float>(rangeFrom: 0, to: 24, stride: 1)
      .reshaped(to: [2, 2, 1, 6])
    let layer = GroupNorm<Float>(featureCount: 6, groupCount: 2)
    let output = layer(tensor)
    let expectedOutput: Tensor<Float> = [
      [
        [[-1.2864685, -0.9648514, -0.64323425, -1.2864685, -0.9648514, -0.64323425]],
        [[0.64323425, 0.9648514, 1.2864685, 0.64323425, 0.9648514, 1.2864685]]
      ],
      [
        [[-1.2864685, -0.9648514, -0.64323425, -1.2864685, -0.9648514, -0.64323425]],
        [[0.64323425, 0.9648514, 1.2864685, 0.64323425, 0.9648514, 1.2864685]]
      ]
    ]
    XCTAssert(output.isAlmostEqual(to: expectedOutput))
    let grad = gradient(at: tensor, layer) { $1($0).squared().sum() }
    let expectedGrad: Tensor<Float> = [
      [[[-8.5592270e-05, -6.4194202e-05, -4.2796135e-05,
         -8.5592270e-05, -6.4194202e-05, -4.2796135e-05]],
       [[4.2796135e-05, 6.4194202e-05, 8.5592270e-05,
         4.2796135e-05, 6.4194202e-05, 8.5592270e-05]]],
      [[[-8.5592270e-05, -6.4194202e-05, -4.2796135e-05,
         -8.5592270e-05, -6.4194202e-05, -4.2796135e-05]],
       [[4.2796135e-05, 6.4194202e-05, 8.5592270e-05,
         4.2796135e-05, 6.4194202e-05, 8.5592270e-05]]]
    ]
    XCTAssert(grad.0.isAlmostEqual(to: expectedGrad))
    XCTAssert(
      grad.1.scale.isAlmostEqual(to: [
        8.275006, 7.4475055, 8.275006,
        8.275006, 7.4475055, 8.275006
      ]))
    XCTAssert(
      grad.1.offset.isAlmostEqual(to: [
        -2.572937, 0, 2.572937,
        -2.572937, 0, 2.572937
      ]))
  }

  func testInstanceNorm() {
    // The expected values were computed using the following Python code:
    // ```
    // import tensorflow as tf
    // import tensorflow_addons as tfa
    // x = tf.reshape(tf.range(24, dtype=tf.float32), [2, 2, 1, 6])
    // layer = tfa.layers.InstanceNormalization()
    // with tf.GradientTape() as tape:
    //     tape.watch(x)
    //     y = layer(x)
    //     z = tf.math.reduce_sum(tf.math.square(y))
    // print(y, tape.gradient(z, [x] + layer.trainable_variables))
    // ```
    let tensor = Tensor<Float>(rangeFrom: 0, to: 24, stride: 1)
      .reshaped(to: [2, 2, 1, 6])
    let layer = InstanceNorm<Float>(featureCount: 6)
    let output = layer(tensor)
    let expected: Tensor<Float> = [
      [
        [[-0.99994445, -0.99994445, -0.9999444, -0.99994445, -0.9999443, -0.99994445]],
        [[0.99994445, 0.9999443, 0.99994445, 0.99994445, 0.99994445, 0.99994445]]
      ],
      [
        [[-0.9999442, -0.9999442, -0.9999447, -0.9999447, -0.9999442, -0.9999442]],
        [[0.9999447, 0.9999442, 0.9999442, 0.9999442, 0.9999447, 0.9999447]]
      ]
    ]
    XCTAssertEqual(output, expected)
    let grad = gradient(at: tensor, layer) { $1($0).squared().sum() }
    let expectedGrad: Tensor<Float> = [
      [[[-7.4148178e-05, -7.4108444e-05, -7.4108444e-05,
         -7.4088573e-05, -7.4068703e-05, -7.4148178e-05]],
       [[7.4148178e-05, 7.4068703e-05, 7.4128307e-05,
         7.4088573e-05, 7.4108444e-05, 7.4148178e-05]]],
      [[[-7.4128300e-05, -7.4207783e-05, -7.4108451e-05,
         -7.4048847e-05, -7.4128300e-05, -7.4128300e-05]],
       [[7.4108451e-05, 7.4207783e-05, 7.4128300e-05,
         7.4068696e-05, 7.4108451e-05, 7.4108451e-05]]]
    ]
    XCTAssert(grad.0.isAlmostEqual(to: expectedGrad))
    XCTAssert(
      grad.1.delegate.scale
        .isAlmostEqual(to: [
          7.999111, 7.9991093, 7.9991107,
          7.9991117, 7.999111, 7.99911
        ]))
    XCTAssert(
      grad.1.delegate.offset
        .isAlmostEqual(to: [
          9.5367432e-07, -2.3841858e-07, -8.3446503e-07,
          -9.5367432e-07, 1.1920929e-06, 9.5367432e-07
        ]))
  }
  func testGaussianNoise() {
    Context.local.learningPhase = .inference
    let gaussianNoise = GaussianNoise<Float>(standardDeviation: 1.0)
    let x = Tensor<Float>(repeating: 1.0, shape: [5, 5])
    XCTAssertEqual(gaussianNoise(x), x)
    withLearningPhase(LearningPhase.inference) {
      XCTAssertEqual(gaussianNoise(x), x)
      withLearningPhase(LearningPhase.training) {
        XCTAssertNotEqual(gaussianNoise(x), x)
      }
      XCTAssertEqual(gaussianNoise(x), x)
    }
    XCTAssertEqual(gaussianNoise(x), x)
  }

  func testGaussianDropout() {
    Context.local.learningPhase = .inference
    let dropout = GaussianDropout<Float>(probability: 0.5)
    let x = Tensor<Float>(repeating: 1.0, shape: [5, 5])
    XCTAssertEqual(dropout(x), x)
    withLearningPhase(LearningPhase.inference) {
      XCTAssertEqual(dropout(x), x)
      withLearningPhase(LearningPhase.training) {
        XCTAssertNotEqual(dropout(x), x)
      }
      XCTAssertEqual(dropout(x), x)
    }
    XCTAssertEqual(dropout(x), x)
  }

  func testAlphaDropout() {
    Context.local.learningPhase = .inference
    let dropout = AlphaDropout<Float>(probability: 0.5)
    let x = Tensor<Float>(repeating: 1.0, shape: [4, 4])
    XCTAssertEqual(dropout(x), x)
    withLearningPhase(LearningPhase.inference) {
      XCTAssertEqual(dropout(x), x)
      withLearningPhase(LearningPhase.training) {
        XCTAssertNotEqual(dropout(x), x)
      }
      XCTAssertEqual(dropout(x), x)
    }
    XCTAssertEqual(dropout(x), x)
  }

  func testDilation2D() {
    let filter = Tensor(shape: [2, 2, 3], scalars: (0..<12).map(Float.init))
    let layer = Dilation2D<Float>(
      filter: filter, strides: (2, 2), padding: .valid)
    let input = Tensor(shape: [2, 2, 2, 3], scalars: (0..<24).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(
      shape: [2, 1, 1, 3],
      scalars: [18, 20, 22, 30, 32, 34])
    XCTAssertEqual(output, expected)
  }

func testDilation2DGradient() {
   let filter = Tensor(shape: [3, 2, 4], scalars: (0..<24).map(Float.init))
   let layer = Dilation2D<Float>(
     filter: filter,
     strides: (1, 1),
     rates: (1, 1),
     padding: .valid)
   let input = Tensor(shape: [2, 4, 4, 4], scalars: (0..<128).map(Float.init))
   let grads = gradient(at: input, layer) { $1($0).sum() }
   // The expected value of the gradient was computed using the following Python code:
   // ```
   // filters = tf.reshape(tf.range(24, dtype=tf.float32), [3, 2, 4])
   // image = tf.reshape(tf.range(128, dtype=tf.float32), [2, 4, 4, 4])
   // with tf.GradientTape() as tape:
   //    tape.watch([image, filters])
   //    y = tf.nn.dilation2d(
   //        image, filters, [1, 1, 1, 1], "VALID", "NHWC", [1, 1, 1, 1], name=None)
   //    print(tape.gradient(y, [image, filters]))
   // ```
   XCTAssertEqual(
     grads.0,
   [
    [[[0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]],

   [[0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]],

   [[0, 0, 0, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]],

   [[0, 0, 0, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]]],

   [[[0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]],

   [[0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]],

   [[0, 0, 0, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]],

   [[0, 0, 0, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]]]
   ]
    )
   XCTAssertEqual(
     grads.1.filter,
     [
        [[0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[0, 0, 0, 0],
        [0, 0, 0, 0]],

        [[0, 0, 0, 0],
         [12, 12, 12, 12]],
     ]

    )
 }

 func testErosion2D() {
    let filter = Tensor(shape: [2, 2, 3], scalars: (0..<12).map(Float.init))
    let layer = Erosion2D<Float>(
      filter: filter, strides: (2, 2), padding: .valid)
    let input = Tensor(shape: [2, 2, 2, 3], scalars: (0..<24).map(Float.init))
    let output = layer.inferring(from: input)
    let expected = Tensor<Float>(
      shape: [2, 1, 1, 3],
      scalars: [-9, -9, -9, 3, 3, 3])
    XCTAssertEqual(output, expected)
  }

func testErosion2DGradient() {
   let filter = Tensor(shape: [3, 2, 4], scalars: (0..<24).map(Float.init))
   let layer = Erosion2D<Float>(
     filter: filter,
     strides: (1, 1),
     rates: (1, 1),
     padding: .valid)
   let input = Tensor(shape: [2, 4, 4, 4], scalars: (0..<128).map(Float.init))
   let grads = gradient(at: input, layer) { $1($0).sum() }
   // The expected value of the gradient was computed using the following Python code:
   // ```
   // filters = tf.reshape(tf.range(24, dtype=tf.float32), [3, 2, 4])
   // image = tf.reshape(tf.range(128, dtype=tf.float32), [2, 4, 4, 4])
   // with tf.GradientTape() as tape:
   //    tape.watch([image, filters])
   //    y = tf.nn.erosion2d(
   //        image, filters, [1, 1, 1, 1], "VALID", "NHWC", [1, 1, 1, 1], name=None)
   //    print(tape.gradient(y, [image, filters]))
   // ```
   XCTAssertEqual(
     grads.0,
   [[[[1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [0, 0, 0, 0]],

    [[1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [0, 0, 0, 0]],

    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]],


    [[[1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [0, 0, 0, 0]],

    [[1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [0, 0, 0, 0]],

    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]]]
    )
   XCTAssertEqual(
     grads.1.filter,
     [[[0, 0, 0, 0],
       [0, 0, 0, 0]],

        [[0, 0, 0, 0],
        [0, 0, 0, 0]],

        [[0, 0, 0, 0],
         [-12, -12, -12, -12]],
     ]

    )
 }

  static var allTests = [
    ("testConv1D", testConv1D),
    ("testConv1DDilation", testConv1DDilation),
    ("testConv2D", testConv2D),
    ("testConv2DGradient", testConv2DGradient),
    ("testConv2DDilation", testConv2DDilation),
    ("testConv3D", testConv3D),
    ("testConv3DGradient", testConv3DGradient),
    ("testTransposedConv1D", testTransposedConv1D),
    ("testTransposedConv2D", testTransposedConv2D),
    ("testTransposedConv2DGradient", testTransposedConv2DGradient),
    ("testTransposedConv3D", testTransposedConv3D),
    ("testDepthwiseConv2D", testDepthwiseConv2D),
    ("testDepthwiseConv2DGradient", testDepthwiseConv2DGradient),
    ("testSeparableConv1D", testSeparableConv1D),
    ("testSeparableConv2D", testSeparableConv2D),
    ("testSeparableConv2DGradient", testSeparableConv2DGradient),
    ("testZeroPadding1D", testZeroPadding1D),
    ("testZeroPadding1DGradient", testZeroPadding1DGradient),
    ("testZeroPadding2D", testZeroPadding2D),
    ("testZeroPadding2DGradient", testZeroPadding2DGradient),
    ("testZeroPadding3D", testZeroPadding3D),
    ("testZeroPadding3DGradient", testZeroPadding3DGradient),
    ("testMaxPool1D", testMaxPool1D),
    ("testMaxPool1DGradient", testMaxPool1DGradient),
    ("testMaxPool2D", testMaxPool2D),
    ("testMaxPool2DGradient", testMaxPool2DGradient),
    ("testMaxPool3D", testMaxPool3D),
    ("testMaxPool3DGradient", testMaxPool3DGradient),
    ("testAvgPool1D", testAvgPool1D),
    ("testAvgPool1DGradient", testAvgPool1DGradient),
    ("testAvgPool2D", testAvgPool2D),
    ("testAvgPool2DGradient", testAvgPool2DGradient),
    ("testAvgPool3D", testAvgPool3D),
    ("testAvgPool3DGradient", testAvgPool3DGradient),
    ("testGlobalAvgPool1D", testGlobalAvgPool1D),
    ("testGlobalAvgPool1DGradient", testGlobalAvgPool1DGradient),
    ("testGlobalAvgPool2D", testGlobalAvgPool2D),
    ("testGlobalAvgPool2DGradient", testGlobalAvgPool2DGradient),
    ("testGlobalAvgPool3D", testGlobalAvgPool3D),
    ("testGlobalAvgPool3DGradient", testGlobalAvgPool3DGradient),
    ("testGlobalMaxPool1D", testGlobalMaxPool1D),
    ("testGlobalMaxPool1DGradient", testGlobalMaxPool1DGradient),
    ("testGlobalMaxPool2D", testGlobalMaxPool2D),
    ("testGlobalMaxPool2DGradient", testGlobalMaxPool2DGradient),
    ("testGlobalMaxPool3D", testGlobalMaxPool3D),
    ("testGlobalMaxPool3DGradient", testGlobalMaxPool3DGradient),
    ("testFractionalMaxPool2D", testFractionalMaxPool2D),
    ("testFractionalMaxPool2DGradient", testFractionalMaxPool2DGradient),
    ("testUpSampling1D", testUpSampling1D),
    ("testUpSampling1DGradient", testUpSampling1DGradient),
    ("testUpSampling2D", testUpSampling2D),
    ("testUpSampling2DGradient", testUpSampling2DGradient),
    ("testUpSampling3D", testUpSampling3D),
    ("testUpSampling3DGradient", testUpSampling3DGradient),
    ("testReshape", testReshape),
    ("testReshapeGradient", testReshapeGradient),
    ("testFlatten", testFlatten),
    ("testFlattenGradient", testFlattenGradient),
    ("testEmbedding", testEmbedding),
    ("testEmbeddingGradient", testEmbeddingGradient),
    ("testBasicRNNCell", testBasicRNNCell),
    ("testDense", testDense),
    ("testDenseGradient", testDenseGradient),
    ("testRNN", testRNN),
    ("testBidirectionalBasicRNNSumMerge", testBidirectionalBasicRNNSumMerge),
    ("testBidirectionalBasicRNNConcatenateMerge", testBidirectionalBasicRNNConcatenateMerge),
    ("testBidirectionalBasicRNNAverageMerge", testBidirectionalBasicRNNAverageMerge),
    ("testBidirectionalBasicRNNMultiplyMerge", testBidirectionalBasicRNNMultiplyMerge),
    ("testBidirectionalBasicRNNStackMerge", testBidirectionalBasicRNNStackMerge),
    ("testLSTM", testLSTM),
    ("testBidirectionalLSTMSumMerge", testBidirectionalLSTMSumMerge),
    ("testBidirectionalLSTMConcatenateMerge", testBidirectionalLSTMConcatenateMerge),
    ("testBidirectionalLSTMAverageMerge", testBidirectionalLSTMAverageMerge),
    ("testBidirectionalLSTMMultiplyMerge", testBidirectionalLSTMMultiplyMerge),
    ("testBidirectionalLSTMStackMerge", testBidirectionalLSTMStackMerge),
    ("testGRU", testGRU),
    ("testBidirectionalGRUSumMerge", testBidirectionalGRUSumMerge),
    ("testBidirectionalGRUConcatenateMerge", testBidirectionalGRUConcatenateMerge),
    ("testBidirectionalGRUAverageMerge", testBidirectionalGRUAverageMerge),
    ("testBidirectionalGRUMultiplyMerge", testBidirectionalGRUMultiplyMerge),
    ("testBidirectionalGRUStackMerge", testBidirectionalGRUStackMerge),
    ("testFunction", testFunction),
    ("testBatchNorm", testBatchNorm),
    ("testBatchNormInference", testBatchNormInference),
    ("testLayerNorm", testLayerNorm),
    ("testLayerNormInference", testLayerNormInference),
    ("testGroupNorm", testGroupNorm),
    ("testInstanceNorm", testInstanceNorm),
    ("testGaussianNoise", testGaussianNoise),
    ("testGaussianDropout", testGaussianDropout),
    ("testAlphaDropout", testAlphaDropout),
    ("testDilation2D", testDilation2D),
    ("testDilation2DGradient", testDilation2DGradient),
    ("testErosion2D", testErosion2D),
    ("testErosion2DGradient", testErosion2DGradient)
  ]
}
