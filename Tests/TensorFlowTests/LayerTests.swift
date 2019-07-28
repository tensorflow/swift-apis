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
    public init() {}

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        sigmoid(input)
    }
}

final class LayerTests: XCTestCase {
    func testSequential() {
        withRandomSeedForTensorFlow((12345, 12345)) {
            let inputSize = 2
            let hiddenSize = 4
            var model = Sequential {
                Dense<Float>(
                    inputSize: inputSize,
                    outputSize: hiddenSize,
                    weightInitializer: glorotUniform())
                Sigmoid<Float>()
                Dense<Float>(
                    inputSize: hiddenSize,
                    outputSize: 1,
                    weightInitializer: glorotUniform())
            }
            let optimizer = SGD(for: model)
            let x = Tensor<Float>([[0, 0], [0, 1], [1, 0], [1, 1]])
            let y = Tensor<Float>([0, 1, 1, 0])
            let initialLoss = meanSquaredError(
                predicted: model(x).squeezingShape(at: 1),
                expected: y)
            withTensorLeakChecking {
                for _ in 0..<10 {
                    let 𝛁model = model.gradient { model -> Tensor<Float> in
                        meanSquaredError(
                            predicted: model(x).squeezingShape(at: 1),
                            expected: y)
                    }
                    optimizer.update(&model, along: 𝛁model)
                }
            }
            let updatedLoss = meanSquaredError(
                predicted: model(x).squeezingShape(at: 1),
                expected: y)
            XCTAssertLessThan(updatedLoss, initialLoss)
        }
    }

    func testConv1D() {
        let filter = Tensor<Float>(ones: [3, 1, 2]) * Tensor<Float>([[[0.5, 1]]])
        let bias = Tensor<Float>([0, 1])
        let layer = Conv1D<Float>(filter: filter, bias: bias, activation: identity, stride: 1,
                                  padding: .valid)
        let input = Tensor<Float>([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]).expandingShape(at: 2)
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>(
          shape: [2, 3, 2],
          scalars: [1.5, 4, 3, 7, 4.5, 10, 16.5, 34, 18, 37, 19.5, 40])
        XCTAssertEqual(output, expected)
    }

    func testConv1DDilation() {
        // Filter shapes.
        let width = 3
        let inputChannels = 1
        let outputChannels = 2

        // Input shapes.
        let inputHeight = 2
        let inputWidth = 5

        let filter = Tensor<Float>(shape: [width, inputChannels, outputChannels],
                                   scalars: [2, 3, 4, 1, 2, 3])
        let bias = Tensor<Float>([0])
        let layer = Conv1D<Float>(filter: filter, bias: bias, activation: identity, stride: 1,
                                  padding: .valid, dilation: 2)
        let input = Tensor<Float>(shape: [inputHeight, inputWidth, 1],
                                  scalars: [0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>(shape: [2, 1, 2], scalars: [16, 14, 96, 84])
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
        let layer = Conv2D<Float>(filter: filter, bias: bias, activation: identity, strides: (1, 1),
                                  padding: .valid, dilations: (2, 2))
        let input = Tensor<Float>(shape: [batchSize, inputHeight, inputWidth, inputChannels],
                                  scalars: (0..<inputSize).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>(shape: [2, 2, 2, 1],
                                     scalars: [48, 54, 72, 78, 144, 150, 168, 174])
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

    func testTransposedConv1D() {
        let filter =  Tensor(shape: [1, 4, 1, 1], scalars: (0..<4).map(Float.init))
        let bias = Tensor<Float>([8])
        let layer = Conv1D<Float>(filter: filter, bias: bias, activation: identity, stride: 1, padding: .valid)
        let input = Tensor<Float>([0, 1, 2, 3]).expandingShape(at: 2)
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([0, 1, 4, 10])
        XCTAssertEqual(output, expected)
    }

    func testTransposedConv3D() {
        let filter =  Tensor(shape: [2, 2, 2, 1, 1], scalars: (0..<8).map(Float.init))
        let bias = Tensor<Float>([8])
        let layer = Conv3D<Float>(filter: filter, bias: bias, activation: identity,
                                  strides: (1, 1, 1), padding: .valid)
        let input = Tensor(shape: [1, 2, 2, 2, 1], scalars: (0..<8).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>(shape: [1, 2, 2, 2, 1],
                                     scalars: [0, 0, 0, 4, 0, 8, 16, 56])
        XCTAssertEqual(output, expected)
    }

    func testDepthConv2D() {
        let filter =  Tensor(shape: [2, 2, 2, 2], scalars: (0..<16).map(Float.init))
        let bias = Tensor<Float>([1, 2, 3, 4])
        let layer = DepthwiseConv2D<Float>(filter: filter, bias: bias, activation: identity,
                                           strides: (2, 2), padding: .valid)
        let input = Tensor(shape: [1, 1, 8, 2], scalars: (0..<16).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>(shape: [1, 1, 4, 4],
                                     scalars: [9, 12, 23, 28, 25, 36, 55, 68, 41, 60, 87, 108,
                                               57, 84, 119, 148])
        XCTAssertEqual(output, expected)
    }

    func testZeroPadding1D() {
        let input = Tensor<Float>([0.0, 1.0, 2.0])
        let layer = ZeroPadding1D<Float>(padding: 2)
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0])
        XCTAssertEqual(output, expected)
    }

    func testZeroPadding2D() {
        let input = Tensor<Float>(shape: [3, 1], scalars: [0.0, 1.0, 2.0])
        let layer = ZeroPadding2D<Float>(padding: ((0, 0), (0, 1)))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        XCTAssertEqual(output, expected)
    }

    func testZeroPadding3D() {
        let input = Tensor<Float>(shape:[3, 1, 1], scalars: [0.0, 1.0, 2.0])
        let layer = ZeroPadding3D<Float>(padding: ((0, 0), (0, 1), (0, 0)))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>(shape: [3, 2, 1], scalars: [0, 0, 1, 0, 2, 0])
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

    func testGlobalMaxPool1D() {
        let layer = GlobalMaxPool1D<Float>()
        let input = Tensor(shape: [1, 10, 1], scalars: (0..<10).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([9])
        XCTAssertEqual(output, expected)
    }

    func testGlobalMaxPool2D() {
        let layer = GlobalMaxPool2D<Float>()
        let input = Tensor(shape: [1, 2, 10, 1], scalars: (0..<20).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([19])
        XCTAssertEqual(output, expected)
    }

    func testGlobalMaxPool3D() {
        let layer = GlobalMaxPool3D<Float>()
        let input = Tensor<Float>(shape: [1, 2, 3, 5, 1], scalars: (0..<30).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([29])
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
      let expected = TensorShape(
          [1, input.shape[1] * size, input.shape[2] * size, input.shape[3] * size, 1])
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

    func testEmbedding() {
        var layer = Embedding<Float>(vocabularySize: 3, embeddingSize: 5)
        var input = Tensor<Int32>(shape: [2, 3], scalars: [0, 1, 2, 1, 2, 2])
        var output = layer(input)
        let expectedShape = TensorShape([2, 3, 5])
        XCTAssertEqual(output.shape, expectedShape)

        let pretrained = Tensor<Float>(shape:[2, 2], scalars: [0.4, 0.3, 0.2, 0.1])
        layer = Embedding<Float>(embeddings: pretrained)
        input = Tensor<Int32>(shape: [2, 2], scalars: [0, 1, 1, 1])
        output = layer(input)
        let expected = Tensor<Float>([[[0.4, 0.3], [0.2, 0.1]], [[0.2, 0.1],[0.2, 0.1]]])
        XCTAssertEqual(output, expected)
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

    func testDense() {
        let weight = Tensor<Float>(shape: [3, 2], scalars: (0..<6).map(Float.init))
        let bias = Tensor<Float>([[1.0, 2.0]])
        let layer = Dense<Float>(weight: weight, bias: bias, activation: identity)
        let input = Tensor<Float>(shape: [1, 3], scalars: (0..<3).map(Float.init))
        let output = layer.inferring(from: input)
        let expected = Tensor<Float>([[11.0, 15.0]])
        XCTAssertEqual(output, expected)
        XCTAssertFalse(layer.batched)

        let weightBatched = Tensor<Float>(shape: [2, 2, 3], scalars: (0..<12).map(Float.init))
        let biasBatched = Tensor<Float>([[1.0, 2.0, 3.0]])
        let layerBatched = Dense<Float>(weight: weightBatched,
                                        bias: biasBatched,
                                        activation: identity)
        let inputBatched = Tensor<Float>(shape: [2, 2], scalars: (0..<4).map(Float.init))
        let outputBatched = layerBatched.inferring(from: inputBatched)
        let expectedBatched = Tensor<Float>([[4.0, 6.0, 8.0], [40.0, 46.0, 52.0]])
        XCTAssertEqual(outputBatched, expectedBatched)
        XCTAssertTrue(layerBatched.batched)
    }

    // TODO(TF-507): Remove references to `SimpleRNNCell.State` after SR-10697 is fixed.
    func testRNN() {
        let x = Tensor<Float>(rangeFrom: 0.0, to: 0.4, stride: 0.1).rankLifted()
        let inputs: [Tensor<Float>] = Array(repeating: x, count: 4)
        let rnn = RNN(SimpleRNNCell<Float>(inputSize: 4, hiddenSize: 4,
                                           seed: (0xFeed, 0xBeef)))
        withTensorLeakChecking {
            let (outputs, _) = rnn.valueWithPullback(at: inputs) { rnn, inputs in
                return rnn(inputs)
            }
            XCTAssertEqual(outputs.map { $0.value },
                           [[[ 0.20775771,  0.20080023, -0.13768704, -0.18534681]],
                            [[ 0.22666009,  0.30019346, -0.19720285, -0.14683801]],
                            [[ 0.23758979,  0.32101023, -0.20359215,  -0.1787096]],
                            [[ 0.24337786,   0.3389194, -0.21143384,  -0.1675081]]])
        }
        // TODO: Figure out why the following is numerically unstable.
        // let (𝛁rnn, _) = pullback(.init(inputs.map { SimpleRNNCell<Float>.State($0) }))
        // XCTAssertEqual(𝛁rnn.cell.weight,
        //                [[         0.0,          0.0,          0.0,          0.0],
        //                 [  0.02496884,   0.06694733,   0.07978788, -0.022378458],
        //                 [  0.04993768,   0.13389467,   0.15957576, -0.044756915],
        //                 [  0.07490652,   0.20084201,   0.23936366,  -0.06713537],
        //                 [         0.0,          0.0,          0.0,          0.0],
        //                 [         0.0,          0.0,          0.0,          0.0],
        //                 [         0.0,          0.0,          0.0,          0.0],
        //                 [         0.0,          0.0,          0.0,          0.0]])
        // XCTAssertEqual(𝛁rnn.cell.bias, [  0.2496884,  0.66947335,   0.7978788, -0.22378457])
    }

    func testFunction() {
        let tanhLayer = Function<Tensor<Float>, Tensor<Float>>(tanh)
        let input = Tensor(shape: [5, 1], scalars: (0..<5).map(Float.init))
        let output = tanhLayer.inferring(from: input)
        let expected = Tensor<Float>([[0.0], [0.7615942], [0.9640276], [0.9950547], [0.9993292]])
        XCTAssertEqual(output, expected)
    }

    func testBatchNorm() {
        let x = Tensor<Float>([
            [  -1.0474433,  -0.11914538,  -0.08634827,   0.15446888,    1.0572497],
            [   1.5165012,    0.3753972,  -0.30856386,   -0.3100725,   -1.9584457],
            [ 0.006384419,    1.4424847,   0.91568077,   0.66328526,   -1.0794537],
            [    1.056803,   0.14263044,   -1.8308276,    0.4189805,    0.6933893],
            [  0.30175626,  -0.16121633,   -0.4191958,  -0.53092813, -0.029484272]])
        let bnLayer = BatchNorm<Float>(featureCount: 5, axis: 0)
        Context.local.learningPhase = .training
        withTensorLeakChecking {
            let output = bnLayer(x)
            let grad = gradient(at: x, bnLayer) { $1($0).squared().sum() }
            // The expected values and gradients were computed using the following Python code:
            // ```
            //   x = tf.constant(
            //         [[  -1.0474433,  -0.11914538,  -0.08634827,   0.15446888,    1.0572497],
            //          [   1.5165012,    0.3753972,  -0.30856386,   -0.3100725,   -1.9584457],
            //          [ 0.006384419,    1.4424847,   0.91568077,   0.66328526,   -1.0794537],
            //          [    1.056803,   0.14263044,   -1.8308276,    0.4189805,    0.6933893],
            //          [  0.30175626,  -0.16121633,   -0.4191958,  -0.53092813, -0.029484272]])
            //  scale = tf.reshape(tf.constant([1., 1., 1., 1., 1.]), [5, 1])
            //  offset = tf.reshape(tf.constant([0., 0., 0., 0., 0.]), [5, 1])
            //  (mean, var) = tf.nn.moments(x, axes=1, keepdims=True)
            //  bn = tf.nn.batch_normalization(x, mean, var, offset=offset, scale=scale,
            //                                 variance_epsilon=0.001)
            //  scaled = tf.reduce_sum(tf.square(bn))
            //  g = tf.gradients(scaled, [x, offset, scale])
            //  init = tf.initialize_all_variables()
            //  with tf.Session() as sess:
            //    sess.run(init)
            //    print(sess.run([bn, g]))
            // ```
            assertEqual(
                output,
                [[-1.5439795 , -0.16477099, -0.11604305,  0.24174842,  1.5830451 ],
                 [ 1.4639764 ,  0.45368853, -0.15186328, -0.15319899, -1.6126028 ],
                 [-0.44139984,  1.2124169 ,  0.60574806,  0.3150888 , -1.6918538 ],
                 [ 0.9507547 ,  0.04595902, -1.9072568 ,  0.31947452,  0.5910686 ],
                 [ 1.5834246 ,  0.02224666, -0.8476793 , -1.2244489 ,  0.46645695]],
                accuracy: 1e-5)
            assertEqual(
                grad.0,
                [[-1.0127544e-02, -1.0807812e-03, -7.6115131e-04,  1.5857220e-03,  1.0383606e-02],
                 [ 2.0323221e-03,  6.2976527e-04, -2.1077941e-04, -2.1265696e-04, -2.2384699e-03],
                 [-1.3483668e-03,  3.7030075e-03,  1.8500184e-03,  9.6232636e-04, -5.1673558e-03],
                 [ 1.8438101e-03,  8.9146197e-05, -3.6990643e-03,  6.1964989e-04,  1.1463165e-03],
                 [ 1.2142579e-01,  1.7060755e-03, -6.5005139e-02, -9.3897656e-02,  3.5770576e-02]],
                accuracy: 1e-5)
            assertEqual(grad.1.offset, [0.0, 0.0, 0.0, 0.0, 0.0], accuracy: 1e-5)
            assertEqual(grad.1.scale, [9.977925, 9.992161, 9.986738, 9.990202, 9.886292],
                        accuracy: 1e-5)
        }
    }

    func testLayerNorm() {
        let x = Tensor<Float>([
            [  -1.0474433,  -0.11914538,  -0.08634827,   0.15446888,    1.0572497],
            [   1.5165012,    0.3753972,  -0.30856386,   -0.3100725,   -1.9584457],
            [ 0.006384419,    1.4424847,   0.91568077,   0.66328526,   -1.0794537],
            [    1.056803,   0.14263044,   -1.8308276,    0.4189805,    0.6933893],
            [  0.30175626,  -0.16121633,   -0.4191958,  -0.53092813, -0.029484272]])
        let lnLayer = LayerNorm<Float>(featureCount: 5, axis: 1)
        let value = lnLayer(x)
        let grad = gradient(at: x, lnLayer) { $1($0).squared().sum() }

        // Uses the same values as `testBatchNorm()` above because `LayerNorm` with features on axis
        // `1` is equivalent to `BatchNorm` with features on axis `0`.
        assertEqual(
            value,
            [[-1.5439795 , -0.16477099, -0.11604305,  0.24174842,  1.5830451 ],
             [ 1.4639764 ,  0.45368853, -0.15186328, -0.15319899, -1.6126028 ],
             [-0.44139984,  1.2124169 ,  0.60574806,  0.3150888 , -1.6918538 ],
             [ 0.9507547 ,  0.04595902, -1.9072568 ,  0.31947452,  0.5910686 ],
             [ 1.5834246 ,  0.02224666, -0.8476793 , -1.2244489 ,  0.46645695]],
            accuracy: 1e-5)
        assertEqual(
            grad.0,
            [[-1.0127544e-02, -1.0807812e-03, -7.6115131e-04,  1.5857220e-03,  1.0383606e-02],
             [ 2.0323221e-03,  6.2976527e-04, -2.1077941e-04, -2.1265696e-04, -2.2384699e-03],
             [-1.3483668e-03,  3.7030075e-03,  1.8500184e-03,  9.6232636e-04, -5.1673558e-03],
             [ 1.8438101e-03,  8.9146197e-05, -3.6990643e-03,  6.1964989e-04,  1.1463165e-03],
             [ 1.2142579e-01,  1.7060755e-03, -6.5005139e-02, -9.3897656e-02,  3.5770576e-02]],
            accuracy: 1e-5)
        assertEqual(grad.1.scale, [9.977925, 9.992161, 9.986738, 9.990202, 9.886292],
                    accuracy: 1e-5)
        assertEqual(grad.1.offset, [0.0, 0.0, 0.0, 0.0, 0.0], accuracy: 1e-5)
    }

    static var allTests = [
        ("testSequential", testSequential),
        ("testConv1D", testConv1D),
        ("testConv1DDilation", testConv1DDilation),
        ("testConv2D", testConv2D),
        ("testConv2DDilation", testConv2DDilation),
        ("testConv3D", testConv3D),
        ("testTransposedConv1D",testTransposedConv1D),
        ("testTransposedConv3D",testTransposedConv3D),
        ("testDepthConv2D", testDepthConv2D),
        ("testZeroPadding1D", testZeroPadding1D),
        ("testZeroPadding2D", testZeroPadding2D),
        ("testZeroPadding3D", testZeroPadding3D),
        ("testMaxPool1D", testMaxPool1D),
        ("testMaxPool2D", testMaxPool2D),
        ("testMaxPool3D", testMaxPool3D),
        ("testAvgPool1D", testAvgPool1D),
        ("testAvgPool2D", testAvgPool2D),
        ("testAvgPool3D", testAvgPool3D),
        ("testGlobalAvgPool1D", testGlobalAvgPool1D),
        ("testGlobalAvgPool2D", testGlobalAvgPool2D),
        ("testGlobalAvgPool3D", testGlobalAvgPool3D),
        ("testGlobalMaxPool1D", testGlobalMaxPool1D),
        ("testGlobalMaxPool2D", testGlobalMaxPool2D),
        ("testGlobalMaxPool3D", testGlobalMaxPool3D),
        ("testUpSampling1D", testUpSampling1D),
        ("testUpSampling2D", testUpSampling2D),
        ("testUpSampling3D", testUpSampling3D),
        ("testReshape", testReshape),
        ("testFlatten", testFlatten),
        ("testEmbedding", testEmbedding),
        ("testSimpleRNNCell", testSimpleRNNCell),
        ("testDense", testDense),
        ("testRNN", testRNN),
        ("testFunction", testFunction),
        ("testBatchNorm", testBatchNorm),
        ("testLayerNorm", testLayerNorm)
    ]
}
