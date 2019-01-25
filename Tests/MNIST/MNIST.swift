// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
import Python
@testable import DeepLearning

let gzip = Python.import("gzip")
let np = Python.import("numpy")

func readImagesFile(_ filename: String) -> [Float] {
    let file = gzip.open(filename, "rb").read()
    let data = np.frombuffer(file, dtype: np.uint8, offset: 16)
    let array = data.astype(np.float32) / 255
    return Array(numpyArray: array)!
}

func readLabelsFile(_ filename: String) -> [Int32] {
    let file = gzip.open(filename, "rb").read()
    let data = np.frombuffer(file, dtype: np.uint8, offset: 8)
    let array = data.astype(np.int32)
    return Array(numpyArray: array)!
}

/// Reads MNIST images and labels from specified file paths.
func readMNIST(imagesFile: String, labelsFile: String)
    -> (images: Tensor<Float>, labels: Tensor<Int32>) {
    print("Reading data.")
    let images = readImagesFile(imagesFile)
    let labels = readLabelsFile(labelsFile)
    let rowCount = Int32(labels.count)
    let columnCount = Int32(images.count) / rowCount

    print("Constructing data tensors.")
    let imagesTensor = Tensor(shape: [rowCount, columnCount], scalars: images) / 255
    let labelsTensor = Tensor(labels)
    return (imagesTensor, labelsTensor)
}

struct MNISTClassifier: Layer {
    var l1, l2: Dense<Float>
    init(hiddenSize: Int) {
        l1 = Dense<Float>(inputSize: 784, outputSize: hiddenSize,
                          activation: sigmoid)
        l2 = Dense<Float>(inputSize: hiddenSize, outputSize: 10,
                          activation: logSoftmax)
    }
    func applied(to input: Tensor<Float>) -> Tensor<Float> {
        let h1 = l1.applied(to: input)
        return l2.applied(to: h1)
    }
}

final class MNISTTests: XCTestCase {
    func testMNIST() {
        // Get training data.
        let (images, numericLabels) = readMNIST(imagesFile: "train-images-idx3-ubyte.gz",
                                                labelsFile: "train-labels-idx1-ubyte.gz")
        let labels = Tensor<Float>(oneHotAtIndices: numericLabels, depth: 10)

        let batchSize = images.shape[0]
        let optimizer = RMSProp<MNISTClassifier, Float>(learningRate: 0.2)
        var classifier = MNISTClassifier(hiddenSize: 30)

        // Hyper-parameters.
        let epochCount = 20
        let minibatchSize: Int32 = 10
        let learningRate: Float = 0.2
        var loss = Float.infinity

        // Training loop.
        print("Begin training for \(epochCount) epochs.")

        func minibatch<Scalar>(_ x: Tensor<Scalar>, index: Int32) -> Tensor<Scalar> {
            let start = index * minibatchSize
            return x[start..<start+minibatchSize]
        }

        for epoch in 0...epochCount {
            // Store information for printing accuracy and loss.
            var correctPredictions = 0
            var totalLoss: Float = 0

            let iterationCount = batchSize / minibatchSize
            for i in 0..<iterationCount {
                let images = minibatch(images, index: i)
                let numericLabels = minibatch(numericLabels, index: i)
                let labels = minibatch(labels, index: i)

                let (loss, 𝛁model) = classifier.valueWithGradient { classifier -> Tensor<Float> in
                    let ŷ = classifier.applied(to: images)

                    // Update number of correct predictions.
                    let correctlyPredicted = ŷ.argmax(squeezingAxis: 1) .== numericLabels
                    correctPredictions += Int(Tensor<Int32>(correctlyPredicted).sum().scalarized())

                    return -(labels * ŷ).sum() / Tensor(10)
                }
                optimizer.update(&classifier.allDifferentiableVariables, along: 𝛁model)
                totalLoss += loss.scalarized()
            }
            print("""
                [Epoch \(epoch)] \
                Accuracy: \(correctPredictions)/\(batchSize) \
                (\(Float(correctPredictions) / Float(batchSize)))\t\
                Loss: \(totalLoss / Float(batchSize))
                """)
        }
        print("Done training MNIST.")
    }

    static var allTests = [
        ("testMNIST", testMNIST),
    ]
}
