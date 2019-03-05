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

final class LossTests: XCTestCase {
    func testSoftmaxCrossEntropyWithOneHotLabelsLoss() {
        let logits = Tensor<Float>(shape: [2, 4], scalars: [1, 2, 3, 4, 5, 6, 7, 8])
        let labels = Tensor<Float>(
            shape: [2, 4],
            scalars: [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1])

        let loss = softmaxCrossEntropy(logits: logits, oneHotLabels: labels)
        // Loss for two rows are 1.44019 and 2.44019 respectively.
        let expectedLoss: Float = (1.44019 + 2.44019) / 2.0
        assertAllClose(expected: Tensor(expectedLoss), got: loss)
    }

    func testSoftmaxCrossEntropyWithOneHotLabelsGrad() {
        let logits = Tensor<Float>(shape: [2, 4], scalars: [1, 2, 3, 4, 5, 6, 7, 8])
        let labels = Tensor<Float>(
            shape: [2, 4],
            scalars: [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1])

        // For the logits and labels above, the gradients below are the golden values. To calcuate
        // them by hand, you can do
        //
        //  D Loss / D logits_i = p_i - labels_i
        //
        //  where p_i is softmax(logits_i).
        let expectedGradientsBeforeMean = Tensor<Float>(
            shape: [2, 4],
            scalars: [-0.067941, -0.112856, -0.063117, 0.243914,
                      -0.367941, -0.212856, 0.036883, 0.543914])

        // As the loss is mean loss, we should scale the golden loss numbers.
        let expectedGradients = expectedGradientsBeforeMean / Float(logits.shape[0])
        let gradients = gradient(
            at: logits,
            in: { softmaxCrossEntropy(logits: $0, oneHotLabels: labels) })
        assertAllClose(expected: expectedGradients, got: gradients)
    }

    func assertAllClose(expected a: Tensor<Float>, got b: Tensor<Float>, tol: Float = 1e-6) {
        XCTAssertEqual(a.shape, b.shape)
        for (index, elementInA) in a.scalars.enumerated() {
            let elementInB = b.scalars[index]
            XCTAssertTrue(
                abs(elementInA - elementInB) < tol,
                "Found difference at \(index), expected: \(elementInA), got: \(elementInB).")
        }
    }

    static var allTests = [
        ("testSoftmaxCrossEntropyWithOneHotLabelsLoss",
         testSoftmaxCrossEntropyWithOneHotLabelsLoss),
        ("testSoftmaxCrossEntropyWithOneHotLabelsGrad",
         testSoftmaxCrossEntropyWithOneHotLabelsGrad),
    ]
}
