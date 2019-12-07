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

final class TrivialModelTests: XCTestCase {
    func testXOR() {
        struct Classifier: Layer {
            var l1, l2: Dense<Float>
            init(hiddenSize: Int) {
                l1 = Dense<Float>(
                    inputSize: 2,
                    outputSize: hiddenSize,
                    activation: relu,
                    weightInitializer: glorotUniform(seed: (0xfffffff, 0xfeeff)))
                l2 = Dense<Float>(
                    inputSize: hiddenSize,
                    outputSize: 1,
                    activation: relu,
                    weightInitializer: glorotUniform(seed: (0xffeffe, 0xfffe)))
            }
            @differentiable
            func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
                let h1 = l1(input)
                return l2(h1)
            }
        }
        var classifier = Classifier(hiddenSize: 4)
        let optimizer = SGD(for: classifier, learningRate: 0.02)
        let x: Tensor<Float> = [[0, 0], [0, 1], [1, 0], [1, 1]]
        let y: Tensor<Float> = [[0], [1], [1], [0]]

        Context.local.learningPhase = .training
        withTensorLeakChecking {
            for _ in 0..<3000 {
                let 𝛁model = gradient(at: classifier) { classifier -> Tensor<Float> in
                    let ŷ = classifier(x)
                    return meanSquaredError(predicted: ŷ, expected: y)
                }
                optimizer.update(&classifier, along: 𝛁model)
            }
        }
        let ŷ = classifier.inferring(from: x)
        XCTAssertEqual(round(ŷ), y)
    }

    static var allTests = [
        ("testXOR", testXOR),
    ]
}
