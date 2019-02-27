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

final class TrainerTests: XCTestCase {
    func testSupervisedLearningTrainer() {
        struct Model: Layer {
            var dense1 = Dense<Float>(inputSize: 2, outputSize: 4, activation: relu,
                                      seed: (0xffffffff, 0xfeeff))
            var dense2 = Dense<Float>(inputSize: 4, outputSize: 1, activation: relu,
                                      seed: (0xfeffeffe, 0xfffe))

            @differentiable
            func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
              return input.sequenced(in: context, through: dense1, dense2)
            }
        }
        var model = Model()
        let optimizer = SGD(for: model, learningRate: 0.02, scalarType: Float.self)
        let x: Tensor<Float> = [[0, 0], [0, 1], [1, 0], [1, 1]]
        let y: Tensor<Float> = [0, 1, 1, 0]
        SupervisedLearningTrainer.fit(
            model: model, parameters: &model.allDifferentiableVariables,
            using: optimizer, loss: meanSquaredError,
            input: x, output: y, stepCount: 1000)
        let ŷ = model.inferring(from: x)
        XCTAssertEqual(round(ŷ), y)
    }

    static var allTests = [
        ("testSupervisedLearningTrainer", testSupervisedLearningTrainer),
    ]
}
