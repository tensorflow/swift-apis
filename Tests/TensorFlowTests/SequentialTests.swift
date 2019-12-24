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

final class SequentialTests: XCTestCase {
    func testSequential() {
        struct Model: Layer {
            var dense1 = Dense<Float>(
                inputSize: 2,
                outputSize: 4,
                activation: relu,
                weightInitializer: glorotUniform(seed: (0xfffffff, 0xfeeff)))
            var dense2 = Dense<Float>(
                inputSize: 4,
                outputSize: 1,
                activation: relu,
                weightInitializer: glorotUniform(seed: (0xeffeffe, 0xfffe)))

            @differentiable
            func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
              return input.sequenced(through: dense1, dense2)
            }
        }
        var model = Model()
        let sgd = SGD(for: model, learningRate: 0.02)
        let rmsprop = RMSProp(for: model, learningRate: 0.02)
        let adam = Adam(for: model, learningRate: 0.02)
        let adamax = AdaMax(for: model, learningRate: 0.02)
        let amsgrad = AMSGrad(for: model, learningRate: 0.02)
        let adagrad = AdaGrad(for: model, learningRate: 0.02)
        let adadelta = AdaDelta(for: model, learningRate: 0.02)
        let radam = RAdam(for: model, learningRate: 0.02)
        let x: Tensor<Float> = [[0, 0], [0, 1], [1, 0], [1, 1]]
        let y: Tensor<Float> = [0, 1, 1, 0]
        Context.local.learningPhase = .training
        withTensorLeakChecking {
            for _ in 0..<1000 {
                let 𝛁model = gradient(at: model) { model -> Tensor<Float> in
                    let ŷ = model(x)
                    return meanSquaredError(predicted: ŷ, expected: y)
                }
                sgd.update(&model, along: 𝛁model)
                rmsprop.update(&model, along: 𝛁model)
                adam.update(&model, along: 𝛁model)
                adamax.update(&model, along: 𝛁model)
                amsgrad.update(&model, along: 𝛁model)
                adagrad.update(&model, along: 𝛁model)
                adadelta.update(&model, along: 𝛁model)
                radam.update(&model, along: 𝛁model)
            }
        }
        XCTAssertEqual(model.inferring(from: [[0, 0], [0, 1], [1, 0], [1, 1]]),
                       [[0.50378805], [0.50378805], [0.50378805], [0.50378805]])
    }

    static var allTests = [
        ("testSequential", testSequential)
    ]
}
