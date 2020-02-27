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

final class OptimizerTests: XCTestCase {
    func testRAdam() {
        struct Model: Layer {
            var w: Tensor<Float>
            var b: Tensor<Float>

            public init(w: Tensor<Float>, b: Tensor<Float>) {
                self.w = w
                self.b = b
            }

            @differentiable
            public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
                return matmul(input, w) + b
            }
        }
        let w = Tensor<Float>([[1.0], [2.0]])
        let b = Tensor<Float>([0.0])
        var model = Model(w: w, b: b)
        let optimizer = RAdam(for: model, learningRate: 1e-3)
        // To obtain gradient of type Model.TangentVector 
        var gradient = model.gradient { model -> Tensor<Float> in 
            return meanAbsoluteError(predicted: model(w.tranposed()), expected: Tensor<Float>(1.0))
        }
        // Custom gradient passed to check optimizer validity
        gradient.w = Tensor<Float>([[0.1], [0.2]])
        for _ in 0..<1000 {
            optimizer.update(&model, along: gradient)
        }
        let expected_gradient = Tensor<Float>([[0.5543607], [1.55286]])
        XCTAssertEqual(gradient.w, expected_gradient)
    }

    static var allTests = [
        ("testRAdam", testRAdam)
    ]
}