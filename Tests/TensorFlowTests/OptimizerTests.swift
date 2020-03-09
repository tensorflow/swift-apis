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
        // The expected value of the gradient was computed using the following Python code
        // Requirements: Tensorflow >= 2.1.0 and tf-addons (see https://github.com/tensorflow/addons)
        // ```
        // import tensorflow as tf
        // from tensorflow_addons.optimizers import RectifiedAdam
        // var_0 = tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)
        // grad_0 = tf.Variable([0.1, 0.2], dtype=tf.dtypes.float32)
        // grads_and_vars = list(zip([grad_0], [var_0]))
        // optimizer = RectifiedAdam(lr=1e-3, epsilon=1e-8)
        // for _ in range(1000)
        //     optimizer.apply_gradients(grads_and_vars)
        // print(var_0.read_value())
        // >>> [0.5553605, 1.5548599]
        // Current implementation = [0.5543607, [1.55286]]
        // Difference of [0.0009997, 0.0019999].
        // ```


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
        var grad = gradient(at: model) { model -> Tensor<Float> in
            let ŷ = model(w.transposed())
            let y = Tensor<Float>(1.0)
            let loss = meanAbsoluteError(predicted: ŷ, expected: y)
            return loss
        }
        // Custom gradient passed to check optimizer validity
        grad.w = Tensor<Float>([[0.1], [0.2]])
        for _ in 0..<1000 {
            optimizer.update(&model, along: grad)
        }
        let expected_gradient = Tensor<Float>([[0.5543607], [1.55286]])
        XCTAssertEqual(model.w, expected_gradient)
    }

    static var allTests = [
        ("testRAdam", testRAdam)
    ]
}