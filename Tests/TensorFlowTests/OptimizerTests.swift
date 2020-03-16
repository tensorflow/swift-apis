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

import TensorFlow
import XCTest

class OptimizerTests: XCTestCase {
  struct Model: Layer {
    var dense1 = Dense<Float>(weight: [[0.8]], bias: [0.8], activation: identity)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
      dense1(input)
    }
  }

  func convergenceTest<Opt: Optimizer>(
    optimizer: Opt,
    model: Model,
    file: StaticString = #file,
    line: UInt = #line
  ) where Opt.Model == Model {
    var optimizer = optimizer
    var model = model
    let x: Tensor<Float> = Tensor(rangeFrom: -1, to: 1, stride: 0.01)
      .reshaped(to: [-1, 1])
    let y: Tensor<Float> = x + 1

    for _ in 0..<1000 {
      let grad = gradient(at: model) { model -> Tensor<Float> in
        let yy = model(x)
        return meanSquaredError(predicted: yy, expected: y)
      }
      optimizer.update(&model, along: grad)

      if model(x).isAlmostEqual(to: y) {
        break
      }
    }

    XCTAssertTrue(model(x).isAlmostEqual(to: y), file: file, line: line)
  }

  func testSGD() {
    let model = Model()
    let optimizer = SGD(for: model)
    convergenceTest(optimizer: optimizer, model: model)
  }

  func testRMSProp() {
    let model = Model()
    let optimizer = RMSProp(for: model)
    convergenceTest(optimizer: optimizer, model: model)
  }

  func testAdaGrad() {
    let model = Model()
    let optimizer = AdaGrad(for: model, learningRate: 0.01)
    convergenceTest(optimizer: optimizer, model: model)
  }

  func testAdaDelta() {
    let model = Model()
    let optimizer = AdaDelta(for: model)
    convergenceTest(optimizer: optimizer, model: model)
  }

  func testAdam() {
    let model = Model()
    let optimizer = Adam(for: model)
    convergenceTest(optimizer: optimizer, model: model)
  }

  func testAdaMax() {
    let model = Model()
    let optimizer = AdaMax(for: model)
    convergenceTest(optimizer: optimizer, model: model)
  }

  func testAMSGrad() {
    let model = Model()
    let optimizer = AMSGrad(for: model)
    convergenceTest(optimizer: optimizer, model: model)
  }

  func testRAdam() {
    let model = Model()
    let optimizer = RAdam(for: model)
    convergenceTest(optimizer: optimizer, model: model)
  }

  struct ModelNumerical: Differentiable, KeyPathIterable {
    var tensor = Tensor<Float>([0, 1, 2])
    static let grad = ModelNumerical.TangentVector(tensor: [0.0, 0.1, 0.2])
  }

  func testAdaMaxNumerical() {
    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // var = tf.Variable([0, 1, 2], dtype=tf.float32)
    // grad = tf.Variable([0, 0.1, 0.2], dtype=tf.dtypes.float32)
    // optimizer = tf.keras.optimizers.Adamax()
    // optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // for i in range(10):
    //     optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // ```
    var model = ModelNumerical()
    let opt = AdaMax(for: model, learningRate: 1e-3, epsilon: 1e-7)
    opt.update(&model, along: ModelNumerical.grad)
    XCTAssertEqual(model.tensor, [0, 0.999, 1.999])
    for _ in 0..<10 {
      opt.update(&model, along: ModelNumerical.grad)
    }
    XCTAssertEqual(model.tensor, [0, 0.98900014, 1.9889995])
  }

  static var allTests = [
    ("testSGD", testSGD),
    ("testRMSProp", testRMSProp),
    ("testAdaGrad", testAdaGrad),
    ("testAdaDelta", testAdaDelta),
    ("testAdam", testAdam),
    ("testAdaMax", testAdaMax),
    ("testAMSGrad", testAMSGrad),
    ("testRAdam", testRAdam),
    ("testAdaMaxNumerical", testAdaMaxNumerical),
  ]
}
