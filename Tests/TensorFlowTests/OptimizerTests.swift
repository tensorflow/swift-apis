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
    var dense: Dense<Float>

    init() {
      self.init(weight: [[0.8]], bias: [0.8])
    }

    init(weight: Tensor<Float>, bias: Tensor<Float>) {
      dense = Dense<Float>(weight: weight, bias: bias, activation: identity)
    }

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
      dense(input)
    }
  }

  /// Check expected weight and bias after updating `model` with `optimizer` `stepCount` times.
  ///
  /// - Note: optimizer correctness reference implementations exist at
  ///   `Utilities/ReferenceImplementations/optimizers.py`.
  func testCorrectness<Opt: Optimizer>(
    optimizer: Opt,
    model: Model,
    expectedWeight: Tensor<Float>,
    expectedBias: Tensor<Float>,
    stepCount: Int = 1000,
    file: StaticString = #file,
    line: UInt = #line
  ) where Opt.Model == Model {
    var optimizer = optimizer
    var model = model
    let grad = Model.TangentVector(dense: .init(weight: [[0.1]], bias: [0.2]))
    for _ in 0..<stepCount {
      optimizer.update(&model, along: grad)
    }
    XCTAssertEqual(model.dense.weight, expectedWeight, file: file, line: line)
    XCTAssertEqual(model.dense.bias, expectedBias, file: file, line: line)
  }

  /// Check that `model` converges after updating it with `optimizer` `stepCount` times.
  func testConvergence<Opt: Optimizer>(
    optimizer: Opt,
    model: Model,
    stepCount: Int = 1000,
    file: StaticString = #file,
    line: UInt = #line
  ) where Opt.Model == Model {
    var optimizer = optimizer
    var model = model
    let x: Tensor<Float> = Tensor(rangeFrom: -1, to: 1, stride: 0.01)
      .reshaped(to: [-1, 1])
    let y: Tensor<Float> = x + 1

    for _ in 0..<stepCount {
      let grad = gradient(at: model) { model -> Tensor<Float> in
        let yy = model(x)
        return meanSquaredError(predicted: yy, expected: y)
      }
      optimizer.update(&model, along: grad)

      // Break if model has converged.
      if model(x).isAlmostEqual(to: y) {
        break
      }
    }

    // Check that model has converged.
    XCTAssertTrue(model(x).isAlmostEqual(to: y), file: file, line: line)
  }

  func testSGD() {
    let model = Model()
    let optimizer = SGD(for: model)
    testConvergence(optimizer: optimizer, model: model)
  }

  func testRMSProp() {
    let model = Model()
    let optimizer = RMSProp(for: model)
    testConvergence(optimizer: optimizer, model: model)
  }

  func testAdaGrad() {
    let model = Model()
    let optimizer = AdaGrad(for: model, learningRate: 0.01)
    testConvergence(optimizer: optimizer, model: model)
  }

  func testAdaDelta() {
    let model = Model()
    let optimizer = AdaDelta(for: model)
    testConvergence(optimizer: optimizer, model: model)
  }

  func testAdam() {
    let model = Model()
    let optimizer = Adam(for: model)
    testConvergence(optimizer: optimizer, model: model)
  }

  func testAdaMax() {
    let model = Model()
    let optimizer = AdaMax(for: model)
    testConvergence(optimizer: optimizer, model: model)
  }

  func testAMSGrad() {
    let model = Model()
    let optimizer = AMSGrad(for: model)
    testConvergence(optimizer: optimizer, model: model)
  }

  func testRAdam() {
    let model = Model()

    // Test convergence.
    do {
      let optimizer = RAdam(for: model)
      testConvergence(optimizer: optimizer, model: model, stepCount: 1400)
    }

    // Test correctness.
    do {
      let optimizer = RAdam(for: model)
      testCorrectness(
        optimizer: optimizer, model: model,
        expectedWeight: [[0.35536084]], expectedBias: [0.3548611])
    }
  }

  struct ModelNumerical: Differentiable, KeyPathIterable {
    var tensor = Tensor<Float>([0, 1, 2])
    static let grad = ModelNumerical.TangentVector(tensor: [0.0, 0.1, 0.2])
  }

  func testSGDNumerical() {
    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // var = tf.Variable([0, 1, 2], dtype=tf.float32)
    // grad = tf.Variable([0, 0.1, 0.2], dtype=tf.dtypes.float32)
    // optimizer = tf.keras.optimizers.SGD()
    // optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // for i in range(10):
    //     optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // ```
    var model = ModelNumerical()
    let opt = SGD(for: model)
    opt.update(&model, along: ModelNumerical.grad)
    XCTAssertEqual(model.tensor, [0, 0.999, 1.998])
    for _ in 0..<10 {
      opt.update(&model, along: ModelNumerical.grad)
    }
    XCTAssertEqual(model.tensor, [0, 0.98900014, 1.9780003])
  }

  func testRMSPropNumerical() {
    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // var = tf.Variable([0, 1, 2], dtype=tf.float32)
    // grad = tf.Variable([0, 0.1, 0.2], dtype=tf.dtypes.float32)
    // optimizer = tf.keras.optimizers.RMSProp()
    // optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // for i in range(10):
    //     optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // ```
    var model = ModelNumerical()
    let opt = RMSProp(for: model, epsilon: 1e-7)
    opt.update(&model, along: ModelNumerical.grad)
    XCTAssertEqual(model.tensor, [0, 0.99683774, 1.9968377])
    for _ in 0..<10 {
      opt.update(&model, along: ModelNumerical.grad)
    }
    XCTAssertEqual(model.tensor, [0, 0.9814604, 1.9814601])
  }

  func testAdamNumerical() {
    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // var = tf.Variable([0, 1, 2], dtype=tf.float32)
    // grad = tf.Variable([0, 0.1, 0.2], dtype=tf.dtypes.float32)
    // optimizer = tf.keras.optimizers.Adam()
    // optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // for i in range(10):
    //     optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // ```
    var model = ModelNumerical()
    let opt = Adam(for: model, epsilon: 1e-7)
    opt.update(&model, along: ModelNumerical.grad)
    XCTAssertEqual(model.tensor, [0, 0.999, 1.9990001])
    for _ in 0..<10 {
      opt.update(&model, along: ModelNumerical.grad)
    }
    XCTAssertEqual(model.tensor, [0, 0.98900014, 1.9889997])
  }

  func testAdaDeltaNumerical() {
    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // var = tf.Variable([0, 1, 2], dtype=tf.float32)
    // grad = tf.Variable([0, 0.1, 0.2], dtype=tf.dtypes.float32)
    // optimizer = tf.keras.optimizers.Adadelta()
    // optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // for i in range(10):
    //     optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // ```
    var model = ModelNumerical()
    let opt = AdaDelta(for: model, learningRate: 1e-3, epsilon: 1e-7)
    opt.update(&model, along: ModelNumerical.grad)
    XCTAssertEqual(model.tensor, [0, 0.99999857, 1.9999986])
    for _ in 0..<10 {
      opt.update(&model, along: ModelNumerical.grad)
    }
    XCTAssertEqual(model.tensor, [0, 0.99998385, 1.9999841])
  }

  func testAMSGradNumerical() {
    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // var = tf.Variable([0, 1, 2], dtype=tf.float32)
    // grad = tf.Variable([0, 0.1, 0.2], dtype=tf.dtypes.float32)
    // optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    // optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // for i in range(10):
    //     optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // ```
    var model = ModelNumerical()
    let opt = AMSGrad(for: model, epsilon: 1e-7)
    opt.update(&model, along: ModelNumerical.grad)
    XCTAssertEqual(model.tensor, [0, 0.999, 1.9990001])
    for _ in 0..<10 {
      opt.update(&model, along: ModelNumerical.grad)
    }
    XCTAssertEqual(model.tensor, [0, 0.98900014, 1.9889997])
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

  func testAdaGradNumerical() {
    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // var = tf.Variable([0, 1, 2], dtype=tf.float32)
    // grad = tf.Variable([0, 0.1, 0.2], dtype=tf.dtypes.float32)
    // optimizer = tf.keras.optimizers.Adagrad()
    // optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // for i in range(10):
    //     optimizer.apply_gradients(list(zip([grad], [var])))
    // print(var.read_value())
    // ```
    var model = ModelNumerical()
    let opt = AdaGrad(for: model, learningRate: 1e-3, epsilon: 1e-7)
    opt.update(&model, along: ModelNumerical.grad)
    XCTAssertEqual(model.tensor, [0, 0.99969846, 1.9994655])
    for _ in 0..<10 {
      opt.update(&model, along: ModelNumerical.grad)
    }
    XCTAssertEqual(model.tensor, [0, 0.9972076, 1.9959843])
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
    ("testSGDNumerical", testSGDNumerical),
    ("testRMSPropNumerical", testRMSPropNumerical),
    ("testAdamNumerical", testAdamNumerical),
    ("testAdaDeltaNumerical", testAdaDeltaNumerical),
    ("testAMSGradNumerical", testAMSGradNumerical),
    ("testAdaMaxNumerical", testAdaMaxNumerical),
    ("testAdaGradNumerical", testAdaGradNumerical),
  ]
}
