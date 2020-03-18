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

  static var allTests = [
    ("testSGD", testSGD),
    ("testRMSProp", testRMSProp),
    ("testAdaGrad", testAdaGrad),
    ("testAdaDelta", testAdaDelta),
    ("testAdam", testAdam),
    ("testAdaMax", testAdaMax),
    ("testAMSGrad", testAMSGrad),
    ("testRAdam", testRAdam),
  ]
}
