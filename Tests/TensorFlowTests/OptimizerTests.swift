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
  /// A dense layer for testing optimizer convergence.
  // TODO: Consider replacing users with `Dense`.
  struct Model: Layer {
    var dense = Dense<Float>(weight: [[0.8]], bias: [0.8], activation: identity)

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
    let optimizer = RAdam(for: model)
    testConvergence(optimizer: optimizer, model: model, stepCount: 1400)
  }

  /// A `Tensor<Float>` wrapper for testing optimizer numerical correctness.
  /// - Note: `KeyPathIterable` conformance is needed for `SGD`.
  struct NumericalValues: Differentiable & KeyPathIterable {
    var value = Tensor<Float>([0, 0, 0])
  }

  /// Check expected weight and bias after updating `model` with `optimizer` `stepCount` times.
  ///
  /// - Note: optimizer correctness reference implementations exist at
  ///   `Utilities/ReferenceImplementations/optimizers.py`.
  func testNumericalCorrectness<Opt: Optimizer>(
    optimizer: Opt,
    startingValues: NumericalValues,
    expectedValues: Tensor<Float>,
    stepCount: Int = 1000,
    file: StaticString = #file,
    line: UInt = #line
  ) where Opt.Model == NumericalValues {
    var optimizer = optimizer
    var values = startingValues
    let gradient = NumericalValues.TangentVector(value: [-5, 0.1, 0.2])
    for _ in 0..<stepCount {
      optimizer.update(&values, along: gradient)
    }
    XCTAssertEqual(values.value, expectedValues, file: file, line: line)
  }

  func testSGDNumerical() {
    let values = NumericalValues()
    let optimizer = SGD(for: values, learningRate: 1e-3)
    testNumericalCorrectness(
      optimizer: optimizer, startingValues: values,
      expectedValues: [5.0000668, -0.10000112, -0.20000224])
  }

  func testRMSPropNumerical() {
    let values = NumericalValues()
    let optimizer = RMSProp(for: values, learningRate: 1e-3, epsilon: 1e-7)
    testNumericalCorrectness(
      optimizer: optimizer, startingValues: values,
      expectedValues: [1.0091327, -1.0091326, -1.0091326])
  }

  func testAdamNumerical() {
    let values = NumericalValues()
    let optimizer = Adam(for: values, learningRate: 1e-3, epsilon: 1e-7)
    // FIXME(TF-759): Investigate small differences with Python reference implementation results:
    // `[ 0.9999907, -0.9999898, -0.9999904]`.
    testNumericalCorrectness(
      optimizer: optimizer, startingValues: values,
      expectedValues: [0.99999064, -0.9999898, -0.9999905])
  }

  func testAdaDeltaNumerical() {
    let values = NumericalValues()
    let optimizer = AdaDelta(for: values, learningRate: 1e-3, epsilon: 1e-7)
    // FIXME(TF-759): Investigate small differences with Python reference implementation results:
    // `[ 0.0021518278, -0.0021515056, -0.0021517489]`.
    testNumericalCorrectness(
      optimizer: optimizer, startingValues: values,
      expectedValues: [0.0021518273, -0.002151505, -0.0021517489])
  }

  func testAMSGradNumerical() {
    let values = NumericalValues()
    let optimizer = AMSGrad(for: values, learningRate: 1e-3, epsilon: 1e-7)
    // FIXME(TF-759): Investigate small differences with Python reference implementation results:
    // `[ 0.9999907, -0.9999898, -0.9999904]`.
    testNumericalCorrectness(
      optimizer: optimizer, startingValues: values,
      expectedValues: [0.99999064, -0.9999898, -0.9999905])
  }

  func testAdaMaxNumerical() {
    let values = NumericalValues()
    let optimizer = AdaMax(for: values, learningRate: 1e-3, epsilon: 1e-7)
    // FIXME(TF-759): Investigate small differences with Python reference implementation results:
    // `[ 0.99999076, -0.99999064, -0.99999064]`.
    testNumericalCorrectness(
      optimizer: optimizer, startingValues: values,
      expectedValues: [0.9999907, -0.99999064, -0.99999064])
  }

  func testAdaGradNumerical() {
    let values = NumericalValues()
    let optimizer = AdaGrad(for: values, learningRate: 1e-3, epsilon: 1e-7)
    // FIXME(TF-759): Investigate small differences with Python reference implementation results:
    // `[ 0.061795924, -0.057095252, -0.059872225]`.
    testNumericalCorrectness(
      optimizer: optimizer, startingValues: values,
      expectedValues: [0.06179592, -0.057095252, -0.059872225])
  }

  func testRAdamNumerical() {
    let values = NumericalValues()
    let optimizer = RAdam(for: values, learningRate: 1e-3, epsilon: 1e-7)
    // FIXME(TF-759): Investigate small differences with Python reference implementation results:
    // `[ 0.46914074, -0.44463935, -0.44513944]`.
    testNumericalCorrectness(
      optimizer: optimizer, startingValues: values,
      expectedValues: [0.46914074, -0.44463903, -0.44513932])
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
