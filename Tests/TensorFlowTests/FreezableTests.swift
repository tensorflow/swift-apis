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

final class FreezableTests: XCTestCase {
  func testFreezableParameters() {
    // A dense layer with freezable properties.
    struct FreezableDense: Layer {
      @_Freezable var weight: Tensor<Float>
      @_Freezable var bias: Tensor<Float>

      init(weight: Tensor<Float>, bias: Tensor<Float>) {
        // Require scalar weight and bias for simplicity.
        precondition(weight.isScalar)
        precondition(bias.isScalar)
        self.weight = weight
        self.bias = bias
      }

      @differentiable
      func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input * weight + bias
      }
    }

    var dense = FreezableDense(weight: Tensor(2), bias: Tensor(3))
    let x = Tensor<Float>(4)
    do {
      let (value, gradient) = valueWithGradient(at: dense, x) { dense, x in dense(x) }
      XCTAssertEqual(Tensor(11), value)
      // The gradient of `dense.weight` should be non-zero.
      XCTAssertEqual(
        FreezableDense.TangentVector(_weight: Tensor(4), _bias: Tensor(1)),
        gradient.0)
      XCTAssertEqual(Tensor(2), gradient.1)
    }

    // Freeze derivatives for `dense.weight`.
    dense.$weight.freeze()
    do {
      let (value, gradient) = valueWithGradient(at: dense, x) { dense, x in dense(x) }
      // The gradient of `dense.weight` should now be zero.
      XCTAssertEqual(Tensor(11), value)
      XCTAssertEqual(
        FreezableDense.TangentVector(_weight: Tensor(0), _bias: Tensor(1)),
        gradient.0)
      XCTAssertEqual(Tensor(2), gradient.1)
    }

    // Unfreeze derivatives for `dense.weight`.
    dense.$weight.unfreeze()
    do {
      let (value, gradient) = valueWithGradient(at: dense, x) { dense, x in dense(x) }
      XCTAssertEqual(Tensor(11), value)
      // The gradient of `dense.weight` should now be non-zero.
      XCTAssertEqual(
        FreezableDense.TangentVector(_weight: Tensor(4), _bias: Tensor(1)),
        gradient.0)
      XCTAssertEqual(Tensor(2), gradient.1)
    }
  }

  static var allTests = [
    ("testFreezableParameters", testFreezableParameters)
  ]
}
