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

final class ShapedLayerExampleTests: XCTestCase {
  func testSimple() {
    let input = Tensor<Float>(zeros: [12, 28, 28, 1])  // MNIST-size; batch size 12.
    let model = SimpleModel(forExample: input)
    let output = model(input)  // Should run without shape errors.
    XCTAssertEqual([12, 10], output.shape)
  }

  static var allTests = [
    ("testSimple", testSimple)
  ]
}

extension ShapedLayerExampleTests {
  struct SimpleModel: Layer {
    var conv: Conv2D<Float>
    var flatten: Flatten<Float>
    var dense: Dense<Float>

    init(forExample sampleInput: Tensor<Float>) {
      var se = sampleInput  // se == shapeExample
      conv = Conv2D(hparams: .init(3, channels: 5), &se)
      flatten = Flatten(hparams: (), &se)
      dense = Dense(hparams: .init(outputSize: 10), &se)
    }

    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
      input.sequenced(through: conv, flatten, dense)
    }
  }
}
