// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow
import XCTest
import x10_optimizers_tensor_visitor_plan

struct Classifier: Layer {
  var layers = [Dense<Float>(inputSize: 784, outputSize: 30, activation: relu)]
  var final_layer = Dense<Float>(inputSize: 30, outputSize: 10)

  @differentiable
  func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return final_layer(layers.differentiableReduce(input) { last, layer in layer(last) })
  }
}

final class TensorVisitorPlanTests: XCTestCase {
  func testBasic() {
    let model = Classifier()
    let plan = TensorVisitorPlan(model.differentiableVectorView)

    var steps = model.differentiableVectorView
    let weights = model.differentiableVectorView

    let weightsList = plan.allTensors(weights)
    let allKps = plan.allTensorKeyPaths

    var numTensors = 0
    plan.mapTensors(&steps, weights) { (step: inout Tensor<Float>, weight: Tensor<Float>, i: Int) in
      XCTAssertTrue(step == weight)
      XCTAssertTrue(weightsList[i] == weight)
      XCTAssertTrue(weights[keyPath: allKps[i]] == weight)
      numTensors += 1
    }
    XCTAssertTrue(numTensors == allKps.count)
    XCTAssertTrue(allKps == weights.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self))
  }

  func testMask() {
    let model = Classifier()
    let plan = TensorVisitorPlan(model.differentiableVectorView)

    XCTAssertEqual(
      [true, false, true, false], plan.keysEnding(with: \Dense<Float>.TangentVector.weight))
  }

  func testAllKeysBetween() {
    var model = Classifier()
    model.layers = [
      Dense<Float>(inputSize: 784, outputSize: 700, activation: relu),
      Dense<Float>(inputSize: 700, outputSize: 600, activation: relu),
      Dense<Float>(inputSize: 600, outputSize: 500, activation: relu),
      Dense<Float>(inputSize: 500, outputSize: 30, activation: relu),
    ]
    let plan = TensorVisitorPlan(model.differentiableVectorView)

    func firstIndex<T>(_ prefix: WritableKeyPath<Classifier.TangentVector, T>) -> Int {
      let mask = plan.allKeysBetween(lower: \Classifier.TangentVector.self, upper: prefix)
      for (i, maskValue) in mask.enumerated() {
        if !maskValue { return i }
      }
      return mask.count
    }
    XCTAssertEqual(0, firstIndex(\Classifier.TangentVector.layers))
    XCTAssertEqual(1, firstIndex(\Classifier.TangentVector.layers[0].bias))
    XCTAssertEqual(5, firstIndex(\Classifier.TangentVector.layers[2].bias))
    XCTAssertEqual(4, firstIndex(\Classifier.TangentVector.layers[2].weight))
    XCTAssertEqual(8, firstIndex(\Classifier.TangentVector.layers[4]))
    XCTAssertEqual(8, firstIndex(\Classifier.TangentVector.final_layer))

    XCTAssertEqual(
      [false, true, true, true, true, true, true, true, false, false],
      plan.allKeysBetween(
        lower: \Classifier.TangentVector.layers[0].bias,
        upper: \Classifier.TangentVector.layers[4]))
  }
}

extension TensorVisitorPlanTests {
  static var allTests = [
    ("testBasic", testBasic),
    ("testMask", testMask),
    ("testAllKeysBetween", testAllKeysBetween),
  ]
}

XCTMain([
  testCase(TensorVisitorPlanTests.allTests)
])
