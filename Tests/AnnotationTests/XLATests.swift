// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

@testable import Tensor

final class AnnotationXLATests: XCTestCase {
  public struct SummaryNet: Layer {
    public var dense1 = Dense<Float>(inputSize: 1, outputSize: 1)
    public var dense2 = Dense<Float>(inputSize: 4, outputSize: 4)
    public var dense3 = Dense<Float>(inputSize: 4, outputSize: 4)
    public var flatten = Flatten<Float>()

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
      let layer1 = dense1(input)
      let layer2 = layer1.reshaped(to: [1, 4])
      let layer3 = dense2(layer2)
      let layer4 = dense3(layer3)
      return flatten(layer4)
    }
  }

  lazy var device: Device = { Device(kind: .CPU, ordinal: 0, backend: .XLA) }()
  lazy var model0: SummaryNet = { SummaryNet() }()
  lazy var model: SummaryNet = { SummaryNet(copying: model0, to: device) }()
  lazy var inputShape: TensorShape = { TensorShape([1, 4, 1, 1]) }()
  lazy var input: Tensor<Float> = { Tensor<Float>(repeating: 1, shape: inputShape, on: device) }()

  override func setUp() {
    super.setUp()
    LazyTensorBarrier()
  }

  private func validateAnnotations(_ annotations: String) -> Bool {
    let lines = annotations.components(separatedBy: "\n")

    if lines.count < 2 {
      return false
    }

    // Isolate layers.
    var contents: [String] = Array()
    for line in lines {
      if line.contains("shape=") {
        contents.append(line)
      }
    }

    return contents.count == 3
  }

  func testLayerAnnotationsTensor() {
    let annotations = model.annotations(input: input)
    XCTAssert(validateAnnotations(annotations))
  }

  func testLayerSummaryTensor() {
    let annotations = model.summary(input: input)
    XCTAssert(validateAnnotations(annotations))
  }

  func testLayerAnnotationsTensorShape() {
    let annotations = model.annotations(inputShape: inputShape)
    XCTAssert(validateAnnotations(annotations))
  }

  func testLayerSummaryTensorShape() {
    let annotations = model.summary(inputShape: inputShape)
    XCTAssert(validateAnnotations(annotations))
  }

  func testTensorAnnotations() {
    let output = model(input)
    let annotations = output.annotations
    XCTAssert(validateAnnotations(annotations))
  }

  func testTensorAnnotationsSummary() {
    let output = model(input)
    let annotations = output.summary
    XCTAssert(validateAnnotations(annotations))
  }
}

extension AnnotationXLATests {
  static var allTests = [
    ("testLayerAnnotationsTensor", testLayerAnnotationsTensor),
    ("testLayerSummaryTensor", testLayerSummaryTensor),
    ("testLayerAnnotationsTensorShape", testLayerAnnotationsTensorShape),
    ("testLayerSummaryTensorShape", testLayerSummaryTensorShape),
    ("testTensorAnnotations", testTensorAnnotations),
    ("testTensorAnnotationsSummary", testTensorAnnotationsSummary),
  ]
}

XCTMain([
  testCase(AnnotationXLATests.allTests)
])
