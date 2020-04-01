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

struct SimpleOutput: TensorGroup {
  let a: TensorHandle<Int32>
  let b: TensorHandle<Int32>
}

@available(*, deprecated)
final class DatasetTests: XCTestCase {
  func testMultiValue() {
    let elements1: Tensor<Int32> = [0, 1, 2]
    let elements2: Tensor<Int32> = [10, 11, 12]
    let outputTypes = [Int32.tensorFlowDataType, Int32.tensorFlowDataType]
    let outputShapes: [TensorShape?] = [nil, nil]
    let dataset: VariantHandle = _Raw.tensorSliceDataset(
      components: [elements1, elements2],
      outputShapes: outputShapes
    )
    let iterator: ResourceHandle = _Raw.iteratorV2(
      sharedName: "blah",
      container: "earth", outputTypes: outputTypes, outputShapes: outputShapes
    )
    _Raw.makeIterator(dataset: dataset, iterator: iterator)
    var next: SimpleOutput = _Raw.iteratorGetNext(
      iterator: iterator, outputShapes: outputShapes
    )
    XCTAssertEqual(Tensor(handle: next.a).scalarized(), 0)
    XCTAssertEqual(Tensor(handle: next.b).scalarized(), 10)
    next = _Raw.iteratorGetNext(
      iterator: iterator, outputShapes: outputShapes
    )
    XCTAssertEqual(Tensor(handle: next.a).scalarized(), 1)
    XCTAssertEqual(Tensor(handle: next.b).scalarized(), 11)
    next = _Raw.iteratorGetNext(
      iterator: iterator, outputShapes: outputShapes
    )
    XCTAssertEqual(Tensor(handle: next.a).scalarized(), 2)
    XCTAssertEqual(Tensor(handle: next.b).scalarized(), 12)
  }

  func testSingleValueManualIterator() {
    // [[1], [2], [3], [4], [5]]
    let scalars = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
      .reshaped(to: [5, 1])
    let dataset = Dataset(elements: scalars)
    var iterator = dataset.makeIterator()
    var i: Int = 0
    while let item = iterator.next() {
      XCTAssertEqual(item.array, scalars[i].array)
      i += 1
    }
  }

  func testDatasetIteration() {
    // [[1], [2], [3], [4], [5]]
    let scalars = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
      .reshaped(to: [5, 1])
    let dataset = Dataset(elements: scalars)
    var i: Int = 0
    for item in dataset {
      XCTAssertEqual(item.array, scalars[i].array)
      i += 1
    }
  }

  func testSingleValueTransformations() {
    let scalars = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
    let dataset = Dataset(elements: scalars)
    let shuffled = dataset.shuffled(sampleCount: 5, randomSeed: 42)
    XCTAssertEqual(shuffled.map { $0.scalar! }, [0, 4, 1, 3, 2])
  }

  func testSingleValueHOFs() {
    let scalars = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
    let dataset = Dataset(elements: scalars)
    let addedOne: Dataset = dataset.map { $0 + 1 }
    XCTAssertEqual([1, 2, 3, 4, 5], addedOne.flatMap { $0.scalars })
    // Use '.==' in the following closure to avoid any conversions to
    // host data types, which is not handled correctly in tracing.
    let evens: Dataset = dataset.filter { Tensor($0 % 2) .== Tensor(0) }
    XCTAssertEqual(evens.flatMap { $0.scalars }, [0, 2, 4])
  }

  func testParallelMap() {
    let scalars = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
    let dataset = Dataset(elements: scalars)
    let addedOne: Dataset = dataset.map(parallelCallCount: 5) { $0 + 1 }
    XCTAssertEqual(addedOne.flatMap { $0.scalars }, [1, 2, 3, 4, 5])
    // Use '.==' in the following closure to avoid any conversions to
    // host data types, which is not handled correctly in tracing.
    let evens: Dataset = dataset.filter { Tensor($0 % 2) .== Tensor(0) }
    XCTAssertEqual(evens.flatMap { $0.scalars }, [0, 2, 4])
  }

  func testMapToDifferentType() {
    let scalars = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
    let dataset = Dataset(elements: scalars)
    let shuffled = dataset.shuffled(sampleCount: 5, randomSeed: 42)
    XCTAssertEqual([0, 4, 1, 3, 2], shuffled.map { $0.scalar! })
    let evens = shuffled.map { Tensor($0 % 2) .== Tensor(0) }
    XCTAssertEqual(evens.map { $0.scalar! }, [true, true, false, false, true])
  }

  func testSingleValueBatched() {
    let scalars = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
    let dataset = Dataset(elements: scalars)
    let batched = dataset.batched(2)

    var iterator = batched.makeIterator()
    XCTAssertEqual(iterator.next()!.scalars, [0, 1])
    XCTAssertEqual(iterator.next()!.scalars, [2, 3])
    XCTAssertEqual(iterator.next()!.scalars, [4])
  }

  func testDoubleValueDatasetIteration() {
    let scalars1 = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
    let scalars2 = Tensor<Int32>(rangeFrom: 5, to: 10, stride: 1)
    let datasetLeft = Dataset(elements: scalars1)
    let datasetRight = Dataset(elements: scalars2)
    var i: Int = 0
    for pair in zip(datasetLeft, datasetRight) {
      XCTAssertEqual(pair.first.array, scalars1[i].array)
      XCTAssertEqual(pair.second.array, scalars2[i].array)
      i += 1
    }
  }

  static var allTests = [
    ("testMultiValue", testMultiValue),
    ("testSingleValueManualIterator", testSingleValueManualIterator),
    ("testDatasetIteration", testDatasetIteration),
    ("testSingleValueTransformations", testSingleValueTransformations),
    ("testSingleValueHOFs", testSingleValueHOFs),
    ("testParallelMap", testParallelMap),
    ("testMapToDifferentType", testMapToDifferentType),
    ("testSingleValueBatched", testSingleValueBatched),
    ("testDoubleValueDatasetIteration", testDoubleValueDatasetIteration),
  ]
}
