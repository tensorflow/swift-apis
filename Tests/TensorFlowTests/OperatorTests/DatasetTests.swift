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

    public init<C: RandomAccessCollection>(
        _handles: C) where C.Element == _AnyTensorHandle {
        precondition(_handles.count == 2)
        let aIndex = _handles.startIndex
        let bIndex = _handles.index(aIndex, offsetBy: 1)
        a = TensorHandle<Int32>(handle: _handles[aIndex])
        b = TensorHandle<Int32>(handle: _handles[bIndex])
    }

    public var _tensorHandles: [_AnyTensorHandle] { [a.handle, b.handle] }
}

final class DatasetTests: XCTestCase {
    func testMultiValue() {
        let elements1: Tensor<Int32> = [0, 1, 2]
        let elements2: Tensor<Int32> = [10, 11, 12]
        let outputTypes = [Int32.tensorFlowDataType, Int32.tensorFlowDataType]
        let outputShapes: [TensorShape?] = [nil, nil]
        let dataset: VariantHandle = Raw.tensorSliceDataset(
            components: [elements1, elements2],
            outputShapes: outputShapes
        )
        let iterator: ResourceHandle = Raw.iteratorV2(sharedName: "blah",
            container: "earth", outputTypes: outputTypes, outputShapes: outputShapes
        )
        Raw.makeIterator(dataset: dataset, iterator: iterator)
        var next: SimpleOutput = Raw.iteratorGetNext(
            iterator: iterator, outputShapes: outputShapes
        )
        XCTAssertEqual(0, Tensor(handle: next.a).scalarized())
        XCTAssertEqual(10, Tensor(handle: next.b).scalarized())
        next = Raw.iteratorGetNext(
            iterator: iterator, outputShapes: outputShapes
        )
        XCTAssertEqual(1, Tensor(handle: next.a).scalarized())
        XCTAssertEqual(11, Tensor(handle: next.b).scalarized())
        next = Raw.iteratorGetNext(
            iterator: iterator, outputShapes: outputShapes
        )
        XCTAssertEqual(2, Tensor(handle: next.a).scalarized())
        XCTAssertEqual(12, Tensor(handle: next.b).scalarized())
    }

    func testSingleValueManualIterator() {
      // [[1], [2], [3], [4], [5]]
      let scalars = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
          .reshaped(to: [5, 1])
      let dataset = Dataset(elements: scalars)
      var iterator = dataset.makeIterator()
      var i: Int = 0
      while let item = iterator.next() {
          XCTAssertEqual(scalars[i].array, item.array)
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
            XCTAssertEqual(scalars[i].array, item.array)
            i += 1
        }
    }

    func testSingleValueTransformations() {
        let scalars = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
        let dataset = Dataset(elements: scalars)
        let shuffled = dataset.shuffled(sampleCount: 5, randomSeed: 42)
        XCTAssertEqual([0, 4, 1, 3, 2], shuffled.map { $0.scalar! })
    }

    func testSingleValueHOFs() {
        let scalars = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
        let dataset = Dataset(elements: scalars)
        let addedOne: Dataset = dataset.map { $0 + 1 }
        XCTAssertEqual([1, 2, 3, 4, 5], addedOne.flatMap { $0.scalars })
        // Use '.==' in the following closure to avoid any conversions to
        // host data types, which is not handled correctly in tracing.
        let evens: Dataset = dataset.filter { Tensor($0 % 2) .== Tensor(0) }
        XCTAssertEqual([0, 2, 4], evens.flatMap { $0.scalars })
    }

    func testParallelMap() {
        let scalars = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
        let dataset = Dataset(elements: scalars)
        let addedOne: Dataset = dataset.map(parallelCallCount: 5) { $0 + 1 }
        XCTAssertEqual([1, 2, 3, 4, 5], addedOne.flatMap { $0.scalars })
        // Use '.==' in the following closure to avoid any conversions to
        // host data types, which is not handled correctly in tracing.
        let evens: Dataset = dataset.filter { Tensor($0 % 2) .== Tensor(0) }
        XCTAssertEqual([0, 2, 4], evens.flatMap { $0.scalars })
    }

    func testMapToDifferentType() {
        let scalars = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
        let dataset = Dataset(elements: scalars)
        let shuffled = dataset.shuffled(sampleCount: 5, randomSeed: 42)
        XCTAssertEqual([0, 4, 1, 3, 2], shuffled.map { $0.scalar! })
        let evens = shuffled.map { Tensor($0 % 2) .== Tensor(0) }
        XCTAssertEqual([true, true, false, false, true], evens.map { $0.scalar! })
    }

    func testSingleValueBatched() {
        let scalars = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
        let dataset = Dataset(elements: scalars)
        let batched = dataset.batched(2)

        var iterator = batched.makeIterator()
        XCTAssertEqual([0, 1], iterator.next()!.scalars)
        XCTAssertEqual([2, 3], iterator.next()!.scalars)
        XCTAssertEqual([4], iterator.next()!.scalars)
    }

/*
    func testDoubleValueDatasetIteration() {
        let scalars1 = Tensor<Float>(rangeFrom: 0, to: 5, stride: 1)
        let scalars2 = Tensor<Int32>(rangeFrom: 5, to: 10, stride: 1)
        let datasetLeft = Dataset(elements: scalars1)
        let datasetRight = Dataset(elements: scalars2)
        var i: Int = 0
        for pair in zip(datasetLeft, datasetRight) {
            XCTAssertEqual(scalars1[i].array, pair.first.array)
            XCTAssertEqual(scalars2[i].array, pair.second.array)
            i += 1
        }
    }
*/

    static var allTests = [
        ("testMultiValue", testMultiValue),
        ("testSingleValueManualIterator", testSingleValueManualIterator),
        ("testDatasetIteration", testDatasetIteration),
        ("testSingleValueTransformations", testSingleValueTransformations),
        ("testSingleValueHOFs", testSingleValueHOFs),
        ("testParallelMap", testParallelMap),
        ("testMapToDifferentType", testMapToDifferentType),
        ("testSingleValueBatched", testSingleValueBatched),
        // Currently broken even in TensorFlow ...
        // This will be easier to fix once everything is moved ...
        // ("testDoubleValueDatasetIteration", testDoubleValueDatasetIteration),
    ]
}
