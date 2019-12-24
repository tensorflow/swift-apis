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

final class InitializerTests: XCTestCase {
    func testInitializers() {
        let scalar = Tensor<Float>(1)
        let matrix: Tensor<Float> = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let broadcastScalar = Tensor<Float>(broadcasting: 10, rank: 3)
        let some4d = Tensor<Float>(shape: [2, 1, 2, 1], scalars: [2, 3, 4, 5])
        XCTAssertEqual(ShapedArray(shape: [2, 1, 2, 1], scalars: [2, 3, 4, 5]), some4d.array)
        XCTAssertEqual(ShapedArray(shape: [], scalars: [1]), scalar.array)
        XCTAssertEqual(ShapedArray(shape: [2, 3], scalars: [1, 2, 3, 4, 5, 6]), matrix.array)
        XCTAssertEqual(ShapedArray(shape: [1, 1, 1], scalars: [10]), broadcastScalar.array)
    }

    func testFactoryInitializers() {
        let x = Tensor<Float>(ones: [1, 10])
        XCTAssertEqual(ShapedArray(repeating: 1, shape: [1, 10]), x.array)
    }

    func testNumericInitializers() {
        let x = Tensor<Float>(oneHotAtIndices: [0, 2, -1, 1], depth: 3)
        XCTAssertEqual(ShapedArray(
            shape: [4, 3],
            scalars: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]), x.array)

        let linearSpaceTensor1 = Tensor<Float>(linearSpaceFrom: 0.1, to: 2.0, count: 10)
        assertEqual(
            [
                0.1, 0.31111111, 0.52222222, 0.73333333, 0.94444444, 1.15555555, 1.36666666,
                1.57777777, 1.78888888, 2.0,
            ], linearSpaceTensor1, accuracy: 0.001)

        let linearSpaceTensor2 = Tensor<Float>(linearSpaceFrom: 2.0, to: 0.1, count: 10)
        assertEqual(
            [
                2, 1.78888889, 1.57777778, 1.36666667, 1.15555556, 0.94444444, 0.73333333,
                0.52222222, 0.31111111, 0.1,
            ], linearSpaceTensor2, accuracy: 0.001)

        let linearSpaceTensor3 = Tensor<Float>(linearSpaceFrom: -5.0, to: 8.5, count: 20)
        assertEqual(
            [
                -5.0, -4.28947368, -3.57894737, -2.86842105, -2.15789474, -1.44736842,
                -0.73684211, -0.02631579, 0.68421053, 1.39473684, 2.10526316, 2.81578947,
                3.52631579, 4.23684211, 4.94736842, 5.65789474, 6.36842105, 7.07894737,
                7.78947368, 8.5,
            ], linearSpaceTensor3, accuracy: 0.001)
    }

    func testScalarToTensorConversion() {
        let tensor = Tensor<Float>(broadcasting: 42, rank: 4)
        XCTAssertEqual([1, 1, 1, 1], tensor.shape)
        XCTAssertEqual([42], tensor.scalars)
    }

    func testArrayConversion() {
        let array3D = ShapedArray(repeating: 1.0, shape: [2, 3, 4])
        let tensor3D = Tensor(array3D)
        XCTAssertEqual(array3D, tensor3D.array)
    }

    func testDataTypeCast() {
        let x = Tensor<Int32>(ones: [5, 5])
        let ints = Tensor<Int64>(x)
        let floats = Tensor<Float>(x)
        let u32s = Tensor<UInt32>(floats)
        XCTAssertEqual(ShapedArray(repeating: 1, shape: [5, 5]), ints.array)
        XCTAssertEqual(ShapedArray(repeating: 1, shape: [5, 5]), floats.array)
        XCTAssertEqual(ShapedArray(repeating: 1, shape: [5, 5]), u32s.array)
    }

    func testBoolToNumericCast() {
        let bools = Tensor<Bool>(shape: [2, 2], scalars: [true, false, true, false])
        let ints = Tensor<Int64>(bools)
        let floats = Tensor<Float>(bools)
        let i8s = Tensor<Int8>(bools)
        XCTAssertEqual(ShapedArray(shape: [2, 2], scalars: [1, 0, 1, 0]), ints.array)
        XCTAssertEqual(ShapedArray(shape: [2, 2], scalars: [1, 0, 1, 0]), floats.array)
        XCTAssertEqual(ShapedArray(shape: [2, 2], scalars: [1, 0, 1, 0]), i8s.array)
    }

    // Constants for testing distribution based initializers.
    private let fcShape = TensorShape([200, 100])
    private let convShape = TensorShape([25, 25, 20, 20])

    func testDistribution(
        _ t: Tensor<Float>,
        expectedMean: Float? = nil,
        expectedStandardDeviation: Float? = nil,
        expectedMin: Float? = nil,
        expectedMax: Float? = nil,
        tolerance: Float = 3e-2
    ) {
        if let expectedMean = expectedMean {
            let mean = t.mean().scalarized()
            XCTAssertTrue(abs(mean - expectedMean) < tolerance)
        }
        if let expectedStandardDeviation = expectedStandardDeviation {
            let standardDeviation = t.standardDeviation().scalarized()
            XCTAssertTrue(abs(standardDeviation - expectedStandardDeviation) < tolerance)
        }
        if let expectedMin = expectedMin {
            let min = t.min().scalarized()
            XCTAssertTrue(abs(min - expectedMin) < tolerance)
        }
        if let expectedMax = expectedMax {
            let max = t.max().scalarized()
            XCTAssertTrue(abs(max - expectedMax) < tolerance)
        }
    }

    func testRandomUniform() {
        do {
            let t = Tensor<Float>(
                randomUniform: fcShape,
                lowerBound: Tensor(2),
                upperBound: Tensor(3))
            testDistribution(t, expectedMean: 2.5, expectedMin: 2, expectedMax: 3)
        }
        do {
            let t = Tensor<Float>(
                randomUniform: fcShape,
                lowerBound: Tensor(-1),
                upperBound: Tensor(1))
            testDistribution(t, expectedMean: 0, expectedMin: -1, expectedMax: 1)
        }
    }

    func testRandomNormal() {
        let t = Tensor<Float>(
            randomNormal: convShape,
            mean: Tensor(1),
            standardDeviation: Tensor(2))
        testDistribution(t, expectedMean: 1, expectedStandardDeviation: 2)
    }

    func testRandomTruncatedNormal() {
        let t = Tensor<Float>(randomTruncatedNormal: convShape)
        testDistribution(t, expectedMean: 0, expectedMin: -2, expectedMax: 2)
    }

    func testGlorotUniform() {
        let t = Tensor<Float>(glorotUniform: convShape)
        let spatialSize = convShape[0..<2].contiguousSize
        let (fanIn, fanOut) = (convShape[2] * spatialSize, convShape[3] * spatialSize)
        let stdDev = sqrt(Float(2.0) / Float(fanIn + fanOut))
        testDistribution(t, expectedMean: 0, expectedStandardDeviation: stdDev, tolerance: 1e-4)
    }

    func testGlorotNormal() {
        let t = Tensor<Float>(glorotNormal: convShape)
        let spatialSize = convShape[0..<2].contiguousSize
        let (fanIn, fanOut) = (convShape[2] * spatialSize, convShape[3] * spatialSize)
        let stdDev = sqrt(Float(2.0) / Float(fanIn + fanOut))
        testDistribution(t, expectedMean: 0, expectedStandardDeviation: stdDev)
    }

    func testCategoricalFromLogits() {
        let probabilities = Tensor<Float>([[0.5, 0.3, 0.2], [0.6, 0.3, 0.1]])
        let logits = log(probabilities)
        let t = Tensor<Int32>(randomCategorialLogits: logits, sampleCount: 1)

        XCTAssertEqual(TensorShape([2, 1]), t.shape)
        // Test all elements are in range of [0, 3)
        XCTAssertTrue((t .>= Tensor<Int32>([[0], [0]])).all())
        XCTAssertTrue((t .< Tensor<Int32>([[3], [3]])).all())
    }

    func testOrthogonalShapesValues() {
        for shape in [[10, 10], [10, 9, 8], [100, 5, 5], [50, 40], [3, 3, 32, 64]] {
            // Check the shape.
            var t = Tensor<Float>(orthogonal: TensorShape(shape))
            XCTAssertEqual(shape, t.shape.dimensions)

            // Check orthogonality by computing the inner product.
            t = t.reshaped(to: [t.shape.dimensions.dropLast().reduce(1, *), t.shape[t.rank - 1]])
            if t.shape[0] > t.shape[1] {
                let eye = _Raw.diag(diagonal: Tensor<Float>(ones: [t.shape[1]]))
                assertEqual(eye, matmul(t.transposed(), t), accuracy: 1e-5)
            } else {
                let eye = _Raw.diag(diagonal: Tensor<Float>(ones: [t.shape[0]]))
                assertEqual(eye, matmul(t, t.transposed()), accuracy: 1e-5)
            }
        }
    }

    static var allTests = [
        ("testInitializers", testInitializers),
        ("testFactoryInitializers", testFactoryInitializers),
        ("testNumericInitializers", testNumericInitializers),
        ("testScalarToTensorConversion", testScalarToTensorConversion),
        ("testArrayConversion", testArrayConversion),
        ("testDataTypeCast", testDataTypeCast),
        ("testBoolToNumericCast", testBoolToNumericCast),
        ("testRandomUniform", testRandomUniform),
        ("testRandomNormal", testRandomNormal),
        ("testRandomTruncatedNormal", testRandomTruncatedNormal),
        ("testGlorotUniform", testGlorotUniform),
        ("testGlorotNormal", testGlorotNormal),
        ("testCategoricalFromLogits", testCategoricalFromLogits),
        ("testOrthogonalShapesValues", testOrthogonalShapesValues)
    ]
}
