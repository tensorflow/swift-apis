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
    private let tolerance = Float(3e-2)

    func testDistribution(
      _ t: Tensor<Float>,
      expectedMean: Float? = nil,
      expectedStandardDeviation: Float? = nil,
      expectedMin: Float? = nil,
      expectedMax: Float? = nil
    ) {
        if let expectedMean = expectedMean {
            XCTAssertTrue(abs(t.mean() - expectedMean) < tolerance)
        }
        if let expectedStandardDeviation = expectedStandardDeviation {
            XCTAssertTrue(abs(t.standardDeviation() - expectedStandardDeviation) < tolerance)
        }
        if let expectedMin = expectedMin {
            XCTAssertTrue(abs(t.min() - expectedMin) < tolerance)
        }
        if let expectedMax = expectedMax {
            XCTAssertTrue(abs(t.max() - expectedMax) < tolerance)
        }
    }

    func testRandomUniform() {
        var t = Tensor<Float>(randomUniform: fcShape, lowerBound: 2, upperBound: 3)
        testDistribution(t, expectedMean: 2.5, expectedMin: 2, expectedMax: 3)
        t = Tensor<Float>(randomUniform: fcShape, lowerBound: -1, upperBound: 1)
        testDistribution(t, expectedMean: 0, expectedMin: -1, expectedMax: 1)
    }

    func testRandomNormal() {
        let t = Tensor<Float>(randomNormal: convShape, mean: 1, standardDeviation: 2)
        testDistribution(t, expectedMean: 1, expectedStandardDeviation: 2)
    }

    func testTruncatedRandomNormal() {
        let t = Tensor<Float>(truncatedRandomNormal: convShape)
        testDistribution(t, expectedMean: 0, expectedMin: -2, expectedMax: 2)        
    }

    func testGlorotUniform() {
        let t = Tensor<Float>(glorotUniform: convShape)
        let spatialSize = convShape[0..<2].contiguousSize
        let (fanIn, fanOut) = (convShape[2] * spatialSize, convShape[3] * spatialSize)
        let stdDev = sqrt(Float(2.0) / Float(fanIn + fanOut))
        testDistribution(t, expectedMean: 0, expectedStandardDeviation: stdDev)
    }

    func testGlorotNormal() {
        let t = Tensor<Float>(glorotNormal: convShape)
        let spatialSize = convShape[0..<2].contiguousSize
        let (fanIn, fanOut) = (convShape[2] * spatialSize, convShape[3] * spatialSize)
        let stdDev = sqrt(Float(2.0) / Float(fanIn + fanOut))
        testDistribution(t, expectedMean: 0, expectedStandardDeviation: stdDev)
    }

    func testOrthogonalShapesValues() {
        for shape in [[10, 10], [10, 9, 8], [100, 5, 5], [50, 40], [3, 3, 32, 64]] {
            // Check the shape.
            var t = Tensor<Float>(orthogonal: TensorShape(shape))
            XCTAssertEqual(shape, t.shape.dimensions)
        
            // Check orthogonality by computing the inner product.
            t = t.reshaped(to: [t.shape.dimensions.dropLast().reduce(1, *), t.shape[t.rank - 1]])
            if t.shape[0] > t.shape[1] {
                let eye = Raw.diag(diagonal: Tensor<Float>(ones: [t.shape[1]]))
                assertEqual(eye, matmul(t.transposed(), t), accuracy: 1e-5)
            } else {
                let eye = Raw.diag(diagonal: Tensor<Float>(ones: [t.shape[0]]))
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
        ("testTruncatedRandomNormal", testTruncatedRandomNormal),
        ("testGlorotUniform", testGlorotUniform),
        ("testGlorotNormal", testGlorotNormal),
        ("testOrthogonalShapesValues", testOrthogonalShapesValues)
    ]
}
