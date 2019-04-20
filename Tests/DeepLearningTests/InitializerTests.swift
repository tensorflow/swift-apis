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
@testable import DeepLearning

final class InitializerTests: XCTestCase {
    func testInitializers() {
        let scalar = Tensor<Float>(1)
        let matrix: Tensor<Float> = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let broadcastScalar = Tensor<Float>(broadcasting: 10, rank: 3)
        let some4d = Tensor<Float>(
            shape: [2, 1, 2, 1],
            scalars: AnyRandomAccessCollection([2, 3, 4, 5]))
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

    func testNonTPUDataTypeCast() {
        // TPU does not support Int8 or 16 casting.
        guard !_RuntimeConfig.executionMode.isTPU else { return }

        let x = Tensor<Int32>(ones: [5, 5])
        let ints = Tensor<Int64>(x)
        let floats = Tensor<Float>(x)
        let i8s = Tensor<Int8>(floats)
        XCTAssertEqual(ShapedArray(repeating: 1, shape: [5, 5]), ints.array)
        XCTAssertEqual(ShapedArray(repeating: 1, shape: [5, 5]), floats.array)
        XCTAssertEqual(ShapedArray(repeating: 1, shape: [5, 5]), i8s.array)
    }

    func testTPUDataTypeCast() {
        // Non-TPU mode (e.g. eager) does not support Uint32 casting.
        guard _RuntimeConfig.executionMode.isTPU else { return }

        let x = Tensor<Int32>(ones: [5, 5])
        let ints = Tensor<Int64>(x)
        let floats = Tensor<Float>(x)
        let u32s = Tensor<UInt32>(floats)
        XCTAssertEqual(ShapedArray(repeating: 1, shape: [5, 5]), ints.array)
        XCTAssertEqual(ShapedArray(repeating: 1, shape: [5, 5]), floats.array)
        XCTAssertEqual(ShapedArray(repeating: 1, shape: [5, 5]), u32s.array)
    }

    func testNonTPUBoolToNumericCast() {
        // TPU does not support Int8 or 16 casting.
        //
        // When changing to UInt32, got another TPU/XLA compilation error when
        // converting from bools to Uint32 (different from missing kernel error).
        if _RuntimeConfig.executionMode.isTPU { return }

        let bools = Tensor<Bool>(shape: [2, 2], scalars: [true, false, true, false])
        let ints = Tensor<Int64>(bools)
        let floats = Tensor<Float>(bools)
        let i8s = Tensor<Int8>(bools)
        XCTAssertEqual(ShapedArray(shape: [2, 2], scalars: [1, 0, 1, 0]), ints.array)
        XCTAssertEqual(ShapedArray(shape: [2, 2], scalars: [1, 0, 1, 0]), floats.array)
        XCTAssertEqual(ShapedArray(shape: [2, 2], scalars: [1, 0, 1, 0]), i8s.array)
    }
}
