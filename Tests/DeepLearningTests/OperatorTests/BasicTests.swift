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

infix operator ++: AdditionPrecedence
infix operator .=

infix operator ..: StridedRangeFormationPrecedence
precedencegroup StridedRangeFormationPrecedence {
    associativity: left
    higherThan: CastingPrecedence
    lowerThan: RangeFormationPrecedence
}

final class BasicOperatorTests: XCTestCase {
    func testElementIndexing() {
        // NOTE: cannot test multiple `Tensor.shape` or `Tensor.scalars` directly
        // until send and receive are implemented (without writing a bunch of mini
        // tests). Instead, `Tensor.array` is called to make a ShapedArray host copy
        // and the ShapedArray is tested.
        let tensor3D = Tensor<Float>(
            shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
        let element2D = tensor3D[2]
        let element1D = tensor3D[1][3]
        let element0D = tensor3D[2][0][3]

        let array2D = element2D.array
        let array1D = element1D.array
        let array0D = element0D.array

        /// Test shapes
        XCTAssertEqual([4, 5], array2D.shape)
        XCTAssertEqual([5], array1D.shape)
        XCTAssertEqual([], array0D.shape)

        /// Test scalars
        XCTAssertEqual(Array(stride(from: 40.0, to: 60, by: 1)), array2D.scalars)
        XCTAssertEqual(Array(stride(from: 35.0, to: 40, by: 1)), array1D.scalars)
        XCTAssertEqual([43], array0D.scalars)
    }

    func testElementIndexingAssignment() {
        // NOTE: cannot test multiple `Tensor.shape` or `Tensor.scalars` directly
        // until send and receive are implemented (without writing a bunch of mini
        // tests). Instead, `Tensor.array` is called to make a ShapedArray host copy
        // and the ShapedArray is tested.
        var tensor3D = Tensor<Float>(
            shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
        tensor3D[2] = Tensor<Float>(
            shape: [4, 5], scalars: Array(stride(from: 20.0, to: 40, by: 1)))
        let element2D = tensor3D[2]
        let element1D = tensor3D[1][3]
        let element0D = tensor3D[2][0][3]

        let array2D = element2D.array
        let array1D = element1D.array
        let array0D = element0D.array

        /// Test shapes
        XCTAssertEqual([4, 5], array2D.shape)
        XCTAssertEqual([5], array1D.shape)
        XCTAssertEqual([], array0D.shape)

        /// Test scalars
        XCTAssertEqual(Array(stride(from: 20.0, to: 40, by: 1)), array2D.scalars)
        XCTAssertEqual(Array(stride(from: 35.0, to: 40, by: 1)), array1D.scalars)
        XCTAssertEqual([23], array0D.scalars)
    }

    func testNestedElementIndexing() {
        // NOTE: This test could use a clearer name, along with other "indexing"
        // tests. Note to update corresponding test names in other files
        // (shaped_array.test) as well.
        let tensor3D = Tensor<Float>(
            shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
        let element1D = tensor3D[1, 3]
        let element0D = tensor3D[2, 0, 3]

        let array1D = element1D.array
        let array0D = element0D.array

        /// Test shapes
        XCTAssertEqual([5], array1D.shape)
        XCTAssertEqual([], array0D.shape)

        /// Test scalars
        XCTAssertEqual(Array(stride(from: 35.0, to: 40, by: 1)), array1D.scalars)
        XCTAssertEqual([43], array0D.scalars)
    }

    func testSliceIndexing() {
        // NOTE: cannot test `Tensor.shape` or `Tensor.scalars` directly until send
        // and receive are implemented (without writing a bunch of mini tests).
        // Instead, `Tensor.array` is called to make a ShapedArray host copy and the
        // ShapedArray is tested instead.
        let tensor3D = Tensor<Float>(
            shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
        let slice3D = tensor3D[2...]
        let slice2D = tensor3D[1][0..<2]
        let slice1D = tensor3D[0][0][3..<5]

        let array3D = slice3D.array
        let array2D = slice2D.array
        let array1D = slice1D.array

        /// Test shapes
        XCTAssertEqual([1, 4, 5], array3D.shape)
        XCTAssertEqual([2, 5], array2D.shape)
        XCTAssertEqual([2], array1D.shape)

        /// Test scalars
        XCTAssertEqual(Array(stride(from: 40.0, to: 60, by: 1)), array3D.scalars)
        XCTAssertEqual(Array(stride(from: 20.0, to: 30, by: 1)), array2D.scalars)
        XCTAssertEqual(Array(stride(from: 3.0, to: 5, by: 1)), array1D.scalars)
    }

    func testSliceIndexingAssignment() {
        // NOTE: cannot test `Tensor.shape` or `Tensor.scalars` directly until send
        // and receive are implemented (without writing a bunch of mini tests).
        // Instead, `Tensor.array` is called to make a ShapedArray host copy and the
        // ShapedArray is tested instead.
        var tensor3D = Tensor<Float>(
            shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
        tensor3D[2, 0..<5, 0..<6] = Tensor<Float>(
            shape: [4, 5], scalars: Array(stride(from: 20.0, to: 40, by: 1)))
        let slice3D = tensor3D[2...]
        let slice2D = tensor3D[1][0..<2]
        let slice1D = tensor3D[0][0][3..<5]

        let array3D = slice3D.array
        let array2D = slice2D.array
        let array1D = slice1D.array

        /// Test shapes
        XCTAssertEqual([1, 4, 5], array3D.shape)
        XCTAssertEqual([2, 5], array2D.shape)
        XCTAssertEqual([2], array1D.shape)

        /// Test scalars
        XCTAssertEqual(Array(stride(from: 20.0, to: 40, by: 1)), array3D.scalars)
        XCTAssertEqual(Array(stride(from: 20.0, to: 30, by: 1)), array2D.scalars)
        XCTAssertEqual(Array(stride(from: 3.0, to: 5, by: 1)), array1D.scalars)
    }

    func testEllipsisIndexing() {
        // NOTE: cannot test `Tensor.shape` or `Tensor.scalars` directly until send
        // and receive are implemented (without writing a bunch of mini tests).
        // Instead, `Tensor.array` is called to make a ShapedArray host copy and the
        // ShapedArray is tested instead.
        var tensor3D = Tensor<Float>(
            shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
        tensor3D[2, TensorRange.ellipsis] = Tensor<Float>(
            shape: [4, 5], scalars: Array(stride(from: 20.0, to: 40, by: 1)))
        let slice3D = tensor3D[2..., TensorRange.ellipsis]
        let slice2D = tensor3D[1][0..<2]
        let slice1D = tensor3D[0][0][3..<5]

        let array3D = slice3D.array
        let array2D = slice2D.array
        let array1D = slice1D.array

        /// Test shapes
        XCTAssertEqual([1, 4, 5], array3D.shape)
        XCTAssertEqual([2, 5], array2D.shape)
        XCTAssertEqual([2], array1D.shape)

        /// Test scalars
        XCTAssertEqual(Array(stride(from: 20.0, to: 40, by: 1)), array3D.scalars)
        XCTAssertEqual(Array(stride(from: 20.0, to: 30, by: 1)), array2D.scalars)
        XCTAssertEqual(Array(stride(from: 3.0, to: 5, by: 1)), array1D.scalars)
    }

    func testNewAxisIndexing() {
        // NOTE: cannot test `Tensor.shape` or `Tensor.scalars` directly until send
        // and receive are implemented (without writing a bunch of mini tests).
        // Instead, `Tensor.array` is called to make a ShapedArray host copy and the
        // ShapedArray is tested instead.
        let tensor3D = Tensor<Float>(
            shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
        let newAxis = TensorRange.newAxis
        let ellipsis = TensorRange.ellipsis
        let slice3D = tensor3D[2..., newAxis, ellipsis]
        let slice2D = tensor3D[1, newAxis][0..<1, 0..<2]
        let slice1D = tensor3D[0][newAxis, 0][0..<1, 3..<5, newAxis]

        let array3D = slice3D.array
        let array2D = slice2D.array
        let array1D = slice1D.array

        /// Test shapes
        XCTAssertEqual([1, 1, 4, 5], array3D.shape)
        XCTAssertEqual([1, 2, 5], array2D.shape)
        XCTAssertEqual([1, 2, 1], array1D.shape)

        /// Test scalars
        XCTAssertEqual(Array(stride(from: 40.0, to: 60, by: 1)), array3D.scalars)
        XCTAssertEqual(Array(stride(from: 20.0, to: 30, by: 1)), array2D.scalars)
        XCTAssertEqual(Array(stride(from: 3.0, to: 5, by: 1)), array1D.scalars)
    }

    func testSqueezeAxisIndexing() {
        // NOTE: cannot test `Tensor.shape` or `Tensor.scalars` directly until send
        // and receive are implemented (without writing a bunch of mini tests).
        // Instead, `Tensor.array` is called to make a ShapedArray host copy and the
        // ShapedArray is tested instead.
        let tensor3D = Tensor<Float>(
            shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
        let newAxis = TensorRange.newAxis
        let ellipsis = TensorRange.ellipsis
        let squeezeAxis = TensorRange.squeezeAxis
        let slice3D = tensor3D[2..., newAxis, ellipsis][squeezeAxis, squeezeAxis]
        let slice2D = tensor3D[1, newAxis][squeezeAxis, 0..<2]
        let slice1D = tensor3D[0..<1, 0, 3..<5, newAxis][
            squeezeAxis, ellipsis, squeezeAxis]

        let array3D = slice3D.array
        let array2D = slice2D.array
        let array1D = slice1D.array

        /// Test shapes
        XCTAssertEqual([4, 5], array3D.shape)
        XCTAssertEqual([2, 5], array2D.shape)
        XCTAssertEqual([2], array1D.shape)

        /// Test scalars
        XCTAssertEqual(Array(stride(from: 40.0, to: 60, by: 1)), array3D.scalars)
        XCTAssertEqual(Array(stride(from: 20.0, to: 30, by: 1)), array2D.scalars)
        XCTAssertEqual(Array(stride(from: 3.0, to: 5, by: 1)), array1D.scalars)
    }

    func testStridedSliceIndexing() {
        // NOTE: cannot test `Tensor.shape` or `Tensor.scalars` directly until send
        // and receive are implemented (without writing a bunch of mini tests).
        // Instead, `Tensor.array` is called to make a ShapedArray host copy and the
        // ShapedArray is tested instead.
        let tensor3D = Tensor<Float>(
            shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
        let slice3D = tensor3D[2...]
        let slice2D = tensor3D[1][0..<3..2]
        let slice1D = tensor3D[0][0][1..<5..2]

        let array3D = slice3D.array
        let array2D = slice2D.array
        let array1D = slice1D.array

        /// Test shapes
        XCTAssertEqual([1, 4, 5], array3D.shape)
        XCTAssertEqual([2, 5], array2D.shape)
        XCTAssertEqual([2], array1D.shape)

        /// Test scalars
        XCTAssertEqual(Array(stride(from: 40.0, to: 60, by: 1)), array3D.scalars)
        XCTAssertEqual(
            Array(stride(from: 20.0, to: 25, by: 1)) + 
            Array(stride(from: 30.0, to: 35, by: 1)), array2D.scalars)
        XCTAssertEqual(Array(stride(from: 1.0, to: 5, by: 2)), array1D.scalars)
    }

    func testStridedSliceIndexingAssignment() {
        // NOTE: cannot test `Tensor.shape` or `Tensor.scalars` directly until send
        // and receive are implemented (without writing a bunch of mini tests).
        // Instead, `Tensor.array` is called to make a ShapedArray host copy and the
        // ShapedArray is tested instead.
        var tensor3D = Tensor<Float>(
            shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
        tensor3D[2, 0..<5..2, 0..<6] = Tensor<Float>(
            shape: [2, 5], scalars: Array(stride(from: 20.0, to: 40, by: 2)))
        let slice3D = tensor3D[2...]
        let slice2D = tensor3D[1][0..<2]
        let slice1D = tensor3D[0][0][3..<5]

        let array3D = slice3D.array
        let array2D = slice2D.array
        let array1D = slice1D.array

        /// Test shapes
        XCTAssertEqual([1, 4, 5], array3D.shape)
        XCTAssertEqual([2, 5], array2D.shape)
        XCTAssertEqual([2], array1D.shape)

        /// Test scalars
        XCTAssertEqual(
            Array(stride(from: 20.0, to: 30, by: 2)) + 
            Array(stride(from: 45.0, to: 50, by: 1)) + 
            Array(stride(from: 30.0, to: 40, by: 2)) + 
            Array(stride(from: 55.0, to: 60, by: 1)), array3D.scalars)
        XCTAssertEqual(Array(stride(from: 20.0, to: 30, by: 1)), array2D.scalars)
        XCTAssertEqual(Array(stride(from: 3.0, to: 5, by: 1)), array1D.scalars)
    }

    func testWholeTensorSlicing() {
        let t: Tensor<Int32> = [[[1, 1, 1], [2, 2, 2]],
                                [[3, 3, 3], [4, 4, 4]],
                                [[5, 5, 5], [6, 6, 6]]]
        let slice2 = t.slice(lowerBounds: [1, 0, 0], upperBounds: [2, 1, 3])
        XCTAssertEqual(ShapedArray(shape: [1, 1, 3], scalars: [3, 3, 3]), slice2.array)
    }

    func testAdvancedIndexing() {
        // NOTE: cannot test multiple `Tensor.shape` or `Tensor.scalars` directly
        // until send and receive are implemented (without writing a bunch of mini
        // tests). Instead, `Tensor.array` is called to make a ShapedArray host copy
        // and the ShapedArray is tested.
        let tensor3D = Tensor<Float>(
            shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
        let element2D = tensor3D[1..<3, 0, 3...]
        let array2D = element2D.array

        // Test shape
        XCTAssertEqual([2, 2], array2D.shape)

        // Test scalars
        XCTAssertEqual(Array([23.0, 24.0, 43.0, 44.0]), array2D.scalars)
    }

    func testConcatenation() {
        // 2 x 3
        let t1 = Tensor<Int32>([[0, 1, 2], [3, 4, 5]])
        // 2 x 3
        let t2 = Tensor<Int32>([[6, 7, 8], [9, 10, 11]])
        let concatenated = t1 ++ t2
        let concatenated0 = t1.concatenated(with: t2)
        let concatenated1 = t1.concatenated(with: t2, alongAxis: 1)
        XCTAssertEqual(ShapedArray(shape: [4, 3], scalars: Array(0..<12)), concatenated.array)
        XCTAssertEqual(ShapedArray(shape: [4, 3], scalars: Array(0..<12)), concatenated0.array)
        XCTAssertEqual(
            ShapedArray(shape: [2, 6], scalars: [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]),
            concatenated1.array)
    }

    func testVJPConcatenation() {
        let a1 = Tensor<Float>([1,2,3,4])
        let b1 = Tensor<Float>([5,6,7,8,9,10])

        let a2 = Tensor<Float>([1,1,1,1])
        let b2 = Tensor<Float>([1,1,1,1,1,1])

        let grads = gradient(at: a2, b2) { a, b in
            return ((a1 * a) ++ (b1 * b)).sum()
        }

        XCTAssertEqual(a1, grads.0)
        XCTAssertEqual(b1, grads.1)
    }

    func testVJPConcatenationNegativeAxis() {
        let a1 = Tensor<Float>([1,2,3,4])
        let b1 = Tensor<Float>([5,6,7,8,9,10])

        let a2 = Tensor<Float>([1,1,1,1])
        let b2 = Tensor<Float>([1,1,1,1,1,1])

        let grads = gradient(at: a2, b2) { a, b in
            return (a1 * a).concatenated(with: b1 * b, alongAxis: -1).sum()
        }

        XCTAssertEqual(a1, grads.0)
        XCTAssertEqual(b1, grads.1)
    }

    func testTranspose() {
        // 3 x 2 -> 2 x 3
        let xT = Tensor<Float>([[1, 2], [3, 4], [5, 6]]).transposed()
        let xTArray = xT.array
        XCTAssertEqual(2, xTArray.rank)
        XCTAssertEqual([2, 3], xTArray.shape)
        XCTAssertEqual([1, 3, 5, 2, 4, 6], xTArray.scalars)
    }

    func testReshape() {
        // 2 x 3 -> 1 x 3 x 1 x 2 x 1
        let matrix = Tensor<Int32>([[0, 1, 2], [3, 4, 5]])
        let reshaped = matrix.reshaped(to: [1, 3, 1, 2, 1])

        XCTAssertEqual([1, 3, 1, 2, 1], reshaped.shape)
        XCTAssertEqual(Array(0..<6), reshaped.scalars)
    }

    func testFlatten() {
        // 2 x 3 -> 6
        let matrix = Tensor<Int32>([[0, 1, 2], [3, 4, 5]])
        let flattened = matrix.flattened()

        XCTAssertEqual([6], flattened.shape)
        XCTAssertEqual(Array(0..<6), flattened.scalars)
    }

    func testFlatten0D() {
        let scalar = Tensor<Float>(5)
        let flattened = scalar.flattened()
        XCTAssertEqual([1], flattened.shape)
        XCTAssertEqual([5], flattened.scalars)
    }

    func testReshapeToScalar() {
        // 1 x 1 -> scalar
        let z = Tensor<Float>([[10]]).reshaped(to: [])
        XCTAssertEqual([], z.shape)
    }

    func testReshapeTensor() {
        // 2 x 3 -> 1 x 3 x 1 x 2 x 1
        let x = Tensor<Float>(repeating: 0.0, shape: [2, 3])
        let y = Tensor<Float>(repeating: 0.0, shape: [1, 3, 1, 2, 1])
        let result = x.reshaped(like: y)
        XCTAssertEqual([1, 3, 1, 2, 1], result.shape)
    }

    func testUnbroadcasted1() {
        let x = Tensor<Float>(repeating: 1, shape: [2, 3, 4, 5])
        let y = Tensor<Float>(repeating: 1, shape: [4, 5])
        let z = x.unbroadcasted(like: y)
        XCTAssertEqual(ShapedArray<Float>(repeating: 6, shape: [4, 5]), z.array)
    }

    func testUnbroadcasted2() {
        let x = Tensor<Float>(repeating: 1, shape: [2, 3, 4, 5])
        let y = Tensor<Float>(repeating: 1, shape: [3, 1, 5])
        let z = x.unbroadcasted(like: y)
        XCTAssertEqual(ShapedArray<Float>(repeating: 8, shape: [3, 1, 5]), z.array)
    }

    func testSliceUpdate() {
        var t1 = Tensor<Float>([[1, 2, 3], [4, 5, 6]])
        t1[0] = Tensor(zeros: [3])
        XCTAssertEqual(ShapedArray(shape:[2, 3], scalars: [0, 0, 0, 4, 5, 6]), t1.array)
        var t2 = t1
        t2[0][2] = Tensor(3)
        XCTAssertEqual(ShapedArray(shape:[2, 3], scalars: [0, 0, 3, 4, 5, 6]), t2.array)
        var t3 = Tensor<Bool>([[true, true, true], [false, false, false]])
        t3[0][1] = Tensor(false)
        XCTAssertEqual(ShapedArray(
            shape:[2, 3], scalars: [true, false, true, false, false, false]), t3.array)
        var t4 = Tensor<Bool>([[true, true, true], [false, false, false]])
        t4[0] = Tensor(repeating: false, shape: [3])
        XCTAssertEqual(ShapedArray(repeating: false, shape: [2, 3]), t4.array)
    }

    func testBroadcastTensor() {
        // 1 -> 2 x 3 x 4
        let one = Tensor<Float>(1)
        var target = Tensor<Float>(repeating: 0.0, shape: [2, 3, 4])
        let broadcasted = one.broadcasted(like: target)
        XCTAssertEqual(Tensor(repeating: 1, shape: [2, 3, 4]), broadcasted)
        target .= Tensor(repeating: 1, shape: [1, 3, 1])
        XCTAssertEqual(Tensor(repeating: 1, shape: [2, 3, 4]), target)
    }

    static var allTests = [
        ("testElementIndexing", testElementIndexing),
        ("testElementIndexingAssignment", testElementIndexingAssignment),
        ("testNestedElementIndexing", testNestedElementIndexing),
        ("testSliceIndexing", testSliceIndexing),
        ("testSliceIndexingAssignment", testSliceIndexingAssignment),
        ("testEllipsisIndexing", testEllipsisIndexing),
        ("testNewAxisIndexing", testNewAxisIndexing),
        ("testSqueezeAxisIndexing", testSqueezeAxisIndexing),
        ("testStridedSliceIndexing", testStridedSliceIndexing),
        ("testStridedSliceIndexingAssignment", testStridedSliceIndexingAssignment),
        ("testWholeTensorSlicing", testWholeTensorSlicing),
        ("testAdvancedIndexing", testAdvancedIndexing),
        ("testConcatenation", testConcatenation),
        ("testVJPConcatenation", testVJPConcatenation),
        ("testTranspose", testTranspose),
        ("testReshape", testReshape),
        ("testFlatten", testFlatten),
        ("testFlatten0D", testFlatten0D),
        ("testReshapeToScalar", testReshapeToScalar),
        ("testReshapeTensor", testReshapeTensor),
        ("testUnbroadcasted1", testUnbroadcasted1),
        ("testUnbroadcasted2", testUnbroadcasted2),
        ("testSliceUpdate", testSliceUpdate),
        ("testBroadcastTensor", testBroadcastTensor)
    ]
}
