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

infix operator ++: AdditionPrecedence
infix operator .=

infix operator ..: StridedRangeFormationPrecedence
precedencegroup StridedRangeFormationPrecedence {
    associativity: left
    higherThan: CastingPrecedence
    lowerThan: RangeFormationPrecedence
}

final class BasicOperatorTests: XCTestCase {
    func testGathering() {
        let x = Tensor<Float>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let y = x.gathering(atIndices: Tensor<Int32>(2), alongAxis: 1)
        XCTAssertEqual(y, Tensor<Float>([3.0, 6.0]))
    }

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
        XCTAssertEqual(array2D.shape, [4, 5])
        XCTAssertEqual(array1D.shape, [5])
        XCTAssertEqual(array0D.shape, [])

        /// Test scalars
        XCTAssertEqual(array2D.scalars, Array(stride(from: 40.0, to: 60, by: 1)))
        XCTAssertEqual(array1D.scalars, Array(stride(from: 35.0, to: 40, by: 1)))
        XCTAssertEqual(array0D.scalars, [43])
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
        XCTAssertEqual(array2D.shape, [4, 5])
        XCTAssertEqual(array1D.shape, [5])
        XCTAssertEqual(array0D.shape, [])

        /// Test scalars
        XCTAssertEqual(array2D.scalars, Array(stride(from: 20.0, to: 40, by: 1)))
        XCTAssertEqual(array1D.scalars, Array(stride(from: 35.0, to: 40, by: 1)))
        XCTAssertEqual(array0D.scalars, [23])
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
        XCTAssertEqual(array1D.shape, [5])
        XCTAssertEqual(array0D.shape, [])

        /// Test scalars
        XCTAssertEqual(array1D.scalars, Array(stride(from: 35.0, to: 40, by: 1)))
        XCTAssertEqual(array0D.scalars, [43])
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
        XCTAssertEqual(array3D.shape, [1, 4, 5])
        XCTAssertEqual(array2D.shape, [2, 5])
        XCTAssertEqual(array1D.shape, [2])

        /// Test scalars
        XCTAssertEqual(array3D.scalars, Array(stride(from: 40.0, to: 60, by: 1)))
        XCTAssertEqual(array2D.scalars, Array(stride(from: 20.0, to: 30, by: 1)))
        XCTAssertEqual(array1D.scalars, Array(stride(from: 3.0, to: 5, by: 1)))
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
        XCTAssertEqual(array3D.shape, [1, 4, 5])
        XCTAssertEqual(array2D.shape, [2, 5])
        XCTAssertEqual(array1D.shape, [2])

        /// Test scalars
        XCTAssertEqual(array3D.scalars, Array(stride(from: 20.0, to: 40, by: 1)))
        XCTAssertEqual(array2D.scalars, Array(stride(from: 20.0, to: 30, by: 1)))
        XCTAssertEqual(array1D.scalars, Array(stride(from: 3.0, to: 5, by: 1)))
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
        XCTAssertEqual(array3D.shape, [1, 4, 5])
        XCTAssertEqual(array2D.shape, [2, 5])
        XCTAssertEqual(array1D.shape, [2])

        /// Test scalars
        XCTAssertEqual(array3D.scalars, Array(stride(from: 20.0, to: 40, by: 1)))
        XCTAssertEqual(array2D.scalars, Array(stride(from: 20.0, to: 30, by: 1)))
        XCTAssertEqual(array1D.scalars, Array(stride(from: 3.0, to: 5, by: 1)))
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
        XCTAssertEqual(array3D.shape, [1, 1, 4, 5])
        XCTAssertEqual(array2D.shape, [1, 2, 5])
        XCTAssertEqual(array1D.shape, [1, 2, 1])

        /// Test scalars
        XCTAssertEqual(array3D.scalars, Array(stride(from: 40.0, to: 60, by: 1)))
        XCTAssertEqual(array2D.scalars, Array(stride(from: 20.0, to: 30, by: 1)))
        XCTAssertEqual(array1D.scalars, Array(stride(from: 3.0, to: 5, by: 1)))
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
        XCTAssertEqual(array3D.shape, [4, 5])
        XCTAssertEqual(array2D.shape, [2, 5])
        XCTAssertEqual(array1D.shape, [2])

        /// Test scalars
        XCTAssertEqual(array3D.scalars, Array(stride(from: 40.0, to: 60, by: 1)))
        XCTAssertEqual(array2D.scalars, Array(stride(from: 20.0, to: 30, by: 1)))
        XCTAssertEqual(array1D.scalars, Array(stride(from: 3.0, to: 5, by: 1)))
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
        XCTAssertEqual(array3D.shape, [1, 4, 5])
        XCTAssertEqual(array2D.shape, [2, 5])
        XCTAssertEqual(array1D.shape, [2])

        /// Test scalars
        XCTAssertEqual(array3D.scalars, Array(stride(from: 40.0, to: 60, by: 1)))
        XCTAssertEqual(
            array2D.scalars,
            Array(stride(from: 20.0, to: 25, by: 1)) + 
            Array(stride(from: 30.0, to: 35, by: 1)))
        XCTAssertEqual(array1D.scalars, Array(stride(from: 1.0, to: 5, by: 2)))
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
        XCTAssertEqual(array3D.shape, [1, 4, 5])
        XCTAssertEqual(array2D.shape, [2, 5])
        XCTAssertEqual(array1D.shape, [2])

        /// Test scalars
        XCTAssertEqual(array3D.scalars,
                       [Float](stride(from: 20.0, to: 30, by: 2)) +
                       [Float](stride(from: 45.0, to: 50, by: 1)) +
                       [Float](stride(from: 30.0, to: 40, by: 2)) +
                       [Float](stride(from: 55.0, to: 60, by: 1)))
        XCTAssertEqual(array2D.scalars, Array(stride(from: 20.0, to: 30, by: 1)))
        XCTAssertEqual(array1D.scalars, Array(stride(from: 3.0, to: 5, by: 1)))
    }

    func testWholeTensorSlicing() {
        let t: Tensor<Int32> = [[[1, 1, 1], [2, 2, 2]],
                                [[3, 3, 3], [4, 4, 4]],
                                [[5, 5, 5], [6, 6, 6]]]
        let slice2 = t.slice(lowerBounds: [1, 0, 0], upperBounds: [2, 1, 3])
        XCTAssertEqual(slice2.array, ShapedArray(shape: [1, 1, 3], scalars: [3, 3, 3]))
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
        XCTAssertEqual(array2D.shape, [2, 2])

        // Test scalars
        XCTAssertEqual(array2D.scalars, Array([23.0, 24.0, 43.0, 44.0]))
    }

    func testConcatenation() {
        // 2 x 3
        let t1 = Tensor<Int32>([[0, 1, 2], [3, 4, 5]])
        // 2 x 3
        let t2 = Tensor<Int32>([[6, 7, 8], [9, 10, 11]])
        let concatenated = t1 ++ t2
        let concatenated0 = t1.concatenated(with: t2)
        let concatenated1 = t1.concatenated(with: t2, alongAxis: 1)
        XCTAssertEqual(concatenated.array, ShapedArray(shape: [4, 3], scalars: Array(0..<12)))
        XCTAssertEqual(concatenated0.array, ShapedArray(shape: [4, 3], scalars: Array(0..<12)))
        XCTAssertEqual(
            concatenated1.array, 
            ShapedArray(shape: [2, 6], scalars: [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]))
    }

    func testVJPConcatenation() {
        let a1 = Tensor<Float>([1,2,3,4])
        let b1 = Tensor<Float>([5,6,7,8,9,10])

        let a2 = Tensor<Float>([1,1,1,1])
        let b2 = Tensor<Float>([1,1,1,1,1,1])

        let grads = gradient(at: a2, b2) { a, b in
            return ((a1 * a) ++ (b1 * b)).sum()
        }

        XCTAssertEqual(grads.0, a1)
        XCTAssertEqual(grads.1, b1)
    }

    func testVJPConcatenationNegativeAxis() {
        let a1 = Tensor<Float>([1,2,3,4])
        let b1 = Tensor<Float>([5,6,7,8,9,10])

        let a2 = Tensor<Float>([1,1,1,1])
        let b2 = Tensor<Float>([1,1,1,1,1,1])

        let grads = gradient(at: a2, b2) { a, b in
            return (a1 * a).concatenated(with: b1 * b, alongAxis: -1).sum()
        }

        XCTAssertEqual(grads.0, a1)
        XCTAssertEqual(grads.1, b1)
    }

    func testTranspose() {
        // 3 x 2 -> 2 x 3
        let xT = Tensor<Float>([[1, 2], [3, 4], [5, 6]]).transposed()
        let xTArray = xT.array
        XCTAssertEqual(xTArray.rank, 2)
        XCTAssertEqual(xTArray.shape, [2, 3])
        XCTAssertEqual(xTArray.scalars, [1, 3, 5, 2, 4, 6])
    }

    func testReshape() {
        // 2 x 3 -> 1 x 3 x 1 x 2 x 1
        let matrix = Tensor<Int32>([[0, 1, 2], [3, 4, 5]])
        let reshaped = matrix.reshaped(to: [1, 3, 1, 2, 1])

        XCTAssertEqual(reshaped.shape, [1, 3, 1, 2, 1])
        XCTAssertEqual(reshaped.scalars, Array(0..<6))
    }

    func testFlatten() {
        // 2 x 3 -> 6
        let matrix = Tensor<Int32>([[0, 1, 2], [3, 4, 5]])
        let flattened = matrix.flattened()

        XCTAssertEqual(flattened.shape, [6])
        XCTAssertEqual(flattened.scalars, Array(0..<6))
    }

    func testFlatten0D() {
        let scalar = Tensor<Float>(5)
        let flattened = scalar.flattened()
        XCTAssertEqual(flattened.shape, [1])
        XCTAssertEqual(flattened.scalars, [5])
    }

    func testReshapeToScalar() {
        // 1 x 1 -> scalar
        let z = Tensor<Float>([[10]]).reshaped(to: [])
        XCTAssertEqual(z.shape, [])
    }

    func testReshapeTensor() {
        // 2 x 3 -> 1 x 3 x 1 x 2 x 1
        let x = Tensor<Float>(repeating: 0.0, shape: [2, 3])
        let y = Tensor<Float>(repeating: 0.0, shape: [1, 3, 1, 2, 1])
        let result = x.reshaped(like: y)
        XCTAssertEqual(result.shape, [1, 3, 1, 2, 1])
    }

    func testUnbroadcast1() {
        let x = Tensor<Float>(repeating: 1, shape: [2, 3, 4, 5])
        let y = Tensor<Float>(repeating: 1, shape: [4, 5])
        let z = x.unbroadcasted(like: y)
        XCTAssertEqual(z.array, ShapedArray<Float>(repeating: 6, shape: [4, 5]))
    }

    func testUnbroadcast2() {
        let x = Tensor<Float>(repeating: 1, shape: [2, 3, 4, 5])
        let y = Tensor<Float>(repeating: 1, shape: [3, 1, 5])
        let z = x.unbroadcasted(like: y)
        XCTAssertEqual(z.array, ShapedArray<Float>(repeating: 8, shape: [3, 1, 5]))
    }

    func testSliceUpdate() {
        var t1 = Tensor<Float>([[1, 2, 3], [4, 5, 6]])
        t1[0] = Tensor(zeros: [3])
        XCTAssertEqual(t1.array, ShapedArray(shape:[2, 3], scalars: [0, 0, 0, 4, 5, 6]))
        var t2 = t1
        t2[0][2] = Tensor(3)
        XCTAssertEqual(t2.array, ShapedArray(shape:[2, 3], scalars: [0, 0, 3, 4, 5, 6]))
        var t3 = Tensor<Bool>([[true, true, true], [false, false, false]])
        t3[0][1] = Tensor(false)
        XCTAssertEqual(t3.array, ShapedArray(
            shape:[2, 3], scalars: [true, false, true, false, false, false]))
        var t4 = Tensor<Bool>([[true, true, true], [false, false, false]])
        t4[0] = Tensor(repeating: false, shape: [3])
        XCTAssertEqual(t4.array, ShapedArray(repeating: false, shape: [2, 3]))
    }

    func testBroadcastTensor() {
        // 1 -> 2 x 3 x 4
        let one = Tensor<Float>(1)
        var target = Tensor<Float>(repeating: 0.0, shape: [2, 3, 4])
        let broadcasted = one.broadcasted(like: target)
        XCTAssertEqual(broadcasted, Tensor(repeating: 1, shape: [2, 3, 4]))
        target .= Tensor(repeating: 1, shape: [1, 3, 1])
        XCTAssertEqual(target, Tensor(repeating: 1, shape: [2, 3, 4]))
    }

    static var allTests = [
        ("testGathering", testGathering),
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
        ("testUnbroadcast1", testUnbroadcast1),
        ("testUnbroadcast2", testUnbroadcast2),
        ("testSliceUpdate", testSliceUpdate),
        ("testBroadcastTensor", testBroadcastTensor)
    ]
}
