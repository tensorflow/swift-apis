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

@testable import Experimental

final class ComplexTests: XCTestCase {
  func testInitializer() {
    let complex = Complex<Float>(real: 2, imaginary: 3)
    XCTAssertEqual(complex.real, 2)
    XCTAssertEqual(complex.imaginary, 3)
  }

  func testStaticImaginary() {
    let imaginary = Complex<Float>(real: 0, imaginary: 1)
    XCTAssertEqual(imaginary, Complex.i)
  }

  func testIsFinite() {
    var complex = Complex<Float>(real: 999, imaginary: 0)
    XCTAssertTrue(complex.isFinite)

    complex = Complex(real: 1.0 / 0.0, imaginary: 1)
    XCTAssertFalse(complex.isFinite)

    complex = Complex(real: 1.0 / 0.0, imaginary: 1.0 / 0.0)
    XCTAssertFalse(complex.isFinite)
  }

  func testIsInfinite() {
    var complex = Complex<Float>(real: 999, imaginary: 0)
    XCTAssertFalse(complex.isInfinite)

    complex = Complex(real: 1.0 / 0.0, imaginary: 1)
    XCTAssertTrue(complex.isInfinite)

    complex = Complex(real: 1.0 / 0.0, imaginary: 1.0 / 0.0)
    XCTAssertTrue(complex.isInfinite)
  }

  func testIsNaN() {
    var complex = Complex<Float>(real: 999, imaginary: 0)
    XCTAssertFalse(complex.isNaN)

    complex = Complex(real: 0.0 * 1.0 / 0.0, imaginary: 1)
    XCTAssertTrue(complex.isNaN)

    complex = Complex(real: 0.0 * 1.0 / 0.0, imaginary: 0.0 * 1.0 / 0.0)
    XCTAssertTrue(complex.isNaN)
  }

  func testIsZero() {
    var complex = Complex<Float>(real: 999, imaginary: 0)
    XCTAssertFalse(complex.isZero)

    complex = Complex(real: 0.0 * 1.0 / 0.0, imaginary: 0)
    XCTAssertFalse(complex.isZero)

    complex = Complex(real: 0.0 * 1.0 / 0.0, imaginary: 0.0 * 1.0 / 0.0)
    XCTAssertFalse(complex.isZero)

    complex = Complex(real: 0, imaginary: 0)
    XCTAssertTrue(complex.isZero)
  }

  func testEquals() {
    var complexA = Complex<Float>(real: 999, imaginary: 0)
    let complexB = Complex<Float>(real: 999, imaginary: 0)
    XCTAssertEqual(complexA, complexB)

    complexA = Complex(real: 5, imaginary: 0)
    XCTAssertNotEqual(complexA, complexB)
  }

  func testPlus() {
    let input = Complex<Float>(real: 5, imaginary: 1)
    let expected = Complex<Float>(real: 10, imaginary: 2)
    XCTAssertEqual(expected, input + input)
  }

  func testMinus() {
    let inputA = Complex<Float>(real: 6, imaginary: 2)
    let inputB = Complex<Float>(real: 5, imaginary: 1)
    let expected = Complex<Float>(real: 1, imaginary: 1)
    XCTAssertEqual(expected, inputA - inputB)
  }

  func testTimes() {
    let inputA = Complex<Float>(real: 6, imaginary: 2)
    let inputB = Complex<Float>(real: 5, imaginary: 1)
    let expected = Complex<Float>(real: 28, imaginary: 16)
    XCTAssertEqual(expected, inputA * inputB)
  }

  func testNegate() {
    var input = Complex<Float>(real: 6, imaginary: 2)
    let negated = Complex<Float>(real: -6, imaginary: -2)
    XCTAssertEqual(-input, negated)
    input.negate()
    XCTAssertEqual(input, negated)
  }

  func testDivide() {
    let inputA = Complex<Float>(real: 20, imaginary: -4)
    let inputB = Complex<Float>(real: 3, imaginary: 2)
    let expected = Complex<Float>(real: 4, imaginary: -4)
    XCTAssertEqual(expected, inputA / inputB)
  }

  func testComplexConjugate() {
    var input = Complex<Float>(real: 2, imaginary: -4)
    var expected = Complex<Float>(real: 2, imaginary: 4)
    XCTAssertEqual(expected, input.complexConjugate())

    input = Complex<Float>(real: -2, imaginary: -4)
    expected = Complex<Float>(real: -2, imaginary: 4)
    XCTAssertEqual(expected, input.complexConjugate())

    input = Complex<Float>(real: 2, imaginary: 4)
    expected = Complex<Float>(real: 2, imaginary: -4)
    XCTAssertEqual(expected, input.complexConjugate())
  }

  func testAdding() {
    var input = Complex<Float>(real: 2, imaginary: -4)
    var expected = Complex<Float>(real: 3, imaginary: -4)
    XCTAssertEqual(expected, input.adding(real: 1))

    input = Complex<Float>(real: 2, imaginary: -4)
    expected = Complex<Float>(real: 2, imaginary: -3)
    XCTAssertEqual(expected, input.adding(imaginary: 1))
  }

  func testSubtracting() {
    var input = Complex<Float>(real: 2, imaginary: -4)
    var expected = Complex<Float>(real: 1, imaginary: -4)
    XCTAssertEqual(expected, input.subtracting(real: 1))

    input = Complex<Float>(real: 2, imaginary: -4)
    expected = Complex<Float>(real: 2, imaginary: -5)
    XCTAssertEqual(expected, input.subtracting(imaginary: 1))
  }

  func testVjpInit() {
    var pb = pullback(at: 4, -3) { r, i in
      return Complex<Float>(real: r, imaginary: i)
    }
    var tanTuple = pb(Complex<Float>(real: -1, imaginary: 2))
    XCTAssertEqual(-1, tanTuple.0)
    XCTAssertEqual(2, tanTuple.1)

    pb = pullback(at: 4, -3) { r, i in
      return Complex<Float>(real: r * r, imaginary: i + i)
    }
    tanTuple = pb(Complex<Float>(real: -1, imaginary: 1))
    XCTAssertEqual(-8, tanTuple.0)
    XCTAssertEqual(2, tanTuple.1)
  }

  func testVjpAdd() {
    let pb: (Complex<Float>) -> Complex<Float> =
      pullback(at: Complex<Float>(real: 2, imaginary: 3)) { x in
        return x + Complex<Float>(real: 5, imaginary: 6)
      }
    XCTAssertEqual(
      pb(Complex(real: 1, imaginary: 1)),
      Complex<Float>(real: 1, imaginary: 1))
  }

  func testVjpSubtract() {
    let pb: (Complex<Float>) -> Complex<Float> =
      pullback(at: Complex<Float>(real: 2, imaginary: 3)) { x in
        return Complex<Float>(real: 5, imaginary: 6) - x
      }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: -1, imaginary: -1))
  }

  func testVjpMultiply() {
    let pb: (Complex<Float>) -> Complex<Float> =
      pullback(at: Complex<Float>(real: 2, imaginary: 3)) { x in
        return x * x
      }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 4, imaginary: 6))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: -6, imaginary: 4))
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: -2, imaginary: 10))
  }

  func testVjpDivide() {
    let pb: (Complex<Float>) -> Complex<Float> =
      pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
        return x / Complex<Float>(real: 2, imaginary: 2)
      }
    XCTAssertEqual(
      pb(Complex(real: 1, imaginary: 0)),
      Complex<Float>(real: 0.25, imaginary: -0.25))
    XCTAssertEqual(
      pb(Complex(real: 0, imaginary: 1)),
      Complex<Float>(real: 0.25, imaginary: 0.25))
  }

  func testVjpNegate() {
    let pb: (Complex<Float>) -> Complex<Float> =
      pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
        return -x
      }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: -1, imaginary: 0))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: -1))
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: -1, imaginary: -1))
  }

  func testVjpComplexConjugate() {
    let pb: (Complex<Float>) -> Complex<Float> =
      pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
        return x.complexConjugate()
      }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 1, imaginary: 0))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: -1))
    XCTAssertEqual(pb(Complex(real: -1, imaginary: 1)), Complex<Float>(real: -1, imaginary: -1))
  }

  func testVjpAddingReal() {
    let pb: (Complex<Float>) -> Complex<Float> =
      pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
        return x.adding(real: 5)
      }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 1, imaginary: 0))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: 1))
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
  }

  func testVjpAddingImaginary() {
    let pb: (Complex<Float>) -> Complex<Float> =
      pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
        return x.adding(imaginary: 5)
      }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 1, imaginary: 0))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: 1))
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
  }

  func testVjpSubtractingReal() {
    let pb: (Complex<Float>) -> Complex<Float> =
      pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
        return x.subtracting(real: 5)
      }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 1, imaginary: 0))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: 1))
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
  }

  func testVjpSubtractingImaginary() {
    let pb: (Complex<Float>) -> Complex<Float> =
      pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
        return x.subtracting(imaginary: 5)
      }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 1, imaginary: 0))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: 1))
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
  }

  func testJvpDotProduct() {
    struct ComplexVector: Differentiable & AdditiveArithmetic {
      var w: Complex<Float>
      var x: Complex<Float>
      var y: Complex<Float>
      var z: Complex<Float>

      init(w: Complex<Float>, x: Complex<Float>, y: Complex<Float>, z: Complex<Float>) {
        self.w = w
        self.x = x
        self.y = y
        self.z = z
      }
    }

    func dot(lhs: ComplexVector, rhs: ComplexVector) -> Complex<Float> {
      var result: Complex<Float> = Complex(real: 0, imaginary: 0)
      result = result + lhs.w.complexConjugate() * rhs.w
      result = result + lhs.x.complexConjugate() * rhs.x
      result = result + lhs.y.complexConjugate() * rhs.y
      result = result + lhs.z.complexConjugate() * rhs.z
      return result
    }

    let atVector = ComplexVector(
      w: Complex(real: 1, imaginary: 1),
      x: Complex(real: 1, imaginary: -1),
      y: Complex(real: -1, imaginary: 1),
      z: Complex(real: -1, imaginary: -1))
    let rhsVector = ComplexVector(
      w: Complex(real: 3, imaginary: -4),
      x: Complex(real: 6, imaginary: -2),
      y: Complex(real: 1, imaginary: 2),
      z: Complex(real: 4, imaginary: 3))
    let expectedVector = ComplexVector(
      w: Complex(real: 7, imaginary: 1),
      x: Complex(real: 8, imaginary: -4),
      y: Complex(real: -1, imaginary: -3),
      z: Complex(real: 1, imaginary: -7))

    let (result, pbComplex) = valueWithPullback(at: atVector) { x in
      return dot(lhs: x, rhs: rhsVector)
    }

    XCTAssertEqual(Complex(real: 1, imaginary: -5), result)
    XCTAssertEqual(expectedVector, pbComplex(Complex(real: 1, imaginary: 1)))
  }

  func testImplicitDifferentiation() {
    func addRealComponents(lhs: Complex<Float>, rhs: Complex<Float>) -> Float {
      return lhs.real + rhs.real
    }

    let (result, pbComplex) = valueWithPullback(at: Complex(real: 2, imaginary: -3)) { x in
      return addRealComponents(lhs: x, rhs: Complex(real: -4, imaginary: 1))
    }

    XCTAssertEqual(-2, result)
    XCTAssertEqual(Complex(real: 1, imaginary: 0), pbComplex(1))
  }

  static var allTests = [
    ("testInitializer", testInitializer),
    ("testStaticImaginary", testStaticImaginary),
    ("testIsFinite", testIsFinite),
    ("testIsInfinite", testIsInfinite),
    ("testIsNaN", testIsNaN),
    ("testIsZero", testIsZero),
    ("testEquals", testEquals),
    ("testPlus", testPlus),
    ("testMinus", testMinus),
    ("testTimes", testTimes),
    ("testNegate", testNegate),
    ("testDivide", testDivide),
    ("testComplexConjugate", testComplexConjugate),
    ("testAdding", testAdding),
    ("testSubtracting", testSubtracting),
    ("testVjpInit", testVjpInit),
    ("testVjpAdd", testVjpAdd),
    ("testVjpSubtract", testVjpSubtract),
    ("testVjpMultiply", testVjpMultiply),
    ("testVjpDivide", testVjpDivide),
    ("testVjpNegate", testVjpNegate),
    ("testVjpComplexConjugate", testVjpComplexConjugate),
    ("testVjpAddingReal", testVjpAddingReal),
    ("testVjpAddingImaginary", testVjpAddingImaginary),
    ("testVjpSubtractingReal", testVjpSubtractingReal),
    ("testVjpSubtractingImaginary", testVjpSubtractingImaginary),
    ("testJvpDotProduct", testJvpDotProduct),
    ("testImplicitDifferentiation", testImplicitDifferentiation),
  ]
}
