// RUN: %target-run-simple-swift
// REQUIRES: executable_test
//
// Complex API tests.

import XCTest
@testable import DeepLearning

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
    expectNotEqual(complexA, complexB)
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

  func testVjpAdd() {
    let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 2, imaginary: 3)) { x in
      return x + Complex<Float>(real: 5, imaginary: 6)
    }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
  }

  func testVjpSubtract() {
    let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 2, imaginary: 3)) { x in
      return Complex<Float>(real: 5, imaginary: 6) - x
    }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: -1, imaginary: -1))
  }

  func testVjpMultiply() {
    let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 2, imaginary: 3)) { x in
      return x * x
    }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 4, imaginary: 6))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: -6, imaginary: 4))
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: -2, imaginary: 10))
  }

  func testVjpDivide() {
    let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
      return x / Complex<Float>(real: 2, imaginary: 2)
    }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 0.25, imaginary: -0.25))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0.25, imaginary: 0.25))
  }

  func testVjpNegate() {
    let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
      return -x
    }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: -1, imaginary: 0))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: -1))
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: -1, imaginary: -1))
  }

  func testVjpComplexConjugate() {
    let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
      return x.complexConjugate()
    }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: -1, imaginary: 0))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: -1))
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: -1, imaginary: -1))
  }

  func testVjpAdding() {
    let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
      return x.adding(real: 5)
    }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 1, imaginary: 0))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: 1))
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
  }

  func testVjpAdding() {
    let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
      return x.adding(imaginary: 5)
    }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 1, imaginary: 0))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: 1))
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
  }

  func testVjpSubtracting() {
    let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
      return x.subtracting(real: 5)
    }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 1, imaginary: 0))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: 1))
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
  }

  func testVjpSubtracting() {
    let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
      return x.subtracting(imaginary: 5)
    }
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 1, imaginary: 0))
    XCTAssertEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: 1))
    XCTAssertEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
  }
}