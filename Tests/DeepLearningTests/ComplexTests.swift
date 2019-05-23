// RUN: %target-run-simple-swift
// REQUIRES: executable_test
//
// Complex API tests.

// TODO: remove import
import TensorFlow

import StdlibUnittest

var ComplexTests = TestSuite("Complex")

ComplexTests.test("Initializer") {
  let complex = Complex<Float>(real: 2, imaginary: 3)
  expectEqual(complex.real, 2)
  expectEqual(complex.imaginary, 3)
}

ComplexTests.test("Static Imaginary") {
  let imaginary = Complex<Float>(real: 0, imaginary: 1)
  expectEqual(imaginary, Complex.i)
}

ComplexTests.test("isFinite") {
  var complex = Complex<Float>(real: 999, imaginary: 0)
  expectTrue(complex.isFinite)

  complex = Complex(real: 1.0 / 0.0, imaginary: 1)
  expectFalse(complex.isFinite)

  complex = Complex(real: 1.0 / 0.0, imaginary: 1.0 / 0.0)
  expectFalse(complex.isFinite)
}

ComplexTests.test("isInfinite") {
  var complex = Complex<Float>(real: 999, imaginary: 0)
  expectFalse(complex.isInfinite)

  complex = Complex(real: 1.0 / 0.0, imaginary: 1)
  expectTrue(complex.isInfinite)

  complex = Complex(real: 1.0 / 0.0, imaginary: 1.0 / 0.0)
  expectTrue(complex.isInfinite)
}

ComplexTests.test("isNaN") {
  var complex = Complex<Float>(real: 999, imaginary: 0)
  expectFalse(complex.isNaN)

  complex = Complex(real: 0.0 * 1.0 / 0.0, imaginary: 1)
  expectTrue(complex.isNaN)

  complex = Complex(real: 0.0 * 1.0 / 0.0, imaginary: 0.0 * 1.0 / 0.0)
  expectTrue(complex.isNaN)
}

ComplexTests.test("isZero") {
  var complex = Complex<Float>(real: 999, imaginary: 0)
  expectFalse(complex.isZero)

  complex = Complex(real: 0.0 * 1.0 / 0.0, imaginary: 0)
  expectFalse(complex.isZero)

  complex = Complex(real: 0.0 * 1.0 / 0.0, imaginary: 0.0 * 1.0 / 0.0)
  expectFalse(complex.isZero)

  complex = Complex(real: 0, imaginary: 0)
  expectTrue(complex.isZero)
}

ComplexTests.test("==") {
  var complexA = Complex<Float>(real: 999, imaginary: 0)
  let complexB = Complex<Float>(real: 999, imaginary: 0)
  expectEqual(complexA, complexB)

  complexA = Complex(real: 5, imaginary: 0)
  expectNotEqual(complexA, complexB)
}

ComplexTests.test("+") {
  let input = Complex<Float>(real: 5, imaginary: 1)
  let expected = Complex<Float>(real: 10, imaginary: 2)
  expectEqual(expected, input + input)
}

ComplexTests.test("-") {
  let inputA = Complex<Float>(real: 6, imaginary: 2)
  let inputB = Complex<Float>(real: 5, imaginary: 1)
  let expected = Complex<Float>(real: 1, imaginary: 1)
  expectEqual(expected, inputA - inputB)
}

ComplexTests.test("*") {
  let inputA = Complex<Float>(real: 6, imaginary: 2)
  let inputB = Complex<Float>(real: 5, imaginary: 1)
  let expected = Complex<Float>(real: 28, imaginary: 16)
  expectEqual(expected, inputA * inputB)
}

ComplexTests.test("negate") {
  var input = Complex<Float>(real: 6, imaginary: 2)
  let negated = Complex<Float>(real: -6, imaginary: -2)
  expectEqual(-input, negated)
  input.negate()
  expectEqual(input, negated)
}

ComplexTests.test("/") {
  let inputA = Complex<Float>(real: 20, imaginary: -4)
  let inputB = Complex<Float>(real: 3, imaginary: 2)
  let expected = Complex<Float>(real: 4, imaginary: -4)
  expectEqual(expected, inputA / inputB)
}

ComplexTests.test("complexConjugate") {
  var input = Complex<Float>(real: 2, imaginary: -4)
  var expected = Complex<Float>(real: 2, imaginary: 4)
  expectEqual(expected, input.complexConjugate())

  input = Complex<Float>(real: -2, imaginary: -4)
  expected = Complex<Float>(real: -2, imaginary: 4)
  expectEqual(expected, input.complexConjugate())

  input = Complex<Float>(real: 2, imaginary: 4)
  expected = Complex<Float>(real: 2, imaginary: -4)
  expectEqual(expected, input.complexConjugate())
}

ComplexTests.test("adding") {
  var input = Complex<Float>(real: 2, imaginary: -4)
  var expected = Complex<Float>(real: 3, imaginary: -4)
  expectEqual(expected, input.adding(real: 1))

  input = Complex<Float>(real: 2, imaginary: -4)
  expected = Complex<Float>(real: 2, imaginary: -3)
  expectEqual(expected, input.adding(imaginary: 1))
}

ComplexTests.test("subtracting") {
  var input = Complex<Float>(real: 2, imaginary: -4)
  var expected = Complex<Float>(real: 1, imaginary: -4)
  expectEqual(expected, input.subtracting(real: 1))

  input = Complex<Float>(real: 2, imaginary: -4)
  expected = Complex<Float>(real: 2, imaginary: -5)
  expectEqual(expected, input.subtracting(imaginary: 1))
}

runAllTests()