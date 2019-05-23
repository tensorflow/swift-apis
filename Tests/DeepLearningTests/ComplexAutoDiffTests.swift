// RUN: %target-run-simple-swift
// REQUIRES: executable_test
//
// Complex API tests.

// TODO: remove import
import TensorFlow

import StdlibUnittest

var AutoDiffComplexTests = TestSuite("AutoDiffComplex")

AutoDiffComplexTests.test("_vjpAdd") {
  let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 2, imaginary: 3)) { x in
    return x + Complex<Float>(real: 5, imaginary: 6)
  }
  expectEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
}

AutoDiffComplexTests.test("_vjpSubtract") {
  let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 2, imaginary: 3)) { x in
    return Complex<Float>(real: 5, imaginary: 6) - x
  }
  expectEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: -1, imaginary: -1))
}

AutoDiffComplexTests.test("_vjpMultiply") {
  let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 2, imaginary: 3)) { x in
    return x * x
  }
  expectEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 4, imaginary: 6))
  expectEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: -6, imaginary: 4))
  expectEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: -2, imaginary: 10))
}

AutoDiffComplexTests.test("_vjpDivide") {
  let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
    return x / Complex<Float>(real: 2, imaginary: 2)
  }
  expectEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 0.25, imaginary: -0.25))
  expectEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0.25, imaginary: 0.25))
}

AutoDiffComplexTests.test("_vjpNegate") {
  let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
    return -x
  }
  expectEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: -1, imaginary: 0))
  expectEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: -1))
  expectEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: -1, imaginary: -1))
}

AutoDiffComplexTests.test("_vjpComplexConjugate") {
  let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
    return x.complexConjugate()
  }
  expectEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: -1, imaginary: 0))
  expectEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: -1))
  expectEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: -1, imaginary: -1))
}

AutoDiffComplexTests.test("_vjpAdding(real:)") {
  let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
    return x.adding(real: 5)
  }
  expectEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 1, imaginary: 0))
  expectEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: 1))
  expectEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
}

AutoDiffComplexTests.test("_vjpAdding(imaginary:)") {
  let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
    return x.adding(imaginary: 5)
  }
  expectEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 1, imaginary: 0))
  expectEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: 1))
  expectEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
}

AutoDiffComplexTests.test("_vjpSubtracting(real:)") {
  let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
    return x.subtracting(real: 5)
  }
  expectEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 1, imaginary: 0))
  expectEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: 1))
  expectEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
}

AutoDiffComplexTests.test("_vjpSubtracting(imaginary:)") {
  let pb: (Complex<Float>) -> Complex<Float> = pullback(at: Complex<Float>(real: 20, imaginary: -4)) { x in
    return x.subtracting(imaginary: 5)
  }
  expectEqual(pb(Complex(real: 1, imaginary: 0)), Complex<Float>(real: 1, imaginary: 0))
  expectEqual(pb(Complex(real: 0, imaginary: 1)), Complex<Float>(real: 0, imaginary: 1))
  expectEqual(pb(Complex(real: 1, imaginary: 1)), Complex<Float>(real: 1, imaginary: 1))
}

runAllTests()