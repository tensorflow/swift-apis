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
//
/// Note
/// ----
///
/// This implementation uses a modified implementation from the
/// xwu/NumericAnnex Swift numeric library repo vy Xiaodi Wu. To view the
/// original code, see the implementation here
///
///    https://github.com/xwu/NumericAnnex/blob/master/Sources/Complex.swift
///
/// Create new instances of `Complex<T>` using integer or floating-point
/// literals and the imaginary unit `Complex<T>.i`. For example:
///
/// ```swift
/// let x: Complex<Double> = 2 + 4 * .i
/// ```
///
/// Additional Considerations
/// -------------------------
///
/// Our implementation of complex number differentiation follows the same
/// convention as Autograd. In short, we can get the derivative of a
/// holomorphic function, functions whose codomain are the Reals, and
/// functions whose codomain and domain are the Reals. You can read more about
/// Autograd at
///
///    https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#complex-numbers
///
/// Floating-point types have special values that represent infinity or NaN
/// ("not a number"). Complex functions in different languages may return
/// different results when working with special values.

import _Differentiation

struct Complex<T: FloatingPoint> {
  var real: T
  var imaginary: T

  @differentiable(where T: Differentiable, T == T.TangentVector)
  init(real: T = 0, imaginary: T = 0) {
    self.real = real
    self.imaginary = imaginary
  }
}

extension Complex: Differentiable where T: Differentiable {
  typealias TangentVector = Complex
}

extension Complex {
  static var i: Complex {
    return Complex(real: 0, imaginary: 1)
  }

  var isFinite: Bool {
    return real.isFinite && imaginary.isFinite
  }

  var isInfinite: Bool {
    return real.isInfinite || imaginary.isInfinite
  }

  var isNaN: Bool {
    return (real.isNaN && !imaginary.isInfinite) || (imaginary.isNaN && !real.isInfinite)
  }

  var isZero: Bool {
    return real.isZero && imaginary.isZero
  }
}

extension Complex: ExpressibleByIntegerLiteral {
  init(integerLiteral value: Int) {
    self.real = T(value)
    self.imaginary = 0
  }
}

extension Complex: CustomStringConvertible {
  var description: String {
    return real.isNaN && real.sign == .minus
      ? imaginary.sign == .minus
        ? "-\(-real) - \(-imaginary)i"
        : "-\(-real) + \(imaginary)i"
      : imaginary.sign == .minus
        ? "\(real) - \(-imaginary)i"
        : "\(real) + \(imaginary)i"
  }
}

extension Complex: Equatable {
  static func == (lhs: Complex, rhs: Complex) -> Bool {
    return lhs.real == rhs.real && lhs.imaginary == rhs.imaginary
  }
}

extension Complex: AdditiveArithmetic {
  static func + (lhs: Complex, rhs: Complex) -> Complex {
    var temp = lhs
    temp += rhs
    return temp
  }

  static func += (lhs: inout Complex, rhs: Complex) {
    lhs.real += rhs.real
    lhs.imaginary += rhs.imaginary
  }

  @differentiable(where T: Differentiable)
  static func - (lhs: Complex, rhs: Complex) -> Complex {
    var temp = lhs
    temp -= rhs
    return temp
  }

  static func -= (lhs: inout Complex, rhs: Complex) {
    lhs.real -= rhs.real
    lhs.imaginary -= rhs.imaginary
  }
}

extension Complex: Numeric {
  init?<U>(exactly source: U) where U: BinaryInteger {
    guard let t = T(exactly: source) else { return nil }
    self.real = t
    self.imaginary = 0
  }

  static private func handleMultiplyNaN(infiniteA: T, infiniteB: T, nanA: T, nanB: T) -> Complex {
    var a = infiniteA
    var b = infiniteB
    var c = nanA
    var d = nanB

    a = T(signOf: infiniteA, magnitudeOf: infiniteA.isInfinite ? 1 : 0)
    b = T(signOf: infiniteB, magnitudeOf: infiniteB.isInfinite ? 1 : 0)

    if nanA.isNaN { c = T(signOf: nanA, magnitudeOf: 0) }
    if nanB.isNaN { d = T(signOf: nanB, magnitudeOf: 0) }

    return Complex(
      real: .infinity * (a * c - b * d),
      imaginary: .infinity * (a * d + b * c)
    )
  }

  @differentiable(where T: Differentiable)
  static func * (lhs: Complex, rhs: Complex) -> Complex {
    var a = lhs.real
    var b = lhs.imaginary
    var c = rhs.real
    var d = rhs.imaginary
    let ac = a * c
    let bd = b * d
    let ad = a * d
    let bc = b * c
    let x = ac - bd
    let y = ad + bc

    if x.isNaN && y.isNaN {
      if a.isInfinite || b.isInfinite {
        return handleMultiplyNaN(infiniteA: a, infiniteB: b, nanA: c, nanB: d)
      } else if c.isInfinite || d.isInfinite {
        return handleMultiplyNaN(infiniteA: c, infiniteB: d, nanA: a, nanB: b)
      } else if ac.isInfinite || bd.isInfinite || ad.isInfinite || bc.isInfinite {
        if a.isNaN { a = T(signOf: a, magnitudeOf: 0) }
        if b.isNaN { b = T(signOf: b, magnitudeOf: 0) }
        if c.isNaN { c = T(signOf: c, magnitudeOf: 0) }
        if d.isNaN { d = T(signOf: d, magnitudeOf: 0) }
        return Complex(
          real: .infinity * (a * c - b * d),
          imaginary: .infinity * (a * d + b * c)
        )
      }
    }
    return Complex(real: x, imaginary: y)
  }

  static func *= (lhs: inout Complex, rhs: Complex) {
    lhs = lhs * rhs
  }

  var magnitude: T {
    var x = abs(real)
    var y = abs(imaginary)
    if x.isInfinite { return x }
    if y.isInfinite { return y }
    if x == 0 { return y }
    if x < y { swap(&x, &y) }
    let ratio = y / x
    return x * (1 + ratio * ratio).squareRoot()
  }
}

extension Complex: SignedNumeric {
  @differentiable(where T: Differentiable)
  static prefix func - (operand: Complex) -> Complex {
    return Complex(real: -operand.real, imaginary: -operand.imaginary)
  }

  mutating func negate() {
    real.negate()
    imaginary.negate()
  }
}

extension Complex {
  @differentiable(where T: Differentiable)
  static func / (lhs: Complex, rhs: Complex) -> Complex {
    var a = lhs.real
    var b = lhs.imaginary
    var c = rhs.real
    var d = rhs.imaginary
    var x: T
    var y: T
    if c.magnitude >= d.magnitude {
      let ratio = d / c
      let denominator = c + d * ratio
      x = (a + b * ratio) / denominator
      y = (b - a * ratio) / denominator
    } else {
      let ratio = c / d
      let denominator = c * ratio + d
      x = (a * ratio + b) / denominator
      y = (b * ratio - a) / denominator
    }
    if x.isNaN && y.isNaN {
      if c == 0 && d == 0 && (!a.isNaN || !b.isNaN) {
        x = T(signOf: c, magnitudeOf: .infinity) * a
        y = T(signOf: c, magnitudeOf: .infinity) * b
      } else if (a.isInfinite || b.isInfinite) && c.isFinite && d.isFinite {
        a = T(signOf: a, magnitudeOf: a.isInfinite ? 1 : 0)
        b = T(signOf: b, magnitudeOf: b.isInfinite ? 1 : 0)
        x = .infinity * (a * c + b * d)
        y = .infinity * (b * c - a * d)
      } else if (c.isInfinite || d.isInfinite) && a.isFinite && b.isFinite {
        c = T(signOf: c, magnitudeOf: c.isInfinite ? 1 : 0)
        d = T(signOf: d, magnitudeOf: d.isInfinite ? 1 : 0)
        x = 0 * (a * c + b * d)
        y = 0 * (b * c - a * d)
      }
    }
    return Complex(real: x, imaginary: y)
  }

  static func /= (lhs: inout Complex, rhs: Complex) {
    lhs = lhs / rhs
  }
}

extension Complex {
  @differentiable(where T: Differentiable)
  func complexConjugate() -> Complex {
    return Complex(real: real, imaginary: -imaginary)
  }
}

func abs<T>(_ z: Complex<T>) -> Complex<T> {
  return Complex(real: z.magnitude)
}

extension Complex {
  @differentiable(where T: Differentiable, T == T.TangentVector)
  func adding(real: T) -> Complex {
    var c = self
    c.real += real
    return c
  }

  @differentiable(where T: Differentiable, T == T.TangentVector)
  func subtracting(real: T) -> Complex {
    var c = self
    c.real -= real
    return c
  }

  @differentiable(where T: Differentiable, T == T.TangentVector)
  func adding(imaginary: T) -> Complex {
    var c = self
    c.imaginary += imaginary
    return c
  }

  @differentiable(where T: Differentiable, T == T.TangentVector)
  func subtracting(imaginary: T) -> Complex {
    var c = self
    c.imaginary -= imaginary
    return c
  }
}

extension Complex where T: Differentiable, T.TangentVector == T {
  @derivative(of: init(real:imaginary:))
  static func _vjpInit(real: T, imaginary: T) -> (value: Complex, pullback: (Complex) -> (T, T)) {
    return (Complex(real: real, imaginary: imaginary), { ($0.real, $0.imaginary) })
  }
}

extension Complex where T: Differentiable {
  @derivative(of: +)
  static func _vjpAdd(lhs: Complex, rhs: Complex)
    -> (value: Complex, pullback: (Complex) -> (Complex, Complex))
  {
    return (lhs + rhs, { v in (v, v) })
  }

  @derivative(of: -)
  static func _vjpSubtract(lhs: Complex, rhs: Complex)
    -> (value: Complex, pullback: (Complex) -> (Complex, Complex))
  {
    return (lhs - rhs, { v in (v, -v) })
  }

  @derivative(of: *)
  static func _vjpMultiply(lhs: Complex, rhs: Complex)
    -> (value: Complex, pullback: (Complex) -> (Complex, Complex))
  {
    return (lhs * rhs, { v in (rhs * v, lhs * v) })
  }

  @derivative(of: /)
  static func _vjpDivide(lhs: Complex, rhs: Complex)
    -> (value: Complex, pullback: (Complex) -> (Complex, Complex))
  {
    return (lhs / rhs, { v in (v / rhs, -lhs / (rhs * rhs) * v) })
  }

  @derivative(of: -)
  static func _vjpNegate(operand: Complex) -> (value: Complex, pullback: (Complex) -> Complex) {
    return (-operand, { -$0 })
  }

  @derivative(of: complexConjugate)
  func _vjpComplexConjugate() -> (value: Complex, pullback: (Complex) -> Complex) {
    return (complexConjugate(), { v in v.complexConjugate() })
  }
}

extension Complex where T: Differentiable, T.TangentVector == T {
  @derivative(of: adding(real:))
  func _vjpAdding(real: T) -> (value: Complex, pullback: (Complex) -> (Complex, T)) {
    return (self.adding(real: real), { ($0, $0.real) })
  }

  @derivative(of: subtracting(real:))
  func _vjpSubtracting(real: T) -> (value: Complex, pullback: (Complex) -> (Complex, T)) {
    return (self.subtracting(real: real), { ($0, -$0.real) })
  }

  @derivative(of: adding(imaginary:))
  func _vjpAdding(imaginary: T) -> (value: Complex, pullback: (Complex) -> (Complex, T)) {
    return (self.adding(real: real), { ($0, $0.imaginary) })
  }

  @derivative(of: subtracting(imaginary:))
  func _vjpSubtracting(imaginary: T) -> (value: Complex, pullback: (Complex) -> (Complex, T)) {
    return (self.subtracting(real: real), { ($0, -$0.imaginary) })
  }
}
