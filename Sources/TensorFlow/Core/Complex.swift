import TensorFlow
// T : FloatingPoint & Differentiable
public struct Complex<T : FloatingPoint> {
  // ---------------------------------------------------------------------------
  // MARK: Stored Properties
  // ---------------------------------------------------------------------------

  /// The real component of the complex value.
  public var real: T

  /// The imaginary component of the complex value.
  public var imaginary: T

  // ---------------------------------------------------------------------------
  // MARK: Initializers
  // ---------------------------------------------------------------------------
  public init(real: T = 0, imaginary: T = 0) {
	self.real = real
	self.imaginary = imaginary
  }
}

extension Complex : Differentiable where T : Differentiable, T.TangentVector == T {
  // ---------------------------------------------------------------------------
  // MARK: Differentiability
  // ---------------------------------------------------------------------------
  public typealias TangentVector = Complex
  public typealias AllDifferentiableVariables = Complex
}

extension Complex {
  // ---------------------------------------------------------------------------
  // MARK: Static Properties
  // ---------------------------------------------------------------------------

  /// The imaginary unit _i_.
  @inlinable
  public static var i: Complex {
    return Complex(real: 0, imaginary: 1)
  }

  /// A Boolean value indicating whether the instance is finite.
  ///
  /// A complex value is finite if its real and imaginary components are both
  /// finite. A component is finite if it is not infinity or NaN.
  @inlinable
  public var isFinite: Bool {
    return real.isFinite && imaginary.isFinite
  }

  /// A Boolean value indicating whether the instance is infinite.
  ///
  /// A complex value is infinite if at least one of its components (real or
  /// imaginary) is infinite, even if the other component is NaN.
  ///
  /// Note that `isFinite` and `isInfinite` do not form a dichotomy because NaN
  /// is neither finite nor infinite.
  @inlinable
  public var isInfinite: Bool {
    return real.isInfinite || imaginary.isInfinite
  }

  /// A Boolean value indicating whether the instance is NaN ("not a number").
  ///
  /// A complex value is NaN if at least one of its components (real or
  /// imaginary) is NaN and the other component is not infinite.
  ///
  /// Because NaN is not equal to any value, including NaN, use this property
  /// instead of the equal-to operator (`==`) or not-equal-to operator (`!=`) to
  /// test whether a value is or is not NaN.
  ///
  /// This property is `true` for both quiet and signaling NaNs.
  @inlinable
  public var isNaN: Bool {
    return (real.isNaN && !imaginary.isInfinite) ||
      (imaginary.isNaN && !real.isInfinite)
  }

  /// A Boolean value indicating whether the instance is equal to zero.
  ///
  /// A complex value is equal to zero if its real and imaginary components both
  /// represent either `-0.0` or `+0.0`.
  @inlinable
  public var isZero: Bool {
    return real.isZero && imaginary.isZero
  }
}

extension Complex : ExpressibleByIntegerLiteral {
  // ---------------------------------------------------------------------------
  // MARK: ExpressibleByIntegerLiteral
  // ---------------------------------------------------------------------------

  @inlinable
  public init(integerLiteral value: Int) {
    self.real = T(value)
    self.imaginary = 0
  }
}

extension Complex : CustomStringConvertible {
  // ---------------------------------------------------------------------------
  // MARK: CustomStringConvertible
  // ---------------------------------------------------------------------------

  @inlinable
  public var description: String {
    return real.isNaN && real.sign == .minus
      // At present, -NaN is described as "nan", which is acceptable for real
      // values. However, it is arguably misleading to describe -NaN - NaNi as
      // "nan + nani" or "nan - nani". Therefore, handle this case separately.
      ? imaginary.sign == .minus
        ? "-\(-real) - \(-imaginary)i"
        : "-\(-real) + \(imaginary)i"
      : imaginary.sign == .minus
        ? "\(real) - \(-imaginary)i"
        : "\(real) + \(imaginary)i"
  }
}

extension Complex : Equatable {
  // ---------------------------------------------------------------------------
  // MARK: Equatable
  // ---------------------------------------------------------------------------

  @inlinable
  public static func == (lhs: Complex, rhs: Complex) -> Bool {
    return lhs.real == rhs.real && lhs.imaginary == rhs.imaginary
  }
}

extension Complex : AdditiveArithmetic {
  // ---------------------------------------------------------------------------
  // MARK: AdditiveArithmetic
  // ---------------------------------------------------------------------------

  @inlinable
  @differentiable(vjp: _vjpAdd(lhs:rhs:) where T : Differentiable, T.TangentVector == T)
  public static func + (lhs: Complex, rhs: Complex) -> Complex {
    var lhs = lhs
    lhs += rhs
    return lhs
  }

  @inlinable
  public static func += (lhs: inout Complex, rhs: Complex) {
    lhs.real += rhs.real
    lhs.imaginary += rhs.imaginary
  }

  @inlinable
  @differentiable(vjp: _vjpSubtract(lhs:rhs:) where T : Differentiable, T.TangentVector == T)
  public static func - (lhs: Complex, rhs: Complex) -> Complex {
    var lhs = lhs
    lhs -= rhs
    return lhs
  }

  @inlinable
  public static func -= (lhs: inout Complex, rhs: Complex) {
    lhs.real -= rhs.real
    lhs.imaginary -= rhs.imaginary
  }
}

extension Complex : Numeric {
  // ---------------------------------------------------------------------------
  // MARK: Numeric
  // ---------------------------------------------------------------------------

  public init?<U>(exactly source: U) where U : BinaryInteger {
    guard let t = T(exactly: source) else { return nil }
    self.real = t
    self.imaginary = 0
  }

  @inlinable
  @differentiable(vjp: _vjpMultiply(lhs:rhs:) where T : Differentiable, T.TangentVector == T)
  public static func * (lhs: Complex, rhs: Complex) -> Complex {
    var a = lhs.real, b = lhs.imaginary, c = rhs.real, d = rhs.imaginary
    let ac = a * c, bd = b * d, ad = a * d, bc = b * c
    let x = ac - bd
    let y = ad + bc
    // Recover infinities that computed as NaN + iNaN.
    // See C11 Annex G.
    if x.isNaN && y.isNaN {
      var recalculate = false
      if a.isInfinite || b.isInfinite {
        // "Box" the infinity and change NaNs in the other operand to 0.
        a = T(signOf: a, magnitudeOf: a.isInfinite ? 1 : 0)
        b = T(signOf: b, magnitudeOf: b.isInfinite ? 1 : 0)
        if c.isNaN { c = T(signOf: c, magnitudeOf: 0) }
        if d.isNaN { d = T(signOf: d, magnitudeOf: 0) }
        recalculate = true
      }
      if c.isInfinite || d.isInfinite {
        // "Box" the infinity and change NaNs in the other operand to 0.
        if a.isNaN { a = T(signOf: a, magnitudeOf: 0) }
        if b.isNaN { b = T(signOf: b, magnitudeOf: 0) }
        c = T(signOf: c, magnitudeOf: c.isInfinite ? 1 : 0)
        d = T(signOf: d, magnitudeOf: d.isInfinite ? 1 : 0)
        recalculate = true
      }
      if !recalculate &&
        (ac.isInfinite || bd.isInfinite || ad.isInfinite || bc.isInfinite) {
        // Recover infinities from overflow by changing NaNs to 0.
        if a.isNaN { a = T(signOf: a, magnitudeOf: 0) }
        if b.isNaN { b = T(signOf: b, magnitudeOf: 0) }
        if c.isNaN { c = T(signOf: c, magnitudeOf: 0) }
        if d.isNaN { d = T(signOf: d, magnitudeOf: 0) }
        recalculate = true
      }
      if recalculate {
        return Complex(
          real: .infinity * (a * c - b * d),
          imaginary: .infinity * (a * d + b * c)
        )
      }
    }
    return Complex(real: x, imaginary: y)
  }

  @inlinable
  public static func *= (lhs: inout Complex, rhs: Complex) {
    lhs = lhs * rhs
  }

  @inlinable
  public var magnitude: T {
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

extension Complex : SignedNumeric {
  // ---------------------------------------------------------------------------
  // MARK: SignedNumeric
  // ---------------------------------------------------------------------------

  @inlinable
  @differentiable(vjp: _vjpNegate where T : Differentiable, T.TangentVector == T)
  public static prefix func - (operand: Complex) -> Complex {
    return Complex(real: -operand.real, imaginary: -operand.imaginary)
  }

  @inlinable
  public mutating func negate() {
    real.negate()
    imaginary.negate()
  }
}

extension Complex {
  // ---------------------------------------------------------------------------
  // MARK: Division
  // ---------------------------------------------------------------------------

  @inlinable
  @differentiable(vjp: _vjpDivide(lhs:rhs:) where T : Differentiable, T.TangentVector == T)
  public static func / (lhs: Complex, rhs: Complex) -> Complex {
    var a = lhs.real, b = lhs.imaginary, c = rhs.real, d = rhs.imaginary
    var x: T
    var y: T
    // Prevent avoidable overflow; see Numerical Recipes.
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
    // Recover infinities and zeros that computed as NaN + iNaN.
    // See C11 Annex G.
    if x.isNaN && y.isNaN {
      if c == 0 && d == 0 && (!a.isNaN || !b.isNaN) {
        x = T(signOf: c, magnitudeOf: .infinity) * a
        y = T(signOf: c /* sic */, magnitudeOf: .infinity) * b
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

  @inlinable
  public static func /= (lhs: inout Complex, rhs: Complex) {
    lhs = lhs / rhs
  }
}

extension Complex {
  @inlinable
  public func complexConjugate() -> Complex {
    return Complex(real: real, imaginary: -imaginary)
  }
}

/// Returns the absolute value (magnitude, modulus) of `z`.
@inlinable
public func abs<T>(_ z: Complex<T>) -> Complex<T> {
  return Complex(real: z.magnitude)
}

extension Complex {
  @inlinable
  @differentiable(vjp: _vjpAdding(real:) where T : Differentiable, T.TangentVector == T)
  public func adding(real: T) -> Complex {
    var c = self
    c.real += real
    return c
  }

  @inlinable
  @differentiable(vjp: _vjpSubtracting(real:) where T : Differentiable, T.TangentVector == T)
  public func subtracting(real: T) -> Complex {
    var c = self
    c.real -= real
    return c
  }

  @inlinable
  @differentiable(vjp: _vjpAdding(imaginary:) where T : Differentiable, T.TangentVector == T)
  public func adding(imaginary: T) -> Complex {
    var c = self
    c.imaginary += imaginary
    return c
  }
  
  @inlinable
  @differentiable(vjp: _vjpSubtracting(imaginary:) where T : Differentiable, T.TangentVector == T)
  public func subtracting(imaginary: T) -> Complex {
    var c = self
    c.imaginary -= imaginary
    return c
  }
}

extension Complex where T : Differentiable, T.TangentVector == T {
  @inlinable
  static func _vjpAdd(lhs: Complex, rhs: Complex) 
  -> (Complex, (Complex) -> (Complex, Complex)) {
    return (lhs * rhs, { v in (v, v) })
  }

  @inlinable
  static func _vjpSubtract(lhs: Complex, rhs: Complex) 
  -> (Complex, (Complex) -> (Complex, Complex)) {
    return (lhs * rhs, { v in (v, -v) })
  }

  @inlinable
  static func _vjpMultiply(lhs: Complex, rhs: Complex) 
  -> (Complex, (Complex) -> (Complex, Complex)) {
    return (lhs * rhs, { v in (rhs * v, lhs * v) })
  }

  @inlinable
  static func _vjpDivide(lhs: Complex, rhs: Complex) 
  -> (Complex, (Complex) -> (Complex, Complex)) {
    return (lhs * rhs, { v in (v / rhs, -lhs / (rhs * rhs) * v) })
  }

  @inlinable
  static func _vjpNegate(operand: Complex)
  -> (Complex, (Complex) -> Complex) {
    return (-operand, { v in -v})
  }

  @inlinable
  func _vjpAdding(real: T) -> (Complex, (Complex) -> (Complex, T)) {
    return (self.adding(real: real), { ($0, $0.real) })
  }

  @inlinable
  func _vjpSubtracting(real: T) -> (Complex, (Complex) -> (Complex, T)) {
    return (self.subtracting(real: real), { ($0, -$0.real) })
  }

  @inlinable
  func _vjpAdding(imaginary: T) -> (Complex, (Complex) -> (Complex, T)) {
    return (self.adding(real: real), { ($0, $0.imaginary) })
  }

  @inlinable
  func _vjpSubtracting(imaginary: T) -> (Complex, (Complex) -> (Complex, T)) {
    return (self.subtracting(real: real), { ($0, -$0.imaginary) })
  }
}