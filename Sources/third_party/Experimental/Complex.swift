//
//  Complex.swift
//  NumericAnnex
//
//  Created by Xiaodi Wu on 3/25/17.
//
//  Note
//  ====
//
//  For maximum consistency with corresponding functions in C/C++, checks for
//  special values in `naturalExponential()`, `squareRoot()`, trigonometric
//  functions, and hyperbolic functions are adapted from libc++.
//
//  Code in libc++ is dual-licensed under the MIT and UIUC/NCSA licenses.
//  Copyright © 2009-2017 contributors to the LLVM/libc++ project, Google LLC.
/// A type to represent a complex value in Cartesian form.
///
/// - Note: `Complex64` is a type alias for `Complex<Float>` and `Complex128` is
///   a type alias for `Complex<Double>`.
///
/// Create new instances of `Complex<T>` using integer or floating-point
/// literals and the imaginary unit `Complex<T>.i`. For example:
///
/// ```swift
/// let x = 2 + 4 * .i // `x` is of type `Complex<Double>`
/// let y = 3.5 + 7 * .i // `y` is of type `Complex<Double>`
///
/// let z: Complex64 = .e + .pi * .i // `z` is of type `Complex<Float>`
/// ```
///
/// Additional Considerations
/// -------------------------
///
/// Floating-point types have special values that represent infinity or NaN
/// ("not a number"). Complex functions in different languages may return
/// different results when working with special values.
///
/// Many complex functions have [branch cuts][dfn], which are curves in the
/// complex plane across which a function is discontinuous. Different languages
/// may adopt different branch cut structures for the same complex function.
///
/// Implementations in `Complex<T>` adhere to the [C standard][std] (Annex G) as
/// closely as possible with respect to special values and branch cuts.
///
/// To users unfamiliar with complex functions, the principal value returned by
/// some complex functions may be unexpected. For example,
/// `Double.cbrt(-8) == -2`, which is the __real root__, while
/// `Complex.cbrt(-8) == 2 * Complex.exp(.i * .pi / 3)`, which is the
/// __principal root__.
///
/// [dfn]: http://mathworld.wolfram.com/BranchCut.html
/// [std]: http://www.open-std.org/JTC1/SC22/WG14/www/standards.html#9899

struct Complex<T: FloatingPoint> {
    var real: T
    var imaginary: T

    @differentiable(vjp: _vjpInit where T: Differentiable, T.TangentVector == T)
    init(real: T = 0, imaginary: T = 0) {
        self.real = real
        self.imaginary = imaginary
    }
}

extension Complex: Differentiable where T: Differentiable {
    typealias TangentVector = Complex
    typealias AllDifferentiableVariables = Complex
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
        return (real.isNaN && !imaginary.isInfinite) ||
        (imaginary.isNaN && !real.isInfinite)
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
    @differentiable(vjp: _vjpAdd(lhs:rhs:) where T: Differentiable)
    static func + (lhs: Complex, rhs: Complex) -> Complex {
        var temp = lhs
        temp += rhs
        return temp
    }

    static func += (lhs: inout Complex, rhs: Complex) {
        lhs.real += rhs.real
        lhs.imaginary += rhs.imaginary
    }

    @differentiable(vjp: _vjpSubtract(lhs:rhs:) where T: Differentiable)
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

    @differentiable(vjp: _vjpMultiply(lhs:rhs:) where T: Differentiable)
    static func * (lhs: Complex, rhs: Complex) -> Complex {
        var a = lhs.real, b = lhs.imaginary, c = rhs.real, d = rhs.imaginary
        let ac = a * c, bd = b * d, ad = a * d, bc = b * c
        let x = ac - bd
        let y = ad + bc

        if x.isNaN && y.isNaN {
            var recalculate = false
            if a.isInfinite || b.isInfinite {
                a = T(signOf: a, magnitudeOf: a.isInfinite ? 1 : 0)
                b = T(signOf: b, magnitudeOf: b.isInfinite ? 1 : 0)
                if c.isNaN { c = T(signOf: c, magnitudeOf: 0) }
                if d.isNaN { d = T(signOf: d, magnitudeOf: 0) }
                recalculate = true
            }
            if c.isInfinite || d.isInfinite {
                if a.isNaN { a = T(signOf: a, magnitudeOf: 0) }
                if b.isNaN { b = T(signOf: b, magnitudeOf: 0) }
                c = T(signOf: c, magnitudeOf: c.isInfinite ? 1 : 0)
                d = T(signOf: d, magnitudeOf: d.isInfinite ? 1 : 0)
                recalculate = true
            }
            if !recalculate &&
                (ac.isInfinite || bd.isInfinite || ad.isInfinite || bc.isInfinite) {
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
    @differentiable(vjp: _vjpNegate where T: Differentiable)
    static prefix func - (operand: Complex) -> Complex {
        return Complex(real: -operand.real, imaginary: -operand.imaginary)
    }

    mutating func negate() {
        real.negate()
        imaginary.negate()
    }
}

extension Complex {
    @differentiable(vjp: _vjpDivide(lhs:rhs:) where T: Differentiable)
    static func / (lhs: Complex, rhs: Complex) -> Complex {
        var a = lhs.real, b = lhs.imaginary, c = rhs.real, d = rhs.imaginary
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
    @differentiable(vjp: _vjpComplexConjugate where T: Differentiable)
    func complexConjugate() -> Complex {
        return Complex(real: real, imaginary: -imaginary)
    }
}

func abs<T>(_ z: Complex<T>) -> Complex<T> {
    return Complex(real: z.magnitude)
}

extension Complex {
    @differentiable(vjp: _vjpAdding(real:) where T: Differentiable, T.TangentVector == T)
    func adding(real: T) -> Complex {
        var c = self
        c.real += real
        return c
    }

    @differentiable(vjp: _vjpSubtracting(real:) where T: Differentiable, T.TangentVector == T)
    func subtracting(real: T) -> Complex {
        var c = self
        c.real -= real
        return c
    }

    @differentiable(vjp: _vjpAdding(imaginary:) where T: Differentiable, T.TangentVector == T)
    func adding(imaginary: T) -> Complex {
        var c = self
        c.imaginary += imaginary
        return c
    }

    @differentiable(vjp: _vjpSubtracting(imaginary:) where T: Differentiable, T.TangentVector == T)
    func subtracting(imaginary: T) -> Complex {
        var c = self
        c.imaginary -= imaginary
        return c
    }
}

extension Complex where T: Differentiable, T.TangentVector == T {
    static func _vjpInit(real: T, imaginary: T) -> (Complex, (Complex) -> (T, T)) {
        return (Complex(real: real, imaginary: imaginary), { ($0.real, $0.imaginary) })
    }
}

extension Complex where T: Differentiable {
    static func _vjpAdd(lhs: Complex, rhs: Complex)
        -> (Complex, (Complex) -> (Complex, Complex)) {
        return (lhs + rhs, { v in (v, v) })
    }

    static func _vjpSubtract(lhs: Complex, rhs: Complex)
        -> (Complex, (Complex) -> (Complex, Complex)) {
        return (lhs - rhs, { v in (v, -v) })
    }

    static func _vjpMultiply(lhs: Complex, rhs: Complex)
        -> (Complex, (Complex) -> (Complex, Complex)) {
        return (lhs * rhs, { v in (rhs * v, lhs * v) })
    }

    static func _vjpDivide(lhs: Complex, rhs: Complex)
        -> (Complex, (Complex) -> (Complex, Complex)) {
        return (lhs / rhs, { v in (v / rhs, -lhs / (rhs * rhs) * v) })
    }

    static func _vjpNegate(operand: Complex)
        -> (Complex, (Complex) -> Complex) {
        return (-operand, { -$0 })
    }

    func _vjpComplexConjugate() -> (Complex, (Complex) -> Complex) {
        return (complexConjugate(), { v in v.complexConjugate() })
    }
}

extension Complex where T: Differentiable, T.TangentVector == T {
    func _vjpAdding(real: T) -> (Complex, (Complex) -> (Complex, T)) {
        return (self.adding(real: real), { ($0, $0.real) })
    }

    func _vjpSubtracting(real: T) -> (Complex, (Complex) -> (Complex, T)) {
        return (self.subtracting(real: real), { ($0, -$0.real) })
    }

    func _vjpAdding(imaginary: T) -> (Complex, (Complex) -> (Complex, T)) {
        return (self.adding(real: real), { ($0, $0.imaginary) })
    }

    func _vjpSubtracting(imaginary: T) -> (Complex, (Complex) -> (Complex, T)) {
        return (self.subtracting(real: real), { ($0, -$0.imaginary) })
    }
}
