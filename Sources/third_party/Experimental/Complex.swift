// A major part of the Complex number implementation comes from
// [@xwu](https://github.com/xwu)'s project, [NumericAnnex](https://github.com/xwu/NumericAnnex).

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
    @differentiable(vjp: _vjpAdding(real:)
        where T: Differentiable,
              T.TangentVector == T)
    func adding(real: T) -> Complex {
        var c = self
        c.real += real
        return c
    }

    @differentiable(vjp: _vjpSubtracting(real:)
        where T: Differentiable,
              T.TangentVector == T)
    func subtracting(real: T) -> Complex {
        var c = self
        c.real -= real
        return c
    }

    @differentiable(vjp: _vjpAdding(imaginary:)
        where T: Differentiable,
              T.TangentVector == T)
    func adding(imaginary: T) -> Complex {
        var c = self
        c.imaginary += imaginary
        return c
    }

    @differentiable(vjp: _vjpSubtracting(imaginary:)
        where T: Differentiable,
              T.TangentVector == T)
    func subtracting(imaginary: T) -> Complex {
        var c = self
        c.imaginary -= imaginary
        return c
    }
}

extension Complex where T: Differentiable, T.TangentVector == T {
    @usableFromInline
    static func _vjpInit(real: T, imaginary: T) -> (Complex, (Complex) -> (T, T)) {
        return (Complex(real: real, imaginary: imaginary), { ($0.real, $0.imaginary) })
    }
}

extension Complex where T: Differentiable {
    @usableFromInline
    static func _vjpAdd(lhs: Complex, rhs: Complex)
    -> (Complex, (Complex) -> (Complex, Complex)) {
        return (lhs + rhs, { v in (v, v) })
    }

    @usableFromInline
    static func _vjpSubtract(lhs: Complex, rhs: Complex)
    -> (Complex, (Complex) -> (Complex, Complex)) {
        return (lhs - rhs, { v in (v, -v) })
    }

    @usableFromInline
    static func _vjpMultiply(lhs: Complex, rhs: Complex)
    -> (Complex, (Complex) -> (Complex, Complex)) {
        return (lhs * rhs, { v in (rhs * v, lhs * v) })
    }

    @usableFromInline
    static func _vjpDivide(lhs: Complex, rhs: Complex)
    -> (Complex, (Complex) -> (Complex, Complex)) {
        return (lhs / rhs, { v in (v / rhs, -lhs / (rhs * rhs) * v) })
    }

    @usableFromInline
    static func _vjpNegate(operand: Complex)
    -> (Complex, (Complex) -> Complex) {
        return (-operand, { -$0 })
    }

    @usableFromInline
    func _vjpComplexConjugate() -> (Complex, (Complex) -> Complex) {
        return (complexConjugate(), { v in v.complexConjugate() })
    }
}

extension Complex where T: Differentiable, T.TangentVector == T {
    @usableFromInline
    func _vjpAdding(real: T) -> (Complex, (Complex) -> (Complex, T)) {
        return (self.adding(real: real), { ($0, $0.real) })
    }

    @usableFromInline
    func _vjpSubtracting(real: T) -> (Complex, (Complex) -> (Complex, T)) {
        return (self.subtracting(real: real), { ($0, -$0.real) })
    }

    @usableFromInline
    func _vjpAdding(imaginary: T) -> (Complex, (Complex) -> (Complex, T)) {
        return (self.adding(real: real), { ($0, $0.imaginary) })
    }

    @usableFromInline
    func _vjpSubtracting(imaginary: T) -> (Complex, (Complex) -> (Complex, T)) {
        return (self.subtracting(real: real), { ($0, -$0.imaginary) })
    }
}
