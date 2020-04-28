/// A `Layer` formed by adapting `Base` in some way I can't describe
/// but you probably can.
struct CollectionModule<Base: Collection> : Module
    where Base: Differentiable,
          Base.Element: Module, 
          Base.Element.Input == Base.Element.Output
{
  typealias Input = Base.Element.Input
  typealias Output = Base.Element.Output
  typealias TangentVector = Self

  private var base: Base
  init(_ base: Base) { self.base = base }

  @differentiable(wrt: self)
  public func callAsFunction(_ input: Input) -> Output {
    base.differentiableReduce(input) { $1($0) }
  }
}

extension CollectionModule : ElementaryFunctions {
  /// The square root of `x`.
  ///
  /// For real types, if `x` is negative the result is `.nan`. For complex
  /// types there is a branch cut on the negative real axis.
  public static func sqrt(_ x: Self) -> Self { Self(.sqrt(x.base)) }

  /// The cosine of `x`, interpreted as an angle in radians.
  public static func cos(_ x: Self) -> Self { Self(.cos(x.base)) }

  /// The sine of `x`, interpreted as an angle in radians.
  public static func sin(_ x: Self) -> Self { Self(.sin(x.base)) }

  /// The tangent of `x`, interpreted as an angle in radians.
  public static func tan(_ x: Self) -> Self { Self(.tan(x.base)) }

  /// The inverse cosine of `x` in radians.
  public static func acos(_ x: Self) -> Self { Self(.acos(x.base)) }

  /// The inverse sine of `x` in radians.
  public static func asin(_ x: Self) -> Self { Self(.asin(x.base)) }

  /// The inverse tangent of `x` in radians.
  public static func atan(_ x: Self) -> Self { Self(.atan(x.base)) }

  /// The hyperbolic cosine of `x`.
  public static func cosh(_ x: Self) -> Self { Self(.cosh(x.base)) }

  /// The hyperbolic sine of `x`.
  public static func sinh(_ x: Self) -> Self { Self(.sinh(x.base)) }

  /// The hyperbolic tangent of `x`.
  public static func tanh(_ x: Self) -> Self { Self(.tanh(x.base)) }

  /// The inverse hyperbolic cosine of `x`.
  public static func acosh(_ x: Self) -> Self { Self(.acosh(x.base)) }

  /// The inverse hyperbolic sine of `x`.
  public static func asinh(_ x: Self) -> Self { Self(.asinh(x.base)) }

  /// The inverse hyperbolic tangent of `x`.
  public static func atanh(_ x: Self) -> Self { Self(.atanh(x.base)) }

  /// The exponential function applied to `x`, or `e**x`.
  public static func exp(_ x: Self) -> Self { Self(.exp(x.base)) }

  /// Two raised to to power `x`.
  public static func exp2(_ x: Self) -> Self { Self(.exp2(x.base)) }

  /// Ten raised to to power `x`.
  public static func exp10(_ x: Self) -> Self { Self(.exp10(x.base)) }

  /// `exp(x) - 1` evaluated so as to preserve accuracy close to zero.
  public static func expm1(_ x: Self) -> Self { Self(.expm1(x.base)) }

  /// The natural logarithm of `x`.
  public static func log(_ x: Self) -> Self { Self(.log(x.base)) }

  /// The base-two logarithm of `x`.
  public static func log2(_ x: Self) -> Self { Self(.log2(x.base)) }

  /// The base-ten logarithm of `x`.
  public static func log10(_ x: Self) -> Self { Self(.log10(x.base)) }

  /// `log(1 + x)` evaluated so as to preserve accuracy close to zero.
  public static func log1p(_ x: Self) -> Self { Self(.log1p(x.base)) }

  /// `exp(y log(x))` computed without loss of intermediate precision.
  ///
  /// For real types, if `x` is negative the result is NaN, even if `y` has
  /// an integral value. For complex types, there is a branch cut on the
  /// negative real axis.
  public static func pow(_ x: Self, _ y: Self) -> Self {
    Self(.pow(x.base, y.base))
  }

  /// `x` raised to the `n`th power.
  ///
  /// The product of `n` copies of `x`.
  public static func pow(_ x: Self, _ n: Int) -> Self { 
    Self(.pow(x.base, n))
  }

  /// The `n`th root of `x`.
  ///
  /// For real types, if `x` is negative and `n` is even, the result is NaN.
  /// For complex types, there is a branch cut along the negative real axis.
  public static func root(_ x: Self, _ n: Int) -> Self { 
    Self(.root(x.base, n))
  }
}

/*

struct CollectionLayer<Base: RandomAccessCollection> : Layer
    where Base: Differentiable, 
          Base.Element: Layer, 
          Base.Element.Input == Base.Element.Output
{
  typealias Input = Base.Element.Input
  typealias Output = Base.Element.Output

  private var base: Base
  init(_ base: Base) { self.base = base }

  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    base.differentiableReduce(input) { $1($0) }
  }
}
*/
