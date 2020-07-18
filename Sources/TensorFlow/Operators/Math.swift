import _Differentiation

extension Tensor: ElementaryFunctions where Scalar: TensorFlowFloatingPoint {
  @differentiable
  public static func sqrt(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func cos(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func sin(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func tan(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func acos(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func asin(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func atan(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func cosh(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func sinh(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func tanh(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func acosh(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func asinh(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func atanh(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func exp(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func exp2(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func exp10(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func expm1(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func log(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func log2(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func log10(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func log1p(_ x: Self) -> Self {
    return x
  }

  @differentiable
  public static func pow(_ x: Self, _ y: Self) -> Self {
    return x
  }

  @differentiable
  public static func pow(_ x: Self, _ n: Int) -> Self {
    return x
  }

  @differentiable
  public static func root(_ x: Self, _ n: Int) -> Self {
    return x
  }
}

extension Tensor: VectorProtocol where Scalar: TensorFlowFloatingPoint {
  public typealias VectorSpaceScalar = Float

  public func scaled(by scale: Float) -> Self {
    self
  }

  public func adding(_ scalar: Float) -> Self {
    self
  }

  public func subtracting(_ scalar: Float) -> Self {
    self
  }
}
