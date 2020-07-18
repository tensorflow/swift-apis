
infix operator .<: ComparisonPrecedence
infix operator .<=: ComparisonPrecedence
infix operator .>=: ComparisonPrecedence
infix operator .>: ComparisonPrecedence
infix operator .==: ComparisonPrecedence
infix operator .!=: ComparisonPrecedence

extension Tensor where Scalar: Numeric & Comparable {
  @inlinable
  public static func .< (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .<= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .> (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .>= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .< (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .<= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .> (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .>= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .< (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .<= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .> (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .>= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    fatalError()
  }
}

extension Tensor where Scalar: Equatable {
  @inlinable
  public static func .== (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .!= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .== (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .!= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .== (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    fatalError()
  }

  @inlinable
  public static func .!= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    fatalError()
  }
}


extension Tensor where Scalar: TensorFlowFloatingPoint & Equatable {
  @inlinable
  public func elementsAlmostEqual(
    _ other: Tensor,
    tolerance: Scalar = Scalar.ulpOfOne.squareRoot()
  ) -> Tensor<Bool> {
    fatalError()
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  public func isAlmostEqual(
    to other: Tensor,
    tolerance: Scalar = Scalar.ulpOfOne.squareRoot()
  ) -> Bool {
  fatalError()
 }
}
