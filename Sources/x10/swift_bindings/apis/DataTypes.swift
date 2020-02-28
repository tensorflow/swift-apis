/// A floating-point data type that conforms to `Differentiable` and is compatible with TensorFlow.
///
/// - Note: `Tensor` conditionally conforms to `Differentiable` when the `Scalar` associated type
///   conforms `TensorFlowFloatingPoint`.
public protocol TensorFlowFloatingPoint:
  XLAScalarType & BinaryFloatingPoint & Differentiable & ElementaryFunctions
where
  Self.RawSignificand: FixedWidthInteger,
  Self == Self.TangentVector
{}
public typealias TensorFlowNumeric = XLAScalarType & Numeric
public typealias TensorFlowScalar = XLAScalarType
public typealias TensorFlowInteger = TensorFlowScalar & BinaryInteger

/// An integer data type that represents integer types which can be used as tensor indices in 
/// TensorFlow.
public protocol TensorFlowIndex: TensorFlowInteger {}

extension Int32: TensorFlowIndex {}
extension Int64: TensorFlowIndex {}

extension Float: TensorFlowFloatingPoint {}
extension Double: TensorFlowFloatingPoint {}
