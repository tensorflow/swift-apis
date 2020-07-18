import _Differentiation
import CTensorFlow

infix operator .==: ComparisonPrecedence
infix operator .!=: ComparisonPrecedence

public protocol AnyTensor {
  var _rawTensorHandle: CTensorHandle { get }
  var _tensorFlowDataType: TensorDataType { get }
}

@frozen
public struct Tensor<Scalar: TensorFlowScalar> {
  public let handle: TensorHandle<Scalar>

  @usableFromInline
  internal var _isScalarZero: Bool = false

  @inlinable
  public init(handle: TensorHandle<Scalar>) {
    self.handle = handle
  }
}

extension Tensor: AnyTensor {
  public var _rawTensorHandle: CTensorHandle { return handle._cTensorHandle }
  public var _tensorFlowDataType: TensorDataType { return Scalar.tensorFlowDataType }
}


extension Tensor {
  public var rank: Int {
    @_semantics("autodiff.nonvarying")
    get { handle.rank }
  }

  #if USING_X10_BACKEND
    @inlinable
    public var scalarCount: Int {
      @_semantics("autodiff.nonvarying")
      get { 0 }
    }
  #else
    @inlinable
    public var scalarCount: Int {
      @_semantics("autodiff.nonvarying")
      get {
        let status = _ExecutionContext.global.status
        let size = TFE_TensorHandleNumElements(handle._cTensorHandle, status)
        checkOk(status)
        return Int(size)
      }
    }
  #endif

  @inlinable
  public var rankTensor: Tensor<Int32> {
    @_semantics("autodiff.nonvarying")
    get {
      fatalError()
    }
  }

  @inlinable
  public var shapeTensor: Tensor<Int32> {
    @_semantics("autodiff.nonvarying")
    get {
      fatalError()
    }
  }

  @inlinable
  public var scalarCountTensor: Tensor<Int32> {
    @_semantics("autodiff.nonvarying")
    get {
      fatalError()
    }
  }
}


extension Tensor {
  @inlinable
  public var isScalar: Bool {
    return rank == 0
  }

  @inlinable
  public var scalar: Scalar? {
    isScalar ? scalars[0] : nil
  }

  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public func scalarized() -> Scalar {
    return scalars[0]
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: scalarized)
  func _vjpScalarized() -> (value: Scalar, pullback: (Scalar) -> Tensor) {
    fatalError()
  }
}

extension TensorFlowScalar {
  @inlinable
  public init?(_ tensor: Tensor<Self>) {
    guard let scalar = tensor.scalar else {
      return nil
    }
    self = scalar
  }
}


extension Tensor {
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public var scalars: [Scalar] {
    return []
  }
}

extension Tensor {
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public init(_ value: Scalar, on device: Device = .default) {
    fatalError()
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: init(_:on:))
  static func _vjpScalarInit(_ value: __owned Scalar, on device: Device = .default) -> (
    value: Tensor, pullback: (Tensor) -> Scalar
  ) {
    return (Tensor(value, on: device), { $0.scalarized() })
  }
}

extension Tensor {
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public init(_ scalars: [Scalar], on device: Device = .default) {
    fatalError()
  }

  @inlinable
  public init<C: RandomAccessCollection>(
    _ vector: C, on device: Device = .default
  ) where C.Element == Scalar {
    #if USING_X10_BACKEND
      self.init([Scalar](vector), on: device)
    #else
      let handle = TensorHandle<Scalar>(
        shape: [vector.count],
        scalarsInitializer: { addr in
          var currentAddr = addr
          for scalar in vector {
            currentAddr.initialize(to: scalar)
            currentAddr = currentAddr.advanced(by: 1)
          }
        })
      self.init(handle: handle)
    #endif
  }

}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: init(_:on:))
  static func _vjpInit(_ scalars: [Scalar], on device: Device = .default) -> (
    value: Tensor, pullback: (Tensor) -> Array<Scalar>.TangentVector
  ) {
    (
      value: Tensor(scalars, on: device),
      pullback: { v in
        Array<Scalar>.TangentVector(v.scalars)
      }
    )
  }
}


@frozen
public struct _TensorElementLiteral<Scalar> where Scalar: TensorFlowScalar {
  @usableFromInline let tensor: Tensor<Scalar>
}

extension _TensorElementLiteral: ExpressibleByBooleanLiteral
where Scalar: ExpressibleByBooleanLiteral {
  public typealias BooleanLiteralType = Scalar.BooleanLiteralType
  @inlinable
  public init(booleanLiteral: BooleanLiteralType) {
    tensor = Tensor(Scalar(booleanLiteral: booleanLiteral))
  }
}

extension _TensorElementLiteral: ExpressibleByIntegerLiteral
where Scalar: ExpressibleByIntegerLiteral {
  public typealias IntegerLiteralType = Scalar.IntegerLiteralType
  @inlinable
  public init(integerLiteral: IntegerLiteralType) {
    tensor = Tensor(Scalar(integerLiteral: integerLiteral))
  }
}

extension _TensorElementLiteral: ExpressibleByFloatLiteral
where Scalar: ExpressibleByFloatLiteral {
  public typealias FloatLiteralType = Scalar.FloatLiteralType
  @inlinable
  public init(floatLiteral: FloatLiteralType) {
    tensor = Tensor(Scalar(floatLiteral: floatLiteral))
  }
}

extension _TensorElementLiteral: ExpressibleByArrayLiteral {
  public typealias ArrayLiteralElement = _TensorElementLiteral<Scalar>
  @inlinable
  public init(arrayLiteral elements: _TensorElementLiteral<Scalar>...) {
    tensor = _Raw.pack(elements.map { $0.tensor })
  }
}

extension Tensor: ExpressibleByArrayLiteral {
  public typealias ArrayLiteralElement = _TensorElementLiteral<Scalar>

  @inlinable
  internal init(_tensorElementLiterals elements: [_TensorElementLiteral<Scalar>]) {
    self = _Raw.pack(elements.map { $0.tensor })
  }

  @inlinable
  public init(arrayLiteral elements: _TensorElementLiteral<Scalar>...) {
    precondition(!elements.isEmpty, "Cannot create a 'Tensor' with no elements.")
    self.init(_tensorElementLiterals: elements)
  }
}


extension Tensor: Equatable where Scalar: Equatable {
  @inlinable
  public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
    fatalError()
  }

  @inlinable
  public static func != (lhs: Tensor, rhs: Tensor) -> Bool {
    fatalError()
  }
}

extension Tensor: AdditiveArithmetic where Scalar: Numeric {
  @inlinable
  public static var zero: Tensor { Tensor(0) }

  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
    _Raw.addV2(lhs, rhs)
  }

  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
    fatalError()
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: +)
  static func _vjpAdd(lhs: Tensor, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    fatalError()
  }

  @inlinable
  @derivative(of: -)
  static func _vjpSubtract(lhs: Tensor, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    fatalError()
  }
}

extension Tensor: PointwiseMultiplicative where Scalar: Numeric {
  @inlinable
  public static var one: Tensor { Tensor(1) }

  @inlinable
  public var reciprocal: Tensor { self }

  public static func .* (lhs: Tensor, rhs: Tensor) -> Tensor {
    return lhs
  }
}

extension Tensor: Differentiable & EuclideanDifferentiable where Scalar: TensorFlowFloatingPoint {
  public typealias TangentVector = Tensor
}
