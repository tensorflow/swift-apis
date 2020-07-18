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

  public var shape: TensorShape {
    @_semantics("autodiff.nonvarying")
    get { handle.shape }
  }

  #if USING_X10_BACKEND
    @inlinable
    public var scalarCount: Int {
      @_semantics("autodiff.nonvarying")
      get { shape.contiguousSize }
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
    precondition(
      shape.contiguousSize == 1,
      "This tensor must have exactly one scalar but contains \(shape.contiguousSize).")
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
  @inlinable
  public var array: ShapedArray<Scalar> {
    debugLog("Returning a host copy of array.")
    #if USING_X10_BACKEND
      if handle.backend == .XLA {
        return ShapedArray<Scalar>(shape: shape.dimensions, scalars: scalars)
      }
    #endif
    return handle.makeHostCopy()
  }

  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public var scalars: [Scalar] {
    #if USING_X10_BACKEND
      if handle.backend == .XLA {
        let (storage, _) = xlaTensor.fetchTensorValues(Scalar.self)
        return storage
      }
    #endif
    return array.scalars
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: scalars)
  func _vjpScalars() -> (value: [Scalar], pullback: (Array<Scalar>.TangentVector) -> Tensor) {
    fatalError()
  }
}


extension Tensor {
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public init(_ value: Scalar, on device: Device = .default) {
    #if USING_X10_BACKEND
      switch device.backend {
      case .XLA:
        self.init(_xla: XLATensor.make(value, on: device))
      case .TF_EAGER:
        self.init(shape: [], scalars: [value], on: device)
      }
    #else
      self.init(shape: [], scalars: [value], on: device)
    #endif
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
    self.init(shape: [scalars.count], scalars: scalars, on: device)
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

  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  public init(shape: TensorShape, scalars: [Scalar], on device: Device = .default) {
    precondition(
      shape.contiguousSize == scalars.count,
      """
      The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
      provided.
      """)
    self = scalars.withUnsafeBufferPointer { bufferPointer in
      Tensor(shape: shape, scalars: bufferPointer, on: device)
    }
  }

  public init(
    shape: TensorShape,
    scalars: UnsafeBufferPointer<Scalar>,
    on device: Device = .default
  ) {
    precondition(
      shape.contiguousSize == scalars.count,
      """
      The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
      provided.
      """)
    #if USING_X10_BACKEND
      switch device.backend {
      case .XLA:
        self.init(_xla: XLATensor.make(scalars, shape.dimensions, on: device))
      case .TF_EAGER:
        let handle = TensorHandle<Scalar>(
          shape: shape.dimensions,
          scalarsInitializer: { address in
            address.initialize(from: scalars.baseAddress!, count: shape.contiguousSize)
          })
        self.init(handle: handle)
      }
    #else
      let handle = TensorHandle<Scalar>(
        shape: shape.dimensions,
        scalarsInitializer: { address in
          address.initialize(from: scalars.baseAddress!, count: shape.contiguousSize)
        })
      self.init(handle: handle)
    #endif
  }

  #if USING_X10_BACKEND
    @inlinable
    public init(
      shape: TensorShape,
      scalars: [Scalar],
      toReducedPrecision: Bool,
      directlyOn device: Device
    ) {
      precondition(
        shape.contiguousSize == scalars.count,
        """
        The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
        provided.
        """)
      self = scalars.withUnsafeBufferPointer { bufferPointer in
        Tensor(
          shape: shape, scalars: bufferPointer, toReducedPrecision: toReducedPrecision,
          directlyOn: device)
      }
    }

    public init(
      shape: TensorShape,
      scalars: UnsafeBufferPointer<Scalar>,
      toReducedPrecision: Bool,
      directlyOn device: Device
    ) {
      precondition(
        shape.contiguousSize == scalars.count,
        """
        The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
        provided.
        """)
      switch device.backend {
      case .XLA:
        self.init(
          _xla: XLATensor.make(
            scalars, shape.dimensions, toReducedPrecision: toReducedPrecision,
            directlyOn: device))
      case .TF_EAGER:
        precondition(!toReducedPrecision)
        self = .init(shape: shape, scalars: scalars, on: device)
      }
    }
  #endif

  public init<C: RandomAccessCollection>(
    shape: TensorShape, scalars: C, on device: Device = .default
  ) where C.Element == Scalar {
    precondition(
      shape.contiguousSize == scalars.count,
      """
      The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
      provided.
      """)
    #if USING_X10_BACKEND
      self.init(shape: shape, scalars: [Scalar](scalars), on: device)
    #else
      let handle = TensorHandle<Scalar>(
        shape: shape.dimensions,
        scalarsInitializer: { addr in
          var currentAddr = addr
          for scalar in scalars {
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

  @inlinable
  @derivative(of: init(shape:scalars:on:))
  static func _vjpInit(
    shape: TensorShape, scalars: [Scalar], on device: Device = .default
  ) -> (value: Tensor, pullback: (Tensor) -> Array<Scalar>.TangentVector) {
    (
      value: Tensor(shape: shape, scalars: scalars, on: device),
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


extension Tensor: CustomStringConvertible {
  public var description: String {
    @_semantics("autodiff.nonvarying")
    get {
      return array.description
    }
  }
}

extension Tensor {
  public func description(
    lineWidth: Int = 80,
    edgeElementCount: Int = 3,
    summarizing: Bool = false
  ) -> String {
    return array.description(
      lineWidth: lineWidth,
      edgeElementCount: edgeElementCount,
      summarizing: summarizing)
  }

  public var fullDescription: String {
    @_semantics("autodiff.nonvarying")
    get {
      return array.fullDescription
    }
  }

  #if USING_X10_BACKEND
    public var irText: String { XLATensor.irText(xlaTensor) }
  #endif
}

extension Tensor: CustomPlaygroundDisplayConvertible {
  public var playgroundDescription: Any {
    @_semantics("autodiff.nonvarying")
    get {
      return description
    }
  }
}

extension Tensor: CustomReflectable {
  public var customMirror: Mirror {
    @_semantics("autodiff.nonvarying")
    get {
      return Mirror(self, children: [], displayStyle: .struct)
    }
  }
}


extension Tensor: Codable where Scalar: Codable {
  @inlinable
  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(array)
  }

  @inlinable
  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    let array = try container.decode(ShapedArray<Scalar>.self)
    self.init(array)
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
