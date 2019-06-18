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

import CTensorFlow

infix operator .==: ComparisonPrecedence
infix operator .!=: ComparisonPrecedence

/// Special protocol for calling tensorflow operations that take heterogeneous arrays as input.
public protocol AnyTensor {
    var _rawTensorHandle: CTensorHandle { get }
    var _tensorFlowDataType: TensorDataType { get }
}

/// `Tensor` is a multi-dimensional array used for computation. It is a wrapper around a
/// `TensorHandle`.
@frozen
public struct Tensor<Scalar: TensorFlowScalar>: TensorProtocol {
    /// The underlying `TensorHandle`.
    /// - Note: `handle` is public to allow user defined ops, but should not normally be used.
    public let handle: TensorHandle<Scalar>

    @inlinable
    public init(handle: TensorHandle<Scalar>) {
        self.handle = handle
    }
}

extension Tensor: AnyTensor {
    public var _rawTensorHandle: CTensorHandle { return handle._cTensorHandle }
    public var _tensorFlowDataType: TensorDataType { return Scalar.tensorFlowDataType }
}

//===------------------------------------------------------------------------------------------===//
// Tensor Properties
//===------------------------------------------------------------------------------------------===//

public extension Tensor {
    /// The number of dimensions of the `Tensor`.
    @inlinable
    var rank: Int {
        @_semantics("autodiff.nonvarying")
        get {
            let status = _ExecutionContext.global.status
            let rank = TFE_TensorHandleNumDims(handle._cTensorHandle, status)
            checkOk(status)
            return Int(rank)
        }
    }

    /// The shape of the `Tensor`.
    @inlinable
    var shape: TensorShape {
        @_semantics("autodiff.nonvarying")
        get {
            let status = _ExecutionContext.global.status
            let dims: [Int] = (0..<Int32(rank)).map { i in
                let dim = TFE_TensorHandleDim(self.handle._cTensorHandle, i, status)
                checkOk(status)
                return Int(dim)
            }
            return TensorShape(dims)
        }
    }

    /// The number of scalars in the `Tensor`.
    @inlinable
    var scalarCount: Int {
        @_semantics("autodiff.nonvarying")
        get {
            let status = _ExecutionContext.global.status
            let size = TFE_TensorHandleNumElements(handle._cTensorHandle, status)
            checkOk(status)
            return Int(size)
        }
    }

    /// The rank of the tensor, represented as a `Tensor<Int32>`.
    @inlinable
    var rankTensor: Tensor<Int32> {
        @_semantics("autodiff.nonvarying")
        get {
            return Raw.rank(self)
        }
    }

    /// The dimensions of the tensor, represented as a `Tensor<Int32>`.
    @inlinable
    var shapeTensor: Tensor<Int32> {
        @_semantics("autodiff.nonvarying")
        get {
            return Raw.shape(self)
        }
    }

    /// The number of scalars in the tensor, represented as a `Tensor<Int32>`.
    @inlinable
    var scalarCountTensor: Tensor<Int32> {
        @_semantics("autodiff.nonvarying")
        get {
            return Raw.size(self)
        }
    }
}

//===------------------------------------------------------------------------------------------===//
// Scalar Conversion
//===------------------------------------------------------------------------------------------===//

public extension Tensor {
    /// Returns `true` if `rank` is equal to 0 and `false` otherwise.
    @inlinable
    var isScalar: Bool {
        return rank == 0
    }

    /// Returns the single scalar element if `rank` is equal to 0 and `nil`
    /// otherwise.
    @inlinable
    var scalar: Scalar? {
        return handle.makeHostCopy().scalar
    }

    /// Reshape to scalar.
    /// - Precondition: The tensor has exactly one scalar.
    @inlinable
    @differentiable(wrt: self, vjp: _vjpScalarized where Scalar: TensorFlowFloatingPoint)
    func scalarized() -> Scalar {
        return reshaped(to: []).scalar!
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    func _vjpScalarized() -> (Scalar, (Scalar) -> Tensor) {
        return (scalarized(), { v in Tensor(v) })
    }
}

public extension TensorFlowScalar {
    @inlinable
    init?(_ tensor: Tensor<Self>) {
        guard let scalar = tensor.scalar else {
            return nil
        }
        self = scalar
    }
}

//===------------------------------------------------------------------------------------------===//
// Array Conversion
//===------------------------------------------------------------------------------------------===//

public extension Tensor {
    @inlinable
    var array: ShapedArray<Scalar> {
        debugLog("Returning a host copy of array.")
        internalConsistencyCheck(handle.isConcrete)
	    return handle.makeHostCopy()
    }

    @inlinable
    var scalars: [Scalar] {
        return array.scalars
    }
}

//===------------------------------------------------------------------------------------------===//
// Initialization
//===------------------------------------------------------------------------------------------===//

public extension Tensor {
    /// Creates a 0-D tensor from a scalar value.
    @inlinable
    @differentiable(vjp: _vjpScalarInit where Scalar: TensorFlowFloatingPoint)
    init(_ value: Scalar) {
        self.init(shape: [], scalars: [value])
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    static func _vjpScalarInit(_ value: Scalar) -> (Tensor, (Tensor) -> Scalar) {
        return (Tensor(value), { $0.scalarized() })
    }
}

public extension Tensor {
    /// Creates a 1D tensor from scalars.
    @inlinable
    init(_ scalars: [Scalar]) {
        self.init(shape: [scalars.count], scalars: scalars)
    }

    /// Creates a 1D tensor from scalars.
    @inlinable
    init<C: RandomAccessCollection>(_ vector: C) where C.Element == Scalar {
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
    }

    /// Creates a tensor with the specified shape and contiguous scalars in row-major order.
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - scalars: The scalar contents of the tensor.
    /// - Precondition: The number of scalars must equal the product of the dimensions of the shape.
    @inlinable
    init(shape: TensorShape, scalars: [Scalar]) {
        self = scalars.withUnsafeBufferPointer { bufferPointer in
	        Tensor(shape: shape, scalars: bufferPointer)
	    }
    }

    /// Creates a tensor with the specified shape and contiguous scalars in row-major order.
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - scalars: The scalar contents of the tensor.
    /// - Precondition: The number of scalars must equal the product of the
    ///   dimensions of the shape.
    @inlinable
    init(shape: TensorShape, scalars: UnsafeBufferPointer<Scalar>) {
        precondition(scalars.count == shape.contiguousSize)
        let handle = TensorHandle<Scalar>(
            shape: shape.dimensions,
            scalarsInitializer: { address in
                address.initialize(from: scalars.baseAddress!, count: shape.contiguousSize)
            })
        self.init(handle: handle)
    }

    /// Creates a tensor with the specified shape and contiguous scalars in row-major order.
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - scalars: The scalar contents of the tensor.
    /// - Precondition: The number of scalars must equal the product of the
    ///   dimensions of the shape.
    @inlinable
    init<C: RandomAccessCollection>(shape: TensorShape, scalars: C) where C.Element == Scalar {
        precondition(scalars.count == shape.contiguousSize)
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
    }
}

// Background story on `TensorElementLiteral` and why it's necessary:
//
// Very importantly, we want users to be able to implicitly convert an array
// literal to a tensor. At first glance, a straightfoward implementation would
// be conforming `Tensor` to `ExpressibleByArrayLiteral` with
// `ExpressibleBy(Float|Int|Bool)Literal` as a base case. However, it is not
// that simple. We have binary operators that take `(Tensor, Scalar)`, `(Scalar,
// Tensor)` as well as `(Tensor, Tensor)`. When `Tensor`s are convertible from
// both a scalar and an array literal, a scalar-tensor binary operator like `+`
// will not type check.
//
// One way to work around it is to define all tensor-tensor operators in a
// protocol extension, and all tensor-scalar and scalar-tensor operators on
// concrete `Tensor`. Protocol extensions are less favorable than concrete
// implementations, so the compiler will prefer the concrete implementation for
// a scalar-tensor operation. However, this would cause enormous code bloat and
// is entirely a hack.
//
// To resolve ambiguity, `Tensor` should not be expressible by scalar literal.
// There's already a lightweight syntax for converting a scalar to a tensor:
// `Tensor(x)`, so there is no strong need for implicit conversion. But we need
// to find a way to give `ExpressibleByArrayLiteral` a base case: what would the
// `ArrayLiteralElement` be if we want to support both `[1,2,3]` and `[[[1,2],
// [1,2]]]`? In the first case the array literal element is an interger, while
// in the second case the array literal itself should be a tensor. Based on this
// observation, we come up with an intermediate type: `TensorElementLiteral` as
// the `ArrayLiteralElement` of `Tensor`. By making `TensorElementLiteral`
// expressible by both array literal and scalar literal, `Tensor` can now be
// converted from an arbitrary-dimensional array literal.
//
// Due to protocol requirements, `TensorElementLiteral` has to be
// public. It is never supposed to be used directly by any user, so the library
// convention is to prepend an underscore to its name, making it
// `_TensorElementLiteral`.
//
// It would be nice to be able to remove this type when we can systematically
// resolve tensor-scalar/scalar-tensor op ambiguity someday, either through an
// improved `Expressible` model, or by introducing an attribute to tell the type
// checker which function to prefer when ambiguity occurs.

/// Represents a literal element for conversion to a `Tensor`.
///
/// - Note: Do not ever use this API directly. This is implicitly created
///   during the conversion from an array literal to a `Tensor`, and is purely
///   for implementation purposes.
@frozen
public struct _TensorElementLiteral<Scalar>: TensorProtocol where Scalar: TensorFlowScalar {
    @usableFromInline let tensor: Tensor<Scalar>

    @inlinable
    public var handle: TensorHandle<Scalar> {
        return tensor.handle
    }

    @inlinable
    public init(handle: TensorHandle<Scalar>) {
        tensor = Tensor(handle: handle)
    }
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
        tensor = Raw.pack(elements.map { $0.tensor })
    }
}

extension Tensor: ExpressibleByArrayLiteral {
    /// The type of the elements of an array literal.
    public typealias ArrayLiteralElement = _TensorElementLiteral<Scalar>

    /// Creates a tensor initialized with the given elements.
    /// - Note: This is for conversion from tensor element literals. This is a
    ///   separate method because `ShapedArray` initializers need to call it.
    @inlinable
    internal init(_tensorElementLiterals elements: [_TensorElementLiteral<Scalar>]) {
        self = Raw.pack(elements.map { $0.tensor })
    }

    /// Creates a tensor initialized with the given elements.
    @inlinable
    public init(arrayLiteral elements: _TensorElementLiteral<Scalar>...) {
        self.init(_tensorElementLiterals: elements)
    }
}

//===------------------------------------------------------------------------------------------===//
// Equatable
//===------------------------------------------------------------------------------------------===//

extension Tensor: Equatable where Scalar: Equatable {
    @inlinable
    public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
        // TODO: This is not correct due to broadcasting.
        return (lhs .== rhs).all()
    }

    @inlinable
    public static func != (lhs: Tensor, rhs: Tensor) -> Bool {
        // TODO: This is not correct due to broadcasting.
        return (lhs .!= rhs).any()
    }
}

//===------------------------------------------------------------------------------------------===//
// Description and Visualization
//===------------------------------------------------------------------------------------------===//

// String conversion.
extension Tensor: CustomStringConvertible {
    /// A textual representation of the tensor.
    ///
    /// - Note: use `fullDescription` for a non-pretty-printed description showing all scalars.
    public var description: String {
        @_semantics("autodiff.nonvarying")
        get {
            return array.description
        }
    }
}

public extension Tensor {
    /// A textual representation of the tensor. Returns a summarized description if `summarize` is
    /// true and the element count exceeds twice the `edgeElementCount`.
    ///
    /// - Parameters:
    ///   - lineWidth: The max line width for printing. Used to determine number of scalars to print
    ///     per line.
    ///   - edgeElementCount: The maximum number of elements to print before and after summarization
    ///     via ellipses (`...`).
    ///   - summarizing: If true, summarize description if element count exceeds twice
    ///     `edgeElementCount`.
    func description(
        lineWidth: Int = 80,
        edgeElementCount: Int = 3,
        summarizing: Bool = false
    ) -> String {
        return array.description(
            lineWidth: lineWidth,
            edgeElementCount: edgeElementCount,
            summarizing: summarizing)
    }

    /// A full, non-pretty-printed textual representation of the tensor, showing
    /// all scalars.
    var fullDescription: String {
        @_semantics("autodiff.nonvarying")
        get {
            return array.fullDescription
        }
    }
}

// Xcode Playground display conversion.
extension Tensor: CustomPlaygroundDisplayConvertible {
    public var playgroundDescription: Any {
        @_semantics("autodiff.nonvarying")
        get {
            return description
        }
    }
}

// Mirror representation, used by debugger/REPL.
extension Tensor: CustomReflectable {
    public var customMirror: Mirror {
        @_semantics("autodiff.nonvarying")
        get {
            return Mirror(self, children: [], displayStyle: .struct)
        }
    }
}

//===------------------------------------------------------------------------------------------===//
// Codable Conformance
//===------------------------------------------------------------------------------------------===//

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

//===------------------------------------------------------------------------------------------===//
// Additive Group
//===------------------------------------------------------------------------------------------===//

extension Tensor: AdditiveArithmetic where Scalar: Numeric {
    /// A scalar zero tensor.
    @inlinable
    public static var zero: Tensor {
        return Tensor(0)
    }

    /// Adds two tensors and produces their sum.
    /// - Note: `+` supports broadcasting.
    @inlinable
    @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Raw.add(lhs, rhs)
    }

    /// Subtracts one tensor from another and produces their difference.
    /// - Note: `-` supports broadcasting.
    @inlinable
    @differentiable(vjp: _vjpSubtract(lhs:rhs:) where Scalar: TensorFlowFloatingPoint)
    public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Raw.sub(lhs, rhs)
    }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    static func _vjpAdd(
        lhs: Tensor,
        rhs: Tensor
    ) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        return (lhs + rhs, { [lhsShape = lhs.shapeTensor, rhsShape = rhs.shapeTensor] v in
            let lhsGrad = v
            let rhsGrad = lhsGrad
            let (lhsAxes, rhsAxes) = Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
            return (lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
                    rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape))
        })
    }

    @inlinable
    static func _vjpSubtract(
        lhs: Tensor,
        rhs: Tensor
    ) -> (Tensor, (Tensor) -> (Tensor, Tensor)) {
        return (lhs - rhs, { [lhsShape = lhs.shapeTensor, rhsShape = rhs.shapeTensor] v in
            let lhsGrad = v
            let rhsGrad = -lhsGrad
            let (lhsAxes, rhsAxes) = Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
            return (lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
                    rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape))
        })
    }
}

//===------------------------------------------------------------------------------------------===//
// Differentiable
//===------------------------------------------------------------------------------------------===//

extension Tensor: Differentiable where Scalar: TensorFlowFloatingPoint {
    public typealias TangentVector = Tensor
    public typealias AllDifferentiableVariables = Tensor
}
