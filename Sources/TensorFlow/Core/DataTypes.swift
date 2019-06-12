// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

/// A TensorFlow dynamic type value that can be created from types that conform to
/// `TensorFlowScalar`.
// This simply wraps a `TF_DataType` and allows user code to handle
// `TF_DataType` without importing CTensorFlow, which pollutes the namespace
// with TensorFlow C API declarations.
public struct TensorDataType {
    public var _cDataType: TF_DataType

    @usableFromInline
	internal init(_ cDataType: TF_DataType) {
	    self._cDataType = cDataType
    }
}

@usableFromInline
internal func makeTensor(
    dataType: TensorDataType,
    owning pointer: CTensorHandle
) -> AnyTensor {
    switch dataType._cDataType {
    case TF_BOOL: return Tensor<Bool>(handle: TensorHandle(_owning: pointer))
    case TF_INT8: return Tensor<Int8>(handle: TensorHandle(_owning: pointer))
    case TF_UINT8: return Tensor<UInt8>(handle: TensorHandle(_owning: pointer))
    case TF_INT16: return Tensor<Int16>(handle: TensorHandle(_owning: pointer))
    case TF_UINT16: return Tensor<UInt16>(handle: TensorHandle(_owning: pointer))
    case TF_INT32: return Tensor<Int32>(handle: TensorHandle(_owning: pointer))
    case TF_UINT32: return Tensor<UInt32>(handle: TensorHandle(_owning: pointer))
    case TF_INT64: return Tensor<Int64>(handle: TensorHandle(_owning: pointer))
    case TF_UINT64: return Tensor<UInt64>(handle: TensorHandle(_owning: pointer))
    case TF_BFLOAT16: return Tensor<BFloat16>(handle: TensorHandle(_owning: pointer))
    case TF_FLOAT: return Tensor<Float>(handle: TensorHandle(_owning: pointer))
    case TF_DOUBLE: return Tensor<Double>(handle: TensorHandle(_owning: pointer))
    case TF_STRING: fatalError("StringTensor does not conform to AnyTensor")
    default: fatalError("Unhandled type: \(dataType)")
    }
}

/// A data type compatible with TensorFlow.
public protocol _TensorFlowDataTypeCompatible {
    /// The underlying TensorFlow data type.
    @inlinable
    static var tensorFlowDataType: TensorDataType { get }
}

/// A scalar data type compatible with TensorFlow.
///
/// Types that conform to `TensorFlowScalar` can be used as the `Scalar` associated type of
/// `Tensor`.
//
// This includes all `_TensorFlowDataTypeCompatible` types except `String`.
public protocol TensorFlowScalar: _TensorFlowDataTypeCompatible {}

public typealias TensorFlowNumeric = TensorFlowScalar & Numeric
public typealias TensorFlowSignedNumeric = TensorFlowScalar & SignedNumeric
public typealias TensorFlowInteger = TensorFlowScalar & BinaryInteger

/// A floating-point data type that conforms to `Differentiable` and is compatible with TensorFlow.
///
/// - Note: `Tensor` conditionally conforms to `Differentiable` when the `Scalar` associated type
///   conforms `TensorFlowFloatingPoint`.
public protocol TensorFlowFloatingPoint: TensorFlowScalar & BinaryFloatingPoint & Differentiable
    where Self.RawSignificand: FixedWidthInteger,
          Self == Self.TangentVector,
          Self == Self.AllDifferentiableVariables {}

extension Float: TensorFlowFloatingPoint {}
extension Double: TensorFlowFloatingPoint {}

extension Bool: TensorFlowScalar {
    @inlinable
    public static var tensorFlowDataType: TensorDataType {
        return TensorDataType(TF_BOOL)
    }
}

extension Int8: TensorFlowScalar {
    @inlinable
    public static var tensorFlowDataType: TensorDataType {
        return TensorDataType(TF_INT8)
    }
}

extension UInt8: TensorFlowScalar {
    @inlinable
    public static var tensorFlowDataType: TensorDataType {
        return TensorDataType(TF_UINT8)
    }
}

extension Int16: TensorFlowScalar {
    @inlinable
    public static var tensorFlowDataType: TensorDataType {
        return TensorDataType(TF_INT16)
    }
}

extension UInt16: TensorFlowScalar {
    @inlinable
    public static var tensorFlowDataType: TensorDataType {
        return TensorDataType(TF_UINT16)
    }
}

extension Int32: TensorFlowScalar {
    @inlinable
    public static var tensorFlowDataType: TensorDataType {
        return TensorDataType(TF_INT32)
    }
}

extension UInt32: TensorFlowScalar {
    @inlinable
    public static var tensorFlowDataType: TensorDataType {
        return TensorDataType(TF_UINT32)
    }
}

extension Int64: TensorFlowScalar {
    @inlinable
    public static var tensorFlowDataType: TensorDataType {
        return TensorDataType(TF_INT64)
    }
}

extension UInt64: TensorFlowScalar {
    @inlinable
    public static var tensorFlowDataType: TensorDataType {
        return TensorDataType(TF_UINT64)
    }
}

@frozen
public struct BFloat16 {
    @usableFromInline var data: Int16 = 0
    private init() {}
}

extension BFloat16: TensorFlowScalar {
    @inlinable
    public static var tensorFlowDataType: TensorDataType {
        return TensorDataType(TF_BFLOAT16)
    }
}

extension Float: TensorFlowScalar {
    @inlinable
    public static var tensorFlowDataType: TensorDataType {
        return TensorDataType(TF_FLOAT)
    }
}

extension Double: TensorFlowScalar {
    @inlinable
    public static var tensorFlowDataType: TensorDataType {
        return TensorDataType(TF_DOUBLE)
    }
}

extension String: _TensorFlowDataTypeCompatible {
    @inlinable
    public static var tensorFlowDataType: TensorDataType {
        return TensorDataType(TF_STRING)
    }
}
