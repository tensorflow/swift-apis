// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

@_implementationOnly import x10_xla_tensor_wrapper

extension XLAScalar {
  init(_ v: Double) {
    self.init()
    self.tag = XLAScalarTypeTag_d
    self.value.d = v
  }

  init(_ v: Int64) {
    self.init()
    self.tag = XLAScalarTypeTag_i
    self.value.i = v
  }
}

public enum XLAScalarWrapper {
  public init(_ v: Double) {
    self = .d(v)
  }
  public init(_ v: Int64) {
    self = .i(v)
  }

  case d(Double)
  case i(Int64)
  var xlaScalar: XLAScalar {
    switch self {
    case .d(let v):
      return XLAScalar(v)
    case .i(let v):
      return XLAScalar(v)
    }
  }
}

/// A supported datatype in x10.
public protocol XLAScalarType {
  var xlaScalarWrapper: XLAScalarWrapper { get }
  static var xlaTensorScalarTypeRawValue: UInt32 { get }
}

extension XLAScalarType {
  var xlaScalar: XLAScalar { xlaScalarWrapper.xlaScalar }
  static var xlaTensorScalarType: XLATensorScalarType {
    #if os(Windows)
      return XLATensorScalarType(rawValue: Int32(xlaTensorScalarTypeRawValue))
    #else
      return XLATensorScalarType(rawValue: UInt32(xlaTensorScalarTypeRawValue))
    #endif
  }
}

extension Float: XLAScalarType {
  public var xlaScalarWrapper: XLAScalarWrapper { XLAScalarWrapper(Double(self)) }

  static public var xlaTensorScalarTypeRawValue: UInt32 {
    return UInt32(XLATensorScalarType_Float.rawValue)
  }
}
extension Double: XLAScalarType {
  public var xlaScalarWrapper: XLAScalarWrapper { XLAScalarWrapper(self) }

  static public var xlaTensorScalarTypeRawValue: UInt32 {
    return UInt32(XLATensorScalarType_Double.rawValue)
  }
}
extension Int64: XLAScalarType {
  public var xlaScalarWrapper: XLAScalarWrapper { XLAScalarWrapper(self) }

  static public var xlaTensorScalarTypeRawValue: UInt32 {
    return UInt32(XLATensorScalarType_Int64.rawValue)
  }
}
extension Int32: XLAScalarType {
  public var xlaScalarWrapper: XLAScalarWrapper { XLAScalarWrapper(Int64(self)) }

  static public var xlaTensorScalarTypeRawValue: UInt32 {
    return UInt32(XLATensorScalarType_Int32.rawValue)
  }
}
extension Int16: XLAScalarType {
  public var xlaScalarWrapper: XLAScalarWrapper { XLAScalarWrapper(Int64(self)) }

  static public var xlaTensorScalarTypeRawValue: UInt32 {
    return UInt32(XLATensorScalarType_Int16.rawValue)
  }
}
extension Int8: XLAScalarType {
  public var xlaScalarWrapper: XLAScalarWrapper { XLAScalarWrapper(Int64(self)) }

  static public var xlaTensorScalarTypeRawValue: UInt32 {
    return UInt32(XLATensorScalarType_Int8.rawValue)
  }
}
extension UInt8: XLAScalarType {
  public var xlaScalarWrapper: XLAScalarWrapper { XLAScalarWrapper(Int64(self)) }

  static public var xlaTensorScalarTypeRawValue: UInt32 {
    return UInt32(XLATensorScalarType_UInt8.rawValue)
  }
}

extension Bool: XLAScalarType {
  public var xlaScalarWrapper: XLAScalarWrapper { XLAScalarWrapper(Int64(self ? 1 : 0)) }

  static public var xlaTensorScalarTypeRawValue: UInt32 {
    return UInt32(XLATensorScalarType_Bool.rawValue)
  }
}

/// Error implementations
extension BFloat16: XLAScalarType {
  public var xlaScalarWrapper: XLAScalarWrapper { fatalError("BFloat16 not suported") }
  static public var xlaTensorScalarTypeRawValue: UInt32 {
    return UInt32(XLATensorScalarType_BFloat16.rawValue)
  }
}

extension UInt64: XLAScalarType {
  public var xlaScalarWrapper: XLAScalarWrapper { fatalError("UInt64 not suported") }
  static public var xlaTensorScalarTypeRawValue: UInt32 {
    fatalError("UInt64 not suported")
  }
}

extension UInt32: XLAScalarType {
  public var xlaScalarWrapper: XLAScalarWrapper { fatalError("UInt32 not suported") }
  static public var xlaTensorScalarTypeRawValue: UInt32 {
    fatalError("UInt32 not suported")
  }
}

extension UInt16: XLAScalarType {
  public var xlaScalarWrapper: XLAScalarWrapper { fatalError("UInt16 not suported") }
  static public var xlaTensorScalarTypeRawValue: UInt32 {
    fatalError("UInt16 not suported")
  }
}
