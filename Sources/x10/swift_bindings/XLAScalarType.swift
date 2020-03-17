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

@_implementationOnly
import x10_xla_tensor_wrapper

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

/// A supported datatype in x10.
public protocol XLAScalarType {
  var xlaScalar: XLAScalar { get }
  static var xlaTensorScalarType: XLATensorScalarType { get }
}

extension Float: XLAScalarType {
  public var xlaScalar: XLAScalar { XLAScalar(Double(self)) }

  static public var xlaTensorScalarType: XLATensorScalarType {
    return XLATensorScalarType_Float
  }
}
extension Double: XLAScalarType {
  public var xlaScalar: XLAScalar { XLAScalar(self) }

  static public var xlaTensorScalarType: XLATensorScalarType {
    return XLATensorScalarType_Double
  }
}
extension Int64: XLAScalarType {
  public var xlaScalar: XLAScalar { XLAScalar(self) }

  static public var xlaTensorScalarType: XLATensorScalarType {
    return XLATensorScalarType_Int64
  }
}
extension Int32: XLAScalarType {
  public var xlaScalar: XLAScalar { XLAScalar(Int64(self)) }

  static public var xlaTensorScalarType: XLATensorScalarType {
    return XLATensorScalarType_Int32
  }
}
extension Int16: XLAScalarType {
  public var xlaScalar: XLAScalar { XLAScalar(Int64(self)) }

  static public var xlaTensorScalarType: XLATensorScalarType {
    return XLATensorScalarType_Int16
  }
}
extension Int8: XLAScalarType {
  public var xlaScalar: XLAScalar { XLAScalar(Int64(self)) }

  static public var xlaTensorScalarType: XLATensorScalarType {
    return XLATensorScalarType_Int8
  }
}

extension Bool: XLAScalarType {
  public var xlaScalar: XLAScalar { XLAScalar(Int64(self ? 1 : 0)) }

  static public var xlaTensorScalarType: XLATensorScalarType {
    return XLATensorScalarType_Bool
  }
}
