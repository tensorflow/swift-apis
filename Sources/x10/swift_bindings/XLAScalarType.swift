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
