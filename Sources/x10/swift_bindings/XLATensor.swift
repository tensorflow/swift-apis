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

@_implementationOnly import x10_xla_tensor_tf_ops
@_implementationOnly import x10_xla_tensor_wrapper

/// Type-erased tensor type on which the fundamental operators are implemented.
struct XLATensor {
  init(_handle: UnsafeMutablePointer<OpaqueXLATensor>) {
    handleDeleter = Handle(_handle: _handle)
  }

  init(_ handle: Handle) {
    handleDeleter = handle
  }

  init?(_ handle: _AnyTensorHandle) {
    if let handle = handle as? Handle {
      self.init(handle)
    } else {
      return nil
    }
  }

  /// The device on which `self` is allocated.
  public var device: Device {
    defer { _fixLifetime(self) }
    return XLATensor_device(handle).device
  }

  var handle: UnsafeMutablePointer<OpaqueXLATensor> {
    return handleDeleter.handle
  }

  // Implementation detail for deleting the pointer.
  class Handle: _AnyTensorHandle {
    init(_handle: UnsafeMutablePointer<OpaqueXLATensor>) {
      handle = _handle
    }

    deinit { destroyTensor(handle) }

    let handle: UnsafeMutablePointer<OpaqueXLATensor>
    var xlaTensor: XLATensor { XLATensor(self) }

    var _tfeTensorHandle: TFETensorHandle { fatalError("Not a tf handle") }
    var rank: Int { xlaTensor.shape.count }
    var shape: TensorShape { TensorShape(xlaTensor.shape) }

    public var backend: Device.Backend { .XLA }
  }

  var tensorHandle: _AnyTensorHandle { handleDeleter }

  let handleDeleter: Handle
}

extension Tensor {
  init(_xla: XLATensor) {
    precondition(
      _xla.dtype == Scalar.xlaTensorScalarType,
      "Type mismatch constructing from XLATensor:"
        + "\(_xla.dtype) vs \(Scalar.xlaTensorScalarType)")
    handle = TensorHandle(handle: _xla.tensorHandle)
  }

  init(_xlaHandle: UnsafeMutablePointer<OpaqueXLATensor>) {
    self.init(_xla: XLATensor(_handle: _xlaHandle))
  }

  var xlaHandle: UnsafeMutablePointer<OpaqueXLATensor> { return xlaTensor.handle }

  var xlaTensor: XLATensor {
    guard let xlaTensor = XLATensor(handle.handle) else {
      fatalError("Must be an XLATensor to convert to XlaTensor")
    }
    return xlaTensor
  }
}

extension XLATensor {
  /// TODO(parkers): Add support for other types and aliasing.
  static func make<Scalar: XLAScalarType>(
    _ data: [Scalar], _ dims: [Int], on device: Device = Device.default
  ) -> XLATensor {
    data.withUnsafeBufferPointer { data in return make(data, dims, on: device) }
  }

  static func make<Scalar: XLAScalarType>(_ data: Scalar, on device: Device = Device.default)
    -> XLATensor
  {
    return XLATensor(
      _handle: XLATensor_makeScalar(data.xlaScalar, Scalar.xlaTensorScalarType, device.cdevice))
  }

  static func make<Scalar: XLAScalarType>(
    _ data: UnsafeBufferPointer<Scalar>, _ dims: [Int], on device: Device = Device.default
  )
    -> XLATensor
  {
    dims.withUnsafeBufferPointer { dims in
      return XLATensor(
        _handle:
          copyTensor(
            Scalar.xlaTensorScalarType, data.baseAddress, data.count, dims.baseAddress, dims.count,
            device.cdevice
          ))
    }
  }

  static func make<Scalar: XLAScalarType>(
    _ data: [Scalar], _ dims: [Int], toReducedPrecision: Bool,
    directlyOn device: Device = Device.default
  ) -> XLATensor {
    data.withUnsafeBufferPointer { data in
      return make(data, dims, toReducedPrecision: toReducedPrecision, directlyOn: device)
    }
  }

  static func make<Scalar: XLAScalarType>(
    _ data: UnsafeBufferPointer<Scalar>, _ dims: [Int], toReducedPrecision: Bool,
    directlyOn device: Device = Device.default
  )
    -> XLATensor
  {
    dims.withUnsafeBufferPointer { dims in
      return XLATensor(
        _handle:
          copyTensorAndMakeResident(
            Scalar.xlaTensorScalarType, data.baseAddress, data.count, dims.baseAddress, dims.count,
            device.cdevice, toReducedPrecision
          ))
    }
  }

  var shape: [Int] {
    defer { _fixLifetime(self) }
    let shape = fetchTensorShape(handle)!
    let rank = XLAShape_getRank(shape)
    let data = XLAShape_getDimensions(shape)
    let result = Array(UnsafeBufferPointer(start: data!, count: rank))
    destroyXLAShape(shape)
    return result.map { Int($0) }
  }

  func fetchTensorValues<Scalar: XLAScalarType>(_ t: Scalar.Type) -> (data: [Scalar], dims: [Int]) {
    defer { _fixLifetime(self) }
    let materialized = XLATensor_materialize(handle)!
    let dims = shape
    let count = shape.reduce(1, *)
    precondition(
      MaterializedTensor_getType(materialized) == Scalar.xlaTensorScalarType,
      "Types mismatch when fetching tensor values.")
    let data = Array(
      UnsafeBufferPointer(
        start:
          UnsafePointer<Scalar>(OpaquePointer(MaterializedTensor_getData(materialized))),
        count: count))
    destroyMaterializedTensor(materialized)
    return (data: data, dims: dims)
  }

  var dtype: XLATensorScalarType {
    defer { _fixLifetime(self) }
    return XLATensor_dtype(handle)
  }
  var physicalScalarType: XLATensorScalarType {
    defer { _fixLifetime(self) }
    return XLATensor_physical_scalar_type(handle)
  }
}

extension Array where Element == Int64 {
  func withArrayRef<Result>(_ body: (Int64ArrayRef) throws -> Result) rethrows -> Result {
    return try withUnsafeBufferPointer { buf in
      return try body(Int64ArrayRef(data: buf.baseAddress, size: buf.count))
    }
  }
}

extension Array where Element == XLATensor {
  func withArrayRef<Result>(_ body: (OpaqueXLATensorArrayRef) throws -> Result) rethrows -> Result {
    defer { _fixLifetime(self) }
    return try map { $0.handle }.withUnsafeBufferPointer { buf in
      return try body(OpaqueXLATensorArrayRef(data: buf.baseAddress, size: buf.count))
    }
  }
}

extension Array where Element: AnyTensor {
  func withArrayRef<T, Result>(_ body: (OpaqueXLATensorArrayRef) throws -> Result) rethrows
    -> Result
  where Element == Tensor<T> {
    defer { _fixLifetime(self) }
    return try map { $0.xlaHandle }.withUnsafeBufferPointer { buf in
      return try body(OpaqueXLATensorArrayRef(data: buf.baseAddress, size: buf.count))
    }
  }
}

extension Array where Element == PaddingConfigDimension {
  func withArrayRef<Result>(_ body: (inout PaddingConfig) -> Result) -> Result {
    defer { _fixLifetime(self) }
    return withUnsafeBufferPointer {
      (_ dimensions: UnsafeBufferPointer<PaddingConfigDimension>) -> Result in
      var paddingConfig = PaddingConfig(dimensions: dimensions.baseAddress, count: count)
      return body(&paddingConfig)
    }
  }
}

extension Optional where Wrapped == XLAScalarType.Type {
  var xlaOptionalType: Optional_XLAScalarType {
    defer { _fixLifetime(self) }
    if let type = self {
      return Optional_XLAScalarType(has_value: true, type: type.xlaTensorScalarType)
    }
    return Optional_XLAScalarType(has_value: false, type: XLATensorScalarType(rawValue: 0))
  }
}

extension Tensor {
  public var xlaIrText: String {
    let str = XLATensor_xla_ir_text(xlaTensor.handle)
    defer { DeleteString(str) }
    return String(cString: GetStringCStr(str))
  }
  var placeholder: Tensor {
    return Tensor(_xlaHandle: XLATensor_makePlaceholder(self.xlaHandle, 0))
  }
}

extension Array where Element == AnyTensor {
  func withArrayRef<Result>(_ body: (OpaqueXLATensorArrayRef) throws -> Result) rethrows -> Result {
    try self.map { $0.scalarType.unwrapTensor($0) }.withArrayRef { try body($0) }
  }
}

extension TensorFlowScalar {
  static func unwrapTensor(_ t: AnyTensor) -> XLATensor {
    return (t as! Tensor<Self>).xlaTensor
  }
  static func wrapTensor(_ t: XLATensor) -> AnyTensor {
    return Tensor<Self>(_xla: t)
  }
  static func makePlaceholder(_ t: AnyTensor, i: Int = 0) -> AnyTensor {
    return Tensor<Self>(
      _xlaHandle: XLATensor_makePlaceholder((t as! Tensor<Self>).xlaHandle, Int32(i)))
  }
}

extension _RawXLA {
  public static func functionalWhile(
    n: Tensor<Int32>,
    initial: [AnyTensor],
    placeholders: [AnyTensor],
    indexPlaceholder: Tensor<Int32>,
    results: [AnyTensor]
  ) -> [AnyTensor] {
    initial.withArrayRef { initial in
      placeholders.withArrayRef { placeholders in
        results.withArrayRef { resultHandles in
          let tensorListHandle = XLATensor_functional_while(
            n.xlaHandle, initial, placeholders, indexPlaceholder.xlaHandle, resultHandles)
          defer { destroyOpaqueXLATensorArrayRef(tensorListHandle) }
          return (0..<tensorListHandle.size).map { i in
            results[i].scalarType.wrapTensor(XLATensor(_handle: tensorListHandle.data[i]!))
          }
        }
      }
    }
  }

  public static func functionalWhile(
    n: Tensor<Int32>, initial: [AnyTensor],
    body: ([AnyTensor], Tensor<Int32>) -> ([AnyTensor])
  ) -> [AnyTensor] {
    var idx = 0
    let placeholders = initial.map { (v: AnyTensor) -> AnyTensor in
      idx += 1
      return v.scalarType.makePlaceholder(v, i: idx)
    }
    let i = n.placeholder
    let results = body(placeholders, i)
    return functionalWhile(
      n: n, initial: initial, placeholders: placeholders,
      indexPlaceholder: i, results: results)
  }
}

/// Add more op wrappers here:
extension XLATensor {
  static func annotate(_ a: XLATensor, _ annotation: String) -> XLATensor {
    return XLATensor(_handle: XLATensor_annotate(a.handle, annotation))
  }

  static func annotations(_ a: XLATensor) -> String {
    // TODO(michellecasbon): Format with header.
    let str = XLATensor_get_annotations(a.handle)
    defer { DeleteString(str) }
    return String(cString: GetStringCStr(str))
  }

  static func avgpool(
    _ value: XLATensor,
    _ ksize: [Int64],
    _ strides: [Int64],
    _ padding: TFPadding,
    _ dataFormat: TFDataFormat
  ) -> XLATensor {
    defer { _fixLifetime(value) }
    return ksize.withArrayRef { ksize in
      strides.withArrayRef { strides in
        XLATensor(
          _handle: tf_AvgPool(value.handle, ksize, strides, padding, dataFormat))
      }
    }
  }

  static func avgpool_grad(
    _ origInputShape: [Int64],
    _ grad: XLATensor,
    _ ksize: [Int64],
    _ strides: [Int64],
    _ padding: TFPadding,
    _ dataFormat: TFDataFormat
  ) -> XLATensor {
    defer { _fixLifetime(grad) }
    return origInputShape.withArrayRef { origInputShape in
      ksize.withArrayRef { ksize in
        strides.withArrayRef { strides in
          XLATensor(
            _handle: tf_AvgPoolGrad(
              origInputShape, grad.handle, ksize, strides, padding, dataFormat))
        }
      }
    }
  }

  static func broadcast_tensors(_ a: XLATensor, _ b: XLATensor) -> (XLATensor, XLATensor) {
    defer { _fixLifetime(a) }
    defer { _fixLifetime(b) }
    let output = XLATensor_broadcast_tensors(a.handle, b.handle)
    return (XLATensor(_handle: output.x), XLATensor(_handle: output.y))
  }

  static func crossReplicaSum(_ inputs: [XLATensor], _ scale: Double) -> [XLATensor] {
    inputs.withArrayRef { inputs in
      let tensorListHandle = XLATensor_cross_replica_sum(inputs, scale)
      defer {
        destroyOpaqueXLATensorArrayRef(tensorListHandle)
      }
      return (0..<tensorListHandle.size).map { i in
        XLATensor(_handle: tensorListHandle.data[i]!)
      }
    }
  }

  static func irText(_ a: XLATensor) -> String {
    let str = XLATensor_ir_text(a.handle)
    defer { DeleteString(str) }
    return String(cString: GetStringCStr(str))
  }

  static func arange(
    _ start: XLAScalarType,
    _ stop: XLAScalarType,
    _ step: XLAScalarType,
    _ type: XLATensorScalarType,
    _ device: Device
  ) -> XLATensor {
    let cdevice = device.cdevice
    return XLATensor(
      _handle: XLATensor_arange(
        start.xlaScalar, stop.xlaScalar, step.xlaScalar, cdevice, type))
  }

  static func linspace(
    _ start: XLAScalarType,
    _ stop: XLAScalarType,
    _ num: Int64,
    _ type: XLATensorScalarType,
    _ device: Device
  ) -> XLATensor {
    let cdevice = device.cdevice
    return XLATensor(
      _handle: XLATensor_linspace(
        start.xlaScalar, stop.xlaScalar, num, cdevice, type))
  }

  static func maxpool(
    _ input: XLATensor,
    _ ksize: [Int64],
    _ strides: [Int64],
    _ padding: TFPadding,
    _ dataFormat: TFDataFormat
  ) -> XLATensor {
    defer { _fixLifetime(input) }
    return ksize.withArrayRef { ksize in
      strides.withArrayRef { strides in
        XLATensor(
          _handle: tf_MaxPool(input.handle, ksize, strides, padding, dataFormat))
      }
    }
  }

  static func maxpool_grad(
    _ input: XLATensor,
    _ grad: XLATensor,
    _ ksize: [Int64],
    _ strides: [Int64],
    _ padding: TFPadding
  ) -> XLATensor {
    defer { _fixLifetime(input) }
    defer { _fixLifetime(grad) }
    return ksize.withArrayRef { ksize in
      strides.withArrayRef { strides in
        XLATensor(
          _handle: tf_MaxPoolGrad(input.handle, grad.handle, ksize, strides, padding))
      }
    }
  }

  static func replica_id(_ device: Device) -> XLATensor {
    return XLATensor(_handle: XLATensor_replica_id(device.cdevice))
  }

  static func to(
    _ a: XLATensor, _ device: Device?, _ dtype: XLAScalarType.Type?
  ) -> XLATensor {
    defer { _fixLifetime(a) }
    if var cdevice = device?.cdevice {
      return XLATensor(_handle: XLATensor_to(a.handle, &cdevice, dtype.xlaOptionalType))
    } else {
      return XLATensor(_handle: XLATensor_to(a.handle, nil, dtype.xlaOptionalType))
    }
  }

  struct StridedSliceSpec {
    let begin: [Int64]
    let end: [Int64]
    let strides: [Int64]
    let processingSizes: [Int64]
    let finalSizes: [Int64]
  }

  static func computeIndexingBoundsAndStrides(
    inputSizes: [Int64], begin: [Int64], end: [Int64], strides: [Int64], beginMask: Int32,
    endMask: Int32, ellipsisMask: Int32, newAxisMask: Int32, shrinkAxisMask: Int32
  ) -> StridedSliceSpec {
    inputSizes.withArrayRef { inputSizes in
      begin.withArrayRef { begin in
        end.withArrayRef { end in
          strides.withArrayRef { strides in
            let stridedSliceSpec = ComputeIndexingBoundsAndStrides(
              inputSizes, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask,
              shrinkAxisMask)!
            defer { destroyStridedSliceSpec(stridedSliceSpec) }
            return StridedSliceSpec(
              begin: arrayFromInt64ArrayRef(stridedSliceSpec.pointee.begin),
              end: arrayFromInt64ArrayRef(stridedSliceSpec.pointee.end),
              strides: arrayFromInt64ArrayRef(stridedSliceSpec.pointee.strides),
              processingSizes: arrayFromInt64ArrayRef(stridedSliceSpec.pointee.processing_sizes),
              finalSizes: arrayFromInt64ArrayRef(stridedSliceSpec.pointee.final_sizes)
            )
          }
        }
      }
    }
  }

  private static func arrayFromInt64ArrayRef(_ arrRef: Int64ArrayRef) -> [Int64] {
    (0..<arrRef.size).map { i in arrRef.data[i] }
  }

  // Currently only used for deterministic testing.
  static func rand(_ dims: [Int64], _ seed: Int64) -> XLATensor {
    dims.withArrayRef { dims in
      XLATensor(_handle: XLATensor_rand(dims, seed))
    }
  }
}

public func PrintX10Metrics() {
  PrintMetrics()
}
