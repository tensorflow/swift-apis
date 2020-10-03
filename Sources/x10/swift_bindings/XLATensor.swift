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
  func withPaddingConfig<Result>(_ body: (inout PaddingConfig) -> Result) -> Result {
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

  static func constantPad(_ input: XLATensor, _ pad: [Int64], _ value: XLAScalarType) -> XLATensor {
    defer { _fixLifetime(input) }
    return pad.withArrayRef { pad in
      XLATensor(_handle: XLATensor_constant_pad_nd(input.handle, pad, value.xlaScalar))
    }
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

  static func logicalCast(_ input: XLATensor, destType: XLATensorScalarType) -> XLATensor {
    defer { _fixLifetime(input) }
    return XLATensor(_handle: XLATensor_logical_cast(input.handle, destType))
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

  static func mirrorPad(_ input: XLATensor, _ padding: [Int64], _ mode: TFMirrorPadMode)
    -> XLATensor
  {
    defer { _fixLifetime(input) }
    return padding.withArrayRef { padding in
      XLATensor(_handle: XLATensor_tf_MirrorPad(input.handle, padding, mode))
    }
  }

  static func mirrorPadGrad(
    _ grad_output: XLATensor, _ inputSize: [Int64], _ padding: [Int64], _ mode: TFMirrorPadMode
  )
    -> XLATensor
  {
    defer { _fixLifetime(grad_output) }
    return inputSize.withArrayRef { inputSize in
      padding.withArrayRef { padding in
        XLATensor(
          _handle: XLATensor_tf_MirrorPadGrad(grad_output.handle, inputSize, padding, mode))
      }
    }
  }

  static func physicalCast(_ input: XLATensor, destType: XLATensorScalarType) -> XLATensor {
    defer { _fixLifetime(input) }
    return XLATensor(_handle: XLATensor_physical_cast(input.handle, destType))
  }

  static func qr(_ input: XLATensor, fullMatrices: Bool) -> (XLATensor, XLATensor) {
    defer { _fixLifetime(input) }
    let output = XLATensor_qr(input.handle, !fullMatrices)
    return (XLATensor(_handle: output.x), XLATensor(_handle: output.y))
  }

  static func replica_id(_ device: Device) -> XLATensor {
    return XLATensor(_handle: XLATensor_replica_id(device.cdevice))
  }

  static func sum(
    _ a: XLATensor, _ dims: [Int64], _ keep_reduced_dimensions: Bool,
    _ dtype: XLAScalarType.Type? = nil
  ) -> XLATensor {
    defer { _fixLifetime(a) }
    return dims.withArrayRef { dims in
      XLATensor(
        _handle: XLATensor_sum(a.handle, dims, keep_reduced_dimensions, dtype.xlaOptionalType))
    }
  }

  static func svd(_ input: XLATensor, computeUv: Bool, fullMatrices: Bool) -> (
    XLATensor, XLATensor, XLATensor
  ) {
    defer { _fixLifetime(input) }
    let output = XLATensor_svd(input.handle, computeUv, fullMatrices)
    return (
      XLATensor(_handle: output.v0), XLATensor(_handle: output.v1), XLATensor(_handle: output.v2)
    )
  }

  static func topk(_ a: XLATensor, k: Int64, dim: Int64, largest: Bool) -> (XLATensor, XLATensor) {
    defer { _fixLifetime(a) }
    let output = XLATensor_topk(a.handle, k, dim, largest)
    return (XLATensor(_handle: output.x), XLATensor(_handle: output.y))
  }

  static func tf_Conv(
    _ input: XLATensor, _ filter: XLATensor, _ depthwise: Bool, _ strides: [Int64],
    _ padding: TFPadding, _ explicit_paddings: [Int64],
    _ data_format: TFDataFormat, _ dilations: [Int64]
  ) -> XLATensor {
    defer { _fixLifetime(input) }
    defer { _fixLifetime(filter) }
    return strides.withArrayRef { strides in
      explicit_paddings.withArrayRef { explicit_paddings in
        dilations.withArrayRef { dilations in
          XLATensor(
            _handle: XLATensor_tf_Conv(
              input.handle, filter.handle, depthwise, strides, padding, explicit_paddings,
              data_format, dilations))
        }
      }
    }
  }

  static func tf_ConvBackpropFilter(
    _ input: XLATensor, _ filter_sizes: [Int64], _ out_backprop: XLATensor, _ depthwise: Bool,
    _ strides: [Int64], _ padding: TFPadding, _ explicit_paddings: [Int64],
    _ data_format: TFDataFormat, _ dilations: [Int64]
  ) -> XLATensor {
    defer { _fixLifetime(input) }
    defer { _fixLifetime(out_backprop) }
    return filter_sizes.withArrayRef { filter_sizes in
      strides.withArrayRef { strides in
        explicit_paddings.withArrayRef { explicit_paddings in
          dilations.withArrayRef { dilations in
            XLATensor(
              _handle: XLATensor_tf_ConvBackpropFilter(
                input.handle, filter_sizes, out_backprop.handle, depthwise, strides,
                padding, explicit_paddings, data_format, dilations))
          }
        }
      }
    }
  }

  static func tf_ConvBackpropInput(
    _ input_sizes: [Int64], _ filter: XLATensor, _ out_backprop: XLATensor, _ depthwise: Bool,
    _ strides: [Int64], _ padding: TFPadding, _ explicit_paddings: [Int64],
    _ data_format: TFDataFormat, _ dilations: [Int64]
  ) -> XLATensor {
    defer { _fixLifetime(filter) }
    defer { _fixLifetime(out_backprop) }
    return input_sizes.withArrayRef { input_sizes in
      strides.withArrayRef { strides in
        explicit_paddings.withArrayRef { explicit_paddings in
          dilations.withArrayRef { dilations in
            XLATensor(
              _handle: XLATensor_tf_ConvBackpropInput(
                input_sizes, filter.handle, out_backprop.handle, depthwise, strides,
                padding, explicit_paddings, data_format, dilations))
          }
        }
      }
    }
  }

  static func tf_OneHot(
    _ indices: XLATensor, _ on_value: XLATensor, _ off_value: XLATensor, _ depth: Int64,
    _ axis: Int64
  ) -> XLATensor {
    defer { _fixLifetime(indices) }
    defer { _fixLifetime(on_value) }
    defer { _fixLifetime(off_value) }
    return XLATensor(
      _handle: XLATensor_tf_OneHot(indices.handle, on_value.handle, off_value.handle, depth, axis))
  }

  static func tf_StatelessRandomNormal(
    _ dims: [Int64],
    _ seeds: XLATensor,
    _ dtype: XLAScalarType.Type,
    _ device: Device
  ) -> XLATensor {
    defer { _fixLifetime(seeds) }
    let cdevice = device.cdevice
    return dims.withArrayRef { dims in
      XLATensor(
        _handle: XLATensor_tf_StatelessRandomNormal(
          dims, seeds.handle, cdevice,
          dtype.xlaTensorScalarType))
    }
  }

  static func tf_StatelessRandomUniform(
    _ dims: [Int64],
    _ seeds: XLATensor,
    _ minvalue: XLATensor,
    _ maxvalue: XLATensor,
    _ dtype: XLAScalarType.Type,
    _ device: Device
  ) -> XLATensor {
    defer { _fixLifetime(seeds) }
    return dims.withArrayRef { dims in
      XLATensor(
        _handle: XLATensor_tf_StatelessRandomUniform(
          dims, seeds.handle, minvalue.handle, maxvalue.handle))
    }
  }

  static func tf_UnsortedSegmentSum(
    _ data: XLATensor, _ indices: XLATensor, _ numSegments: Int64
  ) -> XLATensor {
    defer { _fixLifetime(data) }
    defer { _fixLifetime(indices) }
    return XLATensor(
      _handle: XLATensor_tf_UnsortedSegmentSum(data.handle, indices.handle, numSegments))
  }

  static func threshold_backward(_ grad_output: XLATensor, _ input: XLATensor, _ threshold: Float)
    -> XLATensor
  {
    defer { _fixLifetime(grad_output) }
    defer { _fixLifetime(input) }
    return XLATensor(
      _handle: XLATensor_threshold_backward(grad_output.handle, input.handle, threshold))
  }

  static func truncatedNormal(_ input: XLATensor) -> XLATensor {
    defer { _fixLifetime(input) }
    return XLATensor(_handle: XLATensor_truncated_normal(input.handle))
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

  static func where_(
    _ condition: XLATensor, _ input: XLATensor, _ other: XLATensor
  ) -> XLATensor {
    defer { _fixLifetime(input) }
    defer { _fixLifetime(other) }
    defer { _fixLifetime(condition) }
    return XLATensor(_handle: XLATensor_where(condition.handle, input.handle, other.handle))
  }

  static func xlaPad(
    _ input: XLATensor, paddingValue: XLAScalarType, paddingConfig: [PaddingConfigDimension]
  ) -> XLATensor {
    defer { _fixLifetime(input) }
    return paddingConfig.withPaddingConfig { paddingConfig in
      XLATensor(_handle: XLATensor_xla_pad(input.handle, paddingValue.xlaScalar, paddingConfig))
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
