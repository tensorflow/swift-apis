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
  static func abs(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_abs(a.handle))
  }

  static func acos(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_acos(a.handle))
  }

  static func acosh(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_acosh(a.handle))
  }

  static func add(_ a: XLATensor, _ b: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    defer { _fixLifetime(b) }
    return XLATensor(_handle: XLATensor_add(a.handle, b.handle))
  }

  static func all(_ input: XLATensor, _ reductionIndices: [Int64], _ keepDims: Bool) -> XLATensor {
    defer { _fixLifetime(input) }
    return reductionIndices.withArrayRef { reductionIndices in
      XLATensor(_handle: XLATensor_all(input.handle, reductionIndices, keepDims))
    }
  }

  static func any(_ input: XLATensor, _ reductionIndices: [Int64], _ keepDims: Bool) -> XLATensor {
    defer { _fixLifetime(input) }
    return reductionIndices.withArrayRef { reductionIndices in
      XLATensor(_handle: XLATensor_any(input.handle, reductionIndices, keepDims))
    }
  }

  static func argmax(_ a: XLATensor, _ dim: Int64, _ keepdim: Bool) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_argmax(a.handle, dim, keepdim))
  }

  static func argmin(_ a: XLATensor, _ dim: Int64, _ keepdim: Bool) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_argmin(a.handle, dim, keepdim))
  }

  static func asin(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_asin(a.handle))
  }

  static func asinh(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_asinh(a.handle))
  }

  static func slice(_ a: XLATensor, _ dim: Int64, _ start: Int64, _ end: Int64, _ step: Int64)
    -> XLATensor
  {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_slice(a.handle, dim, start, end, step))
  }

  static func atan(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_atan(a.handle))
  }

  static func atanh(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_atanh(a.handle))
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

  static func cat(_ tensors: [XLATensor], _ dim: Int64) -> XLATensor {
    tensors.withArrayRef { tensors in
      XLATensor(_handle: XLATensor_cat(tensors, dim))
    }
  }

  static func ceil(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_ceil(a.handle))
  }

  static func clamp(_ input: XLATensor, _ min: XLATensor, _ max: XLATensor) -> XLATensor {
    defer { _fixLifetime(input) }
    defer { _fixLifetime(min) }
    defer { _fixLifetime(max) }
    return XLATensor(_handle: XLATensor_clamp(input.handle, min.handle, max.handle))
  }

  static func constantPad(_ input: XLATensor, _ pad: [Int64], _ value: XLAScalarType) -> XLATensor {
    defer { _fixLifetime(input) }
    return pad.withArrayRef { pad in
      XLATensor(_handle: XLATensor_constant_pad_nd(input.handle, pad, value.xlaScalar))
    }
  }

  static func cos(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_cos(a.handle))
  }

  static func cosh(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_cosh(a.handle))
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

  static func cumprod(
    _ a: XLATensor, _ dim: Int64, dtype: XLAScalarType.Type? = nil, exclusive: Bool = false,
    reverse: Bool = false
  ) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(
      _handle: XLATensor_cumprod(a.handle, dim, dtype.xlaOptionalType, exclusive, reverse))
  }

  static func cumsum(
    _ a: XLATensor, _ dim: Int64, dtype: XLAScalarType.Type? = nil, exclusive: Bool = false,
    reverse: Bool = false
  ) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(
      _handle: XLATensor_cumsum(a.handle, dim, dtype.xlaOptionalType, exclusive, reverse))
  }

  static func diagonal_value(
    _ a: XLATensor, _ offset: Int64, _ dim1: Int64, _ dim2: Int64
  ) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_diagonal_value(a.handle, offset, dim1, dim2))
  }

  static func div(_ a: XLATensor, _ b: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    defer { _fixLifetime(b) }
    return XLATensor(_handle: XLATensor_div(a.handle, b.handle))
  }

  static func dynamic_slice(_ base: XLATensor, _ start_indices: [XLATensor], _ slice_shape: [Int64]) -> XLATensor {
    start_indices.withArrayRef { start_indices in
      slice_shape.withArrayRef { slice_shape in
        return XLATensor(_handle: XLATensor_dynamic_slice(base.handle, start_indices, slice_shape))
      }
    }
  }

  static func dynamic_update_slice(
    _ base: XLATensor, _ update: XLATensor, _ start_indices: [XLATensor]
  ) -> XLATensor {
    start_indices.withArrayRef { start_indices in
      return XLATensor(
        _handle: XLATensor_dynamic_update_slice(base.handle, update.handle, start_indices))
    }
  }

  static func eq(_ a: XLATensor, _ b: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    defer { _fixLifetime(b) }
    return XLATensor(_handle: XLATensor_eq(a.handle, b.handle))
  }

  static func exp(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_exp(a.handle))
  }

  static func expand(_ a: XLATensor, _ dims: [Int64]) -> XLATensor {
    defer { _fixLifetime(a) }
    return dims.withArrayRef { dims in
      XLATensor(_handle: XLATensor_expand(a.handle, dims))
    }
  }

  static func expm1(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_expm1(a.handle))
  }

  static func floor(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_floor(a.handle))
  }

  static func flip(_ input: XLATensor, dims: [Int64]) -> XLATensor {
    defer { _fixLifetime(input) }
    return dims.withArrayRef { dims in
      XLATensor(_handle: XLATensor_flip(input.handle, dims))
    }
  }

  static func full(
    _ dims: [Int64],
    _ value: XLAScalarType,
    _ dtype: XLAScalarType.Type,
    _ device: Device
  ) -> XLATensor {
    let cdevice = device.cdevice
    return dims.withArrayRef { dims in
      XLATensor(
        _handle: XLATensor_full(dims, value.xlaScalar, cdevice, dtype.xlaTensorScalarType))
    }
  }

  static func ge(_ x: XLATensor, _ y: XLATensor) -> XLATensor {
    defer { _fixLifetime(x) }
    defer { _fixLifetime(y) }
    return XLATensor(_handle: XLATensor_ge(x.handle, y.handle))
  }

  static func gt(_ x: XLATensor, _ y: XLATensor) -> XLATensor {
    defer { _fixLifetime(x) }
    defer { _fixLifetime(y) }
    return XLATensor(_handle: XLATensor_gt(x.handle, y.handle))
  }

  static func index(_ input: XLATensor, _ indices: [XLATensor], _ startDim: Int64) -> XLATensor {
    defer { _fixLifetime(input) }
    return indices.withArrayRef { indices in
      XLATensor(_handle: XLATensor_index(input.handle, indices, startDim))
    }
  }

  static func isFinite(_ input: XLATensor) -> XLATensor {
    defer { _fixLifetime(input) }
    return XLATensor(_handle: XLATensor_is_finite(input.handle))
  }

  static func isInf(_ input: XLATensor) -> XLATensor {
    defer { _fixLifetime(input) }
    return XLATensor(_handle: XLATensor_is_inf(input.handle))
  }

  static func isNan(_ input: XLATensor) -> XLATensor {
    defer { _fixLifetime(input) }
    return XLATensor(_handle: XLATensor_is_nan(input.handle))
  }

  static func le(_ x: XLATensor, _ y: XLATensor) -> XLATensor {
    defer { _fixLifetime(x) }
    defer { _fixLifetime(y) }
    return XLATensor(_handle: XLATensor_le(x.handle, y.handle))
  }

  static func lt(_ x: XLATensor, _ y: XLATensor) -> XLATensor {
    defer { _fixLifetime(x) }
    defer { _fixLifetime(y) }
    return XLATensor(_handle: XLATensor_lt(x.handle, y.handle))
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

  static func log(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_log(a.handle))
  }

  static func log1p(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_log1p(a.handle))
  }

  static func log_softmax(_ a: XLATensor, _ dim: Int64) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_log_softmax(a.handle, dim))
  }

  static func log_softmax_backward(_ grad_output: XLATensor, _ output: XLATensor, _ dim: Int64)
    -> XLATensor
  {
    defer { _fixLifetime(grad_output) }
    defer { _fixLifetime(output) }
    return XLATensor(
      _handle: XLATensor_log_softmax_backward(grad_output.handle, output.handle, dim))
  }

  static func logicalAnd(_ a: XLATensor, _ b: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    defer { _fixLifetime(b) }
    return XLATensor(_handle: XLATensor_logicalAnd(a.handle, b.handle))
  }

  static func logicalCast(_ input: XLATensor, destType: XLATensorScalarType) -> XLATensor {
    defer { _fixLifetime(input) }
    return XLATensor(_handle: XLATensor_logical_cast(input.handle, destType))
  }

  static func logicalNot(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_logicalNot(a.handle))
  }

  static func logicalOr(_ a: XLATensor, _ b: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    defer { _fixLifetime(b) }
    return XLATensor(_handle: XLATensor_logicalOr(a.handle, b.handle))
  }

  static func matmul(_ a: XLATensor, _ b: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    defer { _fixLifetime(b) }
    return XLATensor(_handle: XLATensor_matmul(a.handle, b.handle))
  }

  static func max(_ a: XLATensor, _ dim: Int64, _ keepdim: Bool) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_max(a.handle, dim, keepdim))
  }

  static func maximum(_ a: XLATensor, _ b: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    defer { _fixLifetime(b) }
    return XLATensor(_handle: XLATensor_maximum(a.handle, b.handle))
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

  static func mean(
    _ a: XLATensor, _ dims: [Int64], _ keep_reduced_dimensions: Bool,
    _ dtype: XLAScalarType.Type? = nil
  ) -> XLATensor {
    defer { _fixLifetime(a) }
    return dims.withArrayRef { dims in
      XLATensor(
        _handle: XLATensor_mean(a.handle, dims, keep_reduced_dimensions, dtype.xlaOptionalType))
    }
  }

  static func min(_ a: XLATensor, _ dim: Int64, _ keepdim: Bool) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_min(a.handle, dim, keepdim))
  }

  static func minimum(_ a: XLATensor, _ b: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    defer { _fixLifetime(b) }
    return XLATensor(_handle: XLATensor_minimum(a.handle, b.handle))
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

  static func mul(_ a: XLATensor, _ b: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    defer { _fixLifetime(b) }
    return XLATensor(_handle: XLATensor_mul(a.handle, b.handle))
  }

  static func mm(_ a: XLATensor, _ b: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    defer { _fixLifetime(b) }
    return XLATensor(_handle: XLATensor_mm(a.handle, b.handle))
  }

  static func ne(_ a: XLATensor, _ b: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    defer { _fixLifetime(b) }
    return XLATensor(_handle: XLATensor_ne(a.handle, b.handle))
  }

  static func neg(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_neg(a.handle))
  }

  static func nll_loss(_ input: XLATensor, _ target: XLATensor, _ ignore_index: Int32) -> XLATensor
  {
    defer { _fixLifetime(input) }
    defer { _fixLifetime(target) }
    return XLATensor(_handle: XLATensor_nll_loss(input.handle, target.handle, ignore_index))
  }

  static func permute_value(_ value: XLATensor, _ dims: [Int64]) -> XLATensor {
    defer { _fixLifetime(value) }
    return dims.withArrayRef { dims in
      XLATensor(_handle: XLATensor_permute_value(value.handle, dims))
    }
  }

  static func physicalCast(_ input: XLATensor, destType: XLATensorScalarType) -> XLATensor {
    defer { _fixLifetime(input) }
    return XLATensor(_handle: XLATensor_physical_cast(input.handle, destType))
  }

  static func pow(_ base: XLATensor, _ exponent: XLATensor) -> XLATensor {
    defer { _fixLifetime(base) }
    defer { _fixLifetime(exponent) }
    return XLATensor(_handle: XLATensor_pow(base.handle, exponent.handle))
  }

  static func prod(
    _ a: XLATensor, _ dims: [Int64], _ keep_reduced_dimensions: Bool,
    _ dtype: XLAScalarType.Type? = nil
  ) -> XLATensor {
    defer { _fixLifetime(a) }
    return dims.withArrayRef { dims in
      XLATensor(
        _handle: XLATensor_prod(a.handle, dims, keep_reduced_dimensions, dtype.xlaOptionalType))
    }
  }

  static func qr(_ input: XLATensor, fullMatrices: Bool) -> (XLATensor, XLATensor) {
    defer { _fixLifetime(input) }
    let output = XLATensor_qr(input.handle, !fullMatrices)
    return (XLATensor(_handle: output.x), XLATensor(_handle: output.y))
  }

  static func rem(_ a: XLATensor, _ b: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_rem(a.handle, b.handle))
  }

  static func relu(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_relu(a.handle))
  }

  static func replica_id(_ device: Device) -> XLATensor {
    return XLATensor(_handle: XLATensor_replica_id(device.cdevice));
  }

  static func resize_value(_ value: XLATensor, _ dims: [Int64]) -> XLATensor {
    defer { _fixLifetime(value) }
    return dims.withArrayRef { dims in
      XLATensor(_handle: XLATensor_resize_value(value.handle, dims))
    }
  }

  static func round_to_even(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_round_to_even(a.handle))
  }

  static func rsqrt(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_rsqrt(a.handle))
  }

  static func select(_ a: XLATensor, _ dim: Int64, _ index: Int64) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_select(a.handle, dim, index))
  }

  static func sigmoid(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_sigmoid(a.handle))
  }

  static func sign(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_sign(a.handle))
  }

  static func sin(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_sin(a.handle))
  }

  static func sinh(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_sinh(a.handle))
  }

  static func softmax(_ a: XLATensor, _ dim: Int64) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_softmax(a.handle, dim))
  }

  static func splitWithSizes(_ input: XLATensor, _ splitSize: [Int64], _ dim: Int64) -> [XLATensor]
  {
    defer { _fixLifetime(input) }
    return splitSize.withArrayRef { splitSize in
      let tensorListHandle = XLATensor_split_with_sizes(input.handle, splitSize, dim)
      defer {
        destroyOpaqueXLATensorArrayRef(tensorListHandle)
      }
      return (0..<tensorListHandle.size).map { i in
        XLATensor(_handle: tensorListHandle.data[i]!)
      }
    }
  }

  static func sqrt(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_sqrt(a.handle))
  }

  static func squeeze(_ a: XLATensor, _ dim: Int64) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_squeeze(a.handle, dim))
  }

  static func stack(_ tensors: [XLATensor], _ dim: Int64) -> XLATensor {
    tensors.withArrayRef { tensors in
      XLATensor(_handle: XLATensor_stack(tensors, dim))
    }
  }

  static func irText(_ a: XLATensor) -> String {
    let str = XLATensor_ir_text(a.handle)
    defer { DeleteString(str) }
    return String(cString: GetStringCStr(str))
  }

  static func sub(_ a: XLATensor, _ b: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    defer { _fixLifetime(b) }
    return XLATensor(_handle: XLATensor_sub(a.handle, b.handle))
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

  static func tan(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_tan(a.handle))
  }

  static func tanh(_ a: XLATensor) -> XLATensor {
    defer { _fixLifetime(a) }
    return XLATensor(_handle: XLATensor_tanh(a.handle))
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
    let cdevice = device.cdevice
    return dims.withArrayRef { dims in
      XLATensor(
        _handle: XLATensor_tf_StatelessRandomUniform(
          dims, seeds.handle, minvalue.handle, maxvalue.handle, cdevice,
          dtype.xlaTensorScalarType))
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

  static func tile(_ input: XLATensor, repetitions: [Int64]) -> XLATensor {
    defer { _fixLifetime(input) }
    return repetitions.withArrayRef { repetitions in
      XLATensor(_handle: XLATensor_repeat(input.handle, repetitions))
    }
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

  static func updateSlice(input: XLATensor, source: XLATensor, baseIndices: [Int64]) -> XLATensor {
    defer { _fixLifetime(input) }
    return baseIndices.withArrayRef { baseIndices in
      XLATensor(_handle: XLATensor_update_slice(input.handle, source.handle, baseIndices))
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

  static func xlaSlice(_ input: XLATensor, begin: [Int64], end: [Int64], strides: [Int64])
    -> XLATensor
  {
    defer { _fixLifetime(input) }
    return begin.withArrayRef { begin in
      end.withArrayRef { end in
        strides.withArrayRef { strides in
          XLATensor(_handle: XLATensor_xla_slice(input.handle, begin, end, strides))
        }
      }
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
