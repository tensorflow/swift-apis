import CTensorFlow

public protocol _AnyTensorHandle: class {
  var _tfeTensorHandle: TFETensorHandle { get }
  var rank: Int { get }
  var backend: Device.Backend { get }
}

extension _AnyTensorHandle {
  public var _cTensorHandle: CTensorHandle {
    return _tfeTensorHandle._cTensorHandle
  }
}

public class TFETensorHandle: _AnyTensorHandle {
  public let _cTensorHandle: CTensorHandle

  public var _tfeTensorHandle: TFETensorHandle { return self }

  public init(_owning base: CTensorHandle) {
    self._cTensorHandle = base
  }

  deinit {
    TFE_DeleteTensorHandle(_cTensorHandle)
  }

  @inlinable
  public var rank: Int {
    @_semantics("autodiff.nonvarying")
    get {
      let status = TF_NewStatus()
      defer { TF_DeleteStatus(status) }
      let rank = TFE_TensorHandleNumDims(_cTensorHandle, status)
      return Int(rank)
    }
  }

  public var backend: Device.Backend { .TF_EAGER }
}

public struct TensorHandle<Scalar> where Scalar: _TensorFlowDataTypeCompatible {
  @usableFromInline let handle: _AnyTensorHandle

  public var _cTensorHandle: CTensorHandle { handle._cTensorHandle }

  public init(_owning cTensorHandle: CTensorHandle) {
    self.handle = TFETensorHandle(_owning: cTensorHandle)
  }

  public init(handle: _AnyTensorHandle) {
    self.handle = handle
  }

  @usableFromInline
  init(copyingFromCTensor cTensor: CTensor) {
    let status = TF_NewStatus()
    let cTensorHandle = TFE_NewTensorHandle(cTensor, status)
    self.init(_owning: cTensorHandle!)
    TF_DeleteStatus(status)
  }

  @inlinable
  init(
    shape: [Int],
    byteCount: Int,
    bufferInitializer: (UnsafeMutableRawPointer) -> Void
  ) {
    let cTensor = TF_AllocateTensor(
      Scalar.tensorFlowDataType._cDataType,
      shape.map(Int64.init),
      Int32(shape.count),
      byteCount)!
    assert(TF_TensorByteSize(cTensor) == byteCount)
    bufferInitializer(TF_TensorData(cTensor))
    self.init(copyingFromCTensor: cTensor)
    TF_DeleteTensor(cTensor)
  }
}

extension TensorHandle where Scalar: TensorFlowScalar {
  @inlinable
  public init(
    shape: [Int],
    scalarsInitializer: (UnsafeMutablePointer<Scalar>) -> Void
  ) {
    let contiguousSize = shape.reduce(1, *)
    let byteCount = contiguousSize * MemoryLayout<Scalar>.stride
    self.init(
      shape: shape, byteCount: byteCount,
      bufferInitializer: { buffer in
        let pointer = buffer.bindMemory(to: Scalar.self, capacity: contiguousSize)
        scalarsInitializer(pointer)
      })
  }
}

extension TensorHandle {
  @inlinable
  public var rank: Int {
    @_semantics("autodiff.nonvarying")
    get { handle.rank }
  }

  @inlinable
  public var backend: Device.Backend {
    @_semantics("autodiff.nonvarying")
    get { handle.backend }
  }
}
