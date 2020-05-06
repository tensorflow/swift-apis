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

/// This protocol abstracts the underlying representation of a tensor. Any type
/// that conforms to this protocol can be used as a `TensorHandle` in the
/// `TensorFlow` library, as it much provide a way to convert the underlying tensor
/// handle into a `ConcreteTensorHandle`, which wraps a `TFE_TensorHandle *`
/// TODO(https://bugs.swift.org/browse/TF-527): This is defined as a class-bound
// protocol to workaround bug TF-527. When it is fixed, we should remove `: class`.
public protocol _AnyTensorHandle: class {
  var _tfeTensorHandle: TFETensorHandle { get }
  var rank: Int { get }
  var shape: TensorShape { get }
  var backend: Device.Backend { get }
}

extension _AnyTensorHandle {
  /// The underlying `TFE_TensorHandle *`.
  public var _cTensorHandle: CTensorHandle {
    return _tfeTensorHandle._cTensorHandle
  }
}

/// Class wrapping a C pointer to a TensorHandle.  This class owns the
/// TensorHandle and is responsible for destroying it.
public class TFETensorHandle: _AnyTensorHandle {
  public let _cTensorHandle: CTensorHandle

  public var _tfeTensorHandle: TFETensorHandle { return self }

  public init(_owning base: CTensorHandle) {
    Context.local.globalTensorCount += 1
    self._cTensorHandle = base
  }

  deinit {
    debugLog("De-initializing TensorHandle.")
    TFE_DeleteTensorHandle(_cTensorHandle)
    Context.local.globalTensorCount -= 1
    debugLog("Returning from deinit of TensorHandle.")
  }

  /// The number of dimensions of the underlying `Tensor`.
  @inlinable
  public var rank: Int {
    @_semantics("autodiff.nonvarying")
    get {
      let status = TF_NewStatus()
      defer { TF_DeleteStatus(status) }
      let rank = TFE_TensorHandleNumDims(_cTensorHandle, status)
      checkOk(status)
      return Int(rank)
    }
  }

  /// The shape of the underlying `Tensor`.
  @inlinable
  public var shape: TensorShape {
    @_semantics("autodiff.nonvarying")
    get {
      let status = TF_NewStatus()
      defer { TF_DeleteStatus(status) }
      let dims: [Int] = (0..<Int32(rank)).map { i in
        let dim = TFE_TensorHandleDim(_cTensorHandle, i, status)
        checkOk(status)
        return Int(dim)
      }
      return TensorShape(dims)
    }
  }

  public var backend: Device.Backend { .TF_EAGER }
}

/// `TensorHandle` is the type used by ops. It includes a `Scalar` type, which
/// compiler internals can use to determine the datatypes of parameters when
/// they are extracted into a tensor program.
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
    checkOk(status)
    self.init(_owning: cTensorHandle!)
    TF_DeleteStatus(status)
  }

  /// Create a `TensorHandle` with a closure that initializes the underlying buffer.
  ///
  /// Users initializing `TensorHandle`s with non-`String` scalars should use the
  /// `init(shape:scalarsInitializer:)` initializer instead of this one. It enforces additional
  /// constraints on the buffer that hold for all non-`String` scalars.
  ///
  /// `bufferInitializer` receives a buffer with exactly `byteCount` bytes of capacity.
  /// `bufferInitializer` must initialize the entire buffer.
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
  /// Create a `TensorHandle` with a closure that initializes the underlying buffer.
  ///
  /// `scalarsInitializer` receives a buffer with exactly enough capacity to hold the scalars in a
  /// tensor with shape `shape`. `scalarsInitializer` must initialize the entire buffer, with
  /// contiguous scalars in row-major order.
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
  /// The number of dimensions of the `Tensor`.
  @inlinable
  public var rank: Int {
    @_semantics("autodiff.nonvarying")
    get { handle.rank }
  }

  /// The shape of the `Tensor`.
  @inlinable
  public var shape: TensorShape {
    @_semantics("autodiff.nonvarying")
    get { handle.shape }
  }

  /// The backend used to dispatch ops.
  @inlinable
  public var backend: Device.Backend {
    @_semantics("autodiff.nonvarying")
    get { handle.backend }
  }
}

extension TensorHandle {
  /// Create a `ShapedArray` with contents of the underlying `TensorHandle`. If the `TensorHandle`
  /// is on the accelerator, it will be copied to the host.
  /// - Returns: A `ShapedArray`.
  @usableFromInline
  @inline(never)
  func makeHostCopy() -> ShapedArray<Scalar> {
    debugLog("Calling makeHostCopy() with c handle \(_cTensorHandle)")
    return ShapedArray(cTensorHandle: _cTensorHandle)
  }
}

public struct ResourceHandle {
  let handle: _AnyTensorHandle

  @usableFromInline
  var _cTensorHandle: CTensorHandle { handle._cTensorHandle }

  @usableFromInline
  init(owning cTensorHandle: CTensorHandle) {
    self.handle = TFETensorHandle(_owning: cTensorHandle)
  }

  @usableFromInline
  init(handle: _AnyTensorHandle) {
    self.handle = handle
  }
}

public struct VariantHandle {
  let handle: _AnyTensorHandle

  @usableFromInline
  var _cTensorHandle: CTensorHandle { handle._cTensorHandle }

  @usableFromInline
  init(owning cTensorHandle: CTensorHandle) {
    self.handle = TFETensorHandle(_owning: cTensorHandle)
  }

  @usableFromInline
  init(handle: _AnyTensorHandle) {
    self.handle = handle
  }
}

//===------------------------------------------------------------------------------------------===//
// TensorBuffer based on a C `TF_Tensor*`.
//===------------------------------------------------------------------------------------------===//

// TF Tensor-specific initializer.
internal class CTensorTensorBuffer<Scalar>: TensorBuffer<Scalar> {
  let cTensor: CTensor

  /// Creates a local tensor buffer from a C `TF_Tensor*` value and takes ownership of the value.
  init(owning cTensor: CTensor, count: Int) {
    debugLog("Initializing TensorBuffer with a cTensor of \(count) elements.")
    let actualCount = (0..<TF_NumDims(cTensor)).reduce(1) { accumulator, next in
      accumulator * Int(TF_Dim(cTensor, next))
    }
    assert(actualCount == count)
    self.cTensor = cTensor
    super.init(count: count)
  }

  override func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Scalar>) throws -> R
  ) rethrows -> R {
    let startAddress = TF_TensorData(cTensor).assumingMemoryBound(to: Scalar.self)
    let bufferPointer = UnsafeBufferPointer(start: startAddress, count: count)
    return try body(bufferPointer)
  }

  override func withUnsafeMutableBufferPointer<R>(
    _ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> R
  ) rethrows -> R {
    let startAddress = TF_TensorData(cTensor).assumingMemoryBound(to: Scalar.self)
    var bufferPointer = UnsafeMutableBufferPointer(start: startAddress, count: count)
    return try body(&bufferPointer)
  }

  deinit {
    TF_DeleteTensor(cTensor)
  }
}

extension ShapedArray where Scalar: _TensorFlowDataTypeCompatible {
  @usableFromInline
  init(owning cTensor: CTensor) {
    // Including \(Scalar.self) into the message would cause non-deterministic crashes.
    debugLog("Initializing ShapedArray from CTensor.")
    let shape = (0..<TF_NumDims(cTensor)).map { Int(TF_Dim(cTensor, $0)) }
    if _RuntimeConfig.printsDebugLog {
      // Without this local variable, passing the string directly into debugLog() would not
      // work, because 'self' is captured by the auto closure param in debugLog().
      let shapeStr = "The shape is \(shape)."
      debugLog(shapeStr)
    }
    self.init(
      buffer: CTensorTensorBuffer<Scalar>(owning: cTensor, count: shape.reduce(1, *)),
      shape: shape)
    debugLog("Done initializing ShapedArray from CTensor.")
  }

  @usableFromInline
  @inline(never)
  init(cTensorHandle: CTensorHandle) {
    let status = TF_NewStatus()
    let cTensor = TFE_TensorHandleResolve(cTensorHandle, status)
    checkOk(status)
    TF_DeleteStatus(status)
    internalConsistencyCheck(cTensor != nil)
    debugLog("# of dims is \(TF_NumDims(cTensor!))")
    debugLog("Returning a shaped array.")
    self.init(owning: cTensor!)
  }
}

// Tensor conversion.
extension Tensor {
  public init(_ array: __owned ShapedArray<Scalar>, on device: Device = .default) {
    precondition(
      array.rank <= Int(Int32.max),
      "Conversion to TensorHandle is undefined when rank exceeds `Int32.max`.")
    precondition(
      array.shape.allSatisfy { $0 <= Int(Int32.max) },
      "Conversion to TensorHandle is undefined when shape dimensions exceed `Int32.max`.")
    if let buffer = array.buffer as? CTensorTensorBuffer<Scalar> {
      #if USING_X10_BACKEND
        let tmp = Tensor(handle: TensorHandle(copyingFromCTensor: buffer.cTensor))
        self = tmp.device == device ? tmp : Tensor(copying: tmp, to: device)
      #else
        self = Tensor(handle: TensorHandle(copyingFromCTensor: buffer.cTensor))
      #endif
    } else {
      self = array.buffer.withUnsafeBufferPointer { buffer in
        return Tensor(shape: TensorShape(array.shape), scalars: buffer, on: device)
      }
    }
  }
}
