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
            let status = _ExecutionContext.global.status
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
            let status = _ExecutionContext.global.status
            let dims: [Int] = (0..<Int32(rank)).map { i in
                let dim = TFE_TensorHandleDim(_cTensorHandle, i, status)
                checkOk(status)
                return Int(dim)
            }
            return TensorShape(dims)
        }
    }
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

    /// Return true if the underlying tensor is concrete (as opposed to being symbolic).
    public var isConcrete: Bool {
        return TFE_TensorHandleIsConcrete(_cTensorHandle) != 0
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
        self.init(shape: shape, byteCount: byteCount, bufferInitializer: { buffer in
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
}

internal extension TensorHandle {
    /// Create a `ShapedArray` with contents of the underlying `TensorHandle`. If the `TensorHandle`
    /// is on the accelerator, it will be copied to the host.
    /// - Returns: A `ShapedArray`.
    @usableFromInline
    @inline(never)
    func makeHostCopy() -> ShapedArray<Scalar> {
        internalConsistencyCheck(isConcrete)
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
