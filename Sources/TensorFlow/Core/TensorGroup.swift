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

/// A protocol representing types that can be mapped to `Array<CTensorHandle>`.
///
/// This protocol is defined separately from `TensorGroup` in order for the number of tensors to be
/// determined at runtime. For example, `[Tensor<Float>]` may have an unknown number of elements at
/// compile time.
///
/// This protocol can be derived automatically for structs whose stored properties all conform to
/// the `TensorGroup` protocol. It cannot be derived automatically for structs whose properties all
/// conform to `TensorArrayProtocol` due to the constructor requirement (i.e., in such cases it
/// would be impossible to know how to break down `count` among the stored properties).
public protocol TensorArrayProtocol {
  /// Writes the tensor handles to `address`, which must be allocated with enough capacity to hold
  /// `_tensorHandleCount` handles. The tensor handles written to `address` are borrowed: this
  /// container still owns them.
  func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?)

  var _tensorHandleCount: Int32 { get }
  var _typeList: [TensorDataType] { get }
  var _tensorHandles: [_AnyTensorHandle] { get }

  init(_owning tensorHandles: UnsafePointer<CTensorHandle>?, count: Int)
  init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle
}

extension TensorArrayProtocol {
  public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle {
    let status = TF_NewStatus()
    defer { TF_DeleteStatus(status) }
    let buffer = UnsafeMutableBufferPointer<CTensorHandle>.allocate(capacity: _handles.count)
    defer { buffer.deallocate() }
    for (i, handle) in _handles.enumerated() {
      // Increment the reference count in TF.
      let handleCopy = TFE_TensorHandleCopySharingTensor(handle._cTensorHandle, status)
      checkOk(status)
      buffer[i] = handleCopy!
    }
    let baseAddress = UnsafeMutablePointer<OpaquePointer>(buffer.baseAddress)
    self.init(_owning: baseAddress, count: _handles.count)
  }

  public var _tensorHandles: [_AnyTensorHandle] {
    let status = TF_NewStatus()
    defer { TF_DeleteStatus(status) }
    let count = Int(_tensorHandleCount)
    let buffer = UnsafeMutableBufferPointer<CTensorHandle>.allocate(capacity: count)
    defer { buffer.deallocate() }
    self._unpackTensorHandles(into: buffer.baseAddress)
    let result: [TFETensorHandle] = (0..<count).map {
      let cTensorHandle = buffer[$0]
      // Increment the reference count in TF.
      let handleCopy = TFE_TensorHandleCopySharingTensor(cTensorHandle, status)
      checkOk(status)
      return TFETensorHandle(_owning: handleCopy!)
    }
    return result
  }
}

/// A protocol representing types that can be mapped to and from `Array<CTensorHandle>`.
///
/// When a `TensorGroup` is used as an argument to a tensor operation, it is passed as an argument
/// list whose elements are the tensor fields of the type.
///
/// When a `TensorGroup` is returned as a result of a tensor operation, it is initialized with its
/// tensor fields set to the tensor operation's tensor results.
public protocol TensorGroup: TensorArrayProtocol {

  /// The types of the tensor stored properties in this type.
  static var _typeList: [TensorDataType] { get }

  /// Initializes a value of this type, taking ownership of the `_tensorHandleCount` tensors
  /// starting at address `tensorHandles`.
  init(_owning tensorHandles: UnsafePointer<CTensorHandle>?)
}

extension TensorGroup {
  /// The number of tensor fields in this type.
  public static var _tensorHandleCount: Int32 { return Int32(Self._typeList.count) }

  /// An array of `nil`s with the same number of elements as `_outputTypeList`. The `nil`
  /// represents unknown shape.
  public static var _unknownShapeList: [TensorShape?] {
    return Array(repeating: nil, count: _typeList.count)
  }

  // The following instance properties are from `TensorArrayProtocol`.
  public var _tensorHandleCount: Int32 { return Int32(Self._typeList.count) }
  public var _typeList: [TensorDataType] { return Self._typeList }

  public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?, count: Int) {
    precondition(count == Self._typeList.count)
    self.init(_owning: tensorHandles)
  }
}

//===------------------------------------------------------------------------------------------===//
// TensorGroup Conformances
//===------------------------------------------------------------------------------------------===//

extension TensorHandle: TensorGroup {
  @inlinable
  public static var _unknownShapeList: [TensorShape?] {
    return [nil]
  }

  @inlinable
  public static var _typeList: [TensorDataType] {
    return [Scalar.tensorFlowDataType]
  }

  public var _tensorHandles: [_AnyTensorHandle] { [self.handle] }

  public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
    address!.initialize(to: _cTensorHandle)
  }

  public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
    self.init(_owning: tensorHandles!.pointee)
  }

  public init<C: RandomAccessCollection>(
    _handles: C
  ) where C.Element: _AnyTensorHandle {
    precondition(_handles.count == 1)
    self.init(handle: _handles[_handles.startIndex])
  }
}

extension ResourceHandle: TensorGroup {
  @inlinable
  public static var _unknownShapeList: [TensorShape?] {
    return [nil]
  }

  @inlinable
  public static var _typeList: [TensorDataType] {
    return [TensorDataType(TF_RESOURCE)]
  }

  public var _tensorHandles: [_AnyTensorHandle] { [self.handle] }

  public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
    address!.initialize(to: _cTensorHandle)
  }

  public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
    self.init(owning: tensorHandles!.pointee)
  }

  public init<C: RandomAccessCollection>(
    _handles: C
  ) where C.Element: _AnyTensorHandle {
    precondition(_handles.count == 1)
    self.init(handle: _handles[_handles.startIndex])
  }
}

extension VariantHandle: TensorGroup {
  @inlinable
  public static var _unknownShapeList: [TensorShape?] {
    return [nil]
  }

  @inlinable
  public static var _typeList: [TensorDataType] {
    return [TensorDataType(TF_VARIANT)]
  }

  public var _tensorHandles: [_AnyTensorHandle] { [self.handle] }

  public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
    address!.initialize(to: _cTensorHandle)
  }

  public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
    self.init(owning: tensorHandles!.pointee)
  }

  public init<C: RandomAccessCollection>(
    _handles: C
  ) where C.Element: _AnyTensorHandle {
    precondition(_handles.count == 1)
    self.init(handle: _handles[_handles.startIndex])
  }
}

extension Tensor: TensorGroup {
  @inlinable
  public static var _unknownShapeList: [TensorShape?] {
    return [nil]
  }

  @inlinable
  public static var _typeList: [TensorDataType] {
    return [Scalar.tensorFlowDataType]
  }

  public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
    address!.initialize(to: handle._cTensorHandle)
  }

  public var _tensorHandles: [_AnyTensorHandle] { [self.handle.handle] }

  public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
    self.init(handle: TensorHandle(_owning: tensorHandles!.pointee))
  }

  public init<C: RandomAccessCollection>(
    _handles: C
  ) where C.Element: _AnyTensorHandle {
    precondition(_handles.count == 1)
    self.init(handle: TensorHandle(handle: _handles[_handles.startIndex]))
  }
}

extension _TensorElementLiteral: TensorGroup {
  @inlinable
  public static var _unknownShapeList: [TensorShape?] {
    return [nil]
  }

  @inlinable
  public static var _typeList: [TensorDataType] {
    return [Scalar.tensorFlowDataType]
  }

  public var _tensorHandles: [_AnyTensorHandle] { tensor._tensorHandles }

  public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
    tensor._unpackTensorHandles(into: address)
  }

  public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
    tensor = Tensor(_owning: tensorHandles)
  }

  public init<C: RandomAccessCollection>(
    _handles: C
  ) where C.Element: _AnyTensorHandle {
    tensor = Tensor(_handles: _handles)
  }
}

extension StringTensor: TensorGroup {
  @inlinable
  public static var _unknownShapeList: [TensorShape?] {
    return [nil]
  }

  @inlinable
  public static var _typeList: [TensorDataType] {
    return [String.tensorFlowDataType]
  }

  public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
    address!.initialize(to: handle._cTensorHandle)
  }

  public var _tensorHandles: [_AnyTensorHandle] { [self.handle.handle] }

  public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
    self.init(handle: TensorHandle(_owning: tensorHandles!.pointee))
  }

  public init<C: RandomAccessCollection>(
    _handles: C
  ) where C.Element: _AnyTensorHandle {
    precondition(_handles.count == 1)
    self.init(handle: TensorHandle(handle: _handles[_handles.startIndex]))
  }
}

extension Array: TensorArrayProtocol where Element: TensorGroup {
  public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
    var ptr = address
    for elem in self {
      elem._unpackTensorHandles(into: ptr)
      ptr = ptr!.advanced(by: Int(elem._tensorHandleCount))
    }
  }

  public var _tensorHandleCount: Int32 {
    return Element._tensorHandleCount * Int32(count)
  }

  public var _typeList: [TensorDataType] {
    return [TensorDataType](
      [[TensorDataType]](
        repeating: Element._typeList,
        count: Int(count)
      ).joined())
  }

  public var _tensorHandles: ([_AnyTensorHandle]) {
    var result: [_AnyTensorHandle] = []
    result.reserveCapacity(Int(self._tensorHandleCount))
    for elem in self {
      result += elem._tensorHandles
    }
    return result
  }

  public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?, count: Int) {
    let size = count / Int(Element._tensorHandleCount)
    self = Array(
      (0..<size).map {
        Element.init(
          _owning: tensorHandles?.advanced(by: $0 * Int(Element._tensorHandleCount)))
      })
  }

  public init<C: RandomAccessCollection>(
    _handles: C
  ) where C.Element: _AnyTensorHandle {
    let size = _handles.count / Int(Element._tensorHandleCount)
    self = (0..<size).map {
      let start = _handles.index(
        _handles.startIndex, offsetBy: $0 * Int(Element._tensorHandleCount))
      let end = _handles.index(
        start, offsetBy: Int(Element._tensorHandleCount))
      return Element.init(_handles: _handles[start..<end])
    }
  }
}
