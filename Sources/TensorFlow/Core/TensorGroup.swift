// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

public extension TensorGroup {
    /// The number of tensor fields in this type.
    static var _tensorHandleCount: Int32 { return Int32(Self._typeList.count) }

    /// An array of `nil`s with the same number of elements as `_outputTypeList`. The `nil`
    /// represents unknown shape.
    static var _unknownShapeList: [TensorShape?] {
        return Array(repeating: nil, count: _typeList.count)
    }
    
    // The following instance properties are from `TensorArrayProtocol`.
    var _tensorHandleCount: Int32 { return Int32(Self._typeList.count) }
    var _typeList: [TensorDataType] { return Self._typeList }

    init(_owning tensorHandles: UnsafePointer<CTensorHandle>?, count: Int) {
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

    public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
        address!.initialize(to: _cTensorHandle)
    }

    public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
        self.init(_owning: tensorHandles!.pointee)
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

    public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
        address!.initialize(to: _cTensorHandle)
    }

    public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
        self.init(owning: tensorHandles!.pointee)
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

    public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
        address!.initialize(to: _cTensorHandle)
    }

    public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
        self.init(owning: tensorHandles!.pointee)
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

    public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
        self.init(handle: TensorHandle(_owning: tensorHandles!.pointee))
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

    public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
        address!.initialize(to: handle._cTensorHandle)
    }

    public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
        self.init(handle: TensorHandle(_owning: tensorHandles!.pointee))
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

    public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
        self.init(handle: TensorHandle(_owning: tensorHandles!.pointee))
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
        return Array<TensorDataType>([[TensorDataType]](
            repeating: Element._typeList,
            count: Int(count)).joined())
    }

    public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?, count: Int) {
        let size = count / Int(Element._tensorHandleCount)
        self = Array((0..<size).map { Element.init(
            _owning: tensorHandles?.advanced(by: $0 * Int(Element._tensorHandleCount)))
        })
    }
}
