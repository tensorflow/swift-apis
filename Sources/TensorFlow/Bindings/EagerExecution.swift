// !!! THIS CODE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND !!!
//
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

/// **WARNING:** After constructing a `TFE_Op`, any one of its `execute` methods must be called
/// *exactly once*. If not called, then a memory leak is introduced due to the underlying TensorFlow
/// eager op object not being freed. If called more than once, then a SEGFAULT may occur due to
/// trying to execute a TensorFlow eager op that has already been freed.
@usableFromInline
internal struct TFE_Op: TFTensorOperation {
    @usableFromInline internal let status: CTFStatus
    @usableFromInline internal let op: CTFEOp
    @usableFromInline internal let outputCount: Int

    @usableFromInline
    internal init(_ name: String, _ outputCount: Int) {
        self.status = TF_NewStatus()
        self.op = TFE_NewOp(_ExecutionContext.global.eagerContext, name, status)
        self.outputCount = outputCount
    }

    @inlinable @inline(__always)
    internal func addInput(_ input: _AnyTensorHandle) {
        TFE_OpAddInput(op, input._cTensorHandle, status)
        checkOk(status)
    }

    @inlinable @inline(__always)
    internal func addInput<Scalar: TensorFlowScalar>(_ input: Tensor<Scalar>) {
        TFE_OpAddInput(op, input.handle._cTensorHandle, status)
        checkOk(status)
    }

    @inlinable @inline(__always)
    internal func addInput(_ input: StringTensor) {
        TFE_OpAddInput(op, input.handle._cTensorHandle, status)
        checkOk(status)
    }

    @inlinable @inline(__always)
    internal func addInput(_ input: ResourceHandle) {
        TFE_OpAddInput(op, input._cTensorHandle, status)
        checkOk(status)
    }

    @inlinable @inline(__always)
    internal func addInput(_ input: VariantHandle) {
        TFE_OpAddInput(op, input._cTensorHandle, status)
        checkOk(status)
    }

    @inlinable @inline(__always)
    internal func addInputList<T: TensorArrayProtocol>(_ input: T) {
        let count = input._tensorHandleCount
        var buffer = UnsafeMutableBufferPointer<CTensorHandle>.allocate(capacity: Int(count))
        defer { buffer.deallocate() }
        let pointer = UnsafeMutablePointer<OpaquePointer?>(buffer.baseAddress)
        input._unpackTensorHandles(into: buffer.baseAddress)
        TFE_OpAddInputList(op, pointer, count, status)
        // TODO: checkOk(status)
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: Bool) {
        TFE_OpSetAttrBool(op, name, value ? 1 : 0)
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: Int) {
        TFE_OpSetAttrInt(op, name, Int64(value))
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: Int32) {
        TFE_OpSetAttrInt(op, name, Int64(value))
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: Int64) {
        TFE_OpSetAttrInt(op, name, value)
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: Float) {
        TFE_OpSetAttrFloat(op, name, value)
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: Double) {
        TFE_OpSetAttrFloat(op, name, Float(value))
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: String) {
        value.utf8CString.withUnsafeBufferPointer { buffer in
            // utf8CString is null-terminated; TFE_OpSetAttrString wants non-null-terminated.
            TFE_OpSetAttrString(op, name, buffer.baseAddress, buffer.count - 1)
        }
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: TensorDataType) {
        TFE_OpSetAttrType(op, name, value._cDataType)
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: TensorShape) {
        let dimensions: [Int64] = value.dimensions.map(Int64.init)
        dimensions.withUnsafeBufferPointer { buffer in
            TFE_OpSetAttrShape(op, name, buffer.baseAddress, Int32(buffer.count), status)
        }
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: TensorShape?) {
        guard let shape = value else {
            TFE_OpSetAttrShape(op, name, nil, -1, status)
            return
        }
        updateAttribute(name, shape)
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: [Bool]) {
        value.map({ $0 ? UInt8(1) : UInt8(0) }).withUnsafeBufferPointer { buffer in
            TFE_OpSetAttrBoolList(op, name, buffer.baseAddress, Int32(buffer.count))
        }
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: [Int]) {
        updateAttribute(name, value.map(Int64.init))
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: [Int32]) {
        updateAttribute(name, value.map(Int64.init))
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: [Int64]) {
        value.withUnsafeBufferPointer { buffer in
            TFE_OpSetAttrIntList(op, name, buffer.baseAddress, Int32(buffer.count))
        }
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: [Float]) {
        value.withUnsafeBufferPointer { buffer in
            TFE_OpSetAttrFloatList(op, name, buffer.baseAddress, Int32(buffer.count))
        }
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: [Double]) {
        updateAttribute(name, value.map(Float.init))
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: [String]) {
        // Collect all the strings' utf8 bytes into a single array so that we can
        // address all the strings with a single
        // `flattenedStringBytes.withUnsafeBufferPointer`.
        var flattenedStringBytes: [CChar] = []
        var lengths: [Int] = []
        for string in value {
            // Don't include the null-terminator because TFE_OpSetAttrStringList uses
            // lengths instead of null-terminators.
            let stringBytes = string.utf8CString.dropLast()
            flattenedStringBytes.append(contentsOf: stringBytes)
            lengths.append(stringBytes.count)
        }

        // Calculate the addresses of all the strings within our single buffer, and then call
        // TFE_OpSetAttrStringList.
        flattenedStringBytes.withUnsafeBufferPointer { flattenedStringBytesBuffer in
            var stringAddrs: [UnsafeRawPointer?] = []
            var currentStringAddr =
                flattenedStringBytesBuffer.baseAddress.map(UnsafeRawPointer.init)
            for length in lengths {
                stringAddrs.append(currentStringAddr)
                currentStringAddr = currentStringAddr?.advanced(by: length)
            }

            stringAddrs.withUnsafeBufferPointer { stringAddrsBuffer in
                lengths.withUnsafeBufferPointer { lengthsBuffer in
                    TFE_OpSetAttrStringList(op, name, stringAddrsBuffer.baseAddress,
                        lengthsBuffer.baseAddress, Int32(value.count))
                }
            }
        }
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: [TensorDataType]) {
        value.withUnsafeBufferPointer { buffer in
            buffer.withMemoryRebound(to: TF_DataType.self) { reboundBuffer in
                TFE_OpSetAttrTypeList(
                    op, name, reboundBuffer.baseAddress, Int32(reboundBuffer.count))
            }
        }
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: [TensorShape]) {
        let flattenedDims = value.flatMap { $0.dimensions.map(Int64.init) }
        let ranks = value.map { Int32($0.rank) }
        flattenedDims.withUnsafeBufferPointer { flattenedDimsBuffer in
            var dimsPtr: UnsafePointer<Int64>? = flattenedDimsBuffer.baseAddress
            var dims: [UnsafePointer<Int64>?] = []
            for rank in ranks {
                dims.append(dimsPtr)
                if rank >= 0 {
                    dimsPtr = dimsPtr.map { $0.advanced(by: Int(rank)) }
                }
            }
            dims.withUnsafeMutableBufferPointer { dimsBuffer in
                ranks.withUnsafeBufferPointer { ranksBuffer in
                    TFE_OpSetAttrShapeList(
                        op, name, dimsBuffer.baseAddress, ranksBuffer.baseAddress,
                        Int32(ranksBuffer.count), status)
                }
            }
        }
    }

    @inlinable @inline(__always)
    internal func updateAttribute(_ name: String, _ value: [TensorShape?]) {
        let flattenedDims = value.flatMap { (tensorShapeOpt) -> [Int64] in
            if let tensorShape = tensorShapeOpt {
                return tensorShape.dimensions.map(Int64.init)
            }
            return []
        }
        let ranks = value.map { shape in (shape?.rank).map(Int32.init) ?? -1 }
        flattenedDims.withUnsafeBufferPointer { flattenedDimsBuffer in
            var dimsPtr: UnsafePointer<Int64>? = flattenedDimsBuffer.baseAddress
            var dims: [UnsafePointer<Int64>?] = []
            for rank in ranks {
                dims.append(dimsPtr)
                if rank >= 0 {
                    dimsPtr = dimsPtr.map { $0.advanced(by: Int(rank)) }
                }
            }
            dims.withUnsafeMutableBufferPointer { dimsBuffer in
                ranks.withUnsafeBufferPointer { ranksBuffer in
                    TFE_OpSetAttrShapeList(
                        op, name, dimsBuffer.baseAddress, ranksBuffer.baseAddress,
                        Int32(ranksBuffer.count), status)
                }
            }
        }
    }

    internal func updateAttribute<In: TensorGroup, Out: TensorGroup>(
        _ name: String,
        _ value: (In) -> Out
    ) {
        updateAttribute(name, _TensorFunctionPointer(name: _tffunc(value)))
    }

    internal func updateAttribute(_ name: String, _ value: _TensorFunctionPointer) {
        value.name.utf8CString.withUnsafeBufferPointer { buffer in
            // utf8CString is null-terminated; TFE_OpSetAttrFunctionName wants
            // non-null-terminated.
            TFE_OpSetAttrFunctionName(op, name, buffer.baseAddress, buffer.count - 1)
        }
    }

    /// **WARNING:** After constructing a `TFE_Op`, any one of its `execute` methods must be called
    /// *exactly once*. If not called, then a memory leak is introduced due to the underlying
    /// TensorFlow eager op object not being freed. If called more than once, then a SEGFAULT may
    /// occur due to trying to execute a TensorFlow eager op that has already been freed.

    @inlinable @inline(__always)
    internal func evaluateUnsafe() -> UnsafeMutablePointer<CTensorHandle> {
        var count: Int32 = Int32(self.outputCount)
        let buffer: UnsafeMutablePointer<CTensorHandle> =
        UnsafeMutablePointer.allocate(capacity: Int(count))
        _TFCOpSetDeviceFromScope(op, status)
        checkOk(status)
        _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, status)
        checkOk(status)
        TFE_DeleteOp(op)
        TF_DeleteStatus(status)
        return buffer
    }

    @inlinable @inline(__always)
    internal func execute() {
        let _ = evaluateUnsafe()
    }

    @inlinable @inline(__always)
    internal func execute<T0: TensorArrayProtocol>(
        _ count0: Int
    ) -> (T0) {
        let buffer = evaluateUnsafe()
        let offset0 = Int32(0)
        let result = (
            T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0))
        buffer.deallocate()
        return result
    }

    @inlinable @inline(__always)
    internal func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int
    ) -> (T0, T1) {
        let buffer = evaluateUnsafe()
        let offset0 = Int32(0)
        let offset1 = offset0 + Int32(count0)
        let result = (
            T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
            T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1))
        buffer.deallocate()
        return result
    }

    @inlinable @inline(__always)
    internal func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int
    ) -> (T0, T1, T2) {
        let buffer = evaluateUnsafe()
        let offset0 = Int32(0)
        let offset1 = offset0 + Int32(count0)
        let offset2 = offset1 + Int32(count1)
        let result = (
            T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
            T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
            T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2))
        buffer.deallocate()
        return result
    }

    @inlinable @inline(__always)
    internal func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int,
        _ count3: Int
    ) -> (T0, T1, T2, T3) {
        let buffer = evaluateUnsafe()
        let offset0 = Int32(0)
        let offset1 = offset0 + Int32(count0)
        let offset2 = offset1 + Int32(count1)
        let offset3 = offset2 + Int32(count2)
        let result = (
            T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
            T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
            T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
            T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3))
        buffer.deallocate()
        return result
    }

    @inlinable @inline(__always)
    internal func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int,
        _ count3: Int,
        _ count4: Int
    ) -> (T0, T1, T2, T3, T4) {
        let buffer = evaluateUnsafe()
        let offset0 = Int32(0)
        let offset1 = offset0 + Int32(count0)
        let offset2 = offset1 + Int32(count1)
        let offset3 = offset2 + Int32(count2)
        let offset4 = offset3 + Int32(count3)
        let result = (
            T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
            T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
            T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
            T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3),
            T4.init(_owning: buffer.advanced(by: Int(offset4)), count: count4))
        buffer.deallocate()
        return result
    }

    @inlinable @inline(__always)
    internal func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int,
        _ count3: Int,
        _ count4: Int,
        _ count5: Int
    ) -> (T0, T1, T2, T3, T4, T5) {
        let buffer = evaluateUnsafe()
        let offset0 = Int32(0)
        let offset1 = offset0 + Int32(count0)
        let offset2 = offset1 + Int32(count1)
        let offset3 = offset2 + Int32(count2)
        let offset4 = offset3 + Int32(count3)
        let offset5 = offset4 + Int32(count4)
        let result = (
            T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
            T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
            T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
            T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3),
            T4.init(_owning: buffer.advanced(by: Int(offset4)), count: count4),
            T5.init(_owning: buffer.advanced(by: Int(offset5)), count: count5))
        buffer.deallocate()
        return result
    }

    @inlinable @inline(__always)
    internal func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int,
        _ count3: Int,
        _ count4: Int,
        _ count5: Int,
        _ count6: Int
    ) -> (T0, T1, T2, T3, T4, T5, T6) {
        let buffer = evaluateUnsafe()
        let offset0 = Int32(0)
        let offset1 = offset0 + Int32(count0)
        let offset2 = offset1 + Int32(count1)
        let offset3 = offset2 + Int32(count2)
        let offset4 = offset3 + Int32(count3)
        let offset5 = offset4 + Int32(count4)
        let offset6 = offset5 + Int32(count5)
        let result = (
            T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
            T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
            T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
            T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3),
            T4.init(_owning: buffer.advanced(by: Int(offset4)), count: count4),
            T5.init(_owning: buffer.advanced(by: Int(offset5)), count: count5),
            T6.init(_owning: buffer.advanced(by: Int(offset6)), count: count6))
        buffer.deallocate()
        return result
    }

    @inlinable @inline(__always)
    internal func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int,
        _ count3: Int,
        _ count4: Int,
        _ count5: Int,
        _ count6: Int,
        _ count7: Int
    ) -> (T0, T1, T2, T3, T4, T5, T6, T7) {
        let buffer = evaluateUnsafe()
        let offset0 = Int32(0)
        let offset1 = offset0 + Int32(count0)
        let offset2 = offset1 + Int32(count1)
        let offset3 = offset2 + Int32(count2)
        let offset4 = offset3 + Int32(count3)
        let offset5 = offset4 + Int32(count4)
        let offset6 = offset5 + Int32(count5)
        let offset7 = offset6 + Int32(count6)
        let result = (
            T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
            T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
            T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
            T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3),
            T4.init(_owning: buffer.advanced(by: Int(offset4)), count: count4),
            T5.init(_owning: buffer.advanced(by: Int(offset5)), count: count5),
            T6.init(_owning: buffer.advanced(by: Int(offset6)), count: count6),
            T7.init(_owning: buffer.advanced(by: Int(offset7)), count: count7))
        buffer.deallocate()
        return result
    }

    @inlinable @inline(__always)
    internal func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int,
        _ count3: Int,
        _ count4: Int,
        _ count5: Int,
        _ count6: Int,
        _ count7: Int,
        _ count8: Int
    ) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8) {
        let buffer = evaluateUnsafe()
        let offset0 = Int32(0)
        let offset1 = offset0 + Int32(count0)
        let offset2 = offset1 + Int32(count1)
        let offset3 = offset2 + Int32(count2)
        let offset4 = offset3 + Int32(count3)
        let offset5 = offset4 + Int32(count4)
        let offset6 = offset5 + Int32(count5)
        let offset7 = offset6 + Int32(count6)
        let offset8 = offset7 + Int32(count7)
        let result = (
            T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
            T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
            T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
            T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3),
            T4.init(_owning: buffer.advanced(by: Int(offset4)), count: count4),
            T5.init(_owning: buffer.advanced(by: Int(offset5)), count: count5),
            T6.init(_owning: buffer.advanced(by: Int(offset6)), count: count6),
            T7.init(_owning: buffer.advanced(by: Int(offset7)), count: count7),
            T8.init(_owning: buffer.advanced(by: Int(offset8)), count: count8))
        buffer.deallocate()
        return result
    }

    @inlinable @inline(__always)
    internal func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol, T9: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int,
        _ count3: Int,
        _ count4: Int,
        _ count5: Int,
        _ count6: Int,
        _ count7: Int,
        _ count8: Int,
        _ count9: Int
    ) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) {
        let buffer = evaluateUnsafe()
        let offset0 = Int32(0)
        let offset1 = offset0 + Int32(count0)
        let offset2 = offset1 + Int32(count1)
        let offset3 = offset2 + Int32(count2)
        let offset4 = offset3 + Int32(count3)
        let offset5 = offset4 + Int32(count4)
        let offset6 = offset5 + Int32(count5)
        let offset7 = offset6 + Int32(count6)
        let offset8 = offset7 + Int32(count7)
        let offset9 = offset8 + Int32(count8)
        let result = (
            T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
            T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
            T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
            T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3),
            T4.init(_owning: buffer.advanced(by: Int(offset4)), count: count4),
            T5.init(_owning: buffer.advanced(by: Int(offset5)), count: count5),
            T6.init(_owning: buffer.advanced(by: Int(offset6)), count: count6),
            T7.init(_owning: buffer.advanced(by: Int(offset7)), count: count7),
            T8.init(_owning: buffer.advanced(by: Int(offset8)), count: count8),
            T9.init(_owning: buffer.advanced(by: Int(offset9)), count: count9))
        buffer.deallocate()
        return result
    }

}
