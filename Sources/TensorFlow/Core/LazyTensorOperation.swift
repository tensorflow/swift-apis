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

@usableFromInline
class LazyTensor: _AnyTensorHandle {
    enum Handle {
        /// Bool indicates if this concrete TFETensorhandle was a result of
        /// materialization.
        case concrete(TFETensorHandle, materialized: Bool)
        /// Bool indicates whether this is a live tensor. This flag is used to
        /// heuristically determine whether this symbolic tensor should also be
        /// materialized whenever materialization of any other tensor is triggered.
        case symbolic(LazyTensorOperation, index: Int, isLive: Bool)
    }

    let handle: Handle

    @usableFromInline
    var _tfeTensorHandle: TFETensorHandle {
        switch handle {
        case .concrete(let h, _):
            return h
        case .symbolic(let op, let index, _):
            assert(false, "TODO: to be send out in a separate PR.")
            // return op.materialized(index: index)
        }
    }

    init(_ base: TFETensorHandle) {
        handle = Handle.concrete(base, materialized: false)
    }

    init(_materialized base: TFETensorHandle) {
        handle = Handle.concrete(base, materialized: true)
    }

    init(_lazy op: LazyTensorOperation, index: Int) {
        precondition(
            index < op.outputCount, "Symbolic Tensor Index is out-of-bounds")
        handle = Handle.symbolic(op, index: index, isLive: false)
        LazyTensor.incrementRefCount(op, isLive: false)
    }

    init(_lazyLive op: LazyTensorOperation, index: Int) {
        precondition(
            index < op.outputCount, "Symbolic Tensor Index is out-of-bounds")
        handle = Handle.symbolic(op, index: index, isLive: true)
        LazyTensor.incrementRefCount(op, isLive: true)
    }

    deinit {
        if case let .symbolic(op, _, isLive) = handle {
            LazyTensor.decrementRefCount(op, isLive: isLive)
        }
    }

    // Liveness tracking for LazyTensorOperations
    //
    struct LazyTensorOperationRefCounts {
        let op: LazyTensorOperation
        let liveRefCount: Int
        let allRefCount: Int
    }

    private static var operationRefCounts: [
        ObjectIdentifier: LazyTensorOperationRefCounts] = [:]

    static func incrementRefCount(_ op: LazyTensorOperation, isLive: Bool) {
        let opID = ObjectIdentifier(op)
        if let counts = operationRefCounts[opID] {
            operationRefCounts[opID] = LazyTensorOperationRefCounts(
                op: op,
                liveRefCount: counts.liveRefCount + (isLive ? 1 : 0),
                allRefCount: counts.allRefCount + 1)
        } else {
            operationRefCounts[opID] = LazyTensorOperationRefCounts(
                op: op, liveRefCount: isLive ? 1 : 0, allRefCount: 1)
        }
    }

    static func decrementRefCount(_ op: LazyTensorOperation, isLive: Bool) {
        let opID = ObjectIdentifier(op)
        if let counts = operationRefCounts[opID] {
            if counts.allRefCount > 1 {
                operationRefCounts[opID] = LazyTensorOperationRefCounts(
                    op: op,
                    liveRefCount: counts.liveRefCount - (isLive ? 1 : 0),
                    allRefCount: counts.allRefCount - 1)
            } else {
                operationRefCounts.removeValue(forKey: opID)
            }
        }
    }

    static func isLive(_ op: LazyTensorOperation) -> Bool {
        let opID = ObjectIdentifier(op)
        if let counts = operationRefCounts[opID] {
            return counts.liveRefCount > 0
        }
        return false
    }

    static func forEachLiveOperation(
        _ perform: (LazyTensorOperation) throws -> Void
    ) rethrows -> Void {
        for (_, counts) in operationRefCounts where counts.liveRefCount > 0 {
            try perform(counts.op)
        }
    }

    static func forEachOperation(
        _ perform: (LazyTensorOperation) throws -> Void
    ) rethrows -> Void {
        for (_, counts) in operationRefCounts { try perform(counts.op) }
    }

    @usableFromInline
    static var _materializationCallback: (String) -> () = { _ in }
}

class LazyTensorOperation: TensorOperation {
     typealias TensorValueHandle = LazyTensor

    enum Input {
        case single(LazyTensor)
        case list([LazyTensor])
    }

    enum Attribute {
        case BoolValue(Bool)
        case IntValue(Int)
        case FloatValue(Float)
        case DoubleValue(Double)
        case StringValue(String)
        case BoolArray([Bool])
        case IntArray([Int])
        case FloatArray([Float])
        case DoubleArray([Double])
        case StringArray([String])
        case ConstTensor(TFETensorHandle)
        case TensorDataTypeValue(TensorDataType)
        case TensorFunctionPointer(_TensorFunctionPointer)
    }

    let name: String
    let outputCount: Int
    var inputs: [Input]
    var attrs: [String: Attribute]
    var outputs: [TFETensorHandle]?
    var id: String?

    var nameWithID: String {
        if let id = self.id {
            return "\(name)_\(id)"
        } else {
            return "\(name)_\(ObjectIdentifier(self))"
        }
    }

    func outputName(at index: Int) -> String {
        precondition(index < outputCount,
            "Output index out of bounds when getting outputName.")
        let ssaID = id ?? "\(ObjectIdentifier(self))"
        var ssaName = "%\(ssaID)"
        if outputCount > 1 {
            ssaName += ".\(index)"
        }
        return ssaName
    }

    var outputName: String {
        switch outputCount {
        case 0: return ""
        case 1: return outputName(at: 0)
        default:
            let outputNames = (0..<outputCount).lazy.map {
                self.outputName(at: $0)
            }
            let aggregateName = outputNames.joined(separator: ", ")
            return "(\(aggregateName))"
        }
    }

    static var liveOperations: Int = 0

    init(_id id: String?, name: String, outputCount: Int) {
        self.name = name
        self.inputs = []
        self.attrs = [:]
        self.outputCount = outputCount
        self.outputs = nil
        self.id = id
        LazyTensorOperation.liveOperations += 1
    }

    required convenience init(_ name: String, _ outputCount: Int) {
        self.init(_id: nil, name: name, outputCount: outputCount)
    }

    deinit {
        LazyTensorOperation.liveOperations -= 1
    }

    func evaluate() -> [LazyTensor] {
        return (0..<outputCount).map {
            LazyTensor(_lazyLive: self, index: $0)
        }
    }

    func addInput(_ input : LazyTensor) {
        inputs.append(Input.single(input))
    }

    func updateAttribute(_ name: String, _ value: Bool) {
        attrs[name] = Attribute.BoolValue(value)
    }
    func updateAttribute(_ name: String, _ value: Int) {
        attrs[name] = Attribute.IntValue(value)
    }
    func updateAttribute(_ name: String, _ value: Int32) {
        attrs[name] = Attribute.IntValue(Int(value))
    }
    func updateAttribute(_ name: String, _ value: Int64) {
        attrs[name] = Attribute.IntValue(Int(value))
    }
    func updateAttribute(_ name: String, _ value: Float) {
        attrs[name] = Attribute.FloatValue(value)
    }
    func updateAttribute(_ name: String, _ value: Double) {
        attrs[name] = Attribute.DoubleValue(value)
    }
    func updateAttribute(_ name: String, _ value: String) {
        attrs[name] = Attribute.StringValue(value)
    }
    func updateAttribute(_ name: String, _ value: [Bool]) {
        attrs[name] = Attribute.BoolArray(value)
    }
    func updateAttribute(_ name: String, _ value: [Int]) {
        attrs[name] = Attribute.IntArray(value)
    }
    func updateAttribute(_ name: String, _ value: [Int32]) {
        attrs[name] = Attribute.IntArray(value.map { Int($0) })
    }
    func updateAttribute(_ name: String, _ value: [Int64]) {
        attrs[name] = Attribute.IntArray(value.map { Int($0) })
    }
    func updateAttribute(_ name: String, _ value: [Float]) {
        attrs[name] = Attribute.FloatArray(value)
    }
    func updateAttribute(_ name: String, _ value: [Double]) {
        attrs[name] = Attribute.DoubleArray(value)
    }
    func updateAttribute(_ name: String, _ value: [String]) {
        attrs[name] = Attribute.StringArray(value)
    }
    func updateAttribute(_ name: String, _ value: _TensorFunctionPointer) {
        attrs[name] = Attribute.TensorFunctionPointer(value)
    }
}

extension LazyTensorOperation: TFTensorOperation {
    private func lazyTensorHandle(_ input: _AnyTensorHandle) -> LazyTensor {
        if let lazyHandle = input as? LazyTensor {
            if case let LazyTensor.Handle.symbolic(
                op, index, true) = lazyHandle.handle {
                // We turn off liveness for the constructed LazyTensor,
                // because it is only referenced internally as a part
                // of the LazyTensorOperation input.
                return LazyTensor(_lazy: op, index: index)
            } else {
                return lazyHandle
            }
        } else {
            return LazyTensor(input._tfeTensorHandle)
        }
    }

    func addInput(_ input: _AnyTensorHandle) {
        addInput(lazyTensorHandle(input))
    }

    func addInput<Scalar: TensorFlowScalar>(_ input: Tensor<Scalar>) {
        addInput(input.handle.handle)
    }

    func addInput(_ input: StringTensor) {
        addInput(input.handle.handle)
    }

    func addInput(_ input: VariantHandle) {
        addInput(input.handle)
    }

    func addInput(_ input: ResourceHandle) {
        addInput(input.handle)
    }

    func addInputList<T: TensorArrayProtocol>(_ input: T) {
        let lazyHandles = input._tensorHandles.map { lazyTensorHandle($0) }
        inputs.append(Input.list(lazyHandles))
    }

    func updateAttribute(_ name: String, _ value: TensorDataType) {
        attrs[name] = Attribute.TensorDataTypeValue(value)
    }
    func updateAttribute(_ name: String, _ value: TensorShape) {
        assert(false, "Unimplemented TensorShape attribute.")
    }
    func updateAttribute(_ name: String, _ value: TensorShape?) {
        assert(false, "Unimplemented TensorShape? attribute.")
    }
    func updateAttribute(_ name: String, _ value: [TensorDataType]) {
        assert(false, "Unimplemented [TensorDataType] attribute.")
    }
    func updateAttribute(_ name: String, _ value: [TensorShape]) {
        assert(false, "Unimplemented [TensorShape] attribute.")
    }
    func updateAttribute(_ name: String, _ value: [TensorShape?]) {
        assert(false, "Unimplemented [TensorShape?] attribute.")
    }
    func updateAttribute<In: TensorGroup, Out: TensorGroup>(
        _ name: String, _ value: (In) -> Out) {
        // TODO:
        assert(false, "Unimplemented [TFFunction] attribute.")
    }

    func execute() {}

    func execute<T0: TensorArrayProtocol>(
        _ count0: Int
    ) -> (T0) {
        let outputs = evaluate()
        let offset0 = 0
        let result = (
            T0.init(_handles: outputs[offset0..<count0]))
        return result
    }

    func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int
    ) -> (T0, T1) {
        let outputs = evaluate()
        let offset0 = 0
        let offset1 = offset0 + count0
        let result = (
            T0.init(_handles: outputs[offset0..<offset1]),
            T1.init(_handles: outputs[offset1..<outputs.count]))
        return result
    }

    func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int
    ) -> (T0, T1, T2) {
        let outputs = evaluate()
        let offset0 = 0
        let offset1 = offset0 + count0
        let offset2 = offset1 + count1
        let result = (
            T0.init(_handles: outputs[offset0..<offset1]),
            T1.init(_handles: outputs[offset1..<offset2]),
            T2.init(_handles: outputs[offset2..<outputs.count]))
        return result
    }

    func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int,
        _ count3: Int
    ) -> (T0, T1, T2, T3) {
        let outputs = evaluate()
        let offset0 = 0
        let offset1 = offset0 + count0
        let offset2 = offset1 + count1
        let offset3 = offset2 + count2
        let result = (
            T0.init(_handles: outputs[offset0..<offset1]),
            T1.init(_handles: outputs[offset1..<offset2]),
            T2.init(_handles: outputs[offset2..<offset3]),
            T3.init(_handles: outputs[offset3..<outputs.count]))
        return result
    }

    func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int,
        _ count3: Int,
        _ count4: Int
    ) -> (T0, T1, T2, T3, T4) {
        let outputs = evaluate()
        let offset0 = 0
        let offset1 = offset0 + count0
        let offset2 = offset1 + count1
        let offset3 = offset2 + count2
        let offset4 = offset3 + count3
        let result = (
            T0.init(_handles: outputs[offset0..<offset1]),
            T1.init(_handles: outputs[offset1..<offset2]),
            T2.init(_handles: outputs[offset2..<offset3]),
            T3.init(_handles: outputs[offset3..<offset4]),
            T4.init(_handles: outputs[offset4..<outputs.count]))
        return result
    }

    func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int,
        _ count3: Int,
        _ count4: Int,
        _ count5: Int
    ) -> (T0, T1, T2, T3, T4, T5) {
        let outputs = evaluate()
        let offset0 = 0
        let offset1 = offset0 + count0
        let offset2 = offset1 + count1
        let offset3 = offset2 + count2
        let offset4 = offset3 + count3
        let offset5 = offset4 + count4
        let result = (
            T0.init(_handles: outputs[offset0..<offset1]),
            T1.init(_handles: outputs[offset1..<offset2]),
            T2.init(_handles: outputs[offset2..<offset3]),
            T3.init(_handles: outputs[offset3..<offset4]),
            T4.init(_handles: outputs[offset4..<offset5]),
            T5.init(_handles: outputs[offset5..<outputs.count]))
        return result
    }

    func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int,
        _ count3: Int,
        _ count4: Int,
        _ count5: Int,
        _ count6: Int
    ) -> (T0, T1, T2, T3, T4, T5, T6) {
        let outputs = evaluate()
        let offset0 = 0
        let offset1 = offset0 + count0
        let offset2 = offset1 + count1
        let offset3 = offset2 + count2
        let offset4 = offset3 + count3
        let offset5 = offset4 + count4
        let offset6 = offset5 + count5
        let result = (
            T0.init(_handles: outputs[offset0..<offset1]),
            T1.init(_handles: outputs[offset1..<offset2]),
            T2.init(_handles: outputs[offset2..<offset3]),
            T3.init(_handles: outputs[offset3..<offset4]),
            T4.init(_handles: outputs[offset4..<offset5]),
            T5.init(_handles: outputs[offset5..<offset6]),
            T6.init(_handles: outputs[offset6..<outputs.count]))
        return result
    }

    func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol>(
        _ count0: Int,
        _ count1: Int,
        _ count2: Int,
        _ count3: Int,
        _ count4: Int,
        _ count5: Int,
        _ count6: Int,
        _ count7: Int
    ) -> (T0, T1, T2, T3, T4, T5, T6, T7) {
        let outputs = evaluate()
        let offset0 = 0
        let offset1 = offset0 + count0
        let offset2 = offset1 + count1
        let offset3 = offset2 + count2
        let offset4 = offset3 + count3
        let offset5 = offset4 + count4
        let offset6 = offset5 + count5
        let offset7 = offset6 + count6
        let result = (
            T0.init(_handles: outputs[offset0..<offset1]),
            T1.init(_handles: outputs[offset1..<offset2]),
            T2.init(_handles: outputs[offset2..<offset3]),
            T3.init(_handles: outputs[offset3..<offset4]),
            T4.init(_handles: outputs[offset4..<offset5]),
            T5.init(_handles: outputs[offset5..<offset6]),
            T6.init(_handles: outputs[offset6..<offset7]),
            T7.init(_handles: outputs[offset7..<outputs.count]))
        return result
    }

    func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol>(
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
        let outputs = evaluate()
        let offset0 = 0
        let offset1 = offset0 + count0
        let offset2 = offset1 + count1
        let offset3 = offset2 + count2
        let offset4 = offset3 + count3
        let offset5 = offset4 + count4
        let offset6 = offset5 + count5
        let offset7 = offset6 + count6
        let offset8 = offset7 + count7
        let result = (
            T0.init(_handles: outputs[offset0..<offset1]),
            T1.init(_handles: outputs[offset1..<offset2]),
            T2.init(_handles: outputs[offset2..<offset3]),
            T3.init(_handles: outputs[offset3..<offset4]),
            T4.init(_handles: outputs[offset4..<offset5]),
            T5.init(_handles: outputs[offset5..<offset6]),
            T6.init(_handles: outputs[offset6..<offset7]),
            T7.init(_handles: outputs[offset7..<offset8]),
            T8.init(_handles: outputs[offset8..<outputs.count]))
    return result
    }

    func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol, T9: TensorArrayProtocol>(
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
        let outputs = evaluate()
        let offset0 = 0
        let offset1 = offset0 + count0
        let offset2 = offset1 + count1
        let offset3 = offset2 + count2
        let offset4 = offset3 + count3
        let offset5 = offset4 + count4
        let offset6 = offset5 + count5
        let offset7 = offset6 + count6
        let offset8 = offset7 + count7
        let offset9 = offset8 + count8
        let result = (
            T0.init(_handles: outputs[offset0..<offset1]),
            T1.init(_handles: outputs[offset1..<offset2]),
            T2.init(_handles: outputs[offset2..<offset3]),
            T3.init(_handles: outputs[offset3..<offset4]),
            T4.init(_handles: outputs[offset4..<offset5]),
            T5.init(_handles: outputs[offset5..<offset6]),
            T6.init(_handles: outputs[offset6..<offset7]),
            T7.init(_handles: outputs[offset7..<offset8]),
            T8.init(_handles: outputs[offset8..<offset9]),
            T9.init(_handles: outputs[offset9..<outputs.count]))
        return result
    }
}

extension TFETensorHandle {
    public var valueDescription: String {
        let dtype = TFE_TensorHandleDataType(self._cTensorHandle)
        switch dtype {
        case TF_FLOAT:
            return Tensor(handle: TensorHandle<Float>(handle: self)).description
        case TF_DOUBLE:
            return Tensor(handle: TensorHandle<Double>(handle: self)).description
        case TF_BFLOAT16:
            return Tensor(handle: TensorHandle<BFloat16>(handle: self)).description
        case TF_INT64:
            return Tensor(handle: TensorHandle<Int64>(handle: self)).description
        case TF_INT32:
            return Tensor(handle: TensorHandle<Int32>(handle: self)).description
        case TF_INT16:
            return Tensor(handle: TensorHandle<Int16>(handle: self)).description
        case TF_INT8:
            return Tensor(handle: TensorHandle<Int8>(handle: self)).description
        case TF_UINT64:
            return Tensor(handle: TensorHandle<UInt64>(handle: self)).description
        case TF_UINT32:
            return Tensor(handle: TensorHandle<UInt32>(handle: self)).description
        case TF_UINT16:
            return Tensor(handle: TensorHandle<UInt16>(handle: self)).description
        case TF_UINT8:
            return Tensor(handle: TensorHandle<UInt8>(handle: self)).description
        case TF_BOOL:
            return Tensor(handle: TensorHandle<Bool>(handle: self)).description
        case TF_STRING:
            // TODO(https://bugs.swift.org/browse/TF-561): The current
            // implementation of ShapedArray<String> is not correct, which
            // causes seg faults.
            return "\"string\""
        default:
            return TFETensorHandle.tfDataTypeAsString(dtype)
        }
    }

    static func tfDataTypeAsString(_ cDataType: TF_DataType) -> String {
        switch(cDataType) {
        case TF_FLOAT: return "float"
        case TF_DOUBLE: return "double"
        case TF_INT32: return "int32"
        case TF_UINT8: return "uint8"
        case TF_INT16: return "int16"
        case TF_INT8: return "int8"
        case TF_STRING: return "string"
        case TF_COMPLEX64, TF_COMPLEX: return "complex"
        case TF_INT64: return "int64"
        case TF_BOOL: return "bool"
        case TF_QINT8: return "qint8"
        case TF_QUINT8: return "quint8"
        case TF_QINT32: return "qint32"
        case TF_BFLOAT16: return "bfloat16"
        case TF_QINT16: return "qint16"
        case TF_QUINT16: return "quint16"
        case TF_UINT16: return "uint16"
        case TF_COMPLEX128: return "complex128"
        case TF_HALF: return "half"
        case TF_RESOURCE: return "resource"
        case TF_VARIANT: return "variant"
        case TF_UINT32: return "uint32"
        case TF_UINT64: return "uint64"
        default: assert(false, "Unhandled type: \(cDataType)")
        }
    }
}

extension LazyTensorOperation.Attribute: CustomStringConvertible {
    var description: String {
        switch self {
        case .BoolValue(let v): return "\(v)"
        case .IntValue(let v): return "Int(\(v))"
        case .FloatValue(let v): return "Float(\(v))"
        case .DoubleValue(let v): return "Double(\(v))"
        case .StringValue(let v): return "\"\(v)\""
        case .BoolArray(let values): return arrayAsString("", values)
        case .IntArray(let values): return arrayAsString("Int", values)
        case .FloatArray(let values): return arrayAsString("Float", values)
        case .DoubleArray(let values): return arrayAsString("Double", values)
        case .StringArray(let values): return arrayAsString("String", values)
        case .ConstTensor(let v): return v.valueDescription
        case .TensorDataTypeValue(let v): return dataTypeAsString(v)
        case .TensorFunctionPointer(let v): return "TFFunction(\(v.name))"
        }
    }

    private func arrayAsString<T>(_ desc: String, _ values: [T]) -> String {
        let arrayDesc = (values.map { "\($0)" }).joined(separator: ", ")
        return "\(desc)[\(arrayDesc)]"
    }

    private func dataTypeAsString(_ dataType: TensorDataType) -> String {
        return TFETensorHandle.tfDataTypeAsString(dataType._cDataType)
    }
}

extension LazyTensor: CustomStringConvertible {
    public var description: String {
        switch self.handle {
        case LazyTensor.Handle.concrete(let h, let isMaterialized):
            return isMaterialized
                ? "\(h.valueDescription)*"
                : "\(h.valueDescription)"
        case LazyTensor.Handle.symbolic(let op, let index, let isLive):
            return op.outputName(at: index) + (isLive ? "*" : "")
        }
    }
}

extension LazyTensorOperation: CustomStringConvertible {
    public var description: String {
        let attrsDesc = attrs.map { (name, value) in "\(name): \(value)" }
        let inputsDesc = inputs.map { (input: Input) -> String in
            switch input {
            case Input.single(let lazyTensor):
                return "\(lazyTensor)"
            case Input.list(let lazyTensorList):
                do {
                    let lazyTensors = lazyTensorList.map { "\($0)" }
                    let lazyTensorsDesc = lazyTensors.joined(separator: ", ")
                    return "[\(lazyTensorsDesc)]"
                }
            }
        }
        var desc = "\(outputName) = \(name)"
        if attrs.count > 0 {
            desc += "["
            desc += attrsDesc.joined(separator: ", ")
            desc += "]"
        }
            desc += "("
        desc += inputsDesc.joined(separator: ", ")
        desc += ")"
        return desc
    }
}
