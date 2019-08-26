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

extension TFETensorHandle {
    /// Returns true if the handles are equivalent for the purposes of comparing lazy tensor traces.
    static func areHandlesEquivalent(_ lhs: TFETensorHandle, _ rhs: TFETensorHandle) -> Bool {
        return lhs._cTensorHandle == rhs._cTensorHandle
    }

    /// Returns true if the underlying tensors are equal.
    static func areTensorsEqual(_ lhs: TFETensorHandle, _ rhs: TFETensorHandle) -> Bool {
        let lhsDtype = TFE_TensorHandleDataType(lhs._cTensorHandle)
        let rhsDtype = TFE_TensorHandleDataType(rhs._cTensorHandle)
        precondition(
            lhsDtype == rhsDtype && lhsDtype != TF_VARIANT && lhsDtype != TF_RESOURCE,
            "Datatypes of tensor handles don't match.")
        let op = TFE_Op("Equal", 1)
        op.updateAttribute("T", TensorDataType(lhsDtype))
        op.addInput(lhs)
        op.addInput(rhs)
        let result: Tensor<Bool> = op.execute(Int(1))
        return result.scalars.allSatisfy { $0 }
    }
}

extension LazyTensorHandle {
    static func areHandlesEquivalent(_ lhs: LazyTensorHandle, _ rhs: LazyTensorHandle) -> Bool {
        switch (lhs.handle, rhs.handle) {
        case let (.concrete(x, _), .concrete(y, _)):
            return TFETensorHandle.areHandlesEquivalent(x, y)
        case let (.symbolic(x, xi, _), .symbolic(y, yi, _)):
            return (xi == yi) && (x.id == y.id)
        default: return false
        }
    }
}

extension LazyTensorOperation {
    /// Returns true if these inputs are equivalent when comparing lazy tensor traces.
    static func areInputsEquivalent(_ lhs: Input, _ rhs: Input) -> Bool {
        switch (lhs, rhs) {
        case let (.single(l), .single(r)):
            return LazyTensorHandle.areHandlesEquivalent(l, r)
        case let (.list(l), .list(r)):
            return l.elementsEqual(r, by: {LazyTensorHandle.areHandlesEquivalent($0, $1) })
        default:
            return false
        }
    }

    /// Returns true if these operations are equivalent when comparing lazy tensor traces.
    static func areEquivalent(_ lhs: LazyTensorOperation, _ rhs: LazyTensorOperation) -> Bool {
        return (lhs.name == rhs.name) &&
            (lhs.outputCount == rhs.outputCount) &&
            (lhs.deviceName == rhs.deviceName) &&
            lhs.inputs.elementsEqual(
                rhs.inputs,
                by: { LazyTensorOperation.areInputsEquivalent($0, $1) }) &&
           (lhs.attributes == rhs.attributes)
    }
}

extension LazyTensorOperation.Attribute: Equatable {}

func ==(_ lhs: LazyTensorOperation.Attribute, _ rhs: LazyTensorOperation.Attribute) -> Bool {
    // We cannot rely on the derived conformance, because it would force us to add a conformance
    // for TFETensorHandle that simply compares the _cTensorHandle property. It is not clear if that
    // is the right thing to do for TFETensorHandle instances in the general case.
    switch (lhs, rhs) {
    case let (.boolValue(l), .boolValue(r)): return l == r
    case let (.intValue(l), .intValue(r)): return l == r
    case let (.floatValue(l), .floatValue(r)): return l == r
    case let (.doubleValue(l), .doubleValue(r)): return l == r
    case let (.stringValue(l), .stringValue(r)): return l == r
    case let (.boolArray(l), .boolArray(r)): return l == r
    case let (.intArray(l), .intArray(r)): return l == r
    case let (.floatArray(l), .floatArray(r)): return l == r
    case let (.doubleArray(l), .doubleArray(r)): return l == r
    case let (.stringArray(l), .stringArray(r)): return l == r
    case let (.constTensor(l), .constTensor(r)):
        return TFETensorHandle.areHandlesEquivalent(l, r)
    case let (.tensorDataTypeValue(l), .tensorDataTypeValue(r)): return l == r
    case let (.tensorFunctionPointer(l), .tensorFunctionPointer(r)): return l == r
    case let (.tensorDataTypeArray(l), .tensorDataTypeArray(r)): return l == r
    case let (.optionalTensorShape(l), .optionalTensorShape(r)): return l == r
    case let (.optionalTensorShapeArray(l), .optionalTensorShapeArray(r)): return l == r
    default: return false
    }
}

// TODO(https://bugs.swift.org/browse/TF-693): This is not thread safe!
struct LazyTensorTraceCache {
    // Cache from signature to traces that match signature.
    static private var cache: [String: [LazyTensorTrace]] = [:]
    static func clearCache() { cache.removeAll() }

    // Returns a `MaterializationTraceInfo` with possibly some constants promoted to inputs.
    static func traceWithPromotedConstants(_ traceInfo: MaterializationTraceInfo) -> MaterializationTraceInfo {
        let trace = traceInfo.trace
        guard var traces = cache[trace.signature] else {
            cache[trace.signature] = [trace]
            return traceInfo
        }
        for cachedTrace in traces {
            if let promotedTrace = traceWithPromotedConstants(traceInfo, cachedTrace) {
                debugLog("Promoted: \(promotedTrace)\n")
                return promotedTrace
            }
        }
        // No match found; cache and return the input `traceInfo` itself.
        traces.append(trace)
        return traceInfo
    }

    static private func traceWithPromotedConstants(
        _ traceInfo: MaterializationTraceInfo,
        _ cachedTrace: LazyTensorTrace
    ) -> MaterializationTraceInfo? {
        let currentTrace = traceInfo.trace
        if currentTrace.operations.count != cachedTrace.operations.count { return nil }
        var promotableConstants: [(Int, TFETensorHandle)] = []
        for (i, current) in currentTrace.operations.enumerated() {
            let cached = cachedTrace.operations[i]
            if let (currentTensor, cachedTensor) = promotableConstant(current, cached) {
                if TFETensorHandle.areTensorsEqual(currentTensor, cachedTensor) { continue }
                promotableConstants.append((i, currentTensor))
                continue
            }
            // TODO: we might avoid running the following check based on results of promotableConstant
            if LazyTensorOperation.areEquivalent(current, cached) { continue }
            return nil
        }

        let newConcreteInputs: [TFETensorHandle] = promotableConstants.map { return $0.1 }
        let newOperations = currentTrace.operations
        let newInputs = promotableConstants.map {
            (promotableConstant: (Int, TFETensorHandle)) -> LazyTensorOperation in
            let constantOp = newOperations[promotableConstant.0]
            constantOp.name = "Placeholder"
            constantOp.attributes.removeValue(forKey: "value")
            return constantOp
        }
        let newTrace = LazyTensorTrace(
            inputs: currentTrace.inputs + newInputs,
            operations: newOperations,
            outputs: currentTrace.outputs)
        return MaterializationTraceInfo(
            lazyOperations: traceInfo.lazyOperations,
            trace: newTrace,
            concreteInputs: traceInfo.concreteInputs + newConcreteInputs)
    }

    /// If `current` and `cached` are compatible constants, returns the constant tensors.
    static private func promotableConstant(
        _ current: LazyTensorOperation,
        _ cached: LazyTensorOperation
    ) -> (TFETensorHandle, TFETensorHandle)? {
        if (current.name != "Const" || cached.name != "Const") { return nil }
        let currentValue = current.attributes["value"]!
        let cachedValue = cached.attributes["value"]!
        guard case let .constTensor(currentTensor) = currentValue else { return nil }
        guard case let .constTensor(cachedTensor) = cachedValue else { return nil }
        let currentDtype = TFE_TensorHandleDataType(currentTensor._cTensorHandle)
        let cachedDtype = TFE_TensorHandleDataType(cachedTensor._cTensorHandle)
        if currentDtype == TF_VARIANT || currentDtype == TF_RESOURCE { return nil }
        if cachedDtype == TF_VARIANT || cachedDtype == TF_RESOURCE { return nil }
        return (currentTensor.shape == cachedTensor.shape) && (currentDtype == cachedDtype)
            ? (currentTensor, cachedTensor)
            : nil
    }
}
