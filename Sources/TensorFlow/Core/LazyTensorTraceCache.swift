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
}

extension LazyTensorHandle {
    static func areHandlesEquivalent(_ lhs: LazyTensorHandle, _ rhs: LazyTensorHandle) -> Bool {
        switch (lhs.handle, rhs.handle) {
        case (let .concrete(x, _), let .concrete(y, _)):
            return TFETensorHandle.areHandlesEquivalent(x, y)
        case (let .symbolic(x, xi, _), let .symbolic(y, yi, _)):
            return (xi == yi) && (x.id == y.id)
        default: return false
        }
    }
}

extension LazyTensorOperation {
    /// Returns true if these inputs are equivalent when comparing lazy tensor traces.
    static func areInputsEquivalent(_ lhs: Input, _ rhs: Input) -> Bool {
        switch (lhs, rhs) {
        case (.single(let l), .single(let r)):
            return LazyTensorHandle.areHandlesEquivalent(l, r)
        case (.list(let l), .list(let r)):
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
            lhs.inputs.elementsEqual(rhs.inputs, by: {
                LazyTensorOperation.areInputsEquivalent($0, $1)
            }) &&
           (lhs.attributes == rhs.attributes)
    }
}

extension LazyTensorOperation.Attribute: Equatable {}
func ==(_ lhs: LazyTensorOperation.Attribute, _ rhs: LazyTensorOperation.Attribute) -> Bool {
    // We cannot rely on the derived conformance, because it would force us to add a conformance
    // for TFETensorHandle that simply compares the _cTensorHandle property. It is not clear if that
    // is the right thing to do for TFETensorHandle instances in the general case.
    switch (lhs, rhs) {
    case (.boolValue(let l), .boolValue(let r)): return l == r
    case (.intValue(let l), .intValue(let r)): return l == r
    case (.floatValue(let l), .floatValue(let r)): return l == r
    case (.doubleValue(let l), .doubleValue(let r)): return l == r
    case (.stringValue(let l), .stringValue(let r)): return l == r
    case (.boolArray(let l), .boolArray(let r)): return l == r
    case (.intArray(let l), .intArray(let r)): return l == r
    case (.floatArray(let l), .floatArray(let r)): return l == r
    case (.doubleArray(let l), .doubleArray(let r)): return l == r
    case (.stringArray(let l), .stringArray(let r)): return l == r
    case (.constTensor(let l), .constTensor(let r)):
        return TFETensorHandle.areHandlesEquivalent(l, r)
    case (.tensorDataTypeValue(let l), .tensorDataTypeValue(let r)): return l == r
    case (.tensorFunctionPointer(let l), .tensorFunctionPointer(let r)): return l == r
    case (.tensorDataTypeArray(let l), .tensorDataTypeArray(let r)): return l == r
    case (.optionalTensorShape(let l), .optionalTensorShape(let r)): return l == r
    case (.optionalTensorShapeArray(let l), .optionalTensorShapeArray(let r)): return l == r
    default: return false
    }
}

struct LazyTensorTraceCache {
    // Cache from signature to traces that match signature.
    static private var cache: [String: [LazyTensorTrace]] = [:]

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
            if let promotableConstant = promotableConstant(current, cached) {
                promotableConstants.append((i, promotableConstant))
                continue
            }
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
    ) -> TFETensorHandle? {
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
            ? currentTensor
            : nil
    }
}
