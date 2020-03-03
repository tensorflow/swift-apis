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

extension TFETensorHandle: Equatable {}

public func == (_ lhs: TFETensorHandle, _ rhs: TFETensorHandle) -> Bool {
  return lhs._cTensorHandle == rhs._cTensorHandle
}

extension TFETensorHandle {
  /// Returns true if the underlying tensors are equal.
  func elementsEqual(_ other: TFETensorHandle) -> Bool {
    let selfDtype = TFE_TensorHandleDataType(self._cTensorHandle)
    let otherDtype = TFE_TensorHandleDataType(other._cTensorHandle)
    precondition(
      selfDtype == otherDtype && selfDtype != TF_VARIANT && selfDtype != TF_RESOURCE,
      "Datatypes of tensor handles don't match.")
    let op = TFE_Op("Equal", 1)
    op.updateAttribute("T", TensorDataType(selfDtype))
    op.addInput(self)
    op.addInput(other)
    let result: Tensor<Bool> = op.execute(Int(1))
    return result.scalars.allSatisfy { $0 }
  }
}

extension LazyTensorHandle {
  func isEquivalent(to other: LazyTensorHandle) -> Bool {
    switch (self.handle, other.handle) {
    case let (.concrete(x, _), .concrete(y, _)):
      return x == y
    case let (.symbolic(x, xi, _), .symbolic(y, yi, _)):
      return xi == yi && x.id == y.id
    default: return false
    }
  }
}

extension LazyTensorOperation.Input {
  /// Returns true if these inputs are equivalent when comparing lazy tensor traces.
  func isEquivalent(to other: LazyTensorOperation.Input) -> Bool {
    switch (self, other) {
    case let (.single(l), .single(r)):
      return l.isEquivalent(to: r)
    case let (.list(l), .list(r)):
      return l.elementsEqual(r, by: { $0.isEquivalent(to: $1) })
    default:
      return false
    }
  }
}

extension LazyTensorOperation {
  /// Returns true if these operations are equivalent when comparing lazy tensor traces.
  func isEquivalent(to other: LazyTensorOperation) -> Bool {
    return self.name == other.name && self.outputCount == other.outputCount
      && self.deviceName == other.deviceName
      && self.inputs.elementsEqual(other.inputs, by: { $0.isEquivalent(to: $1) })
      && self.attributes == other.attributes
  }
}

// TODO(TF-693): This is not thread safe!
struct LazyTensorTraceCache {
  /// Cache from signature to traces that match signature.
  static private var cache: [String: [LazyTensorTrace]] = [:]
  static func clearCache() { cache.removeAll() }

  /// Returns a `MaterializationTraceInfo` with possibly some constants promoted to inputs.
  static func traceWithPromotedConstants(
    _ traceInfo: MaterializationTraceInfo
  ) -> MaterializationTraceInfo {
    let trace = traceInfo.trace
    guard var traces = cache[trace.signature] else {
      cache[trace.signature] = [trace]
      return traceInfo
    }
    for cachedTrace in traces {
      if let promotedTrace = traceInfo.withPromotedConstants(cachedTrace: cachedTrace) {
        debugLog("Promoted: \(promotedTrace)\n")
        return promotedTrace
      }
    }
    // No match found; cache and return the input `traceInfo` itself.
    traces.append(trace)
    return traceInfo
  }
}

extension MaterializationTraceInfo {
  fileprivate func withPromotedConstants(cachedTrace: LazyTensorTrace)
    -> MaterializationTraceInfo?
  {
    let currentTrace = self.trace
    if currentTrace.operations.count != cachedTrace.operations.count { return nil }
    var promotableConstants: [(Int, TFETensorHandle)] = []
    for (i, current) in currentTrace.operations.enumerated() {
      let cached = cachedTrace.operations[i]
      if let (currentTensor, cachedTensor) = Self.promotableConstants(current, cached) {
        if currentTensor.elementsEqual(cachedTensor) { continue }
        promotableConstants.append((i, currentTensor))
        continue
      }
      // TODO: we might avoid running the following check based on results of promotableConstant
      if current.isEquivalent(to: cached) { continue }
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
      lazyOperations: self.lazyOperations,
      trace: newTrace,
      concreteInputs: self.concreteInputs + newConcreteInputs)
  }

  /// If `current` and `cached` are compatible constants, returns the constant tensors.
  static private func promotableConstants(
    _ current: LazyTensorOperation,
    _ cached: LazyTensorOperation
  ) -> (TFETensorHandle, TFETensorHandle)? {
    if current.name != "Const" || cached.name != "Const" { return nil }
    let currentValue = current.attributes["value"]!
    let cachedValue = cached.attributes["value"]!
    guard case let .constTensor(currentTensor) = currentValue,
      case let .constTensor(cachedTensor) = cachedValue
    else { return nil }
    let currentDtype = TFE_TensorHandleDataType(currentTensor._cTensorHandle)
    let cachedDtype = TFE_TensorHandleDataType(cachedTensor._cTensorHandle)
    if currentDtype == TF_VARIANT || currentDtype == TF_RESOURCE { return nil }
    if cachedDtype == TF_VARIANT || cachedDtype == TF_RESOURCE { return nil }
    return currentTensor.shape == cachedTensor.shape && currentDtype == cachedDtype
      ? (currentTensor, cachedTensor)
      : nil
  }
}
