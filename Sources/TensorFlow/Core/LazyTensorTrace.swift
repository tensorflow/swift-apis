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

/// A struct representing a trace of `LazyTensorOperation` instances.
struct LazyTensorTrace {
  let inputs: [LazyTensorOperation]
  let operations: [LazyTensorOperation]
  let outputs: [LazyTensorOperation]

  var signature: String {
    let inputsDesc: [String] = inputs.map { input in
      let dtypeAttr = input.attributes["dtype"]!
      return "\(input.outputName): \(dtypeAttr)"
    }
    let inputDesc = inputsDesc.joined(separator: ", ")
    let outputsDesc = outputs.map { $0.outputName }
    let outputDesc = outputsDesc.joined(separator: ", ")
    return "lazyTrace_\(operations.count)(\(inputDesc)) -> (\(outputDesc))"
  }
}

extension LazyTensorTrace: CustomStringConvertible {
  var description: String {
    var result = "\(signature) {\n"
    for op in operations where op.name != "Placeholder" {
      result += "  \(op)\n"
    }
    result += "}"
    return result
  }
}

/// A struct representing information required to materialize the given
/// `LazyTensorOperation` instances.
struct MaterializationTraceInfo {
  /// The operations that need to be materialized. These correspond to the outputs of `trace`.
  let lazyOperations: [LazyTensorOperation]
  /// Specification of the trace that can be evalaute to materialize `lazyOperations`.
  let trace: LazyTensorTrace
  /// Concrete tensor values for evaluating `trace`.
  let concreteInputs: [TFETensorHandle]
}

/// A builder class that provides various mechanisms to extract traces for
/// evaluating a given collection of `LazyTensorOperation` instances.
class LazyTensorTraceBuilder {
  /// Collect all the direct and transitive dependencies of `lazyOperations`
  /// and package it in a `MaterializationTraceInfo`.
  static func materializationTraceInfo(
    _ lazyOperations: [LazyTensorOperation]
  ) -> MaterializationTraceInfo {
    // TODO: We only pick operations on which `lazyOp` depends on. Note that
    // there may be other live tensors that could also be materialized at
    // this time. e.g.,
    //   x = a + b
    //   y = x + c
    // For `x`, only `a + b` is extracted. One optimization is to also include
    // `y = x + c` into the trace so that we don't have the overhead of creating
    // another trace when we need to materialize `y`.
    //
    let builder = LazyTensorTraceBuilder()
    for lazyOp in lazyOperations { _ = builder.collectLazyOperation(lazyOp) }
    let trace = LazyTensorTrace(
      inputs: builder.inputs,
      operations: builder.operations,
      outputs: builder.outputs)
    let materializationTraceInfo = MaterializationTraceInfo(
      lazyOperations: builder.originalOutputs,
      trace: trace,
      concreteInputs: builder.inputValues)
    return LazyTensorContext.local.shouldPromoteConstants
      ? LazyTensorTraceCache.traceWithPromotedConstants(materializationTraceInfo)
      : materializationTraceInfo
  }

  static func materializationTraceInfo(
    _ lazyOperation: LazyTensorOperation
  ) -> MaterializationTraceInfo {
    return materializationTraceInfo([lazyOperation])
  }

  /// Returns a trace obtained by tracing the given function.
  static func trace<In: TensorGroup, Out: TensorGroup>(_ fn: (In) -> Out) -> LazyTensorTrace {
    precondition(_ThreadLocalState.useLazyTensor, "Lazy tensor is not enabled for tracing.")
    // Disable shape tracking and reset to original state when done.
    let isShapeTrackingEnabled = LazyTensorContext.local.isShapeTrackingEnabled
    defer { LazyTensorContext.local.isShapeTrackingEnabled = isShapeTrackingEnabled }
    LazyTensorContext.local.isShapeTrackingEnabled = false

    // Set up inputs for running `fn`.
    let inputOps = In._typeList.map { Self.makePlaceholder(dataType: $0) }
    let inputHandles = inputOps.map { LazyTensorHandle(_lazy: $0, index: 0) }
    let input = In(_handles: inputHandles)

    // Run the function.
    let output: TensorArrayProtocol = fn(input)

    // Set up the closure that determines if a `LazyTensorOperation` should be an output.
    let outputLazyOperations = output._tensorHandles.map {
      (handle: _AnyTensorHandle) -> LazyTensorOperation in
      let lazyOp = lazyTensorOperation(handle)
      precondition(lazyOp != nil, "Found a non-lazy tensor in output when tracing.")
      return lazyOp!
    }

    // Create the builder and get the trace.
    let builder = LazyTensorTraceBuilder()
    builder.neverPromoteConstants = true
    builder.generateOutputs = false
    // Set up the inputs for the builder as we need to have them in a specific order.
    for inputOp in inputOps {
      builder.updateOperationAndCache(ObjectIdentifier(inputOp), inputOp)
    }
    builder.inputs = inputOps
    for lazyOp in outputLazyOperations { _ = builder.collectLazyOperation(lazyOp) }
    // Set up the outputs for the builder as we need to have them in a specific order.
    builder.outputs = outputLazyOperations.map { builder.lazyOpsCache[ObjectIdentifier($0)]! }
    return LazyTensorTrace(
      inputs: builder.inputs,
      operations: builder.operations,
      outputs: builder.outputs)
  }

  // inputs will be "placeholder" nodes.
  private var inputs: [LazyTensorOperation] = []
  private var inputValues: [TFETensorHandle] = []
  private var operations: [LazyTensorOperation] = []
  private var outputs: [LazyTensorOperation] = []
  private var originalOutputs: [LazyTensorOperation] = []
  private var lazyOpsCache: [ObjectIdentifier: LazyTensorOperation] = [:]
  /// A flag that controls promotion of constants to inputs.
  private var neverPromoteConstants: Bool = false
  /// A flag that controls whether outputs should be automatically generated.
  private var generateOutputs: Bool = true

  private func updateOperationAndCache(
    _ id: ObjectIdentifier, _ node: LazyTensorOperation
  ) {
    lazyOpsCache[id] = node
    node.id = "\(operations.count)"
    operations.append(node)
  }

  private func makeConstTensor(with handle: TFETensorHandle) -> LazyTensorHandle {
    let cTensorHandle = handle._cTensorHandle
    let result = LazyTensorOperation("Const", 1)
    let dtype = TensorDataType(TFE_TensorHandleDataType(cTensorHandle))
    let dtypeAttr = LazyTensorOperation.Attribute.tensorDataTypeValue(dtype)
    result.attributes = [
      "dtype": dtypeAttr,
      "value": LazyTensorOperation.Attribute.constTensor(handle),
    ]
    updateOperationAndCache(ObjectIdentifier(handle), result)
    return LazyTensorHandle(_lazy: result, index: 0)
  }

  /// Returns the `LazyTensorOperation`, if any, for this handle.
  private static func lazyTensorOperation(_ handle: _AnyTensorHandle) -> LazyTensorOperation? {
    guard case let .symbolic(lazyOp, _, _)? = (handle as? LazyTensorHandle)?.handle else {
      return nil
    }
    return lazyOp
  }

  private static func makePlaceholder(dataType: TensorDataType) -> LazyTensorOperation {
    let placeholder = LazyTensorOperation("Placeholder", 1)
    let dtypeAttr = LazyTensorOperation.Attribute.tensorDataTypeValue(dataType)
    placeholder.attributes = ["dtype": dtypeAttr]
    return placeholder
  }

  private func makePlaceholderTensor(handle: TFETensorHandle) -> LazyTensorHandle {
    let cTensorHandle = handle._cTensorHandle
    let dtype = TensorDataType(TFE_TensorHandleDataType(cTensorHandle))
    let placeholder = Self.makePlaceholder(dataType: dtype)
    updateOperationAndCache(ObjectIdentifier(handle), placeholder)
    inputs.append(placeholder)
    inputValues.append(handle)
    return LazyTensorHandle(_lazy: placeholder, index: 0)
  }

  private func makeConstTensorOrPlaceholder(
    with handle: TFETensorHandle, asConst: Bool
  ) -> LazyTensorHandle {
    let id = ObjectIdentifier(handle)
    if let lazyOp = lazyOpsCache[id] {
      return LazyTensorHandle(_lazy: lazyOp, index: 0)
    }
    return asConst || neverPromoteConstants
      ? makeConstTensor(with: handle)
      : makePlaceholderTensor(handle: handle)
  }

  /// Returns the original tensor or a concrete tensor that is promoted to a
  /// placeholder input.
  private func maybePromotedTensor(_ lazyHandle: LazyTensorHandle) -> LazyTensorHandle {
    switch lazyHandle.handle {
    case .concrete(let h, let materialized):
      return makeConstTensorOrPlaceholder(
        with: h, asConst: !materialized)
    case .symbolic(let lazyOp, let index, _):
      if let outputs = lazyOp.outputs {
        return makeConstTensorOrPlaceholder(with: outputs[index], asConst: false)
      } else {
        return LazyTensorHandle(_lazy: collectLazyOperation(lazyOp), index: index)
      }
    }
  }

  private func maybePromotedInput(
    _ input: LazyTensorOperation.Input
  ) -> LazyTensorOperation.Input {
    switch input {
    case .single(let h):
      return LazyTensorOperation.Input.single(maybePromotedTensor(h))
    case .list(let elements):
      return LazyTensorOperation.Input.list(
        elements.map { maybePromotedTensor($0) })
    }
  }

  private func collectLazyOperation(
    _ lazyOp: LazyTensorOperation
  ) -> LazyTensorOperation {
    let id = ObjectIdentifier(lazyOp)
    if let cachedLazyOp = lazyOpsCache[id] {
      return cachedLazyOp
    }
    lazyOp.maybeMaterializeInputs()
    precondition(
      lazyOp.name != "Placeholder",
      "The operation cannot already be a placeholder.")
    let newLazyOp = LazyTensorOperation(lazyOp.name, lazyOp.outputCount)
    newLazyOp.attributes = lazyOp.attributes
    newLazyOp.inputs = lazyOp.inputs.map { maybePromotedInput($0) }
    updateOperationAndCache(id, newLazyOp)

    if generateOutputs && LazyTensorHandle.isLive(lazyOp) {
      outputs.append(newLazyOp)
      originalOutputs.append(lazyOp)
    }
    return newLazyOp
  }
}
