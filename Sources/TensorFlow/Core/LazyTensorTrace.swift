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

/// A class describing the tensor operations that need to be executed
/// to evaluate a given `LazyTensorOperation`.
class LazyTensorTrace {
    // inputs will be "placeholder" nodes.
    var inputs: [LazyTensorOperation] = []
    var inputValues: [TFETensorHandle] = []
    var operations: [LazyTensorOperation] = []
    var outputs: [LazyTensorOperation] = []
    var originalOutputs: [LazyTensorOperation] = []

    init(_ lazyOp: LazyTensorOperation) {
        // TODO: We only pick operations on which `lazyOp` depends on. Note that
        // there may be other live tensors that could also be materialized at
        // this time. e.g.,
        //   x = a + b
        //   y = x + c
        // For `x`, only `a + b` is extracted. One optimization is to also include
        // `y = x + c` into the trace so that we don't have the overhead of creating
        // another trace when we need to materialize `y`.
        //
        _ = collectLazyOperation(lazyOp)
        lazyOpsCache.removeAll()
    }

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

    private var lazyOpsCache: [ObjectIdentifier: LazyTensorOperation] = [:]

    private func updateOperationAndCache(
        _ id: ObjectIdentifier, _ node: LazyTensorOperation
    ) {
        lazyOpsCache[id] = node
        node.id = "\(operations.count)"
        operations.append(node)
    }

    private func makeConstTensor(with handle: TFETensorHandle) -> LazyTensor {
        let cTensorHandle = handle._cTensorHandle
        let result = LazyTensorOperation("Const", 1)
        let dtype = TensorDataType(TFE_TensorHandleDataType(cTensorHandle))
        let dtypeAttr = LazyTensorOperation.Attribute.tensorDataTypeValue(dtype)
        result.attributes = [
            "dtype": dtypeAttr,
            "value": LazyTensorOperation.Attribute.constTensor(handle)]
        updateOperationAndCache(ObjectIdentifier(handle), result)
        return LazyTensor(_lazy: result, index: 0)
    }

    private func makePlaceholderTensor(
        with handle: TFETensorHandle
    ) -> LazyTensor {
        let cTensorHandle = handle._cTensorHandle
        let dtype = TensorDataType(TFE_TensorHandleDataType(cTensorHandle))
        let dtypeAttr = LazyTensorOperation.Attribute.tensorDataTypeValue(dtype)
        let placeholder = LazyTensorOperation("Placeholder", 1)
        placeholder.attributes = ["dtype": dtypeAttr]
        updateOperationAndCache(ObjectIdentifier(handle), placeholder)
        inputs.append(placeholder)
        inputValues.append(handle)
        return LazyTensor(_lazy: placeholder, index: 0)
    }

    private func makeConstTensorOrPlaceholder(
        with handle: TFETensorHandle, asConst: Bool
    ) -> LazyTensor {
        let id = ObjectIdentifier(handle)
        if let lazyOp = lazyOpsCache[id] {
            return LazyTensor(_lazy: lazyOp, index: 0)
        }
        return asConst
            ? makeConstTensor(with: handle)
            : makePlaceholderTensor(with: handle)
    }

    /// Return the original tensor or a concrete tensor that is promoted to a
    /// placeholder input.
    private func maybePromotedTensor(_ lazyHandle: LazyTensor) -> LazyTensor {
        switch lazyHandle.handle {
        case .concrete(let h, let materialized):
            return makeConstTensorOrPlaceholder(
                with: h, asConst: !materialized)
        case .symbolic(let lazyOp, let index, _):
            if let outputs = lazyOp.outputs {
                return makeConstTensorOrPlaceholder(with: outputs[index], asConst: false)
            } else {
                return LazyTensor(_lazy: collectLazyOperation(lazyOp), index: index)
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

        let newLazyOp = LazyTensorOperation(lazyOp.name, lazyOp.outputCount)
        newLazyOp.attributes = lazyOp.attributes
        newLazyOp.inputs = lazyOp.inputs.map { maybePromotedInput($0) }
        updateOperationAndCache(id, newLazyOp)

        if LazyTensor.isLive(lazyOp) {
            outputs.append(newLazyOp)
            originalOutputs.append(lazyOp)
        }
        return newLazyOp
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
