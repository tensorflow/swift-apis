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

class TFGraph {
    /// The `TF_Operation *` type.
    typealias CTFOperation = OpaquePointer
    
    enum Input {
        case single(TF_Output)
        case list([TF_Output])
    }

    /// The pointer to the underlying TF Graph.
    let cTFGraph: CTFGraph = TF_NewGraph()
    var inputs: [TF_Output] = []
    var nodes: [CTFOperation?] = []
    var outputs: [TF_Output] = []
    var outputGroups: [Int] = []
    var name: String { "lazyTrace_\(nodeCounter)" }

    /// A status object to pass to TF graph building operations.
    private let status: CTFStatus = TF_NewStatus()

    /// Counter that is used for number the generated graph nodes. 
    private var nodeCounter: Int = 0

    init(_ lazyTrace: LazyTensorTrace) {
        var nodesCache: [ObjectIdentifier: CTFOperation?] = [:]
        for op in lazyTrace.operations {
            let opInputs = op.inputs.map { input -> TFGraph.Input in
                switch input {
                case LazyTensorOperation.Input.single(let h):
                    return TFGraph.Input.single(makeTFOutput(h, nodesCache))
                case LazyTensorOperation.Input.list(let elements):
                    let tfInputs = elements.map { makeTFOutput($0, nodesCache) }
                    return TFGraph.Input.list(tfInputs)
                }
            }
            let graphNode = makeTFGraphNode(
                name: op.name,
                attrs: op.attributes,
                inputs: opInputs,
                device: op.deviceName)
            nodesCache[ObjectIdentifier(op)] = graphNode
            if op.name != "Placeholder" { nodes.append(graphNode) }
        }
        self.inputs = lazyTrace.inputs.map {
            TF_Output(oper: nodesCache[ObjectIdentifier($0)]!, index: 0)
        }
        for output in lazyTrace.outputs {
            let graphNode = nodesCache[ObjectIdentifier(output)]!
            outputGroups.append(output.outputCount)
            outputs += Array((0..<output.outputCount).map {
                    TF_Output(oper: graphNode, index: Int32($0)) })
        }
    }

    deinit {
        TF_DeleteGraph(cTFGraph)
        TF_DeleteStatus(status)
    }

    private func newNodeName(base: String) -> String {
        let name = "\(base)_\(nodeCounter)"
        nodeCounter += 1
        return name
    }

    private func updateAttribute(
        _ desc: CTFOperationDescription,
        _ name: String,
        _ attrValue: LazyTensorOperation.Attribute
    ) {
        switch attrValue {
        case LazyTensorOperation.Attribute.tensorDataTypeValue(let value):
            TF_SetAttrType(desc, name, value._cDataType)
        case LazyTensorOperation.Attribute.boolValue(let value):
            TF_SetAttrBool(desc, name, value ? 1 : 0)
        case LazyTensorOperation.Attribute.intValue(let value):
            TF_SetAttrInt(desc, name, Int64(value))
        case LazyTensorOperation.Attribute.floatValue(let value):
            TF_SetAttrFloat(desc, name, value)
        case LazyTensorOperation.Attribute.stringValue(let value): do {
                value.utf8CString.withUnsafeBufferPointer { buffer in
                    // utf8CString is null-terminated; TF_SetAttrString wants
                    // non-null-terminated.
                    TF_SetAttrString(
                        desc, name, buffer.baseAddress, buffer.count - 1)
                }
            }
        case LazyTensorOperation.Attribute.intArray(let values): do {
                let values64 = values.map { Int64($0) }
                values64.withUnsafeBufferPointer { buffer in
                    TF_SetAttrIntList(
                        desc, name, buffer.baseAddress, Int32(buffer.count))
                }
            }
        case LazyTensorOperation.Attribute.constTensor(let value): do {
                let cTensor = TFE_TensorHandleResolve(
                    value._cTensorHandle, status)
                checkOk(status)
                TF_SetAttrTensor(desc, name, cTensor!, status)
            }
        case LazyTensorOperation.Attribute.tensorDataTypeArray(let values): do {
                values.withUnsafeBufferPointer { buffer in
                    buffer.withMemoryRebound(to: TF_DataType.self) {
                        reboundBuffer in
                        TF_SetAttrTypeList(
                            desc, name,
                            reboundBuffer.baseAddress,
                            Int32(reboundBuffer.count))
                    }
                }
            }
        case LazyTensorOperation.Attribute.optionalTensorShapeArray(let values): do {
                let flattenedDims = values.flatMap { (tensorShapeOpt) -> [Int64] in
                    if let tensorShape = tensorShapeOpt {
                        return tensorShape.dimensions.map(Int64.init)
                    }
                    return []
                }
                let ranks = values.map { shape in (shape?.rank).map(Int32.init) ?? -1 }
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
                            TF_SetAttrShapeList(
                                desc, name,
                                dimsBuffer.baseAddress,
                                ranksBuffer.baseAddress,
                                Int32(ranksBuffer.count))
                        }
                    }
                }
            }
        default:
            assert(false, "Unhandled attribute \(name):\(attrValue)")
        }
    }

    private func makeTFGraphNode(
        name: String,
        attrs: [String: LazyTensorOperation.Attribute],
        inputs: [Input],
        device: String?
    ) -> CTFOperation? {
        // Create a new graph node now.
        let desc: CTFOperationDescription! = TF_NewOperation(
            cTFGraph, name, newNodeName(base: name))

        // Set Attributes
        for (name, value) in attrs {
            updateAttribute(desc, name, value)
        }

        // Add Inputs
        for input in inputs {
            switch input {
            case Input.single(let singleInput):
                TF_AddInput(desc, singleInput)
            case Input.list(let inputList):
                inputList.withUnsafeBufferPointer { buffer in
                    TF_AddInputList(desc, buffer.baseAddress, Int32(buffer.count))
                }
            }
        }
        
        if let device = device { TF_SetDevice(desc, device) }
        
        // Finalize operation.
        let graphNode = TF_FinishOperation(desc, status)
        checkOk(status)
        return graphNode!
    }

    private func makeTFConstNode(_ handle: TFETensorHandle) -> TF_Output {
        let cTensorHandle = handle._cTensorHandle
        let cTensor = TFE_TensorHandleResolve(cTensorHandle, status)
        checkOk(status)
        let desc = TF_NewOperation(cTFGraph, "Const", newNodeName(base: "Const"))
        checkOk(status)
        TF_SetAttrType(desc, "dtype", TFE_TensorHandleDataType(cTensorHandle))
        TF_SetAttrTensor(desc, "value", cTensor, status)
        checkOk(status)
        let constNode = TF_FinishOperation(desc, status)
        return TF_Output(oper: constNode, index: 0)
    }

    private func makeTFOutput(
        _ lazyHandle: LazyTensor,
        _ nodesCache: [ObjectIdentifier: CTFOperation?]) -> TF_Output {
        if case let LazyTensor.Handle.symbolic(
            lazyOp, index, _) = lazyHandle.handle {
            let id = ObjectIdentifier(lazyOp)
            return TF_Output(oper: nodesCache[id]!, index: Int32(index))
        }
        assert(false, "Should only have symbolic inputs.")
    }
}

class TFFunction {
    let cTFFunction: CTFFunction
    let outputCount: Int
    let outputGroups: [Int]
    
    init(_ lazyTrace: LazyTensorTrace) {
        let status: CTFStatus = TF_NewStatus()
        defer { TF_DeleteStatus(status) }
        let graph = TFGraph(lazyTrace)
        let cTFGraph = graph.cTFGraph
        let inputs = graph.inputs
        let outputs = graph.outputs
        self.outputCount = outputs.count
        self.outputGroups = graph.outputGroups
        self.cTFFunction = graph.nodes.withUnsafeBufferPointer {
            operations -> CTFFunction in
            let base = operations.baseAddress
            let tracedGraphFn = TF_GraphToFunction(cTFGraph, graph.name,
                /*append_hash_to_fn_name*/ 1,
                /*num_opers*/ Int32(operations.count),
                /*opers*/ base,
                /*numinputs*/ Int32(inputs.count),
                /*inputs*/ inputs,
                /*noutputs*/ Int32(outputs.count),
                /*outputs*/ outputs,
                /*outputnames*/ nil,
                /*functionoptions*/ nil, "",
                status)
            checkOk(status)
            if _RuntimeConfig.printsDebugLog {
                var len: Int = 0
                let funcDebugStr = TF_FunctionDebugString(tracedGraphFn, &len)!
                debugLog("The traced function is:\n\(String(cString: funcDebugStr))")
                free(funcDebugStr)
                debugLog("Corresponding lazy tensor operations:\n")
                for output in graph.outputs {
                    debugLog("  \(output)")
                }
            }
            return tracedGraphFn!
        }
        
        let eagerContext = _TFCGetGlobalEagerContext()
        TFE_ContextAddFunction(eagerContext, self.cTFFunction, status)
        checkOk(status)
    }

    func execute(_ inputs: [TFETensorHandle]) -> [TFETensorHandle] {
        let status: CTFStatus = TF_NewStatus()
        defer { TF_DeleteStatus(status) }
        
        let eagerContext = _TFCGetGlobalEagerContext()
        let fname = TF_FunctionName(cTFFunction)!
        let eagerOp: CTFEOp! = TFE_NewOp(eagerContext, fname, status)
        defer { TFE_DeleteOp(eagerOp) }
        checkOk(status)

        let deviceName = _ExecutionContext.global.currentDeviceName
        if let deviceName = deviceName {
            debugLog("Placing the trace func on device \(deviceName).")
            TFE_OpSetDevice(eagerOp, deviceName, status)
            checkOk(status)
        }

        for input in inputs {
            TFE_OpAddInput(eagerOp, input._cTensorHandle, status)
            checkOk(status)
        }

        var returnValues = [CTensorHandle?](repeating: nil, count: outputCount)
        var outputReturnValueCount = Int32(outputCount)
        TFE_Execute(eagerOp, &returnValues, &outputReturnValueCount, status)
        checkOk(status)

        return returnValues.map  { TFETensorHandle(_owning: $0!) }
    }
}

extension TFFunction: CustomStringConvertible {
    public var description: String {
        var len: Int = 0
        let funcDebugStr = TF_FunctionDebugString(cTFFunction, &len)!
        let result = String(cString: funcDebugStr)
        free(funcDebugStr)
        return result
    }
}

