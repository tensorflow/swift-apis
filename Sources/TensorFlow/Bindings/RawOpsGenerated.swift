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

@inlinable @inline(__always)
func makeOp(_ name: String, _ nOutputs: Int) -> TFTensorOperation {
    _ExecutionContext.makeOp(name, nOutputs)
}

public enum Raw {

static let generatedTensorFlowVersion = "2.1.0"
static let generatedTensorFlowGitVersion = "v2.1.0-rc2-17-ge5bf8de"

// @_frozen // SR-9739
public enum A {
    case apples
    case oranges

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .apples: return "apples"
            case .oranges: return "oranges"
            }
        }
    }
}

// @_frozen // SR-9739
public enum DataFormat {
    case nchw
    case nhwc

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .nchw: return "NCHW"
            case .nhwc: return "NHWC"
            }
        }
    }
}

// @_frozen // SR-9739
public enum DataFormat1 {
    case ncdhw
    case ndhwc

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .ncdhw: return "NCDHW"
            case .ndhwc: return "NDHWC"
            }
        }
    }
}

// @_frozen // SR-9739
public enum DataFormat5 {
    case nchw
    case nchwVectC
    case nhwc

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .nchw: return "NCHW"
            case .nchwVectC: return "NCHW_VECT_C"
            case .nhwc: return "NHWC"
            }
        }
    }
}

// @_frozen // SR-9739
public enum DensityUnit {
    case cm
    case in_

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .cm: return "cm"
            case .in_: return "in"
            }
        }
    }
}

// @_frozen // SR-9739
public enum Direction {
    case bidirectional
    case unidirectional

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .bidirectional: return "bidirectional"
            case .unidirectional: return "unidirectional"
            }
        }
    }
}

// @_frozen // SR-9739
public enum Errors {
    case ignore
    case replace
    case strict

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .ignore: return "ignore"
            case .replace: return "replace"
            case .strict: return "strict"
            }
        }
    }
}

// @_frozen // SR-9739
public enum FinalOp {
    case div
    case id

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .div: return "Div"
            case .id: return "Id"
            }
        }
    }
}

// @_frozen // SR-9739
public enum Format {
    case empty
    case grayscale
    case rgb

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .empty: return ""
            case .grayscale: return "grayscale"
            case .rgb: return "rgb"
            }
        }
    }
}

// @_frozen // SR-9739
public enum InputMode {
    case autoSelect
    case linearInput
    case skipInput

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .autoSelect: return "auto_select"
            case .linearInput: return "linear_input"
            case .skipInput: return "skip_input"
            }
        }
    }
}

// @_frozen // SR-9739
public enum InputQuantMode {
    case minFirst
    case scaled

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .minFirst: return "MIN_FIRST"
            case .scaled: return "SCALED"
            }
        }
    }
}

// @_frozen // SR-9739
public enum LossType {
    case hingeLoss
    case logisticLoss
    case poissonLoss
    case smoothHingeLoss
    case squaredLoss

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .hingeLoss: return "hinge_loss"
            case .logisticLoss: return "logistic_loss"
            case .poissonLoss: return "poisson_loss"
            case .smoothHingeLoss: return "smooth_hinge_loss"
            case .squaredLoss: return "squared_loss"
            }
        }
    }
}

// @_frozen // SR-9739
public enum MergeOp {
    case add
    case max
    case min
    case mul

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .add: return "Add"
            case .max: return "Max"
            case .min: return "Min"
            case .mul: return "Mul"
            }
        }
    }
}

// @_frozen // SR-9739
public enum Method {
    case bilinear
    case nearest

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .bilinear: return "bilinear"
            case .nearest: return "nearest"
            }
        }
    }
}

// @_frozen // SR-9739
public enum Method4 {
    case bilinear

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .bilinear: return "bilinear"
            }
        }
    }
}

// @_frozen // SR-9739
public enum Mode {
    case minCombined
    case minFirst
    case scaled

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .minCombined: return "MIN_COMBINED"
            case .minFirst: return "MIN_FIRST"
            case .scaled: return "SCALED"
            }
        }
    }
}

// @_frozen // SR-9739
public enum Mode6 {
    case reflect
    case symmetric

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .reflect: return "REFLECT"
            case .symmetric: return "SYMMETRIC"
            }
        }
    }
}

// @_frozen // SR-9739
public enum OutputEncoding {
    case utf16Be
    case utf32Be
    case utf8

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .utf16Be: return "UTF-16-BE"
            case .utf32Be: return "UTF-32-BE"
            case .utf8: return "UTF-8"
            }
        }
    }
}

// @_frozen // SR-9739
public enum Padding {
    case same
    case valid

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .same: return "SAME"
            case .valid: return "VALID"
            }
        }
    }
}

// @_frozen // SR-9739
public enum Padding3 {
    case explicit
    case same
    case valid

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .explicit: return "EXPLICIT"
            case .same: return "SAME"
            case .valid: return "VALID"
            }
        }
    }
}

// @_frozen // SR-9739
public enum PrecisionMode {
    case fp16
    case fp32
    case int8

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .fp16: return "FP16"
            case .fp32: return "FP32"
            case .int8: return "INT8"
            }
        }
    }
}

// @_frozen // SR-9739
public enum Reduction {
    case max
    case min
    case prod
    case sum

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .max: return "max"
            case .min: return "min"
            case .prod: return "prod"
            case .sum: return "sum"
            }
        }
    }
}

// @_frozen // SR-9739
public enum ReductionType {
    case mean
    case sum

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .mean: return "MEAN"
            case .sum: return "SUM"
            }
        }
    }
}

// @_frozen // SR-9739
public enum RnnMode {
    case gru
    case lstm
    case rnnRelu
    case rnnTanh

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .gru: return "gru"
            case .lstm: return "lstm"
            case .rnnRelu: return "rnn_relu"
            case .rnnTanh: return "rnn_tanh"
            }
        }
    }
}

// @_frozen // SR-9739
public enum RoundMode {
    case halfToEven
    case halfUp

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .halfToEven: return "HALF_TO_EVEN"
            case .halfUp: return "HALF_UP"
            }
        }
    }
}

// @_frozen // SR-9739
public enum RoundMode7 {
    case halfAwayFromZero
    case halfToEven

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .halfAwayFromZero: return "HALF_AWAY_FROM_ZERO"
            case .halfToEven: return "HALF_TO_EVEN"
            }
        }
    }
}

// @_frozen // SR-9739
public enum SplitType {
    case equality
    case inequality

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .equality: return "equality"
            case .inequality: return "inequality"
            }
        }
    }
}

// @_frozen // SR-9739
public enum SplitType2 {
    case inequality

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .inequality: return "inequality"
            }
        }
    }
}

// @_frozen // SR-9739
public enum Unit {
    case byte
    case utf8Char

    @inlinable
    var cName: String {
        @inline(__always)
        get {
            switch self {
            case .byte: return "BYTE"
            case .utf8Char: return "UTF8_CHAR"
            }
        }
    }
}


@inlinable @inline(__always)
public static func a(
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("A", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func abort(
    errorMsg: String,
    exitWithoutError: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("Abort", nOutputs)
    op.updateAttribute("error_msg", errorMsg)
    op.updateAttribute("exit_without_error", exitWithoutError)
    op.execute()
}

@inlinable @inline(__always)
public static func abs<T: TensorFlowNumeric>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Abs", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func accumulateNV2<T: TensorFlowNumeric>(
    inputs: [Tensor<T>],
    shape: TensorShape?
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("AccumulateNV2", nOutputs)
    op.updateAttribute("N", inputs.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("shape", shape)
    op.addInputList(inputs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func acos<T: TensorFlowNumeric>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Acos", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func acosh<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Acosh", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func add<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Add", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func add(
    _ x: StringTensor,
    _ y: StringTensor
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("Add", nOutputs)
    op.updateAttribute("T", TensorDataType(TF_STRING))
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func addManySparseToTensorsMap<T: TensorFlowScalar>(
    sparseIndices: Tensor<Int64>,
    sparseValues: Tensor<T>,
    sparseShape: Tensor<Int64>,
    container: String,
    sharedName: String
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("AddManySparseToTensorsMap", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.addInput(sparseIndices)
    op.addInput(sparseValues)
    op.addInput(sparseShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func addN<T: TensorFlowNumeric>(
    inputs: [Tensor<T>]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("AddN", nOutputs)
    op.updateAttribute("N", inputs.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInputList(inputs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func addSparseToTensorsMap<T: TensorFlowScalar>(
    sparseIndices: Tensor<Int64>,
    sparseValues: Tensor<T>,
    sparseShape: Tensor<Int64>,
    container: String,
    sharedName: String
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("AddSparseToTensorsMap", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.addInput(sparseIndices)
    op.addInput(sparseValues)
    op.addInput(sparseShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func addV2<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("AddV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func adjustContrast<T: TensorFlowNumeric>(
    images: Tensor<T>,
    contrastFactor: Tensor<Float>,
    minValue: Tensor<Float>,
    maxValue: Tensor<Float>
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("AdjustContrast", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(images)
    op.addInput(contrastFactor)
    op.addInput(minValue)
    op.addInput(maxValue)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func adjustContrastv2<T: FloatingPoint & TensorFlowScalar>(
    images: Tensor<T>,
    contrastFactor: Tensor<Float>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("AdjustContrastv2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(images)
    op.addInput(contrastFactor)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func adjustHue<T: FloatingPoint & TensorFlowScalar>(
    images: Tensor<T>,
    delta: Tensor<Float>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("AdjustHue", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(images)
    op.addInput(delta)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func adjustSaturation<T: FloatingPoint & TensorFlowScalar>(
    images: Tensor<T>,
    scale: Tensor<Float>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("AdjustSaturation", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(images)
    op.addInput(scale)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func all<Tidx: TensorFlowIndex>(
    _ input: Tensor<Bool>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("All", nOutputs)
    op.updateAttribute("keep_dims", keepDims)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(input)
    op.addInput(reductionIndices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func allCandidateSampler(
    trueClasses: Tensor<Int64>,
    numTrue: Int64,
    numSampled: Int64,
    unique: Bool,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> (sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>, sampledExpectedCount: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("AllCandidateSampler", nOutputs)
    op.updateAttribute("num_true", numTrue)
    op.updateAttribute("num_sampled", numSampled)
    op.updateAttribute("unique", unique)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.addInput(trueClasses)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func allToAll<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    groupAssignment: Tensor<Int32>,
    concatDimension: Int64,
    splitDimension: Int64,
    splitCount: Int64
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("AllToAll", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("concat_dimension", concatDimension)
    op.updateAttribute("split_dimension", splitDimension)
    op.updateAttribute("split_count", splitCount)
    op.addInput(input)
    op.addInput(groupAssignment)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func angle<
    T: TensorFlowScalar,
    Tout: FloatingPoint & TensorFlowScalar
>(
    _ input: Tensor<T>
) -> Tensor<Tout> {
  let nOutputs = Int(1)
    let op = makeOp("Angle", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tout", Tout.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func anonymousIterator(
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("AnonymousIterator", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func anonymousIteratorV2(
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> (handle: ResourceHandle, deleter: VariantHandle) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("AnonymousIteratorV2", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func anonymousMemoryCache(
) -> (handle: ResourceHandle, deleter: VariantHandle) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("AnonymousMemoryCache", nOutputs)
    
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func anonymousMultiDeviceIterator(
    devices: [String],
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> (handle: ResourceHandle, deleter: VariantHandle) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("AnonymousMultiDeviceIterator", nOutputs)
    op.updateAttribute("devices", devices)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func anonymousRandomSeedGenerator(
    seed: Tensor<Int64>,
    seed2: Tensor<Int64>
) -> (handle: ResourceHandle, deleter: VariantHandle) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("AnonymousRandomSeedGenerator", nOutputs)
    op.addInput(seed)
    op.addInput(seed2)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func any<Tidx: TensorFlowIndex>(
    _ input: Tensor<Bool>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("Any", nOutputs)
    op.updateAttribute("keep_dims", keepDims)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(input)
    op.addInput(reductionIndices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func approximateEqual<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    tolerance: Double = 1e-05
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("ApproximateEqual", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("tolerance", tolerance)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func argMax<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex,
    OutputType: TensorFlowIndex
>(
    _ input: Tensor<T>,
    dimension: Tensor<Tidx>
) -> Tensor<OutputType> {
  let nOutputs = Int(1)
    let op = makeOp("ArgMax", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.updateAttribute("output_type", OutputType.tensorFlowDataType)
    op.addInput(input)
    op.addInput(dimension)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func argMin<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex,
    OutputType: TensorFlowIndex
>(
    _ input: Tensor<T>,
    dimension: Tensor<Tidx>
) -> Tensor<OutputType> {
  let nOutputs = Int(1)
    let op = makeOp("ArgMin", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.updateAttribute("output_type", OutputType.tensorFlowDataType)
    op.addInput(input)
    op.addInput(dimension)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func asString<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    precision: Int64 = -1,
    scientific: Bool = false,
    shortest: Bool = false,
    width: Int64 = -1,
    fill: String
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("AsString", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("precision", precision)
    op.updateAttribute("scientific", scientific)
    op.updateAttribute("shortest", shortest)
    op.updateAttribute("width", width)
    op.updateAttribute("fill", fill)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func asin<T: TensorFlowNumeric>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Asin", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func asinh<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Asinh", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func assert<T: TensorArrayProtocol>(
    condition: Tensor<Bool>,
    data: T,
    summarize: Int64 = 3
) {
  let nOutputs = 0
    let op = makeOp("Assert", nOutputs)
    op.updateAttribute("T", data._typeList)
    op.updateAttribute("summarize", summarize)
    op.addInput(condition)
    op.addInputList(data)
    op.execute()
}

@inlinable @inline(__always)
public static func assertNextDataset(
    inputDataset: VariantHandle,
    transformations: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("AssertNextDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(transformations)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func assignAddVariableOp<Dtype: TensorFlowScalar>(
    resource: ResourceHandle,
    value: Tensor<Dtype>
) {
  let nOutputs = 0
    let op = makeOp("AssignAddVariableOp", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(value)
    op.execute()
}

@inlinable @inline(__always)
public static func assignSubVariableOp<Dtype: TensorFlowScalar>(
    resource: ResourceHandle,
    value: Tensor<Dtype>
) {
  let nOutputs = 0
    let op = makeOp("AssignSubVariableOp", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(value)
    op.execute()
}

@inlinable @inline(__always)
public static func assignVariableOp<Dtype: TensorFlowScalar>(
    resource: ResourceHandle,
    value: Tensor<Dtype>
) {
  let nOutputs = 0
    let op = makeOp("AssignVariableOp", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(value)
    op.execute()
}

@inlinable @inline(__always)
public static func atan<T: TensorFlowNumeric>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Atan", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func atan2<T: FloatingPoint & TensorFlowScalar>(
    _ y: Tensor<T>,
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Atan2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(y)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func atanh<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Atanh", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func attr(
    _ a: Int64
) {
  let nOutputs = 0
    let op = makeOp("Attr", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrBool(
    _ a: Bool
) {
  let nOutputs = 0
    let op = makeOp("AttrBool", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrBoolList(
    _ a: [Bool]
) {
  let nOutputs = 0
    let op = makeOp("AttrBoolList", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrDefault(
    _ a: String = "banana"
) {
  let nOutputs = 0
    let op = makeOp("AttrDefault", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrEmptyListDefault(
    _ a: [Double]
) {
  let nOutputs = 0
    let op = makeOp("AttrEmptyListDefault", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrEnum(
    _ a: A
) {
  let nOutputs = 0
    let op = makeOp("AttrEnum", nOutputs)
    op.updateAttribute("a", a.cName)
    op.execute()
}

@inlinable @inline(__always)
public static func attrEnumList(
    _ a: [String]
) {
  let nOutputs = 0
    let op = makeOp("AttrEnumList", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrFloat(
    _ a: Double
) {
  let nOutputs = 0
    let op = makeOp("AttrFloat", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrListDefault(
    _ a: [Int32] = [5, 15]
) {
  let nOutputs = 0
    let op = makeOp("AttrListDefault", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrListMin(
    _ a: [Int32]
) {
  let nOutputs = 0
    let op = makeOp("AttrListMin", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrListTypeDefault<T: TensorFlowScalar>(
    _ a: [Tensor<T>],
    _ b: [Tensor<T>]
) {
  let nOutputs = 0
    let op = makeOp("AttrListTypeDefault", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("N", a.count)
    op.addInputList(a)
    op.addInputList(b)
    op.execute()
}

@inlinable @inline(__always)
public static func attrMin(
    _ a: Int64
) {
  let nOutputs = 0
    let op = makeOp("AttrMin", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrPartialShape(
    _ a: TensorShape?
) {
  let nOutputs = 0
    let op = makeOp("AttrPartialShape", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrPartialShapeList(
    _ a: [TensorShape?]
) {
  let nOutputs = 0
    let op = makeOp("AttrPartialShapeList", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrShape(
    _ a: TensorShape?
) {
  let nOutputs = 0
    let op = makeOp("AttrShape", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrShapeList(
    _ a: [TensorShape?]
) {
  let nOutputs = 0
    let op = makeOp("AttrShapeList", nOutputs)
    op.updateAttribute("a", a)
    op.execute()
}

@inlinable @inline(__always)
public static func attrTypeDefault<T: TensorFlowScalar>(
    _ a: Tensor<T>
) {
  let nOutputs = 0
    let op = makeOp("AttrTypeDefault", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    op.execute()
}

/// Audio Microfrontend Op.
///
/// This Op converts a sequence of audio data into one or more
/// feature vectors containing filterbanks of the input. The
/// conversion process uses a lightweight library to perform:
///
/// 1. A slicing window function
/// 2. Short-time FFTs
/// 3. Filterbank calculations
/// 4. Noise reduction
/// 5. PCAN Auto Gain Control
/// 6. Logarithmic scaling
///
/// Arguments
///   audio: 1D Tensor, int16 audio data in temporal ordering.
///   sample_rate: Integer, the sample rate of the audio in Hz.
///   window_size: Integer, length of desired time frames in ms.
///   window_step: Integer, length of step size for the next frame in ms.
///   num_channels: Integer, the number of filterbank channels to use.
///   upper_band_limit: Float, the highest frequency included in the filterbanks.
///   lower_band_limit: Float, the lowest frequency included in the filterbanks.
///   smoothing_bits: Int, scale up signal by 2^(smoothing_bits) before reduction.
///   even_smoothing: Float, smoothing coefficient for even-numbered channels.
///   odd_smoothing: Float, smoothing coefficient for odd-numbered channels.
///   min_signal_remaining: Float, fraction of signal to preserve in smoothing.
///   enable_pcan: Bool, enable PCAN auto gain control.
///   pcan_strength: Float, gain normalization exponent.
///   pcan_offset: Float, positive value added in the normalization denominator.
///   gain_bits: Int, number of fractional bits in the gain.
///   enable_log: Bool, enable logarithmic scaling of filterbanks.
///   scale_shift: Integer, scale filterbanks by 2^(scale_shift).
///   left_context: Integer, number of preceding frames to attach to each frame.
///   right_context: Integer, number of preceding frames to attach to each frame.
///   frame_stride: Integer, M frames to skip over, where output[n] = frame[n*M].
///   zero_padding: Bool, if left/right context is out-of-bounds, attach frame of
///                 zeroes. Otherwise, frame[0] or frame[size-1] will be copied.
///   out_scale: Integer, divide all filterbanks by this number.
///   out_type: DType, type of the output Tensor, defaults to UINT16.
///
/// Returns
///   filterbanks: 2D Tensor, each row is a time frame, each column is a channel.
@inlinable @inline(__always)
public static func audioMicrofrontend<OutType: TensorFlowNumeric>(
    audio: Tensor<Int16>,
    sampleRate: Int64 = 16000,
    windowSize: Int64 = 25,
    windowStep: Int64 = 10,
    numChannels: Int64 = 32,
    upperBandLimit: Double = 7500,
    lowerBandLimit: Double = 125,
    smoothingBits: Int64 = 10,
    evenSmoothing: Double = 0.025,
    oddSmoothing: Double = 0.06,
    minSignalRemaining: Double = 0.05,
    enablePcan: Bool = false,
    pcanStrength: Double = 0.95,
    pcanOffset: Double = 80,
    gainBits: Int64 = 21,
    enableLog: Bool = true,
    scaleShift: Int64 = 6,
    leftContext: Int64 = 0,
    rightContext: Int64 = 0,
    frameStride: Int64 = 1,
    zeroPadding: Bool = false,
    outScale: Int64 = 1
) -> Tensor<OutType> {
  let nOutputs = Int(1)
    let op = makeOp("AudioMicrofrontend", nOutputs)
    op.updateAttribute("sample_rate", sampleRate)
    op.updateAttribute("window_size", windowSize)
    op.updateAttribute("window_step", windowStep)
    op.updateAttribute("num_channels", numChannels)
    op.updateAttribute("upper_band_limit", upperBandLimit)
    op.updateAttribute("lower_band_limit", lowerBandLimit)
    op.updateAttribute("smoothing_bits", smoothingBits)
    op.updateAttribute("even_smoothing", evenSmoothing)
    op.updateAttribute("odd_smoothing", oddSmoothing)
    op.updateAttribute("min_signal_remaining", minSignalRemaining)
    op.updateAttribute("enable_pcan", enablePcan)
    op.updateAttribute("pcan_strength", pcanStrength)
    op.updateAttribute("pcan_offset", pcanOffset)
    op.updateAttribute("gain_bits", gainBits)
    op.updateAttribute("enable_log", enableLog)
    op.updateAttribute("scale_shift", scaleShift)
    op.updateAttribute("left_context", leftContext)
    op.updateAttribute("right_context", rightContext)
    op.updateAttribute("frame_stride", frameStride)
    op.updateAttribute("zero_padding", zeroPadding)
    op.updateAttribute("out_scale", outScale)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(audio)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func audioSpectrogram(
    _ input: Tensor<Float>,
    windowSize: Int64,
    stride: Int64,
    magnitudeSquared: Bool = false
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("AudioSpectrogram", nOutputs)
    op.updateAttribute("window_size", windowSize)
    op.updateAttribute("stride", stride)
    op.updateAttribute("magnitude_squared", magnitudeSquared)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func audioSummary(
    tag: StringTensor,
    _ tensor: Tensor<Float>,
    sampleRate: Double,
    maxOutputs: Int64 = 3
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("AudioSummary", nOutputs)
    op.updateAttribute("sample_rate", sampleRate)
    op.updateAttribute("max_outputs", maxOutputs)
    op.addInput(tag)
    op.addInput(tensor)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func audioSummaryV2(
    tag: StringTensor,
    _ tensor: Tensor<Float>,
    sampleRate: Tensor<Float>,
    maxOutputs: Int64 = 3
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("AudioSummaryV2", nOutputs)
    op.updateAttribute("max_outputs", maxOutputs)
    op.addInput(tag)
    op.addInput(tensor)
    op.addInput(sampleRate)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func autoShardDataset(
    inputDataset: VariantHandle,
    numWorkers: Tensor<Int64>,
    index: Tensor<Int64>,
    autoShardPolicy: Int64 = 0,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("AutoShardDataset", nOutputs)
    op.updateAttribute("auto_shard_policy", autoShardPolicy)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(numWorkers)
    op.addInput(index)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func avgPool<T: FloatingPoint & TensorFlowScalar>(
    value: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("AvgPool", nOutputs)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(value)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func avgPool3D<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("AvgPool3D", nOutputs)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func avgPool3DGrad<T: FloatingPoint & TensorFlowScalar>(
    origInputShape: Tensor<Int32>,
    grad: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("AvgPool3DGrad", nOutputs)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(origInputShape)
    op.addInput(grad)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func avgPoolGrad<T: FloatingPoint & TensorFlowScalar>(
    origInputShape: Tensor<Int32>,
    grad: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("AvgPoolGrad", nOutputs)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(origInputShape)
    op.addInput(grad)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func b(
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("B", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batch<T: TensorArrayProtocol>(
    inTensors: T,
    numBatchThreads: Int64,
    maxBatchSize: Int64,
    maxEnqueuedBatches: Int64 = 10,
    batchTimeoutMicros: Int64,
    allowedBatchSizes: [Int32],
    gradTimeoutMicros: Int64,
    container: String,
    sharedName: String,
    batchingQueue: String
) -> (batchedTensors: T, batchIndex: Tensor<Int64>, id: Tensor<Int64>) {
  let nOutputs = Int(inTensors._typeList.count) + Int(1) + Int(1)
    let op = makeOp("Batch", nOutputs)
    op.updateAttribute("num_batch_threads", numBatchThreads)
    op.updateAttribute("max_batch_size", maxBatchSize)
    op.updateAttribute("max_enqueued_batches", maxEnqueuedBatches)
    op.updateAttribute("batch_timeout_micros", batchTimeoutMicros)
    op.updateAttribute("allowed_batch_sizes", allowedBatchSizes)
    op.updateAttribute("grad_timeout_micros", gradTimeoutMicros)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("batching_queue", batchingQueue)
    op.updateAttribute("T", inTensors._typeList)
    op.addInputList(inTensors)
    return op.execute(Int(inTensors._typeList.count), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func batchCholesky<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchCholesky", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchCholeskyGrad<T: FloatingPoint & TensorFlowScalar>(
    l: Tensor<T>,
    grad: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchCholeskyGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(l)
    op.addInput(grad)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchDataset(
    inputDataset: VariantHandle,
    batchSize: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("BatchDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(batchSize)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchDatasetV2(
    inputDataset: VariantHandle,
    batchSize: Tensor<Int64>,
    dropRemainder: Tensor<Bool>,
    parallelCopy: Bool = false,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("BatchDatasetV2", nOutputs)
    op.updateAttribute("parallel_copy", parallelCopy)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(batchSize)
    op.addInput(dropRemainder)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchFunction<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Tin: TensorArrayProtocol,
    Tcaptured: TensorArrayProtocol,
    Tout: TensorGroup
>(
    inTensors: Tin,
    capturedTensors: Tcaptured,
    f: (FIn) -> FOut,
    numBatchThreads: Int64,
    maxBatchSize: Int64,
    batchTimeoutMicros: Int64,
    maxEnqueuedBatches: Int64 = 10,
    allowedBatchSizes: [Int32],
    container: String,
    sharedName: String,
    batchingQueue: String
) -> Tout {
  let nOutputs = Int(Tout._typeList.count)
    let op = makeOp("BatchFunction", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("num_batch_threads", numBatchThreads)
    op.updateAttribute("max_batch_size", maxBatchSize)
    op.updateAttribute("batch_timeout_micros", batchTimeoutMicros)
    op.updateAttribute("max_enqueued_batches", maxEnqueuedBatches)
    op.updateAttribute("allowed_batch_sizes", allowedBatchSizes)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("batching_queue", batchingQueue)
    op.updateAttribute("Tin", inTensors._typeList)
    op.updateAttribute("Tcaptured", capturedTensors._typeList)
    op.updateAttribute("Tout", Tout._typeList)
    op.addInputList(inTensors)
    op.addInputList(capturedTensors)
    return op.execute(Int(Tout._typeList.count))
}

@inlinable @inline(__always)
public static func batchMatMul<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    adjX: Bool = false,
    adjY: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchMatMul", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("adj_x", adjX)
    op.updateAttribute("adj_y", adjY)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchMatMulV2<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    adjX: Bool = false,
    adjY: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchMatMulV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("adj_x", adjX)
    op.updateAttribute("adj_y", adjY)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchMatrixBandPart<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    numLower: Tensor<Int64>,
    numUpper: Tensor<Int64>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchMatrixBandPart", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    op.addInput(numLower)
    op.addInput(numUpper)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchMatrixDeterminant<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchMatrixDeterminant", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchMatrixDiag<T: TensorFlowScalar>(
    diagonal: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchMatrixDiag", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(diagonal)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchMatrixDiagPart<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchMatrixDiagPart", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchMatrixInverse<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    adjoint: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchMatrixInverse", nOutputs)
    op.updateAttribute("adjoint", adjoint)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchMatrixSetDiag<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    diagonal: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchMatrixSetDiag", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    op.addInput(diagonal)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchMatrixSolve<T: FloatingPoint & TensorFlowScalar>(
    matrix: Tensor<T>,
    rhs: Tensor<T>,
    adjoint: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchMatrixSolve", nOutputs)
    op.updateAttribute("adjoint", adjoint)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(matrix)
    op.addInput(rhs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchMatrixSolveLs<T: FloatingPoint & TensorFlowScalar>(
    matrix: Tensor<T>,
    rhs: Tensor<T>,
    l2Regularizer: Tensor<Double>,
    fast: Bool = true
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchMatrixSolveLs", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("fast", fast)
    op.addInput(matrix)
    op.addInput(rhs)
    op.addInput(l2Regularizer)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchMatrixTriangularSolve<T: FloatingPoint & TensorFlowScalar>(
    matrix: Tensor<T>,
    rhs: Tensor<T>,
    lower: Bool = true,
    adjoint: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchMatrixTriangularSolve", nOutputs)
    op.updateAttribute("lower", lower)
    op.updateAttribute("adjoint", adjoint)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(matrix)
    op.addInput(rhs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchNormWithGlobalNormalization<T: TensorFlowNumeric>(
    t: Tensor<T>,
    m: Tensor<T>,
    v: Tensor<T>,
    beta: Tensor<T>,
    gamma: Tensor<T>,
    varianceEpsilon: Double,
    scaleAfterNormalization: Bool
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchNormWithGlobalNormalization", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("variance_epsilon", varianceEpsilon)
    op.updateAttribute("scale_after_normalization", scaleAfterNormalization)
    op.addInput(t)
    op.addInput(m)
    op.addInput(v)
    op.addInput(beta)
    op.addInput(gamma)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchNormWithGlobalNormalizationGrad<T: TensorFlowNumeric>(
    t: Tensor<T>,
    m: Tensor<T>,
    v: Tensor<T>,
    gamma: Tensor<T>,
    backprop: Tensor<T>,
    varianceEpsilon: Double,
    scaleAfterNormalization: Bool
) -> (dx: Tensor<T>, dm: Tensor<T>, dv: Tensor<T>, db: Tensor<T>, dg: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("BatchNormWithGlobalNormalizationGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("variance_epsilon", varianceEpsilon)
    op.updateAttribute("scale_after_normalization", scaleAfterNormalization)
    op.addInput(t)
    op.addInput(m)
    op.addInput(v)
    op.addInput(gamma)
    op.addInput(backprop)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func batchSelfAdjointEig<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchSelfAdjointEig", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchSelfAdjointEigV2<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    computeV: Bool = true
) -> (e: Tensor<T>, v: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("BatchSelfAdjointEigV2", nOutputs)
    op.updateAttribute("compute_v", computeV)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func batchSvd<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    computeUv: Bool = true,
    fullMatrices: Bool = false
) -> (s: Tensor<T>, u: Tensor<T>, v: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("BatchSvd", nOutputs)
    op.updateAttribute("compute_uv", computeUv)
    op.updateAttribute("full_matrices", fullMatrices)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func batchToSpace<
    T: TensorFlowScalar,
    Tidx: TensorFlowIndex
>(
    _ input: Tensor<T>,
    crops: Tensor<Tidx>,
    blockSize: Int64
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchToSpace", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("block_size", blockSize)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(input)
    op.addInput(crops)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func batchToSpaceND<
    T: TensorFlowScalar,
    TblockShape: TensorFlowIndex,
    Tcrops: TensorFlowIndex
>(
    _ input: Tensor<T>,
    blockShape: Tensor<TblockShape>,
    crops: Tensor<Tcrops>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BatchToSpaceND", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tblock_shape", TblockShape.tensorFlowDataType)
    op.updateAttribute("Tcrops", Tcrops.tensorFlowDataType)
    op.addInput(input)
    op.addInput(blockShape)
    op.addInput(crops)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func besselI0e<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BesselI0e", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func besselI1e<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BesselI1e", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func betainc<T: FloatingPoint & TensorFlowScalar>(
    _ a: Tensor<T>,
    _ b: Tensor<T>,
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Betainc", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    op.addInput(b)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func biasAdd<T: TensorFlowNumeric>(
    value: Tensor<T>,
    bias: Tensor<T>,
    dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BiasAdd", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("data_format", dataFormat.cName)
    op.addInput(value)
    op.addInput(bias)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func biasAddGrad<T: TensorFlowNumeric>(
    outBackprop: Tensor<T>,
    dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BiasAddGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("data_format", dataFormat.cName)
    op.addInput(outBackprop)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func biasAddV1<T: TensorFlowNumeric>(
    value: Tensor<T>,
    bias: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BiasAddV1", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(value)
    op.addInput(bias)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func binary<T: TensorFlowScalar>(
    _ a: Tensor<T>,
    _ b: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Binary", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    op.addInput(b)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func bincount<T: TensorFlowNumeric>(
    arr: Tensor<Int32>,
    size: Tensor<Int32>,
    weights: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Bincount", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(arr)
    op.addInput(size)
    op.addInput(weights)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func bitcast<
    T: TensorFlowNumeric,
    Type: TensorFlowNumeric
>(
    _ input: Tensor<T>
) -> Tensor<Type> {
  let nOutputs = Int(1)
    let op = makeOp("Bitcast", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("type", Type.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func bitwiseAnd<T: TensorFlowInteger>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BitwiseAnd", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func bitwiseOr<T: TensorFlowInteger>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BitwiseOr", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func bitwiseXor<T: TensorFlowInteger>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BitwiseXor", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func blockLSTM<T: FloatingPoint & TensorFlowScalar>(
    seqLenMax: Tensor<Int64>,
    _ x: Tensor<T>,
    csPrev: Tensor<T>,
    hPrev: Tensor<T>,
    w: Tensor<T>,
    wci: Tensor<T>,
    wcf: Tensor<T>,
    wco: Tensor<T>,
    _ b: Tensor<T>,
    forgetBias: Double = 1,
    cellClip: Double = 3,
    usePeephole: Bool = false
) -> (i: Tensor<T>, cs: Tensor<T>, f: Tensor<T>, o: Tensor<T>, ci: Tensor<T>, co: Tensor<T>, h: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("BlockLSTM", nOutputs)
    op.updateAttribute("forget_bias", forgetBias)
    op.updateAttribute("cell_clip", cellClip)
    op.updateAttribute("use_peephole", usePeephole)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(seqLenMax)
    op.addInput(x)
    op.addInput(csPrev)
    op.addInput(hPrev)
    op.addInput(w)
    op.addInput(wci)
    op.addInput(wcf)
    op.addInput(wco)
    op.addInput(b)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func blockLSTMGrad<T: FloatingPoint & TensorFlowScalar>(
    seqLenMax: Tensor<Int64>,
    _ x: Tensor<T>,
    csPrev: Tensor<T>,
    hPrev: Tensor<T>,
    w: Tensor<T>,
    wci: Tensor<T>,
    wcf: Tensor<T>,
    wco: Tensor<T>,
    _ b: Tensor<T>,
    i: Tensor<T>,
    cs: Tensor<T>,
    f: Tensor<T>,
    o: Tensor<T>,
    ci: Tensor<T>,
    co: Tensor<T>,
    h: Tensor<T>,
    csGrad: Tensor<T>,
    hGrad: Tensor<T>,
    usePeephole: Bool
) -> (xGrad: Tensor<T>, csPrevGrad: Tensor<T>, hPrevGrad: Tensor<T>, wGrad: Tensor<T>, wciGrad: Tensor<T>, wcfGrad: Tensor<T>, wcoGrad: Tensor<T>, bGrad: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("BlockLSTMGrad", nOutputs)
    op.updateAttribute("use_peephole", usePeephole)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(seqLenMax)
    op.addInput(x)
    op.addInput(csPrev)
    op.addInput(hPrev)
    op.addInput(w)
    op.addInput(wci)
    op.addInput(wcf)
    op.addInput(wco)
    op.addInput(b)
    op.addInput(i)
    op.addInput(cs)
    op.addInput(f)
    op.addInput(o)
    op.addInput(ci)
    op.addInput(co)
    op.addInput(h)
    op.addInput(csGrad)
    op.addInput(hGrad)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func blockLSTMGradV2<T: FloatingPoint & TensorFlowScalar>(
    seqLenMax: Tensor<Int64>,
    _ x: Tensor<T>,
    csPrev: Tensor<T>,
    hPrev: Tensor<T>,
    w: Tensor<T>,
    wci: Tensor<T>,
    wcf: Tensor<T>,
    wco: Tensor<T>,
    _ b: Tensor<T>,
    i: Tensor<T>,
    cs: Tensor<T>,
    f: Tensor<T>,
    o: Tensor<T>,
    ci: Tensor<T>,
    co: Tensor<T>,
    h: Tensor<T>,
    csGrad: Tensor<T>,
    hGrad: Tensor<T>,
    usePeephole: Bool
) -> (xGrad: Tensor<T>, csPrevGrad: Tensor<T>, hPrevGrad: Tensor<T>, wGrad: Tensor<T>, wciGrad: Tensor<T>, wcfGrad: Tensor<T>, wcoGrad: Tensor<T>, bGrad: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("BlockLSTMGradV2", nOutputs)
    op.updateAttribute("use_peephole", usePeephole)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(seqLenMax)
    op.addInput(x)
    op.addInput(csPrev)
    op.addInput(hPrev)
    op.addInput(w)
    op.addInput(wci)
    op.addInput(wcf)
    op.addInput(wco)
    op.addInput(b)
    op.addInput(i)
    op.addInput(cs)
    op.addInput(f)
    op.addInput(o)
    op.addInput(ci)
    op.addInput(co)
    op.addInput(h)
    op.addInput(csGrad)
    op.addInput(hGrad)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func blockLSTMV2<T: FloatingPoint & TensorFlowScalar>(
    seqLenMax: Tensor<Int64>,
    _ x: Tensor<T>,
    csPrev: Tensor<T>,
    hPrev: Tensor<T>,
    w: Tensor<T>,
    wci: Tensor<T>,
    wcf: Tensor<T>,
    wco: Tensor<T>,
    _ b: Tensor<T>,
    cellClip: Double = 0,
    usePeephole: Bool = false
) -> (i: Tensor<T>, cs: Tensor<T>, f: Tensor<T>, o: Tensor<T>, ci: Tensor<T>, co: Tensor<T>, h: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("BlockLSTMV2", nOutputs)
    op.updateAttribute("cell_clip", cellClip)
    op.updateAttribute("use_peephole", usePeephole)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(seqLenMax)
    op.addInput(x)
    op.addInput(csPrev)
    op.addInput(hPrev)
    op.addInput(w)
    op.addInput(wci)
    op.addInput(wcf)
    op.addInput(wco)
    op.addInput(b)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesAggregateStats(
    nodeIds: Tensor<Int32>,
    gradients: Tensor<Float>,
    hessians: Tensor<Float>,
    feature: Tensor<Int32>,
    maxSplits: Int64,
    numBuckets: Int64
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("BoostedTreesAggregateStats", nOutputs)
    op.updateAttribute("max_splits", maxSplits)
    op.updateAttribute("num_buckets", numBuckets)
    op.addInput(nodeIds)
    op.addInput(gradients)
    op.addInput(hessians)
    op.addInput(feature)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesBucketize(
    floatValues: [Tensor<Float>],
    bucketBoundaries: [Tensor<Float>]
) -> [Tensor<Int32>] {
  let nOutputs = Int(floatValues.count)
    let op = makeOp("BoostedTreesBucketize", nOutputs)
    op.updateAttribute("num_features", floatValues.count)
    op.addInputList(floatValues)
    op.addInputList(bucketBoundaries)
    return op.execute(Int(floatValues.count))
}

@inlinable @inline(__always)
public static func boostedTreesCalculateBestFeatureSplit(
    nodeIdRange: Tensor<Int32>,
    statsSummary: Tensor<Float>,
    l1: Tensor<Float>,
    l2: Tensor<Float>,
    treeComplexity: Tensor<Float>,
    minNodeWeight: Tensor<Float>,
    logitsDimension: Int64,
    splitType: SplitType = .inequality
) -> (nodeIds: Tensor<Int32>, gains: Tensor<Float>, featureDimensions: Tensor<Int32>, thresholds: Tensor<Int32>, leftNodeContribs: Tensor<Float>, rightNodeContribs: Tensor<Float>, splitWithDefaultDirections: StringTensor) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("BoostedTreesCalculateBestFeatureSplit", nOutputs)
    op.updateAttribute("logits_dimension", logitsDimension)
    op.updateAttribute("split_type", splitType.cName)
    op.addInput(nodeIdRange)
    op.addInput(statsSummary)
    op.addInput(l1)
    op.addInput(l2)
    op.addInput(treeComplexity)
    op.addInput(minNodeWeight)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesCalculateBestGainsPerFeature(
    nodeIdRange: Tensor<Int32>,
    statsSummaryList: [Tensor<Float>],
    l1: Tensor<Float>,
    l2: Tensor<Float>,
    treeComplexity: Tensor<Float>,
    minNodeWeight: Tensor<Float>,
    maxSplits: Int64
) -> (nodeIdsList: [Tensor<Int32>], gainsList: [Tensor<Float>], thresholdsList: [Tensor<Int32>], leftNodeContribsList: [Tensor<Float>], rightNodeContribsList: [Tensor<Float>]) {
  let nOutputs = Int(statsSummaryList.count) + Int(statsSummaryList.count) + Int(statsSummaryList.count) + Int(statsSummaryList.count) + Int(statsSummaryList.count)
    let op = makeOp("BoostedTreesCalculateBestGainsPerFeature", nOutputs)
    op.updateAttribute("max_splits", maxSplits)
    op.updateAttribute("num_features", statsSummaryList.count)
    op.addInput(nodeIdRange)
    op.addInputList(statsSummaryList)
    op.addInput(l1)
    op.addInput(l2)
    op.addInput(treeComplexity)
    op.addInput(minNodeWeight)
    return op.execute(Int(statsSummaryList.count), Int(statsSummaryList.count), Int(statsSummaryList.count), Int(statsSummaryList.count), Int(statsSummaryList.count))
}

@inlinable @inline(__always)
public static func boostedTreesCenterBias(
    treeEnsembleHandle: ResourceHandle,
    meanGradients: Tensor<Float>,
    meanHessians: Tensor<Float>,
    l1: Tensor<Float>,
    l2: Tensor<Float>
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("BoostedTreesCenterBias", nOutputs)
    op.addInput(treeEnsembleHandle)
    op.addInput(meanGradients)
    op.addInput(meanHessians)
    op.addInput(l1)
    op.addInput(l2)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesCreateEnsemble(
    treeEnsembleHandle: ResourceHandle,
    stampToken: Tensor<Int64>,
    treeEnsembleSerialized: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("BoostedTreesCreateEnsemble", nOutputs)
    op.addInput(treeEnsembleHandle)
    op.addInput(stampToken)
    op.addInput(treeEnsembleSerialized)
    op.execute()
}

@inlinable @inline(__always)
public static func boostedTreesCreateQuantileStreamResource(
    quantileStreamResourceHandle: ResourceHandle,
    epsilon: Tensor<Float>,
    numStreams: Tensor<Int64>,
    maxElements: Int64 = 1099511627776
) {
  let nOutputs = 0
    let op = makeOp("BoostedTreesCreateQuantileStreamResource", nOutputs)
    op.updateAttribute("max_elements", maxElements)
    op.addInput(quantileStreamResourceHandle)
    op.addInput(epsilon)
    op.addInput(numStreams)
    op.execute()
}

@inlinable @inline(__always)
public static func boostedTreesDeserializeEnsemble(
    treeEnsembleHandle: ResourceHandle,
    stampToken: Tensor<Int64>,
    treeEnsembleSerialized: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("BoostedTreesDeserializeEnsemble", nOutputs)
    op.addInput(treeEnsembleHandle)
    op.addInput(stampToken)
    op.addInput(treeEnsembleSerialized)
    op.execute()
}

@inlinable @inline(__always)
public static func boostedTreesEnsembleResourceHandleOp(
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("BoostedTreesEnsembleResourceHandleOp", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesExampleDebugOutputs(
    treeEnsembleHandle: ResourceHandle,
    bucketizedFeatures: [Tensor<Int32>],
    logitsDimension: Int64
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("BoostedTreesExampleDebugOutputs", nOutputs)
    op.updateAttribute("num_bucketized_features", bucketizedFeatures.count)
    op.updateAttribute("logits_dimension", logitsDimension)
    op.addInput(treeEnsembleHandle)
    op.addInputList(bucketizedFeatures)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesFlushQuantileSummaries(
    quantileStreamResourceHandle: ResourceHandle,
    numFeatures: Int64
) -> [Tensor<Float>] {
  let nOutputs = Int(numFeatures)
    let op = makeOp("BoostedTreesFlushQuantileSummaries", nOutputs)
    op.updateAttribute("num_features", numFeatures)
    op.addInput(quantileStreamResourceHandle)
    return op.execute(Int(numFeatures))
}

@inlinable @inline(__always)
public static func boostedTreesGetEnsembleStates(
    treeEnsembleHandle: ResourceHandle
) -> (stampToken: Tensor<Int64>, numTrees: Tensor<Int32>, numFinalizedTrees: Tensor<Int32>, numAttemptedLayers: Tensor<Int32>, lastLayerNodesRange: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("BoostedTreesGetEnsembleStates", nOutputs)
    op.addInput(treeEnsembleHandle)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesMakeQuantileSummaries(
    floatValues: [Tensor<Float>],
    exampleWeights: Tensor<Float>,
    epsilon: Tensor<Float>
) -> [Tensor<Float>] {
  let nOutputs = Int(floatValues.count)
    let op = makeOp("BoostedTreesMakeQuantileSummaries", nOutputs)
    op.updateAttribute("num_features", floatValues.count)
    op.addInputList(floatValues)
    op.addInput(exampleWeights)
    op.addInput(epsilon)
    return op.execute(Int(floatValues.count))
}

@inlinable @inline(__always)
public static func boostedTreesMakeStatsSummary(
    nodeIds: Tensor<Int32>,
    gradients: Tensor<Float>,
    hessians: Tensor<Float>,
    bucketizedFeaturesList: [Tensor<Int32>],
    maxSplits: Int64,
    numBuckets: Int64
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("BoostedTreesMakeStatsSummary", nOutputs)
    op.updateAttribute("max_splits", maxSplits)
    op.updateAttribute("num_buckets", numBuckets)
    op.updateAttribute("num_features", bucketizedFeaturesList.count)
    op.addInput(nodeIds)
    op.addInput(gradients)
    op.addInput(hessians)
    op.addInputList(bucketizedFeaturesList)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesPredict(
    treeEnsembleHandle: ResourceHandle,
    bucketizedFeatures: [Tensor<Int32>],
    logitsDimension: Int64
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("BoostedTreesPredict", nOutputs)
    op.updateAttribute("num_bucketized_features", bucketizedFeatures.count)
    op.updateAttribute("logits_dimension", logitsDimension)
    op.addInput(treeEnsembleHandle)
    op.addInputList(bucketizedFeatures)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesQuantileStreamResourceAddSummaries(
    quantileStreamResourceHandle: ResourceHandle,
    summaries: [Tensor<Float>]
) {
  let nOutputs = 0
    let op = makeOp("BoostedTreesQuantileStreamResourceAddSummaries", nOutputs)
    op.updateAttribute("num_features", summaries.count)
    op.addInput(quantileStreamResourceHandle)
    op.addInputList(summaries)
    op.execute()
}

@inlinable @inline(__always)
public static func boostedTreesQuantileStreamResourceDeserialize(
    quantileStreamResourceHandle: ResourceHandle,
    bucketBoundaries: [Tensor<Float>]
) {
  let nOutputs = 0
    let op = makeOp("BoostedTreesQuantileStreamResourceDeserialize", nOutputs)
    op.updateAttribute("num_streams", bucketBoundaries.count)
    op.addInput(quantileStreamResourceHandle)
    op.addInputList(bucketBoundaries)
    op.execute()
}

@inlinable @inline(__always)
public static func boostedTreesQuantileStreamResourceFlush(
    quantileStreamResourceHandle: ResourceHandle,
    numBuckets: Tensor<Int64>,
    generateQuantiles: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("BoostedTreesQuantileStreamResourceFlush", nOutputs)
    op.updateAttribute("generate_quantiles", generateQuantiles)
    op.addInput(quantileStreamResourceHandle)
    op.addInput(numBuckets)
    op.execute()
}

@inlinable @inline(__always)
public static func boostedTreesQuantileStreamResourceGetBucketBoundaries(
    quantileStreamResourceHandle: ResourceHandle,
    numFeatures: Int64
) -> [Tensor<Float>] {
  let nOutputs = Int(numFeatures)
    let op = makeOp("BoostedTreesQuantileStreamResourceGetBucketBoundaries", nOutputs)
    op.updateAttribute("num_features", numFeatures)
    op.addInput(quantileStreamResourceHandle)
    return op.execute(Int(numFeatures))
}

@inlinable @inline(__always)
public static func boostedTreesQuantileStreamResourceHandleOp(
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("BoostedTreesQuantileStreamResourceHandleOp", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesSerializeEnsemble(
    treeEnsembleHandle: ResourceHandle
) -> (stampToken: Tensor<Int64>, treeEnsembleSerialized: StringTensor) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("BoostedTreesSerializeEnsemble", nOutputs)
    op.addInput(treeEnsembleHandle)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesSparseAggregateStats(
    nodeIds: Tensor<Int32>,
    gradients: Tensor<Float>,
    hessians: Tensor<Float>,
    featureIndices: Tensor<Int32>,
    featureValues: Tensor<Int32>,
    featureShape: Tensor<Int32>,
    maxSplits: Int64,
    numBuckets: Int64
) -> (statsSummaryIndices: Tensor<Int32>, statsSummaryValues: Tensor<Float>, statsSummaryShape: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("BoostedTreesSparseAggregateStats", nOutputs)
    op.updateAttribute("max_splits", maxSplits)
    op.updateAttribute("num_buckets", numBuckets)
    op.addInput(nodeIds)
    op.addInput(gradients)
    op.addInput(hessians)
    op.addInput(featureIndices)
    op.addInput(featureValues)
    op.addInput(featureShape)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesSparseCalculateBestFeatureSplit(
    nodeIdRange: Tensor<Int32>,
    statsSummaryIndices: Tensor<Int32>,
    statsSummaryValues: Tensor<Float>,
    statsSummaryShape: Tensor<Int32>,
    l1: Tensor<Float>,
    l2: Tensor<Float>,
    treeComplexity: Tensor<Float>,
    minNodeWeight: Tensor<Float>,
    logitsDimension: Int64,
    splitType: SplitType2 = .inequality
) -> (nodeIds: Tensor<Int32>, gains: Tensor<Float>, featureDimensions: Tensor<Int32>, thresholds: Tensor<Int32>, leftNodeContribs: Tensor<Float>, rightNodeContribs: Tensor<Float>, splitWithDefaultDirections: StringTensor) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("BoostedTreesSparseCalculateBestFeatureSplit", nOutputs)
    op.updateAttribute("logits_dimension", logitsDimension)
    op.updateAttribute("split_type", splitType.cName)
    op.addInput(nodeIdRange)
    op.addInput(statsSummaryIndices)
    op.addInput(statsSummaryValues)
    op.addInput(statsSummaryShape)
    op.addInput(l1)
    op.addInput(l2)
    op.addInput(treeComplexity)
    op.addInput(minNodeWeight)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesTrainingPredict(
    treeEnsembleHandle: ResourceHandle,
    cachedTreeIds: Tensor<Int32>,
    cachedNodeIds: Tensor<Int32>,
    bucketizedFeatures: [Tensor<Int32>],
    logitsDimension: Int64
) -> (partialLogits: Tensor<Float>, treeIds: Tensor<Int32>, nodeIds: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("BoostedTreesTrainingPredict", nOutputs)
    op.updateAttribute("num_bucketized_features", bucketizedFeatures.count)
    op.updateAttribute("logits_dimension", logitsDimension)
    op.addInput(treeEnsembleHandle)
    op.addInput(cachedTreeIds)
    op.addInput(cachedNodeIds)
    op.addInputList(bucketizedFeatures)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func boostedTreesUpdateEnsemble(
    treeEnsembleHandle: ResourceHandle,
    featureIds: Tensor<Int32>,
    nodeIds: [Tensor<Int32>],
    gains: [Tensor<Float>],
    thresholds: [Tensor<Int32>],
    leftNodeContribs: [Tensor<Float>],
    rightNodeContribs: [Tensor<Float>],
    maxDepth: Tensor<Int32>,
    learningRate: Tensor<Float>,
    pruningMode: Int64
) {
  let nOutputs = 0
    let op = makeOp("BoostedTreesUpdateEnsemble", nOutputs)
    op.updateAttribute("pruning_mode", pruningMode)
    op.updateAttribute("num_features", nodeIds.count)
    op.addInput(treeEnsembleHandle)
    op.addInput(featureIds)
    op.addInputList(nodeIds)
    op.addInputList(gains)
    op.addInputList(thresholds)
    op.addInputList(leftNodeContribs)
    op.addInputList(rightNodeContribs)
    op.addInput(maxDepth)
    op.addInput(learningRate)
    op.execute()
}

@inlinable @inline(__always)
public static func boostedTreesUpdateEnsembleV2(
    treeEnsembleHandle: ResourceHandle,
    featureIds: Tensor<Int32>,
    dimensionIds: [Tensor<Int32>],
    nodeIds: [Tensor<Int32>],
    gains: [Tensor<Float>],
    thresholds: [Tensor<Int32>],
    leftNodeContribs: [Tensor<Float>],
    rightNodeContribs: [Tensor<Float>],
    splitTypes: [StringTensor],
    maxDepth: Tensor<Int32>,
    learningRate: Tensor<Float>,
    pruningMode: Tensor<Int32>,
    logitsDimension: Int64 = 1
) {
  let nOutputs = 0
    let op = makeOp("BoostedTreesUpdateEnsembleV2", nOutputs)
    op.updateAttribute("num_features", dimensionIds.count)
    op.updateAttribute("logits_dimension", logitsDimension)
    op.addInput(treeEnsembleHandle)
    op.addInput(featureIds)
    op.addInputList(dimensionIds)
    op.addInputList(nodeIds)
    op.addInputList(gains)
    op.addInputList(thresholds)
    op.addInputList(leftNodeContribs)
    op.addInputList(rightNodeContribs)
    op.addInputList(splitTypes)
    op.addInput(maxDepth)
    op.addInput(learningRate)
    op.addInput(pruningMode)
    op.execute()
}

@inlinable @inline(__always)
public static func broadcastArgs<T: TensorFlowIndex>(
    s0: Tensor<T>,
    s1: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BroadcastArgs", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(s0)
    op.addInput(s1)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func broadcastGradientArgs<T: TensorFlowIndex>(
    s0: Tensor<T>,
    s1: Tensor<T>
) -> (r0: Tensor<T>, r1: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("BroadcastGradientArgs", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(s0)
    op.addInput(s1)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func broadcastTo<
    T: TensorFlowScalar,
    Tidx: TensorFlowIndex
>(
    _ input: Tensor<T>,
    shape: Tensor<Tidx>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("BroadcastTo", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(input)
    op.addInput(shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func bucketize<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    boundaries: [Double]
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("Bucketize", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("boundaries", boundaries)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func bytesProducedStatsDataset(
    inputDataset: VariantHandle,
    tag: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("BytesProducedStatsDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(tag)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cSRSparseMatrixComponents<Type: FloatingPoint & TensorFlowScalar>(
    csrSparseMatrix: VariantHandle,
    index: Tensor<Int32>
) -> (rowPtrs: Tensor<Int32>, colInds: Tensor<Int32>, values: Tensor<Type>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("CSRSparseMatrixComponents", nOutputs)
    op.updateAttribute("type", Type.tensorFlowDataType)
    op.addInput(csrSparseMatrix)
    op.addInput(index)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func cSRSparseMatrixToDense<Type: FloatingPoint & TensorFlowScalar>(
    sparseInput: VariantHandle
) -> Tensor<Type> {
  let nOutputs = Int(1)
    let op = makeOp("CSRSparseMatrixToDense", nOutputs)
    op.updateAttribute("type", Type.tensorFlowDataType)
    op.addInput(sparseInput)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cSRSparseMatrixToSparseTensor<Type: FloatingPoint & TensorFlowScalar>(
    sparseMatrix: VariantHandle
) -> (indices: Tensor<Int64>, values: Tensor<Type>, denseShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("CSRSparseMatrixToSparseTensor", nOutputs)
    op.updateAttribute("type", Type.tensorFlowDataType)
    op.addInput(sparseMatrix)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func cSVDataset<OutputTypes: TensorArrayProtocol>(
    filenames: StringTensor,
    compressionType: StringTensor,
    bufferSize: Tensor<Int64>,
    header: Tensor<Bool>,
    fieldDelim: StringTensor,
    useQuoteDelim: Tensor<Bool>,
    naValue: StringTensor,
    selectCols: Tensor<Int64>,
    recordDefaults: OutputTypes,
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("CSVDataset", nOutputs)
    op.updateAttribute("output_types", recordDefaults._typeList)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(filenames)
    op.addInput(compressionType)
    op.addInput(bufferSize)
    op.addInput(header)
    op.addInput(fieldDelim)
    op.addInput(useQuoteDelim)
    op.addInput(naValue)
    op.addInput(selectCols)
    op.addInputList(recordDefaults)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cTCBeamSearchDecoder<T: FloatingPoint & TensorFlowScalar>(
    inputs: Tensor<T>,
    sequenceLength: Tensor<Int32>,
    beamWidth: Int64,
    topPaths: Int64,
    mergeRepeated: Bool = true
) -> (decodedIndices: [Tensor<Int64>], decodedValues: [Tensor<Int64>], decodedShape: [Tensor<Int64>], logProbability: Tensor<T>) {
  let nOutputs = Int(topPaths) + Int(topPaths) + Int(topPaths) + Int(1)
    let op = makeOp("CTCBeamSearchDecoder", nOutputs)
    op.updateAttribute("beam_width", beamWidth)
    op.updateAttribute("top_paths", topPaths)
    op.updateAttribute("merge_repeated", mergeRepeated)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(inputs)
    op.addInput(sequenceLength)
    return op.execute(Int(topPaths), Int(topPaths), Int(topPaths), Int(1))
}

@inlinable @inline(__always)
public static func cTCGreedyDecoder<T: FloatingPoint & TensorFlowScalar>(
    inputs: Tensor<T>,
    sequenceLength: Tensor<Int32>,
    mergeRepeated: Bool = false
) -> (decodedIndices: Tensor<Int64>, decodedValues: Tensor<Int64>, decodedShape: Tensor<Int64>, logProbability: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("CTCGreedyDecoder", nOutputs)
    op.updateAttribute("merge_repeated", mergeRepeated)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(inputs)
    op.addInput(sequenceLength)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func cTCLoss<T: FloatingPoint & TensorFlowScalar>(
    inputs: Tensor<T>,
    labelsIndices: Tensor<Int64>,
    labelsValues: Tensor<Int32>,
    sequenceLength: Tensor<Int32>,
    preprocessCollapseRepeated: Bool = false,
    ctcMergeRepeated: Bool = true,
    ignoreLongerOutputsThanInputs: Bool = false
) -> (loss: Tensor<T>, gradient: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("CTCLoss", nOutputs)
    op.updateAttribute("preprocess_collapse_repeated", preprocessCollapseRepeated)
    op.updateAttribute("ctc_merge_repeated", ctcMergeRepeated)
    op.updateAttribute("ignore_longer_outputs_than_inputs", ignoreLongerOutputsThanInputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(inputs)
    op.addInput(labelsIndices)
    op.addInput(labelsValues)
    op.addInput(sequenceLength)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func cacheDataset(
    inputDataset: VariantHandle,
    filename: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("CacheDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(filename)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cacheDatasetV2(
    inputDataset: VariantHandle,
    filename: StringTensor,
    cache: ResourceHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("CacheDatasetV2", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(filename)
    op.addInput(cache)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cast<
    Srct: TensorFlowScalar,
    Dstt: TensorFlowScalar
>(
    _ x: Tensor<Srct>,
    truncate: Bool = false
) -> Tensor<Dstt> {
  let nOutputs = Int(1)
    let op = makeOp("Cast", nOutputs)
    op.updateAttribute("SrcT", Srct.tensorFlowDataType)
    op.updateAttribute("DstT", Dstt.tensorFlowDataType)
    op.updateAttribute("Truncate", truncate)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func ceil<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Ceil", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func checkNumerics<T: FloatingPoint & TensorFlowScalar>(
    _ tensor: Tensor<T>,
    message: String
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CheckNumerics", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("message", message)
    op.addInput(tensor)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cholesky<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Cholesky", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func choleskyGrad<T: FloatingPoint & TensorFlowScalar>(
    l: Tensor<T>,
    grad: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CholeskyGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(l)
    op.addInput(grad)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func chooseFastestDataset(
    inputDatasets: [VariantHandle],
    numExperiments: Int64,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ChooseFastestDataset", nOutputs)
    op.updateAttribute("N", inputDatasets.count)
    op.updateAttribute("num_experiments", numExperiments)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInputList(inputDatasets)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func clipByValue<T: TensorFlowNumeric>(
    t: Tensor<T>,
    clipValueMin: Tensor<T>,
    clipValueMax: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ClipByValue", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(t)
    op.addInput(clipValueMin)
    op.addInput(clipValueMax)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func closeSummaryWriter(
    writer: ResourceHandle
) {
  let nOutputs = 0
    let op = makeOp("CloseSummaryWriter", nOutputs)
    op.addInput(writer)
    op.execute()
}

@inlinable @inline(__always)
public static func collectiveBcastRecv<T: TensorFlowNumeric>(
    groupSize: Int64,
    groupKey: Int64,
    instanceKey: Int64,
    shape: TensorShape?,
    communicationHint: String = "auto"
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CollectiveBcastRecv", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("group_size", groupSize)
    op.updateAttribute("group_key", groupKey)
    op.updateAttribute("instance_key", instanceKey)
    op.updateAttribute("shape", shape)
    op.updateAttribute("communication_hint", communicationHint)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func collectiveBcastSend<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    groupSize: Int64,
    groupKey: Int64,
    instanceKey: Int64,
    shape: TensorShape?,
    communicationHint: String = "auto"
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CollectiveBcastSend", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("group_size", groupSize)
    op.updateAttribute("group_key", groupKey)
    op.updateAttribute("instance_key", instanceKey)
    op.updateAttribute("shape", shape)
    op.updateAttribute("communication_hint", communicationHint)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func collectiveGather<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    groupSize: Int64,
    groupKey: Int64,
    instanceKey: Int64,
    shape: TensorShape?,
    communicationHint: String = "auto"
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CollectiveGather", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("group_size", groupSize)
    op.updateAttribute("group_key", groupKey)
    op.updateAttribute("instance_key", instanceKey)
    op.updateAttribute("shape", shape)
    op.updateAttribute("communication_hint", communicationHint)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func collectivePermute<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    sourceTargetPairs: Tensor<Int32>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CollectivePermute", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    op.addInput(sourceTargetPairs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func collectiveReduce<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    groupSize: Int64,
    groupKey: Int64,
    instanceKey: Int64,
    mergeOp: MergeOp,
    finalOp: FinalOp,
    subdivOffsets: [Int32],
    waitFor: [Int32],
    communicationHint: String = "auto"
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CollectiveReduce", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("group_size", groupSize)
    op.updateAttribute("group_key", groupKey)
    op.updateAttribute("instance_key", instanceKey)
    op.updateAttribute("merge_op", mergeOp.cName)
    op.updateAttribute("final_op", finalOp.cName)
    op.updateAttribute("subdiv_offsets", subdivOffsets)
    op.updateAttribute("wait_for", waitFor)
    op.updateAttribute("communication_hint", communicationHint)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func combinedNonMaxSuppression(
    boxes: Tensor<Float>,
    scores: Tensor<Float>,
    maxOutputSizePerClass: Tensor<Int32>,
    maxTotalSize: Tensor<Int32>,
    iouThreshold: Tensor<Float>,
    scoreThreshold: Tensor<Float>,
    padPerClass: Bool = false,
    clipBoxes: Bool = true
) -> (nmsedBoxes: Tensor<Float>, nmsedScores: Tensor<Float>, nmsedClasses: Tensor<Float>, validDetections: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("CombinedNonMaxSuppression", nOutputs)
    op.updateAttribute("pad_per_class", padPerClass)
    op.updateAttribute("clip_boxes", clipBoxes)
    op.addInput(boxes)
    op.addInput(scores)
    op.addInput(maxOutputSizePerClass)
    op.addInput(maxTotalSize)
    op.addInput(iouThreshold)
    op.addInput(scoreThreshold)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func compareAndBitpack<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    threshold: Tensor<T>
) -> Tensor<UInt8> {
  let nOutputs = Int(1)
    let op = makeOp("CompareAndBitpack", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    op.addInput(threshold)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func complex<
    T: FloatingPoint & TensorFlowScalar,
    Tout: TensorFlowScalar
>(
    real: Tensor<T>,
    imag: Tensor<T>
) -> Tensor<Tout> {
  let nOutputs = Int(1)
    let op = makeOp("Complex", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tout", Tout.tensorFlowDataType)
    op.addInput(real)
    op.addInput(imag)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func complexAbs<
    T: TensorFlowScalar,
    Tout: FloatingPoint & TensorFlowScalar
>(
    _ x: Tensor<T>
) -> Tensor<Tout> {
  let nOutputs = Int(1)
    let op = makeOp("ComplexAbs", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tout", Tout.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func complexStruct<TC: TensorGroup>(
    nA: Int64,
    nB: Int64
) -> (a: [Tensor<Int32>], b: [Tensor<Int64>], c: TC) {
  let nOutputs = Int(nA) + Int(nB) + Int(TC._typeList.count)
    let op = makeOp("ComplexStruct", nOutputs)
    op.updateAttribute("n_a", nA)
    op.updateAttribute("n_b", nB)
    op.updateAttribute("t_c", TC._typeList)
    return op.execute(Int(nA), Int(nB), Int(TC._typeList.count))
}

@inlinable @inline(__always)
public static func computeAccidentalHits(
    trueClasses: Tensor<Int64>,
    sampledCandidates: Tensor<Int64>,
    numTrue: Int64,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> (indices: Tensor<Int32>, ids: Tensor<Int64>, weights: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("ComputeAccidentalHits", nOutputs)
    op.updateAttribute("num_true", numTrue)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.addInput(trueClasses)
    op.addInput(sampledCandidates)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func concat<T: TensorFlowScalar>(
    concatDim: Tensor<Int32>,
    _ values: [Tensor<T>]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Concat", nOutputs)
    op.updateAttribute("N", values.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(concatDim)
    op.addInputList(values)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func concatOffset(
    concatDim: Tensor<Int32>,
    shape: [Tensor<Int32>]
) -> [Tensor<Int32>] {
  let nOutputs = Int(shape.count)
    let op = makeOp("ConcatOffset", nOutputs)
    op.updateAttribute("N", shape.count)
    op.addInput(concatDim)
    op.addInputList(shape)
    return op.execute(Int(shape.count))
}

@inlinable @inline(__always)
public static func concatV2<
    T: TensorFlowScalar,
    Tidx: TensorFlowIndex
>(
    _ values: [Tensor<T>],
    axis: Tensor<Tidx>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ConcatV2", nOutputs)
    op.updateAttribute("N", values.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInputList(values)
    op.addInput(axis)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func concatenateDataset(
    inputDataset: VariantHandle,
    anotherDataset: VariantHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ConcatenateDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(anotherDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func configureDistributedTPU(
    embeddingConfig: String,
    tpuEmbeddingConfig: String,
    isGlobalInit: Bool = false,
    enableWholeMeshCompilations: Bool = false,
    compilationFailureClosesChips: Bool = true
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("ConfigureDistributedTPU", nOutputs)
    op.updateAttribute("embedding_config", embeddingConfig)
    op.updateAttribute("tpu_embedding_config", tpuEmbeddingConfig)
    op.updateAttribute("is_global_init", isGlobalInit)
    op.updateAttribute("enable_whole_mesh_compilations", enableWholeMeshCompilations)
    op.updateAttribute("compilation_failure_closes_chips", compilationFailureClosesChips)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func configureTPUEmbedding(
    config: String
) {
  let nOutputs = 0
    let op = makeOp("ConfigureTPUEmbedding", nOutputs)
    op.updateAttribute("config", config)
    op.execute()
}

@inlinable @inline(__always)
public static func conj<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Conj", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func conjugateTranspose<
    T: TensorFlowScalar,
    Tperm: TensorFlowIndex
>(
    _ x: Tensor<T>,
    perm: Tensor<Tperm>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ConjugateTranspose", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tperm", Tperm.tensorFlowDataType)
    op.addInput(x)
    op.addInput(perm)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func constructionFails(
) {
  let nOutputs = 0
    let op = makeOp("ConstructionFails", nOutputs)
    
    op.execute()
}

@inlinable @inline(__always)
public static func consumeMutexLock(
    mutexLock: VariantHandle
) {
  let nOutputs = 0
    let op = makeOp("ConsumeMutexLock", nOutputs)
    op.addInput(mutexLock)
    op.execute()
}

@inlinable @inline(__always)
public static func controlTrigger(
) {
  let nOutputs = 0
    let op = makeOp("ControlTrigger", nOutputs)
    
    op.execute()
}

@inlinable @inline(__always)
public static func conv2D<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    filter: Tensor<T>,
    strides: [Int32],
    useCudnnOnGpu: Bool = true,
    padding: Padding3,
    explicitPaddings: [Int32],
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Conv2D", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("use_cudnn_on_gpu", useCudnnOnGpu)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("explicit_paddings", explicitPaddings)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filter)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func conv2DBackpropFilter<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    filterSizes: Tensor<Int32>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    useCudnnOnGpu: Bool = true,
    padding: Padding3,
    explicitPaddings: [Int32],
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Conv2DBackpropFilter", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("use_cudnn_on_gpu", useCudnnOnGpu)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("explicit_paddings", explicitPaddings)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filterSizes)
    op.addInput(outBackprop)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func conv2DBackpropInput<T: TensorFlowNumeric>(
    inputSizes: Tensor<Int32>,
    filter: Tensor<T>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    useCudnnOnGpu: Bool = true,
    padding: Padding3,
    explicitPaddings: [Int32],
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Conv2DBackpropInput", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("use_cudnn_on_gpu", useCudnnOnGpu)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("explicit_paddings", explicitPaddings)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(inputSizes)
    op.addInput(filter)
    op.addInput(outBackprop)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func conv3D<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    filter: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc,
    dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Conv3D", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filter)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func conv3DBackpropFilter<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    filter: Tensor<T>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Conv3DBackpropFilter", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(outBackprop)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func conv3DBackpropFilterV2<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    filterSizes: Tensor<Int32>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc,
    dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Conv3DBackpropFilterV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filterSizes)
    op.addInput(outBackprop)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func conv3DBackpropInput<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    filter: Tensor<T>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Conv3DBackpropInput", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(outBackprop)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func conv3DBackpropInputV2<
    T: FloatingPoint & TensorFlowScalar,
    Tshape: TensorFlowIndex
>(
    inputSizes: Tensor<Tshape>,
    filter: Tensor<T>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc,
    dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Conv3DBackpropInputV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("dilations", dilations)
    op.updateAttribute("Tshape", Tshape.tensorFlowDataType)
    op.addInput(inputSizes)
    op.addInput(filter)
    op.addInput(outBackprop)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func copy<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    tensorName: String,
    debugOpsSpec: [String]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Copy", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("tensor_name", tensorName)
    op.updateAttribute("debug_ops_spec", debugOpsSpec)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func copyHost<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    tensorName: String,
    debugOpsSpec: [String]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CopyHost", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("tensor_name", tensorName)
    op.updateAttribute("debug_ops_spec", debugOpsSpec)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func copyOp<T: TensorFlowScalar>(
    _ a: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CopyOp", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cos<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Cos", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cosh<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Cosh", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func createSummaryDbWriter(
    writer: ResourceHandle,
    dbUri: StringTensor,
    experimentName: StringTensor,
    runName: StringTensor,
    userName: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("CreateSummaryDbWriter", nOutputs)
    op.addInput(writer)
    op.addInput(dbUri)
    op.addInput(experimentName)
    op.addInput(runName)
    op.addInput(userName)
    op.execute()
}

@inlinable @inline(__always)
public static func createSummaryFileWriter(
    writer: ResourceHandle,
    logdir: StringTensor,
    maxQueue: Tensor<Int32>,
    flushMillis: Tensor<Int32>,
    filenameSuffix: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("CreateSummaryFileWriter", nOutputs)
    op.addInput(writer)
    op.addInput(logdir)
    op.addInput(maxQueue)
    op.addInput(flushMillis)
    op.addInput(filenameSuffix)
    op.execute()
}

@inlinable @inline(__always)
public static func createTRTResourceHandle(
    resourceName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("CreateTRTResourceHandle", nOutputs)
    op.updateAttribute("resource_name", resourceName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cropAndResize<T: TensorFlowNumeric>(
    image: Tensor<T>,
    boxes: Tensor<Float>,
    boxInd: Tensor<Int32>,
    cropSize: Tensor<Int32>,
    method: Method = .bilinear,
    extrapolationValue: Double = 0
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("CropAndResize", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("method", method.cName)
    op.updateAttribute("extrapolation_value", extrapolationValue)
    op.addInput(image)
    op.addInput(boxes)
    op.addInput(boxInd)
    op.addInput(cropSize)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cropAndResizeGradBoxes<T: TensorFlowNumeric>(
    grads: Tensor<Float>,
    image: Tensor<T>,
    boxes: Tensor<Float>,
    boxInd: Tensor<Int32>,
    method: Method4 = .bilinear
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("CropAndResizeGradBoxes", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("method", method.cName)
    op.addInput(grads)
    op.addInput(image)
    op.addInput(boxes)
    op.addInput(boxInd)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cropAndResizeGradImage<T: FloatingPoint & TensorFlowScalar>(
    grads: Tensor<Float>,
    boxes: Tensor<Float>,
    boxInd: Tensor<Int32>,
    imageSize: Tensor<Int32>,
    method: Method = .bilinear
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CropAndResizeGradImage", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("method", method.cName)
    op.addInput(grads)
    op.addInput(boxes)
    op.addInput(boxInd)
    op.addInput(imageSize)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cross<T: TensorFlowNumeric>(
    _ a: Tensor<T>,
    _ b: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Cross", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    op.addInput(b)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func crossReplicaSum<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    groupAssignment: Tensor<Int32>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CrossReplicaSum", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    op.addInput(groupAssignment)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cudnnRNN<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    inputH: Tensor<T>,
    inputC: Tensor<T>,
    params: Tensor<T>,
    rnnMode: RnnMode = .lstm,
    inputMode: InputMode = .linearInput,
    direction: Direction = .unidirectional,
    dropout: Double = 0,
    seed: Int64 = 0,
    seed2: Int64 = 0,
    isTraining: Bool = true
) -> (output: Tensor<T>, outputH: Tensor<T>, outputC: Tensor<T>, reserveSpace: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("CudnnRNN", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("rnn_mode", rnnMode.cName)
    op.updateAttribute("input_mode", inputMode.cName)
    op.updateAttribute("direction", direction.cName)
    op.updateAttribute("dropout", dropout)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("is_training", isTraining)
    op.addInput(input)
    op.addInput(inputH)
    op.addInput(inputC)
    op.addInput(params)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func cudnnRNNBackprop<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    inputH: Tensor<T>,
    inputC: Tensor<T>,
    params: Tensor<T>,
    output: Tensor<T>,
    outputH: Tensor<T>,
    outputC: Tensor<T>,
    outputBackprop: Tensor<T>,
    outputHBackprop: Tensor<T>,
    outputCBackprop: Tensor<T>,
    reserveSpace: Tensor<T>,
    rnnMode: RnnMode = .lstm,
    inputMode: InputMode = .linearInput,
    direction: Direction = .unidirectional,
    dropout: Double = 0,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> (inputBackprop: Tensor<T>, inputHBackprop: Tensor<T>, inputCBackprop: Tensor<T>, paramsBackprop: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("CudnnRNNBackprop", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("rnn_mode", rnnMode.cName)
    op.updateAttribute("input_mode", inputMode.cName)
    op.updateAttribute("direction", direction.cName)
    op.updateAttribute("dropout", dropout)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.addInput(input)
    op.addInput(inputH)
    op.addInput(inputC)
    op.addInput(params)
    op.addInput(output)
    op.addInput(outputH)
    op.addInput(outputC)
    op.addInput(outputBackprop)
    op.addInput(outputHBackprop)
    op.addInput(outputCBackprop)
    op.addInput(reserveSpace)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func cudnnRNNBackpropV2<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    inputH: Tensor<T>,
    inputC: Tensor<T>,
    params: Tensor<T>,
    output: Tensor<T>,
    outputH: Tensor<T>,
    outputC: Tensor<T>,
    outputBackprop: Tensor<T>,
    outputHBackprop: Tensor<T>,
    outputCBackprop: Tensor<T>,
    reserveSpace: Tensor<T>,
    hostReserved: Tensor<Int8>,
    rnnMode: RnnMode = .lstm,
    inputMode: InputMode = .linearInput,
    direction: Direction = .unidirectional,
    dropout: Double = 0,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> (inputBackprop: Tensor<T>, inputHBackprop: Tensor<T>, inputCBackprop: Tensor<T>, paramsBackprop: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("CudnnRNNBackpropV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("rnn_mode", rnnMode.cName)
    op.updateAttribute("input_mode", inputMode.cName)
    op.updateAttribute("direction", direction.cName)
    op.updateAttribute("dropout", dropout)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.addInput(input)
    op.addInput(inputH)
    op.addInput(inputC)
    op.addInput(params)
    op.addInput(output)
    op.addInput(outputH)
    op.addInput(outputC)
    op.addInput(outputBackprop)
    op.addInput(outputHBackprop)
    op.addInput(outputCBackprop)
    op.addInput(reserveSpace)
    op.addInput(hostReserved)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func cudnnRNNBackpropV3<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    inputH: Tensor<T>,
    inputC: Tensor<T>,
    params: Tensor<T>,
    sequenceLengths: Tensor<Int32>,
    output: Tensor<T>,
    outputH: Tensor<T>,
    outputC: Tensor<T>,
    outputBackprop: Tensor<T>,
    outputHBackprop: Tensor<T>,
    outputCBackprop: Tensor<T>,
    reserveSpace: Tensor<T>,
    hostReserved: Tensor<Int8>,
    rnnMode: RnnMode = .lstm,
    inputMode: InputMode = .linearInput,
    direction: Direction = .unidirectional,
    dropout: Double = 0,
    seed: Int64 = 0,
    seed2: Int64 = 0,
    numProj: Int64 = 0,
    timeMajor: Bool = true
) -> (inputBackprop: Tensor<T>, inputHBackprop: Tensor<T>, inputCBackprop: Tensor<T>, paramsBackprop: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("CudnnRNNBackpropV3", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("rnn_mode", rnnMode.cName)
    op.updateAttribute("input_mode", inputMode.cName)
    op.updateAttribute("direction", direction.cName)
    op.updateAttribute("dropout", dropout)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("num_proj", numProj)
    op.updateAttribute("time_major", timeMajor)
    op.addInput(input)
    op.addInput(inputH)
    op.addInput(inputC)
    op.addInput(params)
    op.addInput(sequenceLengths)
    op.addInput(output)
    op.addInput(outputH)
    op.addInput(outputC)
    op.addInput(outputBackprop)
    op.addInput(outputHBackprop)
    op.addInput(outputCBackprop)
    op.addInput(reserveSpace)
    op.addInput(hostReserved)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func cudnnRNNCanonicalToParams<T: FloatingPoint & TensorFlowScalar>(
    numLayers: Tensor<Int32>,
    numUnits: Tensor<Int32>,
    inputSize: Tensor<Int32>,
    weights: [Tensor<T>],
    biases: [Tensor<T>],
    rnnMode: RnnMode = .lstm,
    inputMode: InputMode = .linearInput,
    direction: Direction = .unidirectional,
    dropout: Double = 0,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CudnnRNNCanonicalToParams", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("num_params", weights.count)
    op.updateAttribute("rnn_mode", rnnMode.cName)
    op.updateAttribute("input_mode", inputMode.cName)
    op.updateAttribute("direction", direction.cName)
    op.updateAttribute("dropout", dropout)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.addInput(numLayers)
    op.addInput(numUnits)
    op.addInput(inputSize)
    op.addInputList(weights)
    op.addInputList(biases)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cudnnRNNCanonicalToParamsV2<T: FloatingPoint & TensorFlowScalar>(
    numLayers: Tensor<Int32>,
    numUnits: Tensor<Int32>,
    inputSize: Tensor<Int32>,
    weights: [Tensor<T>],
    biases: [Tensor<T>],
    rnnMode: RnnMode = .lstm,
    inputMode: InputMode = .linearInput,
    direction: Direction = .unidirectional,
    dropout: Double = 0,
    seed: Int64 = 0,
    seed2: Int64 = 0,
    numProj: Int64 = 0
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CudnnRNNCanonicalToParamsV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("num_params_weights", weights.count)
    op.updateAttribute("num_params_biases", biases.count)
    op.updateAttribute("rnn_mode", rnnMode.cName)
    op.updateAttribute("input_mode", inputMode.cName)
    op.updateAttribute("direction", direction.cName)
    op.updateAttribute("dropout", dropout)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("num_proj", numProj)
    op.addInput(numLayers)
    op.addInput(numUnits)
    op.addInput(inputSize)
    op.addInputList(weights)
    op.addInputList(biases)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cudnnRNNParamsSize<S: TensorFlowIndex>(
    numLayers: Tensor<Int32>,
    numUnits: Tensor<Int32>,
    inputSize: Tensor<Int32>,
    t: TensorDataType,
    rnnMode: RnnMode = .lstm,
    inputMode: InputMode = .linearInput,
    direction: Direction = .unidirectional,
    dropout: Double = 0,
    seed: Int64 = 0,
    seed2: Int64 = 0,
    numProj: Int64 = 0
) -> Tensor<S> {
  let nOutputs = Int(1)
    let op = makeOp("CudnnRNNParamsSize", nOutputs)
    op.updateAttribute("T", t)
    op.updateAttribute("S", S.tensorFlowDataType)
    op.updateAttribute("rnn_mode", rnnMode.cName)
    op.updateAttribute("input_mode", inputMode.cName)
    op.updateAttribute("direction", direction.cName)
    op.updateAttribute("dropout", dropout)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("num_proj", numProj)
    op.addInput(numLayers)
    op.addInput(numUnits)
    op.addInput(inputSize)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cudnnRNNParamsToCanonical<T: FloatingPoint & TensorFlowScalar>(
    numLayers: Tensor<Int32>,
    numUnits: Tensor<Int32>,
    inputSize: Tensor<Int32>,
    params: Tensor<T>,
    numParams: Int64,
    rnnMode: RnnMode = .lstm,
    inputMode: InputMode = .linearInput,
    direction: Direction = .unidirectional,
    dropout: Double = 0,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> (weights: [Tensor<T>], biases: [Tensor<T>]) {
  let nOutputs = Int(numParams) + Int(numParams)
    let op = makeOp("CudnnRNNParamsToCanonical", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("num_params", numParams)
    op.updateAttribute("rnn_mode", rnnMode.cName)
    op.updateAttribute("input_mode", inputMode.cName)
    op.updateAttribute("direction", direction.cName)
    op.updateAttribute("dropout", dropout)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.addInput(numLayers)
    op.addInput(numUnits)
    op.addInput(inputSize)
    op.addInput(params)
    return op.execute(Int(numParams), Int(numParams))
}

@inlinable @inline(__always)
public static func cudnnRNNParamsToCanonicalV2<T: FloatingPoint & TensorFlowScalar>(
    numLayers: Tensor<Int32>,
    numUnits: Tensor<Int32>,
    inputSize: Tensor<Int32>,
    params: Tensor<T>,
    numParamsWeights: Int64,
    numParamsBiases: Int64,
    rnnMode: RnnMode = .lstm,
    inputMode: InputMode = .linearInput,
    direction: Direction = .unidirectional,
    dropout: Double = 0,
    seed: Int64 = 0,
    seed2: Int64 = 0,
    numProj: Int64 = 0
) -> (weights: [Tensor<T>], biases: [Tensor<T>]) {
  let nOutputs = Int(numParamsWeights) + Int(numParamsBiases)
    let op = makeOp("CudnnRNNParamsToCanonicalV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("num_params_weights", numParamsWeights)
    op.updateAttribute("num_params_biases", numParamsBiases)
    op.updateAttribute("rnn_mode", rnnMode.cName)
    op.updateAttribute("input_mode", inputMode.cName)
    op.updateAttribute("direction", direction.cName)
    op.updateAttribute("dropout", dropout)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("num_proj", numProj)
    op.addInput(numLayers)
    op.addInput(numUnits)
    op.addInput(inputSize)
    op.addInput(params)
    return op.execute(Int(numParamsWeights), Int(numParamsBiases))
}

@inlinable @inline(__always)
public static func cudnnRNNV2<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    inputH: Tensor<T>,
    inputC: Tensor<T>,
    params: Tensor<T>,
    rnnMode: RnnMode = .lstm,
    inputMode: InputMode = .linearInput,
    direction: Direction = .unidirectional,
    dropout: Double = 0,
    seed: Int64 = 0,
    seed2: Int64 = 0,
    isTraining: Bool = true
) -> (output: Tensor<T>, outputH: Tensor<T>, outputC: Tensor<T>, reserveSpace: Tensor<T>, hostReserved: Tensor<Int8>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("CudnnRNNV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("rnn_mode", rnnMode.cName)
    op.updateAttribute("input_mode", inputMode.cName)
    op.updateAttribute("direction", direction.cName)
    op.updateAttribute("dropout", dropout)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("is_training", isTraining)
    op.addInput(input)
    op.addInput(inputH)
    op.addInput(inputC)
    op.addInput(params)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func cudnnRNNV3<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    inputH: Tensor<T>,
    inputC: Tensor<T>,
    params: Tensor<T>,
    sequenceLengths: Tensor<Int32>,
    rnnMode: RnnMode = .lstm,
    inputMode: InputMode = .linearInput,
    direction: Direction = .unidirectional,
    dropout: Double = 0,
    seed: Int64 = 0,
    seed2: Int64 = 0,
    numProj: Int64 = 0,
    isTraining: Bool = true,
    timeMajor: Bool = true
) -> (output: Tensor<T>, outputH: Tensor<T>, outputC: Tensor<T>, reserveSpace: Tensor<T>, hostReserved: Tensor<Int8>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("CudnnRNNV3", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("rnn_mode", rnnMode.cName)
    op.updateAttribute("input_mode", inputMode.cName)
    op.updateAttribute("direction", direction.cName)
    op.updateAttribute("dropout", dropout)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("num_proj", numProj)
    op.updateAttribute("is_training", isTraining)
    op.updateAttribute("time_major", timeMajor)
    op.addInput(input)
    op.addInput(inputH)
    op.addInput(inputC)
    op.addInput(params)
    op.addInput(sequenceLengths)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func cumprod<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
>(
    _ x: Tensor<T>,
    axis: Tensor<Tidx>,
    exclusive: Bool = false,
    reverse: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Cumprod", nOutputs)
    op.updateAttribute("exclusive", exclusive)
    op.updateAttribute("reverse", reverse)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(x)
    op.addInput(axis)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cumsum<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
>(
    _ x: Tensor<T>,
    axis: Tensor<Tidx>,
    exclusive: Bool = false,
    reverse: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Cumsum", nOutputs)
    op.updateAttribute("exclusive", exclusive)
    op.updateAttribute("reverse", reverse)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(x)
    op.addInput(axis)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func cumulativeLogsumexp<
    T: FloatingPoint & TensorFlowScalar,
    Tidx: TensorFlowIndex
>(
    _ x: Tensor<T>,
    axis: Tensor<Tidx>,
    exclusive: Bool = false,
    reverse: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("CumulativeLogsumexp", nOutputs)
    op.updateAttribute("exclusive", exclusive)
    op.updateAttribute("reverse", reverse)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(x)
    op.addInput(axis)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func dataFormatDimMap<T: TensorFlowIndex>(
    _ x: Tensor<T>,
    srcFormat: String = "NHWC",
    dstFormat: String = "NCHW"
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DataFormatDimMap", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("src_format", srcFormat)
    op.updateAttribute("dst_format", dstFormat)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func dataFormatVecPermute<T: TensorFlowIndex>(
    _ x: Tensor<T>,
    srcFormat: String = "NHWC",
    dstFormat: String = "NCHW"
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DataFormatVecPermute", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("src_format", srcFormat)
    op.updateAttribute("dst_format", dstFormat)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func datasetCardinality(
    inputDataset: VariantHandle
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("DatasetCardinality", nOutputs)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func datasetFromGraph(
    graphDef: StringTensor
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("DatasetFromGraph", nOutputs)
    op.addInput(graphDef)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func datasetToGraph(
    inputDataset: VariantHandle,
    statefulWhitelist: [String],
    allowStateful: Bool = false,
    stripDeviceAssignment: Bool = false
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("DatasetToGraph", nOutputs)
    op.updateAttribute("stateful_whitelist", statefulWhitelist)
    op.updateAttribute("allow_stateful", allowStateful)
    op.updateAttribute("strip_device_assignment", stripDeviceAssignment)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func datasetToGraphV2(
    inputDataset: VariantHandle,
    externalStatePolicy: Int64 = 0,
    stripDeviceAssignment: Bool = false
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("DatasetToGraphV2", nOutputs)
    op.updateAttribute("external_state_policy", externalStatePolicy)
    op.updateAttribute("strip_device_assignment", stripDeviceAssignment)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func datasetToSingleElement<OutputTypes: TensorGroup>(
    dataset: VariantHandle,
    outputShapes: [TensorShape?]
) -> OutputTypes {
  let nOutputs = Int(OutputTypes._typeList.count)
    let op = makeOp("DatasetToSingleElement", nOutputs)
    op.updateAttribute("output_types", OutputTypes._typeList)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(dataset)
    return op.execute(Int(OutputTypes._typeList.count))
}

@inlinable @inline(__always)
public static func datasetToTFRecord(
    inputDataset: VariantHandle,
    filename: StringTensor,
    compressionType: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("DatasetToTFRecord", nOutputs)
    op.addInput(inputDataset)
    op.addInput(filename)
    op.addInput(compressionType)
    op.execute()
}

@inlinable @inline(__always)
public static func debugGradientIdentity<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DebugGradientIdentity", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func debugIdentity<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    deviceName: String,
    tensorName: String,
    debugUrls: [String],
    gatedGrpc: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DebugIdentity", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("device_name", deviceName)
    op.updateAttribute("tensor_name", tensorName)
    op.updateAttribute("debug_urls", debugUrls)
    op.updateAttribute("gated_grpc", gatedGrpc)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func debugIdentityV2<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    tfdbgContextId: String,
    opName: String,
    outputSlot: Int64 = -1,
    tensorDebugMode: Int64 = -1,
    debugUrls: [String]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DebugIdentityV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("tfdbg_context_id", tfdbgContextId)
    op.updateAttribute("op_name", opName)
    op.updateAttribute("output_slot", outputSlot)
    op.updateAttribute("tensor_debug_mode", tensorDebugMode)
    op.updateAttribute("debug_urls", debugUrls)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func debugNanCount<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    deviceName: String,
    tensorName: String,
    debugUrls: [String],
    gatedGrpc: Bool = false
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("DebugNanCount", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("device_name", deviceName)
    op.updateAttribute("tensor_name", tensorName)
    op.updateAttribute("debug_urls", debugUrls)
    op.updateAttribute("gated_grpc", gatedGrpc)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func debugNumericSummary<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    deviceName: String,
    tensorName: String,
    debugUrls: [String],
    lowerBound: Double = -Double.infinity,
    upperBound: Double = Double.infinity,
    muteIfHealthy: Bool = false,
    gatedGrpc: Bool = false
) -> Tensor<Double> {
  let nOutputs = Int(1)
    let op = makeOp("DebugNumericSummary", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("device_name", deviceName)
    op.updateAttribute("tensor_name", tensorName)
    op.updateAttribute("debug_urls", debugUrls)
    op.updateAttribute("lower_bound", lowerBound)
    op.updateAttribute("upper_bound", upperBound)
    op.updateAttribute("mute_if_healthy", muteIfHealthy)
    op.updateAttribute("gated_grpc", gatedGrpc)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func debugNumericSummaryV2<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    tensorDebugMode: Int64 = -1,
    tensorId: Int64 = -1
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("DebugNumericSummaryV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("tensor_debug_mode", tensorDebugMode)
    op.updateAttribute("tensor_id", tensorId)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func decodeAndCropJpeg(
    contents: StringTensor,
    cropWindow: Tensor<Int32>,
    channels: Int64 = 0,
    ratio: Int64 = 1,
    fancyUpscaling: Bool = true,
    tryRecoverTruncated: Bool = false,
    acceptableFraction: Double = 1,
    dctMethod: String
) -> Tensor<UInt8> {
  let nOutputs = Int(1)
    let op = makeOp("DecodeAndCropJpeg", nOutputs)
    op.updateAttribute("channels", channels)
    op.updateAttribute("ratio", ratio)
    op.updateAttribute("fancy_upscaling", fancyUpscaling)
    op.updateAttribute("try_recover_truncated", tryRecoverTruncated)
    op.updateAttribute("acceptable_fraction", acceptableFraction)
    op.updateAttribute("dct_method", dctMethod)
    op.addInput(contents)
    op.addInput(cropWindow)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func decodeBase64(
    _ input: StringTensor
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("DecodeBase64", nOutputs)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func decodeBmp(
    contents: StringTensor,
    channels: Int64 = 0
) -> Tensor<UInt8> {
  let nOutputs = Int(1)
    let op = makeOp("DecodeBmp", nOutputs)
    op.updateAttribute("channels", channels)
    op.addInput(contents)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func decodeCSV<OutType: TensorArrayProtocol>(
    records: StringTensor,
    recordDefaults: OutType,
    fieldDelim: String = ",",
    useQuoteDelim: Bool = true,
    naValue: String,
    selectCols: [Int32]
) -> OutType {
  let nOutputs = Int(recordDefaults._typeList.count)
    let op = makeOp("DecodeCSV", nOutputs)
    op.updateAttribute("OUT_TYPE", recordDefaults._typeList)
    op.updateAttribute("field_delim", fieldDelim)
    op.updateAttribute("use_quote_delim", useQuoteDelim)
    op.updateAttribute("na_value", naValue)
    op.updateAttribute("select_cols", selectCols)
    op.addInput(records)
    op.addInputList(recordDefaults)
    return op.execute(Int(recordDefaults._typeList.count))
}

@inlinable @inline(__always)
public static func decodeCompressed(
    bytes: StringTensor,
    compressionType: String
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("DecodeCompressed", nOutputs)
    op.updateAttribute("compression_type", compressionType)
    op.addInput(bytes)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func decodeGif(
    contents: StringTensor
) -> Tensor<UInt8> {
  let nOutputs = Int(1)
    let op = makeOp("DecodeGif", nOutputs)
    op.addInput(contents)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func decodeJSONExample(
    jsonExamples: StringTensor
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("DecodeJSONExample", nOutputs)
    op.addInput(jsonExamples)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func decodeJpeg(
    contents: StringTensor,
    channels: Int64 = 0,
    ratio: Int64 = 1,
    fancyUpscaling: Bool = true,
    tryRecoverTruncated: Bool = false,
    acceptableFraction: Double = 1,
    dctMethod: String
) -> Tensor<UInt8> {
  let nOutputs = Int(1)
    let op = makeOp("DecodeJpeg", nOutputs)
    op.updateAttribute("channels", channels)
    op.updateAttribute("ratio", ratio)
    op.updateAttribute("fancy_upscaling", fancyUpscaling)
    op.updateAttribute("try_recover_truncated", tryRecoverTruncated)
    op.updateAttribute("acceptable_fraction", acceptableFraction)
    op.updateAttribute("dct_method", dctMethod)
    op.addInput(contents)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func decodePaddedRaw<OutType: TensorFlowNumeric>(
    inputBytes: StringTensor,
    fixedLength: Tensor<Int32>,
    littleEndian: Bool = true
) -> Tensor<OutType> {
  let nOutputs = Int(1)
    let op = makeOp("DecodePaddedRaw", nOutputs)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("little_endian", littleEndian)
    op.addInput(inputBytes)
    op.addInput(fixedLength)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func decodePng<Dtype: UnsignedInteger & TensorFlowScalar>(
    contents: StringTensor,
    channels: Int64 = 0
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("DecodePng", nOutputs)
    op.updateAttribute("channels", channels)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(contents)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func decodeProtoV2<OutputTypes: TensorGroup>(
    bytes: StringTensor,
    messageType: String,
    fieldNames: [String],
    descriptorSource: String = "local://",
    messageFormat: String = "binary",
    sanitize: Bool = false
) -> (sizes: Tensor<Int32>, values: OutputTypes) {
  let nOutputs = Int(1) + Int(OutputTypes._typeList.count)
    let op = makeOp("DecodeProtoV2", nOutputs)
    op.updateAttribute("message_type", messageType)
    op.updateAttribute("field_names", fieldNames)
    op.updateAttribute("output_types", OutputTypes._typeList)
    op.updateAttribute("descriptor_source", descriptorSource)
    op.updateAttribute("message_format", messageFormat)
    op.updateAttribute("sanitize", sanitize)
    op.addInput(bytes)
    return op.execute(Int(1), Int(OutputTypes._typeList.count))
}

@inlinable @inline(__always)
public static func decodeRaw<OutType: TensorFlowScalar>(
    bytes: StringTensor,
    littleEndian: Bool = true
) -> Tensor<OutType> {
  let nOutputs = Int(1)
    let op = makeOp("DecodeRaw", nOutputs)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("little_endian", littleEndian)
    op.addInput(bytes)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func decodeWav(
    contents: StringTensor,
    desiredChannels: Int64 = -1,
    desiredSamples: Int64 = -1
) -> (audio: Tensor<Float>, sampleRate: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("DecodeWav", nOutputs)
    op.updateAttribute("desired_channels", desiredChannels)
    op.updateAttribute("desired_samples", desiredSamples)
    op.addInput(contents)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func deepCopy<T: TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DeepCopy", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func deleteIterator(
    handle: ResourceHandle,
    deleter: VariantHandle
) {
  let nOutputs = 0
    let op = makeOp("DeleteIterator", nOutputs)
    op.addInput(handle)
    op.addInput(deleter)
    op.execute()
}

@inlinable @inline(__always)
public static func deleteMemoryCache(
    handle: ResourceHandle,
    deleter: VariantHandle
) {
  let nOutputs = 0
    let op = makeOp("DeleteMemoryCache", nOutputs)
    op.addInput(handle)
    op.addInput(deleter)
    op.execute()
}

@inlinable @inline(__always)
public static func deleteMultiDeviceIterator(
    multiDeviceIterator: ResourceHandle,
    iterators: [ResourceHandle],
    deleter: VariantHandle
) {
  let nOutputs = 0
    let op = makeOp("DeleteMultiDeviceIterator", nOutputs)
    op.updateAttribute("N", iterators.count)
    op.addInput(multiDeviceIterator)
    op.addInputList(iterators)
    op.addInput(deleter)
    op.execute()
}

@inlinable @inline(__always)
public static func deleteRandomSeedGenerator(
    handle: ResourceHandle,
    deleter: VariantHandle
) {
  let nOutputs = 0
    let op = makeOp("DeleteRandomSeedGenerator", nOutputs)
    op.addInput(handle)
    op.addInput(deleter)
    op.execute()
}

@inlinable @inline(__always)
public static func deleteSessionTensor(
    handle: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("DeleteSessionTensor", nOutputs)
    op.addInput(handle)
    op.execute()
}

@inlinable @inline(__always)
public static func denseToCSRSparseMatrix<T: FloatingPoint & TensorFlowScalar>(
    denseInput: Tensor<T>,
    indices: Tensor<Int64>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("DenseToCSRSparseMatrix", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(denseInput)
    op.addInput(indices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func denseToDenseSetOperation<T: TensorFlowInteger>(
    set1: Tensor<T>,
    set2: Tensor<T>,
    setOperation: String,
    validateIndices: Bool = true
) -> (resultIndices: Tensor<Int64>, resultValues: Tensor<T>, resultShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("DenseToDenseSetOperation", nOutputs)
    op.updateAttribute("set_operation", setOperation)
    op.updateAttribute("validate_indices", validateIndices)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(set1)
    op.addInput(set2)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func denseToDenseSetOperation(
    set1: StringTensor,
    set2: StringTensor,
    setOperation: String,
    validateIndices: Bool = true
) -> (resultIndices: Tensor<Int64>, resultValues: StringTensor, resultShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("DenseToDenseSetOperation", nOutputs)
    op.updateAttribute("set_operation", setOperation)
    op.updateAttribute("validate_indices", validateIndices)
    op.updateAttribute("T", TensorDataType(TF_STRING))
    op.addInput(set1)
    op.addInput(set2)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func denseToSparseBatchDataset(
    inputDataset: VariantHandle,
    batchSize: Tensor<Int64>,
    rowShape: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("DenseToSparseBatchDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(batchSize)
    op.addInput(rowShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func denseToSparseSetOperation<T: TensorFlowInteger>(
    set1: Tensor<T>,
    set2Indices: Tensor<Int64>,
    set2Values: Tensor<T>,
    set2Shape: Tensor<Int64>,
    setOperation: String,
    validateIndices: Bool = true
) -> (resultIndices: Tensor<Int64>, resultValues: Tensor<T>, resultShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("DenseToSparseSetOperation", nOutputs)
    op.updateAttribute("set_operation", setOperation)
    op.updateAttribute("validate_indices", validateIndices)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(set1)
    op.addInput(set2Indices)
    op.addInput(set2Values)
    op.addInput(set2Shape)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func denseToSparseSetOperation(
    set1: StringTensor,
    set2Indices: Tensor<Int64>,
    set2Values: StringTensor,
    set2Shape: Tensor<Int64>,
    setOperation: String,
    validateIndices: Bool = true
) -> (resultIndices: Tensor<Int64>, resultValues: StringTensor, resultShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("DenseToSparseSetOperation", nOutputs)
    op.updateAttribute("set_operation", setOperation)
    op.updateAttribute("validate_indices", validateIndices)
    op.updateAttribute("T", TensorDataType(TF_STRING))
    op.addInput(set1)
    op.addInput(set2Indices)
    op.addInput(set2Values)
    op.addInput(set2Shape)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func depthToSpace<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    blockSize: Int64,
    dataFormat: DataFormat5 = .nhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DepthToSpace", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("block_size", blockSize)
    op.updateAttribute("data_format", dataFormat.cName)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func depthwiseConv2dNative<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    filter: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DepthwiseConv2dNative", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filter)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func depthwiseConv2dNativeBackpropFilter<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    filterSizes: Tensor<Int32>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DepthwiseConv2dNativeBackpropFilter", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filterSizes)
    op.addInput(outBackprop)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func depthwiseConv2dNativeBackpropInput<T: FloatingPoint & TensorFlowScalar>(
    inputSizes: Tensor<Int32>,
    filter: Tensor<T>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DepthwiseConv2dNativeBackpropInput", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(inputSizes)
    op.addInput(filter)
    op.addInput(outBackprop)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func dequantize<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    minRange: Tensor<Float>,
    maxRange: Tensor<Float>,
    mode: Mode = .minCombined,
    narrowRange: Bool = false,
    axis: Int64 = -1
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("Dequantize", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("mode", mode.cName)
    op.updateAttribute("narrow_range", narrowRange)
    op.updateAttribute("axis", axis)
    op.addInput(input)
    op.addInput(minRange)
    op.addInput(maxRange)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func deserializeIterator(
    resourceHandle: ResourceHandle,
    serialized: VariantHandle
) {
  let nOutputs = 0
    let op = makeOp("DeserializeIterator", nOutputs)
    op.addInput(resourceHandle)
    op.addInput(serialized)
    op.execute()
}

@inlinable @inline(__always)
public static func deserializeManySparse<Dtype: TensorFlowScalar>(
    serializedSparse: StringTensor
) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<Dtype>, sparseShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("DeserializeManySparse", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(serializedSparse)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func deserializeSparse<
    Dtype: TensorFlowScalar,
    Tserialized: TensorFlowScalar
>(
    serializedSparse: Tensor<Tserialized>
) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<Dtype>, sparseShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("DeserializeSparse", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("Tserialized", Tserialized.tensorFlowDataType)
    op.addInput(serializedSparse)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func deserializeSparse<Dtype: TensorFlowScalar>(
    serializedSparse: StringTensor
) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<Dtype>, sparseShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("DeserializeSparse", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("Tserialized", TensorDataType(TF_STRING))
    op.addInput(serializedSparse)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func destroyResourceOp(
    resource: ResourceHandle,
    ignoreLookupError: Bool = true
) {
  let nOutputs = 0
    let op = makeOp("DestroyResourceOp", nOutputs)
    op.updateAttribute("ignore_lookup_error", ignoreLookupError)
    op.addInput(resource)
    op.execute()
}

@inlinable @inline(__always)
public static func devicePlacementOp(
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("DevicePlacementOp", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func diag<T: TensorFlowNumeric>(
    diagonal: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Diag", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(diagonal)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func diagPart<T: TensorFlowNumeric>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DiagPart", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func digamma<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Digamma", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func dilation2D<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    filter: Tensor<T>,
    strides: [Int32],
    rates: [Int32],
    padding: Padding
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Dilation2D", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("rates", rates)
    op.updateAttribute("padding", padding.cName)
    op.addInput(input)
    op.addInput(filter)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func dilation2DBackpropFilter<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    filter: Tensor<T>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    rates: [Int32],
    padding: Padding
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Dilation2DBackpropFilter", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("rates", rates)
    op.updateAttribute("padding", padding.cName)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(outBackprop)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func dilation2DBackpropInput<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    filter: Tensor<T>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    rates: [Int32],
    padding: Padding
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Dilation2DBackpropInput", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("rates", rates)
    op.updateAttribute("padding", padding.cName)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(outBackprop)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func directedInterleaveDataset(
    selectorInputDataset: VariantHandle,
    dataInputDatasets: [VariantHandle],
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("DirectedInterleaveDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("N", dataInputDatasets.count)
    op.addInput(selectorInputDataset)
    op.addInputList(dataInputDatasets)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func div<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Div", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func divNoNan<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DivNoNan", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func drawBoundingBoxes<T: FloatingPoint & TensorFlowScalar>(
    images: Tensor<T>,
    boxes: Tensor<Float>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DrawBoundingBoxes", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(images)
    op.addInput(boxes)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func drawBoundingBoxesV2<T: FloatingPoint & TensorFlowScalar>(
    images: Tensor<T>,
    boxes: Tensor<Float>,
    colors: Tensor<Float>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DrawBoundingBoxesV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(images)
    op.addInput(boxes)
    op.addInput(colors)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func dynamicPartition<T: TensorFlowScalar>(
    data: Tensor<T>,
    partitions: Tensor<Int32>,
    numPartitions: Int64
) -> [Tensor<T>] {
  let nOutputs = Int(numPartitions)
    let op = makeOp("DynamicPartition", nOutputs)
    op.updateAttribute("num_partitions", numPartitions)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(data)
    op.addInput(partitions)
    return op.execute(Int(numPartitions))
}

@inlinable @inline(__always)
public static func dynamicStitch<T: TensorFlowScalar>(
    indices: [Tensor<Int32>],
    data: [Tensor<T>]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("DynamicStitch", nOutputs)
    op.updateAttribute("N", indices.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInputList(indices)
    op.addInputList(data)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func eagerPyFunc<
    Tin: TensorArrayProtocol,
    Tout: TensorGroup
>(
    _ input: Tin,
    token: String,
    isAsync: Bool = false
) -> Tout {
  let nOutputs = Int(Tout._typeList.count)
    let op = makeOp("EagerPyFunc", nOutputs)
    op.updateAttribute("token", token)
    op.updateAttribute("is_async", isAsync)
    op.updateAttribute("Tin", input._typeList)
    op.updateAttribute("Tout", Tout._typeList)
    op.addInputList(input)
    return op.execute(Int(Tout._typeList.count))
}

@inlinable @inline(__always)
public static func editDistance<T: TensorFlowScalar>(
    hypothesisIndices: Tensor<Int64>,
    hypothesisValues: Tensor<T>,
    hypothesisShape: Tensor<Int64>,
    truthIndices: Tensor<Int64>,
    truthValues: Tensor<T>,
    truthShape: Tensor<Int64>,
    normalize: Bool = true
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("EditDistance", nOutputs)
    op.updateAttribute("normalize", normalize)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(hypothesisIndices)
    op.addInput(hypothesisValues)
    op.addInput(hypothesisShape)
    op.addInput(truthIndices)
    op.addInput(truthValues)
    op.addInput(truthShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func eig<
    T: FloatingPoint & TensorFlowScalar,
    Tout: TensorFlowScalar
>(
    _ input: Tensor<T>,
    computeV: Bool = true
) -> (e: Tensor<Tout>, v: Tensor<Tout>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("Eig", nOutputs)
    op.updateAttribute("compute_v", computeV)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tout", Tout.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func einsum<T: TensorFlowScalar>(
    inputs: [Tensor<T>],
    equation: String
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Einsum", nOutputs)
    op.updateAttribute("equation", equation)
    op.updateAttribute("N", inputs.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInputList(inputs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func elu<T: FloatingPoint & TensorFlowScalar>(
    features: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Elu", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(features)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func eluGrad<T: FloatingPoint & TensorFlowScalar>(
    gradients: Tensor<T>,
    outputs: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("EluGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(gradients)
    op.addInput(outputs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func empty<Dtype: TensorFlowScalar>(
    shape: Tensor<Int32>,
    init_: Bool = false
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("Empty", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("init", init_)
    op.addInput(shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func emptyTensorList<ShapeType: TensorFlowIndex>(
    elementShape: Tensor<ShapeType>,
    maxNumElements: Tensor<Int32>,
    elementDtype: TensorDataType
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("EmptyTensorList", nOutputs)
    op.updateAttribute("element_dtype", elementDtype)
    op.updateAttribute("shape_type", ShapeType.tensorFlowDataType)
    op.addInput(elementShape)
    op.addInput(maxNumElements)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func encodeBase64(
    _ input: StringTensor,
    pad: Bool = false
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("EncodeBase64", nOutputs)
    op.updateAttribute("pad", pad)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func encodeJpeg(
    image: Tensor<UInt8>,
    format: Format,
    quality: Int64 = 95,
    progressive: Bool = false,
    optimizeSize: Bool = false,
    chromaDownsampling: Bool = true,
    densityUnit: DensityUnit = .in_,
    xDensity: Int64 = 300,
    yDensity: Int64 = 300,
    xmpMetadata: String
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("EncodeJpeg", nOutputs)
    op.updateAttribute("format", format.cName)
    op.updateAttribute("quality", quality)
    op.updateAttribute("progressive", progressive)
    op.updateAttribute("optimize_size", optimizeSize)
    op.updateAttribute("chroma_downsampling", chromaDownsampling)
    op.updateAttribute("density_unit", densityUnit.cName)
    op.updateAttribute("x_density", xDensity)
    op.updateAttribute("y_density", yDensity)
    op.updateAttribute("xmp_metadata", xmpMetadata)
    op.addInput(image)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func encodeJpegVariableQuality(
    images: Tensor<UInt8>,
    quality: Tensor<Int32>
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("EncodeJpegVariableQuality", nOutputs)
    op.addInput(images)
    op.addInput(quality)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func encodePng<T: UnsignedInteger & TensorFlowScalar>(
    image: Tensor<T>,
    compression: Int64 = -1
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("EncodePng", nOutputs)
    op.updateAttribute("compression", compression)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(image)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func encodeProto<TinputTypes: TensorArrayProtocol>(
    sizes: Tensor<Int32>,
    _ values: TinputTypes,
    fieldNames: [String],
    messageType: String,
    descriptorSource: String = "local://"
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("EncodeProto", nOutputs)
    op.updateAttribute("field_names", fieldNames)
    op.updateAttribute("message_type", messageType)
    op.updateAttribute("descriptor_source", descriptorSource)
    op.updateAttribute("Tinput_types", values._typeList)
    op.addInput(sizes)
    op.addInputList(values)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func encodeWav(
    audio: Tensor<Float>,
    sampleRate: Tensor<Int32>
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("EncodeWav", nOutputs)
    op.addInput(audio)
    op.addInput(sampleRate)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func enqueueTPUEmbeddingIntegerBatch(
    batch: [Tensor<Int32>],
    modeOverride: StringTensor,
    deviceOrdinal: Int64 = -1
) {
  let nOutputs = 0
    let op = makeOp("EnqueueTPUEmbeddingIntegerBatch", nOutputs)
    op.updateAttribute("N", batch.count)
    op.updateAttribute("device_ordinal", deviceOrdinal)
    op.addInputList(batch)
    op.addInput(modeOverride)
    op.execute()
}

@inlinable @inline(__always)
public static func enqueueTPUEmbeddingSparseBatch<
    T1: TensorFlowIndex,
    T2: TensorFlowIndex,
    T3: FloatingPoint & TensorFlowScalar
>(
    sampleIndices: [Tensor<T1>],
    embeddingIndices: [Tensor<T2>],
    aggregationWeights: [Tensor<T3>],
    modeOverride: StringTensor,
    deviceOrdinal: Int64 = -1,
    combiners: [String]
) {
  let nOutputs = 0
    let op = makeOp("EnqueueTPUEmbeddingSparseBatch", nOutputs)
    op.updateAttribute("T1", T1.tensorFlowDataType)
    op.updateAttribute("T2", T2.tensorFlowDataType)
    op.updateAttribute("T3", T3.tensorFlowDataType)
    op.updateAttribute("N", sampleIndices.count)
    op.updateAttribute("device_ordinal", deviceOrdinal)
    op.updateAttribute("combiners", combiners)
    op.addInputList(sampleIndices)
    op.addInputList(embeddingIndices)
    op.addInputList(aggregationWeights)
    op.addInput(modeOverride)
    op.execute()
}

@inlinable @inline(__always)
public static func enqueueTPUEmbeddingSparseTensorBatch<
    T1: TensorFlowIndex,
    T2: TensorFlowIndex,
    T3: FloatingPoint & TensorFlowScalar
>(
    sampleIndices: [Tensor<T1>],
    embeddingIndices: [Tensor<T2>],
    aggregationWeights: [Tensor<T3>],
    modeOverride: StringTensor,
    deviceOrdinal: Int64 = -1,
    combiners: [String],
    tableIds: [Int32],
    maxSequenceLengths: [Int32]
) {
  let nOutputs = 0
    let op = makeOp("EnqueueTPUEmbeddingSparseTensorBatch", nOutputs)
    op.updateAttribute("T1", T1.tensorFlowDataType)
    op.updateAttribute("T2", T2.tensorFlowDataType)
    op.updateAttribute("T3", T3.tensorFlowDataType)
    op.updateAttribute("N", sampleIndices.count)
    op.updateAttribute("device_ordinal", deviceOrdinal)
    op.updateAttribute("combiners", combiners)
    op.updateAttribute("table_ids", tableIds)
    op.updateAttribute("max_sequence_lengths", maxSequenceLengths)
    op.addInputList(sampleIndices)
    op.addInputList(embeddingIndices)
    op.addInputList(aggregationWeights)
    op.addInput(modeOverride)
    op.execute()
}

@inlinable @inline(__always)
public static func ensureShape<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    shape: TensorShape?
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("EnsureShape", nOutputs)
    op.updateAttribute("shape", shape)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func enter<T: TensorFlowScalar>(
    data: Tensor<T>,
    frameName: String,
    isConstant: Bool = false,
    parallelIterations: Int64 = 10
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Enter", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("frame_name", frameName)
    op.updateAttribute("is_constant", isConstant)
    op.updateAttribute("parallel_iterations", parallelIterations)
    op.addInput(data)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func equal<T: TensorFlowScalar>(
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    incompatibleShapeError: Bool = true
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("Equal", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("incompatible_shape_error", incompatibleShapeError)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func equal(
    _ x: StringTensor,
    _ y: StringTensor,
    incompatibleShapeError: Bool = true
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("Equal", nOutputs)
    op.updateAttribute("T", TensorDataType(TF_STRING))
    op.updateAttribute("incompatible_shape_error", incompatibleShapeError)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func erf<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Erf", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func erfc<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Erfc", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func erfinv<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Erfinv", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func euclideanNorm<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
>(
    _ input: Tensor<T>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("EuclideanNorm", nOutputs)
    op.updateAttribute("keep_dims", keepDims)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(input)
    op.addInput(reductionIndices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func exit<T: TensorFlowScalar>(
    data: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Exit", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(data)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func exp<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Exp", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func expandDims<
    T: TensorFlowScalar,
    Tdim: TensorFlowIndex
>(
    _ input: Tensor<T>,
    dim: Tensor<Tdim>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ExpandDims", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tdim", Tdim.tensorFlowDataType)
    op.addInput(input)
    op.addInput(dim)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalAssertNextDataset(
    inputDataset: VariantHandle,
    transformations: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalAssertNextDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(transformations)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalAutoShardDataset(
    inputDataset: VariantHandle,
    numWorkers: Tensor<Int64>,
    index: Tensor<Int64>,
    autoShardPolicy: Int64 = 0,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalAutoShardDataset", nOutputs)
    op.updateAttribute("auto_shard_policy", autoShardPolicy)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(numWorkers)
    op.addInput(index)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalBytesProducedStatsDataset(
    inputDataset: VariantHandle,
    tag: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalBytesProducedStatsDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(tag)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalCSVDataset<OutputTypes: TensorArrayProtocol>(
    filenames: StringTensor,
    compressionType: StringTensor,
    bufferSize: Tensor<Int64>,
    header: Tensor<Bool>,
    fieldDelim: StringTensor,
    useQuoteDelim: Tensor<Bool>,
    naValue: StringTensor,
    selectCols: Tensor<Int64>,
    recordDefaults: OutputTypes,
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalCSVDataset", nOutputs)
    op.updateAttribute("output_types", recordDefaults._typeList)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(filenames)
    op.addInput(compressionType)
    op.addInput(bufferSize)
    op.addInput(header)
    op.addInput(fieldDelim)
    op.addInput(useQuoteDelim)
    op.addInput(naValue)
    op.addInput(selectCols)
    op.addInputList(recordDefaults)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalChooseFastestDataset(
    inputDatasets: [VariantHandle],
    numExperiments: Int64,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalChooseFastestDataset", nOutputs)
    op.updateAttribute("N", inputDatasets.count)
    op.updateAttribute("num_experiments", numExperiments)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInputList(inputDatasets)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalDatasetCardinality(
    inputDataset: VariantHandle
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalDatasetCardinality", nOutputs)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalDatasetToTFRecord(
    inputDataset: VariantHandle,
    filename: StringTensor,
    compressionType: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("ExperimentalDatasetToTFRecord", nOutputs)
    op.addInput(inputDataset)
    op.addInput(filename)
    op.addInput(compressionType)
    op.execute()
}

@inlinable @inline(__always)
public static func experimentalDenseToSparseBatchDataset(
    inputDataset: VariantHandle,
    batchSize: Tensor<Int64>,
    rowShape: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalDenseToSparseBatchDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(batchSize)
    op.addInput(rowShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalDirectedInterleaveDataset(
    selectorInputDataset: VariantHandle,
    dataInputDatasets: [VariantHandle],
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalDirectedInterleaveDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("N", dataInputDatasets.count)
    op.addInput(selectorInputDataset)
    op.addInputList(dataInputDatasets)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalGroupByReducerDataset<
    KeyfuncIn: TensorGroup,
    KeyfuncOut: TensorGroup,
    InitfuncIn: TensorGroup,
    InitfuncOut: TensorGroup,
    ReducefuncIn: TensorGroup,
    ReducefuncOut: TensorGroup,
    FinalizefuncIn: TensorGroup,
    FinalizefuncOut: TensorGroup,
    TkeyFuncOtherArguments: TensorArrayProtocol,
    TinitFuncOtherArguments: TensorArrayProtocol,
    TreduceFuncOtherArguments: TensorArrayProtocol,
    TfinalizeFuncOtherArguments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    keyFuncOtherArguments: TkeyFuncOtherArguments,
    initFuncOtherArguments: TinitFuncOtherArguments,
    reduceFuncOtherArguments: TreduceFuncOtherArguments,
    finalizeFuncOtherArguments: TfinalizeFuncOtherArguments,
    keyFunc: (KeyfuncIn) -> KeyfuncOut,
    initFunc: (InitfuncIn) -> InitfuncOut,
    reduceFunc: (ReducefuncIn) -> ReducefuncOut,
    finalizeFunc: (FinalizefuncIn) -> FinalizefuncOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalGroupByReducerDataset", nOutputs)
    op.updateAttribute("key_func", keyFunc)
    op.updateAttribute("init_func", initFunc)
    op.updateAttribute("reduce_func", reduceFunc)
    op.updateAttribute("finalize_func", finalizeFunc)
    op.updateAttribute("Tkey_func_other_arguments", keyFuncOtherArguments._typeList)
    op.updateAttribute("Tinit_func_other_arguments", initFuncOtherArguments._typeList)
    op.updateAttribute("Treduce_func_other_arguments", reduceFuncOtherArguments._typeList)
    op.updateAttribute("Tfinalize_func_other_arguments", finalizeFuncOtherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInputList(keyFuncOtherArguments)
    op.addInputList(initFuncOtherArguments)
    op.addInputList(reduceFuncOtherArguments)
    op.addInputList(finalizeFuncOtherArguments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalGroupByWindowDataset<
    KeyfuncIn: TensorGroup,
    KeyfuncOut: TensorGroup,
    ReducefuncIn: TensorGroup,
    ReducefuncOut: TensorGroup,
    WindowsizefuncIn: TensorGroup,
    WindowsizefuncOut: TensorGroup,
    TkeyFuncOtherArguments: TensorArrayProtocol,
    TreduceFuncOtherArguments: TensorArrayProtocol,
    TwindowSizeFuncOtherArguments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    keyFuncOtherArguments: TkeyFuncOtherArguments,
    reduceFuncOtherArguments: TreduceFuncOtherArguments,
    windowSizeFuncOtherArguments: TwindowSizeFuncOtherArguments,
    keyFunc: (KeyfuncIn) -> KeyfuncOut,
    reduceFunc: (ReducefuncIn) -> ReducefuncOut,
    windowSizeFunc: (WindowsizefuncIn) -> WindowsizefuncOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalGroupByWindowDataset", nOutputs)
    op.updateAttribute("key_func", keyFunc)
    op.updateAttribute("reduce_func", reduceFunc)
    op.updateAttribute("window_size_func", windowSizeFunc)
    op.updateAttribute("Tkey_func_other_arguments", keyFuncOtherArguments._typeList)
    op.updateAttribute("Treduce_func_other_arguments", reduceFuncOtherArguments._typeList)
    op.updateAttribute("Twindow_size_func_other_arguments", windowSizeFuncOtherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInputList(keyFuncOtherArguments)
    op.addInputList(reduceFuncOtherArguments)
    op.addInputList(windowSizeFuncOtherArguments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalIgnoreErrorsDataset(
    inputDataset: VariantHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalIgnoreErrorsDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalIteratorGetDevice(
    resource: ResourceHandle
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalIteratorGetDevice", nOutputs)
    op.addInput(resource)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalLMDBDataset(
    filenames: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalLMDBDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(filenames)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalLatencyStatsDataset(
    inputDataset: VariantHandle,
    tag: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalLatencyStatsDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(tag)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalMapAndBatchDataset<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    otherArguments: Targuments,
    batchSize: Tensor<Int64>,
    numParallelCalls: Tensor<Int64>,
    dropRemainder: Tensor<Bool>,
    f: (FIn) -> FOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    preserveCardinality: Bool = false
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalMapAndBatchDataset", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("preserve_cardinality", preserveCardinality)
    op.addInput(inputDataset)
    op.addInputList(otherArguments)
    op.addInput(batchSize)
    op.addInput(numParallelCalls)
    op.addInput(dropRemainder)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalMapDataset<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    otherArguments: Targuments,
    f: (FIn) -> FOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    useInterOpParallelism: Bool = true,
    preserveCardinality: Bool = false
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalMapDataset", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("use_inter_op_parallelism", useInterOpParallelism)
    op.updateAttribute("preserve_cardinality", preserveCardinality)
    op.addInput(inputDataset)
    op.addInputList(otherArguments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalMatchingFilesDataset(
    patterns: StringTensor
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalMatchingFilesDataset", nOutputs)
    op.addInput(patterns)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalMaxIntraOpParallelismDataset(
    inputDataset: VariantHandle,
    maxIntraOpParallelism: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalMaxIntraOpParallelismDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(maxIntraOpParallelism)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalNonSerializableDataset(
    inputDataset: VariantHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalNonSerializableDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalParallelInterleaveDataset<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    otherArguments: Targuments,
    cycleLength: Tensor<Int64>,
    blockLength: Tensor<Int64>,
    sloppy: Tensor<Bool>,
    bufferOutputElements: Tensor<Int64>,
    prefetchInputElements: Tensor<Int64>,
    f: (FIn) -> FOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalParallelInterleaveDataset", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInputList(otherArguments)
    op.addInput(cycleLength)
    op.addInput(blockLength)
    op.addInput(sloppy)
    op.addInput(bufferOutputElements)
    op.addInput(prefetchInputElements)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalParseExampleDataset<Tdense: TensorArrayProtocol>(
    inputDataset: VariantHandle,
    numParallelCalls: Tensor<Int64>,
    denseDefaults: Tdense,
    sparseKeys: [String],
    denseKeys: [String],
    sparseTypes: [TensorDataType],
    denseShapes: [TensorShape?],
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    sloppy: Bool = false
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalParseExampleDataset", nOutputs)
    op.updateAttribute("sparse_keys", sparseKeys)
    op.updateAttribute("dense_keys", denseKeys)
    op.updateAttribute("sparse_types", sparseTypes)
    op.updateAttribute("Tdense", denseDefaults._typeList)
    op.updateAttribute("dense_shapes", denseShapes)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("sloppy", sloppy)
    op.addInput(inputDataset)
    op.addInput(numParallelCalls)
    op.addInputList(denseDefaults)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalPrivateThreadPoolDataset(
    inputDataset: VariantHandle,
    numThreads: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalPrivateThreadPoolDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(numThreads)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalRandomDataset(
    seed: Tensor<Int64>,
    seed2: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalRandomDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(seed)
    op.addInput(seed2)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalRebatchDataset(
    inputDataset: VariantHandle,
    numReplicas: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    useFallback: Bool = true
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalRebatchDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("use_fallback", useFallback)
    op.addInput(inputDataset)
    op.addInput(numReplicas)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalScanDataset<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Tstate: TensorArrayProtocol,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    initialState: Tstate,
    otherArguments: Targuments,
    f: (FIn) -> FOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    preserveCardinality: Bool = false
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalScanDataset", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("Tstate", initialState._typeList)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("preserve_cardinality", preserveCardinality)
    op.addInput(inputDataset)
    op.addInputList(initialState)
    op.addInputList(otherArguments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalSetStatsAggregatorDataset(
    inputDataset: VariantHandle,
    statsAggregator: ResourceHandle,
    tag: StringTensor,
    counterPrefix: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalSetStatsAggregatorDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(statsAggregator)
    op.addInput(tag)
    op.addInput(counterPrefix)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalSleepDataset(
    inputDataset: VariantHandle,
    sleepMicroseconds: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalSleepDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(sleepMicroseconds)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalSlidingWindowDataset(
    inputDataset: VariantHandle,
    windowSize: Tensor<Int64>,
    windowShift: Tensor<Int64>,
    windowStride: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalSlidingWindowDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(windowSize)
    op.addInput(windowShift)
    op.addInput(windowStride)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalSqlDataset(
    driverName: StringTensor,
    dataSourceName: StringTensor,
    query: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalSqlDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(driverName)
    op.addInput(dataSourceName)
    op.addInput(query)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalStatsAggregatorHandle(
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalStatsAggregatorHandle", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalStatsAggregatorSummary(
    iterator: ResourceHandle
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalStatsAggregatorSummary", nOutputs)
    op.addInput(iterator)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalTakeWhileDataset<
    PredicateIn: TensorGroup,
    PredicateOut: TensorGroup,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    otherArguments: Targuments,
    predicate: (PredicateIn) -> PredicateOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalTakeWhileDataset", nOutputs)
    op.updateAttribute("predicate", predicate)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInputList(otherArguments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalThreadPoolDataset(
    inputDataset: VariantHandle,
    threadPool: ResourceHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalThreadPoolDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(threadPool)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalThreadPoolHandle(
    numThreads: Int64,
    maxIntraOpParallelism: Int64 = 1,
    displayName: String,
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalThreadPoolHandle", nOutputs)
    op.updateAttribute("num_threads", numThreads)
    op.updateAttribute("max_intra_op_parallelism", maxIntraOpParallelism)
    op.updateAttribute("display_name", displayName)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalUnbatchDataset(
    inputDataset: VariantHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalUnbatchDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func experimentalUniqueDataset(
    inputDataset: VariantHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ExperimentalUniqueDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func expm1<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Expm1", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func extractGlimpse(
    _ input: Tensor<Float>,
    size: Tensor<Int32>,
    offsets: Tensor<Float>,
    centered: Bool = true,
    normalized: Bool = true,
    uniformNoise: Bool = true,
    noise: String = "uniform"
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("ExtractGlimpse", nOutputs)
    op.updateAttribute("centered", centered)
    op.updateAttribute("normalized", normalized)
    op.updateAttribute("uniform_noise", uniformNoise)
    op.updateAttribute("noise", noise)
    op.addInput(input)
    op.addInput(size)
    op.addInput(offsets)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func extractImagePatches<T: TensorFlowNumeric>(
    images: Tensor<T>,
    ksizes: [Int32],
    strides: [Int32],
    rates: [Int32],
    padding: Padding
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ExtractImagePatches", nOutputs)
    op.updateAttribute("ksizes", ksizes)
    op.updateAttribute("strides", strides)
    op.updateAttribute("rates", rates)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("padding", padding.cName)
    op.addInput(images)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func extractJpegShape<OutputType: TensorFlowIndex>(
    contents: StringTensor
) -> Tensor<OutputType> {
  let nOutputs = Int(1)
    let op = makeOp("ExtractJpegShape", nOutputs)
    op.updateAttribute("output_type", OutputType.tensorFlowDataType)
    op.addInput(contents)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func extractVolumePatches<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    ksizes: [Int32],
    strides: [Int32],
    padding: Padding
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ExtractVolumePatches", nOutputs)
    op.updateAttribute("ksizes", ksizes)
    op.updateAttribute("strides", strides)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("padding", padding.cName)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fFT<Tcomplex: TensorFlowScalar>(
    _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  let nOutputs = Int(1)
    let op = makeOp("FFT", nOutputs)
    op.updateAttribute("Tcomplex", Tcomplex.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fFT2D<Tcomplex: TensorFlowScalar>(
    _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  let nOutputs = Int(1)
    let op = makeOp("FFT2D", nOutputs)
    op.updateAttribute("Tcomplex", Tcomplex.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fFT3D<Tcomplex: TensorFlowScalar>(
    _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  let nOutputs = Int(1)
    let op = makeOp("FFT3D", nOutputs)
    op.updateAttribute("Tcomplex", Tcomplex.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fIFOQueueV2(
    componentTypes: [TensorDataType],
    shapes: [TensorShape?],
    capacity: Int64 = -1,
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("FIFOQueueV2", nOutputs)
    op.updateAttribute("component_types", componentTypes)
    op.updateAttribute("shapes", shapes)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fact(
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("Fact", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fakeParam<Dtype: TensorFlowScalar>(
    shape: TensorShape?
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("FakeParam", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape", shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fakeQuantWithMinMaxArgs(
    inputs: Tensor<Float>,
    min: Double = -6,
    max: Double = 6,
    numBits: Int64 = 8,
    narrowRange: Bool = false
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("FakeQuantWithMinMaxArgs", nOutputs)
    op.updateAttribute("min", min)
    op.updateAttribute("max", max)
    op.updateAttribute("num_bits", numBits)
    op.updateAttribute("narrow_range", narrowRange)
    op.addInput(inputs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fakeQuantWithMinMaxArgsGradient(
    gradients: Tensor<Float>,
    inputs: Tensor<Float>,
    min: Double = -6,
    max: Double = 6,
    numBits: Int64 = 8,
    narrowRange: Bool = false
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("FakeQuantWithMinMaxArgsGradient", nOutputs)
    op.updateAttribute("min", min)
    op.updateAttribute("max", max)
    op.updateAttribute("num_bits", numBits)
    op.updateAttribute("narrow_range", narrowRange)
    op.addInput(gradients)
    op.addInput(inputs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fakeQuantWithMinMaxVars(
    inputs: Tensor<Float>,
    min: Tensor<Float>,
    max: Tensor<Float>,
    numBits: Int64 = 8,
    narrowRange: Bool = false
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("FakeQuantWithMinMaxVars", nOutputs)
    op.updateAttribute("num_bits", numBits)
    op.updateAttribute("narrow_range", narrowRange)
    op.addInput(inputs)
    op.addInput(min)
    op.addInput(max)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fakeQuantWithMinMaxVarsGradient(
    gradients: Tensor<Float>,
    inputs: Tensor<Float>,
    min: Tensor<Float>,
    max: Tensor<Float>,
    numBits: Int64 = 8,
    narrowRange: Bool = false
) -> (backpropsWrtInput: Tensor<Float>, backpropWrtMin: Tensor<Float>, backpropWrtMax: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("FakeQuantWithMinMaxVarsGradient", nOutputs)
    op.updateAttribute("num_bits", numBits)
    op.updateAttribute("narrow_range", narrowRange)
    op.addInput(gradients)
    op.addInput(inputs)
    op.addInput(min)
    op.addInput(max)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func fakeQuantWithMinMaxVarsPerChannel(
    inputs: Tensor<Float>,
    min: Tensor<Float>,
    max: Tensor<Float>,
    numBits: Int64 = 8,
    narrowRange: Bool = false
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("FakeQuantWithMinMaxVarsPerChannel", nOutputs)
    op.updateAttribute("num_bits", numBits)
    op.updateAttribute("narrow_range", narrowRange)
    op.addInput(inputs)
    op.addInput(min)
    op.addInput(max)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fakeQuantWithMinMaxVarsPerChannelGradient(
    gradients: Tensor<Float>,
    inputs: Tensor<Float>,
    min: Tensor<Float>,
    max: Tensor<Float>,
    numBits: Int64 = 8,
    narrowRange: Bool = false
) -> (backpropsWrtInput: Tensor<Float>, backpropWrtMin: Tensor<Float>, backpropWrtMax: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("FakeQuantWithMinMaxVarsPerChannelGradient", nOutputs)
    op.updateAttribute("num_bits", numBits)
    op.updateAttribute("narrow_range", narrowRange)
    op.addInput(gradients)
    op.addInput(inputs)
    op.addInput(min)
    op.addInput(max)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func fill<
    T: TensorFlowScalar,
    IndexType: TensorFlowIndex
>(
    dims: Tensor<IndexType>,
    value: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Fill", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("index_type", IndexType.tensorFlowDataType)
    op.addInput(dims)
    op.addInput(value)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func filterByLastComponentDataset(
    inputDataset: VariantHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("FilterByLastComponentDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func filterDataset<
    PredicateIn: TensorGroup,
    PredicateOut: TensorGroup,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    otherArguments: Targuments,
    predicate: (PredicateIn) -> PredicateOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("FilterDataset", nOutputs)
    op.updateAttribute("predicate", predicate)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInputList(otherArguments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fingerprint<T: TensorFlowScalar>(
    data: Tensor<T>,
    method: StringTensor
) -> Tensor<UInt8> {
  let nOutputs = Int(1)
    let op = makeOp("Fingerprint", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(data)
    op.addInput(method)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fiveFloatOutputs(
) -> (a: Tensor<Float>, b: Tensor<Float>, c: Tensor<Float>, d: Tensor<Float>, e: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("FiveFloatOutputs", nOutputs)
    
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func fixedLengthRecordDataset(
    filenames: StringTensor,
    headerBytes: Tensor<Int64>,
    recordBytes: Tensor<Int64>,
    footerBytes: Tensor<Int64>,
    bufferSize: Tensor<Int64>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("FixedLengthRecordDataset", nOutputs)
    op.addInput(filenames)
    op.addInput(headerBytes)
    op.addInput(recordBytes)
    op.addInput(footerBytes)
    op.addInput(bufferSize)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fixedLengthRecordDatasetV2(
    filenames: StringTensor,
    headerBytes: Tensor<Int64>,
    recordBytes: Tensor<Int64>,
    footerBytes: Tensor<Int64>,
    bufferSize: Tensor<Int64>,
    compressionType: StringTensor
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("FixedLengthRecordDatasetV2", nOutputs)
    op.addInput(filenames)
    op.addInput(headerBytes)
    op.addInput(recordBytes)
    op.addInput(footerBytes)
    op.addInput(bufferSize)
    op.addInput(compressionType)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fixedLengthRecordReaderV2(
    headerBytes: Int64 = 0,
    recordBytes: Int64,
    footerBytes: Int64 = 0,
    hopBytes: Int64 = 0,
    container: String,
    sharedName: String,
    encoding: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("FixedLengthRecordReaderV2", nOutputs)
    op.updateAttribute("header_bytes", headerBytes)
    op.updateAttribute("record_bytes", recordBytes)
    op.updateAttribute("footer_bytes", footerBytes)
    op.updateAttribute("hop_bytes", hopBytes)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("encoding", encoding)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fixedUnigramCandidateSampler(
    trueClasses: Tensor<Int64>,
    numTrue: Int64,
    numSampled: Int64,
    unique: Bool,
    rangeMax: Int64,
    vocabFile: String,
    distortion: Double = 1,
    numReservedIds: Int64 = 0,
    numShards: Int64 = 1,
    shard: Int64 = 0,
    unigrams: [Double],
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> (sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>, sampledExpectedCount: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("FixedUnigramCandidateSampler", nOutputs)
    op.updateAttribute("num_true", numTrue)
    op.updateAttribute("num_sampled", numSampled)
    op.updateAttribute("unique", unique)
    op.updateAttribute("range_max", rangeMax)
    op.updateAttribute("vocab_file", vocabFile)
    op.updateAttribute("distortion", distortion)
    op.updateAttribute("num_reserved_ids", numReservedIds)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard", shard)
    op.updateAttribute("unigrams", unigrams)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.addInput(trueClasses)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func flatMapDataset<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    otherArguments: Targuments,
    f: (FIn) -> FOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("FlatMapDataset", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInputList(otherArguments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func floatInput(
    _ a: Tensor<Float>
) {
  let nOutputs = 0
    let op = makeOp("FloatInput", nOutputs)
    op.addInput(a)
    op.execute()
}

@inlinable @inline(__always)
public static func floatOutput(
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("FloatOutput", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func floatOutputStringOutput(
) -> (a: Tensor<Float>, b: StringTensor) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("FloatOutputStringOutput", nOutputs)
    
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func floor<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Floor", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func floorDiv<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("FloorDiv", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func floorMod<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("FloorMod", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func flushSummaryWriter(
    writer: ResourceHandle
) {
  let nOutputs = 0
    let op = makeOp("FlushSummaryWriter", nOutputs)
    op.addInput(writer)
    op.execute()
}

@inlinable @inline(__always)
public static func foo1(
    _ a: Tensor<Float>,
    _ b: Tensor<Int32>,
    c: Tensor<Int32>
) -> (d: Tensor<Float>, e: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("Foo1", nOutputs)
    op.addInput(a)
    op.addInput(b)
    op.addInput(c)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func foo2(
    _ a: Tensor<Float>,
    _ b: StringTensor,
    c: StringTensor
) -> (d: Tensor<Float>, e: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("Foo2", nOutputs)
    op.addInput(a)
    op.addInput(b)
    op.addInput(c)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func foo3(
    _ a: Tensor<Float>,
    _ b: StringTensor,
    c: Tensor<Float>
) -> (d: Tensor<Float>, e: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("Foo3", nOutputs)
    op.addInput(a)
    op.addInput(b)
    op.addInput(c)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func for_<
    T: TensorArrayProtocol,
    BodyIn: TensorGroup,
    BodyOut: TensorGroup
>(
    start: Tensor<Int32>,
    limit: Tensor<Int32>,
    delta: Tensor<Int32>,
    _ input: T,
    body: (BodyIn) -> BodyOut
) -> T {
  let nOutputs = Int(input._typeList.count)
    let op = makeOp("For", nOutputs)
    op.updateAttribute("T", input._typeList)
    op.updateAttribute("body", body)
    op.addInput(start)
    op.addInput(limit)
    op.addInput(delta)
    op.addInputList(input)
    return op.execute(Int(input._typeList.count))
}

@inlinable @inline(__always)
public static func fractionalAvgPool<T: TensorFlowNumeric>(
    value: Tensor<T>,
    poolingRatio: [Double],
    pseudoRandom: Bool = false,
    overlapping: Bool = false,
    deterministic: Bool = false,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> (output: Tensor<T>, rowPoolingSequence: Tensor<Int64>, colPoolingSequence: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("FractionalAvgPool", nOutputs)
    op.updateAttribute("pooling_ratio", poolingRatio)
    op.updateAttribute("pseudo_random", pseudoRandom)
    op.updateAttribute("overlapping", overlapping)
    op.updateAttribute("deterministic", deterministic)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(value)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func fractionalAvgPoolGrad<T: TensorFlowNumeric>(
    origInputTensorShape: Tensor<Int64>,
    outBackprop: Tensor<T>,
    rowPoolingSequence: Tensor<Int64>,
    colPoolingSequence: Tensor<Int64>,
    overlapping: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("FractionalAvgPoolGrad", nOutputs)
    op.updateAttribute("overlapping", overlapping)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(origInputTensorShape)
    op.addInput(outBackprop)
    op.addInput(rowPoolingSequence)
    op.addInput(colPoolingSequence)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fractionalMaxPool<T: TensorFlowNumeric>(
    value: Tensor<T>,
    poolingRatio: [Double],
    pseudoRandom: Bool = false,
    overlapping: Bool = false,
    deterministic: Bool = false,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> (output: Tensor<T>, rowPoolingSequence: Tensor<Int64>, colPoolingSequence: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("FractionalMaxPool", nOutputs)
    op.updateAttribute("pooling_ratio", poolingRatio)
    op.updateAttribute("pseudo_random", pseudoRandom)
    op.updateAttribute("overlapping", overlapping)
    op.updateAttribute("deterministic", deterministic)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(value)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func fractionalMaxPoolGrad<T: TensorFlowNumeric>(
    origInput: Tensor<T>,
    origOutput: Tensor<T>,
    outBackprop: Tensor<T>,
    rowPoolingSequence: Tensor<Int64>,
    colPoolingSequence: Tensor<Int64>,
    overlapping: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("FractionalMaxPoolGrad", nOutputs)
    op.updateAttribute("overlapping", overlapping)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(origInput)
    op.addInput(origOutput)
    op.addInput(outBackprop)
    op.addInput(rowPoolingSequence)
    op.addInput(colPoolingSequence)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func funcAttr<FIn: TensorGroup,
    FOut: TensorGroup>(
    f: (FIn) -> FOut
) {
  let nOutputs = 0
    let op = makeOp("FuncAttr", nOutputs)
    op.updateAttribute("f", f)
    op.execute()
}

@inlinable @inline(__always)
public static func fusedBatchNorm<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>,
    scale: Tensor<T>,
    offset: Tensor<T>,
    mean: Tensor<T>,
    variance: Tensor<T>,
    epsilon: Double = 0.0001,
    dataFormat: DataFormat = .nhwc,
    isTraining: Bool = true
) -> (y: Tensor<T>, batchMean: Tensor<T>, batchVariance: Tensor<T>, reserveSpace1: Tensor<T>, reserveSpace2: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("FusedBatchNorm", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("epsilon", epsilon)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("is_training", isTraining)
    op.addInput(x)
    op.addInput(scale)
    op.addInput(offset)
    op.addInput(mean)
    op.addInput(variance)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func fusedBatchNormGrad<T: FloatingPoint & TensorFlowScalar>(
    yBackprop: Tensor<T>,
    _ x: Tensor<T>,
    scale: Tensor<T>,
    reserveSpace1: Tensor<T>,
    reserveSpace2: Tensor<T>,
    epsilon: Double = 0.0001,
    dataFormat: DataFormat = .nhwc,
    isTraining: Bool = true
) -> (xBackprop: Tensor<T>, scaleBackprop: Tensor<T>, offsetBackprop: Tensor<T>, reserveSpace3: Tensor<T>, reserveSpace4: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("FusedBatchNormGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("epsilon", epsilon)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("is_training", isTraining)
    op.addInput(yBackprop)
    op.addInput(x)
    op.addInput(scale)
    op.addInput(reserveSpace1)
    op.addInput(reserveSpace2)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func fusedBatchNormGradV2<
    T: FloatingPoint & TensorFlowScalar,
    U: FloatingPoint & TensorFlowScalar
>(
    yBackprop: Tensor<T>,
    _ x: Tensor<T>,
    scale: Tensor<Float>,
    reserveSpace1: Tensor<U>,
    reserveSpace2: Tensor<U>,
    epsilon: Double = 0.0001,
    dataFormat: DataFormat = .nhwc,
    isTraining: Bool = true
) -> (xBackprop: Tensor<T>, scaleBackprop: Tensor<U>, offsetBackprop: Tensor<U>, reserveSpace3: Tensor<U>, reserveSpace4: Tensor<U>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("FusedBatchNormGradV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("U", U.tensorFlowDataType)
    op.updateAttribute("epsilon", epsilon)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("is_training", isTraining)
    op.addInput(yBackprop)
    op.addInput(x)
    op.addInput(scale)
    op.addInput(reserveSpace1)
    op.addInput(reserveSpace2)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func fusedBatchNormGradV3<
    T: FloatingPoint & TensorFlowScalar,
    U: FloatingPoint & TensorFlowScalar
>(
    yBackprop: Tensor<T>,
    _ x: Tensor<T>,
    scale: Tensor<Float>,
    reserveSpace1: Tensor<U>,
    reserveSpace2: Tensor<U>,
    reserveSpace3: Tensor<U>,
    epsilon: Double = 0.0001,
    dataFormat: DataFormat = .nhwc,
    isTraining: Bool = true
) -> (xBackprop: Tensor<T>, scaleBackprop: Tensor<U>, offsetBackprop: Tensor<U>, reserveSpace4: Tensor<U>, reserveSpace5: Tensor<U>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("FusedBatchNormGradV3", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("U", U.tensorFlowDataType)
    op.updateAttribute("epsilon", epsilon)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("is_training", isTraining)
    op.addInput(yBackprop)
    op.addInput(x)
    op.addInput(scale)
    op.addInput(reserveSpace1)
    op.addInput(reserveSpace2)
    op.addInput(reserveSpace3)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func fusedBatchNormV2<
    T: FloatingPoint & TensorFlowScalar,
    U: FloatingPoint & TensorFlowScalar
>(
    _ x: Tensor<T>,
    scale: Tensor<U>,
    offset: Tensor<U>,
    mean: Tensor<U>,
    variance: Tensor<U>,
    epsilon: Double = 0.0001,
    dataFormat: DataFormat = .nhwc,
    isTraining: Bool = true
) -> (y: Tensor<T>, batchMean: Tensor<U>, batchVariance: Tensor<U>, reserveSpace1: Tensor<U>, reserveSpace2: Tensor<U>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("FusedBatchNormV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("U", U.tensorFlowDataType)
    op.updateAttribute("epsilon", epsilon)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("is_training", isTraining)
    op.addInput(x)
    op.addInput(scale)
    op.addInput(offset)
    op.addInput(mean)
    op.addInput(variance)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func fusedBatchNormV3<
    T: FloatingPoint & TensorFlowScalar,
    U: FloatingPoint & TensorFlowScalar
>(
    _ x: Tensor<T>,
    scale: Tensor<U>,
    offset: Tensor<U>,
    mean: Tensor<U>,
    variance: Tensor<U>,
    epsilon: Double = 0.0001,
    dataFormat: DataFormat = .nhwc,
    isTraining: Bool = true
) -> (y: Tensor<T>, batchMean: Tensor<U>, batchVariance: Tensor<U>, reserveSpace1: Tensor<U>, reserveSpace2: Tensor<U>, reserveSpace3: Tensor<U>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("FusedBatchNormV3", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("U", U.tensorFlowDataType)
    op.updateAttribute("epsilon", epsilon)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("is_training", isTraining)
    op.addInput(x)
    op.addInput(scale)
    op.addInput(offset)
    op.addInput(mean)
    op.addInput(variance)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func fusedPadConv2D<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    paddings: Tensor<Int32>,
    filter: Tensor<T>,
    mode: Mode6,
    strides: [Int32],
    padding: Padding
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("FusedPadConv2D", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("mode", mode.cName)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.addInput(input)
    op.addInput(paddings)
    op.addInput(filter)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func fusedResizeAndPadConv2D<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    size: Tensor<Int32>,
    paddings: Tensor<Int32>,
    filter: Tensor<T>,
    resizeAlignCorners: Bool = false,
    mode: Mode6,
    strides: [Int32],
    padding: Padding
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("FusedResizeAndPadConv2D", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("resize_align_corners", resizeAlignCorners)
    op.updateAttribute("mode", mode.cName)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.addInput(input)
    op.addInput(size)
    op.addInput(paddings)
    op.addInput(filter)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func gRUBlockCell<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>,
    hPrev: Tensor<T>,
    wRu: Tensor<T>,
    wC: Tensor<T>,
    bRu: Tensor<T>,
    bC: Tensor<T>
) -> (r: Tensor<T>, u: Tensor<T>, c: Tensor<T>, h: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("GRUBlockCell", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(hPrev)
    op.addInput(wRu)
    op.addInput(wC)
    op.addInput(bRu)
    op.addInput(bC)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func gRUBlockCellGrad<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>,
    hPrev: Tensor<T>,
    wRu: Tensor<T>,
    wC: Tensor<T>,
    bRu: Tensor<T>,
    bC: Tensor<T>,
    r: Tensor<T>,
    u: Tensor<T>,
    c: Tensor<T>,
    dH: Tensor<T>
) -> (dX: Tensor<T>, dHPrev: Tensor<T>, dCBar: Tensor<T>, dRBarUBar: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("GRUBlockCellGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(hPrev)
    op.addInput(wRu)
    op.addInput(wC)
    op.addInput(bRu)
    op.addInput(bC)
    op.addInput(r)
    op.addInput(u)
    op.addInput(c)
    op.addInput(dH)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func gather<
    Tparams: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    params: Tensor<Tparams>,
    indices: Tensor<Tindices>,
    validateIndices: Bool = true
) -> Tensor<Tparams> {
  let nOutputs = Int(1)
    let op = makeOp("Gather", nOutputs)
    op.updateAttribute("validate_indices", validateIndices)
    op.updateAttribute("Tparams", Tparams.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(params)
    op.addInput(indices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func gatherNd<
    Tparams: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    params: Tensor<Tparams>,
    indices: Tensor<Tindices>
) -> Tensor<Tparams> {
  let nOutputs = Int(1)
    let op = makeOp("GatherNd", nOutputs)
    op.updateAttribute("Tparams", Tparams.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(params)
    op.addInput(indices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func gatherV2<
    Tparams: TensorFlowScalar,
    Tindices: TensorFlowIndex,
    Taxis: TensorFlowIndex
>(
    params: Tensor<Tparams>,
    indices: Tensor<Tindices>,
    axis: Tensor<Taxis>,
    batchDims: Int64 = 0
) -> Tensor<Tparams> {
  let nOutputs = Int(1)
    let op = makeOp("GatherV2", nOutputs)
    op.updateAttribute("batch_dims", batchDims)
    op.updateAttribute("Tparams", Tparams.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("Taxis", Taxis.tensorFlowDataType)
    op.addInput(params)
    op.addInput(indices)
    op.addInput(axis)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func generateBoundingBoxProposals(
    scores: Tensor<Float>,
    bboxDeltas: Tensor<Float>,
    imageInfo: Tensor<Float>,
    anchors: Tensor<Float>,
    nmsThreshold: Tensor<Float>,
    preNmsTopn: Tensor<Int32>,
    minSize: Tensor<Float>,
    postNmsTopn: Int64 = 300
) -> (rois: Tensor<Float>, roiProbabilities: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("GenerateBoundingBoxProposals", nOutputs)
    op.updateAttribute("post_nms_topn", postNmsTopn)
    op.addInput(scores)
    op.addInput(bboxDeltas)
    op.addInput(imageInfo)
    op.addInput(anchors)
    op.addInput(nmsThreshold)
    op.addInput(preNmsTopn)
    op.addInput(minSize)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func generateVocabRemapping(
    newVocabFile: StringTensor,
    oldVocabFile: StringTensor,
    newVocabOffset: Int64,
    numNewVocab: Int64,
    oldVocabSize: Int64 = -1
) -> (remapping: Tensor<Int64>, numPresent: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("GenerateVocabRemapping", nOutputs)
    op.updateAttribute("new_vocab_offset", newVocabOffset)
    op.updateAttribute("num_new_vocab", numNewVocab)
    op.updateAttribute("old_vocab_size", oldVocabSize)
    op.addInput(newVocabFile)
    op.addInput(oldVocabFile)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func generatorDataset<
    InitfuncIn: TensorGroup,
    InitfuncOut: TensorGroup,
    NextfuncIn: TensorGroup,
    NextfuncOut: TensorGroup,
    FinalizefuncIn: TensorGroup,
    FinalizefuncOut: TensorGroup,
    TinitFuncArgs: TensorArrayProtocol,
    TnextFuncArgs: TensorArrayProtocol,
    TfinalizeFuncArgs: TensorArrayProtocol
>(
    initFuncOtherArgs: TinitFuncArgs,
    nextFuncOtherArgs: TnextFuncArgs,
    finalizeFuncOtherArgs: TfinalizeFuncArgs,
    initFunc: (InitfuncIn) -> InitfuncOut,
    nextFunc: (NextfuncIn) -> NextfuncOut,
    finalizeFunc: (FinalizefuncIn) -> FinalizefuncOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("GeneratorDataset", nOutputs)
    op.updateAttribute("init_func", initFunc)
    op.updateAttribute("next_func", nextFunc)
    op.updateAttribute("finalize_func", finalizeFunc)
    op.updateAttribute("Tinit_func_args", initFuncOtherArgs._typeList)
    op.updateAttribute("Tnext_func_args", nextFuncOtherArgs._typeList)
    op.updateAttribute("Tfinalize_func_args", finalizeFuncOtherArgs._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInputList(initFuncOtherArgs)
    op.addInputList(nextFuncOtherArgs)
    op.addInputList(finalizeFuncOtherArgs)
    return op.execute(Int(1))
}

/// Returns calibration data for the given resource name
@inlinable @inline(__always)
public static func getCalibrationDataOp(
    resourceName: StringTensor
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("GetCalibrationDataOp", nOutputs)
    op.addInput(resourceName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func getSessionHandle<T: TensorFlowScalar>(
    value: Tensor<T>
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("GetSessionHandle", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(value)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func getSessionHandleV2<T: TensorFlowScalar>(
    value: Tensor<T>
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("GetSessionHandleV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(value)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func getSessionTensor<Dtype: TensorFlowScalar>(
    handle: StringTensor
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("GetSessionTensor", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(handle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func graphDefVersion(
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("GraphDefVersion", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func greater<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("Greater", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func greaterEqual<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("GreaterEqual", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func groupByReducerDataset<
    KeyfuncIn: TensorGroup,
    KeyfuncOut: TensorGroup,
    InitfuncIn: TensorGroup,
    InitfuncOut: TensorGroup,
    ReducefuncIn: TensorGroup,
    ReducefuncOut: TensorGroup,
    FinalizefuncIn: TensorGroup,
    FinalizefuncOut: TensorGroup,
    TkeyFuncOtherArguments: TensorArrayProtocol,
    TinitFuncOtherArguments: TensorArrayProtocol,
    TreduceFuncOtherArguments: TensorArrayProtocol,
    TfinalizeFuncOtherArguments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    keyFuncOtherArguments: TkeyFuncOtherArguments,
    initFuncOtherArguments: TinitFuncOtherArguments,
    reduceFuncOtherArguments: TreduceFuncOtherArguments,
    finalizeFuncOtherArguments: TfinalizeFuncOtherArguments,
    keyFunc: (KeyfuncIn) -> KeyfuncOut,
    initFunc: (InitfuncIn) -> InitfuncOut,
    reduceFunc: (ReducefuncIn) -> ReducefuncOut,
    finalizeFunc: (FinalizefuncIn) -> FinalizefuncOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("GroupByReducerDataset", nOutputs)
    op.updateAttribute("key_func", keyFunc)
    op.updateAttribute("init_func", initFunc)
    op.updateAttribute("reduce_func", reduceFunc)
    op.updateAttribute("finalize_func", finalizeFunc)
    op.updateAttribute("Tkey_func_other_arguments", keyFuncOtherArguments._typeList)
    op.updateAttribute("Tinit_func_other_arguments", initFuncOtherArguments._typeList)
    op.updateAttribute("Treduce_func_other_arguments", reduceFuncOtherArguments._typeList)
    op.updateAttribute("Tfinalize_func_other_arguments", finalizeFuncOtherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInputList(keyFuncOtherArguments)
    op.addInputList(initFuncOtherArguments)
    op.addInputList(reduceFuncOtherArguments)
    op.addInputList(finalizeFuncOtherArguments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func groupByWindowDataset<
    KeyfuncIn: TensorGroup,
    KeyfuncOut: TensorGroup,
    ReducefuncIn: TensorGroup,
    ReducefuncOut: TensorGroup,
    WindowsizefuncIn: TensorGroup,
    WindowsizefuncOut: TensorGroup,
    TkeyFuncOtherArguments: TensorArrayProtocol,
    TreduceFuncOtherArguments: TensorArrayProtocol,
    TwindowSizeFuncOtherArguments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    keyFuncOtherArguments: TkeyFuncOtherArguments,
    reduceFuncOtherArguments: TreduceFuncOtherArguments,
    windowSizeFuncOtherArguments: TwindowSizeFuncOtherArguments,
    keyFunc: (KeyfuncIn) -> KeyfuncOut,
    reduceFunc: (ReducefuncIn) -> ReducefuncOut,
    windowSizeFunc: (WindowsizefuncIn) -> WindowsizefuncOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("GroupByWindowDataset", nOutputs)
    op.updateAttribute("key_func", keyFunc)
    op.updateAttribute("reduce_func", reduceFunc)
    op.updateAttribute("window_size_func", windowSizeFunc)
    op.updateAttribute("Tkey_func_other_arguments", keyFuncOtherArguments._typeList)
    op.updateAttribute("Treduce_func_other_arguments", reduceFuncOtherArguments._typeList)
    op.updateAttribute("Twindow_size_func_other_arguments", windowSizeFuncOtherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInputList(keyFuncOtherArguments)
    op.addInputList(reduceFuncOtherArguments)
    op.addInputList(windowSizeFuncOtherArguments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func guaranteeConst<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("GuaranteeConst", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func hSVToRGB<T: FloatingPoint & TensorFlowScalar>(
    images: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("HSVToRGB", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(images)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func hashTableV2(
    container: String,
    sharedName: String,
    useNodeNameSharing: Bool = false,
    keyDtype: TensorDataType,
    valueDtype: TensorDataType
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("HashTableV2", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("use_node_name_sharing", useNodeNameSharing)
    op.updateAttribute("key_dtype", keyDtype)
    op.updateAttribute("value_dtype", valueDtype)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func histogramFixedWidth<
    T: TensorFlowNumeric,
    Dtype: TensorFlowIndex
>(
    _ values: Tensor<T>,
    valueRange: Tensor<T>,
    nbins: Tensor<Int32>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("HistogramFixedWidth", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(values)
    op.addInput(valueRange)
    op.addInput(nbins)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func histogramSummary<T: TensorFlowNumeric>(
    tag: StringTensor,
    _ values: Tensor<T>
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("HistogramSummary", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(tag)
    op.addInput(values)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func iFFT<Tcomplex: TensorFlowScalar>(
    _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  let nOutputs = Int(1)
    let op = makeOp("IFFT", nOutputs)
    op.updateAttribute("Tcomplex", Tcomplex.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func iFFT2D<Tcomplex: TensorFlowScalar>(
    _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  let nOutputs = Int(1)
    let op = makeOp("IFFT2D", nOutputs)
    op.updateAttribute("Tcomplex", Tcomplex.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func iFFT3D<Tcomplex: TensorFlowScalar>(
    _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  let nOutputs = Int(1)
    let op = makeOp("IFFT3D", nOutputs)
    op.updateAttribute("Tcomplex", Tcomplex.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func iRFFT<
    Treal: FloatingPoint & TensorFlowScalar,
    Tcomplex: TensorFlowScalar
>(
    _ input: Tensor<Tcomplex>,
    fftLength: Tensor<Int32>
) -> Tensor<Treal> {
  let nOutputs = Int(1)
    let op = makeOp("IRFFT", nOutputs)
    op.updateAttribute("Treal", Treal.tensorFlowDataType)
    op.updateAttribute("Tcomplex", Tcomplex.tensorFlowDataType)
    op.addInput(input)
    op.addInput(fftLength)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func iRFFT2D<
    Treal: FloatingPoint & TensorFlowScalar,
    Tcomplex: TensorFlowScalar
>(
    _ input: Tensor<Tcomplex>,
    fftLength: Tensor<Int32>
) -> Tensor<Treal> {
  let nOutputs = Int(1)
    let op = makeOp("IRFFT2D", nOutputs)
    op.updateAttribute("Treal", Treal.tensorFlowDataType)
    op.updateAttribute("Tcomplex", Tcomplex.tensorFlowDataType)
    op.addInput(input)
    op.addInput(fftLength)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func iRFFT3D<
    Treal: FloatingPoint & TensorFlowScalar,
    Tcomplex: TensorFlowScalar
>(
    _ input: Tensor<Tcomplex>,
    fftLength: Tensor<Int32>
) -> Tensor<Treal> {
  let nOutputs = Int(1)
    let op = makeOp("IRFFT3D", nOutputs)
    op.updateAttribute("Treal", Treal.tensorFlowDataType)
    op.updateAttribute("Tcomplex", Tcomplex.tensorFlowDataType)
    op.addInput(input)
    op.addInput(fftLength)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func identity<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Identity", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func identityN<T: TensorArrayProtocol>(
    _ input: T
) -> T {
  let nOutputs = Int(input._typeList.count)
    let op = makeOp("IdentityN", nOutputs)
    op.updateAttribute("T", input._typeList)
    op.addInputList(input)
    return op.execute(Int(input._typeList.count))
}

@inlinable @inline(__always)
public static func identityReaderV2(
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("IdentityReaderV2", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func if_<
    Tcond: TensorFlowScalar,
    Tin: TensorArrayProtocol,
    Tout: TensorGroup,
    ThenbranchIn: TensorGroup,
    ThenbranchOut: TensorGroup,
    ElsebranchIn: TensorGroup,
    ElsebranchOut: TensorGroup
>(
    cond: Tensor<Tcond>,
    _ input: Tin,
    thenBranch: (ThenbranchIn) -> ThenbranchOut,
    elseBranch: (ElsebranchIn) -> ElsebranchOut,
    outputShapes: [TensorShape?]
) -> Tout {
  let nOutputs = Int(Tout._typeList.count)
    let op = makeOp("If", nOutputs)
    op.updateAttribute("Tcond", Tcond.tensorFlowDataType)
    op.updateAttribute("Tin", input._typeList)
    op.updateAttribute("Tout", Tout._typeList)
    op.updateAttribute("then_branch", thenBranch)
    op.updateAttribute("else_branch", elseBranch)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(cond)
    op.addInputList(input)
    return op.execute(Int(Tout._typeList.count))
}

@inlinable @inline(__always)
public static func igamma<T: FloatingPoint & TensorFlowScalar>(
    _ a: Tensor<T>,
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Igamma", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func igammaGradA<T: FloatingPoint & TensorFlowScalar>(
    _ a: Tensor<T>,
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("IgammaGradA", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func igammac<T: FloatingPoint & TensorFlowScalar>(
    _ a: Tensor<T>,
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Igammac", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func ignoreErrorsDataset(
    inputDataset: VariantHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("IgnoreErrorsDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func imag<
    T: TensorFlowScalar,
    Tout: FloatingPoint & TensorFlowScalar
>(
    _ input: Tensor<T>
) -> Tensor<Tout> {
  let nOutputs = Int(1)
    let op = makeOp("Imag", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tout", Tout.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func immutableConst<Dtype: TensorFlowScalar>(
    shape: TensorShape?,
    memoryRegionName: String
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("ImmutableConst", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape", shape)
    op.updateAttribute("memory_region_name", memoryRegionName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func importEvent(
    writer: ResourceHandle,
    event: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("ImportEvent", nOutputs)
    op.addInput(writer)
    op.addInput(event)
    op.execute()
}

@inlinable @inline(__always)
public static func inPolymorphicTwice<T: TensorFlowScalar>(
    _ a: [Tensor<T>],
    _ b: [Tensor<T>]
) {
  let nOutputs = 0
    let op = makeOp("InPolymorphicTwice", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("N", a.count)
    op.updateAttribute("M", b.count)
    op.addInputList(a)
    op.addInputList(b)
    op.execute()
}

@inlinable @inline(__always)
public static func inTopK<T: TensorFlowIndex>(
    predictions: Tensor<Float>,
    targets: Tensor<T>,
    k: Int64
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("InTopK", nOutputs)
    op.updateAttribute("k", k)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(predictions)
    op.addInput(targets)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func inTopKV2<T: TensorFlowIndex>(
    predictions: Tensor<Float>,
    targets: Tensor<T>,
    k: Tensor<T>
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("InTopKV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(predictions)
    op.addInput(targets)
    op.addInput(k)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func infeedDequeue<Dtype: TensorFlowScalar>(
    shape: TensorShape?
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("InfeedDequeue", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape", shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func infeedDequeueTuple<Dtypes: TensorGroup>(
    shapes: [TensorShape?]
) -> Dtypes {
  let nOutputs = Int(Dtypes._typeList.count)
    let op = makeOp("InfeedDequeueTuple", nOutputs)
    op.updateAttribute("dtypes", Dtypes._typeList)
    op.updateAttribute("shapes", shapes)
    return op.execute(Int(Dtypes._typeList.count))
}

@inlinable @inline(__always)
public static func infeedEnqueue<Dtype: TensorFlowScalar>(
    _ input: Tensor<Dtype>,
    shape: TensorShape?,
    layout: [Int32],
    deviceOrdinal: Int64 = -1
) {
  let nOutputs = 0
    let op = makeOp("InfeedEnqueue", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape", shape)
    op.updateAttribute("layout", layout)
    op.updateAttribute("device_ordinal", deviceOrdinal)
    op.addInput(input)
    op.execute()
}

@inlinable @inline(__always)
public static func infeedEnqueuePrelinearizedBuffer(
    _ input: VariantHandle,
    deviceOrdinal: Int64 = -1
) {
  let nOutputs = 0
    let op = makeOp("InfeedEnqueuePrelinearizedBuffer", nOutputs)
    op.updateAttribute("device_ordinal", deviceOrdinal)
    op.addInput(input)
    op.execute()
}

@inlinable @inline(__always)
public static func infeedEnqueueTuple<Dtypes: TensorArrayProtocol>(
    inputs: Dtypes,
    shapes: [TensorShape?],
    layouts: [Int32],
    deviceOrdinal: Int64 = -1
) {
  let nOutputs = 0
    let op = makeOp("InfeedEnqueueTuple", nOutputs)
    op.updateAttribute("dtypes", inputs._typeList)
    op.updateAttribute("shapes", shapes)
    op.updateAttribute("layouts", layouts)
    op.updateAttribute("device_ordinal", deviceOrdinal)
    op.addInputList(inputs)
    op.execute()
}

@inlinable @inline(__always)
public static func initializeTRTResource(
    resourceHandle: ResourceHandle,
    filename: StringTensor,
    maxCachedEnginesCount: Int64 = 1
) {
  let nOutputs = 0
    let op = makeOp("InitializeTRTResource", nOutputs)
    op.updateAttribute("max_cached_engines_count", maxCachedEnginesCount)
    op.addInput(resourceHandle)
    op.addInput(filename)
    op.execute()
}

@inlinable @inline(__always)
public static func initializeTableFromTextFileV2(
    tableHandle: ResourceHandle,
    filename: StringTensor,
    keyIndex: Int64,
    valueIndex: Int64,
    vocabSize: Int64 = -1,
    delimiter: String = "\t"
) {
  let nOutputs = 0
    let op = makeOp("InitializeTableFromTextFileV2", nOutputs)
    op.updateAttribute("key_index", keyIndex)
    op.updateAttribute("value_index", valueIndex)
    op.updateAttribute("vocab_size", vocabSize)
    op.updateAttribute("delimiter", delimiter)
    op.addInput(tableHandle)
    op.addInput(filename)
    op.execute()
}

@inlinable @inline(__always)
public static func initializeTableV2<
    Tkey: TensorFlowScalar,
    Tval: TensorFlowScalar
>(
    tableHandle: ResourceHandle,
    keys: Tensor<Tkey>,
    _ values: Tensor<Tval>
) {
  let nOutputs = 0
    let op = makeOp("InitializeTableV2", nOutputs)
    op.updateAttribute("Tkey", Tkey.tensorFlowDataType)
    op.updateAttribute("Tval", Tval.tensorFlowDataType)
    op.addInput(tableHandle)
    op.addInput(keys)
    op.addInput(values)
    op.execute()
}

@inlinable @inline(__always)
public static func inplaceAdd<T: TensorFlowScalar>(
    _ x: Tensor<T>,
    i: Tensor<Int32>,
    v: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("InplaceAdd", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(i)
    op.addInput(v)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func inplaceSub<T: TensorFlowScalar>(
    _ x: Tensor<T>,
    i: Tensor<Int32>,
    v: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("InplaceSub", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(i)
    op.addInput(v)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func inplaceUpdate<T: TensorFlowScalar>(
    _ x: Tensor<T>,
    i: Tensor<Int32>,
    v: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("InplaceUpdate", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(i)
    op.addInput(v)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func int64Output(
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("Int64Output", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func intAttr(
    foo: Int64 = 1
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("IntAttr", nOutputs)
    op.updateAttribute("foo", foo)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func intInput(
    _ a: Tensor<Int32>
) {
  let nOutputs = 0
    let op = makeOp("IntInput", nOutputs)
    op.addInput(a)
    op.execute()
}

@inlinable @inline(__always)
public static func intInputFloatInput(
    _ a: Tensor<Int32>,
    _ b: Tensor<Float>
) {
  let nOutputs = 0
    let op = makeOp("IntInputFloatInput", nOutputs)
    op.addInput(a)
    op.addInput(b)
    op.execute()
}

@inlinable @inline(__always)
public static func intInputIntOutput(
    _ a: Tensor<Int32>
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("IntInputIntOutput", nOutputs)
    op.addInput(a)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func intOutput(
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("IntOutput", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func intOutputFloatOutput(
) -> (a: Tensor<Int32>, b: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("IntOutputFloatOutput", nOutputs)
    
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func interleaveDataset<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    otherArguments: Targuments,
    cycleLength: Tensor<Int64>,
    blockLength: Tensor<Int64>,
    f: (FIn) -> FOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("InterleaveDataset", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInputList(otherArguments)
    op.addInput(cycleLength)
    op.addInput(blockLength)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func inv<T: TensorFlowNumeric>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Inv", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func invGrad<T: FloatingPoint & TensorFlowScalar>(
    _ y: Tensor<T>,
    dy: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("InvGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(y)
    op.addInput(dy)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func invert<T: TensorFlowInteger>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Invert", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func invertPermutation<T: TensorFlowIndex>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("InvertPermutation", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func isBoostedTreesEnsembleInitialized(
    treeEnsembleHandle: ResourceHandle
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("IsBoostedTreesEnsembleInitialized", nOutputs)
    op.addInput(treeEnsembleHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func isBoostedTreesQuantileStreamResourceInitialized(
    quantileStreamResourceHandle: ResourceHandle
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("IsBoostedTreesQuantileStreamResourceInitialized", nOutputs)
    op.addInput(quantileStreamResourceHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func isFinite<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("IsFinite", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func isInf<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("IsInf", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func isNan<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("IsNan", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func iterator(
    sharedName: String,
    container: String,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("Iterator", nOutputs)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("container", container)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func iteratorFromStringHandle(
    stringHandle: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("IteratorFromStringHandle", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(stringHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func iteratorFromStringHandleV2(
    stringHandle: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("IteratorFromStringHandleV2", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(stringHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func iteratorGetDevice(
    resource: ResourceHandle
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("IteratorGetDevice", nOutputs)
    op.addInput(resource)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func iteratorGetNext<OutputTypes: TensorGroup>(
    iterator: ResourceHandle,
    outputShapes: [TensorShape?]
) -> OutputTypes {
  let nOutputs = Int(OutputTypes._typeList.count)
    let op = makeOp("IteratorGetNext", nOutputs)
    op.updateAttribute("output_types", OutputTypes._typeList)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(iterator)
    return op.execute(Int(OutputTypes._typeList.count))
}

@inlinable @inline(__always)
public static func iteratorGetNextAsOptional(
    iterator: ResourceHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("IteratorGetNextAsOptional", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(iterator)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func iteratorGetNextSync<OutputTypes: TensorGroup>(
    iterator: ResourceHandle,
    outputShapes: [TensorShape?]
) -> OutputTypes {
  let nOutputs = Int(OutputTypes._typeList.count)
    let op = makeOp("IteratorGetNextSync", nOutputs)
    op.updateAttribute("output_types", OutputTypes._typeList)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(iterator)
    return op.execute(Int(OutputTypes._typeList.count))
}

@inlinable @inline(__always)
public static func iteratorToStringHandle(
    resourceHandle: ResourceHandle
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("IteratorToStringHandle", nOutputs)
    op.addInput(resourceHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func iteratorV2(
    sharedName: String,
    container: String,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("IteratorV2", nOutputs)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("container", container)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func kMC2ChainInitialization(
    distances: Tensor<Float>,
    seed: Tensor<Int64>
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("KMC2ChainInitialization", nOutputs)
    op.addInput(distances)
    op.addInput(seed)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func kernelLabel(
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("KernelLabel", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func kernelLabelRequired(
    _ input: Tensor<Int32>
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("KernelLabelRequired", nOutputs)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func kmeansPlusPlusInitialization(
    points: Tensor<Float>,
    numToSample: Tensor<Int64>,
    seed: Tensor<Int64>,
    numRetriesPerSample: Tensor<Int64>
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("KmeansPlusPlusInitialization", nOutputs)
    op.addInput(points)
    op.addInput(numToSample)
    op.addInput(seed)
    op.addInput(numRetriesPerSample)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func l2Loss<T: FloatingPoint & TensorFlowScalar>(
    t: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("L2Loss", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(t)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func lMDBDataset(
    filenames: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("LMDBDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(filenames)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func lRN<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    depthRadius: Int64 = 5,
    bias: Double = 1,
    alpha: Double = 1,
    beta: Double = 0.5
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("LRN", nOutputs)
    op.updateAttribute("depth_radius", depthRadius)
    op.updateAttribute("bias", bias)
    op.updateAttribute("alpha", alpha)
    op.updateAttribute("beta", beta)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func lRNGrad<T: FloatingPoint & TensorFlowScalar>(
    inputGrads: Tensor<T>,
    inputImage: Tensor<T>,
    outputImage: Tensor<T>,
    depthRadius: Int64 = 5,
    bias: Double = 1,
    alpha: Double = 1,
    beta: Double = 0.5
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("LRNGrad", nOutputs)
    op.updateAttribute("depth_radius", depthRadius)
    op.updateAttribute("bias", bias)
    op.updateAttribute("alpha", alpha)
    op.updateAttribute("beta", beta)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(inputGrads)
    op.addInput(inputImage)
    op.addInput(outputImage)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func lSTMBlockCell<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>,
    csPrev: Tensor<T>,
    hPrev: Tensor<T>,
    w: Tensor<T>,
    wci: Tensor<T>,
    wcf: Tensor<T>,
    wco: Tensor<T>,
    _ b: Tensor<T>,
    forgetBias: Double = 1,
    cellClip: Double = 3,
    usePeephole: Bool = false
) -> (i: Tensor<T>, cs: Tensor<T>, f: Tensor<T>, o: Tensor<T>, ci: Tensor<T>, co: Tensor<T>, h: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("LSTMBlockCell", nOutputs)
    op.updateAttribute("forget_bias", forgetBias)
    op.updateAttribute("cell_clip", cellClip)
    op.updateAttribute("use_peephole", usePeephole)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(csPrev)
    op.addInput(hPrev)
    op.addInput(w)
    op.addInput(wci)
    op.addInput(wcf)
    op.addInput(wco)
    op.addInput(b)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func lSTMBlockCellGrad<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>,
    csPrev: Tensor<T>,
    hPrev: Tensor<T>,
    w: Tensor<T>,
    wci: Tensor<T>,
    wcf: Tensor<T>,
    wco: Tensor<T>,
    _ b: Tensor<T>,
    i: Tensor<T>,
    cs: Tensor<T>,
    f: Tensor<T>,
    o: Tensor<T>,
    ci: Tensor<T>,
    co: Tensor<T>,
    csGrad: Tensor<T>,
    hGrad: Tensor<T>,
    usePeephole: Bool
) -> (csPrevGrad: Tensor<T>, dicfo: Tensor<T>, wciGrad: Tensor<T>, wcfGrad: Tensor<T>, wcoGrad: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("LSTMBlockCellGrad", nOutputs)
    op.updateAttribute("use_peephole", usePeephole)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(csPrev)
    op.addInput(hPrev)
    op.addInput(w)
    op.addInput(wci)
    op.addInput(wcf)
    op.addInput(wco)
    op.addInput(b)
    op.addInput(i)
    op.addInput(cs)
    op.addInput(f)
    op.addInput(o)
    op.addInput(ci)
    op.addInput(co)
    op.addInput(csGrad)
    op.addInput(hGrad)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func latencyStatsDataset(
    inputDataset: VariantHandle,
    tag: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("LatencyStatsDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(tag)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func leakyRelu<T: FloatingPoint & TensorFlowScalar>(
    features: Tensor<T>,
    alpha: Double = 0.2
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("LeakyRelu", nOutputs)
    op.updateAttribute("alpha", alpha)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(features)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func leakyReluGrad<T: FloatingPoint & TensorFlowScalar>(
    gradients: Tensor<T>,
    features: Tensor<T>,
    alpha: Double = 0.2
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("LeakyReluGrad", nOutputs)
    op.updateAttribute("alpha", alpha)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(gradients)
    op.addInput(features)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func learnedUnigramCandidateSampler(
    trueClasses: Tensor<Int64>,
    numTrue: Int64,
    numSampled: Int64,
    unique: Bool,
    rangeMax: Int64,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> (sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>, sampledExpectedCount: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("LearnedUnigramCandidateSampler", nOutputs)
    op.updateAttribute("num_true", numTrue)
    op.updateAttribute("num_sampled", numSampled)
    op.updateAttribute("unique", unique)
    op.updateAttribute("range_max", rangeMax)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.addInput(trueClasses)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func leftShift<T: TensorFlowInteger>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("LeftShift", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func less<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("Less", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func lessEqual<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("LessEqual", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func lgamma<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Lgamma", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func linSpace<
    T: FloatingPoint & TensorFlowScalar,
    Tidx: TensorFlowIndex
>(
    start: Tensor<T>,
    stop: Tensor<T>,
    num: Tensor<Tidx>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("LinSpace", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(start)
    op.addInput(stop)
    op.addInput(num)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func listDiff<
    T: TensorFlowScalar,
    OutIdx: TensorFlowIndex
>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> (out: Tensor<T>, idx: Tensor<OutIdx>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("ListDiff", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("out_idx", OutIdx.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func listInput<T: TensorFlowScalar>(
    _ a: [Tensor<T>]
) {
  let nOutputs = 0
    let op = makeOp("ListInput", nOutputs)
    op.updateAttribute("N", a.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInputList(a)
    op.execute()
}

@inlinable @inline(__always)
public static func listOutput<T: TensorGroup>(
) -> T {
  let nOutputs = Int(T._typeList.count)
    let op = makeOp("ListOutput", nOutputs)
    op.updateAttribute("T", T._typeList)
    return op.execute(Int(T._typeList.count))
}

@inlinable @inline(__always)
public static func loadAndRemapMatrix(
    ckptPath: StringTensor,
    oldTensorName: StringTensor,
    rowRemapping: Tensor<Int64>,
    colRemapping: Tensor<Int64>,
    initializingValues: Tensor<Float>,
    numRows: Int64,
    numCols: Int64,
    maxRowsInMemory: Int64 = -1
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("LoadAndRemapMatrix", nOutputs)
    op.updateAttribute("num_rows", numRows)
    op.updateAttribute("num_cols", numCols)
    op.updateAttribute("max_rows_in_memory", maxRowsInMemory)
    op.addInput(ckptPath)
    op.addInput(oldTensorName)
    op.addInput(rowRemapping)
    op.addInput(colRemapping)
    op.addInput(initializingValues)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingADAMParameters(
    parameters: Tensor<Float>,
    momenta: Tensor<Float>,
    velocities: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingADAMParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(momenta)
    op.addInput(velocities)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingADAMParametersGradAccumDebug(
    parameters: Tensor<Float>,
    momenta: Tensor<Float>,
    velocities: Tensor<Float>,
    gradientAccumulators: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingADAMParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(momenta)
    op.addInput(velocities)
    op.addInput(gradientAccumulators)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingAdadeltaParameters(
    parameters: Tensor<Float>,
    accumulators: Tensor<Float>,
    updates: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingAdadeltaParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(accumulators)
    op.addInput(updates)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingAdadeltaParametersGradAccumDebug(
    parameters: Tensor<Float>,
    accumulators: Tensor<Float>,
    updates: Tensor<Float>,
    gradientAccumulators: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingAdadeltaParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(accumulators)
    op.addInput(updates)
    op.addInput(gradientAccumulators)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingAdagradParameters(
    parameters: Tensor<Float>,
    accumulators: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingAdagradParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(accumulators)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingAdagradParametersGradAccumDebug(
    parameters: Tensor<Float>,
    accumulators: Tensor<Float>,
    gradientAccumulators: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingAdagradParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(accumulators)
    op.addInput(gradientAccumulators)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingCenteredRMSPropParameters(
    parameters: Tensor<Float>,
    ms: Tensor<Float>,
    mom: Tensor<Float>,
    mg: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingCenteredRMSPropParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(ms)
    op.addInput(mom)
    op.addInput(mg)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingFTRLParameters(
    parameters: Tensor<Float>,
    accumulators: Tensor<Float>,
    linears: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingFTRLParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(accumulators)
    op.addInput(linears)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingFTRLParametersGradAccumDebug(
    parameters: Tensor<Float>,
    accumulators: Tensor<Float>,
    linears: Tensor<Float>,
    gradientAccumulators: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingFTRLParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(accumulators)
    op.addInput(linears)
    op.addInput(gradientAccumulators)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingMDLAdagradLightParameters(
    parameters: Tensor<Float>,
    accumulators: Tensor<Float>,
    weights: Tensor<Float>,
    benefits: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingMDLAdagradLightParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(accumulators)
    op.addInput(weights)
    op.addInput(benefits)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingMomentumParameters(
    parameters: Tensor<Float>,
    momenta: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingMomentumParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(momenta)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingMomentumParametersGradAccumDebug(
    parameters: Tensor<Float>,
    momenta: Tensor<Float>,
    gradientAccumulators: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingMomentumParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(momenta)
    op.addInput(gradientAccumulators)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingProximalAdagradParameters(
    parameters: Tensor<Float>,
    accumulators: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingProximalAdagradParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(accumulators)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingProximalAdagradParametersGradAccumDebug(
    parameters: Tensor<Float>,
    accumulators: Tensor<Float>,
    gradientAccumulators: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingProximalAdagradParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(accumulators)
    op.addInput(gradientAccumulators)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingRMSPropParameters(
    parameters: Tensor<Float>,
    ms: Tensor<Float>,
    mom: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingRMSPropParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(ms)
    op.addInput(mom)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingRMSPropParametersGradAccumDebug(
    parameters: Tensor<Float>,
    ms: Tensor<Float>,
    mom: Tensor<Float>,
    gradientAccumulators: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingRMSPropParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.addInput(ms)
    op.addInput(mom)
    op.addInput(gradientAccumulators)
    op.execute()
}

@inlinable @inline(__always)
public static func loadTPUEmbeddingStochasticGradientDescentParameters(
    parameters: Tensor<Float>,
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) {
  let nOutputs = 0
    let op = makeOp("LoadTPUEmbeddingStochasticGradientDescentParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    op.addInput(parameters)
    op.execute()
}

@inlinable @inline(__always)
public static func log<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Log", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func log1p<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Log1p", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func logMatrixDeterminant<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>
) -> (sign: Tensor<T>, logAbsDeterminant: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("LogMatrixDeterminant", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func logSoftmax<T: FloatingPoint & TensorFlowScalar>(
    logits: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("LogSoftmax", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(logits)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func logUniformCandidateSampler(
    trueClasses: Tensor<Int64>,
    numTrue: Int64,
    numSampled: Int64,
    unique: Bool,
    rangeMax: Int64,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> (sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>, sampledExpectedCount: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("LogUniformCandidateSampler", nOutputs)
    op.updateAttribute("num_true", numTrue)
    op.updateAttribute("num_sampled", numSampled)
    op.updateAttribute("unique", unique)
    op.updateAttribute("range_max", rangeMax)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.addInput(trueClasses)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func logicalAnd(
    _ x: Tensor<Bool>,
    _ y: Tensor<Bool>
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("LogicalAnd", nOutputs)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func logicalNot(
    _ x: Tensor<Bool>
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("LogicalNot", nOutputs)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func logicalOr(
    _ x: Tensor<Bool>,
    _ y: Tensor<Bool>
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("LogicalOr", nOutputs)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func lookupTableExportV2<
    Tkeys: TensorFlowScalar,
    Tvalues: TensorFlowScalar
>(
    tableHandle: ResourceHandle
) -> (keys: Tensor<Tkeys>, values: Tensor<Tvalues>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("LookupTableExportV2", nOutputs)
    op.updateAttribute("Tkeys", Tkeys.tensorFlowDataType)
    op.updateAttribute("Tvalues", Tvalues.tensorFlowDataType)
    op.addInput(tableHandle)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func lookupTableFindV2<
    Tin: TensorFlowScalar,
    Tout: TensorFlowScalar
>(
    tableHandle: ResourceHandle,
    keys: Tensor<Tin>,
    defaultValue: Tensor<Tout>
) -> Tensor<Tout> {
  let nOutputs = Int(1)
    let op = makeOp("LookupTableFindV2", nOutputs)
    op.updateAttribute("Tin", Tin.tensorFlowDataType)
    op.updateAttribute("Tout", Tout.tensorFlowDataType)
    op.addInput(tableHandle)
    op.addInput(keys)
    op.addInput(defaultValue)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func lookupTableImportV2<
    Tin: TensorFlowScalar,
    Tout: TensorFlowScalar
>(
    tableHandle: ResourceHandle,
    keys: Tensor<Tin>,
    _ values: Tensor<Tout>
) {
  let nOutputs = 0
    let op = makeOp("LookupTableImportV2", nOutputs)
    op.updateAttribute("Tin", Tin.tensorFlowDataType)
    op.updateAttribute("Tout", Tout.tensorFlowDataType)
    op.addInput(tableHandle)
    op.addInput(keys)
    op.addInput(values)
    op.execute()
}

@inlinable @inline(__always)
public static func lookupTableInsertV2<
    Tin: TensorFlowScalar,
    Tout: TensorFlowScalar
>(
    tableHandle: ResourceHandle,
    keys: Tensor<Tin>,
    _ values: Tensor<Tout>
) {
  let nOutputs = 0
    let op = makeOp("LookupTableInsertV2", nOutputs)
    op.updateAttribute("Tin", Tin.tensorFlowDataType)
    op.updateAttribute("Tout", Tout.tensorFlowDataType)
    op.addInput(tableHandle)
    op.addInput(keys)
    op.addInput(values)
    op.execute()
}

@inlinable @inline(__always)
public static func lookupTableRemoveV2<Tin: TensorFlowScalar>(
    tableHandle: ResourceHandle,
    keys: Tensor<Tin>
) {
  let nOutputs = 0
    let op = makeOp("LookupTableRemoveV2", nOutputs)
    op.updateAttribute("Tin", Tin.tensorFlowDataType)
    op.addInput(tableHandle)
    op.addInput(keys)
    op.execute()
}

@inlinable @inline(__always)
public static func lookupTableSizeV2(
    tableHandle: ResourceHandle
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("LookupTableSizeV2", nOutputs)
    op.addInput(tableHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func loopCond(
    _ input: Tensor<Bool>
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("LoopCond", nOutputs)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func lowerBound<
    T: TensorFlowScalar,
    OutType: TensorFlowIndex
>(
    sortedInputs: Tensor<T>,
    _ values: Tensor<T>
) -> Tensor<OutType> {
  let nOutputs = Int(1)
    let op = makeOp("LowerBound", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(sortedInputs)
    op.addInput(values)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func lu<
    T: FloatingPoint & TensorFlowScalar,
    OutputIdxType: TensorFlowIndex
>(
    _ input: Tensor<T>
) -> (lu: Tensor<T>, p: Tensor<OutputIdxType>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("Lu", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("output_idx_type", OutputIdxType.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func makeIterator(
    dataset: VariantHandle,
    iterator: ResourceHandle
) {
  let nOutputs = 0
    let op = makeOp("MakeIterator", nOutputs)
    op.addInput(dataset)
    op.addInput(iterator)
    op.execute()
}

@inlinable @inline(__always)
public static func mapAndBatchDataset<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    otherArguments: Targuments,
    batchSize: Tensor<Int64>,
    numParallelCalls: Tensor<Int64>,
    dropRemainder: Tensor<Bool>,
    f: (FIn) -> FOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    preserveCardinality: Bool = false
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("MapAndBatchDataset", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("preserve_cardinality", preserveCardinality)
    op.addInput(inputDataset)
    op.addInputList(otherArguments)
    op.addInput(batchSize)
    op.addInput(numParallelCalls)
    op.addInput(dropRemainder)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mapClear(
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    dtypes: [TensorDataType],
    container: String,
    sharedName: String
) {
  let nOutputs = 0
    let op = makeOp("MapClear", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", dtypes)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.execute()
}

@inlinable @inline(__always)
public static func mapDataset<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    otherArguments: Targuments,
    f: (FIn) -> FOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    useInterOpParallelism: Bool = true,
    preserveCardinality: Bool = false
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("MapDataset", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("use_inter_op_parallelism", useInterOpParallelism)
    op.updateAttribute("preserve_cardinality", preserveCardinality)
    op.addInput(inputDataset)
    op.addInputList(otherArguments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mapDefun<
    Targuments: TensorArrayProtocol,
    Tcaptured: TensorArrayProtocol,
    OutputTypes: TensorGroup,
    FIn: TensorGroup,
    FOut: TensorGroup
>(
    arguments: Targuments,
    capturedInputs: Tcaptured,
    outputShapes: [TensorShape?],
    f: (FIn) -> FOut,
    maxIntraOpParallelism: Int64 = 1
) -> OutputTypes {
  let nOutputs = Int(OutputTypes._typeList.count)
    let op = makeOp("MapDefun", nOutputs)
    op.updateAttribute("Targuments", arguments._typeList)
    op.updateAttribute("Tcaptured", capturedInputs._typeList)
    op.updateAttribute("output_types", OutputTypes._typeList)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("f", f)
    op.updateAttribute("max_intra_op_parallelism", maxIntraOpParallelism)
    op.addInputList(arguments)
    op.addInputList(capturedInputs)
    return op.execute(Int(OutputTypes._typeList.count))
}

@inlinable @inline(__always)
public static func mapIncompleteSize(
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    dtypes: [TensorDataType],
    container: String,
    sharedName: String
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("MapIncompleteSize", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", dtypes)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mapPeek<Dtypes: TensorGroup>(
    key: Tensor<Int64>,
    indices: Tensor<Int32>,
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    container: String,
    sharedName: String
) -> Dtypes {
  let nOutputs = Int(Dtypes._typeList.count)
    let op = makeOp("MapPeek", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", Dtypes._typeList)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.addInput(key)
    op.addInput(indices)
    return op.execute(Int(Dtypes._typeList.count))
}

@inlinable @inline(__always)
public static func mapSize(
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    dtypes: [TensorDataType],
    container: String,
    sharedName: String
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("MapSize", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", dtypes)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mapStage<FakeDtypes: TensorArrayProtocol>(
    key: Tensor<Int64>,
    indices: Tensor<Int32>,
    _ values: FakeDtypes,
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    dtypes: [TensorDataType],
    container: String,
    sharedName: String
) {
  let nOutputs = 0
    let op = makeOp("MapStage", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", dtypes)
    op.updateAttribute("fake_dtypes", values._typeList)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.addInput(key)
    op.addInput(indices)
    op.addInputList(values)
    op.execute()
}

@inlinable @inline(__always)
public static func mapUnstage<Dtypes: TensorGroup>(
    key: Tensor<Int64>,
    indices: Tensor<Int32>,
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    container: String,
    sharedName: String
) -> Dtypes {
  let nOutputs = Int(Dtypes._typeList.count)
    let op = makeOp("MapUnstage", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", Dtypes._typeList)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.addInput(key)
    op.addInput(indices)
    return op.execute(Int(Dtypes._typeList.count))
}

@inlinable @inline(__always)
public static func mapUnstageNoKey<Dtypes: TensorGroup>(
    indices: Tensor<Int32>,
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    container: String,
    sharedName: String
) -> (key: Tensor<Int64>, values: Dtypes) {
  let nOutputs = Int(1) + Int(Dtypes._typeList.count)
    let op = makeOp("MapUnstageNoKey", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", Dtypes._typeList)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.addInput(indices)
    return op.execute(Int(1), Int(Dtypes._typeList.count))
}

@inlinable @inline(__always)
public static func matMul<T: TensorFlowNumeric>(
    _ a: Tensor<T>,
    _ b: Tensor<T>,
    transposeA: Bool = false,
    transposeB: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatMul", nOutputs)
    op.updateAttribute("transpose_a", transposeA)
    op.updateAttribute("transpose_b", transposeB)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    op.addInput(b)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matchingFiles(
    pattern: StringTensor
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("MatchingFiles", nOutputs)
    op.addInput(pattern)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matchingFilesDataset(
    patterns: StringTensor
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("MatchingFilesDataset", nOutputs)
    op.addInput(patterns)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixBandPart<
    T: TensorFlowScalar,
    Tindex: TensorFlowIndex
>(
    _ input: Tensor<T>,
    numLower: Tensor<Tindex>,
    numUpper: Tensor<Tindex>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixBandPart", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindex", Tindex.tensorFlowDataType)
    op.addInput(input)
    op.addInput(numLower)
    op.addInput(numUpper)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixDeterminant<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixDeterminant", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixDiag<T: TensorFlowScalar>(
    diagonal: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixDiag", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(diagonal)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixDiagPart<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixDiagPart", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixDiagPartV2<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    k: Tensor<Int32>,
    paddingValue: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixDiagPartV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    op.addInput(k)
    op.addInput(paddingValue)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixDiagV2<T: TensorFlowScalar>(
    diagonal: Tensor<T>,
    k: Tensor<Int32>,
    numRows: Tensor<Int32>,
    numCols: Tensor<Int32>,
    paddingValue: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixDiagV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(diagonal)
    op.addInput(k)
    op.addInput(numRows)
    op.addInput(numCols)
    op.addInput(paddingValue)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixExponential<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixExponential", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixInverse<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    adjoint: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixInverse", nOutputs)
    op.updateAttribute("adjoint", adjoint)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixLogarithm<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixLogarithm", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixSetDiag<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    diagonal: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixSetDiag", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    op.addInput(diagonal)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixSetDiagV2<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    diagonal: Tensor<T>,
    k: Tensor<Int32>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixSetDiagV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    op.addInput(diagonal)
    op.addInput(k)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixSolve<T: FloatingPoint & TensorFlowScalar>(
    matrix: Tensor<T>,
    rhs: Tensor<T>,
    adjoint: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixSolve", nOutputs)
    op.updateAttribute("adjoint", adjoint)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(matrix)
    op.addInput(rhs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixSolveLs<T: FloatingPoint & TensorFlowScalar>(
    matrix: Tensor<T>,
    rhs: Tensor<T>,
    l2Regularizer: Tensor<Double>,
    fast: Bool = true
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixSolveLs", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("fast", fast)
    op.addInput(matrix)
    op.addInput(rhs)
    op.addInput(l2Regularizer)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixSquareRoot<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixSquareRoot", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func matrixTriangularSolve<T: FloatingPoint & TensorFlowScalar>(
    matrix: Tensor<T>,
    rhs: Tensor<T>,
    lower: Bool = true,
    adjoint: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MatrixTriangularSolve", nOutputs)
    op.updateAttribute("lower", lower)
    op.updateAttribute("adjoint", adjoint)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(matrix)
    op.addInput(rhs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func max<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
>(
    _ input: Tensor<T>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Max", nOutputs)
    op.updateAttribute("keep_dims", keepDims)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(input)
    op.addInput(reductionIndices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func maxIntraOpParallelismDataset(
    inputDataset: VariantHandle,
    maxIntraOpParallelism: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("MaxIntraOpParallelismDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(maxIntraOpParallelism)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func maxPool<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat5 = .nhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MaxPool", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func maxPool3D<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MaxPool3D", nOutputs)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func maxPool3DGrad<
    T: FloatingPoint & TensorFlowScalar,
    Tinput: FloatingPoint & TensorFlowScalar
>(
    origInput: Tensor<Tinput>,
    origOutput: Tensor<Tinput>,
    grad: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MaxPool3DGrad", nOutputs)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("TInput", Tinput.tensorFlowDataType)
    op.addInput(origInput)
    op.addInput(origOutput)
    op.addInput(grad)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func maxPool3DGradGrad<T: TensorFlowNumeric>(
    origInput: Tensor<T>,
    origOutput: Tensor<T>,
    grad: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MaxPool3DGradGrad", nOutputs)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(origInput)
    op.addInput(origOutput)
    op.addInput(grad)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func maxPoolGrad<T: TensorFlowNumeric>(
    origInput: Tensor<T>,
    origOutput: Tensor<T>,
    grad: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MaxPoolGrad", nOutputs)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(origInput)
    op.addInput(origOutput)
    op.addInput(grad)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func maxPoolGradGrad<T: TensorFlowNumeric>(
    origInput: Tensor<T>,
    origOutput: Tensor<T>,
    grad: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MaxPoolGradGrad", nOutputs)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(origInput)
    op.addInput(origOutput)
    op.addInput(grad)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func maxPoolGradGradV2<T: TensorFlowNumeric>(
    origInput: Tensor<T>,
    origOutput: Tensor<T>,
    grad: Tensor<T>,
    ksize: Tensor<Int32>,
    strides: Tensor<Int32>,
    padding: Padding,
    dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MaxPoolGradGradV2", nOutputs)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(origInput)
    op.addInput(origOutput)
    op.addInput(grad)
    op.addInput(ksize)
    op.addInput(strides)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func maxPoolGradGradWithArgmax<
    Targmax: TensorFlowIndex,
    T: TensorFlowNumeric
>(
    _ input: Tensor<T>,
    grad: Tensor<T>,
    argmax: Tensor<Targmax>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    includeBatchInIndex: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MaxPoolGradGradWithArgmax", nOutputs)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("include_batch_in_index", includeBatchInIndex)
    op.updateAttribute("Targmax", Targmax.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    op.addInput(grad)
    op.addInput(argmax)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func maxPoolGradV2<T: TensorFlowNumeric>(
    origInput: Tensor<T>,
    origOutput: Tensor<T>,
    grad: Tensor<T>,
    ksize: Tensor<Int32>,
    strides: Tensor<Int32>,
    padding: Padding,
    dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MaxPoolGradV2", nOutputs)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(origInput)
    op.addInput(origOutput)
    op.addInput(grad)
    op.addInput(ksize)
    op.addInput(strides)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func maxPoolGradWithArgmax<
    Targmax: TensorFlowIndex,
    T: TensorFlowNumeric
>(
    _ input: Tensor<T>,
    grad: Tensor<T>,
    argmax: Tensor<Targmax>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    includeBatchInIndex: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MaxPoolGradWithArgmax", nOutputs)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("include_batch_in_index", includeBatchInIndex)
    op.updateAttribute("Targmax", Targmax.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    op.addInput(grad)
    op.addInput(argmax)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func maxPoolV2<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    ksize: Tensor<Int32>,
    strides: Tensor<Int32>,
    padding: Padding,
    dataFormat: DataFormat5 = .nhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MaxPoolV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("data_format", dataFormat.cName)
    op.addInput(input)
    op.addInput(ksize)
    op.addInput(strides)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func maxPoolWithArgmax<
    Targmax: TensorFlowIndex,
    T: TensorFlowNumeric
>(
    _ input: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    includeBatchInIndex: Bool = false
) -> (output: Tensor<T>, argmax: Tensor<Targmax>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("MaxPoolWithArgmax", nOutputs)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("Targmax", Targmax.tensorFlowDataType)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("include_batch_in_index", includeBatchInIndex)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func maximum<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Maximum", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mean<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
>(
    _ input: Tensor<T>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Mean", nOutputs)
    op.updateAttribute("keep_dims", keepDims)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(input)
    op.addInput(reductionIndices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func merge<T: TensorFlowScalar>(
    inputs: [Tensor<T>]
) -> (output: Tensor<T>, valueIndex: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("Merge", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("N", inputs.count)
    op.addInputList(inputs)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func mergeSummary(
    inputs: [StringTensor]
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("MergeSummary", nOutputs)
    op.updateAttribute("N", inputs.count)
    op.addInputList(inputs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mergeV2Checkpoints(
    checkpointPrefixes: StringTensor,
    destinationPrefix: StringTensor,
    deleteOldDirs: Bool = true
) {
  let nOutputs = 0
    let op = makeOp("MergeV2Checkpoints", nOutputs)
    op.updateAttribute("delete_old_dirs", deleteOldDirs)
    op.addInput(checkpointPrefixes)
    op.addInput(destinationPrefix)
    op.execute()
}

@inlinable @inline(__always)
public static func mfcc(
    spectrogram: Tensor<Float>,
    sampleRate: Tensor<Int32>,
    upperFrequencyLimit: Double = 4000,
    lowerFrequencyLimit: Double = 20,
    filterbankChannelCount: Int64 = 40,
    dctCoefficientCount: Int64 = 13
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("Mfcc", nOutputs)
    op.updateAttribute("upper_frequency_limit", upperFrequencyLimit)
    op.updateAttribute("lower_frequency_limit", lowerFrequencyLimit)
    op.updateAttribute("filterbank_channel_count", filterbankChannelCount)
    op.updateAttribute("dct_coefficient_count", dctCoefficientCount)
    op.addInput(spectrogram)
    op.addInput(sampleRate)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func min<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
>(
    _ input: Tensor<T>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Min", nOutputs)
    op.updateAttribute("keep_dims", keepDims)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(input)
    op.addInput(reductionIndices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func minimum<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Minimum", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mirrorPad<
    T: TensorFlowScalar,
    Tpaddings: TensorFlowIndex
>(
    _ input: Tensor<T>,
    paddings: Tensor<Tpaddings>,
    mode: Mode6
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MirrorPad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tpaddings", Tpaddings.tensorFlowDataType)
    op.updateAttribute("mode", mode.cName)
    op.addInput(input)
    op.addInput(paddings)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mirrorPadGrad<
    T: TensorFlowScalar,
    Tpaddings: TensorFlowIndex
>(
    _ input: Tensor<T>,
    paddings: Tensor<Tpaddings>,
    mode: Mode6
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MirrorPadGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tpaddings", Tpaddings.tensorFlowDataType)
    op.updateAttribute("mode", mode.cName)
    op.addInput(input)
    op.addInput(paddings)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mixedStruct(
    nA: Int64
) -> (a: [Tensor<Int32>], b: Tensor<Float>) {
  let nOutputs = Int(nA) + Int(1)
    let op = makeOp("MixedStruct", nOutputs)
    op.updateAttribute("n_a", nA)
    return op.execute(Int(nA), Int(1))
}

@inlinable @inline(__always)
public static func mlirPassthroughOp<
    Tinputs: TensorArrayProtocol,
    Toutputs: TensorGroup
>(
    inputs: Tinputs,
    mlirModule: String
) -> Toutputs {
  let nOutputs = Int(Toutputs._typeList.count)
    let op = makeOp("MlirPassthroughOp", nOutputs)
    op.updateAttribute("mlir_module", mlirModule)
    op.updateAttribute("Tinputs", inputs._typeList)
    op.updateAttribute("Toutputs", Toutputs._typeList)
    op.addInputList(inputs)
    return op.execute(Int(Toutputs._typeList.count))
}

@inlinable @inline(__always)
public static func mod<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Mod", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func modelDataset(
    inputDataset: VariantHandle,
    algorithm: Int64 = 0,
    cpuBudget: Int64 = 0,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ModelDataset", nOutputs)
    op.updateAttribute("algorithm", algorithm)
    op.updateAttribute("cpu_budget", cpuBudget)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mul<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Mul", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mulNoNan<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("MulNoNan", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func multiDeviceIterator(
    devices: [String],
    sharedName: String,
    container: String,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("MultiDeviceIterator", nOutputs)
    op.updateAttribute("devices", devices)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("container", container)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func multiDeviceIteratorFromStringHandle(
    stringHandle: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("MultiDeviceIteratorFromStringHandle", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(stringHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func multiDeviceIteratorGetNextFromShard<OutputTypes: TensorGroup>(
    multiDeviceIterator: ResourceHandle,
    shardNum: Tensor<Int32>,
    incarnationId: Tensor<Int64>,
    outputShapes: [TensorShape?]
) -> OutputTypes {
  let nOutputs = Int(OutputTypes._typeList.count)
    let op = makeOp("MultiDeviceIteratorGetNextFromShard", nOutputs)
    op.updateAttribute("output_types", OutputTypes._typeList)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(multiDeviceIterator)
    op.addInput(shardNum)
    op.addInput(incarnationId)
    return op.execute(Int(OutputTypes._typeList.count))
}

@inlinable @inline(__always)
public static func multiDeviceIteratorInit(
    dataset: VariantHandle,
    multiDeviceIterator: ResourceHandle,
    maxBufferSize: Tensor<Int64>
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("MultiDeviceIteratorInit", nOutputs)
    op.addInput(dataset)
    op.addInput(multiDeviceIterator)
    op.addInput(maxBufferSize)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func multiDeviceIteratorToStringHandle(
    multiDeviceIterator: ResourceHandle
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("MultiDeviceIteratorToStringHandle", nOutputs)
    op.addInput(multiDeviceIterator)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func multinomial<
    T: TensorFlowNumeric,
    OutputDtype: TensorFlowIndex
>(
    logits: Tensor<T>,
    numSamples: Tensor<Int32>,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> Tensor<OutputDtype> {
  let nOutputs = Int(1)
    let op = makeOp("Multinomial", nOutputs)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("output_dtype", OutputDtype.tensorFlowDataType)
    op.addInput(logits)
    op.addInput(numSamples)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mutableDenseHashTableV2<KeyDtype: TensorFlowScalar>(
    emptyKey: Tensor<KeyDtype>,
    deletedKey: Tensor<KeyDtype>,
    container: String,
    sharedName: String,
    useNodeNameSharing: Bool = false,
    valueDtype: TensorDataType,
    valueShape: TensorShape?,
    initialNumBuckets: Int64 = 131072,
    maxLoadFactor: Double = 0.8
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("MutableDenseHashTableV2", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("use_node_name_sharing", useNodeNameSharing)
    op.updateAttribute("key_dtype", KeyDtype.tensorFlowDataType)
    op.updateAttribute("value_dtype", valueDtype)
    op.updateAttribute("value_shape", valueShape)
    op.updateAttribute("initial_num_buckets", initialNumBuckets)
    op.updateAttribute("max_load_factor", maxLoadFactor)
    op.addInput(emptyKey)
    op.addInput(deletedKey)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mutableHashTableOfTensorsV2(
    container: String,
    sharedName: String,
    useNodeNameSharing: Bool = false,
    keyDtype: TensorDataType,
    valueDtype: TensorDataType,
    valueShape: TensorShape?
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("MutableHashTableOfTensorsV2", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("use_node_name_sharing", useNodeNameSharing)
    op.updateAttribute("key_dtype", keyDtype)
    op.updateAttribute("value_dtype", valueDtype)
    op.updateAttribute("value_shape", valueShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mutableHashTableV2(
    container: String,
    sharedName: String,
    useNodeNameSharing: Bool = false,
    keyDtype: TensorDataType,
    valueDtype: TensorDataType
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("MutableHashTableV2", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("use_node_name_sharing", useNodeNameSharing)
    op.updateAttribute("key_dtype", keyDtype)
    op.updateAttribute("value_dtype", valueDtype)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mutexLock(
    mutex: ResourceHandle
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("MutexLock", nOutputs)
    op.addInput(mutex)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func mutexV2(
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("MutexV2", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func nInPolymorphicTwice<T: TensorFlowScalar>(
    _ a: [Tensor<T>],
    _ b: [Tensor<T>]
) {
  let nOutputs = 0
    let op = makeOp("NInPolymorphicTwice", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("N", a.count)
    op.addInputList(a)
    op.addInputList(b)
    op.execute()
}

@inlinable @inline(__always)
public static func nInTwice(
    _ a: [Tensor<Int32>],
    _ b: [StringTensor]
) {
  let nOutputs = 0
    let op = makeOp("NInTwice", nOutputs)
    op.updateAttribute("N", a.count)
    op.addInputList(a)
    op.addInputList(b)
    op.execute()
}

@inlinable @inline(__always)
public static func nInTwoTypeVariables<
    S: TensorFlowScalar,
    T: TensorFlowScalar
>(
    _ a: [Tensor<S>],
    _ b: [Tensor<T>]
) {
  let nOutputs = 0
    let op = makeOp("NInTwoTypeVariables", nOutputs)
    op.updateAttribute("S", S.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("N", a.count)
    op.addInputList(a)
    op.addInputList(b)
    op.execute()
}

@inlinable @inline(__always)
public static func nIntsIn(
    _ a: [Tensor<Int32>]
) {
  let nOutputs = 0
    let op = makeOp("NIntsIn", nOutputs)
    op.updateAttribute("N", a.count)
    op.addInputList(a)
    op.execute()
}

@inlinable @inline(__always)
public static func nIntsOut(
    n: Int64
) -> [Tensor<Int32>] {
  let nOutputs = Int(n)
    let op = makeOp("NIntsOut", nOutputs)
    op.updateAttribute("N", n)
    return op.execute(Int(n))
}

@inlinable @inline(__always)
public static func nIntsOutDefault(
    n: Int64 = 3
) -> [Tensor<Int32>] {
  let nOutputs = Int(n)
    let op = makeOp("NIntsOutDefault", nOutputs)
    op.updateAttribute("N", n)
    return op.execute(Int(n))
}

@inlinable @inline(__always)
public static func nPolymorphicIn<T: TensorFlowScalar>(
    _ a: [Tensor<T>]
) {
  let nOutputs = 0
    let op = makeOp("NPolymorphicIn", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("N", a.count)
    op.addInputList(a)
    op.execute()
}

@inlinable @inline(__always)
public static func nPolymorphicOut<T: TensorFlowScalar>(
    n: Int64
) -> [Tensor<T>] {
  let nOutputs = Int(n)
    let op = makeOp("NPolymorphicOut", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("N", n)
    return op.execute(Int(n))
}

@inlinable @inline(__always)
public static func nPolymorphicOutDefault<T: TensorFlowScalar>(
    n: Int64 = 2
) -> [Tensor<T>] {
  let nOutputs = Int(n)
    let op = makeOp("NPolymorphicOutDefault", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("N", n)
    return op.execute(Int(n))
}

@inlinable @inline(__always)
public static func nPolymorphicRestrictIn<T: TensorFlowScalar>(
    _ a: [Tensor<T>]
) {
  let nOutputs = 0
    let op = makeOp("NPolymorphicRestrictIn", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("N", a.count)
    op.addInputList(a)
    op.execute()
}

@inlinable @inline(__always)
public static func nPolymorphicRestrictIn(
    _ a: [StringTensor]
) {
  let nOutputs = 0
    let op = makeOp("NPolymorphicRestrictIn", nOutputs)
    op.updateAttribute("T", TensorDataType(TF_STRING))
    op.updateAttribute("N", a.count)
    op.addInputList(a)
    op.execute()
}

@inlinable @inline(__always)
public static func nPolymorphicRestrictOut<T: TensorFlowScalar>(
    n: Int64
) -> [Tensor<T>] {
  let nOutputs = Int(n)
    let op = makeOp("NPolymorphicRestrictOut", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("N", n)
    return op.execute(Int(n))
}

@inlinable @inline(__always)
public static func nPolymorphicRestrictOut(
    n: Int64
) -> [StringTensor] {
  let nOutputs = Int(n)
    let op = makeOp("NPolymorphicRestrictOut", nOutputs)
    op.updateAttribute("T", TensorDataType(TF_STRING))
    op.updateAttribute("N", n)
    return op.execute(Int(n))
}

@inlinable @inline(__always)
public static func namespace>TestStringOutput(
    _ input: Tensor<Float>
) -> (output1: Tensor<Float>, output2: StringTensor) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("Namespace>TestStringOutput", nOutputs)
    op.addInput(input)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func ncclAllReduce<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    reduction: Reduction,
    numDevices: Int64,
    sharedName: String
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("NcclAllReduce", nOutputs)
    op.updateAttribute("reduction", reduction.cName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("num_devices", numDevices)
    op.updateAttribute("shared_name", sharedName)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func ncclBroadcast<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    shape: TensorShape?
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("NcclBroadcast", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("shape", shape)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func ncclReduce<T: TensorFlowNumeric>(
    _ input: [Tensor<T>],
    reduction: Reduction
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("NcclReduce", nOutputs)
    op.updateAttribute("reduction", reduction.cName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("num_devices", input.count)
    op.addInputList(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func ndtri<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Ndtri", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func nearestNeighbors(
    points: Tensor<Float>,
    centers: Tensor<Float>,
    k: Tensor<Int64>
) -> (nearestCenterIndices: Tensor<Int64>, nearestCenterDistances: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("NearestNeighbors", nOutputs)
    op.addInput(points)
    op.addInput(centers)
    op.addInput(k)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func neg<T: TensorFlowNumeric>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Neg", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func nextAfter<T: FloatingPoint & TensorFlowScalar>(
    x1: Tensor<T>,
    x2: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("NextAfter", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x1)
    op.addInput(x2)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func nextIteration<T: TensorFlowScalar>(
    data: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("NextIteration", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(data)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func noOp(
) {
  let nOutputs = 0
    let op = makeOp("NoOp", nOutputs)
    
    op.execute()
}

@inlinable @inline(__always)
public static func nonDeterministicInts<
    Dtype: TensorFlowScalar,
    ShapeDtype: TensorFlowScalar
>(
    shape: Tensor<ShapeDtype>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("NonDeterministicInts", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape_dtype", ShapeDtype.tensorFlowDataType)
    op.addInput(shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func nonMaxSuppression(
    boxes: Tensor<Float>,
    scores: Tensor<Float>,
    maxOutputSize: Tensor<Int32>,
    iouThreshold: Double = 0.5
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("NonMaxSuppression", nOutputs)
    op.updateAttribute("iou_threshold", iouThreshold)
    op.addInput(boxes)
    op.addInput(scores)
    op.addInput(maxOutputSize)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func nonMaxSuppressionV2<
    T: FloatingPoint & TensorFlowScalar,
    TThreshold: FloatingPoint & TensorFlowScalar
>(
    boxes: Tensor<T>,
    scores: Tensor<T>,
    maxOutputSize: Tensor<Int32>,
    iouThreshold: Tensor<TThreshold>
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("NonMaxSuppressionV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("T_threshold", TThreshold.tensorFlowDataType)
    op.addInput(boxes)
    op.addInput(scores)
    op.addInput(maxOutputSize)
    op.addInput(iouThreshold)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func nonMaxSuppressionV3<
    T: FloatingPoint & TensorFlowScalar,
    TThreshold: FloatingPoint & TensorFlowScalar
>(
    boxes: Tensor<T>,
    scores: Tensor<T>,
    maxOutputSize: Tensor<Int32>,
    iouThreshold: Tensor<TThreshold>,
    scoreThreshold: Tensor<TThreshold>
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("NonMaxSuppressionV3", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("T_threshold", TThreshold.tensorFlowDataType)
    op.addInput(boxes)
    op.addInput(scores)
    op.addInput(maxOutputSize)
    op.addInput(iouThreshold)
    op.addInput(scoreThreshold)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func nonMaxSuppressionV4<
    T: FloatingPoint & TensorFlowScalar,
    TThreshold: FloatingPoint & TensorFlowScalar
>(
    boxes: Tensor<T>,
    scores: Tensor<T>,
    maxOutputSize: Tensor<Int32>,
    iouThreshold: Tensor<TThreshold>,
    scoreThreshold: Tensor<TThreshold>,
    padToMaxOutputSize: Bool = false
) -> (selectedIndices: Tensor<Int32>, validOutputs: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("NonMaxSuppressionV4", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("T_threshold", TThreshold.tensorFlowDataType)
    op.updateAttribute("pad_to_max_output_size", padToMaxOutputSize)
    op.addInput(boxes)
    op.addInput(scores)
    op.addInput(maxOutputSize)
    op.addInput(iouThreshold)
    op.addInput(scoreThreshold)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func nonMaxSuppressionV5<T: FloatingPoint & TensorFlowScalar>(
    boxes: Tensor<T>,
    scores: Tensor<T>,
    maxOutputSize: Tensor<Int32>,
    iouThreshold: Tensor<T>,
    scoreThreshold: Tensor<T>,
    softNmsSigma: Tensor<T>,
    padToMaxOutputSize: Bool = false
) -> (selectedIndices: Tensor<Int32>, selectedScores: Tensor<T>, validOutputs: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("NonMaxSuppressionV5", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("pad_to_max_output_size", padToMaxOutputSize)
    op.addInput(boxes)
    op.addInput(scores)
    op.addInput(maxOutputSize)
    op.addInput(iouThreshold)
    op.addInput(scoreThreshold)
    op.addInput(softNmsSigma)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func nonMaxSuppressionWithOverlaps(
    overlaps: Tensor<Float>,
    scores: Tensor<Float>,
    maxOutputSize: Tensor<Int32>,
    overlapThreshold: Tensor<Float>,
    scoreThreshold: Tensor<Float>
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("NonMaxSuppressionWithOverlaps", nOutputs)
    op.addInput(overlaps)
    op.addInput(scores)
    op.addInput(maxOutputSize)
    op.addInput(overlapThreshold)
    op.addInput(scoreThreshold)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func nonSerializableDataset(
    inputDataset: VariantHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("NonSerializableDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func none(
) {
  let nOutputs = 0
    let op = makeOp("None", nOutputs)
    
    op.execute()
}

@inlinable @inline(__always)
public static func notEqual<T: TensorFlowScalar>(
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    incompatibleShapeError: Bool = true
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("NotEqual", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("incompatible_shape_error", incompatibleShapeError)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func notEqual(
    _ x: StringTensor,
    _ y: StringTensor,
    incompatibleShapeError: Bool = true
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("NotEqual", nOutputs)
    op.updateAttribute("T", TensorDataType(TF_STRING))
    op.updateAttribute("incompatible_shape_error", incompatibleShapeError)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func nthElement<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    n: Tensor<Int32>,
    reverse: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("NthElement", nOutputs)
    op.updateAttribute("reverse", reverse)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    op.addInput(n)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func old(
) {
  let nOutputs = 0
    let op = makeOp("Old", nOutputs)
    
    op.execute()
}

@inlinable @inline(__always)
public static func oneHot<
    T: TensorFlowScalar,
    Ti: TensorFlowInteger
>(
    indices: Tensor<Ti>,
    depth: Tensor<Int32>,
    onValue: Tensor<T>,
    offValue: Tensor<T>,
    axis: Int64 = -1
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("OneHot", nOutputs)
    op.updateAttribute("axis", axis)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("TI", Ti.tensorFlowDataType)
    op.addInput(indices)
    op.addInput(depth)
    op.addInput(onValue)
    op.addInput(offValue)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func oneShotIterator<DatasetfactoryIn: TensorGroup,
    DatasetfactoryOut: TensorGroup>(
    datasetFactory: (DatasetfactoryIn) -> DatasetfactoryOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("OneShotIterator", nOutputs)
    op.updateAttribute("dataset_factory", datasetFactory)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func onesLike<T: TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("OnesLike", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func opWithDefaultAttr(
    defaultFloat: Double = 123
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("OpWithDefaultAttr", nOutputs)
    op.updateAttribute("default_float", defaultFloat)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func opWithFutureDefaultAttr(
) {
  let nOutputs = 0
    let op = makeOp("OpWithFutureDefaultAttr", nOutputs)
    
    op.execute()
}

@inlinable @inline(__always)
public static func optimizeDataset(
    inputDataset: VariantHandle,
    optimizations: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    optimizationConfigs: [String]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("OptimizeDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("optimization_configs", optimizationConfigs)
    op.addInput(inputDataset)
    op.addInput(optimizations)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func optionalFromValue<ToutputTypes: TensorArrayProtocol>(
    components: ToutputTypes
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("OptionalFromValue", nOutputs)
    op.updateAttribute("Toutput_types", components._typeList)
    op.addInputList(components)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func optionalGetValue<OutputTypes: TensorGroup>(
    optional: VariantHandle,
    outputShapes: [TensorShape?]
) -> OutputTypes {
  let nOutputs = Int(OutputTypes._typeList.count)
    let op = makeOp("OptionalGetValue", nOutputs)
    op.updateAttribute("output_types", OutputTypes._typeList)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(optional)
    return op.execute(Int(OutputTypes._typeList.count))
}

@inlinable @inline(__always)
public static func optionalHasValue(
    optional: VariantHandle
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("OptionalHasValue", nOutputs)
    op.addInput(optional)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func optionalNone(
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("OptionalNone", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func orderedMapClear(
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    dtypes: [TensorDataType],
    container: String,
    sharedName: String
) {
  let nOutputs = 0
    let op = makeOp("OrderedMapClear", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", dtypes)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.execute()
}

@inlinable @inline(__always)
public static func orderedMapIncompleteSize(
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    dtypes: [TensorDataType],
    container: String,
    sharedName: String
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("OrderedMapIncompleteSize", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", dtypes)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func orderedMapPeek<Dtypes: TensorGroup>(
    key: Tensor<Int64>,
    indices: Tensor<Int32>,
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    container: String,
    sharedName: String
) -> Dtypes {
  let nOutputs = Int(Dtypes._typeList.count)
    let op = makeOp("OrderedMapPeek", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", Dtypes._typeList)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.addInput(key)
    op.addInput(indices)
    return op.execute(Int(Dtypes._typeList.count))
}

@inlinable @inline(__always)
public static func orderedMapSize(
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    dtypes: [TensorDataType],
    container: String,
    sharedName: String
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("OrderedMapSize", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", dtypes)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func orderedMapStage<FakeDtypes: TensorArrayProtocol>(
    key: Tensor<Int64>,
    indices: Tensor<Int32>,
    _ values: FakeDtypes,
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    dtypes: [TensorDataType],
    container: String,
    sharedName: String
) {
  let nOutputs = 0
    let op = makeOp("OrderedMapStage", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", dtypes)
    op.updateAttribute("fake_dtypes", values._typeList)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.addInput(key)
    op.addInput(indices)
    op.addInputList(values)
    op.execute()
}

@inlinable @inline(__always)
public static func orderedMapUnstage<Dtypes: TensorGroup>(
    key: Tensor<Int64>,
    indices: Tensor<Int32>,
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    container: String,
    sharedName: String
) -> Dtypes {
  let nOutputs = Int(Dtypes._typeList.count)
    let op = makeOp("OrderedMapUnstage", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", Dtypes._typeList)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.addInput(key)
    op.addInput(indices)
    return op.execute(Int(Dtypes._typeList.count))
}

@inlinable @inline(__always)
public static func orderedMapUnstageNoKey<Dtypes: TensorGroup>(
    indices: Tensor<Int32>,
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    container: String,
    sharedName: String
) -> (key: Tensor<Int64>, values: Dtypes) {
  let nOutputs = Int(1) + Int(Dtypes._typeList.count)
    let op = makeOp("OrderedMapUnstageNoKey", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", Dtypes._typeList)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.addInput(indices)
    return op.execute(Int(1), Int(Dtypes._typeList.count))
}

@inlinable @inline(__always)
public static func outT<T: TensorFlowScalar>(
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("OutT", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func outTypeList<T: TensorGroup>(
) -> T {
  let nOutputs = Int(T._typeList.count)
    let op = makeOp("OutTypeList", nOutputs)
    op.updateAttribute("T", T._typeList)
    return op.execute(Int(T._typeList.count))
}

@inlinable @inline(__always)
public static func outTypeListRestrict<T: TensorGroup>(
) -> T {
  let nOutputs = Int(T._typeList.count)
    let op = makeOp("OutTypeListRestrict", nOutputs)
    op.updateAttribute("t", T._typeList)
    return op.execute(Int(T._typeList.count))
}

@inlinable @inline(__always)
public static func outfeedDequeue<Dtype: TensorFlowScalar>(
    shape: TensorShape?,
    deviceOrdinal: Int64 = -1
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("OutfeedDequeue", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape", shape)
    op.updateAttribute("device_ordinal", deviceOrdinal)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func outfeedDequeueTuple<Dtypes: TensorGroup>(
    shapes: [TensorShape?],
    deviceOrdinal: Int64 = -1
) -> Dtypes {
  let nOutputs = Int(Dtypes._typeList.count)
    let op = makeOp("OutfeedDequeueTuple", nOutputs)
    op.updateAttribute("dtypes", Dtypes._typeList)
    op.updateAttribute("shapes", shapes)
    op.updateAttribute("device_ordinal", deviceOrdinal)
    return op.execute(Int(Dtypes._typeList.count))
}

@inlinable @inline(__always)
public static func outfeedEnqueue<Dtype: TensorFlowScalar>(
    _ input: Tensor<Dtype>
) {
  let nOutputs = 0
    let op = makeOp("OutfeedEnqueue", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(input)
    op.execute()
}

@inlinable @inline(__always)
public static func outfeedEnqueueTuple<Dtypes: TensorArrayProtocol>(
    inputs: Dtypes
) {
  let nOutputs = 0
    let op = makeOp("OutfeedEnqueueTuple", nOutputs)
    op.updateAttribute("dtypes", inputs._typeList)
    op.addInputList(inputs)
    op.execute()
}

@inlinable @inline(__always)
public static func pack<T: TensorFlowScalar>(
    _ values: [Tensor<T>],
    axis: Int64 = 0
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Pack", nOutputs)
    op.updateAttribute("N", values.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("axis", axis)
    op.addInputList(values)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func pad<
    T: TensorFlowScalar,
    Tpaddings: TensorFlowIndex
>(
    _ input: Tensor<T>,
    paddings: Tensor<Tpaddings>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Pad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tpaddings", Tpaddings.tensorFlowDataType)
    op.addInput(input)
    op.addInput(paddings)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func padV2<
    T: TensorFlowScalar,
    Tpaddings: TensorFlowIndex
>(
    _ input: Tensor<T>,
    paddings: Tensor<Tpaddings>,
    constantValues: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("PadV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tpaddings", Tpaddings.tensorFlowDataType)
    op.addInput(input)
    op.addInput(paddings)
    op.addInput(constantValues)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func paddedBatchDataset<ToutputTypes: TensorArrayProtocol>(
    inputDataset: VariantHandle,
    batchSize: Tensor<Int64>,
    paddedShapes: [Tensor<Int64>],
    paddingValues: ToutputTypes,
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("PaddedBatchDataset", nOutputs)
    op.updateAttribute("Toutput_types", paddingValues._typeList)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("N", paddedShapes.count)
    op.addInput(inputDataset)
    op.addInput(batchSize)
    op.addInputList(paddedShapes)
    op.addInputList(paddingValues)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func paddedBatchDatasetV2<ToutputTypes: TensorArrayProtocol>(
    inputDataset: VariantHandle,
    batchSize: Tensor<Int64>,
    paddedShapes: [Tensor<Int64>],
    paddingValues: ToutputTypes,
    dropRemainder: Tensor<Bool>,
    parallelCopy: Bool = false,
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("PaddedBatchDatasetV2", nOutputs)
    op.updateAttribute("parallel_copy", parallelCopy)
    op.updateAttribute("Toutput_types", paddingValues._typeList)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("N", paddedShapes.count)
    op.addInput(inputDataset)
    op.addInput(batchSize)
    op.addInputList(paddedShapes)
    op.addInputList(paddingValues)
    op.addInput(dropRemainder)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func paddingFIFOQueueV2(
    componentTypes: [TensorDataType],
    shapes: [TensorShape?],
    capacity: Int64 = -1,
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("PaddingFIFOQueueV2", nOutputs)
    op.updateAttribute("component_types", componentTypes)
    op.updateAttribute("shapes", shapes)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func parallelConcat<T: TensorFlowScalar>(
    _ values: [Tensor<T>],
    shape: TensorShape?
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ParallelConcat", nOutputs)
    op.updateAttribute("N", values.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("shape", shape)
    op.addInputList(values)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func parallelDynamicStitch<T: TensorFlowScalar>(
    indices: [Tensor<Int32>],
    data: [Tensor<T>]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ParallelDynamicStitch", nOutputs)
    op.updateAttribute("N", indices.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInputList(indices)
    op.addInputList(data)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func parallelInterleaveDataset<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    otherArguments: Targuments,
    cycleLength: Tensor<Int64>,
    blockLength: Tensor<Int64>,
    sloppy: Tensor<Bool>,
    bufferOutputElements: Tensor<Int64>,
    prefetchInputElements: Tensor<Int64>,
    f: (FIn) -> FOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ParallelInterleaveDataset", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInputList(otherArguments)
    op.addInput(cycleLength)
    op.addInput(blockLength)
    op.addInput(sloppy)
    op.addInput(bufferOutputElements)
    op.addInput(prefetchInputElements)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func parallelInterleaveDatasetV2<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    otherArguments: Targuments,
    cycleLength: Tensor<Int64>,
    blockLength: Tensor<Int64>,
    numParallelCalls: Tensor<Int64>,
    f: (FIn) -> FOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    sloppy: Bool = false
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ParallelInterleaveDatasetV2", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("sloppy", sloppy)
    op.addInput(inputDataset)
    op.addInputList(otherArguments)
    op.addInput(cycleLength)
    op.addInput(blockLength)
    op.addInput(numParallelCalls)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func parallelMapDataset<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    otherArguments: Targuments,
    numParallelCalls: Tensor<Int32>,
    f: (FIn) -> FOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    useInterOpParallelism: Bool = true,
    sloppy: Bool = false,
    preserveCardinality: Bool = false
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ParallelMapDataset", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("use_inter_op_parallelism", useInterOpParallelism)
    op.updateAttribute("sloppy", sloppy)
    op.updateAttribute("preserve_cardinality", preserveCardinality)
    op.addInput(inputDataset)
    op.addInputList(otherArguments)
    op.addInput(numParallelCalls)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func parameterizedTruncatedNormal<
    Dtype: FloatingPoint & TensorFlowScalar,
    T: TensorFlowIndex
>(
    shape: Tensor<T>,
    means: Tensor<Dtype>,
    stdevs: Tensor<Dtype>,
    minvals: Tensor<Dtype>,
    maxvals: Tensor<Dtype>,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("ParameterizedTruncatedNormal", nOutputs)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(shape)
    op.addInput(means)
    op.addInput(stdevs)
    op.addInput(minvals)
    op.addInput(maxvals)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func parseExample<
    SparseTypes: TensorGroup,
    Tdense: TensorArrayProtocol
>(
    serialized: StringTensor,
    names: StringTensor,
    sparseKeys: [StringTensor],
    denseKeys: [StringTensor],
    denseDefaults: Tdense,
    denseShapes: [TensorShape?]
) -> (sparseIndices: [Tensor<Int64>], sparseValues: SparseTypes, sparseShapes: [Tensor<Int64>], denseValues: Tdense) {
  let nOutputs = Int(sparseKeys.count) + Int(SparseTypes._typeList.count) + Int(sparseKeys.count) + Int(denseDefaults._typeList.count)
    let op = makeOp("ParseExample", nOutputs)
    op.updateAttribute("Nsparse", sparseKeys.count)
    op.updateAttribute("Ndense", denseKeys.count)
    op.updateAttribute("sparse_types", SparseTypes._typeList)
    op.updateAttribute("Tdense", denseDefaults._typeList)
    op.updateAttribute("dense_shapes", denseShapes)
    op.addInput(serialized)
    op.addInput(names)
    op.addInputList(sparseKeys)
    op.addInputList(denseKeys)
    op.addInputList(denseDefaults)
    return op.execute(Int(sparseKeys.count), Int(SparseTypes._typeList.count), Int(sparseKeys.count), Int(denseDefaults._typeList.count))
}

@inlinable @inline(__always)
public static func parseExampleDataset<Tdense: TensorArrayProtocol>(
    inputDataset: VariantHandle,
    numParallelCalls: Tensor<Int64>,
    denseDefaults: Tdense,
    sparseKeys: [String],
    denseKeys: [String],
    sparseTypes: [TensorDataType],
    denseShapes: [TensorShape?],
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    sloppy: Bool = false,
    raggedKeys: [String],
    raggedValueTypes: [TensorDataType],
    raggedSplitTypes: [TensorDataType]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ParseExampleDataset", nOutputs)
    op.updateAttribute("sparse_keys", sparseKeys)
    op.updateAttribute("dense_keys", denseKeys)
    op.updateAttribute("sparse_types", sparseTypes)
    op.updateAttribute("Tdense", denseDefaults._typeList)
    op.updateAttribute("dense_shapes", denseShapes)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("sloppy", sloppy)
    op.updateAttribute("ragged_keys", raggedKeys)
    op.updateAttribute("ragged_value_types", raggedValueTypes)
    op.updateAttribute("ragged_split_types", raggedSplitTypes)
    op.addInput(inputDataset)
    op.addInput(numParallelCalls)
    op.addInputList(denseDefaults)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func parseExampleV2<
    Tdense: TensorArrayProtocol,
    SparseTypes: TensorGroup,
    RaggedValueTypes: TensorGroup,
    RaggedSplitTypes: TensorGroup
>(
    serialized: StringTensor,
    names: StringTensor,
    sparseKeys: StringTensor,
    denseKeys: StringTensor,
    raggedKeys: StringTensor,
    denseDefaults: Tdense,
    numSparse: Int64,
    denseShapes: [TensorShape?]
) -> (sparseIndices: [Tensor<Int64>], sparseValues: SparseTypes, sparseShapes: [Tensor<Int64>], denseValues: Tdense, raggedValues: RaggedValueTypes, raggedRowSplits: RaggedSplitTypes) {
  let nOutputs = Int(numSparse) + Int(SparseTypes._typeList.count) + Int(numSparse) + Int(denseDefaults._typeList.count) + Int(RaggedValueTypes._typeList.count) + Int(RaggedSplitTypes._typeList.count)
    let op = makeOp("ParseExampleV2", nOutputs)
    op.updateAttribute("Tdense", denseDefaults._typeList)
    op.updateAttribute("num_sparse", numSparse)
    op.updateAttribute("sparse_types", SparseTypes._typeList)
    op.updateAttribute("ragged_value_types", RaggedValueTypes._typeList)
    op.updateAttribute("ragged_split_types", RaggedSplitTypes._typeList)
    op.updateAttribute("dense_shapes", denseShapes)
    op.addInput(serialized)
    op.addInput(names)
    op.addInput(sparseKeys)
    op.addInput(denseKeys)
    op.addInput(raggedKeys)
    op.addInputList(denseDefaults)
    return op.execute(Int(numSparse), Int(SparseTypes._typeList.count), Int(numSparse), Int(denseDefaults._typeList.count), Int(RaggedValueTypes._typeList.count), Int(RaggedSplitTypes._typeList.count))
}

@inlinable @inline(__always)
public static func parseSequenceExample<
    ContextSparseTypes: TensorGroup,
    TcontextDense: TensorArrayProtocol,
    FeatureListDenseTypes: TensorGroup,
    FeatureListSparseTypes: TensorGroup
>(
    serialized: StringTensor,
    debugName: StringTensor,
    contextDenseDefaults: TcontextDense,
    featureListDenseMissingAssumedEmpty: [String],
    contextSparseKeys: [String],
    contextDenseKeys: [String],
    featureListSparseKeys: [String],
    featureListDenseKeys: [String],
    ncontextSparse: Int64 = 0,
    ncontextDense: Int64 = 0,
    nfeatureListSparse: Int64 = 0,
    nfeatureListDense: Int64 = 0,
    contextDenseShapes: [TensorShape?],
    featureListDenseShapes: [TensorShape?]
) -> (contextSparseIndices: [Tensor<Int64>], contextSparseValues: ContextSparseTypes, contextSparseShapes: [Tensor<Int64>], contextDenseValues: TcontextDense, featureListSparseIndices: [Tensor<Int64>], featureListSparseValues: FeatureListSparseTypes, featureListSparseShapes: [Tensor<Int64>], featureListDenseValues: FeatureListDenseTypes, featureListDenseLengths: [Tensor<Int64>]) {
  let nOutputs = Int(ncontextSparse) + Int(ContextSparseTypes._typeList.count) + Int(ncontextSparse) + Int(contextDenseDefaults._typeList.count) + Int(nfeatureListSparse) + Int(FeatureListSparseTypes._typeList.count) + Int(nfeatureListSparse) + Int(FeatureListDenseTypes._typeList.count) + Int(nfeatureListDense)
    let op = makeOp("ParseSequenceExample", nOutputs)
    op.updateAttribute("feature_list_dense_missing_assumed_empty", featureListDenseMissingAssumedEmpty)
    op.updateAttribute("context_sparse_keys", contextSparseKeys)
    op.updateAttribute("context_dense_keys", contextDenseKeys)
    op.updateAttribute("feature_list_sparse_keys", featureListSparseKeys)
    op.updateAttribute("feature_list_dense_keys", featureListDenseKeys)
    op.updateAttribute("Ncontext_sparse", ncontextSparse)
    op.updateAttribute("Ncontext_dense", ncontextDense)
    op.updateAttribute("Nfeature_list_sparse", nfeatureListSparse)
    op.updateAttribute("Nfeature_list_dense", nfeatureListDense)
    op.updateAttribute("context_sparse_types", ContextSparseTypes._typeList)
    op.updateAttribute("Tcontext_dense", contextDenseDefaults._typeList)
    op.updateAttribute("feature_list_dense_types", FeatureListDenseTypes._typeList)
    op.updateAttribute("context_dense_shapes", contextDenseShapes)
    op.updateAttribute("feature_list_sparse_types", FeatureListSparseTypes._typeList)
    op.updateAttribute("feature_list_dense_shapes", featureListDenseShapes)
    op.addInput(serialized)
    op.addInput(debugName)
    op.addInputList(contextDenseDefaults)
    return op.execute(Int(ncontextSparse), Int(ContextSparseTypes._typeList.count), Int(ncontextSparse), Int(contextDenseDefaults._typeList.count), Int(nfeatureListSparse), Int(FeatureListSparseTypes._typeList.count), Int(nfeatureListSparse), Int(FeatureListDenseTypes._typeList.count), Int(nfeatureListDense))
}

@inlinable @inline(__always)
public static func parseSequenceExampleV2<
    TcontextDense: TensorArrayProtocol,
    ContextSparseTypes: TensorGroup,
    ContextRaggedValueTypes: TensorGroup,
    ContextRaggedSplitTypes: TensorGroup,
    FeatureListDenseTypes: TensorGroup,
    FeatureListSparseTypes: TensorGroup,
    FeatureListRaggedValueTypes: TensorGroup,
    FeatureListRaggedSplitTypes: TensorGroup
>(
    serialized: StringTensor,
    debugName: StringTensor,
    contextSparseKeys: StringTensor,
    contextDenseKeys: StringTensor,
    contextRaggedKeys: StringTensor,
    featureListSparseKeys: StringTensor,
    featureListDenseKeys: StringTensor,
    featureListRaggedKeys: StringTensor,
    featureListDenseMissingAssumedEmpty: Tensor<Bool>,
    contextDenseDefaults: TcontextDense,
    ncontextSparse: Int64 = 0,
    contextDenseShapes: [TensorShape?],
    nfeatureListSparse: Int64 = 0,
    nfeatureListDense: Int64 = 0,
    featureListDenseShapes: [TensorShape?]
) -> (contextSparseIndices: [Tensor<Int64>], contextSparseValues: ContextSparseTypes, contextSparseShapes: [Tensor<Int64>], contextDenseValues: TcontextDense, contextRaggedValues: ContextRaggedValueTypes, contextRaggedRowSplits: ContextRaggedSplitTypes, featureListSparseIndices: [Tensor<Int64>], featureListSparseValues: FeatureListSparseTypes, featureListSparseShapes: [Tensor<Int64>], featureListDenseValues: FeatureListDenseTypes, featureListDenseLengths: [Tensor<Int64>], featureListRaggedValues: FeatureListRaggedValueTypes, featureListRaggedOuterSplits: FeatureListRaggedSplitTypes, featureListRaggedInnerSplits: FeatureListRaggedSplitTypes) {
  let nOutputs = Int(ncontextSparse) + Int(ContextSparseTypes._typeList.count) + Int(ncontextSparse) + Int(contextDenseDefaults._typeList.count) + Int(ContextRaggedValueTypes._typeList.count) + Int(ContextRaggedSplitTypes._typeList.count) + Int(nfeatureListSparse) + Int(FeatureListSparseTypes._typeList.count) + Int(nfeatureListSparse) + Int(FeatureListDenseTypes._typeList.count) + Int(nfeatureListDense) + Int(FeatureListRaggedValueTypes._typeList.count) + Int(FeatureListRaggedSplitTypes._typeList.count) + Int(FeatureListRaggedSplitTypes._typeList.count)
    let op = makeOp("ParseSequenceExampleV2", nOutputs)
    op.updateAttribute("Ncontext_sparse", ncontextSparse)
    op.updateAttribute("Tcontext_dense", contextDenseDefaults._typeList)
    op.updateAttribute("context_sparse_types", ContextSparseTypes._typeList)
    op.updateAttribute("context_ragged_value_types", ContextRaggedValueTypes._typeList)
    op.updateAttribute("context_ragged_split_types", ContextRaggedSplitTypes._typeList)
    op.updateAttribute("context_dense_shapes", contextDenseShapes)
    op.updateAttribute("Nfeature_list_sparse", nfeatureListSparse)
    op.updateAttribute("Nfeature_list_dense", nfeatureListDense)
    op.updateAttribute("feature_list_dense_types", FeatureListDenseTypes._typeList)
    op.updateAttribute("feature_list_sparse_types", FeatureListSparseTypes._typeList)
    op.updateAttribute("feature_list_ragged_value_types", FeatureListRaggedValueTypes._typeList)
    op.updateAttribute("feature_list_ragged_split_types", FeatureListRaggedSplitTypes._typeList)
    op.updateAttribute("feature_list_dense_shapes", featureListDenseShapes)
    op.addInput(serialized)
    op.addInput(debugName)
    op.addInput(contextSparseKeys)
    op.addInput(contextDenseKeys)
    op.addInput(contextRaggedKeys)
    op.addInput(featureListSparseKeys)
    op.addInput(featureListDenseKeys)
    op.addInput(featureListRaggedKeys)
    op.addInput(featureListDenseMissingAssumedEmpty)
    op.addInputList(contextDenseDefaults)
    return op.execute(Int(ncontextSparse), Int(ContextSparseTypes._typeList.count), Int(ncontextSparse), Int(contextDenseDefaults._typeList.count), Int(ContextRaggedValueTypes._typeList.count), Int(ContextRaggedSplitTypes._typeList.count), Int(nfeatureListSparse), Int(FeatureListSparseTypes._typeList.count), Int(nfeatureListSparse), Int(FeatureListDenseTypes._typeList.count), Int(nfeatureListDense), Int(FeatureListRaggedValueTypes._typeList.count), Int(FeatureListRaggedSplitTypes._typeList.count), Int(FeatureListRaggedSplitTypes._typeList.count))
}

@inlinable @inline(__always)
public static func parseSingleExample<
    SparseTypes: TensorGroup,
    Tdense: TensorArrayProtocol
>(
    serialized: StringTensor,
    denseDefaults: Tdense,
    numSparse: Int64,
    sparseKeys: [String],
    denseKeys: [String],
    denseShapes: [TensorShape?]
) -> (sparseIndices: [Tensor<Int64>], sparseValues: SparseTypes, sparseShapes: [Tensor<Int64>], denseValues: Tdense) {
  let nOutputs = Int(numSparse) + Int(SparseTypes._typeList.count) + Int(numSparse) + Int(denseDefaults._typeList.count)
    let op = makeOp("ParseSingleExample", nOutputs)
    op.updateAttribute("num_sparse", numSparse)
    op.updateAttribute("sparse_keys", sparseKeys)
    op.updateAttribute("dense_keys", denseKeys)
    op.updateAttribute("sparse_types", SparseTypes._typeList)
    op.updateAttribute("Tdense", denseDefaults._typeList)
    op.updateAttribute("dense_shapes", denseShapes)
    op.addInput(serialized)
    op.addInputList(denseDefaults)
    return op.execute(Int(numSparse), Int(SparseTypes._typeList.count), Int(numSparse), Int(denseDefaults._typeList.count))
}

@inlinable @inline(__always)
public static func parseSingleSequenceExample<
    ContextSparseTypes: TensorGroup,
    TcontextDense: TensorArrayProtocol,
    FeatureListDenseTypes: TensorGroup,
    FeatureListSparseTypes: TensorGroup
>(
    serialized: StringTensor,
    featureListDenseMissingAssumedEmpty: StringTensor,
    contextSparseKeys: [StringTensor],
    contextDenseKeys: [StringTensor],
    featureListSparseKeys: [StringTensor],
    featureListDenseKeys: [StringTensor],
    contextDenseDefaults: TcontextDense,
    debugName: StringTensor,
    contextDenseShapes: [TensorShape?],
    featureListDenseShapes: [TensorShape?]
) -> (contextSparseIndices: [Tensor<Int64>], contextSparseValues: ContextSparseTypes, contextSparseShapes: [Tensor<Int64>], contextDenseValues: TcontextDense, featureListSparseIndices: [Tensor<Int64>], featureListSparseValues: FeatureListSparseTypes, featureListSparseShapes: [Tensor<Int64>], featureListDenseValues: FeatureListDenseTypes) {
  let nOutputs = Int(contextSparseKeys.count) + Int(ContextSparseTypes._typeList.count) + Int(contextSparseKeys.count) + Int(contextDenseDefaults._typeList.count) + Int(featureListSparseKeys.count) + Int(FeatureListSparseTypes._typeList.count) + Int(featureListSparseKeys.count) + Int(FeatureListDenseTypes._typeList.count)
    let op = makeOp("ParseSingleSequenceExample", nOutputs)
    op.updateAttribute("Ncontext_sparse", contextSparseKeys.count)
    op.updateAttribute("Ncontext_dense", contextDenseKeys.count)
    op.updateAttribute("Nfeature_list_sparse", featureListSparseKeys.count)
    op.updateAttribute("Nfeature_list_dense", featureListDenseKeys.count)
    op.updateAttribute("context_sparse_types", ContextSparseTypes._typeList)
    op.updateAttribute("Tcontext_dense", contextDenseDefaults._typeList)
    op.updateAttribute("feature_list_dense_types", FeatureListDenseTypes._typeList)
    op.updateAttribute("context_dense_shapes", contextDenseShapes)
    op.updateAttribute("feature_list_sparse_types", FeatureListSparseTypes._typeList)
    op.updateAttribute("feature_list_dense_shapes", featureListDenseShapes)
    op.addInput(serialized)
    op.addInput(featureListDenseMissingAssumedEmpty)
    op.addInputList(contextSparseKeys)
    op.addInputList(contextDenseKeys)
    op.addInputList(featureListSparseKeys)
    op.addInputList(featureListDenseKeys)
    op.addInputList(contextDenseDefaults)
    op.addInput(debugName)
    return op.execute(Int(contextSparseKeys.count), Int(ContextSparseTypes._typeList.count), Int(contextSparseKeys.count), Int(contextDenseDefaults._typeList.count), Int(featureListSparseKeys.count), Int(FeatureListSparseTypes._typeList.count), Int(featureListSparseKeys.count), Int(FeatureListDenseTypes._typeList.count))
}

@inlinable @inline(__always)
public static func parseTensor<OutType: TensorFlowScalar>(
    serialized: StringTensor
) -> Tensor<OutType> {
  let nOutputs = Int(1)
    let op = makeOp("ParseTensor", nOutputs)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(serialized)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func partitionedCall<
    Tin: TensorArrayProtocol,
    Tout: TensorGroup,
    FIn: TensorGroup,
    FOut: TensorGroup
>(
    args: Tin,
    f: (FIn) -> FOut,
    config: String,
    configProto: String,
    executorType: String
) -> Tout {
  let nOutputs = Int(Tout._typeList.count)
    let op = makeOp("PartitionedCall", nOutputs)
    op.updateAttribute("Tin", args._typeList)
    op.updateAttribute("Tout", Tout._typeList)
    op.updateAttribute("f", f)
    op.updateAttribute("config", config)
    op.updateAttribute("config_proto", configProto)
    op.updateAttribute("executor_type", executorType)
    op.addInputList(args)
    return op.execute(Int(Tout._typeList.count))
}

@inlinable @inline(__always)
public static func placeholder<Dtype: TensorFlowScalar>(
    shape: TensorShape?
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("Placeholder", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape", shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func placeholderV2<Dtype: TensorFlowScalar>(
    shape: TensorShape?
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("PlaceholderV2", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape", shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func placeholderWithDefault<Dtype: TensorFlowScalar>(
    _ input: Tensor<Dtype>,
    shape: TensorShape?
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("PlaceholderWithDefault", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape", shape)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func polygamma<T: FloatingPoint & TensorFlowScalar>(
    _ a: Tensor<T>,
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Polygamma", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func polymorphic<T: TensorFlowScalar>(
    _ a: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Polymorphic", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func polymorphicDefaultOut<T: TensorFlowScalar>(
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("PolymorphicDefaultOut", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func polymorphicOut<T: TensorFlowScalar>(
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("PolymorphicOut", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func populationCount<T: TensorFlowInteger>(
    _ x: Tensor<T>
) -> Tensor<UInt8> {
  let nOutputs = Int(1)
    let op = makeOp("PopulationCount", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func pow<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Pow", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func prefetchDataset(
    inputDataset: VariantHandle,
    bufferSize: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    slackPeriod: Int64 = 0,
    legacyAutotune: Bool = true
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("PrefetchDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("slack_period", slackPeriod)
    op.updateAttribute("legacy_autotune", legacyAutotune)
    op.addInput(inputDataset)
    op.addInput(bufferSize)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func prelinearize<Dtype: TensorFlowScalar>(
    _ input: Tensor<Dtype>,
    shape: TensorShape?,
    layout: [Int32]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("Prelinearize", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape", shape)
    op.updateAttribute("layout", layout)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func prelinearizeTuple<Dtypes: TensorArrayProtocol>(
    inputs: Dtypes,
    shapes: [TensorShape?],
    layouts: [Int32]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("PrelinearizeTuple", nOutputs)
    op.updateAttribute("dtypes", inputs._typeList)
    op.updateAttribute("shapes", shapes)
    op.updateAttribute("layouts", layouts)
    op.addInputList(inputs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func preventGradient<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    message: String
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("PreventGradient", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("message", message)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func print<
    T: TensorFlowScalar,
    U: TensorArrayProtocol
>(
    _ input: Tensor<T>,
    data: U,
    message: String,
    firstN: Int64 = -1,
    summarize: Int64 = 3
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Print", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("U", data._typeList)
    op.updateAttribute("message", message)
    op.updateAttribute("first_n", firstN)
    op.updateAttribute("summarize", summarize)
    op.addInput(input)
    op.addInputList(data)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func printV2(
    _ input: StringTensor,
    outputStream: String = "stderr",
    end: String = "
"
) {
  let nOutputs = 0
    let op = makeOp("PrintV2", nOutputs)
    op.updateAttribute("output_stream", outputStream)
    op.updateAttribute("end", end)
    op.addInput(input)
    op.execute()
}

@inlinable @inline(__always)
public static func priorityQueueV2(
    componentTypes: [TensorDataType],
    shapes: [TensorShape?],
    capacity: Int64 = -1,
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("PriorityQueueV2", nOutputs)
    op.updateAttribute("component_types", componentTypes)
    op.updateAttribute("shapes", shapes)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func privateThreadPoolDataset(
    inputDataset: VariantHandle,
    numThreads: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("PrivateThreadPoolDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(numThreads)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func prod<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
>(
    _ input: Tensor<T>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Prod", nOutputs)
    op.updateAttribute("keep_dims", keepDims)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(input)
    op.addInput(reductionIndices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func pyFunc<
    Tin: TensorArrayProtocol,
    Tout: TensorGroup
>(
    _ input: Tin,
    token: String
) -> Tout {
  let nOutputs = Int(Tout._typeList.count)
    let op = makeOp("PyFunc", nOutputs)
    op.updateAttribute("token", token)
    op.updateAttribute("Tin", input._typeList)
    op.updateAttribute("Tout", Tout._typeList)
    op.addInputList(input)
    return op.execute(Int(Tout._typeList.count))
}

@inlinable @inline(__always)
public static func pyFuncStateless<
    Tin: TensorArrayProtocol,
    Tout: TensorGroup
>(
    _ input: Tin,
    token: String
) -> Tout {
  let nOutputs = Int(Tout._typeList.count)
    let op = makeOp("PyFuncStateless", nOutputs)
    op.updateAttribute("token", token)
    op.updateAttribute("Tin", input._typeList)
    op.updateAttribute("Tout", Tout._typeList)
    op.addInputList(input)
    return op.execute(Int(Tout._typeList.count))
}

@inlinable @inline(__always)
public static func qr<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    fullMatrices: Bool = false
) -> (q: Tensor<T>, r: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("Qr", nOutputs)
    op.updateAttribute("full_matrices", fullMatrices)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizeAndDequantize<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    signedInput: Bool = true,
    numBits: Int64 = 8,
    rangeGiven: Bool = false,
    inputMin: Double = 0,
    inputMax: Double = 0
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("QuantizeAndDequantize", nOutputs)
    op.updateAttribute("signed_input", signedInput)
    op.updateAttribute("num_bits", numBits)
    op.updateAttribute("range_given", rangeGiven)
    op.updateAttribute("input_min", inputMin)
    op.updateAttribute("input_max", inputMax)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func quantizeAndDequantizeV2<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    inputMin: Tensor<T>,
    inputMax: Tensor<T>,
    signedInput: Bool = true,
    numBits: Int64 = 8,
    rangeGiven: Bool = false,
    roundMode: RoundMode = .halfToEven,
    narrowRange: Bool = false,
    axis: Int64 = -1
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("QuantizeAndDequantizeV2", nOutputs)
    op.updateAttribute("signed_input", signedInput)
    op.updateAttribute("num_bits", numBits)
    op.updateAttribute("range_given", rangeGiven)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("round_mode", roundMode.cName)
    op.updateAttribute("narrow_range", narrowRange)
    op.updateAttribute("axis", axis)
    op.addInput(input)
    op.addInput(inputMin)
    op.addInput(inputMax)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func quantizeAndDequantizeV3<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    inputMin: Tensor<T>,
    inputMax: Tensor<T>,
    numBits: Tensor<Int32>,
    signedInput: Bool = true,
    rangeGiven: Bool = true,
    narrowRange: Bool = false,
    axis: Int64 = -1
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("QuantizeAndDequantizeV3", nOutputs)
    op.updateAttribute("signed_input", signedInput)
    op.updateAttribute("range_given", rangeGiven)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("narrow_range", narrowRange)
    op.updateAttribute("axis", axis)
    op.addInput(input)
    op.addInput(inputMin)
    op.addInput(inputMax)
    op.addInput(numBits)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func quantizeDownAndShrinkRange<
    Tinput: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    inputMin: Tensor<Float>,
    inputMax: Tensor<Float>
) -> (output: Tensor<OutType>, outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizeDownAndShrinkRange", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(input)
    op.addInput(inputMin)
    op.addInput(inputMax)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizeV2<T: TensorFlowScalar>(
    _ input: Tensor<Float>,
    minRange: Tensor<Float>,
    maxRange: Tensor<Float>,
    mode: Mode = .minCombined,
    roundMode: RoundMode7 = .halfAwayFromZero,
    narrowRange: Bool = false,
    axis: Int64 = -1,
    ensureMinimumRange: Double = 0.01
) -> (output: Tensor<T>, outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizeV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("mode", mode.cName)
    op.updateAttribute("round_mode", roundMode.cName)
    op.updateAttribute("narrow_range", narrowRange)
    op.updateAttribute("axis", axis)
    op.updateAttribute("ensure_minimum_range", ensureMinimumRange)
    op.addInput(input)
    op.addInput(minRange)
    op.addInput(maxRange)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedAdd<
    T1: TensorFlowScalar,
    T2: TensorFlowScalar,
    Toutput: TensorFlowScalar
>(
    _ x: Tensor<T1>,
    _ y: Tensor<T2>,
    minX: Tensor<Float>,
    maxX: Tensor<Float>,
    minY: Tensor<Float>,
    maxY: Tensor<Float>
) -> (z: Tensor<Toutput>, minZ: Tensor<Float>, maxZ: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedAdd", nOutputs)
    op.updateAttribute("T1", T1.tensorFlowDataType)
    op.updateAttribute("T2", T2.tensorFlowDataType)
    op.updateAttribute("Toutput", Toutput.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    op.addInput(minX)
    op.addInput(maxX)
    op.addInput(minY)
    op.addInput(maxY)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedAvgPool<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding
) -> (output: Tensor<T>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedAvgPool", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.addInput(input)
    op.addInput(minInput)
    op.addInput(maxInput)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedBatchNormWithGlobalNormalization<
    Tinput: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    t: Tensor<Tinput>,
    tMin: Tensor<Float>,
    tMax: Tensor<Float>,
    m: Tensor<Tinput>,
    mMin: Tensor<Float>,
    mMax: Tensor<Float>,
    v: Tensor<Tinput>,
    vMin: Tensor<Float>,
    vMax: Tensor<Float>,
    beta: Tensor<Tinput>,
    betaMin: Tensor<Float>,
    betaMax: Tensor<Float>,
    gamma: Tensor<Tinput>,
    gammaMin: Tensor<Float>,
    gammaMax: Tensor<Float>,
    varianceEpsilon: Double,
    scaleAfterNormalization: Bool
) -> (result: Tensor<OutType>, resultMin: Tensor<Float>, resultMax: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedBatchNormWithGlobalNormalization", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("variance_epsilon", varianceEpsilon)
    op.updateAttribute("scale_after_normalization", scaleAfterNormalization)
    op.addInput(t)
    op.addInput(tMin)
    op.addInput(tMax)
    op.addInput(m)
    op.addInput(mMin)
    op.addInput(mMax)
    op.addInput(v)
    op.addInput(vMin)
    op.addInput(vMax)
    op.addInput(beta)
    op.addInput(betaMin)
    op.addInput(betaMax)
    op.addInput(gamma)
    op.addInput(gammaMin)
    op.addInput(gammaMax)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedBiasAdd<
    T1: TensorFlowScalar,
    T2: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<T1>,
    bias: Tensor<T2>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minBias: Tensor<Float>,
    maxBias: Tensor<Float>
) -> (output: Tensor<OutType>, minOut: Tensor<Float>, maxOut: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedBiasAdd", nOutputs)
    op.updateAttribute("T1", T1.tensorFlowDataType)
    op.updateAttribute("T2", T2.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(input)
    op.addInput(bias)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minBias)
    op.addInput(maxBias)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedConcat<T: TensorFlowScalar>(
    concatDim: Tensor<Int32>,
    _ values: [Tensor<T>],
    inputMins: [Tensor<Float>],
    inputMaxes: [Tensor<Float>]
) -> (output: Tensor<T>, outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedConcat", nOutputs)
    op.updateAttribute("N", values.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(concatDim)
    op.addInputList(values)
    op.addInputList(inputMins)
    op.addInputList(inputMaxes)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedConv2D<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedConv2D", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedConv2DAndRelu<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1],
    paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedConv2DAndRelu", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.updateAttribute("padding_list", paddingList)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedConv2DAndReluAndRequantize<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    minFreezedOutput: Tensor<Float>,
    maxFreezedOutput: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1],
    paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedConv2DAndReluAndRequantize", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.updateAttribute("padding_list", paddingList)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    op.addInput(minFreezedOutput)
    op.addInput(maxFreezedOutput)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedConv2DAndRequantize<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    minFreezedOutput: Tensor<Float>,
    maxFreezedOutput: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1],
    paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedConv2DAndRequantize", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.updateAttribute("padding_list", paddingList)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    op.addInput(minFreezedOutput)
    op.addInput(maxFreezedOutput)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedConv2DPerChannel<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedConv2DPerChannel", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBias<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    bias: Tensor<Float>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1],
    paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedConv2DWithBias", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.updateAttribute("padding_list", paddingList)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(bias)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBiasAndRelu<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    bias: Tensor<Float>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1],
    paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedConv2DWithBiasAndRelu", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.updateAttribute("padding_list", paddingList)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(bias)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBiasAndReluAndRequantize<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    Tbias: FloatingPoint & TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    bias: Tensor<Tbias>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    minFreezedOutput: Tensor<Float>,
    maxFreezedOutput: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1],
    paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedConv2DWithBiasAndReluAndRequantize", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("Tbias", Tbias.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.updateAttribute("padding_list", paddingList)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(bias)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    op.addInput(minFreezedOutput)
    op.addInput(maxFreezedOutput)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBiasAndRequantize<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    Tbias: FloatingPoint & TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    bias: Tensor<Tbias>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    minFreezedOutput: Tensor<Float>,
    maxFreezedOutput: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1],
    paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedConv2DWithBiasAndRequantize", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("Tbias", Tbias.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.updateAttribute("padding_list", paddingList)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(bias)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    op.addInput(minFreezedOutput)
    op.addInput(maxFreezedOutput)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBiasSignedSumAndReluAndRequantize<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    Tbias: FloatingPoint & TensorFlowScalar,
    Tsummand: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    bias: Tensor<Tbias>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    minFreezedOutput: Tensor<Float>,
    maxFreezedOutput: Tensor<Float>,
    summand: Tensor<Tsummand>,
    minSummand: Tensor<Float>,
    maxSummand: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1],
    paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedConv2DWithBiasSignedSumAndReluAndRequantize", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("Tbias", Tbias.tensorFlowDataType)
    op.updateAttribute("Tsummand", Tsummand.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.updateAttribute("padding_list", paddingList)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(bias)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    op.addInput(minFreezedOutput)
    op.addInput(maxFreezedOutput)
    op.addInput(summand)
    op.addInput(minSummand)
    op.addInput(maxSummand)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBiasSumAndRelu<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    bias: Tensor<Float>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    summand: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1],
    paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedConv2DWithBiasSumAndRelu", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.updateAttribute("padding_list", paddingList)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(bias)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    op.addInput(summand)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBiasSumAndReluAndRequantize<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    Tbias: FloatingPoint & TensorFlowScalar,
    Tsummand: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    bias: Tensor<Tbias>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    minFreezedOutput: Tensor<Float>,
    maxFreezedOutput: Tensor<Float>,
    summand: Tensor<Tsummand>,
    minSummand: Tensor<Float>,
    maxSummand: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1],
    paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedConv2DWithBiasSumAndReluAndRequantize", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("Tbias", Tbias.tensorFlowDataType)
    op.updateAttribute("Tsummand", Tsummand.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.updateAttribute("padding_list", paddingList)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(bias)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    op.addInput(minFreezedOutput)
    op.addInput(maxFreezedOutput)
    op.addInput(summand)
    op.addInput(minSummand)
    op.addInput(maxSummand)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedDepthwiseConv2D<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedDepthwiseConv2D", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedDepthwiseConv2DWithBias<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    bias: Tensor<Float>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedDepthwiseConv2DWithBias", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(bias)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedDepthwiseConv2DWithBiasAndRelu<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    bias: Tensor<Float>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedDepthwiseConv2DWithBiasAndRelu", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(bias)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedDepthwiseConv2DWithBiasAndReluAndRequantize<
    Tinput: TensorFlowScalar,
    Tfilter: TensorFlowScalar,
    Tbias: FloatingPoint & TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    filter: Tensor<Tfilter>,
    bias: Tensor<Tbias>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    minFilter: Tensor<Float>,
    maxFilter: Tensor<Float>,
    minFreezedOutput: Tensor<Float>,
    maxFreezedOutput: Tensor<Float>,
    strides: [Int32],
    padding: Padding,
    dilations: [Int32] = [1, 1, 1, 1]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("Tfilter", Tfilter.tensorFlowDataType)
    op.updateAttribute("Tbias", Tbias.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.updateAttribute("dilations", dilations)
    op.addInput(input)
    op.addInput(filter)
    op.addInput(bias)
    op.addInput(minInput)
    op.addInput(maxInput)
    op.addInput(minFilter)
    op.addInput(maxFilter)
    op.addInput(minFreezedOutput)
    op.addInput(maxFreezedOutput)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedInstanceNorm<T: TensorFlowScalar>(
    _ x: Tensor<T>,
    xMin: Tensor<Float>,
    xMax: Tensor<Float>,
    outputRangeGiven: Bool = false,
    givenYMin: Double = 0,
    givenYMax: Double = 0,
    varianceEpsilon: Double = 1e-05,
    minSeparation: Double = 0.001
) -> (y: Tensor<T>, yMin: Tensor<Float>, yMax: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedInstanceNorm", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("output_range_given", outputRangeGiven)
    op.updateAttribute("given_y_min", givenYMin)
    op.updateAttribute("given_y_max", givenYMax)
    op.updateAttribute("variance_epsilon", varianceEpsilon)
    op.updateAttribute("min_separation", minSeparation)
    op.addInput(x)
    op.addInput(xMin)
    op.addInput(xMax)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedMatMul<
    T1: TensorFlowScalar,
    T2: TensorFlowScalar,
    Toutput: TensorFlowScalar
>(
    _ a: Tensor<T1>,
    _ b: Tensor<T2>,
    minA: Tensor<Float>,
    maxA: Tensor<Float>,
    minB: Tensor<Float>,
    maxB: Tensor<Float>,
    transposeA: Bool = false,
    transposeB: Bool = false,
    tactivation: TensorDataType
) -> (out: Tensor<Toutput>, minOut: Tensor<Float>, maxOut: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedMatMul", nOutputs)
    op.updateAttribute("T1", T1.tensorFlowDataType)
    op.updateAttribute("T2", T2.tensorFlowDataType)
    op.updateAttribute("Toutput", Toutput.tensorFlowDataType)
    op.updateAttribute("transpose_a", transposeA)
    op.updateAttribute("transpose_b", transposeB)
    op.updateAttribute("Tactivation", tactivation)
    op.addInput(a)
    op.addInput(b)
    op.addInput(minA)
    op.addInput(maxA)
    op.addInput(minB)
    op.addInput(maxB)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedMatMulWithBias<
    T1: TensorFlowScalar,
    T2: TensorFlowScalar,
    Tbias: FloatingPoint & TensorFlowScalar,
    Toutput: TensorFlowScalar
>(
    _ a: Tensor<T1>,
    _ b: Tensor<T2>,
    bias: Tensor<Tbias>,
    minA: Tensor<Float>,
    maxA: Tensor<Float>,
    minB: Tensor<Float>,
    maxB: Tensor<Float>,
    transposeA: Bool = false,
    transposeB: Bool = false,
    inputQuantMode: InputQuantMode = .minFirst
) -> (out: Tensor<Toutput>, minOut: Tensor<Float>, maxOut: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedMatMulWithBias", nOutputs)
    op.updateAttribute("T1", T1.tensorFlowDataType)
    op.updateAttribute("T2", T2.tensorFlowDataType)
    op.updateAttribute("Tbias", Tbias.tensorFlowDataType)
    op.updateAttribute("Toutput", Toutput.tensorFlowDataType)
    op.updateAttribute("transpose_a", transposeA)
    op.updateAttribute("transpose_b", transposeB)
    op.updateAttribute("input_quant_mode", inputQuantMode.cName)
    op.addInput(a)
    op.addInput(b)
    op.addInput(bias)
    op.addInput(minA)
    op.addInput(maxA)
    op.addInput(minB)
    op.addInput(maxB)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedMatMulWithBiasAndRelu<
    T1: TensorFlowScalar,
    T2: TensorFlowScalar,
    Toutput: TensorFlowScalar
>(
    _ a: Tensor<T1>,
    _ b: Tensor<T2>,
    bias: Tensor<Float>,
    minA: Tensor<Float>,
    maxA: Tensor<Float>,
    minB: Tensor<Float>,
    maxB: Tensor<Float>,
    transposeA: Bool = false,
    transposeB: Bool = false,
    inputQuantMode: InputQuantMode = .minFirst
) -> (out: Tensor<Toutput>, minOut: Tensor<Float>, maxOut: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedMatMulWithBiasAndRelu", nOutputs)
    op.updateAttribute("T1", T1.tensorFlowDataType)
    op.updateAttribute("T2", T2.tensorFlowDataType)
    op.updateAttribute("Toutput", Toutput.tensorFlowDataType)
    op.updateAttribute("transpose_a", transposeA)
    op.updateAttribute("transpose_b", transposeB)
    op.updateAttribute("input_quant_mode", inputQuantMode.cName)
    op.addInput(a)
    op.addInput(b)
    op.addInput(bias)
    op.addInput(minA)
    op.addInput(maxA)
    op.addInput(minB)
    op.addInput(maxB)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedMatMulWithBiasAndReluAndRequantize<
    T1: TensorFlowScalar,
    T2: TensorFlowScalar,
    Tbias: FloatingPoint & TensorFlowScalar,
    Toutput: TensorFlowScalar
>(
    _ a: Tensor<T1>,
    _ b: Tensor<T2>,
    bias: Tensor<Tbias>,
    minA: Tensor<Float>,
    maxA: Tensor<Float>,
    minB: Tensor<Float>,
    maxB: Tensor<Float>,
    minFreezedOutput: Tensor<Float>,
    maxFreezedOutput: Tensor<Float>,
    transposeA: Bool = false,
    transposeB: Bool = false,
    inputQuantMode: InputQuantMode = .minFirst
) -> (out: Tensor<Toutput>, minOut: Tensor<Float>, maxOut: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedMatMulWithBiasAndReluAndRequantize", nOutputs)
    op.updateAttribute("T1", T1.tensorFlowDataType)
    op.updateAttribute("T2", T2.tensorFlowDataType)
    op.updateAttribute("Tbias", Tbias.tensorFlowDataType)
    op.updateAttribute("Toutput", Toutput.tensorFlowDataType)
    op.updateAttribute("transpose_a", transposeA)
    op.updateAttribute("transpose_b", transposeB)
    op.updateAttribute("input_quant_mode", inputQuantMode.cName)
    op.addInput(a)
    op.addInput(b)
    op.addInput(bias)
    op.addInput(minA)
    op.addInput(maxA)
    op.addInput(minB)
    op.addInput(maxB)
    op.addInput(minFreezedOutput)
    op.addInput(maxFreezedOutput)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedMaxPool<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    minInput: Tensor<Float>,
    maxInput: Tensor<Float>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding
) -> (output: Tensor<T>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedMaxPool", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("ksize", ksize)
    op.updateAttribute("strides", strides)
    op.updateAttribute("padding", padding.cName)
    op.addInput(input)
    op.addInput(minInput)
    op.addInput(maxInput)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedMul<
    T1: TensorFlowScalar,
    T2: TensorFlowScalar,
    Toutput: TensorFlowScalar
>(
    _ x: Tensor<T1>,
    _ y: Tensor<T2>,
    minX: Tensor<Float>,
    maxX: Tensor<Float>,
    minY: Tensor<Float>,
    maxY: Tensor<Float>
) -> (z: Tensor<Toutput>, minZ: Tensor<Float>, maxZ: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedMul", nOutputs)
    op.updateAttribute("T1", T1.tensorFlowDataType)
    op.updateAttribute("T2", T2.tensorFlowDataType)
    op.updateAttribute("Toutput", Toutput.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    op.addInput(minX)
    op.addInput(maxX)
    op.addInput(minY)
    op.addInput(maxY)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedRelu<
    Tinput: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    features: Tensor<Tinput>,
    minFeatures: Tensor<Float>,
    maxFeatures: Tensor<Float>
) -> (activations: Tensor<OutType>, minActivations: Tensor<Float>, maxActivations: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedRelu", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(features)
    op.addInput(minFeatures)
    op.addInput(maxFeatures)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedRelu6<
    Tinput: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    features: Tensor<Tinput>,
    minFeatures: Tensor<Float>,
    maxFeatures: Tensor<Float>
) -> (activations: Tensor<OutType>, minActivations: Tensor<Float>, maxActivations: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedRelu6", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(features)
    op.addInput(minFeatures)
    op.addInput(maxFeatures)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedReluX<
    Tinput: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    features: Tensor<Tinput>,
    maxValue: Tensor<Float>,
    minFeatures: Tensor<Float>,
    maxFeatures: Tensor<Float>
) -> (activations: Tensor<OutType>, minActivations: Tensor<Float>, maxActivations: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedReluX", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(features)
    op.addInput(maxValue)
    op.addInput(minFeatures)
    op.addInput(maxFeatures)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedReshape<
    T: TensorFlowScalar,
    Tshape: TensorFlowIndex
>(
    _ tensor: Tensor<T>,
    shape: Tensor<Tshape>,
    inputMin: Tensor<Float>,
    inputMax: Tensor<Float>
) -> (output: Tensor<T>, outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedReshape", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tshape", Tshape.tensorFlowDataType)
    op.addInput(tensor)
    op.addInput(shape)
    op.addInput(inputMin)
    op.addInput(inputMax)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func quantizedResizeBilinear<T: FloatingPoint & TensorFlowScalar>(
    images: Tensor<T>,
    size: Tensor<Int32>,
    min: Tensor<Float>,
    max: Tensor<Float>,
    alignCorners: Bool = false,
    halfPixelCenters: Bool = false
) -> (resizedImages: Tensor<T>, outMin: Tensor<Float>, outMax: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("QuantizedResizeBilinear", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("align_corners", alignCorners)
    op.updateAttribute("half_pixel_centers", halfPixelCenters)
    op.addInput(images)
    op.addInput(size)
    op.addInput(min)
    op.addInput(max)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func queueCloseV2(
    handle: ResourceHandle,
    cancelPendingEnqueues: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("QueueCloseV2", nOutputs)
    op.updateAttribute("cancel_pending_enqueues", cancelPendingEnqueues)
    op.addInput(handle)
    op.execute()
}

@inlinable @inline(__always)
public static func queueDequeueManyV2<ComponentTypes: TensorGroup>(
    handle: ResourceHandle,
    n: Tensor<Int32>,
    timeoutMs: Int64 = -1
) -> ComponentTypes {
  let nOutputs = Int(ComponentTypes._typeList.count)
    let op = makeOp("QueueDequeueManyV2", nOutputs)
    op.updateAttribute("component_types", ComponentTypes._typeList)
    op.updateAttribute("timeout_ms", timeoutMs)
    op.addInput(handle)
    op.addInput(n)
    return op.execute(Int(ComponentTypes._typeList.count))
}

@inlinable @inline(__always)
public static func queueDequeueUpToV2<ComponentTypes: TensorGroup>(
    handle: ResourceHandle,
    n: Tensor<Int32>,
    timeoutMs: Int64 = -1
) -> ComponentTypes {
  let nOutputs = Int(ComponentTypes._typeList.count)
    let op = makeOp("QueueDequeueUpToV2", nOutputs)
    op.updateAttribute("component_types", ComponentTypes._typeList)
    op.updateAttribute("timeout_ms", timeoutMs)
    op.addInput(handle)
    op.addInput(n)
    return op.execute(Int(ComponentTypes._typeList.count))
}

@inlinable @inline(__always)
public static func queueDequeueV2<ComponentTypes: TensorGroup>(
    handle: ResourceHandle,
    timeoutMs: Int64 = -1
) -> ComponentTypes {
  let nOutputs = Int(ComponentTypes._typeList.count)
    let op = makeOp("QueueDequeueV2", nOutputs)
    op.updateAttribute("component_types", ComponentTypes._typeList)
    op.updateAttribute("timeout_ms", timeoutMs)
    op.addInput(handle)
    return op.execute(Int(ComponentTypes._typeList.count))
}

@inlinable @inline(__always)
public static func queueEnqueueManyV2<Tcomponents: TensorArrayProtocol>(
    handle: ResourceHandle,
    components: Tcomponents,
    timeoutMs: Int64 = -1
) {
  let nOutputs = 0
    let op = makeOp("QueueEnqueueManyV2", nOutputs)
    op.updateAttribute("Tcomponents", components._typeList)
    op.updateAttribute("timeout_ms", timeoutMs)
    op.addInput(handle)
    op.addInputList(components)
    op.execute()
}

@inlinable @inline(__always)
public static func queueEnqueueV2<Tcomponents: TensorArrayProtocol>(
    handle: ResourceHandle,
    components: Tcomponents,
    timeoutMs: Int64 = -1
) {
  let nOutputs = 0
    let op = makeOp("QueueEnqueueV2", nOutputs)
    op.updateAttribute("Tcomponents", components._typeList)
    op.updateAttribute("timeout_ms", timeoutMs)
    op.addInput(handle)
    op.addInputList(components)
    op.execute()
}

@inlinable @inline(__always)
public static func queueIsClosedV2(
    handle: ResourceHandle
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("QueueIsClosedV2", nOutputs)
    op.addInput(handle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func queueSizeV2(
    handle: ResourceHandle
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("QueueSizeV2", nOutputs)
    op.addInput(handle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func rFFT<
    Treal: FloatingPoint & TensorFlowScalar,
    Tcomplex: TensorFlowScalar
>(
    _ input: Tensor<Treal>,
    fftLength: Tensor<Int32>
) -> Tensor<Tcomplex> {
  let nOutputs = Int(1)
    let op = makeOp("RFFT", nOutputs)
    op.updateAttribute("Treal", Treal.tensorFlowDataType)
    op.updateAttribute("Tcomplex", Tcomplex.tensorFlowDataType)
    op.addInput(input)
    op.addInput(fftLength)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func rFFT2D<
    Treal: FloatingPoint & TensorFlowScalar,
    Tcomplex: TensorFlowScalar
>(
    _ input: Tensor<Treal>,
    fftLength: Tensor<Int32>
) -> Tensor<Tcomplex> {
  let nOutputs = Int(1)
    let op = makeOp("RFFT2D", nOutputs)
    op.updateAttribute("Treal", Treal.tensorFlowDataType)
    op.updateAttribute("Tcomplex", Tcomplex.tensorFlowDataType)
    op.addInput(input)
    op.addInput(fftLength)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func rFFT3D<
    Treal: FloatingPoint & TensorFlowScalar,
    Tcomplex: TensorFlowScalar
>(
    _ input: Tensor<Treal>,
    fftLength: Tensor<Int32>
) -> Tensor<Tcomplex> {
  let nOutputs = Int(1)
    let op = makeOp("RFFT3D", nOutputs)
    op.updateAttribute("Treal", Treal.tensorFlowDataType)
    op.updateAttribute("Tcomplex", Tcomplex.tensorFlowDataType)
    op.addInput(input)
    op.addInput(fftLength)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func rGBToHSV<T: FloatingPoint & TensorFlowScalar>(
    images: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("RGBToHSV", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(images)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func raggedGather<
    Tvalues: TensorFlowScalar,
    Tindices: TensorFlowIndex,
    Tsplits: TensorFlowIndex
>(
    paramsNestedSplits: [Tensor<Tsplits>],
    paramsDenseValues: Tensor<Tvalues>,
    indices: Tensor<Tindices>,
    oUTPUTRAGGEDRANK: Int64
) -> (outputNestedSplits: [Tensor<Tsplits>], outputDenseValues: Tensor<Tvalues>) {
  let nOutputs = Int(oUTPUTRAGGEDRANK) + Int(1)
    let op = makeOp("RaggedGather", nOutputs)
    op.updateAttribute("Tvalues", Tvalues.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("Tsplits", Tsplits.tensorFlowDataType)
    op.updateAttribute("PARAMS_RAGGED_RANK", paramsNestedSplits.count)
    op.updateAttribute("OUTPUT_RAGGED_RANK", oUTPUTRAGGEDRANK)
    op.addInputList(paramsNestedSplits)
    op.addInput(paramsDenseValues)
    op.addInput(indices)
    return op.execute(Int(oUTPUTRAGGEDRANK), Int(1))
}

@inlinable @inline(__always)
public static func raggedRange<
    T: TensorFlowNumeric,
    Tsplits: TensorFlowIndex
>(
    starts: Tensor<T>,
    limits: Tensor<T>,
    deltas: Tensor<T>
) -> (rtNestedSplits: Tensor<Tsplits>, rtDenseValues: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("RaggedRange", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tsplits", Tsplits.tensorFlowDataType)
    op.addInput(starts)
    op.addInput(limits)
    op.addInput(deltas)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func raggedTensorFromVariant<
    Tvalues: TensorFlowScalar,
    Tsplits: TensorFlowIndex
>(
    encodedRagged: VariantHandle,
    inputRaggedRank: Int64,
    outputRaggedRank: Int64
) -> (outputNestedSplits: [Tensor<Tsplits>], outputDenseValues: Tensor<Tvalues>) {
  let nOutputs = Int(outputRaggedRank) + Int(1)
    let op = makeOp("RaggedTensorFromVariant", nOutputs)
    op.updateAttribute("input_ragged_rank", inputRaggedRank)
    op.updateAttribute("output_ragged_rank", outputRaggedRank)
    op.updateAttribute("Tvalues", Tvalues.tensorFlowDataType)
    op.updateAttribute("Tsplits", Tsplits.tensorFlowDataType)
    op.addInput(encodedRagged)
    return op.execute(Int(outputRaggedRank), Int(1))
}

@inlinable @inline(__always)
public static func raggedTensorToSparse<
    T: TensorFlowScalar,
    Tsplits: TensorFlowIndex
>(
    rtNestedSplits: [Tensor<Tsplits>],
    rtDenseValues: Tensor<T>
) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<T>, sparseDenseShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("RaggedTensorToSparse", nOutputs)
    op.updateAttribute("RAGGED_RANK", rtNestedSplits.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tsplits", Tsplits.tensorFlowDataType)
    op.addInputList(rtNestedSplits)
    op.addInput(rtDenseValues)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func raggedTensorToTensor<
    T: TensorFlowScalar,
    Tindex: TensorFlowIndex,
    Tshape: TensorFlowIndex
>(
    shape: Tensor<Tshape>,
    _ values: Tensor<T>,
    defaultValue: Tensor<T>,
    rowPartitionTensors: [Tensor<Tindex>],
    rowPartitionTypes: [String]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("RaggedTensorToTensor", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindex", Tindex.tensorFlowDataType)
    op.updateAttribute("Tshape", Tshape.tensorFlowDataType)
    op.updateAttribute("num_row_partition_tensors", rowPartitionTensors.count)
    op.updateAttribute("row_partition_types", rowPartitionTypes)
    op.addInput(shape)
    op.addInput(values)
    op.addInput(defaultValue)
    op.addInputList(rowPartitionTensors)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func raggedTensorToVariant<
    Tvalues: TensorFlowScalar,
    Tsplits: TensorFlowIndex
>(
    rtNestedSplits: [Tensor<Tsplits>],
    rtDenseValues: Tensor<Tvalues>,
    batchedInput: Bool
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("RaggedTensorToVariant", nOutputs)
    op.updateAttribute("RAGGED_RANK", rtNestedSplits.count)
    op.updateAttribute("Tvalues", Tvalues.tensorFlowDataType)
    op.updateAttribute("Tsplits", Tsplits.tensorFlowDataType)
    op.updateAttribute("batched_input", batchedInput)
    op.addInputList(rtNestedSplits)
    op.addInput(rtDenseValues)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func randomCrop<T: TensorFlowNumeric>(
    image: Tensor<T>,
    size: Tensor<Int64>,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("RandomCrop", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.addInput(image)
    op.addInput(size)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func randomDataset(
    seed: Tensor<Int64>,
    seed2: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("RandomDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(seed)
    op.addInput(seed2)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func randomGamma<
    S: TensorFlowIndex,
    T: FloatingPoint & TensorFlowScalar
>(
    shape: Tensor<S>,
    alpha: Tensor<T>,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("RandomGamma", nOutputs)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("S", S.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(shape)
    op.addInput(alpha)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func randomGammaGrad<T: FloatingPoint & TensorFlowScalar>(
    alpha: Tensor<T>,
    sample: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("RandomGammaGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(alpha)
    op.addInput(sample)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func randomPoisson<
    S: TensorFlowIndex,
    Dtype: FloatingPoint & TensorFlowScalar
>(
    shape: Tensor<S>,
    rate: Tensor<Dtype>,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("RandomPoisson", nOutputs)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("S", S.tensorFlowDataType)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(shape)
    op.addInput(rate)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func randomPoissonV2<
    S: TensorFlowIndex,
    R: TensorFlowNumeric,
    Dtype: TensorFlowNumeric
>(
    shape: Tensor<S>,
    rate: Tensor<R>,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("RandomPoissonV2", nOutputs)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("S", S.tensorFlowDataType)
    op.updateAttribute("R", R.tensorFlowDataType)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(shape)
    op.addInput(rate)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func randomShuffle<T: TensorFlowScalar>(
    value: Tensor<T>,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("RandomShuffle", nOutputs)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(value)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func randomShuffleQueueV2(
    componentTypes: [TensorDataType],
    shapes: [TensorShape?],
    capacity: Int64 = -1,
    minAfterDequeue: Int64 = 0,
    seed: Int64 = 0,
    seed2: Int64 = 0,
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("RandomShuffleQueueV2", nOutputs)
    op.updateAttribute("component_types", componentTypes)
    op.updateAttribute("shapes", shapes)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("min_after_dequeue", minAfterDequeue)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func randomStandardNormal<
    Dtype: FloatingPoint & TensorFlowScalar,
    T: TensorFlowIndex
>(
    shape: Tensor<T>,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("RandomStandardNormal", nOutputs)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func randomUniform<
    Dtype: FloatingPoint & TensorFlowScalar,
    T: TensorFlowIndex
>(
    shape: Tensor<T>,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("RandomUniform", nOutputs)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func randomUniformInt<
    Tout: TensorFlowIndex,
    T: TensorFlowIndex
>(
    shape: Tensor<T>,
    minval: Tensor<Tout>,
    maxval: Tensor<Tout>,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> Tensor<Tout> {
  let nOutputs = Int(1)
    let op = makeOp("RandomUniformInt", nOutputs)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("Tout", Tout.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(shape)
    op.addInput(minval)
    op.addInput(maxval)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func range<Tidx: TensorFlowNumeric>(
    start: Tensor<Tidx>,
    limit: Tensor<Tidx>,
    delta: Tensor<Tidx>
) -> Tensor<Tidx> {
  let nOutputs = Int(1)
    let op = makeOp("Range", nOutputs)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(start)
    op.addInput(limit)
    op.addInput(delta)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func rangeDataset(
    start: Tensor<Int64>,
    stop: Tensor<Int64>,
    step: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("RangeDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(start)
    op.addInput(stop)
    op.addInput(step)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func rank<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("Rank", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func readFile(
    filename: StringTensor
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("ReadFile", nOutputs)
    op.addInput(filename)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func readVariableOp<Dtype: TensorFlowScalar>(
    resource: ResourceHandle
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("ReadVariableOp", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(resource)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func readerNumRecordsProducedV2(
    readerHandle: ResourceHandle
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("ReaderNumRecordsProducedV2", nOutputs)
    op.addInput(readerHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func readerNumWorkUnitsCompletedV2(
    readerHandle: ResourceHandle
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("ReaderNumWorkUnitsCompletedV2", nOutputs)
    op.addInput(readerHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func readerReadUpToV2(
    readerHandle: ResourceHandle,
    queueHandle: ResourceHandle,
    numRecords: Tensor<Int64>
) -> (keys: StringTensor, values: StringTensor) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("ReaderReadUpToV2", nOutputs)
    op.addInput(readerHandle)
    op.addInput(queueHandle)
    op.addInput(numRecords)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func readerReadV2(
    readerHandle: ResourceHandle,
    queueHandle: ResourceHandle
) -> (key: StringTensor, value: StringTensor) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("ReaderReadV2", nOutputs)
    op.addInput(readerHandle)
    op.addInput(queueHandle)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func readerResetV2(
    readerHandle: ResourceHandle
) {
  let nOutputs = 0
    let op = makeOp("ReaderResetV2", nOutputs)
    op.addInput(readerHandle)
    op.execute()
}

@inlinable @inline(__always)
public static func readerRestoreStateV2(
    readerHandle: ResourceHandle,
    state: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("ReaderRestoreStateV2", nOutputs)
    op.addInput(readerHandle)
    op.addInput(state)
    op.execute()
}

@inlinable @inline(__always)
public static func readerSerializeStateV2(
    readerHandle: ResourceHandle
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("ReaderSerializeStateV2", nOutputs)
    op.addInput(readerHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func real<
    T: TensorFlowScalar,
    Tout: FloatingPoint & TensorFlowScalar
>(
    _ input: Tensor<T>
) -> Tensor<Tout> {
  let nOutputs = Int(1)
    let op = makeOp("Real", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tout", Tout.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func realDiv<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("RealDiv", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func rebatchDataset(
    inputDataset: VariantHandle,
    numReplicas: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    useFallback: Bool = true
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("RebatchDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("use_fallback", useFallback)
    op.addInput(inputDataset)
    op.addInput(numReplicas)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func reciprocal<T: TensorFlowNumeric>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Reciprocal", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func reciprocalGrad<T: FloatingPoint & TensorFlowScalar>(
    _ y: Tensor<T>,
    dy: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ReciprocalGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(y)
    op.addInput(dy)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func recordInput(
    filePattern: String,
    fileRandomSeed: Int64 = 301,
    fileShuffleShiftRatio: Double = 0,
    fileBufferSize: Int64 = 10000,
    fileParallelism: Int64 = 16,
    batchSize: Int64 = 32,
    compressionType: String
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("RecordInput", nOutputs)
    op.updateAttribute("file_pattern", filePattern)
    op.updateAttribute("file_random_seed", fileRandomSeed)
    op.updateAttribute("file_shuffle_shift_ratio", fileShuffleShiftRatio)
    op.updateAttribute("file_buffer_size", fileBufferSize)
    op.updateAttribute("file_parallelism", fileParallelism)
    op.updateAttribute("batch_size", batchSize)
    op.updateAttribute("compression_type", compressionType)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func recv<TensorType: TensorFlowScalar>(
    tensorName: String,
    sendDevice: String,
    sendDeviceIncarnation: Int64,
    recvDevice: String,
    clientTerminated: Bool = false
) -> Tensor<TensorType> {
  let nOutputs = Int(1)
    let op = makeOp("Recv", nOutputs)
    op.updateAttribute("tensor_type", TensorType.tensorFlowDataType)
    op.updateAttribute("tensor_name", tensorName)
    op.updateAttribute("send_device", sendDevice)
    op.updateAttribute("send_device_incarnation", sendDeviceIncarnation)
    op.updateAttribute("recv_device", recvDevice)
    op.updateAttribute("client_terminated", clientTerminated)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func recvTPUEmbeddingActivations(
    numOutputs: Int64,
    config: String
) -> [Tensor<Float>] {
  let nOutputs = Int(numOutputs)
    let op = makeOp("RecvTPUEmbeddingActivations", nOutputs)
    op.updateAttribute("num_outputs", numOutputs)
    op.updateAttribute("config", config)
    return op.execute(Int(numOutputs))
}

@inlinable @inline(__always)
public static func reduceDataset<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Tstate: TensorArrayProtocol,
    Targuments: TensorArrayProtocol,
    OutputTypes: TensorGroup
>(
    inputDataset: VariantHandle,
    initialState: Tstate,
    otherArguments: Targuments,
    f: (FIn) -> FOut,
    outputShapes: [TensorShape?],
    useInterOpParallelism: Bool = true
) -> OutputTypes {
  let nOutputs = Int(OutputTypes._typeList.count)
    let op = makeOp("ReduceDataset", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("Tstate", initialState._typeList)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", OutputTypes._typeList)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("use_inter_op_parallelism", useInterOpParallelism)
    op.addInput(inputDataset)
    op.addInputList(initialState)
    op.addInputList(otherArguments)
    return op.execute(Int(OutputTypes._typeList.count))
}

@inlinable @inline(__always)
public static func reduceJoin(
    inputs: StringTensor,
    reductionIndices: Tensor<Int32>,
    keepDims: Bool = false,
    separator: String
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("ReduceJoin", nOutputs)
    op.updateAttribute("keep_dims", keepDims)
    op.updateAttribute("separator", separator)
    op.addInput(inputs)
    op.addInput(reductionIndices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func regexFullMatch(
    _ input: StringTensor,
    pattern: StringTensor
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("RegexFullMatch", nOutputs)
    op.addInput(input)
    op.addInput(pattern)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func regexReplace(
    _ input: StringTensor,
    pattern: StringTensor,
    rewrite: StringTensor,
    replaceGlobal: Bool = true
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("RegexReplace", nOutputs)
    op.updateAttribute("replace_global", replaceGlobal)
    op.addInput(input)
    op.addInput(pattern)
    op.addInput(rewrite)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func relu<T: TensorFlowNumeric>(
    features: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Relu", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(features)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func relu6<T: TensorFlowNumeric>(
    features: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Relu6", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(features)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func relu6Grad<T: TensorFlowNumeric>(
    gradients: Tensor<T>,
    features: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Relu6Grad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(gradients)
    op.addInput(features)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func reluGrad<T: TensorFlowNumeric>(
    gradients: Tensor<T>,
    features: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ReluGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(gradients)
    op.addInput(features)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func remoteCall<
    Tin: TensorArrayProtocol,
    Tout: TensorGroup,
    FIn: TensorGroup,
    FOut: TensorGroup
>(
    target: StringTensor,
    args: Tin,
    f: (FIn) -> FOut
) -> Tout {
  let nOutputs = Int(Tout._typeList.count)
    let op = makeOp("RemoteCall", nOutputs)
    op.updateAttribute("Tin", args._typeList)
    op.updateAttribute("Tout", Tout._typeList)
    op.updateAttribute("f", f)
    op.addInput(target)
    op.addInputList(args)
    return op.execute(Int(Tout._typeList.count))
}

@inlinable @inline(__always)
public static func remoteFusedGraphExecute<
    Tinputs: TensorArrayProtocol,
    Toutputs: TensorGroup
>(
    inputs: Tinputs,
    serializedRemoteFusedGraphExecuteInfo: String
) -> Toutputs {
  let nOutputs = Int(Toutputs._typeList.count)
    let op = makeOp("RemoteFusedGraphExecute", nOutputs)
    op.updateAttribute("Tinputs", inputs._typeList)
    op.updateAttribute("Toutputs", Toutputs._typeList)
    op.updateAttribute("serialized_remote_fused_graph_execute_info", serializedRemoteFusedGraphExecuteInfo)
    op.addInputList(inputs)
    return op.execute(Int(Toutputs._typeList.count))
}

@inlinable @inline(__always)
public static func repeatDataset(
    inputDataset: VariantHandle,
    count: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("RepeatDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(count)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func requantizationRange<Tinput: TensorFlowScalar>(
    _ input: Tensor<Tinput>,
    inputMin: Tensor<Float>,
    inputMax: Tensor<Float>
) -> (outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("RequantizationRange", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.addInput(input)
    op.addInput(inputMin)
    op.addInput(inputMax)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func requantizationRangePerChannel<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    inputMin: Tensor<Float>,
    inputMax: Tensor<Float>,
    clipValueMax: Double
) -> (outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("RequantizationRangePerChannel", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("clip_value_max", clipValueMax)
    op.addInput(input)
    op.addInput(inputMin)
    op.addInput(inputMax)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func requantize<
    Tinput: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<Tinput>,
    inputMin: Tensor<Float>,
    inputMax: Tensor<Float>,
    requestedOutputMin: Tensor<Float>,
    requestedOutputMax: Tensor<Float>
) -> (output: Tensor<OutType>, outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("Requantize", nOutputs)
    op.updateAttribute("Tinput", Tinput.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(input)
    op.addInput(inputMin)
    op.addInput(inputMax)
    op.addInput(requestedOutputMin)
    op.addInput(requestedOutputMax)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func requantizePerChannel<
    T: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    _ input: Tensor<T>,
    inputMin: Tensor<Float>,
    inputMax: Tensor<Float>,
    requestedOutputMin: Tensor<Float>,
    requestedOutputMax: Tensor<Float>
) -> (output: Tensor<OutType>, outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("RequantizePerChannel", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(input)
    op.addInput(inputMin)
    op.addInput(inputMax)
    op.addInput(requestedOutputMin)
    op.addInput(requestedOutputMax)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func requiresOlderGraphVersion(
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("RequiresOlderGraphVersion", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func reservedAttr(
    range: Int64
) {
  let nOutputs = 0
    let op = makeOp("ReservedAttr", nOutputs)
    op.updateAttribute("range", range)
    op.execute()
}

@inlinable @inline(__always)
public static func reservedInput(
    _ input: Tensor<Int32>
) {
  let nOutputs = 0
    let op = makeOp("ReservedInput", nOutputs)
    op.addInput(input)
    op.execute()
}

@inlinable @inline(__always)
public static func reshape<
    T: TensorFlowScalar,
    Tshape: TensorFlowIndex
>(
    _ tensor: Tensor<T>,
    shape: Tensor<Tshape>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Reshape", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tshape", Tshape.tensorFlowDataType)
    op.addInput(tensor)
    op.addInput(shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resizeArea<T: TensorFlowNumeric>(
    images: Tensor<T>,
    size: Tensor<Int32>,
    alignCorners: Bool = false
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("ResizeArea", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("align_corners", alignCorners)
    op.addInput(images)
    op.addInput(size)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resizeBicubic<T: TensorFlowNumeric>(
    images: Tensor<T>,
    size: Tensor<Int32>,
    alignCorners: Bool = false,
    halfPixelCenters: Bool = false
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("ResizeBicubic", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("align_corners", alignCorners)
    op.updateAttribute("half_pixel_centers", halfPixelCenters)
    op.addInput(images)
    op.addInput(size)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resizeBicubicGrad<T: FloatingPoint & TensorFlowScalar>(
    grads: Tensor<Float>,
    originalImage: Tensor<T>,
    alignCorners: Bool = false,
    halfPixelCenters: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ResizeBicubicGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("align_corners", alignCorners)
    op.updateAttribute("half_pixel_centers", halfPixelCenters)
    op.addInput(grads)
    op.addInput(originalImage)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resizeBilinear<T: TensorFlowNumeric>(
    images: Tensor<T>,
    size: Tensor<Int32>,
    alignCorners: Bool = false,
    halfPixelCenters: Bool = false
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("ResizeBilinear", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("align_corners", alignCorners)
    op.updateAttribute("half_pixel_centers", halfPixelCenters)
    op.addInput(images)
    op.addInput(size)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resizeBilinearGrad<T: FloatingPoint & TensorFlowScalar>(
    grads: Tensor<Float>,
    originalImage: Tensor<T>,
    alignCorners: Bool = false,
    halfPixelCenters: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ResizeBilinearGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("align_corners", alignCorners)
    op.updateAttribute("half_pixel_centers", halfPixelCenters)
    op.addInput(grads)
    op.addInput(originalImage)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resizeNearestNeighbor<T: TensorFlowNumeric>(
    images: Tensor<T>,
    size: Tensor<Int32>,
    alignCorners: Bool = false,
    halfPixelCenters: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ResizeNearestNeighbor", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("align_corners", alignCorners)
    op.updateAttribute("half_pixel_centers", halfPixelCenters)
    op.addInput(images)
    op.addInput(size)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resizeNearestNeighborGrad<T: TensorFlowNumeric>(
    grads: Tensor<T>,
    size: Tensor<Int32>,
    alignCorners: Bool = false,
    halfPixelCenters: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ResizeNearestNeighborGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("align_corners", alignCorners)
    op.updateAttribute("half_pixel_centers", halfPixelCenters)
    op.addInput(grads)
    op.addInput(size)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resourceAccumulatorApplyGradient<Dtype: TensorFlowNumeric>(
    handle: ResourceHandle,
    localStep: Tensor<Int64>,
    gradient: Tensor<Dtype>
) {
  let nOutputs = 0
    let op = makeOp("ResourceAccumulatorApplyGradient", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(handle)
    op.addInput(localStep)
    op.addInput(gradient)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceAccumulatorNumAccumulated(
    handle: ResourceHandle
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("ResourceAccumulatorNumAccumulated", nOutputs)
    op.addInput(handle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resourceAccumulatorSetGlobalStep(
    handle: ResourceHandle,
    newGlobalStep: Tensor<Int64>
) {
  let nOutputs = 0
    let op = makeOp("ResourceAccumulatorSetGlobalStep", nOutputs)
    op.addInput(handle)
    op.addInput(newGlobalStep)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceAccumulatorTakeGradient<Dtype: TensorFlowNumeric>(
    handle: ResourceHandle,
    numRequired: Tensor<Int32>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("ResourceAccumulatorTakeGradient", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(handle)
    op.addInput(numRequired)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resourceApplyAdaMax<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    m: ResourceHandle,
    v: ResourceHandle,
    beta1Power: Tensor<T>,
    lr: Tensor<T>,
    beta1: Tensor<T>,
    beta2: Tensor<T>,
    epsilon: Tensor<T>,
    grad: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyAdaMax", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(m)
    op.addInput(v)
    op.addInput(beta1Power)
    op.addInput(lr)
    op.addInput(beta1)
    op.addInput(beta2)
    op.addInput(epsilon)
    op.addInput(grad)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyAdadelta<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    accumUpdate: ResourceHandle,
    lr: Tensor<T>,
    rho: Tensor<T>,
    epsilon: Tensor<T>,
    grad: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyAdadelta", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(accumUpdate)
    op.addInput(lr)
    op.addInput(rho)
    op.addInput(epsilon)
    op.addInput(grad)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyAdagrad<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    lr: Tensor<T>,
    grad: Tensor<T>,
    useLocking: Bool = false,
    updateSlots: Bool = true
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyAdagrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.updateAttribute("update_slots", updateSlots)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(lr)
    op.addInput(grad)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyAdagradDA<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    gradientAccumulator: ResourceHandle,
    gradientSquaredAccumulator: ResourceHandle,
    grad: Tensor<T>,
    lr: Tensor<T>,
    l1: Tensor<T>,
    l2: Tensor<T>,
    globalStep: Tensor<Int64>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyAdagradDA", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(gradientAccumulator)
    op.addInput(gradientSquaredAccumulator)
    op.addInput(grad)
    op.addInput(lr)
    op.addInput(l1)
    op.addInput(l2)
    op.addInput(globalStep)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyAdagradV2<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    lr: Tensor<T>,
    epsilon: Tensor<T>,
    grad: Tensor<T>,
    useLocking: Bool = false,
    updateSlots: Bool = true
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyAdagradV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.updateAttribute("update_slots", updateSlots)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(lr)
    op.addInput(epsilon)
    op.addInput(grad)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyAdam<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    m: ResourceHandle,
    v: ResourceHandle,
    beta1Power: Tensor<T>,
    beta2Power: Tensor<T>,
    lr: Tensor<T>,
    beta1: Tensor<T>,
    beta2: Tensor<T>,
    epsilon: Tensor<T>,
    grad: Tensor<T>,
    useLocking: Bool = false,
    useNesterov: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyAdam", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.updateAttribute("use_nesterov", useNesterov)
    op.addInput(var_)
    op.addInput(m)
    op.addInput(v)
    op.addInput(beta1Power)
    op.addInput(beta2Power)
    op.addInput(lr)
    op.addInput(beta1)
    op.addInput(beta2)
    op.addInput(epsilon)
    op.addInput(grad)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyAdamWithAmsgrad<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    m: ResourceHandle,
    v: ResourceHandle,
    vhat: ResourceHandle,
    beta1Power: Tensor<T>,
    beta2Power: Tensor<T>,
    lr: Tensor<T>,
    beta1: Tensor<T>,
    beta2: Tensor<T>,
    epsilon: Tensor<T>,
    grad: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyAdamWithAmsgrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(m)
    op.addInput(v)
    op.addInput(vhat)
    op.addInput(beta1Power)
    op.addInput(beta2Power)
    op.addInput(lr)
    op.addInput(beta1)
    op.addInput(beta2)
    op.addInput(epsilon)
    op.addInput(grad)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyAddSign<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    m: ResourceHandle,
    lr: Tensor<T>,
    alpha: Tensor<T>,
    signDecay: Tensor<T>,
    beta: Tensor<T>,
    grad: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyAddSign", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(m)
    op.addInput(lr)
    op.addInput(alpha)
    op.addInput(signDecay)
    op.addInput(beta)
    op.addInput(grad)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyCenteredRMSProp<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    mg: ResourceHandle,
    ms: ResourceHandle,
    mom: ResourceHandle,
    lr: Tensor<T>,
    rho: Tensor<T>,
    momentum: Tensor<T>,
    epsilon: Tensor<T>,
    grad: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyCenteredRMSProp", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(mg)
    op.addInput(ms)
    op.addInput(mom)
    op.addInput(lr)
    op.addInput(rho)
    op.addInput(momentum)
    op.addInput(epsilon)
    op.addInput(grad)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyFtrl<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    linear: ResourceHandle,
    grad: Tensor<T>,
    lr: Tensor<T>,
    l1: Tensor<T>,
    l2: Tensor<T>,
    lrPower: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyFtrl", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(linear)
    op.addInput(grad)
    op.addInput(lr)
    op.addInput(l1)
    op.addInput(l2)
    op.addInput(lrPower)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyFtrlV2<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    linear: ResourceHandle,
    grad: Tensor<T>,
    lr: Tensor<T>,
    l1: Tensor<T>,
    l2: Tensor<T>,
    l2Shrinkage: Tensor<T>,
    lrPower: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyFtrlV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(linear)
    op.addInput(grad)
    op.addInput(lr)
    op.addInput(l1)
    op.addInput(l2)
    op.addInput(l2Shrinkage)
    op.addInput(lrPower)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyGradientDescent<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    alpha: Tensor<T>,
    delta: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyGradientDescent", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(alpha)
    op.addInput(delta)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyKerasMomentum<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    lr: Tensor<T>,
    grad: Tensor<T>,
    momentum: Tensor<T>,
    useLocking: Bool = false,
    useNesterov: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyKerasMomentum", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.updateAttribute("use_nesterov", useNesterov)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(lr)
    op.addInput(grad)
    op.addInput(momentum)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyMomentum<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    lr: Tensor<T>,
    grad: Tensor<T>,
    momentum: Tensor<T>,
    useLocking: Bool = false,
    useNesterov: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyMomentum", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.updateAttribute("use_nesterov", useNesterov)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(lr)
    op.addInput(grad)
    op.addInput(momentum)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyPowerSign<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    m: ResourceHandle,
    lr: Tensor<T>,
    logbase: Tensor<T>,
    signDecay: Tensor<T>,
    beta: Tensor<T>,
    grad: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyPowerSign", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(m)
    op.addInput(lr)
    op.addInput(logbase)
    op.addInput(signDecay)
    op.addInput(beta)
    op.addInput(grad)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyProximalAdagrad<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    lr: Tensor<T>,
    l1: Tensor<T>,
    l2: Tensor<T>,
    grad: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyProximalAdagrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(lr)
    op.addInput(l1)
    op.addInput(l2)
    op.addInput(grad)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyProximalGradientDescent<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    alpha: Tensor<T>,
    l1: Tensor<T>,
    l2: Tensor<T>,
    delta: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyProximalGradientDescent", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(alpha)
    op.addInput(l1)
    op.addInput(l2)
    op.addInput(delta)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceApplyRMSProp<T: TensorFlowNumeric>(
    var_: ResourceHandle,
    ms: ResourceHandle,
    mom: ResourceHandle,
    lr: Tensor<T>,
    rho: Tensor<T>,
    momentum: Tensor<T>,
    epsilon: Tensor<T>,
    grad: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceApplyRMSProp", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(ms)
    op.addInput(mom)
    op.addInput(lr)
    op.addInput(rho)
    op.addInput(momentum)
    op.addInput(epsilon)
    op.addInput(grad)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceConditionalAccumulator(
    dtype: TensorDataType,
    shape: TensorShape?,
    container: String,
    sharedName: String,
    reductionType: ReductionType = .mean
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("ResourceConditionalAccumulator", nOutputs)
    op.updateAttribute("dtype", dtype)
    op.updateAttribute("shape", shape)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("reduction_type", reductionType.cName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resourceCountUpTo<T: TensorFlowIndex>(
    resource: ResourceHandle,
    limit: Int64
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ResourceCountUpTo", nOutputs)
    op.updateAttribute("limit", limit)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(resource)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resourceCreateOp(
    resource: ResourceHandle
) {
  let nOutputs = 0
    let op = makeOp("ResourceCreateOp", nOutputs)
    op.addInput(resource)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceGather<
    Dtype: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    resource: ResourceHandle,
    indices: Tensor<Tindices>,
    batchDims: Int64 = 0,
    validateIndices: Bool = true
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("ResourceGather", nOutputs)
    op.updateAttribute("batch_dims", batchDims)
    op.updateAttribute("validate_indices", validateIndices)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(indices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resourceGatherNd<
    Dtype: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    resource: ResourceHandle,
    indices: Tensor<Tindices>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("ResourceGatherNd", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(indices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resourceInitializedOp(
    resource: ResourceHandle
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("ResourceInitializedOp", nOutputs)
    op.addInput(resource)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func resourceScatterAdd<
    Dtype: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    resource: ResourceHandle,
    indices: Tensor<Tindices>,
    updates: Tensor<Dtype>
) {
  let nOutputs = 0
    let op = makeOp("ResourceScatterAdd", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(indices)
    op.addInput(updates)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceScatterDiv<
    Dtype: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    resource: ResourceHandle,
    indices: Tensor<Tindices>,
    updates: Tensor<Dtype>
) {
  let nOutputs = 0
    let op = makeOp("ResourceScatterDiv", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(indices)
    op.addInput(updates)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceScatterMax<
    Dtype: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    resource: ResourceHandle,
    indices: Tensor<Tindices>,
    updates: Tensor<Dtype>
) {
  let nOutputs = 0
    let op = makeOp("ResourceScatterMax", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(indices)
    op.addInput(updates)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceScatterMin<
    Dtype: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    resource: ResourceHandle,
    indices: Tensor<Tindices>,
    updates: Tensor<Dtype>
) {
  let nOutputs = 0
    let op = makeOp("ResourceScatterMin", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(indices)
    op.addInput(updates)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceScatterMul<
    Dtype: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    resource: ResourceHandle,
    indices: Tensor<Tindices>,
    updates: Tensor<Dtype>
) {
  let nOutputs = 0
    let op = makeOp("ResourceScatterMul", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(indices)
    op.addInput(updates)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceScatterNdAdd<
    T: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    ref: ResourceHandle,
    indices: Tensor<Tindices>,
    updates: Tensor<T>,
    useLocking: Bool = true
) {
  let nOutputs = 0
    let op = makeOp("ResourceScatterNdAdd", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(ref)
    op.addInput(indices)
    op.addInput(updates)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceScatterNdSub<
    T: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    ref: ResourceHandle,
    indices: Tensor<Tindices>,
    updates: Tensor<T>,
    useLocking: Bool = true
) {
  let nOutputs = 0
    let op = makeOp("ResourceScatterNdSub", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(ref)
    op.addInput(indices)
    op.addInput(updates)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceScatterNdUpdate<
    T: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    ref: ResourceHandle,
    indices: Tensor<Tindices>,
    updates: Tensor<T>,
    useLocking: Bool = true
) {
  let nOutputs = 0
    let op = makeOp("ResourceScatterNdUpdate", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(ref)
    op.addInput(indices)
    op.addInput(updates)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceScatterSub<
    Dtype: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    resource: ResourceHandle,
    indices: Tensor<Tindices>,
    updates: Tensor<Dtype>
) {
  let nOutputs = 0
    let op = makeOp("ResourceScatterSub", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(indices)
    op.addInput(updates)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceScatterUpdate<
    Dtype: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    resource: ResourceHandle,
    indices: Tensor<Tindices>,
    updates: Tensor<Dtype>
) {
  let nOutputs = 0
    let op = makeOp("ResourceScatterUpdate", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(indices)
    op.addInput(updates)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceSparseApplyAdadelta<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    accumUpdate: ResourceHandle,
    lr: Tensor<T>,
    rho: Tensor<T>,
    epsilon: Tensor<T>,
    grad: Tensor<T>,
    indices: Tensor<Tindices>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceSparseApplyAdadelta", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(accumUpdate)
    op.addInput(lr)
    op.addInput(rho)
    op.addInput(epsilon)
    op.addInput(grad)
    op.addInput(indices)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceSparseApplyAdagrad<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    lr: Tensor<T>,
    grad: Tensor<T>,
    indices: Tensor<Tindices>,
    useLocking: Bool = false,
    updateSlots: Bool = true
) {
  let nOutputs = 0
    let op = makeOp("ResourceSparseApplyAdagrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.updateAttribute("update_slots", updateSlots)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(lr)
    op.addInput(grad)
    op.addInput(indices)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceSparseApplyAdagradDA<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    var_: ResourceHandle,
    gradientAccumulator: ResourceHandle,
    gradientSquaredAccumulator: ResourceHandle,
    grad: Tensor<T>,
    indices: Tensor<Tindices>,
    lr: Tensor<T>,
    l1: Tensor<T>,
    l2: Tensor<T>,
    globalStep: Tensor<Int64>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceSparseApplyAdagradDA", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(gradientAccumulator)
    op.addInput(gradientSquaredAccumulator)
    op.addInput(grad)
    op.addInput(indices)
    op.addInput(lr)
    op.addInput(l1)
    op.addInput(l2)
    op.addInput(globalStep)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceSparseApplyAdagradV2<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    lr: Tensor<T>,
    epsilon: Tensor<T>,
    grad: Tensor<T>,
    indices: Tensor<Tindices>,
    useLocking: Bool = false,
    updateSlots: Bool = true
) {
  let nOutputs = 0
    let op = makeOp("ResourceSparseApplyAdagradV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.updateAttribute("update_slots", updateSlots)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(lr)
    op.addInput(epsilon)
    op.addInput(grad)
    op.addInput(indices)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceSparseApplyCenteredRMSProp<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    var_: ResourceHandle,
    mg: ResourceHandle,
    ms: ResourceHandle,
    mom: ResourceHandle,
    lr: Tensor<T>,
    rho: Tensor<T>,
    momentum: Tensor<T>,
    epsilon: Tensor<T>,
    grad: Tensor<T>,
    indices: Tensor<Tindices>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceSparseApplyCenteredRMSProp", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(mg)
    op.addInput(ms)
    op.addInput(mom)
    op.addInput(lr)
    op.addInput(rho)
    op.addInput(momentum)
    op.addInput(epsilon)
    op.addInput(grad)
    op.addInput(indices)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceSparseApplyFtrl<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    linear: ResourceHandle,
    grad: Tensor<T>,
    indices: Tensor<Tindices>,
    lr: Tensor<T>,
    l1: Tensor<T>,
    l2: Tensor<T>,
    lrPower: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceSparseApplyFtrl", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(linear)
    op.addInput(grad)
    op.addInput(indices)
    op.addInput(lr)
    op.addInput(l1)
    op.addInput(l2)
    op.addInput(lrPower)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceSparseApplyFtrlV2<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    linear: ResourceHandle,
    grad: Tensor<T>,
    indices: Tensor<Tindices>,
    lr: Tensor<T>,
    l1: Tensor<T>,
    l2: Tensor<T>,
    l2Shrinkage: Tensor<T>,
    lrPower: Tensor<T>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceSparseApplyFtrlV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(linear)
    op.addInput(grad)
    op.addInput(indices)
    op.addInput(lr)
    op.addInput(l1)
    op.addInput(l2)
    op.addInput(l2Shrinkage)
    op.addInput(lrPower)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceSparseApplyKerasMomentum<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    lr: Tensor<T>,
    grad: Tensor<T>,
    indices: Tensor<Tindices>,
    momentum: Tensor<T>,
    useLocking: Bool = false,
    useNesterov: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceSparseApplyKerasMomentum", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.updateAttribute("use_nesterov", useNesterov)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(lr)
    op.addInput(grad)
    op.addInput(indices)
    op.addInput(momentum)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceSparseApplyMomentum<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    lr: Tensor<T>,
    grad: Tensor<T>,
    indices: Tensor<Tindices>,
    momentum: Tensor<T>,
    useLocking: Bool = false,
    useNesterov: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceSparseApplyMomentum", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.updateAttribute("use_nesterov", useNesterov)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(lr)
    op.addInput(grad)
    op.addInput(indices)
    op.addInput(momentum)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceSparseApplyProximalAdagrad<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    var_: ResourceHandle,
    accum: ResourceHandle,
    lr: Tensor<T>,
    l1: Tensor<T>,
    l2: Tensor<T>,
    grad: Tensor<T>,
    indices: Tensor<Tindices>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceSparseApplyProximalAdagrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(accum)
    op.addInput(lr)
    op.addInput(l1)
    op.addInput(l2)
    op.addInput(grad)
    op.addInput(indices)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceSparseApplyProximalGradientDescent<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    var_: ResourceHandle,
    alpha: Tensor<T>,
    l1: Tensor<T>,
    l2: Tensor<T>,
    grad: Tensor<T>,
    indices: Tensor<Tindices>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceSparseApplyProximalGradientDescent", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(alpha)
    op.addInput(l1)
    op.addInput(l2)
    op.addInput(grad)
    op.addInput(indices)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceSparseApplyRMSProp<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    var_: ResourceHandle,
    ms: ResourceHandle,
    mom: ResourceHandle,
    lr: Tensor<T>,
    rho: Tensor<T>,
    momentum: Tensor<T>,
    epsilon: Tensor<T>,
    grad: Tensor<T>,
    indices: Tensor<Tindices>,
    useLocking: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("ResourceSparseApplyRMSProp", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("use_locking", useLocking)
    op.addInput(var_)
    op.addInput(ms)
    op.addInput(mom)
    op.addInput(lr)
    op.addInput(rho)
    op.addInput(momentum)
    op.addInput(epsilon)
    op.addInput(grad)
    op.addInput(indices)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceStridedSliceAssign<
    T: TensorFlowScalar,
    Index: TensorFlowIndex
>(
    ref: ResourceHandle,
    begin: Tensor<Index>,
    end: Tensor<Index>,
    strides: Tensor<Index>,
    value: Tensor<T>,
    beginMask: Int64 = 0,
    endMask: Int64 = 0,
    ellipsisMask: Int64 = 0,
    newAxisMask: Int64 = 0,
    shrinkAxisMask: Int64 = 0
) {
  let nOutputs = 0
    let op = makeOp("ResourceStridedSliceAssign", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Index", Index.tensorFlowDataType)
    op.updateAttribute("begin_mask", beginMask)
    op.updateAttribute("end_mask", endMask)
    op.updateAttribute("ellipsis_mask", ellipsisMask)
    op.updateAttribute("new_axis_mask", newAxisMask)
    op.updateAttribute("shrink_axis_mask", shrinkAxisMask)
    op.addInput(ref)
    op.addInput(begin)
    op.addInput(end)
    op.addInput(strides)
    op.addInput(value)
    op.execute()
}

@inlinable @inline(__always)
public static func resourceUsingOp(
    resource: ResourceHandle
) {
  let nOutputs = 0
    let op = makeOp("ResourceUsingOp", nOutputs)
    op.addInput(resource)
    op.execute()
}

@inlinable @inline(__always)
public static func restore<Dt: TensorFlowScalar>(
    filePattern: StringTensor,
    tensorName: StringTensor,
    preferredShard: Int64 = -1
) -> Tensor<Dt> {
  let nOutputs = Int(1)
    let op = makeOp("Restore", nOutputs)
    op.updateAttribute("dt", Dt.tensorFlowDataType)
    op.updateAttribute("preferred_shard", preferredShard)
    op.addInput(filePattern)
    op.addInput(tensorName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func restoreSlice<Dt: TensorFlowScalar>(
    filePattern: StringTensor,
    tensorName: StringTensor,
    shapeAndSlice: StringTensor,
    preferredShard: Int64 = -1
) -> Tensor<Dt> {
  let nOutputs = Int(1)
    let op = makeOp("RestoreSlice", nOutputs)
    op.updateAttribute("dt", Dt.tensorFlowDataType)
    op.updateAttribute("preferred_shard", preferredShard)
    op.addInput(filePattern)
    op.addInput(tensorName)
    op.addInput(shapeAndSlice)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func restoreV2<Dtypes: TensorGroup>(
    prefix: StringTensor,
    tensorNames: StringTensor,
    shapeAndSlices: StringTensor
) -> Dtypes {
  let nOutputs = Int(Dtypes._typeList.count)
    let op = makeOp("RestoreV2", nOutputs)
    op.updateAttribute("dtypes", Dtypes._typeList)
    op.addInput(prefix)
    op.addInput(tensorNames)
    op.addInput(shapeAndSlices)
    return op.execute(Int(Dtypes._typeList.count))
}

@inlinable @inline(__always)
public static func restrict<T: TensorFlowScalar>(
    _ a: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Restrict", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func restrict(
    _ a: StringTensor
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("Restrict", nOutputs)
    op.updateAttribute("T", TensorDataType(TF_STRING))
    op.addInput(a)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingADAMParameters(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, momenta: Tensor<Float>, velocities: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingADAMParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingADAMParametersGradAccumDebug(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, momenta: Tensor<Float>, velocities: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingADAMParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingAdadeltaParameters(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, updates: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingAdadeltaParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingAdadeltaParametersGradAccumDebug(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, updates: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingAdadeltaParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingAdagradParameters(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingAdagradParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingAdagradParametersGradAccumDebug(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingAdagradParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingCenteredRMSPropParameters(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, ms: Tensor<Float>, mom: Tensor<Float>, mg: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingCenteredRMSPropParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingFTRLParameters(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, linears: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingFTRLParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingFTRLParametersGradAccumDebug(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, linears: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingFTRLParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingMDLAdagradLightParameters(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, weights: Tensor<Float>, benefits: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingMDLAdagradLightParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingMomentumParameters(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, momenta: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingMomentumParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingMomentumParametersGradAccumDebug(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, momenta: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingMomentumParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingProximalAdagradParameters(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingProximalAdagradParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingRMSPropParameters(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, ms: Tensor<Float>, mom: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingRMSPropParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingRMSPropParametersGradAccumDebug(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> (parameters: Tensor<Float>, ms: Tensor<Float>, mom: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("RetrieveTPUEmbeddingRMSPropParametersGradAccumDebug", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func retrieveTPUEmbeddingStochasticGradientDescentParameters(
    tableId: Int64 = -1,
    tableName: String,
    numShards: Int64,
    shardId: Int64,
    config: String
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("RetrieveTPUEmbeddingStochasticGradientDescentParameters", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("table_name", tableName)
    op.updateAttribute("num_shards", numShards)
    op.updateAttribute("shard_id", shardId)
    op.updateAttribute("config", config)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func reverse<T: TensorFlowScalar>(
    _ tensor: Tensor<T>,
    dims: Tensor<Bool>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Reverse", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(tensor)
    op.addInput(dims)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func reverse(
    _ tensor: StringTensor,
    dims: Tensor<Bool>
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("Reverse", nOutputs)
    op.updateAttribute("T", TensorDataType(TF_STRING))
    op.addInput(tensor)
    op.addInput(dims)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func reverseSequence<
    T: TensorFlowScalar,
    Tlen: TensorFlowIndex
>(
    _ input: Tensor<T>,
    seqLengths: Tensor<Tlen>,
    seqDim: Int64,
    batchDim: Int64 = 0
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ReverseSequence", nOutputs)
    op.updateAttribute("seq_dim", seqDim)
    op.updateAttribute("batch_dim", batchDim)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tlen", Tlen.tensorFlowDataType)
    op.addInput(input)
    op.addInput(seqLengths)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func reverseV2<
    Tidx: TensorFlowIndex,
    T: TensorFlowScalar
>(
    _ tensor: Tensor<T>,
    axis: Tensor<Tidx>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ReverseV2", nOutputs)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(tensor)
    op.addInput(axis)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func reverseV2<Tidx: TensorFlowIndex>(
    _ tensor: StringTensor,
    axis: Tensor<Tidx>
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("ReverseV2", nOutputs)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.updateAttribute("T", TensorDataType(TF_STRING))
    op.addInput(tensor)
    op.addInput(axis)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func rightShift<T: TensorFlowInteger>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("RightShift", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func rint<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Rint", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func rngSkip(
    resource: ResourceHandle,
    algorithm: Tensor<Int64>,
    delta: Tensor<Int64>
) {
  let nOutputs = 0
    let op = makeOp("RngSkip", nOutputs)
    op.addInput(resource)
    op.addInput(algorithm)
    op.addInput(delta)
    op.execute()
}

@inlinable @inline(__always)
public static func roll<
    T: TensorFlowScalar,
    Tshift: TensorFlowIndex,
    Taxis: TensorFlowIndex
>(
    _ input: Tensor<T>,
    shift: Tensor<Tshift>,
    axis: Tensor<Taxis>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Roll", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tshift", Tshift.tensorFlowDataType)
    op.updateAttribute("Taxis", Taxis.tensorFlowDataType)
    op.addInput(input)
    op.addInput(shift)
    op.addInput(axis)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func round<T: TensorFlowNumeric>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Round", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func rpc(
    address: StringTensor,
    method: StringTensor,
    request: StringTensor,
    protocol_: String,
    failFast: Bool = true,
    timeoutInMs: Int64 = 0
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("Rpc", nOutputs)
    op.updateAttribute("protocol", protocol_)
    op.updateAttribute("fail_fast", failFast)
    op.updateAttribute("timeout_in_ms", timeoutInMs)
    op.addInput(address)
    op.addInput(method)
    op.addInput(request)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func rsqrt<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Rsqrt", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func rsqrtGrad<T: FloatingPoint & TensorFlowScalar>(
    _ y: Tensor<T>,
    dy: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("RsqrtGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(y)
    op.addInput(dy)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sampleDistortedBoundingBox<T: TensorFlowInteger>(
    imageSize: Tensor<T>,
    boundingBoxes: Tensor<Float>,
    seed: Int64 = 0,
    seed2: Int64 = 0,
    minObjectCovered: Double = 0.1,
    aspectRatioRange: [Double] = [0.75, 1.33],
    areaRange: [Double] = [0.05, 1],
    maxAttempts: Int64 = 100,
    useImageIfNoBoundingBoxes: Bool = false
) -> (begin: Tensor<T>, size: Tensor<T>, bboxes: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("SampleDistortedBoundingBox", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("min_object_covered", minObjectCovered)
    op.updateAttribute("aspect_ratio_range", aspectRatioRange)
    op.updateAttribute("area_range", areaRange)
    op.updateAttribute("max_attempts", maxAttempts)
    op.updateAttribute("use_image_if_no_bounding_boxes", useImageIfNoBoundingBoxes)
    op.addInput(imageSize)
    op.addInput(boundingBoxes)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sampleDistortedBoundingBoxV2<T: TensorFlowInteger>(
    imageSize: Tensor<T>,
    boundingBoxes: Tensor<Float>,
    minObjectCovered: Tensor<Float>,
    seed: Int64 = 0,
    seed2: Int64 = 0,
    aspectRatioRange: [Double] = [0.75, 1.33],
    areaRange: [Double] = [0.05, 1],
    maxAttempts: Int64 = 100,
    useImageIfNoBoundingBoxes: Bool = false
) -> (begin: Tensor<T>, size: Tensor<T>, bboxes: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("SampleDistortedBoundingBoxV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("aspect_ratio_range", aspectRatioRange)
    op.updateAttribute("area_range", areaRange)
    op.updateAttribute("max_attempts", maxAttempts)
    op.updateAttribute("use_image_if_no_bounding_boxes", useImageIfNoBoundingBoxes)
    op.addInput(imageSize)
    op.addInput(boundingBoxes)
    op.addInput(minObjectCovered)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func samplingDataset(
    inputDataset: VariantHandle,
    rate: Tensor<Float>,
    seed: Tensor<Int64>,
    seed2: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SamplingDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(rate)
    op.addInput(seed)
    op.addInput(seed2)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func save<T: TensorArrayProtocol>(
    filename: StringTensor,
    tensorNames: StringTensor,
    data: T
) {
  let nOutputs = 0
    let op = makeOp("Save", nOutputs)
    op.updateAttribute("T", data._typeList)
    op.addInput(filename)
    op.addInput(tensorNames)
    op.addInputList(data)
    op.execute()
}

@inlinable @inline(__always)
public static func saveSlices<T: TensorArrayProtocol>(
    filename: StringTensor,
    tensorNames: StringTensor,
    shapesAndSlices: StringTensor,
    data: T
) {
  let nOutputs = 0
    let op = makeOp("SaveSlices", nOutputs)
    op.updateAttribute("T", data._typeList)
    op.addInput(filename)
    op.addInput(tensorNames)
    op.addInput(shapesAndSlices)
    op.addInputList(data)
    op.execute()
}

@inlinable @inline(__always)
public static func saveV2<Dtypes: TensorArrayProtocol>(
    prefix: StringTensor,
    tensorNames: StringTensor,
    shapeAndSlices: StringTensor,
    tensors: Dtypes
) {
  let nOutputs = 0
    let op = makeOp("SaveV2", nOutputs)
    op.updateAttribute("dtypes", tensors._typeList)
    op.addInput(prefix)
    op.addInput(tensorNames)
    op.addInput(shapeAndSlices)
    op.addInputList(tensors)
    op.execute()
}

@inlinable @inline(__always)
public static func scalarSummary<T: TensorFlowNumeric>(
    tags: StringTensor,
    _ values: Tensor<T>
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("ScalarSummary", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(tags)
    op.addInput(values)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func scaleAndTranslate<T: TensorFlowNumeric>(
    images: Tensor<T>,
    size: Tensor<Int32>,
    scale: Tensor<Float>,
    translation: Tensor<Float>,
    kernelType: String = "lanczos3",
    antialias: Bool = true
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("ScaleAndTranslate", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("kernel_type", kernelType)
    op.updateAttribute("antialias", antialias)
    op.addInput(images)
    op.addInput(size)
    op.addInput(scale)
    op.addInput(translation)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func scaleAndTranslateGrad<T: FloatingPoint & TensorFlowScalar>(
    grads: Tensor<T>,
    originalImage: Tensor<T>,
    scale: Tensor<Float>,
    translation: Tensor<Float>,
    kernelType: String = "lanczos3",
    antialias: Bool = true
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ScaleAndTranslateGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("kernel_type", kernelType)
    op.updateAttribute("antialias", antialias)
    op.addInput(grads)
    op.addInput(originalImage)
    op.addInput(scale)
    op.addInput(translation)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func scanDataset<
    FIn: TensorGroup,
    FOut: TensorGroup,
    Tstate: TensorArrayProtocol,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    initialState: Tstate,
    otherArguments: Targuments,
    f: (FIn) -> FOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    preserveCardinality: Bool = false,
    useDefaultDevice: Bool = true
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ScanDataset", nOutputs)
    op.updateAttribute("f", f)
    op.updateAttribute("Tstate", initialState._typeList)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("preserve_cardinality", preserveCardinality)
    op.updateAttribute("use_default_device", useDefaultDevice)
    op.addInput(inputDataset)
    op.addInputList(initialState)
    op.addInputList(otherArguments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func scatterNd<
    T: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    indices: Tensor<Tindices>,
    updates: Tensor<T>,
    shape: Tensor<Tindices>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ScatterNd", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(indices)
    op.addInput(updates)
    op.addInput(shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func scatterNdNonAliasingAdd<
    T: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    _ input: Tensor<T>,
    indices: Tensor<Tindices>,
    updates: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ScatterNdNonAliasingAdd", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(input)
    op.addInput(indices)
    op.addInput(updates)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sdcaFprint(
    _ input: StringTensor
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("SdcaFprint", nOutputs)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sdcaOptimizer(
    sparseExampleIndices: [Tensor<Int64>],
    sparseFeatureIndices: [Tensor<Int64>],
    sparseFeatureValues: [Tensor<Float>],
    denseFeatures: [Tensor<Float>],
    exampleWeights: Tensor<Float>,
    exampleLabels: Tensor<Float>,
    sparseIndices: [Tensor<Int64>],
    sparseWeights: [Tensor<Float>],
    denseWeights: [Tensor<Float>],
    exampleStateData: Tensor<Float>,
    lossType: LossType,
    adaptative: Bool = false,
    l1: Double,
    l2: Double,
    numLossPartitions: Int64,
    numInnerIterations: Int64
) -> (outExampleStateData: Tensor<Float>, outDeltaSparseWeights: [Tensor<Float>], outDeltaDenseWeights: [Tensor<Float>]) {
  let nOutputs = Int(1) + Int(sparseExampleIndices.count) + Int(denseFeatures.count)
    let op = makeOp("SdcaOptimizer", nOutputs)
    op.updateAttribute("loss_type", lossType.cName)
    op.updateAttribute("adaptative", adaptative)
    op.updateAttribute("num_sparse_features", sparseExampleIndices.count)
    op.updateAttribute("num_sparse_features_with_values", sparseFeatureValues.count)
    op.updateAttribute("num_dense_features", denseFeatures.count)
    op.updateAttribute("l1", l1)
    op.updateAttribute("l2", l2)
    op.updateAttribute("num_loss_partitions", numLossPartitions)
    op.updateAttribute("num_inner_iterations", numInnerIterations)
    op.addInputList(sparseExampleIndices)
    op.addInputList(sparseFeatureIndices)
    op.addInputList(sparseFeatureValues)
    op.addInputList(denseFeatures)
    op.addInput(exampleWeights)
    op.addInput(exampleLabels)
    op.addInputList(sparseIndices)
    op.addInputList(sparseWeights)
    op.addInputList(denseWeights)
    op.addInput(exampleStateData)
    return op.execute(Int(1), Int(sparseExampleIndices.count), Int(denseFeatures.count))
}

@inlinable @inline(__always)
public static func sdcaOptimizerV2(
    sparseExampleIndices: [Tensor<Int64>],
    sparseFeatureIndices: [Tensor<Int64>],
    sparseFeatureValues: [Tensor<Float>],
    denseFeatures: [Tensor<Float>],
    exampleWeights: Tensor<Float>,
    exampleLabels: Tensor<Float>,
    sparseIndices: [Tensor<Int64>],
    sparseWeights: [Tensor<Float>],
    denseWeights: [Tensor<Float>],
    exampleStateData: Tensor<Float>,
    lossType: LossType,
    adaptive: Bool = false,
    l1: Double,
    l2: Double,
    numLossPartitions: Int64,
    numInnerIterations: Int64
) -> (outExampleStateData: Tensor<Float>, outDeltaSparseWeights: [Tensor<Float>], outDeltaDenseWeights: [Tensor<Float>]) {
  let nOutputs = Int(1) + Int(sparseExampleIndices.count) + Int(denseFeatures.count)
    let op = makeOp("SdcaOptimizerV2", nOutputs)
    op.updateAttribute("loss_type", lossType.cName)
    op.updateAttribute("adaptive", adaptive)
    op.updateAttribute("num_sparse_features", sparseExampleIndices.count)
    op.updateAttribute("num_sparse_features_with_values", sparseFeatureValues.count)
    op.updateAttribute("num_dense_features", denseFeatures.count)
    op.updateAttribute("l1", l1)
    op.updateAttribute("l2", l2)
    op.updateAttribute("num_loss_partitions", numLossPartitions)
    op.updateAttribute("num_inner_iterations", numInnerIterations)
    op.addInputList(sparseExampleIndices)
    op.addInputList(sparseFeatureIndices)
    op.addInputList(sparseFeatureValues)
    op.addInputList(denseFeatures)
    op.addInput(exampleWeights)
    op.addInput(exampleLabels)
    op.addInputList(sparseIndices)
    op.addInputList(sparseWeights)
    op.addInputList(denseWeights)
    op.addInput(exampleStateData)
    return op.execute(Int(1), Int(sparseExampleIndices.count), Int(denseFeatures.count))
}

@inlinable @inline(__always)
public static func segmentMax<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    data: Tensor<T>,
    segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SegmentMax", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(data)
    op.addInput(segmentIds)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func segmentMean<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    data: Tensor<T>,
    segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SegmentMean", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(data)
    op.addInput(segmentIds)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func segmentMin<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    data: Tensor<T>,
    segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SegmentMin", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(data)
    op.addInput(segmentIds)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func segmentProd<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    data: Tensor<T>,
    segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SegmentProd", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(data)
    op.addInput(segmentIds)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func segmentSum<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    data: Tensor<T>,
    segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SegmentSum", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(data)
    op.addInput(segmentIds)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func select<T: TensorFlowScalar>(
    condition: Tensor<Bool>,
    t: Tensor<T>,
    e: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Select", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(condition)
    op.addInput(t)
    op.addInput(e)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func selectV2<T: TensorFlowScalar>(
    condition: Tensor<Bool>,
    t: Tensor<T>,
    e: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SelectV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(condition)
    op.addInput(t)
    op.addInput(e)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func selfAdjointEig<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SelfAdjointEig", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func selfAdjointEigV2<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    computeV: Bool = true
) -> (e: Tensor<T>, v: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("SelfAdjointEigV2", nOutputs)
    op.updateAttribute("compute_v", computeV)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func selu<T: FloatingPoint & TensorFlowScalar>(
    features: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Selu", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(features)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func seluGrad<T: FloatingPoint & TensorFlowScalar>(
    gradients: Tensor<T>,
    outputs: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SeluGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(gradients)
    op.addInput(outputs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func send<T: TensorFlowScalar>(
    _ tensor: Tensor<T>,
    tensorName: String,
    sendDevice: String,
    sendDeviceIncarnation: Int64,
    recvDevice: String,
    clientTerminated: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("Send", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("tensor_name", tensorName)
    op.updateAttribute("send_device", sendDevice)
    op.updateAttribute("send_device_incarnation", sendDeviceIncarnation)
    op.updateAttribute("recv_device", recvDevice)
    op.updateAttribute("client_terminated", clientTerminated)
    op.addInput(tensor)
    op.execute()
}

@inlinable @inline(__always)
public static func sendTPUEmbeddingGradients(
    inputs: [Tensor<Float>],
    learningRates: [Tensor<Float>],
    config: String
) {
  let nOutputs = 0
    let op = makeOp("SendTPUEmbeddingGradients", nOutputs)
    op.updateAttribute("N", inputs.count)
    op.updateAttribute("NN", learningRates.count)
    op.updateAttribute("config", config)
    op.addInputList(inputs)
    op.addInputList(learningRates)
    op.execute()
}

@inlinable @inline(__always)
public static func serializeIterator(
    resourceHandle: ResourceHandle
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SerializeIterator", nOutputs)
    op.addInput(resourceHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func serializeManySparse<
    T: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    sparseIndices: Tensor<Int64>,
    sparseValues: Tensor<T>,
    sparseShape: Tensor<Int64>
) -> Tensor<OutType> {
  let nOutputs = Int(1)
    let op = makeOp("SerializeManySparse", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(sparseIndices)
    op.addInput(sparseValues)
    op.addInput(sparseShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func serializeManySparse<T: TensorFlowScalar>(
    sparseIndices: Tensor<Int64>,
    sparseValues: Tensor<T>,
    sparseShape: Tensor<Int64>
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("SerializeManySparse", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("out_type", TensorDataType(TF_STRING))
    op.addInput(sparseIndices)
    op.addInput(sparseValues)
    op.addInput(sparseShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func serializeSparse<
    T: TensorFlowScalar,
    OutType: TensorFlowScalar
>(
    sparseIndices: Tensor<Int64>,
    sparseValues: Tensor<T>,
    sparseShape: Tensor<Int64>
) -> Tensor<OutType> {
  let nOutputs = Int(1)
    let op = makeOp("SerializeSparse", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(sparseIndices)
    op.addInput(sparseValues)
    op.addInput(sparseShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func serializeSparse<T: TensorFlowScalar>(
    sparseIndices: Tensor<Int64>,
    sparseValues: Tensor<T>,
    sparseShape: Tensor<Int64>
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("SerializeSparse", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("out_type", TensorDataType(TF_STRING))
    op.addInput(sparseIndices)
    op.addInput(sparseValues)
    op.addInput(sparseShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func serializeTRTResource(
    resourceName: StringTensor,
    filename: StringTensor,
    deleteResource: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("SerializeTRTResource", nOutputs)
    op.updateAttribute("delete_resource", deleteResource)
    op.addInput(resourceName)
    op.addInput(filename)
    op.execute()
}

@inlinable @inline(__always)
public static func serializeTensor<T: TensorFlowScalar>(
    _ tensor: Tensor<T>
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("SerializeTensor", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(tensor)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func setSize<T: TensorFlowInteger>(
    setIndices: Tensor<Int64>,
    setValues: Tensor<T>,
    setShape: Tensor<Int64>,
    validateIndices: Bool = true
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("SetSize", nOutputs)
    op.updateAttribute("validate_indices", validateIndices)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(setIndices)
    op.addInput(setValues)
    op.addInput(setShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func setSize(
    setIndices: Tensor<Int64>,
    setValues: StringTensor,
    setShape: Tensor<Int64>,
    validateIndices: Bool = true
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("SetSize", nOutputs)
    op.updateAttribute("validate_indices", validateIndices)
    op.updateAttribute("T", TensorDataType(TF_STRING))
    op.addInput(setIndices)
    op.addInput(setValues)
    op.addInput(setShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func setStatsAggregatorDataset(
    inputDataset: VariantHandle,
    statsAggregator: ResourceHandle,
    tag: StringTensor,
    counterPrefix: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SetStatsAggregatorDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(statsAggregator)
    op.addInput(tag)
    op.addInput(counterPrefix)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func shape<
    T: TensorFlowScalar,
    OutType: TensorFlowIndex
>(
    _ input: Tensor<T>
) -> Tensor<OutType> {
  let nOutputs = Int(1)
    let op = makeOp("Shape", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func shapeN<
    T: TensorFlowScalar,
    OutType: TensorFlowIndex
>(
    _ input: [Tensor<T>]
) -> [Tensor<OutType>] {
  let nOutputs = Int(input.count)
    let op = makeOp("ShapeN", nOutputs)
    op.updateAttribute("N", input.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInputList(input)
    return op.execute(Int(input.count))
}

@inlinable @inline(__always)
public static func shardDataset(
    inputDataset: VariantHandle,
    numShards: Tensor<Int64>,
    index: Tensor<Int64>,
    requireNonEmpty: Bool = false,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ShardDataset", nOutputs)
    op.updateAttribute("require_non_empty", requireNonEmpty)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(numShards)
    op.addInput(index)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func shardedFilename(
    basename: StringTensor,
    shard: Tensor<Int32>,
    numShards: Tensor<Int32>
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("ShardedFilename", nOutputs)
    op.addInput(basename)
    op.addInput(shard)
    op.addInput(numShards)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func shardedFilespec(
    basename: StringTensor,
    numShards: Tensor<Int32>
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("ShardedFilespec", nOutputs)
    op.addInput(basename)
    op.addInput(numShards)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func shuffleAndRepeatDataset(
    inputDataset: VariantHandle,
    bufferSize: Tensor<Int64>,
    seed: Tensor<Int64>,
    seed2: Tensor<Int64>,
    count: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ShuffleAndRepeatDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(bufferSize)
    op.addInput(seed)
    op.addInput(seed2)
    op.addInput(count)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func shuffleDataset(
    inputDataset: VariantHandle,
    bufferSize: Tensor<Int64>,
    seed: Tensor<Int64>,
    seed2: Tensor<Int64>,
    reshuffleEachIteration: Bool = true,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ShuffleDataset", nOutputs)
    op.updateAttribute("reshuffle_each_iteration", reshuffleEachIteration)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(bufferSize)
    op.addInput(seed)
    op.addInput(seed2)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func shuffleDatasetV2(
    inputDataset: VariantHandle,
    bufferSize: Tensor<Int64>,
    seedGenerator: ResourceHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ShuffleDatasetV2", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(bufferSize)
    op.addInput(seedGenerator)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func shutdownDistributedTPU(
) {
  let nOutputs = 0
    let op = makeOp("ShutdownDistributedTPU", nOutputs)
    
    op.execute()
}

@inlinable @inline(__always)
public static func sigmoid<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Sigmoid", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sigmoidGrad<T: FloatingPoint & TensorFlowScalar>(
    _ y: Tensor<T>,
    dy: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SigmoidGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(y)
    op.addInput(dy)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sign<T: TensorFlowNumeric>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Sign", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func simple(
    _ a: Tensor<Int32>
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("Simple", nOutputs)
    op.addInput(a)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func simpleStruct(
    nA: Int64
) -> [Tensor<Int32>] {
  let nOutputs = Int(nA)
    let op = makeOp("SimpleStruct", nOutputs)
    op.updateAttribute("n_a", nA)
    return op.execute(Int(nA))
}

@inlinable @inline(__always)
public static func sin<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Sin", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sinh<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Sinh", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func size<
    T: TensorFlowScalar,
    OutType: TensorFlowIndex
>(
    _ input: Tensor<T>
) -> Tensor<OutType> {
  let nOutputs = Int(1)
    let op = makeOp("Size", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func skipDataset(
    inputDataset: VariantHandle,
    count: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SkipDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(count)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func skipgram(
    filename: String,
    batchSize: Int64,
    windowSize: Int64 = 5,
    minCount: Int64 = 5,
    subsample: Double = 0.001
) -> (vocabWord: StringTensor, vocabFreq: Tensor<Int32>, wordsPerEpoch: Tensor<Int64>, currentEpoch: Tensor<Int32>, totalWordsProcessed: Tensor<Int64>, examples: Tensor<Int32>, labels: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("Skipgram", nOutputs)
    op.updateAttribute("filename", filename)
    op.updateAttribute("batch_size", batchSize)
    op.updateAttribute("window_size", windowSize)
    op.updateAttribute("min_count", minCount)
    op.updateAttribute("subsample", subsample)
    return op.execute(Int(1), Int(1), Int(1), Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sleepDataset(
    inputDataset: VariantHandle,
    sleepMicroseconds: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SleepDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(sleepMicroseconds)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func slice<
    T: TensorFlowScalar,
    Index: TensorFlowIndex
>(
    _ input: Tensor<T>,
    begin: Tensor<Index>,
    size: Tensor<Index>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Slice", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Index", Index.tensorFlowDataType)
    op.addInput(input)
    op.addInput(begin)
    op.addInput(size)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func slidingWindowDataset(
    inputDataset: VariantHandle,
    windowSize: Tensor<Int64>,
    windowShift: Tensor<Int64>,
    windowStride: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SlidingWindowDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(windowSize)
    op.addInput(windowShift)
    op.addInput(windowStride)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func snapshot<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Snapshot", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func snapshotDataset(
    inputDataset: VariantHandle,
    path: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?],
    compression: String,
    readerPathPrefix: String,
    writerPathPrefix: String,
    shardSizeBytes: Int64 = 10737418240,
    pendingSnapshotExpirySeconds: Int64 = 86400,
    numReaderThreads: Int64 = 1,
    readerBufferSize: Int64 = 1,
    numWriterThreads: Int64 = 1,
    writerBufferSize: Int64 = 1,
    shuffleOnRead: Bool = false,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SnapshotDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("compression", compression)
    op.updateAttribute("reader_path_prefix", readerPathPrefix)
    op.updateAttribute("writer_path_prefix", writerPathPrefix)
    op.updateAttribute("shard_size_bytes", shardSizeBytes)
    op.updateAttribute("pending_snapshot_expiry_seconds", pendingSnapshotExpirySeconds)
    op.updateAttribute("num_reader_threads", numReaderThreads)
    op.updateAttribute("reader_buffer_size", readerBufferSize)
    op.updateAttribute("num_writer_threads", numWriterThreads)
    op.updateAttribute("writer_buffer_size", writerBufferSize)
    op.updateAttribute("shuffle_on_read", shuffleOnRead)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.addInput(inputDataset)
    op.addInput(path)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func softmax<T: FloatingPoint & TensorFlowScalar>(
    logits: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Softmax", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(logits)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func softmaxCrossEntropyWithLogits<T: FloatingPoint & TensorFlowScalar>(
    features: Tensor<T>,
    labels: Tensor<T>
) -> (loss: Tensor<T>, backprop: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("SoftmaxCrossEntropyWithLogits", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(features)
    op.addInput(labels)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func softplus<T: FloatingPoint & TensorFlowScalar>(
    features: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Softplus", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(features)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func softplusGrad<T: FloatingPoint & TensorFlowScalar>(
    gradients: Tensor<T>,
    features: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SoftplusGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(gradients)
    op.addInput(features)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func softsign<T: FloatingPoint & TensorFlowScalar>(
    features: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Softsign", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(features)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func softsignGrad<T: FloatingPoint & TensorFlowScalar>(
    gradients: Tensor<T>,
    features: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SoftsignGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(gradients)
    op.addInput(features)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func spaceToBatch<
    T: TensorFlowScalar,
    Tpaddings: TensorFlowIndex
>(
    _ input: Tensor<T>,
    paddings: Tensor<Tpaddings>,
    blockSize: Int64
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SpaceToBatch", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tpaddings", Tpaddings.tensorFlowDataType)
    op.updateAttribute("block_size", blockSize)
    op.addInput(input)
    op.addInput(paddings)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func spaceToBatchND<
    T: TensorFlowScalar,
    TblockShape: TensorFlowIndex,
    Tpaddings: TensorFlowIndex
>(
    _ input: Tensor<T>,
    blockShape: Tensor<TblockShape>,
    paddings: Tensor<Tpaddings>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SpaceToBatchND", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tblock_shape", TblockShape.tensorFlowDataType)
    op.updateAttribute("Tpaddings", Tpaddings.tensorFlowDataType)
    op.addInput(input)
    op.addInput(blockShape)
    op.addInput(paddings)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func spaceToDepth<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    blockSize: Int64,
    dataFormat: DataFormat5 = .nhwc
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SpaceToDepth", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("block_size", blockSize)
    op.updateAttribute("data_format", dataFormat.cName)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseAdd<
    T: TensorFlowNumeric,
    Treal: TensorFlowNumeric
>(
    aIndices: Tensor<Int64>,
    aValues: Tensor<T>,
    aShape: Tensor<Int64>,
    bIndices: Tensor<Int64>,
    bValues: Tensor<T>,
    bShape: Tensor<Int64>,
    thresh: Tensor<Treal>
) -> (sumIndices: Tensor<Int64>, sumValues: Tensor<T>, sumShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("SparseAdd", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Treal", Treal.tensorFlowDataType)
    op.addInput(aIndices)
    op.addInput(aValues)
    op.addInput(aShape)
    op.addInput(bIndices)
    op.addInput(bValues)
    op.addInput(bShape)
    op.addInput(thresh)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseAddGrad<T: TensorFlowNumeric>(
    backpropValGrad: Tensor<T>,
    aIndices: Tensor<Int64>,
    bIndices: Tensor<Int64>,
    sumIndices: Tensor<Int64>
) -> (aValGrad: Tensor<T>, bValGrad: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("SparseAddGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(backpropValGrad)
    op.addInput(aIndices)
    op.addInput(bIndices)
    op.addInput(sumIndices)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseConcat<T: TensorFlowScalar>(
    indices: [Tensor<Int64>],
    _ values: [Tensor<T>],
    shapes: [Tensor<Int64>],
    concatDim: Int64
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, outputShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("SparseConcat", nOutputs)
    op.updateAttribute("concat_dim", concatDim)
    op.updateAttribute("N", indices.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInputList(indices)
    op.addInputList(values)
    op.addInputList(shapes)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseCross<
    SparseTypes: TensorArrayProtocol,
    DenseTypes: TensorArrayProtocol,
    OutType: TensorFlowIndex
>(
    indices: [Tensor<Int64>],
    _ values: SparseTypes,
    shapes: [Tensor<Int64>],
    denseInputs: DenseTypes,
    hashedOutput: Bool,
    numBuckets: Int64,
    hashKey: Int64,
    internalType: TensorDataType
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<OutType>, outputShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("SparseCross", nOutputs)
    op.updateAttribute("N", indices.count)
    op.updateAttribute("hashed_output", hashedOutput)
    op.updateAttribute("num_buckets", numBuckets)
    op.updateAttribute("hash_key", hashKey)
    op.updateAttribute("sparse_types", values._typeList)
    op.updateAttribute("dense_types", denseInputs._typeList)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.updateAttribute("internal_type", internalType)
    op.addInputList(indices)
    op.addInputList(values)
    op.addInputList(shapes)
    op.addInputList(denseInputs)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseCross<
    SparseTypes: TensorArrayProtocol,
    DenseTypes: TensorArrayProtocol
>(
    indices: [Tensor<Int64>],
    _ values: SparseTypes,
    shapes: [Tensor<Int64>],
    denseInputs: DenseTypes,
    hashedOutput: Bool,
    numBuckets: Int64,
    hashKey: Int64,
    internalType: TensorDataType
) -> (outputIndices: Tensor<Int64>, outputValues: StringTensor, outputShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("SparseCross", nOutputs)
    op.updateAttribute("N", indices.count)
    op.updateAttribute("hashed_output", hashedOutput)
    op.updateAttribute("num_buckets", numBuckets)
    op.updateAttribute("hash_key", hashKey)
    op.updateAttribute("sparse_types", values._typeList)
    op.updateAttribute("dense_types", denseInputs._typeList)
    op.updateAttribute("out_type", TensorDataType(TF_STRING))
    op.updateAttribute("internal_type", internalType)
    op.addInputList(indices)
    op.addInputList(values)
    op.addInputList(shapes)
    op.addInputList(denseInputs)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseDenseCwiseAdd<T: TensorFlowNumeric>(
    spIndices: Tensor<Int64>,
    spValues: Tensor<T>,
    spShape: Tensor<Int64>,
    dense: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseDenseCwiseAdd", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(spIndices)
    op.addInput(spValues)
    op.addInput(spShape)
    op.addInput(dense)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseDenseCwiseDiv<T: TensorFlowNumeric>(
    spIndices: Tensor<Int64>,
    spValues: Tensor<T>,
    spShape: Tensor<Int64>,
    dense: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseDenseCwiseDiv", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(spIndices)
    op.addInput(spValues)
    op.addInput(spShape)
    op.addInput(dense)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseDenseCwiseMul<T: TensorFlowNumeric>(
    spIndices: Tensor<Int64>,
    spValues: Tensor<T>,
    spShape: Tensor<Int64>,
    dense: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseDenseCwiseMul", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(spIndices)
    op.addInput(spValues)
    op.addInput(spShape)
    op.addInput(dense)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseFillEmptyRows<T: TensorFlowScalar>(
    indices: Tensor<Int64>,
    _ values: Tensor<T>,
    denseShape: Tensor<Int64>,
    defaultValue: Tensor<T>
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, emptyRowIndicator: Tensor<Bool>, reverseIndexMap: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1) + Int(1)
    let op = makeOp("SparseFillEmptyRows", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(indices)
    op.addInput(values)
    op.addInput(denseShape)
    op.addInput(defaultValue)
    return op.execute(Int(1), Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseFillEmptyRowsGrad<T: TensorFlowScalar>(
    reverseIndexMap: Tensor<Int64>,
    gradValues: Tensor<T>
) -> (dValues: Tensor<T>, dDefaultValue: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("SparseFillEmptyRowsGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(reverseIndexMap)
    op.addInput(gradValues)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseMatMul<
    Ta: FloatingPoint & TensorFlowScalar,
    Tb: FloatingPoint & TensorFlowScalar
>(
    _ a: Tensor<Ta>,
    _ b: Tensor<Tb>,
    transposeA: Bool = false,
    transposeB: Bool = false,
    aIsSparse: Bool = false,
    bIsSparse: Bool = false
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("SparseMatMul", nOutputs)
    op.updateAttribute("transpose_a", transposeA)
    op.updateAttribute("transpose_b", transposeB)
    op.updateAttribute("a_is_sparse", aIsSparse)
    op.updateAttribute("b_is_sparse", bIsSparse)
    op.updateAttribute("Ta", Ta.tensorFlowDataType)
    op.updateAttribute("Tb", Tb.tensorFlowDataType)
    op.addInput(a)
    op.addInput(b)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseMatrixAdd<T: FloatingPoint & TensorFlowScalar>(
    _ a: VariantHandle,
    _ b: VariantHandle,
    alpha: Tensor<T>,
    beta: Tensor<T>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SparseMatrixAdd", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    op.addInput(b)
    op.addInput(alpha)
    op.addInput(beta)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseMatrixMatMul<T: TensorFlowScalar>(
    _ a: VariantHandle,
    _ b: Tensor<T>,
    transposeA: Bool = false,
    transposeB: Bool = false,
    adjointA: Bool = false,
    adjointB: Bool = false,
    transposeOutput: Bool = false,
    conjugateOutput: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseMatrixMatMul", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("transpose_a", transposeA)
    op.updateAttribute("transpose_b", transposeB)
    op.updateAttribute("adjoint_a", adjointA)
    op.updateAttribute("adjoint_b", adjointB)
    op.updateAttribute("transpose_output", transposeOutput)
    op.updateAttribute("conjugate_output", conjugateOutput)
    op.addInput(a)
    op.addInput(b)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseMatrixMul<T: TensorFlowScalar>(
    _ a: VariantHandle,
    _ b: Tensor<T>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SparseMatrixMul", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    op.addInput(b)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseMatrixNNZ(
    sparseMatrix: VariantHandle
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("SparseMatrixNNZ", nOutputs)
    op.addInput(sparseMatrix)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseMatrixOrderingAMD(
    _ input: VariantHandle
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("SparseMatrixOrderingAMD", nOutputs)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseMatrixSoftmax(
    logits: VariantHandle,
    type: TensorDataType
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SparseMatrixSoftmax", nOutputs)
    op.updateAttribute("type", type)
    op.addInput(logits)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseMatrixSoftmaxGrad(
    softmax: VariantHandle,
    gradSoftmax: VariantHandle,
    type: TensorDataType
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SparseMatrixSoftmaxGrad", nOutputs)
    op.updateAttribute("type", type)
    op.addInput(softmax)
    op.addInput(gradSoftmax)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseMatrixSparseCholesky(
    _ input: VariantHandle,
    permutation: Tensor<Int32>,
    type: TensorDataType
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SparseMatrixSparseCholesky", nOutputs)
    op.updateAttribute("type", type)
    op.addInput(input)
    op.addInput(permutation)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseMatrixSparseMatMul(
    _ a: VariantHandle,
    _ b: VariantHandle,
    type: TensorDataType,
    transposeA: Bool = false,
    transposeB: Bool = false,
    adjointA: Bool = false,
    adjointB: Bool = false
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SparseMatrixSparseMatMul", nOutputs)
    op.updateAttribute("type", type)
    op.updateAttribute("transpose_a", transposeA)
    op.updateAttribute("transpose_b", transposeB)
    op.updateAttribute("adjoint_a", adjointA)
    op.updateAttribute("adjoint_b", adjointB)
    op.addInput(a)
    op.addInput(b)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseMatrixTranspose(
    _ input: VariantHandle,
    conjugate: Bool = false,
    type: TensorDataType
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SparseMatrixTranspose", nOutputs)
    op.updateAttribute("conjugate", conjugate)
    op.updateAttribute("type", type)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseMatrixZeros(
    denseShape: Tensor<Int64>,
    type: TensorDataType
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SparseMatrixZeros", nOutputs)
    op.updateAttribute("type", type)
    op.addInput(denseShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseReduceMax<T: TensorFlowNumeric>(
    inputIndices: Tensor<Int64>,
    inputValues: Tensor<T>,
    inputShape: Tensor<Int64>,
    reductionAxes: Tensor<Int32>,
    keepDims: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseReduceMax", nOutputs)
    op.updateAttribute("keep_dims", keepDims)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(inputIndices)
    op.addInput(inputValues)
    op.addInput(inputShape)
    op.addInput(reductionAxes)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseReduceMaxSparse<T: TensorFlowNumeric>(
    inputIndices: Tensor<Int64>,
    inputValues: Tensor<T>,
    inputShape: Tensor<Int64>,
    reductionAxes: Tensor<Int32>,
    keepDims: Bool = false
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, outputShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("SparseReduceMaxSparse", nOutputs)
    op.updateAttribute("keep_dims", keepDims)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(inputIndices)
    op.addInput(inputValues)
    op.addInput(inputShape)
    op.addInput(reductionAxes)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseReduceSum<T: TensorFlowNumeric>(
    inputIndices: Tensor<Int64>,
    inputValues: Tensor<T>,
    inputShape: Tensor<Int64>,
    reductionAxes: Tensor<Int32>,
    keepDims: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseReduceSum", nOutputs)
    op.updateAttribute("keep_dims", keepDims)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(inputIndices)
    op.addInput(inputValues)
    op.addInput(inputShape)
    op.addInput(reductionAxes)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseReduceSumSparse<T: TensorFlowNumeric>(
    inputIndices: Tensor<Int64>,
    inputValues: Tensor<T>,
    inputShape: Tensor<Int64>,
    reductionAxes: Tensor<Int32>,
    keepDims: Bool = false
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, outputShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("SparseReduceSumSparse", nOutputs)
    op.updateAttribute("keep_dims", keepDims)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(inputIndices)
    op.addInput(inputValues)
    op.addInput(inputShape)
    op.addInput(reductionAxes)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseReorder<T: TensorFlowScalar>(
    inputIndices: Tensor<Int64>,
    inputValues: Tensor<T>,
    inputShape: Tensor<Int64>
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("SparseReorder", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(inputIndices)
    op.addInput(inputValues)
    op.addInput(inputShape)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseReshape(
    inputIndices: Tensor<Int64>,
    inputShape: Tensor<Int64>,
    newShape: Tensor<Int64>
) -> (outputIndices: Tensor<Int64>, outputShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("SparseReshape", nOutputs)
    op.addInput(inputIndices)
    op.addInput(inputShape)
    op.addInput(newShape)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseSegmentMean<
    T: FloatingPoint & TensorFlowScalar,
    Tidx: TensorFlowIndex
>(
    data: Tensor<T>,
    indices: Tensor<Tidx>,
    segmentIds: Tensor<Int32>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseSegmentMean", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(data)
    op.addInput(indices)
    op.addInput(segmentIds)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseSegmentMeanGrad<
    T: FloatingPoint & TensorFlowScalar,
    Tidx: TensorFlowIndex
>(
    grad: Tensor<T>,
    indices: Tensor<Tidx>,
    segmentIds: Tensor<Int32>,
    outputDim0: Tensor<Int32>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseSegmentMeanGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(grad)
    op.addInput(indices)
    op.addInput(segmentIds)
    op.addInput(outputDim0)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseSegmentMeanWithNumSegments<
    T: FloatingPoint & TensorFlowScalar,
    Tidx: TensorFlowIndex,
    Tnumsegments: TensorFlowIndex
>(
    data: Tensor<T>,
    indices: Tensor<Tidx>,
    segmentIds: Tensor<Int32>,
    numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseSegmentMeanWithNumSegments", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.updateAttribute("Tnumsegments", Tnumsegments.tensorFlowDataType)
    op.addInput(data)
    op.addInput(indices)
    op.addInput(segmentIds)
    op.addInput(numSegments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseSegmentSqrtN<
    T: FloatingPoint & TensorFlowScalar,
    Tidx: TensorFlowIndex
>(
    data: Tensor<T>,
    indices: Tensor<Tidx>,
    segmentIds: Tensor<Int32>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseSegmentSqrtN", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(data)
    op.addInput(indices)
    op.addInput(segmentIds)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseSegmentSqrtNGrad<
    T: FloatingPoint & TensorFlowScalar,
    Tidx: TensorFlowIndex
>(
    grad: Tensor<T>,
    indices: Tensor<Tidx>,
    segmentIds: Tensor<Int32>,
    outputDim0: Tensor<Int32>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseSegmentSqrtNGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(grad)
    op.addInput(indices)
    op.addInput(segmentIds)
    op.addInput(outputDim0)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseSegmentSqrtNWithNumSegments<
    T: FloatingPoint & TensorFlowScalar,
    Tidx: TensorFlowIndex,
    Tnumsegments: TensorFlowIndex
>(
    data: Tensor<T>,
    indices: Tensor<Tidx>,
    segmentIds: Tensor<Int32>,
    numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseSegmentSqrtNWithNumSegments", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.updateAttribute("Tnumsegments", Tnumsegments.tensorFlowDataType)
    op.addInput(data)
    op.addInput(indices)
    op.addInput(segmentIds)
    op.addInput(numSegments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseSegmentSum<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
>(
    data: Tensor<T>,
    indices: Tensor<Tidx>,
    segmentIds: Tensor<Int32>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseSegmentSum", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(data)
    op.addInput(indices)
    op.addInput(segmentIds)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseSegmentSumWithNumSegments<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex,
    Tnumsegments: TensorFlowIndex
>(
    data: Tensor<T>,
    indices: Tensor<Tidx>,
    segmentIds: Tensor<Int32>,
    numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseSegmentSumWithNumSegments", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.updateAttribute("Tnumsegments", Tnumsegments.tensorFlowDataType)
    op.addInput(data)
    op.addInput(indices)
    op.addInput(segmentIds)
    op.addInput(numSegments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseSlice<T: TensorFlowScalar>(
    indices: Tensor<Int64>,
    _ values: Tensor<T>,
    shape: Tensor<Int64>,
    start: Tensor<Int64>,
    size: Tensor<Int64>
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, outputShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("SparseSlice", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(indices)
    op.addInput(values)
    op.addInput(shape)
    op.addInput(start)
    op.addInput(size)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseSliceGrad<T: TensorFlowNumeric>(
    backpropValGrad: Tensor<T>,
    inputIndices: Tensor<Int64>,
    inputStart: Tensor<Int64>,
    outputIndices: Tensor<Int64>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseSliceGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(backpropValGrad)
    op.addInput(inputIndices)
    op.addInput(inputStart)
    op.addInput(outputIndices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseSoftmax<T: FloatingPoint & TensorFlowScalar>(
    spIndices: Tensor<Int64>,
    spValues: Tensor<T>,
    spShape: Tensor<Int64>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseSoftmax", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(spIndices)
    op.addInput(spValues)
    op.addInput(spShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseSoftmaxCrossEntropyWithLogits<
    T: FloatingPoint & TensorFlowScalar,
    Tlabels: TensorFlowIndex
>(
    features: Tensor<T>,
    labels: Tensor<Tlabels>
) -> (loss: Tensor<T>, backprop: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("SparseSoftmaxCrossEntropyWithLogits", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tlabels", Tlabels.tensorFlowDataType)
    op.addInput(features)
    op.addInput(labels)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseSparseMaximum<T: TensorFlowNumeric>(
    aIndices: Tensor<Int64>,
    aValues: Tensor<T>,
    aShape: Tensor<Int64>,
    bIndices: Tensor<Int64>,
    bValues: Tensor<T>,
    bShape: Tensor<Int64>
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("SparseSparseMaximum", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(aIndices)
    op.addInput(aValues)
    op.addInput(aShape)
    op.addInput(bIndices)
    op.addInput(bValues)
    op.addInput(bShape)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseSparseMinimum<T: TensorFlowNumeric>(
    aIndices: Tensor<Int64>,
    aValues: Tensor<T>,
    aShape: Tensor<Int64>,
    bIndices: Tensor<Int64>,
    bValues: Tensor<T>,
    bShape: Tensor<Int64>
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("SparseSparseMinimum", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(aIndices)
    op.addInput(aValues)
    op.addInput(aShape)
    op.addInput(bIndices)
    op.addInput(bValues)
    op.addInput(bShape)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseSplit<T: TensorFlowScalar>(
    splitDim: Tensor<Int64>,
    indices: Tensor<Int64>,
    _ values: Tensor<T>,
    shape: Tensor<Int64>,
    numSplit: Int64
) -> (outputIndices: [Tensor<Int64>], outputValues: [Tensor<T>], outputShape: [Tensor<Int64>]) {
  let nOutputs = Int(numSplit) + Int(numSplit) + Int(numSplit)
    let op = makeOp("SparseSplit", nOutputs)
    op.updateAttribute("num_split", numSplit)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(splitDim)
    op.addInput(indices)
    op.addInput(values)
    op.addInput(shape)
    return op.execute(Int(numSplit), Int(numSplit), Int(numSplit))
}

@inlinable @inline(__always)
public static func sparseTensorDenseAdd<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    aIndices: Tensor<Tindices>,
    aValues: Tensor<T>,
    aShape: Tensor<Tindices>,
    _ b: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseTensorDenseAdd", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(aIndices)
    op.addInput(aValues)
    op.addInput(aShape)
    op.addInput(b)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseTensorDenseMatMul<
    T: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    aIndices: Tensor<Tindices>,
    aValues: Tensor<T>,
    aShape: Tensor<Int64>,
    _ b: Tensor<T>,
    adjointA: Bool = false,
    adjointB: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseTensorDenseMatMul", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("adjoint_a", adjointA)
    op.updateAttribute("adjoint_b", adjointB)
    op.addInput(aIndices)
    op.addInput(aValues)
    op.addInput(aShape)
    op.addInput(b)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseTensorSliceDataset<Tvalues: TensorFlowScalar>(
    indices: Tensor<Int64>,
    _ values: Tensor<Tvalues>,
    denseShape: Tensor<Int64>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SparseTensorSliceDataset", nOutputs)
    op.updateAttribute("Tvalues", Tvalues.tensorFlowDataType)
    op.addInput(indices)
    op.addInput(values)
    op.addInput(denseShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseTensorToCSRSparseMatrix<T: FloatingPoint & TensorFlowScalar>(
    indices: Tensor<Int64>,
    _ values: Tensor<T>,
    denseShape: Tensor<Int64>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SparseTensorToCSRSparseMatrix", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(indices)
    op.addInput(values)
    op.addInput(denseShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseToDense<
    T: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    sparseIndices: Tensor<Tindices>,
    outputShape: Tensor<Tindices>,
    sparseValues: Tensor<T>,
    defaultValue: Tensor<T>,
    validateIndices: Bool = true
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SparseToDense", nOutputs)
    op.updateAttribute("validate_indices", validateIndices)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(sparseIndices)
    op.addInput(outputShape)
    op.addInput(sparseValues)
    op.addInput(defaultValue)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sparseToSparseSetOperation<T: TensorFlowInteger>(
    set1Indices: Tensor<Int64>,
    set1Values: Tensor<T>,
    set1Shape: Tensor<Int64>,
    set2Indices: Tensor<Int64>,
    set2Values: Tensor<T>,
    set2Shape: Tensor<Int64>,
    setOperation: String,
    validateIndices: Bool = true
) -> (resultIndices: Tensor<Int64>, resultValues: Tensor<T>, resultShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("SparseToSparseSetOperation", nOutputs)
    op.updateAttribute("set_operation", setOperation)
    op.updateAttribute("validate_indices", validateIndices)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(set1Indices)
    op.addInput(set1Values)
    op.addInput(set1Shape)
    op.addInput(set2Indices)
    op.addInput(set2Values)
    op.addInput(set2Shape)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func sparseToSparseSetOperation(
    set1Indices: Tensor<Int64>,
    set1Values: StringTensor,
    set1Shape: Tensor<Int64>,
    set2Indices: Tensor<Int64>,
    set2Values: StringTensor,
    set2Shape: Tensor<Int64>,
    setOperation: String,
    validateIndices: Bool = true
) -> (resultIndices: Tensor<Int64>, resultValues: StringTensor, resultShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("SparseToSparseSetOperation", nOutputs)
    op.updateAttribute("set_operation", setOperation)
    op.updateAttribute("validate_indices", validateIndices)
    op.updateAttribute("T", TensorDataType(TF_STRING))
    op.addInput(set1Indices)
    op.addInput(set1Values)
    op.addInput(set1Shape)
    op.addInput(set2Indices)
    op.addInput(set2Values)
    op.addInput(set2Shape)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func split<T: TensorFlowScalar>(
    splitDim: Tensor<Int32>,
    value: Tensor<T>,
    numSplit: Int64
) -> [Tensor<T>] {
  let nOutputs = Int(numSplit)
    let op = makeOp("Split", nOutputs)
    op.updateAttribute("num_split", numSplit)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(splitDim)
    op.addInput(value)
    return op.execute(Int(numSplit))
}

@inlinable @inline(__always)
public static func splitV<
    T: TensorFlowScalar,
    Tlen: TensorFlowIndex
>(
    value: Tensor<T>,
    sizeSplits: Tensor<Tlen>,
    splitDim: Tensor<Int32>,
    numSplit: Int64
) -> [Tensor<T>] {
  let nOutputs = Int(numSplit)
    let op = makeOp("SplitV", nOutputs)
    op.updateAttribute("num_split", numSplit)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tlen", Tlen.tensorFlowDataType)
    op.addInput(value)
    op.addInput(sizeSplits)
    op.addInput(splitDim)
    return op.execute(Int(numSplit))
}

@inlinable @inline(__always)
public static func sqlDataset(
    driverName: StringTensor,
    dataSourceName: StringTensor,
    query: StringTensor,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("SqlDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(driverName)
    op.addInput(dataSourceName)
    op.addInput(query)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sqrt<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Sqrt", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sqrtGrad<T: FloatingPoint & TensorFlowScalar>(
    _ y: Tensor<T>,
    dy: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SqrtGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(y)
    op.addInput(dy)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func square<T: TensorFlowNumeric>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Square", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func squaredDifference<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("SquaredDifference", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func squeeze<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    squeezeDims: [Int32]
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Squeeze", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("squeeze_dims", squeezeDims)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stackCloseV2(
    handle: ResourceHandle
) {
  let nOutputs = 0
    let op = makeOp("StackCloseV2", nOutputs)
    op.addInput(handle)
    op.execute()
}

@inlinable @inline(__always)
public static func stackPopV2<ElemType: TensorFlowScalar>(
    handle: ResourceHandle
) -> Tensor<ElemType> {
  let nOutputs = Int(1)
    let op = makeOp("StackPopV2", nOutputs)
    op.updateAttribute("elem_type", ElemType.tensorFlowDataType)
    op.addInput(handle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stackPushV2<T: TensorFlowScalar>(
    handle: ResourceHandle,
    elem: Tensor<T>,
    swapMemory: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("StackPushV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("swap_memory", swapMemory)
    op.addInput(handle)
    op.addInput(elem)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stackV2(
    maxSize: Tensor<Int32>,
    elemType: TensorDataType,
    stackName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("StackV2", nOutputs)
    op.updateAttribute("elem_type", elemType)
    op.updateAttribute("stack_name", stackName)
    op.addInput(maxSize)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stage<Dtypes: TensorArrayProtocol>(
    _ values: Dtypes,
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    container: String,
    sharedName: String
) {
  let nOutputs = 0
    let op = makeOp("Stage", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", values._typeList)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.addInputList(values)
    op.execute()
}

@inlinable @inline(__always)
public static func stageClear(
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    dtypes: [TensorDataType],
    container: String,
    sharedName: String
) {
  let nOutputs = 0
    let op = makeOp("StageClear", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", dtypes)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.execute()
}

@inlinable @inline(__always)
public static func stagePeek<Dtypes: TensorGroup>(
    index: Tensor<Int32>,
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    container: String,
    sharedName: String
) -> Dtypes {
  let nOutputs = Int(Dtypes._typeList.count)
    let op = makeOp("StagePeek", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", Dtypes._typeList)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.addInput(index)
    return op.execute(Int(Dtypes._typeList.count))
}

@inlinable @inline(__always)
public static func stageSize(
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    dtypes: [TensorDataType],
    container: String,
    sharedName: String
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("StageSize", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", dtypes)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statefulPartitionedCall<
    Tin: TensorArrayProtocol,
    Tout: TensorGroup,
    FIn: TensorGroup,
    FOut: TensorGroup
>(
    args: Tin,
    f: (FIn) -> FOut,
    config: String,
    configProto: String,
    executorType: String
) -> Tout {
  let nOutputs = Int(Tout._typeList.count)
    let op = makeOp("StatefulPartitionedCall", nOutputs)
    op.updateAttribute("Tin", args._typeList)
    op.updateAttribute("Tout", Tout._typeList)
    op.updateAttribute("f", f)
    op.updateAttribute("config", config)
    op.updateAttribute("config_proto", configProto)
    op.updateAttribute("executor_type", executorType)
    op.addInputList(args)
    return op.execute(Int(Tout._typeList.count))
}

@inlinable @inline(__always)
public static func statefulRandomBinomial<
    S: TensorFlowIndex,
    T: TensorFlowNumeric,
    Dtype: TensorFlowNumeric
>(
    resource: ResourceHandle,
    algorithm: Tensor<Int64>,
    shape: Tensor<S>,
    counts: Tensor<T>,
    probs: Tensor<T>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("StatefulRandomBinomial", nOutputs)
    op.updateAttribute("S", S.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(algorithm)
    op.addInput(shape)
    op.addInput(counts)
    op.addInput(probs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statefulStandardNormal<
    Dtype: TensorFlowScalar,
    ShapeDtype: TensorFlowScalar
>(
    resource: ResourceHandle,
    shape: Tensor<ShapeDtype>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("StatefulStandardNormal", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape_dtype", ShapeDtype.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statefulStandardNormalV2<
    Dtype: TensorFlowScalar,
    ShapeDtype: TensorFlowScalar
>(
    resource: ResourceHandle,
    algorithm: Tensor<Int64>,
    shape: Tensor<ShapeDtype>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("StatefulStandardNormalV2", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape_dtype", ShapeDtype.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(algorithm)
    op.addInput(shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statefulTruncatedNormal<
    Dtype: TensorFlowScalar,
    ShapeDtype: TensorFlowScalar
>(
    resource: ResourceHandle,
    algorithm: Tensor<Int64>,
    shape: Tensor<ShapeDtype>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("StatefulTruncatedNormal", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape_dtype", ShapeDtype.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(algorithm)
    op.addInput(shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statefulUniform<
    Dtype: TensorFlowScalar,
    ShapeDtype: TensorFlowScalar
>(
    resource: ResourceHandle,
    algorithm: Tensor<Int64>,
    shape: Tensor<ShapeDtype>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("StatefulUniform", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape_dtype", ShapeDtype.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(algorithm)
    op.addInput(shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statefulUniformFullInt<
    Dtype: TensorFlowScalar,
    ShapeDtype: TensorFlowScalar
>(
    resource: ResourceHandle,
    algorithm: Tensor<Int64>,
    shape: Tensor<ShapeDtype>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("StatefulUniformFullInt", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape_dtype", ShapeDtype.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(algorithm)
    op.addInput(shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statefulUniformInt<
    Dtype: TensorFlowScalar,
    ShapeDtype: TensorFlowScalar
>(
    resource: ResourceHandle,
    algorithm: Tensor<Int64>,
    shape: Tensor<ShapeDtype>,
    minval: Tensor<Dtype>,
    maxval: Tensor<Dtype>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("StatefulUniformInt", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("shape_dtype", ShapeDtype.tensorFlowDataType)
    op.addInput(resource)
    op.addInput(algorithm)
    op.addInput(shape)
    op.addInput(minval)
    op.addInput(maxval)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statelessIf<
    Tcond: TensorFlowScalar,
    Tin: TensorArrayProtocol,
    Tout: TensorGroup,
    ThenbranchIn: TensorGroup,
    ThenbranchOut: TensorGroup,
    ElsebranchIn: TensorGroup,
    ElsebranchOut: TensorGroup
>(
    cond: Tensor<Tcond>,
    _ input: Tin,
    thenBranch: (ThenbranchIn) -> ThenbranchOut,
    elseBranch: (ElsebranchIn) -> ElsebranchOut,
    outputShapes: [TensorShape?]
) -> Tout {
  let nOutputs = Int(Tout._typeList.count)
    let op = makeOp("StatelessIf", nOutputs)
    op.updateAttribute("Tcond", Tcond.tensorFlowDataType)
    op.updateAttribute("Tin", input._typeList)
    op.updateAttribute("Tout", Tout._typeList)
    op.updateAttribute("then_branch", thenBranch)
    op.updateAttribute("else_branch", elseBranch)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(cond)
    op.addInputList(input)
    return op.execute(Int(Tout._typeList.count))
}

@inlinable @inline(__always)
public static func statelessMultinomial<
    T: TensorFlowNumeric,
    Tseed: TensorFlowIndex,
    OutputDtype: TensorFlowIndex
>(
    logits: Tensor<T>,
    numSamples: Tensor<Int32>,
    seed: Tensor<Tseed>
) -> Tensor<OutputDtype> {
  let nOutputs = Int(1)
    let op = makeOp("StatelessMultinomial", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tseed", Tseed.tensorFlowDataType)
    op.updateAttribute("output_dtype", OutputDtype.tensorFlowDataType)
    op.addInput(logits)
    op.addInput(numSamples)
    op.addInput(seed)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statelessRandomNormal<
    Dtype: FloatingPoint & TensorFlowScalar,
    T: TensorFlowIndex,
    Tseed: TensorFlowIndex
>(
    shape: Tensor<T>,
    seed: Tensor<Tseed>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("StatelessRandomNormal", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tseed", Tseed.tensorFlowDataType)
    op.addInput(shape)
    op.addInput(seed)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statelessRandomUniform<
    Dtype: FloatingPoint & TensorFlowScalar,
    T: TensorFlowIndex,
    Tseed: TensorFlowIndex
>(
    shape: Tensor<T>,
    seed: Tensor<Tseed>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("StatelessRandomUniform", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tseed", Tseed.tensorFlowDataType)
    op.addInput(shape)
    op.addInput(seed)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statelessRandomUniformInt<
    Dtype: TensorFlowIndex,
    T: TensorFlowIndex,
    Tseed: TensorFlowIndex
>(
    shape: Tensor<T>,
    seed: Tensor<Tseed>,
    minval: Tensor<Dtype>,
    maxval: Tensor<Dtype>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("StatelessRandomUniformInt", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tseed", Tseed.tensorFlowDataType)
    op.addInput(shape)
    op.addInput(seed)
    op.addInput(minval)
    op.addInput(maxval)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statelessTruncatedNormal<
    Dtype: FloatingPoint & TensorFlowScalar,
    T: TensorFlowIndex,
    Tseed: TensorFlowIndex
>(
    shape: Tensor<T>,
    seed: Tensor<Tseed>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("StatelessTruncatedNormal", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tseed", Tseed.tensorFlowDataType)
    op.addInput(shape)
    op.addInput(seed)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statelessWhile<
    T: TensorArrayProtocol,
    CondIn: TensorGroup,
    CondOut: TensorGroup,
    BodyIn: TensorGroup,
    BodyOut: TensorGroup
>(
    _ input: T,
    cond: (CondIn) -> CondOut,
    body: (BodyIn) -> BodyOut,
    outputShapes: [TensorShape?],
    parallelIterations: Int64 = 10
) -> T {
  let nOutputs = Int(input._typeList.count)
    let op = makeOp("StatelessWhile", nOutputs)
    op.updateAttribute("T", input._typeList)
    op.updateAttribute("cond", cond)
    op.updateAttribute("body", body)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("parallel_iterations", parallelIterations)
    op.addInputList(input)
    return op.execute(Int(input._typeList.count))
}

@inlinable @inline(__always)
public static func staticRegexFullMatch(
    _ input: StringTensor,
    pattern: String
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("StaticRegexFullMatch", nOutputs)
    op.updateAttribute("pattern", pattern)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func staticRegexReplace(
    _ input: StringTensor,
    pattern: String,
    rewrite: String,
    replaceGlobal: Bool = true
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("StaticRegexReplace", nOutputs)
    op.updateAttribute("pattern", pattern)
    op.updateAttribute("rewrite", rewrite)
    op.updateAttribute("replace_global", replaceGlobal)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statsAggregatorHandle(
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("StatsAggregatorHandle", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statsAggregatorHandleV2(
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("StatsAggregatorHandleV2", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func statsAggregatorSetSummaryWriter(
    statsAggregator: ResourceHandle,
    summary: ResourceHandle
) {
  let nOutputs = 0
    let op = makeOp("StatsAggregatorSetSummaryWriter", nOutputs)
    op.addInput(statsAggregator)
    op.addInput(summary)
    op.execute()
}

@inlinable @inline(__always)
public static func statsAggregatorSummary(
    iterator: ResourceHandle
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("StatsAggregatorSummary", nOutputs)
    op.addInput(iterator)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stopGradient<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("StopGradient", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stridedSlice<
    T: TensorFlowScalar,
    Index: TensorFlowIndex
>(
    _ input: Tensor<T>,
    begin: Tensor<Index>,
    end: Tensor<Index>,
    strides: Tensor<Index>,
    beginMask: Int64 = 0,
    endMask: Int64 = 0,
    ellipsisMask: Int64 = 0,
    newAxisMask: Int64 = 0,
    shrinkAxisMask: Int64 = 0
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("StridedSlice", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Index", Index.tensorFlowDataType)
    op.updateAttribute("begin_mask", beginMask)
    op.updateAttribute("end_mask", endMask)
    op.updateAttribute("ellipsis_mask", ellipsisMask)
    op.updateAttribute("new_axis_mask", newAxisMask)
    op.updateAttribute("shrink_axis_mask", shrinkAxisMask)
    op.addInput(input)
    op.addInput(begin)
    op.addInput(end)
    op.addInput(strides)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stridedSliceGrad<
    T: TensorFlowScalar,
    Index: TensorFlowIndex
>(
    shape: Tensor<Index>,
    begin: Tensor<Index>,
    end: Tensor<Index>,
    strides: Tensor<Index>,
    dy: Tensor<T>,
    beginMask: Int64 = 0,
    endMask: Int64 = 0,
    ellipsisMask: Int64 = 0,
    newAxisMask: Int64 = 0,
    shrinkAxisMask: Int64 = 0
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("StridedSliceGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Index", Index.tensorFlowDataType)
    op.updateAttribute("begin_mask", beginMask)
    op.updateAttribute("end_mask", endMask)
    op.updateAttribute("ellipsis_mask", ellipsisMask)
    op.updateAttribute("new_axis_mask", newAxisMask)
    op.updateAttribute("shrink_axis_mask", shrinkAxisMask)
    op.addInput(shape)
    op.addInput(begin)
    op.addInput(end)
    op.addInput(strides)
    op.addInput(dy)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stringFormat<T: TensorArrayProtocol>(
    inputs: T,
    template: String = "%s",
    placeholder: String = "%s",
    summarize: Int64 = 3
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("StringFormat", nOutputs)
    op.updateAttribute("T", inputs._typeList)
    op.updateAttribute("template", template)
    op.updateAttribute("placeholder", placeholder)
    op.updateAttribute("summarize", summarize)
    op.addInputList(inputs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stringJoin(
    inputs: [StringTensor],
    separator: String
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("StringJoin", nOutputs)
    op.updateAttribute("N", inputs.count)
    op.updateAttribute("separator", separator)
    op.addInputList(inputs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stringLength(
    _ input: StringTensor,
    unit: Unit = .byte
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("StringLength", nOutputs)
    op.updateAttribute("unit", unit.cName)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stringListAttr(
    _ a: [String],
    _ b: String
) {
  let nOutputs = 0
    let op = makeOp("StringListAttr", nOutputs)
    op.updateAttribute("a", a)
    op.updateAttribute("b", b)
    op.execute()
}

@inlinable @inline(__always)
public static func stringLower(
    _ input: StringTensor,
    encoding: String
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("StringLower", nOutputs)
    op.updateAttribute("encoding", encoding)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stringNGrams<Tsplits: TensorFlowIndex>(
    data: StringTensor,
    dataSplits: Tensor<Tsplits>,
    separator: String,
    ngramWidths: [Int32],
    leftPad: String,
    rightPad: String,
    padWidth: Int64,
    preserveShortSequences: Bool
) -> (ngrams: StringTensor, ngramsSplits: Tensor<Tsplits>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("StringNGrams", nOutputs)
    op.updateAttribute("separator", separator)
    op.updateAttribute("ngram_widths", ngramWidths)
    op.updateAttribute("left_pad", leftPad)
    op.updateAttribute("right_pad", rightPad)
    op.updateAttribute("pad_width", padWidth)
    op.updateAttribute("preserve_short_sequences", preserveShortSequences)
    op.updateAttribute("Tsplits", Tsplits.tensorFlowDataType)
    op.addInput(data)
    op.addInput(dataSplits)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func stringSplit(
    _ input: StringTensor,
    delimiter: StringTensor,
    skipEmpty: Bool = true
) -> (indices: Tensor<Int64>, values: StringTensor, shape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("StringSplit", nOutputs)
    op.updateAttribute("skip_empty", skipEmpty)
    op.addInput(input)
    op.addInput(delimiter)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func stringSplitV2(
    _ input: StringTensor,
    sep: StringTensor,
    maxsplit: Int64 = -1
) -> (indices: Tensor<Int64>, values: StringTensor, shape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("StringSplitV2", nOutputs)
    op.updateAttribute("maxsplit", maxsplit)
    op.addInput(input)
    op.addInput(sep)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func stringStrip(
    _ input: StringTensor
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("StringStrip", nOutputs)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stringToHashBucket(
    stringTensor: StringTensor,
    numBuckets: Int64
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("StringToHashBucket", nOutputs)
    op.updateAttribute("num_buckets", numBuckets)
    op.addInput(stringTensor)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stringToHashBucketFast(
    _ input: StringTensor,
    numBuckets: Int64
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("StringToHashBucketFast", nOutputs)
    op.updateAttribute("num_buckets", numBuckets)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stringToHashBucketStrong(
    _ input: StringTensor,
    numBuckets: Int64,
    key: [Int32]
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("StringToHashBucketStrong", nOutputs)
    op.updateAttribute("num_buckets", numBuckets)
    op.updateAttribute("key", key)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stringToNumber<OutType: TensorFlowNumeric>(
    stringTensor: StringTensor
) -> Tensor<OutType> {
  let nOutputs = Int(1)
    let op = makeOp("StringToNumber", nOutputs)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(stringTensor)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stringUpper(
    _ input: StringTensor,
    encoding: String
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("StringUpper", nOutputs)
    op.updateAttribute("encoding", encoding)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func stubResourceHandleOp(
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("StubResourceHandleOp", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sub<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Sub", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func substr<T: TensorFlowIndex>(
    _ input: StringTensor,
    pos: Tensor<T>,
    len: Tensor<T>,
    unit: Unit = .byte
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("Substr", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("unit", unit.cName)
    op.addInput(input)
    op.addInput(pos)
    op.addInput(len)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func sum<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
>(
    _ input: Tensor<T>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Sum", nOutputs)
    op.updateAttribute("keep_dims", keepDims)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(input)
    op.addInput(reductionIndices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func summaryWriter(
    sharedName: String,
    container: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("SummaryWriter", nOutputs)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("container", container)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func svd<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    computeUv: Bool = true,
    fullMatrices: Bool = false
) -> (s: Tensor<T>, u: Tensor<T>, v: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("Svd", nOutputs)
    op.updateAttribute("compute_uv", computeUv)
    op.updateAttribute("full_matrices", fullMatrices)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func switch_<T: TensorFlowScalar>(
    data: Tensor<T>,
    pred: Tensor<Bool>
) -> (outputFalse: Tensor<T>, outputTrue: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("Switch", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(data)
    op.addInput(pred)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func symbolicGradient<
    Tin: TensorArrayProtocol,
    Tout: TensorGroup,
    FIn: TensorGroup,
    FOut: TensorGroup
>(
    _ input: Tin,
    f: (FIn) -> FOut
) -> Tout {
  let nOutputs = Int(Tout._typeList.count)
    let op = makeOp("SymbolicGradient", nOutputs)
    op.updateAttribute("Tin", input._typeList)
    op.updateAttribute("Tout", Tout._typeList)
    op.updateAttribute("f", f)
    op.addInputList(input)
    return op.execute(Int(Tout._typeList.count))
}

@inlinable @inline(__always)
public static func tFRecordDataset(
    filenames: StringTensor,
    compressionType: StringTensor,
    bufferSize: Tensor<Int64>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TFRecordDataset", nOutputs)
    op.addInput(filenames)
    op.addInput(compressionType)
    op.addInput(bufferSize)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tFRecordReaderV2(
    container: String,
    sharedName: String,
    compressionType: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("TFRecordReaderV2", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("compression_type", compressionType)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tPUCompilationResult(
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("TPUCompilationResult", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tPUEmbeddingActivations(
    embeddingVariable: Tensor<Float>,
    slicedActivations: Tensor<Float>,
    tableId: Int64,
    lookupId: Int64
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("TPUEmbeddingActivations", nOutputs)
    op.updateAttribute("table_id", tableId)
    op.updateAttribute("lookup_id", lookupId)
    op.addInput(embeddingVariable)
    op.addInput(slicedActivations)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tPUOrdinalSelector(
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("TPUOrdinalSelector", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tPUPartitionedCall<
    Tin: TensorArrayProtocol,
    Tout: TensorGroup,
    FIn: TensorGroup,
    FOut: TensorGroup
>(
    args: Tin,
    deviceOrdinal: Tensor<Int32>,
    f: (FIn) -> FOut,
    autotunerThresh: Int64 = 0
) -> Tout {
  let nOutputs = Int(Tout._typeList.count)
    let op = makeOp("TPUPartitionedCall", nOutputs)
    op.updateAttribute("Tin", args._typeList)
    op.updateAttribute("Tout", Tout._typeList)
    op.updateAttribute("f", f)
    op.updateAttribute("autotuner_thresh", autotunerThresh)
    op.addInputList(args)
    op.addInput(deviceOrdinal)
    return op.execute(Int(Tout._typeList.count))
}

@inlinable @inline(__always)
public static func tPUReplicateMetadata(
    numReplicas: Int64,
    numCoresPerReplica: Int64 = 1,
    topology: String,
    useTpu: Bool = true,
    deviceAssignment: [Int32],
    computationShape: [Int32],
    hostComputeCore: [String],
    paddingMap: [String],
    stepMarkerLocation: String = "STEP_MARK_AT_ENTRY",
    allowSoftPlacement: Bool = false
) {
  let nOutputs = 0
    let op = makeOp("TPUReplicateMetadata", nOutputs)
    op.updateAttribute("num_replicas", numReplicas)
    op.updateAttribute("num_cores_per_replica", numCoresPerReplica)
    op.updateAttribute("topology", topology)
    op.updateAttribute("use_tpu", useTpu)
    op.updateAttribute("device_assignment", deviceAssignment)
    op.updateAttribute("computation_shape", computationShape)
    op.updateAttribute("host_compute_core", hostComputeCore)
    op.updateAttribute("padding_map", paddingMap)
    op.updateAttribute("step_marker_location", stepMarkerLocation)
    op.updateAttribute("allow_soft_placement", allowSoftPlacement)
    op.execute()
}

@inlinable @inline(__always)
public static func tPUReplicatedInput<T: TensorFlowScalar>(
    inputs: [Tensor<T>],
    isMirroredVariable: Bool = false,
    index: Int64 = -1
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("TPUReplicatedInput", nOutputs)
    op.updateAttribute("N", inputs.count)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("is_mirrored_variable", isMirroredVariable)
    op.updateAttribute("index", index)
    op.addInputList(inputs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tPUReplicatedOutput<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    numReplicas: Int64
) -> [Tensor<T>] {
  let nOutputs = Int(numReplicas)
    let op = makeOp("TPUReplicatedOutput", nOutputs)
    op.updateAttribute("num_replicas", numReplicas)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(numReplicas))
}

@inlinable @inline(__always)
public static func tRTEngineOp<
    SegmentfuncIn: TensorGroup,
    SegmentfuncOut: TensorGroup,
    Int: TensorArrayProtocol,
    Outt: TensorGroup
>(
    inTensor: Int,
    serializedSegment: String,
    segmentFunc: (SegmentfuncIn) -> SegmentfuncOut,
    maxCachedEnginesCount: Int64 = 1,
    workspaceSizeBytes: Int64,
    precisionMode: PrecisionMode,
    calibrationData: String,
    useCalibration: Bool = true,
    segmentFuncdefName: String,
    cachedEngineBatches: [Int32],
    fixedInputSize: Bool = true,
    inputShapes: [TensorShape?],
    outputShapes: [TensorShape?],
    staticEngine: Bool = true
) -> Outt {
  let nOutputs = Int(Outt._typeList.count)
    let op = makeOp("TRTEngineOp", nOutputs)
    op.updateAttribute("serialized_segment", serializedSegment)
    op.updateAttribute("segment_func", segmentFunc)
    op.updateAttribute("InT", inTensor._typeList)
    op.updateAttribute("OutT", Outt._typeList)
    op.updateAttribute("max_cached_engines_count", maxCachedEnginesCount)
    op.updateAttribute("workspace_size_bytes", workspaceSizeBytes)
    op.updateAttribute("precision_mode", precisionMode.cName)
    op.updateAttribute("calibration_data", calibrationData)
    op.updateAttribute("use_calibration", useCalibration)
    op.updateAttribute("segment_funcdef_name", segmentFuncdefName)
    op.updateAttribute("cached_engine_batches", cachedEngineBatches)
    op.updateAttribute("fixed_input_size", fixedInputSize)
    op.updateAttribute("input_shapes", inputShapes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("static_engine", staticEngine)
    op.addInputList(inTensor)
    return op.execute(Int(Outt._typeList.count))
}

@inlinable @inline(__always)
public static func takeDataset(
    inputDataset: VariantHandle,
    count: Tensor<Int64>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TakeDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(count)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func takeManySparseFromTensorsMap<Dtype: TensorFlowScalar>(
    sparseHandles: Tensor<Int64>,
    container: String,
    sharedName: String
) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<Dtype>, sparseShape: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("TakeManySparseFromTensorsMap", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.addInput(sparseHandles)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func takeWhileDataset<
    PredicateIn: TensorGroup,
    PredicateOut: TensorGroup,
    Targuments: TensorArrayProtocol
>(
    inputDataset: VariantHandle,
    otherArguments: Targuments,
    predicate: (PredicateIn) -> PredicateOut,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TakeWhileDataset", nOutputs)
    op.updateAttribute("predicate", predicate)
    op.updateAttribute("Targuments", otherArguments._typeList)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInputList(otherArguments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tan<T: TensorFlowNumeric>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Tan", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tanh<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Tanh", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tanhGrad<T: FloatingPoint & TensorFlowScalar>(
    _ y: Tensor<T>,
    dy: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("TanhGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(y)
    op.addInput(dy)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayCloseV2(
    handle: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("TensorArrayCloseV2", nOutputs)
    op.addInput(handle)
    op.execute()
}

@inlinable @inline(__always)
public static func tensorArrayCloseV3(
    handle: ResourceHandle
) {
  let nOutputs = 0
    let op = makeOp("TensorArrayCloseV3", nOutputs)
    op.addInput(handle)
    op.execute()
}

@inlinable @inline(__always)
public static func tensorArrayConcatV2<Dtype: TensorFlowScalar>(
    handle: StringTensor,
    flowIn: Tensor<Float>,
    elementShapeExcept0: TensorShape?
) -> (value: Tensor<Dtype>, lengths: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("TensorArrayConcatV2", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("element_shape_except0", elementShapeExcept0)
    op.addInput(handle)
    op.addInput(flowIn)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayConcatV3<Dtype: TensorFlowScalar>(
    handle: ResourceHandle,
    flowIn: Tensor<Float>,
    elementShapeExcept0: TensorShape?
) -> (value: Tensor<Dtype>, lengths: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("TensorArrayConcatV3", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("element_shape_except0", elementShapeExcept0)
    op.addInput(handle)
    op.addInput(flowIn)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayGatherV2<Dtype: TensorFlowScalar>(
    handle: StringTensor,
    indices: Tensor<Int32>,
    flowIn: Tensor<Float>,
    elementShape: TensorShape?
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("TensorArrayGatherV2", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("element_shape", elementShape)
    op.addInput(handle)
    op.addInput(indices)
    op.addInput(flowIn)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayGatherV3<Dtype: TensorFlowScalar>(
    handle: ResourceHandle,
    indices: Tensor<Int32>,
    flowIn: Tensor<Float>,
    elementShape: TensorShape?
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("TensorArrayGatherV3", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("element_shape", elementShape)
    op.addInput(handle)
    op.addInput(indices)
    op.addInput(flowIn)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayGradV2(
    handle: StringTensor,
    flowIn: Tensor<Float>,
    source: String
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("TensorArrayGradV2", nOutputs)
    op.updateAttribute("source", source)
    op.addInput(handle)
    op.addInput(flowIn)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayGradV3(
    handle: ResourceHandle,
    flowIn: Tensor<Float>,
    source: String
) -> (gradHandle: ResourceHandle, flowOut: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("TensorArrayGradV3", nOutputs)
    op.updateAttribute("source", source)
    op.addInput(handle)
    op.addInput(flowIn)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayGradWithShape(
    handle: ResourceHandle,
    flowIn: Tensor<Float>,
    shapeToPrepend: Tensor<Int32>,
    source: String
) -> (gradHandle: ResourceHandle, flowOut: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("TensorArrayGradWithShape", nOutputs)
    op.updateAttribute("source", source)
    op.addInput(handle)
    op.addInput(flowIn)
    op.addInput(shapeToPrepend)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayReadV2<Dtype: TensorFlowScalar>(
    handle: StringTensor,
    index: Tensor<Int32>,
    flowIn: Tensor<Float>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("TensorArrayReadV2", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(handle)
    op.addInput(index)
    op.addInput(flowIn)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayReadV3<Dtype: TensorFlowScalar>(
    handle: ResourceHandle,
    index: Tensor<Int32>,
    flowIn: Tensor<Float>
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("TensorArrayReadV3", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.addInput(handle)
    op.addInput(index)
    op.addInput(flowIn)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayScatterV2<T: TensorFlowScalar>(
    handle: StringTensor,
    indices: Tensor<Int32>,
    value: Tensor<T>,
    flowIn: Tensor<Float>
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("TensorArrayScatterV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(handle)
    op.addInput(indices)
    op.addInput(value)
    op.addInput(flowIn)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayScatterV3<T: TensorFlowScalar>(
    handle: ResourceHandle,
    indices: Tensor<Int32>,
    value: Tensor<T>,
    flowIn: Tensor<Float>
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("TensorArrayScatterV3", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(handle)
    op.addInput(indices)
    op.addInput(value)
    op.addInput(flowIn)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArraySizeV2(
    handle: StringTensor,
    flowIn: Tensor<Float>
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("TensorArraySizeV2", nOutputs)
    op.addInput(handle)
    op.addInput(flowIn)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArraySizeV3(
    handle: ResourceHandle,
    flowIn: Tensor<Float>
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("TensorArraySizeV3", nOutputs)
    op.addInput(handle)
    op.addInput(flowIn)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArraySplitV2<T: TensorFlowScalar>(
    handle: StringTensor,
    value: Tensor<T>,
    lengths: Tensor<Int64>,
    flowIn: Tensor<Float>
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("TensorArraySplitV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(handle)
    op.addInput(value)
    op.addInput(lengths)
    op.addInput(flowIn)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArraySplitV3<T: TensorFlowScalar>(
    handle: ResourceHandle,
    value: Tensor<T>,
    lengths: Tensor<Int64>,
    flowIn: Tensor<Float>
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("TensorArraySplitV3", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(handle)
    op.addInput(value)
    op.addInput(lengths)
    op.addInput(flowIn)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayV2(
    size: Tensor<Int32>,
    dtype: TensorDataType,
    elementShape: TensorShape?,
    dynamicSize: Bool = false,
    clearAfterRead: Bool = true,
    tensorArrayName: String
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("TensorArrayV2", nOutputs)
    op.updateAttribute("dtype", dtype)
    op.updateAttribute("element_shape", elementShape)
    op.updateAttribute("dynamic_size", dynamicSize)
    op.updateAttribute("clear_after_read", clearAfterRead)
    op.updateAttribute("tensor_array_name", tensorArrayName)
    op.addInput(size)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayV3(
    size: Tensor<Int32>,
    dtype: TensorDataType,
    elementShape: TensorShape?,
    dynamicSize: Bool = false,
    clearAfterRead: Bool = true,
    identicalElementShapes: Bool = false,
    tensorArrayName: String
) -> (handle: ResourceHandle, flow: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("TensorArrayV3", nOutputs)
    op.updateAttribute("dtype", dtype)
    op.updateAttribute("element_shape", elementShape)
    op.updateAttribute("dynamic_size", dynamicSize)
    op.updateAttribute("clear_after_read", clearAfterRead)
    op.updateAttribute("identical_element_shapes", identicalElementShapes)
    op.updateAttribute("tensor_array_name", tensorArrayName)
    op.addInput(size)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayWriteV2<T: TensorFlowScalar>(
    handle: StringTensor,
    index: Tensor<Int32>,
    value: Tensor<T>,
    flowIn: Tensor<Float>
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("TensorArrayWriteV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(handle)
    op.addInput(index)
    op.addInput(value)
    op.addInput(flowIn)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorArrayWriteV3<T: TensorFlowScalar>(
    handle: ResourceHandle,
    index: Tensor<Int32>,
    value: Tensor<T>,
    flowIn: Tensor<Float>
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("TensorArrayWriteV3", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(handle)
    op.addInput(index)
    op.addInput(value)
    op.addInput(flowIn)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorDataset<ToutputTypes: TensorArrayProtocol>(
    components: ToutputTypes,
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorDataset", nOutputs)
    op.updateAttribute("Toutput_types", components._typeList)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInputList(components)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorForestCreateTreeVariable(
    treeHandle: ResourceHandle,
    treeConfig: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("TensorForestCreateTreeVariable", nOutputs)
    op.addInput(treeHandle)
    op.addInput(treeConfig)
    op.execute()
}

@inlinable @inline(__always)
public static func tensorForestTreeDeserialize(
    treeHandle: ResourceHandle,
    treeConfig: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("TensorForestTreeDeserialize", nOutputs)
    op.addInput(treeHandle)
    op.addInput(treeConfig)
    op.execute()
}

@inlinable @inline(__always)
public static func tensorForestTreeIsInitializedOp(
    treeHandle: ResourceHandle
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("TensorForestTreeIsInitializedOp", nOutputs)
    op.addInput(treeHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorForestTreePredict(
    treeHandle: ResourceHandle,
    denseFeatures: Tensor<Float>,
    logitsDimension: Int64
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("TensorForestTreePredict", nOutputs)
    op.updateAttribute("logits_dimension", logitsDimension)
    op.addInput(treeHandle)
    op.addInput(denseFeatures)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorForestTreeResourceHandleOp(
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorForestTreeResourceHandleOp", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorForestTreeSerialize(
    treeHandle: ResourceHandle
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("TensorForestTreeSerialize", nOutputs)
    op.addInput(treeHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorForestTreeSize(
    treeHandle: ResourceHandle
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("TensorForestTreeSize", nOutputs)
    op.addInput(treeHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListConcat<ElementDtype: TensorFlowScalar>(
    inputHandle: VariantHandle,
    elementShape: TensorShape?
) -> (tensor: Tensor<ElementDtype>, lengths: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("TensorListConcat", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.updateAttribute("element_shape", elementShape)
    op.addInput(inputHandle)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func tensorListConcatLists(
    inputA: VariantHandle,
    inputB: VariantHandle,
    elementDtype: TensorDataType
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorListConcatLists", nOutputs)
    op.updateAttribute("element_dtype", elementDtype)
    op.addInput(inputA)
    op.addInput(inputB)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListConcatV2<
    ElementDtype: TensorFlowScalar,
    ShapeType: TensorFlowIndex
>(
    inputHandle: VariantHandle,
    elementShape: Tensor<ShapeType>,
    leadingDims: Tensor<Int64>
) -> (tensor: Tensor<ElementDtype>, lengths: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("TensorListConcatV2", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.updateAttribute("shape_type", ShapeType.tensorFlowDataType)
    op.addInput(inputHandle)
    op.addInput(elementShape)
    op.addInput(leadingDims)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func tensorListElementShape<ShapeType: TensorFlowIndex>(
    inputHandle: VariantHandle
) -> Tensor<ShapeType> {
  let nOutputs = Int(1)
    let op = makeOp("TensorListElementShape", nOutputs)
    op.updateAttribute("shape_type", ShapeType.tensorFlowDataType)
    op.addInput(inputHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListFromTensor<
    ElementDtype: TensorFlowScalar,
    ShapeType: TensorFlowIndex
>(
    _ tensor: Tensor<ElementDtype>,
    elementShape: Tensor<ShapeType>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorListFromTensor", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.updateAttribute("shape_type", ShapeType.tensorFlowDataType)
    op.addInput(tensor)
    op.addInput(elementShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListGather<ElementDtype: TensorFlowScalar>(
    inputHandle: VariantHandle,
    indices: Tensor<Int32>,
    elementShape: Tensor<Int32>
) -> Tensor<ElementDtype> {
  let nOutputs = Int(1)
    let op = makeOp("TensorListGather", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.addInput(inputHandle)
    op.addInput(indices)
    op.addInput(elementShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListGetItem<ElementDtype: TensorFlowScalar>(
    inputHandle: VariantHandle,
    index: Tensor<Int32>,
    elementShape: Tensor<Int32>
) -> Tensor<ElementDtype> {
  let nOutputs = Int(1)
    let op = makeOp("TensorListGetItem", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.addInput(inputHandle)
    op.addInput(index)
    op.addInput(elementShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListLength(
    inputHandle: VariantHandle
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("TensorListLength", nOutputs)
    op.addInput(inputHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListPopBack<ElementDtype: TensorFlowScalar>(
    inputHandle: VariantHandle,
    elementShape: Tensor<Int32>
) -> (outputHandle: VariantHandle, tensor: Tensor<ElementDtype>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("TensorListPopBack", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.addInput(inputHandle)
    op.addInput(elementShape)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func tensorListPushBack<ElementDtype: TensorFlowScalar>(
    inputHandle: VariantHandle,
    _ tensor: Tensor<ElementDtype>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorListPushBack", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.addInput(inputHandle)
    op.addInput(tensor)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListPushBackBatch<ElementDtype: TensorFlowScalar>(
    inputHandles: VariantHandle,
    _ tensor: Tensor<ElementDtype>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorListPushBackBatch", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.addInput(inputHandles)
    op.addInput(tensor)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListReserve<ShapeType: TensorFlowIndex>(
    elementShape: Tensor<ShapeType>,
    numElements: Tensor<Int32>,
    elementDtype: TensorDataType
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorListReserve", nOutputs)
    op.updateAttribute("element_dtype", elementDtype)
    op.updateAttribute("shape_type", ShapeType.tensorFlowDataType)
    op.addInput(elementShape)
    op.addInput(numElements)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListResize(
    inputHandle: VariantHandle,
    size: Tensor<Int32>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorListResize", nOutputs)
    op.addInput(inputHandle)
    op.addInput(size)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListScatter<
    ElementDtype: TensorFlowScalar,
    ShapeType: TensorFlowIndex
>(
    _ tensor: Tensor<ElementDtype>,
    indices: Tensor<Int32>,
    elementShape: Tensor<ShapeType>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorListScatter", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.updateAttribute("shape_type", ShapeType.tensorFlowDataType)
    op.addInput(tensor)
    op.addInput(indices)
    op.addInput(elementShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListScatterIntoExistingList<ElementDtype: TensorFlowScalar>(
    inputHandle: VariantHandle,
    _ tensor: Tensor<ElementDtype>,
    indices: Tensor<Int32>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorListScatterIntoExistingList", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.addInput(inputHandle)
    op.addInput(tensor)
    op.addInput(indices)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListScatterV2<
    ElementDtype: TensorFlowScalar,
    ShapeType: TensorFlowIndex
>(
    _ tensor: Tensor<ElementDtype>,
    indices: Tensor<Int32>,
    elementShape: Tensor<ShapeType>,
    numElements: Tensor<Int32>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorListScatterV2", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.updateAttribute("shape_type", ShapeType.tensorFlowDataType)
    op.addInput(tensor)
    op.addInput(indices)
    op.addInput(elementShape)
    op.addInput(numElements)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListSetItem<ElementDtype: TensorFlowScalar>(
    inputHandle: VariantHandle,
    index: Tensor<Int32>,
    item: Tensor<ElementDtype>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorListSetItem", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.addInput(inputHandle)
    op.addInput(index)
    op.addInput(item)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListSplit<
    ElementDtype: TensorFlowScalar,
    ShapeType: TensorFlowIndex
>(
    _ tensor: Tensor<ElementDtype>,
    elementShape: Tensor<ShapeType>,
    lengths: Tensor<Int64>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorListSplit", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.updateAttribute("shape_type", ShapeType.tensorFlowDataType)
    op.addInput(tensor)
    op.addInput(elementShape)
    op.addInput(lengths)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorListStack<ElementDtype: TensorFlowScalar>(
    inputHandle: VariantHandle,
    elementShape: Tensor<Int32>,
    numElements: Int64 = -1
) -> Tensor<ElementDtype> {
  let nOutputs = Int(1)
    let op = makeOp("TensorListStack", nOutputs)
    op.updateAttribute("element_dtype", ElementDtype.tensorFlowDataType)
    op.updateAttribute("num_elements", numElements)
    op.addInput(inputHandle)
    op.addInput(elementShape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorScatterAdd<
    T: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    _ tensor: Tensor<T>,
    indices: Tensor<Tindices>,
    updates: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("TensorScatterAdd", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(tensor)
    op.addInput(indices)
    op.addInput(updates)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorScatterSub<
    T: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    _ tensor: Tensor<T>,
    indices: Tensor<Tindices>,
    updates: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("TensorScatterSub", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(tensor)
    op.addInput(indices)
    op.addInput(updates)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorScatterUpdate<
    T: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    _ tensor: Tensor<T>,
    indices: Tensor<Tindices>,
    updates: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("TensorScatterUpdate", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(tensor)
    op.addInput(indices)
    op.addInput(updates)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorSliceDataset<ToutputTypes: TensorArrayProtocol>(
    components: ToutputTypes,
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TensorSliceDataset", nOutputs)
    op.updateAttribute("Toutput_types", components._typeList)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInputList(components)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorStridedSliceUpdate<
    T: TensorFlowScalar,
    Index: TensorFlowIndex
>(
    _ input: Tensor<T>,
    begin: Tensor<Index>,
    end: Tensor<Index>,
    strides: Tensor<Index>,
    value: Tensor<T>,
    beginMask: Int64 = 0,
    endMask: Int64 = 0,
    ellipsisMask: Int64 = 0,
    newAxisMask: Int64 = 0,
    shrinkAxisMask: Int64 = 0
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("TensorStridedSliceUpdate", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Index", Index.tensorFlowDataType)
    op.updateAttribute("begin_mask", beginMask)
    op.updateAttribute("end_mask", endMask)
    op.updateAttribute("ellipsis_mask", ellipsisMask)
    op.updateAttribute("new_axis_mask", newAxisMask)
    op.updateAttribute("shrink_axis_mask", shrinkAxisMask)
    op.addInput(input)
    op.addInput(begin)
    op.addInput(end)
    op.addInput(strides)
    op.addInput(value)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorSummary<T: TensorFlowScalar>(
    _ tensor: Tensor<T>,
    description: String,
    labels: [String],
    displayName: String
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("TensorSummary", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("description", description)
    op.updateAttribute("labels", labels)
    op.updateAttribute("display_name", displayName)
    op.addInput(tensor)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tensorSummaryV2<T: TensorFlowScalar>(
    tag: StringTensor,
    _ tensor: Tensor<T>,
    serializedSummaryMetadata: StringTensor
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("TensorSummaryV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(tag)
    op.addInput(tensor)
    op.addInput(serializedSummaryMetadata)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func testAttr<T: FloatingPoint & TensorFlowScalar>(
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("TestAttr", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func testStringOutput(
    _ input: Tensor<Float>
) -> (output1: Tensor<Float>, output2: StringTensor) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("TestStringOutput", nOutputs)
    op.addInput(input)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func textLineDataset(
    filenames: StringTensor,
    compressionType: StringTensor,
    bufferSize: Tensor<Int64>
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("TextLineDataset", nOutputs)
    op.addInput(filenames)
    op.addInput(compressionType)
    op.addInput(bufferSize)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func textLineReaderV2(
    skipHeaderLines: Int64 = 0,
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("TextLineReaderV2", nOutputs)
    op.updateAttribute("skip_header_lines", skipHeaderLines)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func threadPoolDataset(
    inputDataset: VariantHandle,
    threadPool: ResourceHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ThreadPoolDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(threadPool)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func threadPoolHandle(
    numThreads: Int64,
    maxIntraOpParallelism: Int64 = 1,
    displayName: String,
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("ThreadPoolHandle", nOutputs)
    op.updateAttribute("num_threads", numThreads)
    op.updateAttribute("max_intra_op_parallelism", maxIntraOpParallelism)
    op.updateAttribute("display_name", displayName)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func threadUnsafeUnigramCandidateSampler(
    trueClasses: Tensor<Int64>,
    numTrue: Int64,
    numSampled: Int64,
    unique: Bool,
    rangeMax: Int64,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> (sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>, sampledExpectedCount: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("ThreadUnsafeUnigramCandidateSampler", nOutputs)
    op.updateAttribute("num_true", numTrue)
    op.updateAttribute("num_sampled", numSampled)
    op.updateAttribute("unique", unique)
    op.updateAttribute("range_max", rangeMax)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.addInput(trueClasses)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func tile<
    T: TensorFlowScalar,
    Tmultiples: TensorFlowIndex
>(
    _ input: Tensor<T>,
    multiples: Tensor<Tmultiples>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Tile", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tmultiples", Tmultiples.tensorFlowDataType)
    op.addInput(input)
    op.addInput(multiples)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tileGrad<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    multiples: Tensor<Int32>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("TileGrad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    op.addInput(multiples)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func timestamp(
) -> Tensor<Double> {
  let nOutputs = Int(1)
    let op = makeOp("Timestamp", nOutputs)
    
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func topK<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    k: Int64,
    sorted: Bool = true
) -> (values: Tensor<T>, indices: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("TopK", nOutputs)
    op.updateAttribute("k", k)
    op.updateAttribute("sorted", sorted)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func topKV2<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    k: Tensor<Int32>,
    sorted: Bool = true
) -> (values: Tensor<T>, indices: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("TopKV2", nOutputs)
    op.updateAttribute("sorted", sorted)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    op.addInput(k)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func transpose<
    T: TensorFlowScalar,
    Tperm: TensorFlowIndex
>(
    _ x: Tensor<T>,
    perm: Tensor<Tperm>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Transpose", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tperm", Tperm.tensorFlowDataType)
    op.addInput(x)
    op.addInput(perm)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tridiagonalMatMul<T: FloatingPoint & TensorFlowScalar>(
    superdiag: Tensor<T>,
    maindiag: Tensor<T>,
    subdiag: Tensor<T>,
    rhs: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("TridiagonalMatMul", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(superdiag)
    op.addInput(maindiag)
    op.addInput(subdiag)
    op.addInput(rhs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tridiagonalSolve<T: FloatingPoint & TensorFlowScalar>(
    diagonals: Tensor<T>,
    rhs: Tensor<T>,
    partialPivoting: Bool = true
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("TridiagonalSolve", nOutputs)
    op.updateAttribute("partial_pivoting", partialPivoting)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(diagonals)
    op.addInput(rhs)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func truncateDiv<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("TruncateDiv", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func truncateMod<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("TruncateMod", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func truncatedNormal<
    Dtype: FloatingPoint & TensorFlowScalar,
    T: TensorFlowIndex
>(
    shape: Tensor<T>,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("TruncatedNormal", nOutputs)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func tryRpc(
    address: StringTensor,
    method: StringTensor,
    request: StringTensor,
    protocol_: String,
    failFast: Bool = true,
    timeoutInMs: Int64 = 0
) -> (response: StringTensor, statusCode: Tensor<Int32>, statusMessage: StringTensor) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("TryRpc", nOutputs)
    op.updateAttribute("protocol", protocol_)
    op.updateAttribute("fail_fast", failFast)
    op.updateAttribute("timeout_in_ms", timeoutInMs)
    op.addInput(address)
    op.addInput(method)
    op.addInput(request)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func twoFloatInputs(
    _ a: Tensor<Float>,
    _ b: Tensor<Float>
) {
  let nOutputs = 0
    let op = makeOp("TwoFloatInputs", nOutputs)
    op.addInput(a)
    op.addInput(b)
    op.execute()
}

@inlinable @inline(__always)
public static func twoFloatInputsFloatOutput(
    _ a: Tensor<Float>,
    _ b: Tensor<Float>
) -> Tensor<Float> {
  let nOutputs = Int(1)
    let op = makeOp("TwoFloatInputsFloatOutput", nOutputs)
    op.addInput(a)
    op.addInput(b)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func twoFloatInputsIntOutput(
    _ a: Tensor<Float>,
    _ b: Tensor<Float>
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("TwoFloatInputsIntOutput", nOutputs)
    op.addInput(a)
    op.addInput(b)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func twoFloatOutputs(
) -> (a: Tensor<Float>, b: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("TwoFloatOutputs", nOutputs)
    
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func twoIntInputs(
    _ a: Tensor<Int32>,
    _ b: Tensor<Int32>
) {
  let nOutputs = 0
    let op = makeOp("TwoIntInputs", nOutputs)
    op.addInput(a)
    op.addInput(b)
    op.execute()
}

@inlinable @inline(__always)
public static func twoIntOutputs(
) -> (a: Tensor<Int32>, b: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("TwoIntOutputs", nOutputs)
    
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func typeList<T: TensorArrayProtocol>(
    _ a: T
) {
  let nOutputs = 0
    let op = makeOp("TypeList", nOutputs)
    op.updateAttribute("T", a._typeList)
    op.addInputList(a)
    op.execute()
}

@inlinable @inline(__always)
public static func typeListRestrict<T: TensorArrayProtocol>(
    _ a: T
) {
  let nOutputs = 0
    let op = makeOp("TypeListRestrict", nOutputs)
    op.updateAttribute("T", a._typeList)
    op.addInputList(a)
    op.execute()
}

@inlinable @inline(__always)
public static func typeListTwice<T: TensorArrayProtocol>(
    _ a: T,
    _ b: T
) {
  let nOutputs = 0
    let op = makeOp("TypeListTwice", nOutputs)
    op.updateAttribute("T", a._typeList)
    op.addInputList(a)
    op.addInputList(b)
    op.execute()
}

@inlinable @inline(__always)
public static func unary<T: TensorFlowScalar>(
    _ a: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Unary", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func unbatch<T: TensorFlowScalar>(
    batchedTensor: Tensor<T>,
    batchIndex: Tensor<Int64>,
    id: Tensor<Int64>,
    timeoutMicros: Int64,
    container: String,
    sharedName: String
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Unbatch", nOutputs)
    op.updateAttribute("timeout_micros", timeoutMicros)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(batchedTensor)
    op.addInput(batchIndex)
    op.addInput(id)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func unbatchDataset(
    inputDataset: VariantHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("UnbatchDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func unbatchGrad<T: TensorFlowScalar>(
    originalInput: Tensor<T>,
    batchIndex: Tensor<Int64>,
    grad: Tensor<T>,
    id: Tensor<Int64>,
    container: String,
    sharedName: String
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("UnbatchGrad", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(originalInput)
    op.addInput(batchIndex)
    op.addInput(grad)
    op.addInput(id)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func unicodeDecode<Tsplits: TensorFlowIndex>(
    _ input: StringTensor,
    inputEncoding: String,
    errors: Errors = .replace,
    replacementChar: Int64 = 65533,
    replaceControlCharacters: Bool = false
) -> (rowSplits: Tensor<Tsplits>, charValues: Tensor<Int32>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("UnicodeDecode", nOutputs)
    op.updateAttribute("input_encoding", inputEncoding)
    op.updateAttribute("errors", errors.cName)
    op.updateAttribute("replacement_char", replacementChar)
    op.updateAttribute("replace_control_characters", replaceControlCharacters)
    op.updateAttribute("Tsplits", Tsplits.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func unicodeDecodeWithOffsets<Tsplits: TensorFlowIndex>(
    _ input: StringTensor,
    inputEncoding: String,
    errors: Errors = .replace,
    replacementChar: Int64 = 65533,
    replaceControlCharacters: Bool = false
) -> (rowSplits: Tensor<Tsplits>, charValues: Tensor<Int32>, charToByteStarts: Tensor<Int64>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("UnicodeDecodeWithOffsets", nOutputs)
    op.updateAttribute("input_encoding", inputEncoding)
    op.updateAttribute("errors", errors.cName)
    op.updateAttribute("replacement_char", replacementChar)
    op.updateAttribute("replace_control_characters", replaceControlCharacters)
    op.updateAttribute("Tsplits", Tsplits.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func unicodeEncode<Tsplits: TensorFlowIndex>(
    inputValues: Tensor<Int32>,
    inputSplits: Tensor<Tsplits>,
    errors: Errors = .replace,
    outputEncoding: OutputEncoding,
    replacementChar: Int64 = 65533
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("UnicodeEncode", nOutputs)
    op.updateAttribute("errors", errors.cName)
    op.updateAttribute("output_encoding", outputEncoding.cName)
    op.updateAttribute("replacement_char", replacementChar)
    op.updateAttribute("Tsplits", Tsplits.tensorFlowDataType)
    op.addInput(inputValues)
    op.addInput(inputSplits)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func unicodeScript(
    _ input: Tensor<Int32>
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("UnicodeScript", nOutputs)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func unicodeTranscode(
    _ input: StringTensor,
    inputEncoding: String,
    outputEncoding: OutputEncoding,
    errors: Errors = .replace,
    replacementChar: Int64 = 65533,
    replaceControlCharacters: Bool = false
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("UnicodeTranscode", nOutputs)
    op.updateAttribute("input_encoding", inputEncoding)
    op.updateAttribute("output_encoding", outputEncoding.cName)
    op.updateAttribute("errors", errors.cName)
    op.updateAttribute("replacement_char", replacementChar)
    op.updateAttribute("replace_control_characters", replaceControlCharacters)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func uniformCandidateSampler(
    trueClasses: Tensor<Int64>,
    numTrue: Int64,
    numSampled: Int64,
    unique: Bool,
    rangeMax: Int64,
    seed: Int64 = 0,
    seed2: Int64 = 0
) -> (sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>, sampledExpectedCount: Tensor<Float>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("UniformCandidateSampler", nOutputs)
    op.updateAttribute("num_true", numTrue)
    op.updateAttribute("num_sampled", numSampled)
    op.updateAttribute("unique", unique)
    op.updateAttribute("range_max", rangeMax)
    op.updateAttribute("seed", seed)
    op.updateAttribute("seed2", seed2)
    op.addInput(trueClasses)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func unique<
    T: TensorFlowScalar,
    OutIdx: TensorFlowIndex
>(
    _ x: Tensor<T>
) -> (y: Tensor<T>, idx: Tensor<OutIdx>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("Unique", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("out_idx", OutIdx.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func uniqueDataset(
    inputDataset: VariantHandle,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("UniqueDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func uniqueV2<
    T: TensorFlowScalar,
    Taxis: TensorFlowIndex,
    OutIdx: TensorFlowIndex
>(
    _ x: Tensor<T>,
    axis: Tensor<Taxis>
) -> (y: Tensor<T>, idx: Tensor<OutIdx>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("UniqueV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Taxis", Taxis.tensorFlowDataType)
    op.updateAttribute("out_idx", OutIdx.tensorFlowDataType)
    op.addInput(x)
    op.addInput(axis)
    return op.execute(Int(1), Int(1))
}

@inlinable @inline(__always)
public static func uniqueWithCounts<
    T: TensorFlowScalar,
    OutIdx: TensorFlowIndex
>(
    _ x: Tensor<T>
) -> (y: Tensor<T>, idx: Tensor<OutIdx>, count: Tensor<OutIdx>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("UniqueWithCounts", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("out_idx", OutIdx.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func uniqueWithCountsV2<
    T: TensorFlowScalar,
    Taxis: TensorFlowIndex,
    OutIdx: TensorFlowIndex
>(
    _ x: Tensor<T>,
    axis: Tensor<Taxis>
) -> (y: Tensor<T>, idx: Tensor<OutIdx>, count: Tensor<OutIdx>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("UniqueWithCountsV2", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Taxis", Taxis.tensorFlowDataType)
    op.updateAttribute("out_idx", OutIdx.tensorFlowDataType)
    op.addInput(x)
    op.addInput(axis)
    return op.execute(Int(1), Int(1), Int(1))
}

@inlinable @inline(__always)
public static func unpack<T: TensorFlowScalar>(
    value: Tensor<T>,
    num: Int64,
    axis: Int64 = 0
) -> [Tensor<T>] {
  let nOutputs = Int(num)
    let op = makeOp("Unpack", nOutputs)
    op.updateAttribute("num", num)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("axis", axis)
    op.addInput(value)
    return op.execute(Int(num))
}

@inlinable @inline(__always)
public static func unravelIndex<Tidx: TensorFlowIndex>(
    indices: Tensor<Tidx>,
    dims: Tensor<Tidx>
) -> Tensor<Tidx> {
  let nOutputs = Int(1)
    let op = makeOp("UnravelIndex", nOutputs)
    op.updateAttribute("Tidx", Tidx.tensorFlowDataType)
    op.addInput(indices)
    op.addInput(dims)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func unsortedSegmentJoin<
    Tindices: TensorFlowIndex,
    Tnumsegments: TensorFlowIndex
>(
    inputs: StringTensor,
    segmentIds: Tensor<Tindices>,
    numSegments: Tensor<Tnumsegments>,
    separator: String
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("UnsortedSegmentJoin", nOutputs)
    op.updateAttribute("separator", separator)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("Tnumsegments", Tnumsegments.tensorFlowDataType)
    op.addInput(inputs)
    op.addInput(segmentIds)
    op.addInput(numSegments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func unsortedSegmentMax<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex,
    Tnumsegments: TensorFlowIndex
>(
    data: Tensor<T>,
    segmentIds: Tensor<Tindices>,
    numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("UnsortedSegmentMax", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("Tnumsegments", Tnumsegments.tensorFlowDataType)
    op.addInput(data)
    op.addInput(segmentIds)
    op.addInput(numSegments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func unsortedSegmentMin<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex,
    Tnumsegments: TensorFlowIndex
>(
    data: Tensor<T>,
    segmentIds: Tensor<Tindices>,
    numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("UnsortedSegmentMin", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("Tnumsegments", Tnumsegments.tensorFlowDataType)
    op.addInput(data)
    op.addInput(segmentIds)
    op.addInput(numSegments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func unsortedSegmentProd<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex,
    Tnumsegments: TensorFlowIndex
>(
    data: Tensor<T>,
    segmentIds: Tensor<Tindices>,
    numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("UnsortedSegmentProd", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("Tnumsegments", Tnumsegments.tensorFlowDataType)
    op.addInput(data)
    op.addInput(segmentIds)
    op.addInput(numSegments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func unsortedSegmentSum<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex,
    Tnumsegments: TensorFlowIndex
>(
    data: Tensor<T>,
    segmentIds: Tensor<Tindices>,
    numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("UnsortedSegmentSum", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("Tnumsegments", Tnumsegments.tensorFlowDataType)
    op.addInput(data)
    op.addInput(segmentIds)
    op.addInput(numSegments)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func unstage<Dtypes: TensorGroup>(
    capacity: Int64 = 0,
    memoryLimit: Int64 = 0,
    container: String,
    sharedName: String
) -> Dtypes {
  let nOutputs = Int(Dtypes._typeList.count)
    let op = makeOp("Unstage", nOutputs)
    op.updateAttribute("capacity", capacity)
    op.updateAttribute("memory_limit", memoryLimit)
    op.updateAttribute("dtypes", Dtypes._typeList)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(Dtypes._typeList.count))
}

@inlinable @inline(__always)
public static func unwrapDatasetVariant(
    inputHandle: VariantHandle
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("UnwrapDatasetVariant", nOutputs)
    op.addInput(inputHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func upperBound<
    T: TensorFlowScalar,
    OutType: TensorFlowIndex
>(
    sortedInputs: Tensor<T>,
    _ values: Tensor<T>
) -> Tensor<OutType> {
  let nOutputs = Int(1)
    let op = makeOp("UpperBound", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(sortedInputs)
    op.addInput(values)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func varHandleOp(
    container: String,
    sharedName: String,
    dtype: TensorDataType,
    shape: TensorShape?
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("VarHandleOp", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    op.updateAttribute("dtype", dtype)
    op.updateAttribute("shape", shape)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func varIsInitializedOp(
    resource: ResourceHandle
) -> Tensor<Bool> {
  let nOutputs = Int(1)
    let op = makeOp("VarIsInitializedOp", nOutputs)
    op.addInput(resource)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func variableShape<OutType: TensorFlowIndex>(
    _ input: ResourceHandle
) -> Tensor<OutType> {
  let nOutputs = Int(1)
    let op = makeOp("VariableShape", nOutputs)
    op.updateAttribute("out_type", OutType.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func where_<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<Int64> {
  let nOutputs = Int(1)
    let op = makeOp("Where", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func while_<
    T: TensorArrayProtocol,
    CondIn: TensorGroup,
    CondOut: TensorGroup,
    BodyIn: TensorGroup,
    BodyOut: TensorGroup
>(
    _ input: T,
    cond: (CondIn) -> CondOut,
    body: (BodyIn) -> BodyOut,
    outputShapes: [TensorShape?],
    parallelIterations: Int64 = 10
) -> T {
  let nOutputs = Int(input._typeList.count)
    let op = makeOp("While", nOutputs)
    op.updateAttribute("T", input._typeList)
    op.updateAttribute("cond", cond)
    op.updateAttribute("body", body)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("parallel_iterations", parallelIterations)
    op.addInputList(input)
    return op.execute(Int(input._typeList.count))
}

@inlinable @inline(__always)
public static func wholeFileReaderV2(
    container: String,
    sharedName: String
) -> ResourceHandle {
  let nOutputs = Int(1)
    let op = makeOp("WholeFileReaderV2", nOutputs)
    op.updateAttribute("container", container)
    op.updateAttribute("shared_name", sharedName)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func windowDataset(
    inputDataset: VariantHandle,
    size: Tensor<Int64>,
    shift: Tensor<Int64>,
    stride: Tensor<Int64>,
    dropRemainder: Tensor<Bool>,
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("WindowDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.addInput(inputDataset)
    op.addInput(size)
    op.addInput(shift)
    op.addInput(stride)
    op.addInput(dropRemainder)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func workerHeartbeat(
    request: StringTensor
) -> StringTensor {
  let nOutputs = Int(1)
    let op = makeOp("WorkerHeartbeat", nOutputs)
    op.addInput(request)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func wrapDatasetVariant(
    inputHandle: VariantHandle
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("WrapDatasetVariant", nOutputs)
    op.addInput(inputHandle)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func writeAudioSummary(
    writer: ResourceHandle,
    step: Tensor<Int64>,
    tag: StringTensor,
    _ tensor: Tensor<Float>,
    sampleRate: Tensor<Float>,
    maxOutputs: Int64 = 3
) {
  let nOutputs = 0
    let op = makeOp("WriteAudioSummary", nOutputs)
    op.updateAttribute("max_outputs", maxOutputs)
    op.addInput(writer)
    op.addInput(step)
    op.addInput(tag)
    op.addInput(tensor)
    op.addInput(sampleRate)
    op.execute()
}

@inlinable @inline(__always)
public static func writeFile(
    filename: StringTensor,
    contents: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("WriteFile", nOutputs)
    op.addInput(filename)
    op.addInput(contents)
    op.execute()
}

@inlinable @inline(__always)
public static func writeGraphSummary(
    writer: ResourceHandle,
    step: Tensor<Int64>,
    _ tensor: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("WriteGraphSummary", nOutputs)
    op.addInput(writer)
    op.addInput(step)
    op.addInput(tensor)
    op.execute()
}

@inlinable @inline(__always)
public static func writeHistogramSummary<T: TensorFlowNumeric>(
    writer: ResourceHandle,
    step: Tensor<Int64>,
    tag: StringTensor,
    _ values: Tensor<T>
) {
  let nOutputs = 0
    let op = makeOp("WriteHistogramSummary", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(writer)
    op.addInput(step)
    op.addInput(tag)
    op.addInput(values)
    op.execute()
}

@inlinable @inline(__always)
public static func writeImageSummary<T: TensorFlowNumeric>(
    writer: ResourceHandle,
    step: Tensor<Int64>,
    tag: StringTensor,
    _ tensor: Tensor<T>,
    badColor: Tensor<UInt8>,
    maxImages: Int64 = 3
) {
  let nOutputs = 0
    let op = makeOp("WriteImageSummary", nOutputs)
    op.updateAttribute("max_images", maxImages)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(writer)
    op.addInput(step)
    op.addInput(tag)
    op.addInput(tensor)
    op.addInput(badColor)
    op.execute()
}

@inlinable @inline(__always)
public static func writeRawProtoSummary(
    writer: ResourceHandle,
    step: Tensor<Int64>,
    _ tensor: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("WriteRawProtoSummary", nOutputs)
    op.addInput(writer)
    op.addInput(step)
    op.addInput(tensor)
    op.execute()
}

@inlinable @inline(__always)
public static func writeScalarSummary<T: TensorFlowNumeric>(
    writer: ResourceHandle,
    step: Tensor<Int64>,
    tag: StringTensor,
    value: Tensor<T>
) {
  let nOutputs = 0
    let op = makeOp("WriteScalarSummary", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(writer)
    op.addInput(step)
    op.addInput(tag)
    op.addInput(value)
    op.execute()
}

@inlinable @inline(__always)
public static func writeSummary<T: TensorFlowScalar>(
    writer: ResourceHandle,
    step: Tensor<Int64>,
    _ tensor: Tensor<T>,
    tag: StringTensor,
    summaryMetadata: StringTensor
) {
  let nOutputs = 0
    let op = makeOp("WriteSummary", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(writer)
    op.addInput(step)
    op.addInput(tensor)
    op.addInput(tag)
    op.addInput(summaryMetadata)
    op.execute()
}

@inlinable @inline(__always)
public static func xdivy<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Xdivy", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

/// Helper operator for performing XLA-style broadcasts
///
/// Broadcasts `lhs` and `rhs` to the same rank, by adding size 1 dimensions to
/// whichever of `lhs` and `rhs` has the lower rank, using XLA's broadcasting rules
/// for binary operators.
///
/// - Parameters:
///     - lhs: the LHS input tensor
///     - rhs: the RHS input tensor
///     - broadcast_dims: an XLA-style broadcast dimension specification
///
/// - Outputs:
///     - lhs_output: the broadcasted LHS tensor
///     - rhs_output: the broadcasted RHS tensor
@inlinable @inline(__always)
public static func xlaBroadcastHelper<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    lhs: Tensor<T>,
    rhs: Tensor<T>,
    broadcastDims: Tensor<Tindices>
) -> (lhsOutput: Tensor<T>, rhsOutput: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("XlaBroadcastHelper", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(lhs)
    op.addInput(rhs)
    op.addInput(broadcastDims)
    return op.execute(Int(1), Int(1))
}

/// Operator that connects the output of an XLA computation to other consumer graph nodes.
@inlinable @inline(__always)
public static func xlaClusterOutput<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("XlaClusterOutput", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

/// Wraps the XLA ConvGeneralDilated operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution
/// .
///
/// - Parameters:
///     - lhs: the input tensor
///     - rhs: the kernel tensor
///     - window_strides: the inter-window strides
///     - padding: the padding to apply at the start and end of each input dimensions
///     - lhs_dilation: dilation to apply between input elements
///     - rhs_dilation: dilation to apply between kernel elements
///     - feature_group_count: number of feature groups for grouped convolution.
///
/// - Attrs:
///     - dimension_numbers: a serialized xla::ConvolutionDimensionNumbers proto.
///     - precision_config: a serialized xla::PrecisionConfig proto.
@inlinable @inline(__always)
public static func xlaConv<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
>(
    lhs: Tensor<T>,
    rhs: Tensor<T>,
    windowStrides: Tensor<Tindices>,
    padding: Tensor<Tindices>,
    lhsDilation: Tensor<Tindices>,
    rhsDilation: Tensor<Tindices>,
    featureGroupCount: Tensor<Tindices>,
    dimensionNumbers: String,
    precisionConfig: String
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("XlaConv", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("dimension_numbers", dimensionNumbers)
    op.updateAttribute("precision_config", precisionConfig)
    op.addInput(lhs)
    op.addInput(rhs)
    op.addInput(windowStrides)
    op.addInput(padding)
    op.addInput(lhsDilation)
    op.addInput(rhsDilation)
    op.addInput(featureGroupCount)
    return op.execute(Int(1))
}

/// Wraps the XLA DotGeneral operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#dotgeneral
/// .
///
/// - Parameters:
///     - lhs: the LHS tensor
///     - rhs: the RHS tensor
///
/// - Attrs:
///     - dimension_numbers: a serialized xla::DotDimensionNumbers proto.
///     - precision_config: a serialized xla::PrecisionConfig proto.
@inlinable @inline(__always)
public static func xlaDot<T: TensorFlowNumeric>(
    lhs: Tensor<T>,
    rhs: Tensor<T>,
    dimensionNumbers: String,
    precisionConfig: String
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("XlaDot", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("dimension_numbers", dimensionNumbers)
    op.updateAttribute("precision_config", precisionConfig)
    op.addInput(lhs)
    op.addInput(rhs)
    return op.execute(Int(1))
}

/// Wraps the XLA DynamicSlice operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#dynamicslice
/// .
///
/// DynamicSlice extracts a sub-array from the input array at dynamic
/// start_indices. The size of the slice in each dimension is passed in
/// size_indices, which specify the end point of exclusive slice intervals in each
/// dimension -- [start, start + size). The shape of start_indices must have rank 1,
/// with dimension size equal to the rank of operand.
///
/// - Parameters:
///     - input: A `Tensor` of type T.
///     - start_indices: List of N integers containing the slice size for each
///         dimension. Each value must be strictly greater than zero, and start + size
///         must be less than or equal to the size of the dimension to avoid
///         implementation defined behavior.
@inlinable @inline(__always)
public static func xlaDynamicSlice<
    T: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    _ input: Tensor<T>,
    startIndices: Tensor<Tindices>,
    sizeIndices: Tensor<Tindices>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("XlaDynamicSlice", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(input)
    op.addInput(startIndices)
    op.addInput(sizeIndices)
    return op.execute(Int(1))
}

/// Wraps the XLA DynamicUpdateSlice operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#dynamicupdateslice
/// .
///
/// XlaDynamicUpdateSlice generates a result which is the value of the `input`
/// operand, with a slice update overwritten at `indices`. The shape of `update`
/// determines the shape of the sub-array of the result which is updated. The shape
/// of indices must be rank == 1, with dimension size equal to the rank of `input`.
///
/// Handling of out-of-bounds slice indices is implementation-defined.
///
/// - Parameters:
///     - input: A `Tensor` of type T.
///     - update: A `Tensor` of type T. Same rank as `input`.
///     - indices: A vector of indices into `input`. Must have length equal to the rank of
///         `input`.
///
/// - Output output: A `Tensor` of type T.
@inlinable @inline(__always)
public static func xlaDynamicUpdateSlice<
    T: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    _ input: Tensor<T>,
    update: Tensor<T>,
    indices: Tensor<Tindices>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("XlaDynamicUpdateSlice", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(input)
    op.addInput(update)
    op.addInput(indices)
    return op.execute(Int(1))
}

/// An op which supports basic einsum op with 2 inputs and 1 output.
///
/// This op has better TPU performnce since it doesn't have explicitly reshape and
/// transpose operations as tf.einsum does.
@inlinable @inline(__always)
public static func xlaEinsum<T: FloatingPoint & TensorFlowScalar>(
    _ a: Tensor<T>,
    _ b: Tensor<T>,
    equation: String
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("XlaEinsum", nOutputs)
    op.updateAttribute("equation", equation)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    op.addInput(b)
    return op.execute(Int(1))
}

/// output = cond ? then_branch(inputs) : else_branch(inputs).
///
/// - Parameters:
///     - cond: A boolean scalar.
///     - inputs: A list of input tensors.
///
/// - Attrs:
///     - then_branch: A function takes 'inputs' and returns a list of tensors,
///         whose types are the same as what else_branch returns.
///     - else_branch: A function takes 'inputs' and returns a list of tensors.
///         whose types are the same as what then_branch returns.
///
/// - Output output: A list of tensors returned by either then_branch(inputs) or
///     else_branch(inputs). The input shapes of the then_branch and
///     else_branch must match.
@inlinable @inline(__always)
public static func xlaIf<
    Tcond: TensorFlowScalar,
    ThenbranchIn: TensorGroup,
    ThenbranchOut: TensorGroup,
    ElsebranchIn: TensorGroup,
    ElsebranchOut: TensorGroup,
    Tin: TensorArrayProtocol,
    Tout: TensorGroup
>(
    cond: Tensor<Tcond>,
    inputs: Tin,
    thenBranch: (ThenbranchIn) -> ThenbranchOut,
    elseBranch: (ElsebranchIn) -> ElsebranchOut
) -> Tout {
  let nOutputs = Int(Tout._typeList.count)
    let op = makeOp("XlaIf", nOutputs)
    op.updateAttribute("Tcond", Tcond.tensorFlowDataType)
    op.updateAttribute("then_branch", thenBranch)
    op.updateAttribute("else_branch", elseBranch)
    op.updateAttribute("Tin", inputs._typeList)
    op.updateAttribute("Tout", Tout._typeList)
    op.addInput(cond)
    op.addInputList(inputs)
    return op.execute(Int(Tout._typeList.count))
}

/// Wraps the XLA Sort operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#sort
/// .
///
/// Sorts a tensor. Currently only sorts in ascending order are supported.
///
/// - Parameters:
///     - keys: A `Tensor` of type K.
///     - values: A `Tensor` of type V.
///
/// - Outputs:
///     - sorted_keys: A `Tensor` of type K.
///     - sorted_values: A `Tensor` of type V.
@inlinable @inline(__always)
public static func xlaKeyValueSort<
    K: TensorFlowNumeric,
    V: TensorFlowScalar
>(
    keys: Tensor<K>,
    _ values: Tensor<V>
) -> (sortedKeys: Tensor<K>, sortedValues: Tensor<V>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("XlaKeyValueSort", nOutputs)
    op.updateAttribute("K", K.tensorFlowDataType)
    op.updateAttribute("V", V.tensorFlowDataType)
    op.addInput(keys)
    op.addInput(values)
    return op.execute(Int(1), Int(1))
}

/// XLA Launch Op. For use by the XLA JIT only.
@inlinable @inline(__always)
public static func xlaLaunch<
    Tconstants: TensorArrayProtocol,
    Targs: TensorArrayProtocol,
    Tresults: TensorGroup,
    FunctionIn: TensorGroup,
    FunctionOut: TensorGroup
>(
    constants: Tconstants,
    args: Targs,
    resources: [ResourceHandle],
    function: (FunctionIn) -> FunctionOut
) -> Tresults {
  let nOutputs = Int(Tresults._typeList.count)
    let op = makeOp("XlaLaunch", nOutputs)
    op.updateAttribute("Tconstants", constants._typeList)
    op.updateAttribute("Targs", args._typeList)
    op.updateAttribute("Nresources", resources.count)
    op.updateAttribute("Tresults", Tresults._typeList)
    op.updateAttribute("function", function)
    op.addInputList(constants)
    op.addInputList(args)
    op.addInputList(resources)
    return op.execute(Int(Tresults._typeList.count))
}

/// Wraps the XLA Pad operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#pad
/// .
///
/// - Parameters:
///     - input: A `Tensor` of type T.
///     - padding_value: A scalar `Tensor` of type T.
///     - padding_low: the padding to apply at the start of each input dimensions
///     - padding_high: the padding to apply at the end of each input dimension.
///     - padding_interior: the padding to apply between each input element.
///
/// - Output output: A `Tensor` of type T.
@inlinable @inline(__always)
public static func xlaPad<
    T: TensorFlowScalar,
    Tindices: TensorFlowIndex
>(
    _ input: Tensor<T>,
    paddingValue: Tensor<T>,
    paddingLow: Tensor<Tindices>,
    paddingHigh: Tensor<Tindices>,
    paddingInterior: Tensor<Tindices>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("XlaPad", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.addInput(input)
    op.addInput(paddingValue)
    op.addInput(paddingLow)
    op.addInput(paddingHigh)
    op.addInput(paddingInterior)
    return op.execute(Int(1))
}

/// Receives the named tensor from another XLA computation. Wraps the XLA Recv
///
/// operator documented at
///  https://www.tensorflow.org/performance/xla/operation_semantics#recv .
///
/// - Attrs:
///     - dtype: The type of the tensor.
///     - tensor_name: A string key that identifies the channel.
///     - shape: The shape of the tensor.
///
/// - Output tensor: The tensor to receive.
@inlinable @inline(__always)
public static func xlaRecv<Dtype: TensorFlowScalar>(
    tensorName: String,
    shape: TensorShape?
) -> Tensor<Dtype> {
  let nOutputs = Int(1)
    let op = makeOp("XlaRecv", nOutputs)
    op.updateAttribute("dtype", Dtype.tensorFlowDataType)
    op.updateAttribute("tensor_name", tensorName)
    op.updateAttribute("shape", shape)
    return op.execute(Int(1))
}

/// Wraps the XLA Reduce operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#reduce .
///
/// - Parameters:
///     - input: the input tensor
///     - init_value: a scalar representing the initial value for the reduction
///
/// - Attrs:
///     - dimensions_to_reduce: dimension numbers over which to reduce
///     - reducer: a reducer function to apply
@inlinable @inline(__always)
public static func xlaReduce<
    T: TensorFlowNumeric,
    ReducerIn: TensorGroup,
    ReducerOut: TensorGroup
>(
    _ input: Tensor<T>,
    initValue: Tensor<T>,
    dimensionsToReduce: [Int32],
    reducer: (ReducerIn) -> ReducerOut
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("XlaReduce", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("dimensions_to_reduce", dimensionsToReduce)
    op.updateAttribute("reducer", reducer)
    op.addInput(input)
    op.addInput(initValue)
    return op.execute(Int(1))
}

/// Wraps the XLA ReduceWindow operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#reducewindow .
///
/// - Parameters:
///     - input: the input tensor
///     - init_value: a scalar representing the initial value for the reduction
///     - window_dimensions: the shape of the window
///     - window_strides: the inter-window strides
///     - padding: the padding to apply at the start and end of each input dimensions
///
/// - Attr computation: a reducer function to apply
@inlinable @inline(__always)
public static func xlaReduceWindow<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex,
    ComputationIn: TensorGroup,
    ComputationOut: TensorGroup
>(
    _ input: Tensor<T>,
    initValue: Tensor<T>,
    windowDimensions: Tensor<Tindices>,
    windowStrides: Tensor<Tindices>,
    baseDilations: Tensor<Tindices>,
    windowDilations: Tensor<Tindices>,
    padding: Tensor<Tindices>,
    computation: (ComputationIn) -> ComputationOut
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("XlaReduceWindow", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("computation", computation)
    op.addInput(input)
    op.addInput(initValue)
    op.addInput(windowDimensions)
    op.addInput(windowStrides)
    op.addInput(baseDilations)
    op.addInput(windowDilations)
    op.addInput(padding)
    return op.execute(Int(1))
}

/// Replica ID.
@inlinable @inline(__always)
public static func xlaReplicaId(
) -> Tensor<Int32> {
  let nOutputs = Int(1)
    let op = makeOp("XlaReplicaId", nOutputs)
    
    return op.execute(Int(1))
}

/// Wraps the XLA SelectAndScatter operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
/// .
///
/// - Parameters:
///     - operand: the input tensor
///     - window_dimensions: the shape of the window
///     - window_strides: the inter-window strides
///     - padding: the padding to apply at the start and end of each input dimensions
///     - source: a tensor of values to scatter
///     - init_value: a scalar representing the initial value for the output tensor
///
/// - Attrs:
///     - select: a selection function to apply
///     - scatter: a scatter function to apply
@inlinable @inline(__always)
public static func xlaSelectAndScatter<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex,
    SelectIn: TensorGroup,
    SelectOut: TensorGroup,
    ScatterIn: TensorGroup,
    ScatterOut: TensorGroup
>(
    operand: Tensor<T>,
    windowDimensions: Tensor<Tindices>,
    windowStrides: Tensor<Tindices>,
    padding: Tensor<Tindices>,
    source: Tensor<T>,
    initValue: Tensor<T>,
    select: (SelectIn) -> SelectOut,
    scatter: (ScatterIn) -> ScatterOut
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("XlaSelectAndScatter", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("Tindices", Tindices.tensorFlowDataType)
    op.updateAttribute("select", select)
    op.updateAttribute("scatter", scatter)
    op.addInput(operand)
    op.addInput(windowDimensions)
    op.addInput(windowStrides)
    op.addInput(padding)
    op.addInput(source)
    op.addInput(initValue)
    return op.execute(Int(1))
}

/// Computes the eigen decomposition of a batch of self-adjoint matrices
///
/// (Note: Only real inputs are supported).
///
/// Computes the eigenvalues and eigenvectors of the innermost N-by-N matrices in
/// tensor such that tensor[...,:,:] * v[..., :,i] = e[..., i] * v[...,:,i], for
/// i=0...N-1.
///
/// - Parameter a: the input tensor.
///
/// - Attrs:
///     - lower: a boolean specifies whether the calculation is done with the lower
///         triangular part or the upper triangular part.
///     - max_iter: maximum number of sweep update, i.e., the whole lower triangular
///         part or upper triangular part based on parameter lower. Heuristically, it has
///         been argued that approximatly logN sweeps are needed in practice (Ref: Golub &
///         van Loan "Matrix Computation").
///     - epsilon: the tolerance ratio.
///
/// - Outputs:
///     - w: The eigenvalues in ascending order, each repeated according to its
///         multiplicity.
///     - v: The column v[..., :, i] is the normalized eigenvector corresponding to the
///         eigenvalue w[..., i].
@inlinable @inline(__always)
public static func xlaSelfAdjointEig<T: TensorFlowNumeric>(
    _ a: Tensor<T>,
    lower: Bool,
    maxIter: Int64,
    epsilon: Double
) -> (w: Tensor<T>, v: Tensor<T>) {
  let nOutputs = Int(1) + Int(1)
    let op = makeOp("XlaSelfAdjointEig", nOutputs)
    op.updateAttribute("lower", lower)
    op.updateAttribute("max_iter", maxIter)
    op.updateAttribute("epsilon", epsilon)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    return op.execute(Int(1), Int(1))
}

/// Sends the named tensor to another XLA computation. Wraps the XLA Send operator
///
/// documented at
///  https://www.tensorflow.org/performance/xla/operation_semantics#send .
///
/// - Parameter tensor: The tensor to send.
///
/// - Attr tensor_name: A string key that identifies the channel.
@inlinable @inline(__always)
public static func xlaSend<T: TensorFlowScalar>(
    _ tensor: Tensor<T>,
    tensorName: String
) {
  let nOutputs = 0
    let op = makeOp("XlaSend", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.updateAttribute("tensor_name", tensorName)
    op.addInput(tensor)
    op.execute()
}

/// An op which shards the input based on the given sharding attribute.
@inlinable @inline(__always)
public static func xlaSharding<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("XlaSharding", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

/// Wraps the XLA Sort operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#sort
/// .
///
/// Sorts a tensor. Currently only sorts in ascending order are supported.
///
/// - Parameter input: A `Tensor` of type T.
///
/// - Output output: A `Tensor` of type T.
@inlinable @inline(__always)
public static func xlaSort<T: TensorFlowScalar>(
    _ input: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("XlaSort", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(input)
    return op.execute(Int(1))
}

/// Computes the eigen decomposition of a batch of self-adjoint matrices
///
/// (Note: Only real inputs are supported).
///
/// Computes the eigenvalues and eigenvectors of the innermost M-by-N matrices in
/// tensor such that tensor[...,:,:] = u[..., :, :] * Diag(s[..., :]) * Transpose(v[...,:,:]).
///
/// - Parameter a: the input tensor.
///
/// - Attrs:
///     - max_iter: maximum number of sweep update, i.e., the whole lower triangular
///         part or upper triangular part based on parameter lower. Heuristically, it has
///         been argued that approximatly log(min (M, N)) sweeps are needed in practice
///         (Ref: Golub & van Loan "Matrix Computation").
///     - epsilon: the tolerance ratio.
///     - precision_config: a serialized xla::PrecisionConfig proto.
///
/// - Outputs:
///     - s: Singular values. The values are sorted in reverse order of magnitude, so
///         s[..., 0] is the largest value, s[..., 1] is the second largest, etc.
///     - u: Left singular vectors.
///     - v: Right singular vectors.
@inlinable @inline(__always)
public static func xlaSvd<T: TensorFlowNumeric>(
    _ a: Tensor<T>,
    maxIter: Int64,
    epsilon: Double,
    precisionConfig: String
) -> (s: Tensor<T>, u: Tensor<T>, v: Tensor<T>) {
  let nOutputs = Int(1) + Int(1) + Int(1)
    let op = makeOp("XlaSvd", nOutputs)
    op.updateAttribute("max_iter", maxIter)
    op.updateAttribute("epsilon", epsilon)
    op.updateAttribute("precision_config", precisionConfig)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(a)
    return op.execute(Int(1), Int(1), Int(1))
}

/// output = input; While (Cond(output)) { output = Body(output) }
///
/// - Parameter input: A list of input tensors whose types are T.
///
/// - Attrs:
///     - cond: A function takes 'input' and returns a tensor.  If the tensor is
///         a scalar of non-boolean, the scalar is converted to a boolean
///         according to the following rule: if the scalar is a numerical
///         value, non-zero means True and zero means False; if the scalar is
///         a string, non-empty means True and empty means False. If the
///         tensor is not a scalar, non-emptiness means True and False
///         otherwise.
///     - body: A function that takes a list of tensors and returns another
///         list of tensors. Both lists have the same types as specified by T.
///
/// - Output output: A list of output tensors whose types are T.
@inlinable @inline(__always)
public static func xlaWhile<
    T: TensorArrayProtocol,
    CondIn: TensorGroup,
    CondOut: TensorGroup,
    BodyIn: TensorGroup,
    BodyOut: TensorGroup
>(
    _ input: T,
    cond: (CondIn) -> CondOut,
    body: (BodyIn) -> BodyOut
) -> T {
  let nOutputs = Int(input._typeList.count)
    let op = makeOp("XlaWhile", nOutputs)
    op.updateAttribute("T", input._typeList)
    op.updateAttribute("cond", cond)
    op.updateAttribute("body", body)
    op.addInputList(input)
    return op.execute(Int(input._typeList.count))
}

@inlinable @inline(__always)
public static func xlogy<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Xlogy", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(y)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func zerosLike<T: TensorFlowScalar>(
    _ x: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("ZerosLike", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func zeta<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>,
    q: Tensor<T>
) -> Tensor<T> {
  let nOutputs = Int(1)
    let op = makeOp("Zeta", nOutputs)
    op.updateAttribute("T", T.tensorFlowDataType)
    op.addInput(x)
    op.addInput(q)
    return op.execute(Int(1))
}

@inlinable @inline(__always)
public static func zipDataset(
    inputDatasets: [VariantHandle],
    outputTypes: [TensorDataType],
    outputShapes: [TensorShape?]
) -> VariantHandle {
  let nOutputs = Int(1)
    let op = makeOp("ZipDataset", nOutputs)
    op.updateAttribute("output_types", outputTypes)
    op.updateAttribute("output_shapes", outputShapes)
    op.updateAttribute("N", inputDatasets.count)
    op.addInputList(inputDatasets)
    return op.execute(Int(1))
}

}
