// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow
@_exported import x10_optimizers_tensor_visitor_plan

/// State for a single step of a single weight inside an optimizer.
public struct OptimizerWeightStepState {
  /// Hyperparameters.
  public let globals: [Tensor<Float>]

  /// Temporary values (can only be assigned once).
  public var locals: [Tensor<Float>] = []

  /// The actual derivative of weight wrt to the loss function.
  public var grad: Tensor<Float>

  /// The weight being optimized.
  public let weight: Tensor<Float>

  /// Used for indexing into auxiliary arrays (like OptimizerState).
  var weightId: Int

  /// The final output of the optimizer. (should really only be set once).
  /// nil means that the weight will not be touched.
  /// This will be applied to the true weight at the end: `weight += step`.
  public var step: Tensor<Float>? = nil

  public subscript(_ local: LocalAccessor) -> Tensor<Float> {
    get { return locals[local.index] }
    set {
      // Can only set to the next index.
      precondition(locals.count == local.index)
      locals.append(newValue)
    }
  }

  public subscript(_ global: GlobalAccessor) -> Tensor<Float> {
    get { return globals[global.index] }
  }
}

/// Global state accessed through `StateAccessor`.
public struct OptimizerState {
  public init(_ zeros: [Tensor<Float>], stateCount: Int) {
    self.state = (0..<stateCount * zeros.count).map { zeros[$0 % zeros.count] }
    self.stride = zeros.count
  }

  public init(copying other: OptimizerState, to device: Device) {
    self.stride = other.stride
    self.state = other.state.map { Tensor<Float>(copying: $0, to: device) }
  }

  var state: [Tensor<Float>]
  var stride: Int

  public subscript(_ stateId: Int, _ weightId: Int) -> Tensor<Float> {
    get { state[stateId * stride + weightId] }
    _modify { yield &state[stateId * stride + weightId] }
  }

  public subscript(_ state: OptimizerWeightStepState, _ index: StateAccessor) -> Tensor<Float> {
    get { return self[index.index, state.weightId] }
    _modify { yield &self[index.index, state.weightId] }
  }
}

/// `[String: Float]` but elements can be accessed as though they were members.
@dynamicMemberLookup
public struct HyperparameterDictionary {
  public init() {}

  var values: [String: Float] = [String: Float]()

  public subscript(dynamicMember name: String) -> Float? {
    get { return values[name] }
    _modify { yield &values[name] }
  }

  public subscript(name: String) -> Float? {
    get { return values[name] }
    _modify { yield &values[name] }
  }
}

// TODO: Experiment with efficiently fusing these...
public typealias OptimizerCallback = (inout OptimizerWeightStepState, inout OptimizerState) -> Void

/// An optimizer that works on a single parameter group.
public struct ParameterGroupOptimizer {
  public init() {}
  public var hyperparameters = HyperparameterDictionary()
  public var globals: [(HyperparameterDictionary, Device) -> Tensor<Float>] = []
  public var localCount: Int = 0
  public var callbacks: [OptimizerCallback] = []
  public var stateCount: Int = 0
}

/// General optimizer that should be able to express multiple possible optimizations.
/// The optimizer is composed of a mapping from ParameterGroup to ParameterGroupOptimizer.
/// This optimizer also contains the number of elements working in a cross replica sum.
/// This is for efficiency to prevent multiple inefficient iterations over the gradient.
public class GeneralOptimizer<Model: EuclideanDifferentiable>: Optimizer
where
  Model.TangentVector: VectorProtocol & ElementaryFunctions & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float
{
  public typealias Model = Model
  /// The set of steps taken.
  public var step: Int = 0

  /// Used to determine the scaling factor of the cross replica sum.
  public var crossReplicaSumCount: Int? = nil

  /// global optimizer state.
  public var optimizerState: OptimizerState

  /// Current device of the model. (Used for constructing hyperparameters)
  public var device: Device

  /// An array mapping nested weight indices to parameter group optimizers?
  /// Weight i will be optimized by `parameterGroups[parameterGroupIndices[i]]`
  public var parameterGroupIndices: [Int]

  /// An array of parameter group optimizers.
  public var parameterGroups: [ParameterGroupOptimizer]

  /// The plan used to iterate over the Tensors of the model.
  let kpPlan: TensorVisitorPlan<Model.TangentVector>

  /// Overall learning rate of the optimizer.
  public var learningRate: Float {
    get { parameterGroups[0].hyperparameters.learningRate ?? 0.0 }
    set {
      for index in parameterGroups.indices {
        parameterGroups[index].hyperparameters.learningRate = newValue
      }
    }
  }

  /// Per-parameter group optimizer learning rates.
  public var learningRates: [Float] {
    get { parameterGroups.map { $0.hyperparameters.learningRate ?? 0.0 } }
    set {
      for index in parameterGroups.indices {
        parameterGroups[index].hyperparameters.learningRate = newValue[index]
      }
    }
  }

  /// Constructs an optimizer from a list of parameter group optimizers
  /// and a selector that divides the weights into different parameter groups.
  /// This is the most general constructor as there are many ways to construct
  /// this selector vector.
  public init(
    for model: __shared Model,
    _ kpPlan: TensorVisitorPlan<Model.TangentVector>,
    parameterGroupIndices: [Int],
    parameterGroups: [ParameterGroupOptimizer]
  ) {
    self.kpPlan = kpPlan
    let zerosPattern = model.differentiableVectorView
    // TODO(parkers): Be more precise here...
    let stateCount = parameterGroups.map { $0.stateCount }.max() ?? 0
    self.optimizerState = OptimizerState(
      kpPlan.allTensorKeyPaths.map { kp in
        Tensor<Float>(zerosLike: zerosPattern[keyPath: kp])
      }, stateCount: stateCount)
    self.device = zerosPattern[keyPath: kpPlan.allTensorKeyPaths[0]].device
    self.parameterGroupIndices = parameterGroupIndices
    self.parameterGroups = parameterGroups
  }

  /// Constructs an optimizer from a sequence of per-parameter group optimizers
  /// and then a final default parameter group optimizer. The `[Bool]` array
  /// is per weight and is true for the weights in that param group. The
  /// first parameterGroup will be used over subsequent ones.
  public convenience init(
    for model: __shared Model,
    _ kpPlan: TensorVisitorPlan<Model.TangentVector>,
    parameterGroups: ([Bool], ParameterGroupOptimizer)...,
    defaultOptimizer: ParameterGroupOptimizer
  ) {
    self.init(
      for: model, kpPlan,
      parameterGroupIndices: kpPlan.allTensorKeyPaths.indices.map { (index: Int) -> Int in
        for (i, pg) in parameterGroups.enumerated() {
          if pg.0[index] { return i }
        }
        return parameterGroups.count
      }, parameterGroups: parameterGroups.map { $0.1 } + [defaultOptimizer])
  }

  /// The actual optimizer step. Maps over all the tensors of the gradient
  /// and applies per-weight optimizers defined by ParameterGroupOptimizer.
  public func update(_ model: inout Model, along direction: Model.TangentVector) {
    step += 1
    let globals = parameterGroups.map { pg in
      pg.globals.map { globalInit in globalInit(pg.hyperparameters, device) }
    }
    var step = direction
    let crsScale : Double? = crossReplicaSumCount.map { 1.0 / Double($0) }
    // step plays dual-duties as an inout parameter for efficiency.
    let _ = kpPlan.mapTensors(&step, model.differentiableVectorView) {
      (step: inout Tensor<Float>, weight: Tensor<Float>, i: Int) in
      let selector = parameterGroupIndices[i]
      let paramGroup = parameterGroups[selector]
      var state = OptimizerWeightStepState(
        globals: globals[selector], grad: step, weight: weight, weightId: i)
      if let crsScale = crsScale {
        state.grad = _Raw.crossReplicaSum([state.grad], crsScale).first!
      }
      for cb in paramGroup.callbacks { cb(&state, &optimizerState) }
      step = state.step ?? Tensor<Float>(zerosLike: step)
    }
    model.move(along: step)
  }

  /// Copies the optimizer to the specified device.
  public required init(copying other: GeneralOptimizer, to device: Device) {
    step = other.step
    crossReplicaSumCount = other.crossReplicaSumCount
    kpPlan = other.kpPlan
    optimizerState = .init(copying: other.optimizerState, to: device)
    parameterGroupIndices = other.parameterGroupIndices
    parameterGroups = other.parameterGroups
    self.device = device
  }
}

/// A type-safe wrapper around an `Int` index value for optimizer local values.
public struct LocalAccessor {
  init(_ index: Int) {
    self.index = index
  }
  var index: Int
}

/// A type-safe wrapper around an `Int` index value for optimizer global values.
public struct GlobalAccessor {
  init(_ index: Int) {
    self.index = index
  }
  var index: Int
}

/// A type-safe wrapper around an `Int` index value for optimizer state values.
public struct StateAccessor {
  init(_ index: Int) {
    self.index = index
  }
  var index: Int
}

/// Builds a `ParameterGroupOptimizer`. This is used at essentially the level
/// of a single weight in the model. A mapping from parameter groups
/// selected by (`[Bool]` to ParameterGroupOptimizer) defines the final
/// optimizer.
public struct ParameterGroupOptimizerBuilder {
  public init() {}
  var result = ParameterGroupOptimizer()
  var globalValues: [Float] = []
  var globals: [String: GlobalAccessor] = [:]
  var locals: [String: LocalAccessor] = [:]
  var states: [String: StateAccessor] = [:]

  public mutating func makeParameter(_ name: String, _ value: Float) -> GlobalAccessor {
    precondition(globals[name] == nil, "Already a global parameter named \(name)")
    let index = result.globals.count
    result.hyperparameters[name] = value
    result.globals.append({ Tensor<Float>($0[name]!, on: $1) })
    globals[name] = GlobalAccessor(index)
    globalValues.append(value)
    return GlobalAccessor(index)
  }

  public subscript(_ global: GlobalAccessor) -> Float {
    globalValues[global.index]
  }

  public subscript(state name: String) -> StateAccessor {
    mutating get {
      if let res = states[name] { return res }
      let index = result.stateCount
      result.stateCount += 1
      states[name] = StateAccessor(index)
      return StateAccessor(index)
    }
  }

  public subscript(local name: String) -> LocalAccessor {
    mutating get {
      if let res = locals[name] { return res }
      let index = result.localCount
      result.localCount += 1
      locals[name] = LocalAccessor(index)
      return LocalAccessor(index)
    }
  }

  /// Appends a callback to the list of callbacks.
  public mutating func appendCallback(_ cb: @escaping OptimizerCallback) {
    result.callbacks.append(cb)
  }

  /// Returns the optimizer and clears the builder.
  public mutating func makeOptimizer() -> ParameterGroupOptimizer {
    let tmp = result
    self = ParameterGroupOptimizerBuilder()
    return tmp
  }
}
