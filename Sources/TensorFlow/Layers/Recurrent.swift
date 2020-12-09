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

import _Differentiation
#if !COMPILING_TENSORFLOW_STDLIB_MODULE
  import Tensor
#endif

/// An input to a recurrent neural network.
public struct RNNCellInput<Input: Differentiable, State: Differentiable>: Differentiable {
  /// The input at the current time step.
  public var input: Input
  /// The previous state.
  public var state: State

  @differentiable
  public init(input: Input, state: State) {
    self.input = input
    self.state = state
  }
}

extension RNNCellInput: EuclideanDifferentiable
where Input: EuclideanDifferentiable, State: EuclideanDifferentiable {}

/// An output to a recurrent neural network.
public struct RNNCellOutput<Output: Differentiable, State: Differentiable>: Differentiable {
  /// The output at the current time step.
  public var output: Output
  /// The current state.
  public var state: State

  @differentiable
  public init(output: Output, state: State) {
    self.output = output
    self.state = state
  }
}

extension RNNCellOutput: EuclideanDifferentiable
where Output: EuclideanDifferentiable, State: EuclideanDifferentiable {}

/// A recurrent layer cell.
public protocol RecurrentLayerCell: Layer
where
  Input == RNNCellInput<TimeStepInput, State>,
  Output == RNNCellOutput<TimeStepOutput, State>
{
  /// The input at a time step.
  associatedtype TimeStepInput: Differentiable
  /// The output at a time step.
  associatedtype TimeStepOutput: Differentiable
  /// The state that may be preserved across time steps.
  associatedtype State: Differentiable

  /// Returns a zero-valued state with shape compatible with the provided input.
  func zeroState(for input: TimeStepInput) -> State
}

extension RecurrentLayerCell {
  /// Returns the new state obtained from applying the recurrent layer cell to the input at the
  /// current time step and the previous state.
  ///
  /// - Parameters:
  ///   - timeStepInput: The input at the current time step.
  ///   - previousState: The previous state of the recurrent layer cell.
  /// - Returns: The output.
  @differentiable
  public func callAsFunction(
    input: TimeStepInput,
    state: State
  ) -> RNNCellOutput<TimeStepOutput, State> {
    self(RNNCellInput(input: input, state: state))
  }

  @differentiable
  public func call(input: TimeStepInput, state: State) -> RNNCellOutput<TimeStepOutput, State> {
    self(RNNCellInput(input: input, state: state))
  }
}

/// A basic RNN cell.
public struct BasicRNNCell<Scalar: TensorFlowFloatingPoint>: RecurrentLayerCell {
  public var weight: Tensor<Scalar>
  public var bias: Tensor<Scalar>

  public typealias State = Tensor<Scalar>
  public typealias TimeStepInput = Tensor<Scalar>
  public typealias TimeStepOutput = State
  public typealias Input = RNNCellInput<TimeStepInput, State>
  public typealias Output = RNNCellOutput<TimeStepOutput, State>

  /// Creates a `SimpleRNNCell` with the specified input size and hidden state size.
  ///
  /// - Parameters:
  ///   - inputSize: The number of features in 2-D input tensors.
  ///   - hiddenSize: The number of features in 2-D hidden states.
  ///   - seed: The random seed for initialization. The default value is random.
  public init(inputSize: Int, hiddenSize: Int, seed: TensorFlowSeed = Context.local.randomSeed) {
    let concatenatedInputSize = inputSize + hiddenSize
    self.weight = Tensor(glorotUniform: [concatenatedInputSize, hiddenSize], seed: seed)
    self.bias = Tensor(zeros: [hiddenSize])
  }

  /// Returns a zero-valued state with shape compatible with the provided input.
  public func zeroState(for input: Tensor<Scalar>) -> State {
    Tensor(zeros: [input.shape[0], weight.shape[1]], on: input.device)
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The hidden state.
  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    let concatenatedInput = input.input.concatenated(with: input.state, alongAxis: 1)
    let newState = tanh(matmul(concatenatedInput, weight) + bias)
    return Output(output: newState, state: newState)
  }
}

/// An LSTM cell.
public struct LSTMCell<Scalar: TensorFlowFloatingPoint>: RecurrentLayerCell {
  public var fusedWeight: Tensor<Scalar>
  public var fusedBias: Tensor<Scalar>

  public var inputWeight: Tensor<Scalar> {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedWeight.slice(
      lowerBounds: [0, 0],
      upperBounds: [fusedWeight.shape[0], hiddenSize])
  }

  public var updateWeight: Tensor<Scalar> {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedWeight.slice(
      lowerBounds: [0, hiddenSize],
      upperBounds: [fusedWeight.shape[0], 2 * hiddenSize])
  }

  public var forgetWeight: Tensor<Scalar> {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedWeight.slice(
      lowerBounds: [0, 2 * hiddenSize],
      upperBounds: [fusedWeight.shape[0], 3 * hiddenSize])
  }

  public var outputWeight: Tensor<Scalar> {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedWeight.slice(
      lowerBounds: [0, 3 * hiddenSize],
      upperBounds: [fusedWeight.shape[0], 4 * hiddenSize])
  }

  public var inputBias: Tensor<Scalar> {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedBias.slice(lowerBounds: [0], upperBounds: [hiddenSize])
  }

  public var updateBias: Tensor<Scalar> {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedBias.slice(lowerBounds: [hiddenSize], upperBounds: [2 * hiddenSize])
  }

  public var forgetBias: Tensor<Scalar> {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedBias.slice(lowerBounds: [2 * hiddenSize], upperBounds: [3 * hiddenSize])
  }

  public var outputBias: Tensor<Scalar> {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedBias.slice(lowerBounds: [3 * hiddenSize], upperBounds: [4 * hiddenSize])
  }

  public typealias TimeStepInput = Tensor<Scalar>
  public typealias TimeStepOutput = State
  public typealias Input = RNNCellInput<TimeStepInput, State>
  public typealias Output = RNNCellOutput<TimeStepOutput, State>

  /// Creates a `LSTMCell` with the specified input size and hidden state size.
  ///
  /// - Parameters:
  ///   - inputSize: The number of features in 2-D input tensors.
  ///   - hiddenSize: The number of features in 2-D hidden states.
  public init(inputSize: Int, hiddenSize: Int) {
    self.fusedWeight = Tensor(glorotUniform: [inputSize + hiddenSize, 4 * hiddenSize])
    self.fusedBias = Tensor(zeros: [4 * hiddenSize])
  }

  public struct State: Equatable, Differentiable, VectorProtocol, KeyPathIterable, Mergeable {
    public var cell: Tensor<Scalar>
    public var hidden: Tensor<Scalar>

    @differentiable
    public init(cell: Tensor<Scalar>, hidden: Tensor<Scalar>) {
      self.cell = cell
      self.hidden = hidden
    }

    /// Concatenates two values.
    @differentiable
    public static func concatenate(_ lhs: Self, _ rhs: Self) -> Self {
      // TODO(TF-1005): Remove workaround for differenting concatenated.
      let concatCell = lhs.cell.concatenated(with: rhs.cell, alongAxis: -1)
      let concatHidden = lhs.hidden.concatenated(with: rhs.hidden, alongAxis: -1)
      let cell = concatCell.withDerivative { [shape = concatCell.shape] in
          if $0 == Tensor(0) { $0 = Tensor(zeros: shape) }
      }
      let hidden = concatHidden.withDerivative { [shape = concatHidden.shape] in
          if $0 == Tensor(0) { $0 = Tensor(zeros: shape) }
      }
      return Self(cell: cell, hidden: hidden)
    }

    /// Adds two values and produces their sum.
    @differentiable
    public static func sum(_ lhs: Self, _ rhs: Self) -> Self {
      Self(cell: lhs.cell + rhs.cell, hidden: lhs.hidden + rhs.hidden)
    }

    /// Averages two values.
    @differentiable
    public static func average(_ lhs: Self, _ rhs: Self) -> Self {
      Self(cell: (lhs.cell + rhs.cell) / 2, hidden: (lhs.hidden + rhs.hidden) / 2)
    }

    /// Multiplies two values.
    @differentiable
    public static func multiply(_ lhs: Self, _ rhs: Self) -> Self {
      Self(cell: lhs.cell * rhs.cell, hidden: lhs.hidden * rhs.hidden)
    }

    /// Stack two values.
    @differentiable
    public static func stack(_ lhs: Self, _ rhs: Self) -> Self {
      // TODO(TF-1005): Remove workaround for differenting stacking.
      let stackCell = Tensor(stacking: [lhs.cell, rhs.cell])
      let stackHidden = Tensor(stacking: [lhs.hidden, rhs.hidden])
      let cell = stackCell.withDerivative { [shape = stackCell.shape] in
          if $0 == Tensor(0) { $0 = Tensor(zeros: shape) }
      }
      let hidden = stackHidden.withDerivative { [shape = stackHidden.shape] in
          if $0 == Tensor(0) { $0 = Tensor(zeros: shape) }
      }
      return Self(cell: cell, hidden: hidden)
    }
  }

  /// Returns a zero-valued state with shape compatible with the provided input.
  public func zeroState(for input: Tensor<Scalar>) -> State {
    let hiddenSize = fusedWeight.shape[1] / 4
    return State(
      cell: Tensor(zeros: [input.shape[0], hiddenSize], on: input.device),
      hidden: Tensor(zeros: [input.shape[0], hiddenSize], on: input.device))
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The hidden state.
  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    let gateInput = input.input.concatenated(with: input.state.hidden, alongAxis: 1)

    let fused = matmul(gateInput, fusedWeight) + fusedBias
    let fusedParts = fused.split(count: 4, alongAxis: 1)
    let inputGate = sigmoid(fusedParts[0])
    let updateGate = tanh(fusedParts[1])
    let forgetGate = sigmoid(fusedParts[2])
    let outputGate = sigmoid(fusedParts[3])

    let newCellState = input.state.cell * forgetGate + inputGate * updateGate
    let newHiddenState = tanh(newCellState) * outputGate

    let newState = State(cell: newCellState, hidden: newHiddenState)

    return Output(output: newState, state: newState)
  }
}

/// An GRU cell.
public struct GRUCell<Scalar: TensorFlowFloatingPoint>: RecurrentLayerCell {
  public var updateKernel, updateRecurrentKernel: Tensor<Scalar>
  public var resetKernel, resetRecurrentKernel: Tensor<Scalar>
  public var outputKernel, outputRecurrentKernel: Tensor<Scalar>
  public var updateBias, updateRecurrentBias: Tensor<Scalar>
  public var resetBias, resetRecurrentBias: Tensor<Scalar>
  public var outputBias, outputRecurrentBias: Tensor<Scalar>

  @noDerivative public var stateShape: TensorShape {
    [1, updateKernel.shape[0]]
  }

  public func zeroState(for input: Tensor<Scalar>) -> State {
    return Tensor(zeros: stateShape, on: input.device)
  }

  public typealias State = Tensor<Scalar>
  public typealias TimeStepInput = Tensor<Scalar>
  public typealias TimeStepOutput = State
  public typealias Input = RNNCellInput<TimeStepInput, State>
  public typealias Output = RNNCellOutput<TimeStepOutput, State>

  /// Creates a `GRUCell` with the specified input size and hidden state size.
  ///
  /// - Parameters:
  ///   - inputSize: The number of features in 2-D input tensors.
  ///   - hiddenSize: The number of features in 2-D hidden states.
  public init(
    inputSize: Int,
    hiddenSize: Int,
    kernelInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    let gateKernelShape = TensorShape([inputSize, hiddenSize])
    let gateRecurrentKernelShape = TensorShape([hiddenSize, hiddenSize])
    let gateBiasShape = TensorShape([hiddenSize])
    self.updateKernel = kernelInitializer(gateKernelShape)
    self.updateRecurrentKernel = kernelInitializer(gateRecurrentKernelShape)
    self.updateBias = biasInitializer(gateBiasShape)
    self.updateRecurrentBias = biasInitializer(gateBiasShape)
    self.resetKernel = kernelInitializer(gateKernelShape)
    self.resetRecurrentKernel = kernelInitializer(gateRecurrentKernelShape)
    self.resetBias = biasInitializer(gateBiasShape)
    self.resetRecurrentBias = biasInitializer(gateBiasShape)
    self.outputKernel = kernelInitializer(gateKernelShape)
    self.outputRecurrentKernel = kernelInitializer(gateRecurrentKernelShape)
    self.outputBias = biasInitializer(gateBiasShape)
    self.outputRecurrentBias = biasInitializer(gateBiasShape)
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The hidden state.
  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    let updateGate = sigmoid(
      (matmul(input.input, updateKernel) + updateBias)
      + (matmul(input.state, updateRecurrentKernel) + updateRecurrentBias)
    )
    let resetGate = sigmoid(
      (matmul(input.input, resetKernel) + resetBias)
      + (matmul(input.state, resetRecurrentKernel) + resetRecurrentBias)
    )
    let outputGate = tanh(
      (matmul(input.input, outputKernel) + outputBias)
      + resetGate * (matmul(input.state, outputRecurrentKernel) + outputRecurrentBias)
    )

    let updateHidden = updateGate * input.state
    let updateOutput = (1 - updateGate) * outputGate
    let newState = State(updateHidden + updateOutput)

    return Output(output: newState, state: newState)
  }
}

public struct RecurrentLayer<Cell: RecurrentLayerCell>: Layer {
  public typealias Input = [Cell.TimeStepInput]
  public typealias Output = [Cell.TimeStepOutput]

  public var cell: Cell

  public init(_ cell: @autoclosure () -> Cell) {
    self.cell = cell()
  }

  @differentiable(wrt: (self, inputs, initialState))
  public func callAsFunction(
    _ inputs: [Cell.TimeStepInput],
    initialState: Cell.State
  ) -> [Cell.TimeStepOutput] {
    if inputs.isEmpty { return [Cell.TimeStepOutput]() }
    var currentHiddenState = initialState
    var timeStepOutputs: [Cell.TimeStepOutput] = []
    for timeStepInput in inputs {
      let output = cell(input: timeStepInput, state: currentHiddenState)
      currentHiddenState = output.state
      timeStepOutputs.append(output.output)
    }
    return timeStepOutputs
  }

  @differentiable(wrt: (self, inputs, initialState))
  public func call(
    _ inputs: [Cell.TimeStepInput],
    initialState: Cell.State
  ) -> [Cell.TimeStepOutput] {
    callAsFunction(inputs, initialState: initialState)
  }

  @usableFromInline
  @derivative(of: callAsFunction, wrt: (self, inputs, initialState))
  internal func _vjpCallAsFunction(
    _ inputs: [Cell.TimeStepInput],
    initialState: Cell.State
  ) -> (
    value: [Cell.TimeStepOutput],
    pullback: (Array<Cell.TimeStepOutput>.TangentVector)
      -> (TangentVector, Array<Cell.TimeStepInput>.TangentVector, Cell.State.TangentVector)
  ) {
    let timeStepCount = inputs.count
    var currentHiddenState = initialState
    var timeStepOutputs: [Cell.TimeStepOutput] = []
    timeStepOutputs.reserveCapacity(timeStepCount)
    var backpropagators: [Cell.Backpropagator] = []
    backpropagators.reserveCapacity(timeStepCount)
    for timestep in inputs {
      let (output, backpropagator) = cell.appliedForBackpropagation(
        to: .init(input: timestep, state: currentHiddenState))
      currentHiddenState = output.state
      timeStepOutputs.append(output.output)
      backpropagators.append(backpropagator)
    }
    return (
      timeStepOutputs,
      { 𝛁outputs in
        precondition(
          𝛁outputs.base.count == timeStepCount,
          "The number of output gradients must equal the number of time steps")
        var 𝛁cell = Cell.TangentVector.zero
        var 𝛁state = Cell.State.TangentVector.zero
        var reversed𝛁inputs: [Cell.TimeStepInput.TangentVector] = []
        reversed𝛁inputs.reserveCapacity(timeStepCount)
        for (𝛁output, backpropagator) in zip(𝛁outputs.base, backpropagators).reversed() {
          let (new𝛁cell, 𝛁input) = backpropagator(.init(output: 𝛁output, state: 𝛁state))
          𝛁cell += new𝛁cell
          𝛁state = 𝛁input.state
          reversed𝛁inputs.append(𝛁input.input)
        }
        return (.init(cell: 𝛁cell), .init(Array(reversed𝛁inputs.reversed())), 𝛁state)
      }
    )
  }

  @differentiable
  public func callAsFunction(_ inputs: [Cell.TimeStepInput]) -> [Cell.TimeStepOutput] {
    let initialState = withoutDerivative(at: cell.zeroState(for: inputs[0]))
    return self(inputs, initialState: initialState)
  }

  @differentiable(wrt: (self, inputs, initialState))
  public func lastOutput(
    from inputs: [Cell.TimeStepInput],
    initialState: Cell.State
  ) -> Cell.TimeStepOutput {
    precondition(!inputs.isEmpty, "'inputs' must be non-empty.")
    return self(inputs, initialState: initialState)[withoutDerivative(at: inputs.count - 1)]
  }

  @differentiable(wrt: (self, inputs))
  public func lastOutput(from inputs: [Cell.TimeStepInput]) -> Cell.TimeStepOutput {
    precondition(!inputs.isEmpty, "'inputs' must be non-empty.")
    let initialState = withoutDerivative(at: cell.zeroState(for: inputs[0]))
    return lastOutput(from: inputs, initialState: initialState)
  }
}

/// A type with values that support differentiable binary operations.
///
/// Used by `BidirectionalRecurrentLayer` as a generic requirement for merge functions.
public protocol Mergeable: Differentiable, AdditiveArithmetic {
  /// Concatenates two values.
  @differentiable
  static func concatenate(_ lhs: Self, _ rhs: Self) -> Self

  /// Adds two values and produces their sum.
  ///
  /// - Note: renaming `sum` to `+` results in a compiler crash when conforming `Tensor` to
  /// `Mergeable` (SR-13229).
  @differentiable
  static func sum(_ lhs: Self, _ rhs: Self) -> Self

  /// Averages two values.
  @differentiable
  static func average(_ lhs: Self, _ rhs: Self) -> Self

  /// Multiplies two values.
  @differentiable
  static func multiply(_ lhs: Self, _ rhs: Self) -> Self

  /// Stack two values.
  @differentiable
  static func stack(_ lhs: Self, _ rhs: Self) -> Self
}

extension Tensor: Mergeable where Scalar: TensorFlowFloatingPoint {
  /// Concatenates two tensors along last axis.
  @differentiable
  public static func concatenate(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    // TODO(TF-1005): Remove workaround for differenting concatenated.
    let concat = lhs.concatenated(with: rhs, alongAxis: -1)
    return concat.withDerivative { [shape = concat.shape] in
        if $0 == Tensor(0) { $0 = Tensor(zeros: shape) }
    }
  }

  /// Adds two values and produces their sum.
  @differentiable
  public static func sum(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    lhs + rhs
  }

  /// Averages two values.
  @differentiable
  public static func average(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    (lhs + rhs) / 2
  }

  /// Multiplies two values.
  @differentiable
  public static func multiply(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    lhs * rhs
  }

  /// Stack two values.
  @differentiable
  public static func stack(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    // TODO(TF-1005): Remove workaround for differenting stacking.
    let stack = Tensor(stacking: [lhs, rhs])
    return stack.withDerivative { [shape = stack.shape] in
        if $0 == Tensor(0) { $0 = Tensor(zeros: shape) }
    }
  }
}

/// Concatenates two values.
@differentiable
public func concatenate<T: Mergeable>(
  _ first: T,
  _ second: T
) -> T {
  T.concatenate(first, second)
}

/// Adds two values and produces their sum.
@differentiable
public func sum<T: Mergeable>(
  _ first: T,
  _ second: T
) -> T {
  T.sum(first, second)
}

/// Averages two values.
@differentiable
public func average<T: Mergeable>(
  _ first: T,
  _ second: T
) -> T {
  T.average(first, second)
}

/// Multiplies two values.
@differentiable
public func multiply<T: Mergeable>(
  _ first: T,
  _ second: T
) -> T {
  T.multiply(first, second)
}

/// Stack two values.
@differentiable
public func stack<T: Mergeable>(
  _ first: T,
  _ second: T
) -> T {
  T.stack(first, second)
}

public struct BidirectionalRecurrentLayer<Cell: RecurrentLayerCell>: Layer
where Cell.TimeStepOutput: Mergeable {
  public typealias Input = [Cell.TimeStepInput]
  public typealias Output = [Cell.TimeStepOutput]
  public typealias MergeFunction = @differentiable (Cell.TimeStepOutput, Cell.TimeStepOutput) -> Cell.TimeStepOutput

  /// A wrapper around a `@differentiable` merge function.
  ///
  /// - Note: this exists as a workaround for runtime crashes regarding `@differentiable`function
  ///   stored properties (TF-1122).
  private class _MergeFunction {
    var function: MergeFunction
    init(_ function: @escaping MergeFunction) {
      self.function = function
    }
  }

  @noDerivative private let _mergeFunction: _MergeFunction
  /// The forward recurrent layer.
  public var forward: RecurrentLayer<Cell>
  /// The backward recurrent layer.
  public var backward: RecurrentLayer<Cell>
  /// The differentiable function used for merging forward and backward recurrent layer outputs.
  @noDerivative public var mergeFunction: MergeFunction {
    _mergeFunction.function
  }

  /// Creates an instance from the given recurrent layer cell and merge function.
  public init(_ cell: @autoclosure () -> Cell, mergeFunction: @escaping MergeFunction = concatenate) {
    forward = RecurrentLayer(cell())
    backward = RecurrentLayer(cell())
    _mergeFunction = .init(mergeFunction)
  }

  @differentiable
  public func callAsFunction(
    _ inputs: Input,
    initialForwardLayerState: Cell.State,
    initialBackwardLayerState: Cell.State
  ) -> Output {
    let forwardOutputs = forward(
      inputs, initialState: initialForwardLayerState)
    let backwardOutputs = backward(
        inputs.differentiableReversed(), initialState: initialBackwardLayerState)
    return forwardOutputs.differentiableMerging(
      backwardOutputs.differentiableReversed(), mergeFunction: mergeFunction)
  }

  @differentiable
  public func callAsFunction(_ inputs: Input) -> Output {
    precondition(!inputs.isEmpty, "'inputs' must be non-empty.")
    let initialForwardLayerState = withoutDerivative(
      at: forward.cell.zeroState(for: inputs.first!))
    let initialBackwardLayerState = withoutDerivative(
      at: backward.cell.zeroState(for: inputs.last!))
    return self(
      inputs,
      initialForwardLayerState: initialForwardLayerState,
      initialBackwardLayerState: initialBackwardLayerState
    )
  }

  @differentiable
  public func lastOutput(
    from inputs: Input,
    initialForwardLayerState: Cell.State,
    initialBackwardLayerState: Cell.State
  ) -> Cell.TimeStepOutput {
    precondition(!inputs.isEmpty, "'inputs' must be non-empty.")
    return self(
      inputs,
      initialForwardLayerState: initialForwardLayerState,
      initialBackwardLayerState: initialBackwardLayerState
    )[withoutDerivative(at: inputs.count - 1)]
  }

  @differentiable
  public func lastOutput(from inputs: Input) -> Cell.TimeStepOutput {
    precondition(!inputs.isEmpty, "'inputs' must be non-empty.")
    return self(inputs)[withoutDerivative(at: inputs.count - 1)]
  }
}

extension RecurrentLayer: Equatable where Cell: Equatable {}
extension RecurrentLayer: AdditiveArithmetic where Cell: AdditiveArithmetic {}

public typealias BasicRNN<Scalar: TensorFlowFloatingPoint> = RecurrentLayer<BasicRNNCell<Scalar>>
public typealias LSTM<Scalar: TensorFlowFloatingPoint> = RecurrentLayer<LSTMCell<Scalar>>
public typealias GRU<Scalar: TensorFlowFloatingPoint> = RecurrentLayer<GRUCell<Scalar>>
public typealias BidirectionalBasicRNN<Scalar: TensorFlowFloatingPoint> = BidirectionalRecurrentLayer<BasicRNNCell<Scalar>>
public typealias BidirectionalLSTM<Scalar: TensorFlowFloatingPoint> = BidirectionalRecurrentLayer<LSTMCell<Scalar>>
public typealias BidirectionalGRU<Scalar: TensorFlowFloatingPoint> = BidirectionalRecurrentLayer<GRUCell<Scalar>>

// - MARK: Deprecated names

@available(*, deprecated, renamed: "RecurrentLayerCell")
public typealias RNNCell = RecurrentLayerCell

@available(*, deprecated, renamed: "RecurrentLayer")
public typealias RNN = RecurrentLayer

@available(*, deprecated, renamed: "BasicRNNCell")
public typealias SimpleRNNCell = BasicRNNCell

@available(*, deprecated, renamed: "BasicRNN")
public typealias SimpleRNN = BasicRNN

// - MARK: Workaround helpers.

fileprivate extension Array where Element: Differentiable {
  /// Returns a reversed copy of `self`.
  ///
  /// This has a custom derivative, which works around the SR-13945 segfault that you would
  /// encounter if you tried to implement this at the callsite using a for loop.
  @differentiable
  func differentiableReversed() -> Self {
    .init(self.reversed())
  }

  @derivative(of: differentiableReversed)
  func vjpDifferentiableReversed()
    -> (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    return (self.differentiableReversed(), { .init(.init($0.base.reversed())) })
  }

  /// Returns `zip(self, other).map { mergeFunction($0.0, $0.1) }`.
  ///
  /// This has a custom derivative, which works around the SR-13945 segfault that you would
  /// encounter if you tried to implement this at the callsite using a for loop.
  @differentiable
  func differentiableMerging(
    _ other: Self, mergeFunction: @differentiable (Element, Element) -> Element
  ) -> Self {
    zip(self, other).map { mergeFunction($0.0, $0.1) }
  }

  @derivative(of: differentiableMerging)
  func vjpDifferentiableMerging(
    _ other: Self, mergeFunction: @differentiable (Element, Element) -> Element
  ) -> (value: Self, pullback: (TangentVector) -> (TangentVector, TangentVector)) {
    let valuesWithPullbacks = zip(self, other).map {
      valueWithPullback(at: $0.0, $0.1, in: mergeFunction)
    }
    let pullbacks = valuesWithPullbacks.map { $0.pullback }
    return (
      valuesWithPullbacks.map { $0.value },
      { vs in
        let resultPairs = zip(vs.base, pullbacks).map { (v, pb) in
          pb(v)
        }
        return (.init(resultPairs.map { $0.0 }), .init(resultPairs.map { $0.1 }))
      }
    )
  }
}
