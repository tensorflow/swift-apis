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

  // TODO(TF-507): Revert to `typealias State = Tensor<Scalar>` after SR-10697 is fixed.
  public struct State: Equatable, Differentiable, VectorProtocol, KeyPathIterable, Mergeable {
    public var value: Tensor<Scalar>

    @differentiable
    public init(_ value: Tensor<Scalar>) {
      self.value = value
    }

    @differentiable
    public static func +(lhs: Self, rhs: Self) -> Self {
      Self(lhs.value + rhs.value)
    }
  }

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
    State(Tensor(zeros: [input.shape[0], weight.shape[1]], on: input.device))
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The hidden state.
  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    let concatenatedInput = input.input.concatenated(with: input.state.value, alongAxis: 1)
    let newState = State(tanh(matmul(concatenatedInput, weight) + bias))
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

    @differentiable
    public static func +(lhs: Self, rhs: Self) -> Self {
      Self(cell: lhs.cell + rhs.cell, hidden: lhs.hidden + rhs.hidden)
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
    let batchSize = fused.shape[0]
    let hiddenSize = fused.shape[1] / 4
    let inputGate = sigmoid(
      fused.slice(
        lowerBounds: [0, 0],
        upperBounds: [batchSize, hiddenSize]))
    let updateGate = tanh(
      fused.slice(
        lowerBounds: [0, hiddenSize],
        upperBounds: [batchSize, 2 * hiddenSize]))
    let forgetGate = sigmoid(
      fused.slice(
        lowerBounds: [0, 2 * hiddenSize],
        upperBounds: [batchSize, 3 * hiddenSize]))
    let outputGate = sigmoid(
      fused.slice(
        lowerBounds: [0, 3 * hiddenSize],
        upperBounds: [batchSize, 4 * hiddenSize]))
    // TODO(SR-10697/TF-507): Replace with the following once it does not crash the compiler.
    // let fusedParts = fused.split(count: 4, alongAxis: 1)
    // let inputGate = sigmoid(fusedParts[0])
    // let updateGate = tanh(fusedParts[1])
    // let forgetGate = sigmoid(fusedParts[2])
    // let outputGate = sigmoid(fusedParts[3])

    let newCellState = input.state.cell * forgetGate + inputGate * updateGate
    let newHiddenState = tanh(newCellState) * outputGate

    let newState = State(cell: newCellState, hidden: newHiddenState)

    return Output(output: newState, state: newState)
  }
}

/// An GRU cell.
public struct GRUCell<Scalar: TensorFlowFloatingPoint>: RecurrentLayerCell {
  public var updateWeight1, updateWeight2: Tensor<Scalar>
  public var resetWeight1, resetWeight2: Tensor<Scalar>
  public var outputWeight1, outputWeight2: Tensor<Scalar>
  public var updateBias, outputBias, resetBias: Tensor<Scalar>

  @noDerivative public var stateShape: TensorShape {
    [1, updateWeight1.shape[0]]
  }

  public func zeroState(for input: Tensor<Scalar>) -> State {
    return State(hidden: Tensor(zeros: stateShape, on: input.device))
  }

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
    weightInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    let gateWeightShape = TensorShape([inputSize, 1])
    let gateBiasShape = TensorShape([hiddenSize])
    self.updateWeight1 = weightInitializer(gateWeightShape)
    self.updateWeight2 = weightInitializer(gateWeightShape)
    self.updateBias = biasInitializer(gateBiasShape)
    self.resetWeight1 = weightInitializer(gateWeightShape)
    self.resetWeight2 = weightInitializer(gateWeightShape)
    self.resetBias = biasInitializer(gateBiasShape)
    self.outputWeight1 = weightInitializer(gateWeightShape)
    self.outputWeight2 = weightInitializer(gateWeightShape)
    self.outputBias = biasInitializer(gateBiasShape)
  }

  // TODO(TF-507): Revert to `typealias State = Tensor<Scalar>` after
  // SR-10697 is fixed.
  public struct State: Equatable, Differentiable, VectorProtocol, KeyPathIterable, Mergeable {
    public var hidden: Tensor<Scalar>

    @differentiable
    public init(hidden: Tensor<Scalar>) {
      self.hidden = hidden
    }

    @differentiable
    public static func +(lhs: Self, rhs: Self) -> Self {
      Self(hidden: lhs.hidden + rhs.hidden)
    }
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The hidden state.
  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    let resetGate = sigmoid(
      matmul(input.input, resetWeight1) + matmul(input.state.hidden, resetWeight2) + resetBias
    )
    let updateGate = sigmoid(
      matmul(input.input, updateWeight1) + matmul(input.state.hidden, updateWeight2)
        + updateBias)
    let outputGate = tanh(
      matmul(input.input, outputWeight1)
        + matmul(resetGate * input.state.hidden, outputWeight2) + outputBias)
    let updateHidden = (1 - updateGate) * input.state.hidden
    let updateOutput = (1 - updateGate) * outputGate
    let newState = State(hidden: updateHidden + updateOutput)
    return Output(output: newState, state: newState)
  }
}

public struct RecurrentLayer<Cell: RecurrentLayerCell>: Layer {
  public typealias Input = [Cell.TimeStepInput]
  public typealias Output = [Cell.TimeStepOutput]

  public var cell: Cell
  @noDerivative public let backwardDirection: Bool 

  public init(_ cell: @autoclosure () -> Cell, backwardDirection: Bool = false) {
    self.cell = cell()
    self.backwardDirection = backwardDirection
  }

  @differentiable(wrt: (self, inputs, initialState))
  public func callAsFunction(
    _ inputs: [Cell.TimeStepInput],
    initialState: Cell.State
  ) -> [Cell.TimeStepOutput] {
    if inputs.isEmpty { return [Cell.TimeStepOutput]() }
    var currentHiddenState = initialState
    var timeStepOutputs: [Cell.TimeStepOutput] = []
    for index in 0 ..< inputs.count {
      let timeStepIndex = backwardDirection ? inputs.count - 1 - index : index
      let timeStepInput = inputs[timeStepIndex]
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
    for index in 0 ..< timeStepCount {
      let timestepIndex = backwardDirection ? timeStepCount - 1 - index : index
      let timestep = inputs[timestepIndex]
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
        return (.init(cell: 𝛁cell), .init(Array(backwardDirection ? reversed𝛁inputs : reversed𝛁inputs.reversed())), 𝛁state)
      }
    )
  }

  @differentiable
  public func callAsFunction(_ inputs: [Cell.TimeStepInput]) -> [Cell.TimeStepOutput] {
    let initialState = withoutDerivative(at: cell.zeroState(for: backwardDirection ? inputs.last! : inputs.first!))
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
    let initialState = withoutDerivative(at: cell.zeroState(for: backwardDirection ? inputs.last! : inputs.first!))
    return lastOutput(from: inputs, initialState: initialState)
  }
}

public protocol Mergeable: Differentiable, AdditiveArithmetic {
  @differentiable
  static func +(lhs: Self, rhs: Self) -> Self
}

@differentiable
public func sum<T: Mergeable>(
  _ first: T, 
  _ second: T
) -> T {
  first + second
}

public struct BidirectionalRecurrentLayer<Cell: RecurrentLayerCell>: Layer
where Cell.TimeStepOutput: Mergeable {
  public typealias Input = [Cell.TimeStepInput]
  public typealias Output = [Cell.TimeStepOutput]
  public typealias MergeFunction = @differentiable (Cell.TimeStepOutput, Cell.TimeStepOutput) -> Cell.TimeStepOutput

  // TODO: Runtime crash when I try to set `sum` function to property
  // @noDerivative public let merge: MergeFunction
  public var forward, backward: RecurrentLayer<Cell>

  public init(_ cell: @autoclosure () -> Cell, merge: @escaping MergeFunction = sum) {
    forward = RecurrentLayer(cell())
    backward = RecurrentLayer(cell(), backwardDirection: true)
    // TODO: Runtime crash when I try to set `sum` function to property
    // self.merge = merge
  }

  @differentiable
  public func callAsFunction(
    _ inputs: Input,
    initialStateForward: Cell.State,
    initialStateBackward: Cell.State
  ) -> Output {
    let forwardOutputs = forward(
      inputs, initialState: initialStateForward)
    let backwardOutputs = backward(
      inputs, initialState: initialStateBackward)

    var outputs = Output()

    for forwardIndex in  0 ..< withoutDerivative(at: inputs.count) {
        let backwardIndex = withoutDerivative(at: inputs.count - 1 - forwardIndex)
        // TODO: Replace with merge function
        outputs.append(sum(forwardOutputs[forwardIndex], backwardOutputs[backwardIndex]))
    }

    return outputs
  }

  @differentiable
  public func callAsFunction(_ inputs: Input) -> Output {
    let initialStateForward = withoutDerivative(
      at: forward.cell.zeroState(for: inputs.first!))
    let initialStateBackward = withoutDerivative(
      at: backward.cell.zeroState(for: inputs.last!))
    return self(
      inputs, 
      initialStateForward: initialStateForward, 
      initialStateBackward: initialStateBackward
    )
  }

  @differentiable
  public func lastOutput(
    from inputs: Input,
    initialStateForward: Cell.State,
    initialStateBackward: Cell.State
  ) -> Cell.TimeStepOutput {
    self(
      inputs, 
      initialStateForward: initialStateForward, 
      initialStateBackward: initialStateBackward
    )[withoutDerivative(at: inputs.count - 1)]
  }

  @differentiable
  public func lastOutput(from inputs: Input) -> Cell.TimeStepOutput {
    self(inputs)[withoutDerivative(at: inputs.count - 1)]
  }
}

extension RecurrentLayer: Equatable where Cell: Equatable {}

extension RecurrentLayer: AdditiveArithmetic where Cell: AdditiveArithmetic {
  public static var zero: Self {
    return Self(Cell.zero)
  }

  public static func +(lhs: Self, rhs: Self) -> Self {
    precondition(lhs.backwardDirection == rhs.backwardDirection)
    return Self(lhs.cell + rhs.cell, backwardDirection: lhs.backwardDirection)
  }

  public static func -(lhs: Self, rhs: Self) -> Self {
    precondition(lhs.backwardDirection == rhs.backwardDirection)
    return Self(lhs.cell - rhs.cell, backwardDirection: lhs.backwardDirection)
  }
}

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
