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

/// A recurrent neural network cell.
public protocol RNNCell: Layer
    where Input == RNNCellInput<TimeStepInput, State>, 
          Output == RNNCellOutput<TimeStepOutput, State> {
    /// The input at a time step.
    associatedtype TimeStepInput: Differentiable
    /// The output at a time step.
    associatedtype TimeStepOutput: Differentiable
    /// The state that may be preserved across time steps.
    associatedtype State: Differentiable
    /// The zero state.
    var zeroState: State { get }
}

public extension RNNCell {
    /// Returns the new state obtained from applying the RNN cell to the input at the current time
    /// step and the previous state.
    ///
    /// - Parameters:
    ///   - timeStepInput: The input at the current time step.
    ///   - previousState: The previous state of the RNN cell.
    /// - Returns: The output.
    @differentiable
    func callAsFunction(
        input: TimeStepInput,
        state: State
    ) -> RNNCellOutput<TimeStepOutput, State> {
        self(RNNCellInput(input: input, state: state))
    }

    @differentiable
    func call(input: TimeStepInput, state: State) -> RNNCellOutput<TimeStepOutput, State> {
        self(RNNCellInput(input: input, state: state))
    }
}

/// A simple RNN cell.
public struct SimpleRNNCell<Scalar: TensorFlowFloatingPoint>: RNNCell {
    public var weight: Tensor<Scalar>
    public var bias: Tensor<Scalar>

    @noDerivative public var stateShape: TensorShape {
        TensorShape([1, weight.shape[1]])
    }

    public var zeroState: State {
        State(Tensor(zeros: stateShape))
    }

    // TODO(TF-507): Revert to `typealias State = Tensor<Scalar>` after SR-10697 is fixed.
    public struct State: Equatable, Differentiable, VectorProtocol, KeyPathIterable {
        public var value: Tensor<Scalar>
        public init(_ value: Tensor<Scalar>) {
            self.value = value
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
public struct LSTMCell<Scalar: TensorFlowFloatingPoint>: RNNCell {
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

    @noDerivative public var stateShape: TensorShape {
        TensorShape([1, fusedWeight.shape[1] / 4])
    }

    public var zeroState: State {
        State(cell: Tensor(zeros: stateShape), hidden: Tensor(zeros: stateShape))
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

    public struct State: Differentiable {
        public var cell: Tensor<Scalar>
        public var hidden: Tensor<Scalar>

        @differentiable
        public init(cell: Tensor<Scalar>, hidden: Tensor<Scalar>) {
            self.cell = cell
            self.hidden = hidden
        }
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
        let inputGate = sigmoid(fused.slice(
            lowerBounds: [0, 0],
            upperBounds: [batchSize, hiddenSize]))
        let updateGate = tanh(fused.slice(
            lowerBounds: [0, hiddenSize],
            upperBounds: [batchSize, 2 * hiddenSize]))
        let forgetGate = sigmoid(fused.slice(
            lowerBounds: [0, 2 * hiddenSize],
            upperBounds: [batchSize, 3 * hiddenSize]))
        let outputGate = sigmoid(fused.slice(
            lowerBounds: [0, 3 * hiddenSize],
            upperBounds: [batchSize,4 * hiddenSize]))
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

public struct RNN<Cell: RNNCell>: Layer {
    public typealias Input = [Cell.TimeStepInput]
    public typealias Output = [Cell.TimeStepOutput]

    public var cell: Cell

    public init(_ cell: @autoclosure () -> Cell) {
        self.cell = cell()
    }

    @differentiable(wrt: (self, input), vjp: _vjpCallAsFunction(_:initialState:))
    public func callAsFunction(
        _ input: [Cell.TimeStepInput],
        initialState: Cell.State
    ) -> [Cell.TimeStepOutput] {
        var currentHiddenState = initialState
        var timeStepOutputs: [Cell.TimeStepOutput] = []
        for timestep in input {
            let output = cell(input: timestep, state: currentHiddenState)
            currentHiddenState = output.state
            timeStepOutputs.append(output.output)
        }
        return timeStepOutputs
    }

    @differentiable(wrt: (self, input))
    public func call(
        _ input: [Cell.TimeStepInput],
        initialState: Cell.State
    ) -> [Cell.TimeStepOutput] {
        callAsFunction(input, initialState: initialState)
    }

    @usableFromInline
    internal func _vjpCallAsFunction(
        _ inputs: [Cell.TimeStepInput],
        initialState: Cell.State
    ) -> ([Cell.TimeStepOutput],
          (Array<Cell.TimeStepOutput>.TangentVector)
              -> (TangentVector, Array<Cell.TimeStepInput>.TangentVector)) {
        let timeStepCount = inputs.count
        var currentHiddenState = cell.zeroState
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
        return (timeStepOutputs, { 𝛁outputs in
            precondition(𝛁outputs.base.count == timeStepCount,
                         "The number of output gradients must equal the number of time steps")
            var 𝛁cell = Cell.TangentVector.zero
            var 𝛁state = Cell.State.TangentVector.zero
            var reversed𝛁inputs: [Cell.TimeStepInput.TangentVector] = []
            reversed𝛁inputs.reserveCapacity(timeStepCount)
            for (𝛁output, backpropagator) in zip(𝛁outputs.base, backpropagators).reversed() {
                let (new𝛁cell, 𝛁input) = backpropagator(.init(output: 𝛁output, state: 𝛁state))
                𝛁cell = new𝛁cell
                𝛁state = 𝛁input.state
                reversed𝛁inputs.append(𝛁input.input)
            }
            return (.init(cell: 𝛁cell), .init(Array(reversed𝛁inputs.reversed())))
        })
    }

    @differentiable
    public func callAsFunction(_ inputs: [Cell.TimeStepInput]) -> [Cell.TimeStepOutput] {
        return self(inputs, initialState: withoutDerivative(at: cell.zeroState))
    }

    /* TODO: Uncomment once control flow and differentiation through force unwrapping is supported.
    @differentiable(wrt: (self, inputs))
    public func lastOutput(from inputs: [Cell.TimeStepInput],
                           initialState: Cell.State) -> Cell.TimeStepOutput {
        precondition(!inputs.isEmpty, "inputs cannot be empty")
        return self(inputs, initialState: initialState).last!
    }

    @differentiable(wrt: (self, inputs))
    public func lastOutput(from inputs: [Cell.TimeStepInput]) -> Cell.TimeStepOutput {
        precondition(!inputs.isEmpty, "inputs cannot be empty")
        return self(inputs, initialState: cell.zeroState).last!
    }
    */
}

extension RNN: Equatable where Cell: Equatable {}
extension RNN: AdditiveArithmetic where Cell: AdditiveArithmetic {}

public typealias SimpleRNN<Scalar: TensorFlowFloatingPoint> = RNN<SimpleRNNCell<Scalar>>
public typealias LSTM<Scalar: TensorFlowFloatingPoint> = RNN<LSTMCell<Scalar>>
