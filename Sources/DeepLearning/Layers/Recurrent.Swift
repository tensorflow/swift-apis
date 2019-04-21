// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#if !COMPILING_TENSORFLOW_MODULE
@_exported import TensorFlow
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

/// A recurrent neural network cell.
public protocol RNNCell: Layer where Input == RNNCellInput<TimeStepInput, State>,
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
    func call(input: TimeStepInput, state: State) -> RNNCellOutput<TimeStepOutput, State> {
        return self(RNNCellInput(input: input, state: state))
    }
}

/// A Simple RNN Cell.
public struct SimpleRNNCell<Scalar: TensorFlowFloatingPoint>: RNNCell {
    public var weight: Tensor<Scalar>
    public var bias: Tensor<Scalar>

    @noDerivative public var stateShape: TensorShape {
        return TensorShape([1, weight.shape[1]])
    }

    public var zeroState: Tensor<Scalar> {
        return Tensor(zeros: stateShape)
    }

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
    public init(inputSize: Int, hiddenSize: Int) {
        let concatenatedInputSize = inputSize + hiddenSize
        self.weight = Tensor(glorotUniform: [concatenatedInputSize, hiddenSize])
        self.bias = Tensor(zeros: [hiddenSize])
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The hidden state.
    @differentiable
    public func call(_ input: Input) -> Output {
        let concatenatedInput = input.input.concatenated(with: input.state, alongAxis: 1)
        let newState = tanh(matmul(concatenatedInput, weight) + bias)
        return Output(output: newState, state: newState)
    }
}

/// An LSTM Cell.
public struct LSTMCell<Scalar: TensorFlowFloatingPoint>: RNNCell {
    public var inputWeight, updateWeight, forgetWeight, outputWeight: Tensor<Scalar>
    public var inputBias, updateBias, forgetBias, outputBias: Tensor<Scalar>

    @noDerivative public var stateShape: TensorShape {
        return TensorShape([1, inputWeight.shape[1]])
    }

    public var zeroState: State {
        return State(cell: Tensor(zeros: stateShape), hidden: Tensor(zeros: stateShape))
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
        let concatenatedInputSize = inputSize + hiddenSize
        let gateWeightShape = TensorShape([concatenatedInputSize, hiddenSize])
        let gateBiasShape = TensorShape([hiddenSize])
        self.inputWeight = Tensor(glorotUniform: gateWeightShape)
        self.inputBias = Tensor(zeros: gateBiasShape)
        self.updateWeight = Tensor(glorotUniform: gateWeightShape)
        self.updateBias = Tensor(zeros: gateBiasShape)
        self.forgetWeight = Tensor(glorotUniform: gateWeightShape)
        self.forgetBias = Tensor(ones: gateBiasShape)
        self.outputWeight = Tensor(glorotUniform: gateWeightShape)
        self.outputBias = Tensor(zeros: gateBiasShape)
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
    /// - Parameters:
    ///   - input: The input to the layer.
    ///   - context: The contextual information for the layer application, e.g. the current learning
    ///     phase.
    /// - Returns: The hidden state.
    @differentiable
    public func call(_ input: Input) -> Output {
        let gateInput = input.input.concatenated(with: input.state.hidden, alongAxis: 1)

        let inputGate = sigmoid(matmul(gateInput, inputWeight) + inputBias)
        let updateGate = tanh(matmul(gateInput, updateWeight) + updateBias)
        let forgetGate = sigmoid(matmul(gateInput, forgetWeight) + forgetBias)
        let outputGate = sigmoid(matmul(gateInput, outputWeight) + outputBias)

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

    @differentiable(wrt: (self, input), vjp: _vjpCall(_:initialState:))
    public func call(_ input: [Cell.TimeStepInput],
                     initialState: Cell.State) -> [Cell.TimeStepOutput] {
        var currentHiddenState = initialState
        var timeStepOutputs: [Cell.TimeStepOutput] = []
        for timestep in input {
            let output = cell(input: timestep, state: currentHiddenState)
            currentHiddenState = output.state
            timeStepOutputs.append(output.output)
        }
        return timeStepOutputs
    }

    @usableFromInline
    internal func _vjpCall(
        _ inputs: [Cell.TimeStepInput], initialState: Cell.State
    ) -> ([Cell.TimeStepOutput],
          (Array<Cell.TimeStepOutput>.CotangentVector)
              -> (CotangentVector, Array<Cell.TimeStepInput>.CotangentVector)) {
        let timeStepCount = inputs.count
        var currentHiddenState = cell.zeroState
        var timeStepOutputs: [Cell.TimeStepOutput] = []
        timeStepOutputs.reserveCapacity(timeStepCount)
        var backpropagators: [Cell.Backpropagator] = []
        backpropagators.reserveCapacity(timeStepCount)
        for timestep in inputs {
            let (output, backpropagator) =
                cell.appliedForBackpropagation(to: .init(input: timestep,
                                                         state: currentHiddenState))
            currentHiddenState = output.state
            timeStepOutputs.append(output.output)
            backpropagators.append(backpropagator)
        }
        return (timeStepOutputs, { 𝛁outputs in
            precondition(𝛁outputs.base.count == timeStepCount,
                         "The number of output gradients must equal the number of time steps")
            var 𝛁cell = Cell.CotangentVector.zero
            var 𝛁state = Cell.State.CotangentVector.zero
            var reversed𝛁inputs: [Cell.TimeStepInput.CotangentVector] = []
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

    @differentiable(wrt: (self, inputs))
    public func call(_ inputs: [Cell.TimeStepInput]) -> [Cell.TimeStepOutput] {
        return self(inputs, initialState: cell.zeroState.withoutDerivative())
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
extension RNN: VectorNumeric where Cell: VectorNumeric {}

public typealias SimpleRNN<Scalar: TensorFlowFloatingPoint> = RNN<SimpleRNNCell<Scalar>>
public typealias LSTM<Scalar: TensorFlowFloatingPoint> = RNN<LSTMCell<Scalar>>
