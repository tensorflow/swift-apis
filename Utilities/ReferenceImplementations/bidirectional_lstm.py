# Computes expected results for `testBidirectionalBasicRNN()` in `Tests/TensorFlowTests/LayerTests.swift`.
# Requires 'tensorflow>=2.0.0a0' (e.g. "pip install tensorflow==2.0.0b1").

import numpy
import tensorflow as tf

# Set random seed for repetable results
tf.random.set_seed(0)

def indented(s):
    return '\n'.join(['    ' + l for l in s.split('\n')])

def swift_tensor(name, tensor):
    if hasattr(tensor, 'numpy'):
        tensor = tensor.numpy()
    def format_float(x):
        formatted = numpy.format_float_positional(x, unique=True)
        if formatted[-1] == '.':
            return formatted + '0'
        return formatted
    formatter = {
        'float_kind': format_float
    }
    return 'let {} = Tensor<Float>(\n{}\n)'.format(
        name,
        indented(numpy.array2string(tensor, separator=',', formatter=formatter)))

input_dim = 3
input_length = 4
units = 4

# Initialize the keras model with the Bidirectional RNN.
forward = tf.keras.layers.LSTM(
    input_dim=input_dim, units=units, activation='tanh', 
    return_sequences=True, return_state=True)
backward = tf.keras.layers.LSTM(
    input_dim=input_dim, units=units, activation='tanh', 
    return_sequences=True, return_state=True, 
    go_backwards=True)
bidirectional = tf.keras.layers.Bidirectional(
    forward,
    backward_layer=backward,
    merge_mode='sum'
)

x_input = tf.keras.Input(shape=[input_length, input_dim])

initial_state_hidden_forward = tf.keras.Input(shape=[units])
initial_state_cell_forward = tf.keras.Input(shape=[units])
initial_state_hidden_backward = tf.keras.Input(shape=[units])
initial_state_cell_backward = tf.keras.Input(shape=[units])
initial_state_input = [
    initial_state_hidden_forward, initial_state_cell_forward, 
    initial_state_hidden_backward, initial_state_cell_backward
]

output = bidirectional(x_input, initial_state=initial_state_input)
model = tf.keras.Model(inputs=[x_input, initial_state_input], outputs=[output])

# Print the Bidirectional RNN weights.
[kernel_forward, recurrent_kernel_forward, bias_forward, 
 kernel_backward, recurrent_kernel_backward, bias_backward] = bidirectional.get_weights()
print(swift_tensor('kernelForward', kernel_forward))
print(swift_tensor('recurrentKernelForward', recurrent_kernel_forward))
print(swift_tensor('biasForward', bias_forward))
print(swift_tensor('kernelBackward', kernel_backward))
print(swift_tensor('recurrentKernelBackward', recurrent_kernel_backward))
print(swift_tensor('biasBackward', bias_backward))

# Initialize input data and print it.
x = tf.keras.initializers.GlorotUniform()(shape=[1, input_length, input_dim])
initial_state = [
    tf.keras.initializers.GlorotUniform()(shape=[1, units]),
    tf.keras.initializers.GlorotUniform()(shape=[1, units]),
    tf.keras.initializers.GlorotUniform()(shape=[1, units]),
    tf.keras.initializers.GlorotUniform()(shape=[1, units]),
]
print(swift_tensor('x', x))
print(swift_tensor('initialStateHiddenForward', initial_state[0]))
print(swift_tensor('initialStateCellForward', initial_state[1]))
print(swift_tensor('initialStateHiddenBackward', initial_state[2]))
print(swift_tensor('initialStateCellBackward', initial_state[3]))

# Run forwards and backwards pass and print the results.
with tf.GradientTape() as tape:
    tape.watch(x)
    tape.watch(initial_state)
    [[states, 
      final_state_hidden_forward, final_state_cell_forward, 
      final_state_hidden_backward, final_state_cell_backward]] = model([x, initial_state])
    sum_output = tf.reduce_sum(states[0][-1])

[grad_model, grad_x, grad_initial_state] = tape.gradient(sum_output, [model.variables, x, initial_state])
[grad_kernel_forward, grad_recurrent_kernel_forward, grad_bias_forward,
 grad_kernel_backward, grad_recurrent_kernel_backward, grad_bias_backward] = grad_model
[grad_initial_state_hidden_forward, grad_initial_state_cell_forward,
 grad_initial_state_hidden_backward, grad_initial_state_cell_backward] = grad_initial_state
print(swift_tensor('expectedSum', sum_output))
print(swift_tensor('expectedStates', states))
print(swift_tensor('expectedFinalStateHiddenForward', final_state_hidden_forward))
print(swift_tensor('expectedFinalStateCellForward', final_state_cell_forward))
print(swift_tensor('expectedFinalStateHiddenBackward', final_state_hidden_backward))
print(swift_tensor('expectedFinalStateCellBackward', final_state_cell_backward))
print(swift_tensor('expectedGradKernelForward', grad_kernel_forward))
print(swift_tensor('expectedGradRecurrentKernelForward', grad_recurrent_kernel_forward))
print(swift_tensor('expectedGradBiasForward', grad_bias_forward))
print(swift_tensor('expectedGradKernelBackward', grad_kernel_backward))
print(swift_tensor('expectedGradRecurrentKernelBackward', grad_recurrent_kernel_backward))
print(swift_tensor('expectedGradBiasBackward', grad_bias_backward))
print(swift_tensor('expectedGradX', grad_x))
print(swift_tensor('expectedGradInitialStateHiddenForward', grad_initial_state_hidden_forward))
print(swift_tensor('expectedGradInitialStateCellForward', grad_initial_state_cell_forward))
print(swift_tensor('expectedGradInitialStateHiddenBackward', grad_initial_state_hidden_backward))
print(swift_tensor('expectedGradInitialStateCellBackward', grad_initial_state_cell_backward))
