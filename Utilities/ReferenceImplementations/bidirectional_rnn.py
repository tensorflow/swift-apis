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

# Initialize the keras model with the Bidirectional RNN.
forward = tf.keras.layers.SimpleRNN(
    units=4, activation='tanh',
    return_sequences=True, return_state=True)
backward = tf.keras.layers.SimpleRNN(
    units=4, activation='tanh',
    return_sequences=True, return_state=True,
    go_backwards=True)
bidirectional = tf.keras.layers.Bidirectional(
    forward,
    backward_layer=backward,
    merge_mode='sum'
)

x_input = tf.keras.Input(shape=[4, 4])

initial_state_forward = tf.keras.Input(shape=[4])
initial_state_backward = tf.keras.Input(shape=[4])
initial_state_input = [initial_state_forward, initial_state_backward]

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
x = tf.keras.initializers.GlorotUniform()(shape=[1, 4, 4])
initial_state = [
    tf.keras.initializers.GlorotUniform()(shape=[1, 4]),
    tf.keras.initializers.GlorotUniform()(shape=[1, 4]),
]
print(swift_tensor('x', x))
print(swift_tensor('initialStateForward', initial_state[0]))
print(swift_tensor('initialStateBackward', initial_state[1]))

# Run forwards and backwards pass and print the results.
with tf.GradientTape() as tape:
    tape.watch(x)
    tape.watch(initial_state)
    [[states, final_state_forward, final_state_backward]] = model([x, initial_state])
    sum_output = tf.reduce_sum(states[0][-1])

[grad_model, grad_x, grad_initial_state] = tape.gradient(sum_output, [model.variables, x, initial_state])
[grad_kernel_forward, grad_recurrent_kernel_forward, grad_bias_forward,
 grad_kernel_backward, grad_recurrent_kernel_backward, grad_bias_backward] = grad_model
[grad_initial_state_forward, grad_initial_state_backward] = grad_initial_state
print(swift_tensor('expectedSum', sum_output))
print(swift_tensor('expectedStates', states))
print(swift_tensor('expectedFinalStateForward', final_state_forward))
print(swift_tensor('expectedFinalStateBackward', final_state_backward))
print(swift_tensor('expectedGradKernelForward', grad_kernel_forward))
print(swift_tensor('expectedGradRecurrentKernelForward', grad_recurrent_kernel_forward))
print(swift_tensor('expectedGradBiasForward', grad_bias_forward))
print(swift_tensor('expectedGradKernelBackward', grad_kernel_backward))
print(swift_tensor('expectedGradRecurrentKernelBackward', grad_recurrent_kernel_backward))
print(swift_tensor('expectedGradBiasBackward', grad_bias_backward))
print(swift_tensor('expectedGradX', grad_x))
print(swift_tensor('expectedGradInitialStateForward', grad_initial_state_forward))
print(swift_tensor('expectedGradInitialStateBackward', grad_initial_state_backward))
