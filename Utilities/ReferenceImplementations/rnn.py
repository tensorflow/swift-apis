# Computes expected results for `testRNN()` in `Tests/TensorFlowTests/LayerTests.swift`.
# Requires 'tensorflow>=2.0.0a0' (e.g. "pip install tensorflow==2.2.0").

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

# Initialize the keras model with the SimpleRNN.
rnn = tf.keras.layers.SimpleRNN(
    units=4, activation="tanh", 
    return_sequences=True, return_state=True)

x_input = tf.keras.Input(shape=[4, 4])

initial_state = tf.keras.Input(shape=[4])
initial_state_input = [initial_state]

output = rnn(x_input, initial_state=initial_state_input)
model = tf.keras.Model(inputs=[x_input, initial_state_input], outputs=[output])

# Print the SimpleRNN weights.
[kernel, recurrent_kernel, bias] = rnn.get_weights()
print(swift_tensor('kernel', kernel))
print(swift_tensor('recurrentKernel', recurrent_kernel))
print(swift_tensor('bias', bias))

# Initialize input data and print it.
x = tf.keras.initializers.GlorotUniform()(shape=[1, 4, 4])
initial_state = [
    tf.keras.initializers.GlorotUniform()(shape=[1, 4]),
]
print(swift_tensor('x', x))
print(swift_tensor('initialState', initial_state[0]))

# Run forwards and backwards pass and print the results.
with tf.GradientTape() as tape:
    tape.watch(x)
    tape.watch(initial_state)
    [[states, final_state]] = model([x, initial_state])
    sum_output = tf.reduce_sum(states[0][-1])

[grad_model, grad_x, grad_initial_state] = tape.gradient(sum_output, [model.variables, x, initial_state])
[grad_kernel, grad_recurrent_kernel, grad_bias] = grad_model
[grad_initial_state] = grad_initial_state
print(swift_tensor('expectedSum', sum_output))
print(swift_tensor('expectedStates', states))
print(swift_tensor('expectedFinalState', final_state))
print(swift_tensor('expectedGradKernel', grad_kernel))
print(swift_tensor('expectedGradRecurrentKernel', grad_recurrent_kernel))
print(swift_tensor('expectedGradBias', grad_bias))
print(swift_tensor('expectedGradX', grad_x))
print(swift_tensor('expectedGradInitialState', grad_initial_state))
