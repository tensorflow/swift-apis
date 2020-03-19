# Computes expected results for `testLSTM()` in `Tests/TensorFlowTests/LayerTests.swift`.
# Requires 'tensorflow>=2.0.0a0' (e.g. "pip install tensorflow==2.0.0b1").

import numpy
import tensorflow as tf

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

# Initialize the keras model with the LSTM.
lstm = tf.keras.layers.LSTM(units=4, return_sequences=True, return_state=True)
x_input = tf.keras.Input(shape=[4, 4])
initial_state_hidden_input = tf.keras.Input(shape=[4])
initial_state_cell_input = tf.keras.Input(shape=[4])
initial_state_input = [initial_state_hidden_input, initial_state_cell_input]
output = lstm(x_input, initial_state=initial_state_input)
model = tf.keras.Model(inputs=[x_input, initial_state_input], outputs=[output])

# Print the LSTM weights.
[kernel, recurrent_kernel, bias] = lstm.get_weights()
print(swift_tensor('kernel', kernel))
print(swift_tensor('recurrentKernel', recurrent_kernel))
print(swift_tensor('bias', bias))

# Initialize input data and print it.
x = tf.keras.initializers.GlorotUniform()(shape=[1, 4, 4])
initial_state = [
    tf.keras.initializers.GlorotUniform()(shape=[1, 4]),
    tf.keras.initializers.GlorotUniform()(shape=[1, 4])
]
print(swift_tensor('x', x))
print(swift_tensor('initialStateHidden', initial_state[0]))
print(swift_tensor('initialStateCell', initial_state[1]))

# Run forwards and backwards pass and print the results.
with tf.GradientTape() as tape:
  tape.watch(x)
  tape.watch(initial_state)
  [[states, _, output]] = model([x, initial_state])
  sum_output = tf.reduce_sum(output)
[grad_model, grad_x, grad_initial_state] = tape.gradient(sum_output, [model.variables, x, initial_state])
[grad_kernel, grad_recurrent_kernel, grad_bias] = grad_model
[grad_initial_state_hidden, grad_initial_state_cell] = grad_initial_state
print(swift_tensor('expectedStates', states))
print(swift_tensor('expectedOutput', output))
print(swift_tensor('expectedGradKernel', grad_kernel))
print(swift_tensor('expectedGradRecurrentKernel', grad_recurrent_kernel))
print(swift_tensor('expectedGradBias', grad_bias))
print(swift_tensor('expectedGradX', grad_x))
print(swift_tensor('expectedGradInitialStateHidden', grad_initial_state_hidden))
print(swift_tensor('expectedGradInitialStateCell', grad_initial_state_cell))
