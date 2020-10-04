# Computes expected results for Bidirectional GRU layers in `Tests/TensorFlowTests/LayerTests.swift`.
# Requires 'tensorflow>=2.0.0a0' (e.g. "pip install tensorflow==2.2.0").

import sys
import numpy
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--input-dim", default=3)
parser.add_argument("--input-length", default=4)
parser.add_argument("--units", default=4)
parser.add_argument("--merge-mode", default="concat")
args = parser.parse_args()

# Initialize the keras model with the Bidirectional GRU.
forward = tf.keras.layers.GRU(
    input_dim=args.input_dim, units=args.units,
    activation='tanh', recurrent_activation="sigmoid",
    return_sequences=True, return_state=True)
backward = tf.keras.layers.GRU(
    input_dim=args.input_dim, units=args.units,
    activation='tanh', recurrent_activation="sigmoid",
    return_sequences=True, return_state=True, 
    go_backwards=True)
bidirectional = tf.keras.layers.Bidirectional(
    forward,
    backward_layer=backward,
    merge_mode=args.merge_mode
)

x_input = tf.keras.Input(shape=[args.input_length, args.input_dim])

initial_state_forward = tf.keras.Input(shape=[args.units])
initial_state_backward = tf.keras.Input(shape=[args.units])
initial_state_input = [initial_state_forward, initial_state_backward]

output = bidirectional(x_input, initial_state=initial_state_input)
model = tf.keras.Model(inputs=[x_input, initial_state_input], outputs=[output])

[kernel_forward, recurrent_kernel_forward, bias_forward,
 kernel_backward, recurrent_kernel_backward, bias_backward] = bidirectional.get_weights()

# Assign individual kernels to variables
update_kernel_forward = kernel_forward[:, :args.units]
update_recurrent_kernel_forward = recurrent_kernel_forward[:, :args.units]
reset_kernel_forward = kernel_forward[:, args.units: args.units * 2]
reset_recurrent_kernel_forward = recurrent_kernel_forward[:, args.units: args.units * 2]
new_kernel_forward = kernel_forward[:, args.units * 2:]
new_recurrent_kernel_forward = recurrent_kernel_forward[:, args.units * 2:]
update_bias_forward = bias_forward[0][:args.units]
update_recurrent_bias_forward = bias_forward[1][:args.units]
reset_bias_forward = bias_forward[0][args.units: args.units * 2]
reset_recurrent_bias_forward = bias_forward[1][args.units: args.units * 2]
new_bias_forward = bias_forward[0][args.units * 2:]
new_recurrent_bias_forward = bias_forward[1][args.units * 2:]

update_kernel_backward = kernel_backward[:, :args.units]
update_recurrent_kernel_backward = recurrent_kernel_backward[:, :args.units]
reset_kernel_backward = kernel_backward[:, args.units: args.units * 2]
reset_recurrent_kernel_backward = recurrent_kernel_backward[:, args.units: args.units * 2]
new_kernel_backward = kernel_backward[:, args.units * 2:]
new_recurrent_kernel_backward = recurrent_kernel_backward[:, args.units * 2:]
update_bias_backward = bias_backward[0][:args.units]
update_recurrent_bias_backward = bias_backward[1][:args.units]
reset_bias_backward = bias_backward[0][args.units: args.units * 2]
reset_recurrent_bias_backward = bias_backward[1][args.units: args.units * 2]
new_bias_backward = bias_backward[0][args.units * 2:]
new_recurrent_bias_backward = bias_backward[1][args.units * 2:]

# Print the BidirectionalGRU weights.
print(swift_tensor('updateKernelForward', update_kernel_forward))
print(swift_tensor('resetKernelForward', reset_kernel_forward))
print(swift_tensor('outputKernelForward', new_kernel_forward))
print(swift_tensor('updateRecurrentKernelForward', update_recurrent_kernel_forward))
print(swift_tensor('resetRecurrentKernelForward', reset_recurrent_kernel_forward))
print(swift_tensor('outputRecurrentKernelForward', new_recurrent_kernel_forward))
print(swift_tensor('updateBiasForward', update_bias_forward))
print(swift_tensor('resetBiasForward', reset_bias_forward))
print(swift_tensor('outputBiasForward', new_bias_forward))
print(swift_tensor('updateRecurrentBiasForward', update_recurrent_bias_forward))
print(swift_tensor('resetRecurrentBiasForward', reset_recurrent_bias_forward))
print(swift_tensor('outputRecurrentBiasForward', new_recurrent_bias_forward))

print(swift_tensor('updateKernelBackward', update_kernel_backward))
print(swift_tensor('resetKernelBackward', reset_kernel_backward))
print(swift_tensor('outputKernelBackward', new_kernel_backward))
print(swift_tensor('updateRecurrentKernelBackward', update_recurrent_kernel_backward))
print(swift_tensor('resetRecurrentKernelBackward', reset_recurrent_kernel_backward))
print(swift_tensor('outputRecurrentKernelBackward', new_recurrent_kernel_backward))
print(swift_tensor('updateBiasBackward', update_bias_backward))
print(swift_tensor('resetBiasBackward', reset_bias_backward))
print(swift_tensor('outputBiasBackward', new_bias_backward))
print(swift_tensor('updateRecurrentBiasBackward', update_recurrent_bias_backward))
print(swift_tensor('resetRecurrentBiasBackward', reset_recurrent_bias_backward))
print(swift_tensor('outputRecurrentBiasBackward', new_recurrent_bias_backward))

# Initialize input data and print it.
x = tf.keras.initializers.GlorotUniform()(shape=[1, args.input_length, args.input_dim])
initial_state = [
    tf.keras.initializers.GlorotUniform()(shape=[1, args.units]),
    tf.keras.initializers.GlorotUniform()(shape=[1, args.units]),
]
print(swift_tensor('x', x))
print(swift_tensor('initialForwardLayerState', initial_state[0]))
print(swift_tensor('initialBackwardLayerState', initial_state[1]))

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

grad_update_kernel_forward = grad_kernel_forward[:, :args.units]
grad_update_recurrent_kernel_forward = grad_recurrent_kernel_forward[:, :args.units]
grad_reset_kernel_forward = grad_kernel_forward[:, args.units: args.units * 2]
grad_reset_recurrent_kernel_forward = grad_recurrent_kernel_forward[:, args.units: args.units * 2]
grad_new_kernel_forward = grad_kernel_forward[:, args.units * 2:]
grad_new_recurrent_kernel_forward = grad_recurrent_kernel_forward[:, args.units * 2:]
grad_update_bias_forward = grad_bias_forward[0][:args.units]
grad_update_recurrent_bias_forward = grad_bias_forward[1][:args.units]
grad_reset_bias_forward = grad_bias_forward[0][args.units: args.units * 2]
grad_reset_recurrent_bias_forward = grad_bias_forward[1][args.units: args.units * 2]
grad_new_bias_forward = grad_bias_forward[0][args.units * 2:]
grad_new_recurrent_bias_forward = grad_bias_forward[1][args.units * 2:]

grad_update_kernel_backward = grad_kernel_backward[:, :args.units]
grad_update_recurrent_kernel_backward = grad_recurrent_kernel_backward[:, :args.units]
grad_reset_kernel_backward = grad_kernel_backward[:, args.units: args.units * 2]
grad_reset_recurrent_kernel_backward = grad_recurrent_kernel_backward[:, args.units: args.units * 2]
grad_new_kernel_backward = grad_kernel_backward[:, args.units * 2:]
grad_new_recurrent_kernel_backward = grad_recurrent_kernel_backward[:, args.units * 2:]
grad_update_bias_backward = grad_bias_backward[0][:args.units]
grad_update_recurrent_bias_backward = grad_bias_backward[1][:args.units]
grad_reset_bias_backward = grad_bias_backward[0][args.units: args.units * 2]
grad_reset_recurrent_bias_backward = grad_bias_backward[1][args.units: args.units * 2]
grad_new_bias_backward = grad_bias_backward[0][args.units * 2:]
grad_new_recurrent_bias_backward = grad_bias_backward[1][args.units * 2:]

print(swift_tensor('expectedSum', sum_output))
print(swift_tensor('expectedStates', states))
print(swift_tensor('expectedFinalStateForward', final_state_forward))
print(swift_tensor('expectedFinalStateBackward', final_state_backward))
print(swift_tensor('expectedGradX', grad_x))
print(swift_tensor('expectedGradInitialStateForward', grad_initial_state_forward))
print(swift_tensor('expectedGradUpdateKernelForward', grad_update_kernel_forward))
print(swift_tensor('expectedGradResetKernelForward', grad_reset_kernel_forward))
print(swift_tensor('expectedGradOutputKernelForward', grad_new_kernel_forward))
print(swift_tensor('expectedGradUpdateRecurrentKernelForward', grad_update_recurrent_kernel_forward))
print(swift_tensor('expectedGradResetRecurrentKernelForward', grad_reset_recurrent_kernel_forward))
print(swift_tensor('expectedGradOutputRecurrentKernelForward', grad_new_recurrent_kernel_forward))
print(swift_tensor('expectedGradUpdateBiasForward', grad_update_bias_forward))
print(swift_tensor('expectedGradResetBiasForward', grad_reset_bias_forward))
print(swift_tensor('expectedGradOutputBiasForward', grad_new_bias_forward))
print(swift_tensor('expectedGradUpdateRecurrentBiasForward', grad_update_recurrent_bias_forward))
print(swift_tensor('expectedGradResetRecurrentBiasForward', grad_reset_recurrent_bias_forward))
print(swift_tensor('expectedGradOutputRecurrentBiasForward', grad_new_recurrent_bias_forward))
print(swift_tensor('expectedGradInitialStateBackward', grad_initial_state_backward))
print(swift_tensor('expectedGradUpdateKernelBackward', grad_update_kernel_backward))
print(swift_tensor('expectedGradResetKernelBackward', grad_reset_kernel_backward))
print(swift_tensor('expectedGradOutputKernelBackward', grad_new_kernel_backward))
print(swift_tensor('expectedGradUpdateRecurrentKernelBackward', grad_update_recurrent_kernel_backward))
print(swift_tensor('expectedGradResetRecurrentKernelBackward', grad_reset_recurrent_kernel_backward))
print(swift_tensor('expectedGradOutputRecurrentKernelBackward', grad_new_recurrent_kernel_backward))
print(swift_tensor('expectedGradUpdateBiasBackward', grad_update_bias_backward))
print(swift_tensor('expectedGradResetBiasBackward', grad_reset_bias_backward))
print(swift_tensor('expectedGradOutputBiasBackward', grad_new_bias_backward))
print(swift_tensor('expectedGradUpdateRecurrentBiasBackward', grad_update_recurrent_bias_backward))
print(swift_tensor('expectedGradResetRecurrentBiasBackward', grad_reset_recurrent_bias_backward))
print(swift_tensor('expectedGradOutputRecurrentBiasBackward', grad_new_recurrent_bias_backward))
