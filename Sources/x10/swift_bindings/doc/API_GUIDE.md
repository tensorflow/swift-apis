# X10: S4TF on XLA Devices

S4TF runs on XLA devices, like TPUs, using the
[X10 tensor library][x10_lib].
This document describes how to run your models on these devices.

## Creating an X10 Tensor

X10 exposes device type and ordinals to S4TF. Device type can be TPU, CPU or
GPU. For example, here's how to create and print an X10 tensor on a CPU device:

```swift
import TensorFlow

let device = Device(kind: .CPU, ordinal: 0, backend: .XLA)
let t = Tensor(shape: [3, 2], scalars: [1, 2, 3, 4, 5, 6], on: device)
print(t.device)
print(t)
```

This snippet will output the following:

```
Device(kind: .CPU, ordinal: 0)
[[1.0, 2.0],
 [3.0, 4.0],
 [5.0, 6.0]]
```

On a machine without TPU, the following snippet will have the same behavior:

```swift
import TensorFlow

let t = Tensor(shape: [3, 2], scalars: [1, 2, 3, 4, 5, 6], on: Device.defaultXLA)
print(t.device)
print(t)
```

When the device isn't specified, the default device is picked: TPU if available,
otherwise CPU, in both cases with the ordinal 0. Note that 0 is the only valid
ordinal for CPU devices, whereas each TPU node has its own ordinal. For example,
a 4x2 TPUv3 slice will cover ordinals from 0 to 15.

This code should look familiar. X10 uses the same interfaces as regular S4TF
with a few additions which cover support for multiple TPU devices, gradient
reduction across them and ways to force evaluation of an accumulated
computational graph. More details about the latter can be found in the
[X10 Tensor Deep Dive](#x10-tensor-deep-dive) section.

## X10 Tensors are S4TF Tensors

Usual S4TF operations can be performed on X10 tensors. For example, X10 tensors
can be added together:

```swift
let t0 = Tensor(shape: [3, 2], scalars: [1, 2, 3, 4, 5, 6], on: Device.defaultXLA)
let t1 = Tensor(shape: [3, 2], scalars: [2, 3, 4, 5, 6, 7], on: Device.defaultXLA)
print(t0 + t1)
```

Matrix multiplication also works:

```swift
let t0 = Tensor(shape: [3, 2], scalars: [1, 2, 3, 4, 5, 6], on: Device.defaultXLA)
let t1 = Tensor(shape: [2, 3], scalars: [2, 3, 4, 5, 6, 7], on: Device.defaultXLA)
print(matmul(t0, t1))
```

The existing S4TF neural network layers work and behave correctly with X10 as
well.

Note that operations on X10 tensors expect all of them to be on the same device.
In other words, transfers don't happen automatically for tensors on different
devices and the following code won't work:

```swift
let tpu0 = Device(kind: .TPU, ordinal: 0, backend: .XLA)
let tpu1 = Device(kind: .TPU, ordinal: 1, backend: .XLA)
let t0 = Tensor(shape: [3, 2], scalars: [1, 2, 3, 4, 5, 6], on: tpu0)
let t1 = Tensor(shape: [3, 2], scalars: [2, 3, 4, 5, 6, 7], on: tpu1)
```

We made this choice in order to prevent unwanted, expensive transfers across
devices triggered by user errors.

## Running Models on XLA Devices

Building a new S4TF network or converting an existing one to run on XLA devices
requires only a few lines of X10-specific code. The following snippets highlight
these lines when running on a single or multiple devices.

### Running on a Single XLA Device

For training on a single device, it's sufficient to add a call to
`LazyTensorBarrier` after the optimizer update:

```swift
...
optimizer.update(&model, along: 𝛁model)
LazyTensorBarrier(on: device, devices: [])
...
```

This snippet highlights how easy it is to switch your model to run on X10. The
model definition, input pipeline, optimizer and training loop can work on any
device. The only X10-specific code is the call to `LazyTensorBarrier`, which
marks the end of a training iteration. Calling it forces the evaluation of the
computation graph and updates the model parameters. See
[X10 Tensor Deep Dive](#x10-tensor-deep-dive) for more on how XLA creates graphs
and runs operations.

### Running on Multiple XLA Devices

X10 offers support for
[copying tensors, models and optimizers][copying]
to a given device. On top of this low-level functionality, we provide a
[training loop high-level API][training]
which automates a lot of the work for image models. In a nutshell,
this helper does the following:

*   Copies the initial model weights and optimizer state to each TPU core.
*   Reads multiple minibatches from the dataset and transfers them to TPU cores
    in round-robin fashion.
*   Runs each copy of the model on each core.
*   Averages the gradients from all cores using cross replica sum.
*   Applies the averaged gradients to all the copies of the model weights.

### Running with mixed precision

Training with mixed precision is supported and we provide both low-level and
high-level API to control it. The
[low-level API][low-level]
offers two computed properties: `toReducedPrecision` and `toFullPrecision` which
convert between full and reduced precision, alongside with `isReducedPrecision`
to query the precision. Besides tensors, models and optimizers can be converted
between full and reduced precision using this API.

Note that conversion to reduced precision doesn't change the logical type of a
tensor. If `t` is a `Tensor<Float>`, `t.toReducedPrecision` is also a
`Tensor<Float>` with a reduced precision underlying representation.

As with devices, operations between tensors of different precisions are not
allowed. This avoids silent and unwanted promotion to `F32`, which would be hard
to detect by the user.

The
[training loop high-level API][training]
we've mentioned earlier also provides a flag to allow for automatic mixed
precision. In this mode, weights are kept in full precision but inputs and
activations are reduced precision, following the precedent set by other
frameworks.

## X10 Tensor Deep Dive

Using X10 tensors and devices requires changing only a few lines of code. While
we preserved the semantics of regular S4TF tensors, the implementation is very
different. This section describes what makes X10 tensors unique.

### X10 Tensors are Lazy

Regular S4TF tensors launch operations immediately (eagerly). On the other hand,
X10 tensor operations are lazily evaluated. They record operations in a graph
until the results are needed. Deferring execution like this lets XLA optimize
it, allowing training computation to be one fused graph.

Lazy execution is generally invisible to the caller. X10 automatically
constructs the graphs, sends them to X10 devices and synchronizes when copying
data between an XLA device and the CPU. Inserting the `LazyTensorBarrier` call
after the optimizer step explicitly synchronizes the CPU and the X10 device.

### Memory Layout

The internal data representation of X10 tensors is opaque to the user. This
allows XLA to control a tensor's memory layout for better performance.

[x10_lib]: https://github.com/tensorflow/swift-apis/tree/master/Sources/x10
[copying]: https://github.com/tensorflow/swift-apis/tree/master/Sources/TensorFlow/Core/CopyableToDevice.swift
[training]: https://github.com/tensorflow/swift-apis/tree/master/Sources/x10/swift_bindings/training_loop.swift
[low-level]: https://github.com/tensorflow/swift-apis/blob/master/Sources/TensorFlow/Core/MixedPrecision.swift
