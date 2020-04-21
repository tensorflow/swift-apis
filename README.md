# Swift for TensorFlow Deep Learning Library

Get a taste of *protocol-oriented differentiable programming*.

This repository hosts [Swift for TensorFlow](https://github.com/tensorflow/swift)'s deep learning library, available both as a part of Swift for TensorFlow toolchains and as a Swift package. 

## Usage

This library is being [automatically integrated](https://github.com/apple/swift/tree/tensorflow#customize-tensorflow-support) in Swift for TensorFlow toolchains. You do not need to add this library as a Swift Package Manager dependency.

### Use Google Colaboratory

[**Open an empty Colaboratory now**](https://colab.research.google.com/github/tensorflow/swift/blob/master/notebooks/blank_swift.ipynb) to try out Swift, TensorFlow, differentiable programming, and deep learning.

> For detailed usage and troubleshooting, see [Usage](https://github.com/tensorflow/swift/blob/master/Usage.md) on the Swift for TensorFlow project homepage.

#### Define a model

Simply import `TensorFlow` to get the full power of TensorFlow.

```swift
import TensorFlow

let hiddenSize: Int = 10

struct Model: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: 3, activation: identity)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}
```

#### Initialize a model and an optimizer

```swift
var classifier = Model()
let optimizer = SGD(for: classifier, learningRate: 0.02)
Context.local.learningPhase = .training
// Dummy data.
let x: Tensor<Float> = Tensor(randomNormal: [100, 4])
let y: Tensor<Int32> = Tensor(randomUniform: [100])
```

#### Run a training loop

One way to define a training epoch is to use the [`gradient(at:in:)`](https://www.tensorflow.org/swift/api_docs/Functions#/s:10TensorFlow8gradient2at2in13TangentVectorQzx_AA0A0Vyq_GxXEtAA14DifferentiableRzAA0aB13FloatingPointR_r0_lF) function.

```swift
for _ in 0..<1000 {
    let 𝛁model = gradient(at: classifier) { classifier -> Tensor<Float> in
        let ŷ = classifier(x)
        let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
        print("Loss: \(loss)")
        return loss
    }
    optimizer.update(&classifier, along: 𝛁model)
}
```

Another way is to make use of methods on `Differentiable` or `Layer` that produce a backpropagation function. This allows you to compose your derivative computation with great flexibility.

```swift
for _ in 0..<1000 {
    let (ŷ, backprop) = classifier.appliedForBackpropagation(to: x)
    let (loss, 𝛁ŷ) = valueWithGradient(at: ŷ) { ŷ in softmaxCrossEntropy(logits: ŷ, labels: y) }
    print("Model output: \(ŷ), Loss: \(loss)")
    let (𝛁model, _) = backprop(𝛁ŷ)
    optimizer.update(&classifier, along: 𝛁model)
}
```

For more models, go to [**tensorflow/swift-models**](https://github.com/tensorflow/swift-models).

## Development

### Requirements

* [Swift for TensorFlow toolchain](https://github.com/tensorflow/swift/blob/master/Installation.md).
* An environment that can run the Swift for TensorFlow toolchains: Linux 18.04 or macOS with Xcode 10.

### Building and testing

#### SwiftPM

```
$ swift build
```
```
$ swift test
```

#### CMake

*Note: CMake support is experimental and under development.*

In-tree builds are not supported.  The instructions here expect CMake 3.16
or newer, although the minimum required version is 3.15.1.  Older releases
will not allow the use of the `-B` option to specific the build tree and
require that you are in the location of the build tree (and the `-B` option
and its argument are elided).

If `swiftc` is not in your `PATH`, you must specify the path to it using
`-D CMAKE_Swift_COMPILER=`.

```
cmake -G Ninja -B out -S swift-apis -D BUILD_X10=yes
cmake --build out  # Alternate spellings of this command are below.
```

> Note: `cmake --build out` can be alternatively spelled:
>  - `cd out && ninja`: This runs `ninja` inside the `out` directory.
>  - `ninja -C out`: This tells `ninja` to `cd` into `out` first and then
>    run.

## Bugs

Please report bugs and feature requests using GitHub issues in this repository.

## Community

Discussion about Swift for TensorFlow happens on the
[swift@tensorflow.org](https://groups.google.com/a/tensorflow.org/d/forum/swift)
mailing list.

## Contributing

We welcome contributions: please read the [Contributor Guide](CONTRIBUTING.md)
to get started. It's always a good idea to discuss your plans on the mailing
list before making any major submissions.

## Code of Conduct

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of
experience, education, socio-economic status, nationality, personal appearance,
race, religion, or sexual identity and orientation.

The Swift for TensorFlow community is guided by our [Code of
Conduct](CODE_OF_CONDUCT.md), which we encourage everybody to read before
participating.
