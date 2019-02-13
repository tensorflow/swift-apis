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
    var layer1 = Dense(inputSize: 4, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense(inputSize: hiddenSize, outputSize: 3, activation: {$0})
    
    @differentiable(wrt: (self, input))
    func applied(to input: Tensor<Float>) -> Tensor<Float> {
        let l1 = layer1.applied(to: input)
        let l2 = layer2.applied(to: l1)
        return layer3.applied(to: l2)
    }
}
```

#### Run a training loop

```swift
let optimizer = SGD<Model, Float>(learningRate: 0.02)
var classifier = Model()
let x: Tensor<Float> = ...
let y: Tensor<Float> = ...

for _ in 0..<1000 {
    let 𝛁model = classifier.gradient { classifier -> Tensor<Float> in
        let ŷ = classifier.applied(to: x)
        let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
        print("Loss: \(loss)")
        return loss
    }
    optimizer.update(&classifier, along: 𝛁model)
}
```

For more tutorials and models, go to [**tensorflow/swift-tutorials**](https://github.com/tensorflow/swift-tutorials) and [**tensorflow/swift-models**](https://github.com/tensorflow/swift-models).

## Development

### Requirements

* [Swift for TensorFlow toolchain](https://github.com/tensorflow/swift/blob/master/Installation.md).
* An environment that can run the Swift for TensorFlow toolchains: Linux 18.04 or macOS with Xcode 10.

### Building and testing

```
$ swift build
```
```
$ swift test
```

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
