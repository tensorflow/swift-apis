# [Swift for TensorFlow](https://github.com/tensorflow/swift) APIs

Deep learning library for Swift for TensorFlow.

## Requirements

* A latest Swift for TensorFlow toolchain.

## Usage

A [Swift for TensorFlow toolchain](https://github.com/tensorflow/swift/blob/master/Installation.md)
is required to use this package. Add the following to your Swift package manifest.

```swift
packages: [
  .package(url: "https://github.com/tensorflow/swift-apis.git")
]
```

To get started, simply import `TensorFlow` in your Swift code.

```swift
import TensorFlow

// Define a model.
struct Classifier: Layer {
    var l1, l2: Dense<Float>

    func applied(to input: Tensor<Float>) -> Tensor<Float> {
        let h1 = l1.applied(to: input)
        return l2.applied(to: h1)
    }
}

var model = Classifier(...)
let optimizer = SGD<Classifier, Float>(learningRate: 0.02)
for _ in 0..<1000 {
    let (loss, 𝛁model) = model.valueWithGradient { model in
        let ŷ = model.applied(to: x)
        print("Prediction: \(ŷ)")
        return (y - ŷ).squared().mean()
    }
    print("Loss: \(loss)")
    optimizer.update(&model.allDifferentiableVariables,
                     along: 𝛁model)
}
```

## Building

```bash
swift build
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
