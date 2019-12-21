# Contributing guidelines

## How to become a contributor and submit your own code

### Contributor License Agreements

We'd love to accept your patches! Before we can take them, there are
just a few small guidelines to follow.

Please fill out either the individual or corporate Contributor License Agreement
(CLA).

  * If you are an individual writing original source code and you're sure you
    own the intellectual property, then you'll need to sign an [individual
    CLA](https://code.google.com/legal/individual-cla-v1.0.html).
  * If you work for a company that wants to allow you to contribute your work,
    then you'll need to sign a [corporate
    CLA](https://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and
instructions for how to sign and return it. Once we receive it, we'll be able to
accept your pull requests.

***NOTE***: Only original source code from you and other people that have signed
the CLA can be accepted into the main repository.

### Contributing code

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult [GitHub
Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

### Contribution guidelines and standards

Before sending your pull request for 
[review](https://github.com/tensorflow/swift-apis/pulls), 
make sure your changes are consistent with the guidelines.

#### Testing

*   Include unit tests when you contribute new features, as they help to a)
    prove that your code works correctly, and b) guard against future breaking
    changes to lower the maintenance cost.
*   Bug fixes also generally require unit tests, because the presence of bugs
    usually indicates insufficient test coverage.

#### License

Include a license at the top of new files.
* [License example](https://github.com/tensorflow/swift-apis/blob/master/Sources/TensorFlow/Random.swift)

#### Swift coding style

Changes should conform to:

* [Google Swift Style Guide](https://google.github.io/swift/)
* [Swift API Design Guidelines](https://swift.org/documentation/api-design-guidelines/)

With the exception that 4-space indendation be used.

#### API documentation contribution guidelines

* For APIs ported from Python TensorFlow, adopt the API documentation from [tensorflow.org/api_docs](https://www.tensorflow.org/api_docs) as a starting point.
  - If required, replace variable names from Python docs to reflect the Swift code. 
  For example: `y_true` and `y_predict` in TensorFlow 2.x API docs will become 
  `expected` and `predicted` in Swift for TensorFlow, respectively.
* When you contribute a new feature to Swift for TensorFlow, the maintenance burden is 
(by default) transferred to the Swift to TensorFlow team. This means that the benefit 
of the contribution must be compared against the cost of maintaining the 
feature.

## Community

It's a good idea to discuss any non-trivial submissions with the project
maintainers before submitting a pull request: please join the
[swift@tensorflow.org](https://groups.google.com/a/tensorflow.org/d/forum/swift)
mailing list to do this.
