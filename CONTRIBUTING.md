# Contributing guidelines

## Welcome!

`swift-apis` is a carefully curated set of maintained APIs and functionality. 
We generally aim to incubate new features in the library ecosystem that builds 
on top of Swift for TensorFlow. Popular functionality, such as new layers or 
helpful abstractions, often graduate into `swift-apis` after maturing in other 
repositories first (e.g. `swift-models`), where the development team and the 
community can try them out in context and iterate quickly.

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
* [License example](https://github.com/tensorflow/swift-apis/blob/master/Sources/TensorFlow/Layer.swift)

#### Swift coding style

Changes should conform to:

* [Google Swift Style Guide](https://google.github.io/swift/)
* [Swift API Design Guidelines](https://swift.org/documentation/api-design-guidelines/)

With the exception that 4-space indendation be used.

#### API documentation guidelines

API documentation should follow guidelines from the
["Write a documentation comment"](https://swift.org/documentation/api-design-guidelines/#write-doc-comment)
section of the Swift API Design Guidelines:

> *   **Use Swift’s
>     [dialect of Markdown](https://developer.apple.com/library/archive/documentation/Xcode/Reference/xcode_markup_formatting_ref)**.
> *   **Begin with a summary** that describes the entity being declared. Often,
>     an API can be completely understood from its declaration and its summary.
> *   **Optionally, continue** with one or more paragraphs and bullet items.
>     Paragraphs are separated by blank lines and use complete sentences.

For APIs ported from Python TensorFlow, use the API documentation from
[tensorflow.org/api_docs](https://www.tensorflow.org/api_docs) (e.g. function
parameter descriptions) as a starting point.

## Community

It's a good idea to discuss any non-trivial submissions with the project
maintainers before submitting a pull request: please join the
[swift@tensorflow.org](https://groups.google.com/a/tensorflow.org/d/forum/swift)
mailing list to do this.
