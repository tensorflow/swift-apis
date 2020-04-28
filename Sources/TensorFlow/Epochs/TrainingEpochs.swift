// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// An infinite sequence of collections of batch samples suitable for training a
/// DNN when samples are uniform.
///
/// - Parameter `Samples`: the type of collection from which samples will be
///   drawn.
/// - Parameter `Entropy`: a source of entropy used to randomize sample order in
///   each epoch.  See the `init` documentation for details.
///
/// The batches in each epoch all have exactly the same size.
public final class TrainingEpochs<
  Samples: Collection,
  Entropy: RandomNumberGenerator
>: Sequence, IteratorProtocol {
  private let samples: Samples

  /// The number of samples in a batch.
  let batchSize: Int

  /// The ordering of samples in the current epoch.
  private var sampleOrder: [Samples.Index]

  // TODO: Figure out how to handle non-threasafe PRNGs with a parallel shuffle
  // algorithm.
  /// A source of entropy for shuffling samples.
  private var entropy: Entropy

  /// Creates an instance drawing samples from `samples` into batches of size
  /// `batchSize`.
  ///
  /// - Parameter entropy: a source of randomness used to shuffle sample 
  ///   ordering.  It  will be stored in `self`, so if it is only pseudorandom 
  ///   and has value semantics, the sequence of epochs is determinstic and not 
  ///   dependent on other operations.
  public init(
    samples: Samples,
    batchSize: Int,
    entropy: Entropy
  ) {
    self.samples = samples
    self.batchSize = batchSize
    sampleOrder = Array(samples.indices)
    self.entropy = entropy
  }

  /// The type of each epoch, a collection of batches of samples.
  public typealias Element = Slices<
    Sampling<Samples, Array<Samples.Index>.SubSequence>
  >

  /// Returns the next epoch in sequence.
  public func next() -> Element? {
    let remainder = sampleOrder.count % batchSize

    // TODO: use a parallel shuffle like mergeshuffle
    // (http://ceur-ws.org/Vol-2113/paper3.pdf)
    sampleOrder.shuffle(using: &entropy)

    return samples.sampled(at: sampleOrder.dropLast(remainder))
      .inBatches(of: batchSize)
  }
}

extension TrainingEpochs where Entropy == SystemRandomNumberGenerator {
  /// Creates an instance drawing samples from `samples` into batches of size
  /// `batchSize`.
  public convenience init(
    samples: Samples,
    batchSize: Int
  ) {
    self.init(
      samples: samples, batchSize: batchSize,
      entropy: SystemRandomNumberGenerator())
  }
}
