// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import XCTest

@testable import TensorFlow

var rng = ARC4RandomNumberGenerator(seed: [42])

final class EpochsTests: XCTestCase {

  // An element that keeps track of when it was first accessed.
  class AccessTracker {
    var accessed: Bool = false
  }

  // A struct keeping track of when its elements have been first accessed. We
  // use it in the tests to check whether methods that are not supposed to break
  // the laziness work as intended.
  /// An adapted collection that presents the elements of `Base` but
  /// tracks whether elements have been read.
  ///
  /// - Warning: distinct elements may be read concurrently, but reading
  ///   the same element from two threads is a race condition.
  struct ReadTracker<Base: RandomAccessCollection>: RandomAccessCollection {
    let base: Base
    let accessed_: [AccessTracker]

    public typealias Element = Base.Element
    /// A type whose instances represent positions in `self`.
    public typealias Index = Base.Index
    /// The position of the first element.
    public var startIndex: Index { base.startIndex }
    /// The position one past the last element.
    public var endIndex: Index { base.endIndex }
    /// Returns the position after `i`.
    public func index(after i: Index) -> Index { base.index(after: i) }
    /// Returns the position after `i`.
    public func index(before i: Index) -> Index { base.index(before: i) }

    init(_ base: Base) {
      self.base = base
      accessed_ = (0..<base.count).map { _ in AccessTracker() }
    }

    subscript(i: Base.Index) -> Base.Element {
      accessed_[base.distance(from: base.startIndex, to: i)].accessed = true
      return base[i]
    }

    var accessed: LazyMapCollection<[AccessTracker], Bool> {
      accessed_.lazy.map(\.accessed)
    }
  }

  func testBaseUse() {
    let batchSize = 64
    let dataset = (0..<512).map { (_) -> Tensor<Float> in
      Tensor<Float>(randomNormal: [224, 224, 3])
    }
    let batches = dataset.inBatches(of: batchSize).lazy.map(\.collated)

    XCTAssertEqual(
      batches.count, dataset.count / batchSize,
      "Incorrect number of batches.")
    for batch in batches {
      XCTAssertEqual(
        batch.shape, TensorShape([64, 224, 224, 3]),
        "Wrong shape for batch: \(batch.shape), should be [64, 224, 224, 3]")
    }
  }

  func testInBatchesIsLazy() {
    let batchSize = 64
    let items = Array(0..<512)
    let dataset = ReadTracker(items)
    let batches = dataset.inBatches(of: batchSize)

    // `inBatches` is lazy so no elements were accessed.
    XCTAssert(
      dataset.accessed.allSatisfy { !$0 },
      "Laziness failure: no elements should have been accessed yet.")
    for (i, batch) in batches.enumerated() {
      // Elements are not accessed until we do something with `batch` so only
      // the elements up to `i * batchSize` have been accessed yet.
      XCTAssert(
        dataset.accessed[..<(i * batchSize)].allSatisfy { $0 },
        "Some samples in a prior batch were unexpectedly skipped.")
      XCTAssert(
        dataset.accessed[(i * batchSize)...].allSatisfy { !$0 },
        "Laziness failure: some samples were read prematurely.")
      let _ = Array(batch)
      let limit = (i + 1) * batchSize
      // We accessed elements up to `limit` but no further.
      XCTAssert(
        dataset.accessed[..<limit].allSatisfy { $0 },
        "Some samples in a prior batch were unexpectedly skipped.")
      XCTAssert(
        dataset.accessed[limit...].allSatisfy { !$0 },
        "Laziness failure: some samples were read prematurely.")
    }
  }

  func testTrainingEpochsShuffles() {
    let batchSize = 64
    let dataset = Array(0..<512)
    let epochs = TrainingEpochs(
      samples: dataset, batchSize: batchSize,
      entropy: rng
    ).prefix(10)
    var lastEpochSampleOrder: [Int]? = nil
    for batches in epochs {
      var newEpochSampleOrder: [Int] = []
      for batch in batches {
        XCTAssertEqual(batches.count, 8, "Incorrect number of batches.")
        let samples = Array(batch)
        XCTAssertEqual(
          samples.count, batchSize,
          "This batch doesn't have batchSize elements.")

        newEpochSampleOrder += samples
      }
      if let l = lastEpochSampleOrder {
        XCTAssertNotEqual(
          l, newEpochSampleOrder,
          "Dataset should have been reshuffled.")
      }

      let uniqueSamples = Set(newEpochSampleOrder)
      XCTAssertEqual(
        uniqueSamples.count, newEpochSampleOrder.count,
        "Every epoch sample should be drawn from a different input sample.")
      lastEpochSampleOrder = newEpochSampleOrder
    }
  }

  func testTrainingEpochsShapes() {
    let batchSize = 64
    let dataset = 0..<500
    let epochs = TrainingEpochs(
      samples: dataset, batchSize: batchSize,
      entropy: rng
    ).prefix(1)

    for epochBatches in epochs {
      XCTAssertEqual(epochBatches.count, 7, "Incorrect number of batches.")
      var epochSampleCount = 0
      for batch in epochBatches {
        XCTAssertEqual(
          batch.count, batchSize, "unexpected batch size: \(batch.count)")
        epochSampleCount += batch.count
      }
      let expectedDropCount = dataset.count % 64
      let actualDropCount = dataset.count - epochSampleCount
      XCTAssertEqual(
        expectedDropCount, actualDropCount,
        "Dropped \(actualDropCount) samples but expected \(expectedDropCount).")
    }
  }

  func testTrainingEpochsIsLazy() {
    let batchSize = 64
    let items = Array(0..<512)
    let dataset = ReadTracker(items)
    let epochs = TrainingEpochs(
      samples: dataset, batchSize: batchSize,
      entropy: rng
    ).prefix(1)

    // `inBatches` is lazy so no elements were accessed.
    XCTAssert(
      dataset.accessed.allSatisfy { !$0 },
      "No elements should have been accessed yet.")
    for batches in epochs {
      for (i, batch) in batches.enumerated() {
        // Elements are not accessed until we do something with `batch` so only
        // `i * batchSize` elements have been accessed yet.
        XCTAssertEqual(
          dataset.accessed.filter { $0 }.count, i * batchSize,
          "Should have accessed \(i * batchSize) elements.")
        let _ = Array(batch)
        XCTAssertEqual(
          dataset.accessed.filter { $0 }.count, (i + 1) * batchSize,
          "Should have accessed \((i + 1) * batchSize) elements.")
      }
    }
  }

  // Use with padding
  // Let's create an array of things of various lengths (for instance texts)
  let nonuniformDataset: [Tensor<Int32>] = {
    var dataset: [Tensor<Int32>] = []
    for _ in 0..<512 {
      dataset.append(
        Tensor<Int32>(
          repeating: 1,
          shape: [Int.random(in: 1...200, using: &rng)]
        ))
    }
    return dataset
  }()

  func paddingTest(padValue: Int32, atStart: Bool) {
    let batches = nonuniformDataset.inBatches(of: 64)
      .lazy.map { $0.paddedAndCollated(with: padValue, atStart: atStart) }
    for (i, b) in batches.enumerated() {
      let shapes = nonuniformDataset[(i * 64)..<((i + 1) * 64)]
        .map { Int($0.shape[0]) }
      let expectedShape = shapes.reduce(0) { max($0, $1) }
      XCTAssertEqual(
        Int(b.shape[1]), expectedShape,
        "The batch does not have the expected shape: \(expectedShape).")

      for k in 0..<64 {
        let currentShape = nonuniformDataset[i * 64 + k].shape[0]
        let paddedPart =
          atStart ? b[k, 0..<(expectedShape - currentShape)] : (b[k, currentShape..<expectedShape])
        XCTAssertEqual(
          paddedPart,
          Tensor<Int32>(
            repeating: padValue,
            shape: [expectedShape - currentShape]),
          "Padding was not found where it should be.")
      }
    }
  }

  func testAllPadding() {
    paddingTest(padValue: 0, atStart: false)
    paddingTest(padValue: 42, atStart: false)
    paddingTest(padValue: 0, atStart: true)
    paddingTest(padValue: -1, atStart: true)
  }

  let cuts = [0, 5, 8, 15, 24, 30]
  var texts: [[Int]] { (0..<5).map { Array(cuts[$0]..<cuts[$0 + 1]) } }

  // To reindex the dataset such that the first batch samples are given by
  // indices (0, batchCount, batchCount * 2, ...
  func preBatchTranspose<C: Collection>(_ base: C, for batchSize: Int)
    -> [C.Index]
  {
    let batchCount = base.count / batchSize
    return (0..<base.count).map { (i: Int) -> C.Index in
      let j = batchCount * (i % batchSize) + i / batchSize
      return base.index(base.startIndex, offsetBy: j)
    }
  }

  //Now let's look at what it gives us:
  func testLanguageModel() {
    let sequenceLength = 3
    let batchSize = 2

    let sequences = texts.joined()
      .inBatches(of: sequenceLength)
    let indices = preBatchTranspose(sequences, for: batchSize)
    let batches = sequences.sampled(at: indices).inBatches(of: batchSize)

    var results: [[Int32]] = [[], []]
    for batch in batches {
      let tensor = Tensor<Int32>(
        batch.map {
          Tensor<Int32>(
            $0.map { Int32($0) })
        })
      XCTAssertEqual(tensor.shape, TensorShape([2, 3]))
      results[0] += tensor[0].scalars
      results[1] += tensor[1].scalars
    }
    XCTAssertEqual(results[0] + results[1], (0..<30).map { Int32($0) })
  }

  func isSubset(_ x: [Int], from y: [Int]) -> Bool {
    if let i = y.firstIndex(of: x[0]) {
      return x.enumerated().allSatisfy { (k: Int, o: Int) -> Bool in
        o == y[i + k]
      }
    }
    return false
  }

  func testLanguageModelShuffled() {
    let sequenceLength = 3
    let batchSize = 2

    let sequences = texts.shuffled().joined()
      .inBatches(of: sequenceLength)
    let indices = preBatchTranspose(sequences, for: batchSize)
    let batches = sequences.sampled(at: indices).inBatches(of: batchSize)

    var results: [[Int32]] = [[], []]
    for batch in batches {
      let tensor = Tensor<Int32>(
        batch.map {
          Tensor<Int32>(
            $0.map { Int32($0) })
        })
      XCTAssertEqual(tensor.shape, TensorShape([2, 3]))
      results[0] += tensor[0].scalars
      results[1] += tensor[1].scalars
    }
    let stream = (results[0] + results[1]).map { Int($0) }
    XCTAssertEqual(stream.count, 30)
    XCTAssert(texts.allSatisfy { isSubset($0, from: stream) })
  }

  class SizedSample {
    init(size: Int) { self.size = size }
    var size: Int
  }

  func testNonuniformInferenceBatches() {
    let sampleCount = 503
    let batchSize = 7
    let samples = (0..<sampleCount).map {
      _ in SizedSample.init(size: Int.random(in: 0..<1000, using: &rng))
    }
    let batches = NonuniformInferenceBatches(
      samples: samples, batchSize: batchSize
    ) { $0.size < $1.size }

    XCTAssertEqual(
      batches.count, sampleCount / batchSize + 1,
      "Wrong number of batches")
    var previousSize: Int? = nil
    for (i, batchSamples) in batches.enumerated() {
      let batch = Array(batchSamples)
      XCTAssertEqual(
        batch.count,
        i == batches.count - 1 ? sampleCount % batchSize : batchSize,
        "Wrong number of samples in this batch.")
      let newSize = batch.map(\.size).max()!
      if let size = previousSize {
        XCTAssert(
          size >= newSize,
          "Batch should be sorted through size.")
      }
      previousSize = Int(newSize)
    }
  }

  func testNonuniformTrainingEpochs() {
    let sampleCount = 503
    let batchSize = 7
    let samples = (0..<sampleCount).map {
      _ in SizedSample.init(size: Int.random(in: 0..<1000, using: &rng))
    }

    let epochs = NonuniformTrainingEpochs(
      samples: samples,
      batchSize: batchSize,
      entropy: rng
    ) { $0.size < $1.size }

    // The first sample ordering observed during this test.
    var observedSampleOrder: [ObjectIdentifier]?

    for batches in epochs.prefix(10) {
      XCTAssertEqual(batches.count, sampleCount / batchSize)
      XCTAssert(batches.allSatisfy { $0.count == batchSize })
      let epochSamples = batches.joined()
      let epochSampleOrder = epochSamples.lazy.map(ObjectIdentifier.init)

      if let o = observedSampleOrder {
        XCTAssertFalse(
          o.elementsEqual(epochSampleOrder),
          "Batches should be randomized")
      } else {
        observedSampleOrder = Array(epochSampleOrder)
      }

      let maxEpochSampleSize = epochSamples.lazy.map(\.size).max()!
      XCTAssertEqual(
        batches.first!.lazy.map(\.size).max(),
        maxEpochSampleSize,
        "The first batch should contain a sample of maximal size.")

      let uniqueSamples = Set(epochSampleOrder)
      XCTAssertEqual(
        uniqueSamples.count, epochSamples.count,
        "Every epoch sample should be drawn from a different input sample.")
    }
  }
}

extension EpochsTests {
  static var allTests = [
    ("testAllPadding", testAllPadding),
    ("testInBatchesIsLazy", testInBatchesIsLazy),
    ("testBaseUse", testBaseUse),
    ("testTrainingEpochsShuffles", testTrainingEpochsShuffles),
    ("testTrainingEpochsShapes", testTrainingEpochsShapes),
    ("testTrainingEpochsIsLazy", testTrainingEpochsIsLazy),
    ("testLanguageModel", testLanguageModel),
    ("testLanguageModelShuffled", testLanguageModelShuffled),
    ("testNonuniformInferenceBatches", testNonuniformInferenceBatches),
    ("testNonuniformTrainingEpochs", testNonuniformTrainingEpochs),
  ]
}
