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

/// The default graph seed.
///
/// - Note: See TensorFlow's `python.framework.random_seed.DEFAULT_GRAPH_SEED`.
@available(*, deprecated, message: "Graph-level tracing will be removed in S4TF v0.10")
@usableFromInline let _defaultGraphSeed: Int64 = 87_654_321

/// Returns the local seeds an operation should use given an op-specific seed.
///
/// Given operation-specific seed, `seed`, this helper function returns two seeds derived from
/// graph-level and op-level seeds. Many random operations internally use the two seeds to allow
/// user to change the seed globally for a graph, or for only specific operations.
///
/// - Note: See TensorFlow's `python.framework.random_seed.get_seed`.
///
// TODO: There's no support for TF's "global seed" yet, so we always use the default graph seed as
// the first seed. Need to investigate the best way to model TF's "global seed".
@available(*, deprecated, message: "Graph-level tracing will be removed in S4TF v0.10")
@usableFromInline
func _tensorSeeds(_ seed: Tensor<Int64>) -> (Tensor<Int64>, Tensor<Int64>) {
  return (Tensor(_defaultGraphSeed, on: .defaultTFEager), seed)
}

//===------------------------------------------------------------------------------------------===//
// Single Value Dataset
//===------------------------------------------------------------------------------------------===//

/// Represents a potentially large set of elements.
///
/// A `Dataset` can be used to represent an input pipeline as a collection of element tensors.
@available(
  *, deprecated,
  message:
    """
  Datasets will be removed in S4TF v0.10. Please use the new Batches API instead.
  """
)
@frozen
public struct Dataset<Element: TensorGroup> {
  public let _handle: VariantHandle

  @inlinable
  public init(_handle: VariantHandle) {
    self._handle = _handle
  }
}

@available(*, deprecated)
extension Dataset {
  @inlinable
  public init(randomSeed: Int64) {
    let (seed1, seed2) = _tensorSeeds(Tensor(randomSeed, on: .defaultTFEager))
    self.init(
      _handle: _Raw.experimentalRandomDataset(
        seed: seed1,
        seed2: seed2,
        outputTypes: Element._typeList,
        outputShapes: Element._unknownShapeList))
  }
}

@available(*, deprecated)
extension Dataset {
  /// Creates a dataset from a batch of elements as a tensor.
  @inlinable
  public init(elements: Element) {
    self.init(
      _handle: _Raw.tensorSliceDataset(
        components: [elements],
        outputShapes: Element._unknownShapeList))
  }
}

@available(*, deprecated)
extension Dataset: Sequence {
  public typealias Iterator = DatasetIterator<Element>

  /// Returns an iterator over the elements of this dataset.
  @inlinable
  public func makeIterator() -> DatasetIterator<Element> {
    let resource = _Raw.anonymousIterator(
      outputTypes: Element._typeList,
      outputShapes: Element._unknownShapeList)
    _Raw.makeIterator(dataset: _handle, iterator: resource)
    return DatasetIterator(_handle: resource)
  }
}

@available(*, deprecated)
extension Dataset {
  // Note that this Dataset API implementation uses an experimental tracing feature, which is not
  // robust and does not have great diagnostics yet.
  @inlinable
  public func map<ResultElement: TensorGroup>(
    _ transform: (Element) -> ResultElement
  ) -> Dataset<ResultElement> {
    return Dataset<ResultElement>(
      _handle: _Raw.mapDataset(
        inputDataset: _handle,
        otherArguments: Tensor<Int32>(0, on: .defaultTFEager),
        f: transform,
        outputTypes: ResultElement._typeList,
        outputShapes: ResultElement._unknownShapeList,
        useInterOpParallelism: true,
        preserveCardinality: false))
  }

  @inlinable
  public func map<ResultElement: TensorGroup>(
    parallelCallCount: Int,
    _ transform: (Element) -> ResultElement
  ) -> Dataset<ResultElement> {
    return Dataset<ResultElement>(
      _handle: _Raw.parallelMapDataset(
        inputDataset: _handle,
        otherArguments: Tensor<Int32>(0, on: .defaultTFEager),
        numParallelCalls: Tensor<Int32>(Int32(parallelCallCount), on: .defaultTFEager),
        f: transform,
        outputTypes: ResultElement._typeList,
        outputShapes: ResultElement._unknownShapeList,
        useInterOpParallelism: true,
        sloppy: false,
        preserveCardinality: false))
  }

  @inlinable
  public func filter(_ isIncluded: (Element) -> Tensor<Bool>) -> Dataset {
    return Dataset(
      _handle: _Raw.filterDataset(
        inputDataset: _handle,
        otherArguments: Tensor<Int32>(0, on: .defaultTFEager),
        predicate: isIncluded,
        outputTypes: Element._typeList,
        outputShapes: Element._unknownShapeList))
  }
}

@available(*, deprecated)
extension Dataset {
  @inlinable
  public func prefetched(count: Int) -> Dataset {
    return Dataset(
      _handle: _Raw.prefetchDataset(
        inputDataset: _handle,
        bufferSize: Tensor(Int64(count), on: .defaultTFEager),
        outputTypes: Element._typeList,
        outputShapes: Element._unknownShapeList))
  }

  @inlinable
  public func shuffled(
    sampleCount: Int,
    randomSeed: Int64,
    reshuffleForEachIterator: Bool = true
  ) -> Dataset {
    let (seed1, seed2) = _tensorSeeds(Tensor(randomSeed, on: .defaultTFEager))
    return Dataset(
      _handle: _Raw.shuffleDataset(
        inputDataset: _handle,
        bufferSize: Tensor(Int64(sampleCount), on: .defaultTFEager),
        seed: seed1,
        seed2: seed2,
        reshuffleEachIteration: reshuffleForEachIterator,
        outputTypes: Element._typeList,
        outputShapes: Element._unknownShapeList))
  }

  @inlinable
  public func batched(_ batchSize: Int) -> Dataset {
    return Dataset(
      _handle: _Raw.batchDataset(
        inputDataset: _handle,
        batchSize: Tensor(Int64(batchSize), on: .defaultTFEager),
        outputTypes: Element._typeList,
        outputShapes: Element._unknownShapeList))
  }

  @inlinable
  public func repeated(count: Int? = nil) -> Dataset {
    return Dataset(
      _handle: _Raw.repeatDataset(
        inputDataset: _handle,
        count: Tensor(Int64(count ?? -1), on: .defaultTFEager),
        outputTypes: Element._typeList,
        outputShapes: Element._unknownShapeList))
  }
}

/// The type that allows iteration over a dataset's elements.
@available(*, deprecated)
@frozen
public struct DatasetIterator<Element: TensorGroup> {
  @usableFromInline let _handle: ResourceHandle

  @usableFromInline
  internal init(_handle: ResourceHandle) {
    self._handle = _handle
  }
}

@available(*, deprecated)
extension DatasetIterator: IteratorProtocol {
  /// Advances to the next element and returns it, or `nil` if no next element exists.
  @inlinable
  public mutating func next() -> Element? {
    let optional = _Raw.iteratorGetNextAsOptional(
      iterator: _handle,
      outputTypes: Element._typeList,
      outputShapes: Element._unknownShapeList)
    guard _Raw.optionalHasValue(optional: optional).scalarized() else {
      return nil
    }
    return _Raw.optionalGetValue(
      optional: optional,
      outputShapes: Element._unknownShapeList)
  }
}

/// A 2-tuple-like struct that conforms to TensorGroup that represents a tuple of 2 types conforming
/// to `TensorGroup`.
@frozen
public struct Zip2TensorGroup<T: TensorGroup, U: TensorGroup>: TensorGroup {
  public var first: T
  public var second: U

  public init(_ first: T, _ second: U) {
    self.first = first
    self.second = second
  }

  public static var _typeList: [TensorDataType] { return T._typeList + U._typeList }

  public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
    first = .init(_owning: tensorHandles)
    second = .init(_owning: tensorHandles?.advanced(by: Int(T._tensorHandleCount)))
  }

  public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
    var ptr = address
    first._unpackTensorHandles(into: ptr)
    ptr = ptr!.advanced(by: Int(first._tensorHandleCount))
    second._unpackTensorHandles(into: ptr)
  }

  public var _tensorHandles: [_AnyTensorHandle] {
    first._tensorHandles + second._tensorHandles
  }

  public init<C: RandomAccessCollection>(
    _handles: C
  ) where C.Element: _AnyTensorHandle {
    let firstStart = _handles.startIndex
    let firstEnd = _handles.index(
      firstStart, offsetBy: Int(T._tensorHandleCount))
    self.first = T.init(_handles: _handles[firstStart..<firstEnd])
    self.second = U.init(_handles: _handles[firstEnd..<_handles.endIndex])
  }
}

// TODO(SR-9156): This does not work in graph mode.
@available(*, deprecated, message: "Graph-level tracing will be removed in S4TF v0.10")
@inlinable
public func zip<T: TensorGroup, U: TensorGroup>(
  _ dataset1: Dataset<T>, _ dataset2: Dataset<U>
) -> Dataset<Zip2TensorGroup<T, U>> {
  let handle = _Raw.zipDataset(
    inputDatasets: [dataset1._handle, dataset2._handle],
    outputTypes: Zip2TensorGroup<T, U>._typeList,
    outputShapes: Zip2TensorGroup<T, U>._unknownShapeList)
  return Dataset(_handle: handle)
}
