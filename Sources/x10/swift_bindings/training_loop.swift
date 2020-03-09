// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import x10_device
import x10_tensor
import x10_xla_tensor_wrapper

/// Collects correct prediction counters and loss totals.
public struct HostStatistics {
  public init() {}

  public init(correctGuessCount: Int, totalSamples: Int, totalLoss: Float) {
    self.correctGuessCount = correctGuessCount
    self.totalSamples = totalSamples
    self.totalLoss = totalLoss
  }
  public var correctGuessCount: Int = 0
  public var totalSamples: Int = 0
  public var totalLoss: Float = 0

  static func += (lhs: inout HostStatistics, rhs: HostStatistics) {
    lhs.correctGuessCount += rhs.correctGuessCount
    lhs.totalSamples += rhs.totalSamples
    lhs.totalLoss += rhs.totalLoss
  }
}

fileprivate func makeNibbleTensor(_ input: [Int], on device: Device) -> Tensor<Float> {
  var scalars = [Float]()
  for i in input {
    var iCopy = i
    for _ in 0..<16 {
      scalars.append(Float(iCopy & 15))
      iCopy = iCopy >> 4
    }
  }
  return Tensor<Float>(shape: [input.count, 16], scalars: scalars, on: device)
}

fileprivate func unpackNibbles(_ scalars: [Float]) -> Int {
  var out: Int = 0
  for i in 0..<scalars.count {
    out += Int(scalars[i]) << (i * 4)
  }
  return out
}

fileprivate func unpackNibbles(_ input: Tensor<Float>) -> [Int] {
  let shape = input.shape.dimensions
  precondition(shape.count == 2 && shape[1] == 16)
  var out = [Int]()
  let scalars = input.scalars
  for i in 0..<shape[0] {
    out.append(unpackNibbles((0..<16).map { scalars[i * 16 + $0] }))
  }
  return out
}

fileprivate func crsHostStats(
  _ stats: inout HostStatistics,
  on device: Device, devices: [Device]
) {
  var ints = makeNibbleTensor(
    [
      stats.correctGuessCount,
      stats.totalSamples,
    ], on: device)
  var floats = Tensor<Float>([stats.totalLoss], on: device)
  ints.crossReplicaSum(1)
  floats.crossReplicaSum(1)
  LazyTensorBarrier(on: device, devices: devices, wait: true)
  let intsScalars = unpackNibbles(ints)
  let floatsScalars = floats.scalars

  stats.correctGuessCount = Int(intsScalars[0])
  stats.totalSamples = Int(intsScalars[1])
  stats.totalLoss = floatsScalars[0]
}

public func summarizeStatistics(
  _ stats: [(train: HostStatistics, test: HostStatistics)]
) -> (train: HostStatistics, test: HostStatistics) {
  var trainStats = HostStatistics()
  var testStats = HostStatistics()
  for result in stats {
    trainStats += result.train
    testStats += result.test
  }
  // crs the stats if we're running multihost:
  let crsDevices = Device.crossReplicaSumDevices
  let trainingDevices = Device.trainingDevices
  if crsDevices.count != trainingDevices.count {
    let devices = stride(from: 0, to: crsDevices.count, by: trainingDevices.count).map {
      crsDevices[$0]
    }
    let device = devices.first { !$0.isRemote }!
    crsHostStats(&trainStats, on: device, devices: devices)
    crsHostStats(&testStats, on: device, devices: devices)
  }
  return (train: trainStats, test: testStats)
}

/// Creates a string summary of a list of training and testing stats.
public func formatStatistics(_ stats: [(train: HostStatistics, test: HostStatistics)]) -> String {
  let (trainStats, testStats) = summarizeStatistics(stats)
  return formatStatistics(train: trainStats, test: testStats)
}
public func formatStatistics(train trainStats: HostStatistics, test testStats: HostStatistics)
  -> String
{
  let trainAccuracy = Float(trainStats.correctGuessCount) / Float(trainStats.totalSamples)
  let testAccuracy = Float(testStats.correctGuessCount) / Float(testStats.totalSamples)
  return """
    Training Loss: \(trainStats.totalLoss / Float(trainStats.totalSamples)), \
    Training Accuracy: \(trainStats.correctGuessCount)/\(trainStats.totalSamples) \
    (\(trainAccuracy)), \
    Test Loss: \(testStats.totalLoss / Float(testStats.totalSamples)), \
    Test Accuracy: \(testStats.correctGuessCount)/\(testStats.totalSamples) \
    (\(testAccuracy))
    """
}

/// Collects correct prediction totals and losses on a device.
struct Statistics {
  var correctGuessCountTensor: Tensor<Int32>
  var correctGuessCount: Int { return Int(correctGuessCountTensor.scalarized()) }
  var totalSamples: Int = 0
  var totalLoss: Float { return totalLossTensor.scalarized() }
  var totalLossTensor: Tensor<Float>

  var hostStats: HostStatistics {
    HostStatistics(
      correctGuessCount: correctGuessCount,
      totalSamples: totalSamples, totalLoss: totalLoss)
  }

  public init(on device: Device) {
    correctGuessCountTensor = Tensor<Int32>(0, on: device)
    totalLossTensor = Tensor<Float>(0, on: device)
  }
}

@differentiable
public func _defaultLossFunction(_ ŷ: Tensor<Float>, _ y: Tensor<Int32>) -> Tensor<Float> {
  softmaxCrossEntropy(logits: ŷ, labels: y)
}

/// The state of a training loop on a device.
public class ThreadState<Model: Layer, Opt: Optimizer>
where
  Opt.Model == Model, Opt.Scalar == Float, Model.Input == Tensor<Float>,
  Model.Output == Tensor<Float>,
  Model.TangentVector.VectorSpaceScalar == Float
{
  public var classifier: Model
  public var optimizer: Opt
  let threadId: Int
  let devices: [Device]
  let useAutomaticMixedPrecision: Bool

  public init(
    model: Model, optimizer: Opt, id: Int, devices: [Device], useAutomaticMixedPrecision: Bool
  ) {
    self.threadId = id
    self.devices = devices
    self.classifier = Model(copying: model, to: devices[id])
    self.optimizer = Opt(copying: optimizer, to: devices[id])
    self.useAutomaticMixedPrecision = useAutomaticMixedPrecision
  }

  public func run<Dataset: Sequence>(
    train: Dataset, test: Dataset, crossReplicaSumDevices: [Device]? = nil,
    scheduleLearningRate: (Opt) -> Void = { _ in },
    lossFunction: @differentiable (Tensor<Float>, @noDerivative Tensor<Int32>) -> Tensor<Float> =
      _defaultLossFunction
  )
    -> (train: HostStatistics, test: HostStatistics)
  where Dataset.Iterator.Element == (x: Tensor<Float>, y: Tensor<Int32>) {
    let device = devices[threadId]
    let crsDevices = crossReplicaSumDevices ?? devices

    LazyTensorBarrier(on: device, wait: true)

    var trainStats = Statistics(on: device)
    var testStats = Statistics(on: device)
    Context.local.learningPhase = .training
    for (x, y) in train {
      let scope = MakeAnnotationScope("training")
      let scopeTracing = MakeAnnotationScope("training-tracing")
      var detailedScopeTracing = MakeAnnotationScope("fwd-training-tracing")
      // x might have been constructed directly with reduced precision, check for that.
      let input = (useAutomaticMixedPrecision && !x.isReducedPrecision) ? x.toReducedPrecision : x
      // Compute the gradient with respect to the model.
      let reducedPrecisionClassifier =
        useAutomaticMixedPrecision
        ? classifier.toReducedPrecision : classifier
      let 𝛁model = gradient(at: reducedPrecisionClassifier) {
        reducedPrecisionClassifier -> Tensor<Float> in
        let ŷ = reducedPrecisionClassifier(input)
        let correctPredictions = ŷ.argmax(squeezingAxis: 1) .== y
        trainStats.correctGuessCountTensor +=
          Tensor<Int32>(correctPredictions).sum()
        trainStats.totalSamples += y.shape[0]
        let loss = lossFunction(ŷ, y)
        trainStats.totalLossTensor +=
          Float(y.shape[0]) * (self.useAutomaticMixedPrecision ? loss.toFullPrecision : loss)
        DestroyAnnotationScope(detailedScopeTracing)
        detailedScopeTracing = MakeAnnotationScope("back-training-tracing")
        return loss
      }
      DestroyAnnotationScope(detailedScopeTracing)
      detailedScopeTracing = MakeAnnotationScope("optimizer-training-tracing")
      // Update the model's differentiable variables along the gradient vector.
      scheduleLearningRate(optimizer)
      optimizer.update(
        &classifier, along: useAutomaticMixedPrecision ? 𝛁model.toFullPrecision : 𝛁model)
      DestroyAnnotationScope(detailedScopeTracing)
      DestroyAnnotationScope(scopeTracing)
      LazyTensorBarrier(on: device, devices: crsDevices)
      DestroyAnnotationScope(scope)
    }

    Context.local.learningPhase = .inference
    let reducedPrecisionClassifier =
      useAutomaticMixedPrecision
      ? classifier.toReducedPrecision : classifier
    for (x, y) in test {
      let scope = MakeAnnotationScope("test")
      // x might have been constructed directly with reduced precision, check for that.
      let input = (useAutomaticMixedPrecision && !x.isReducedPrecision) ? x.toReducedPrecision : x
      // Compute loss on test set
      let ŷ = reducedPrecisionClassifier(input)
      let correctPredictions = ŷ.argmax(squeezingAxis: 1) .== y
      testStats.correctGuessCountTensor += Tensor<Int32>(correctPredictions).sum()
      testStats.totalSamples += y.shape[0]
      let loss = lossFunction(ŷ, y)
      testStats.totalLossTensor +=
        Float(y.shape[0]) * (useAutomaticMixedPrecision ? loss.toFullPrecision : loss)
      LazyTensorBarrier(on: device)
      DestroyAnnotationScope(scope)
    }
    return (train: trainStats.hostStats, test: testStats.hostStats)
  }
}

class ThreadResultBox<T> {
  init() {}
  var data: T? = nil
}

/// Maps a function over n threads.
public func runOnNThreads<R>(_ nThreads: Int, _ body: @escaping (Int) -> R) -> [R] {
  let results = (0..<nThreads).map { _ in ThreadResultBox<R>() }

  // TODO(parkers): Don't use Tensorflow version of _runOnNDevices
  // because it doesn't use a threadpool.
  _runOnNDevices(nThreads) { threadId in
    results[threadId].data = body(threadId)
  }
  return results.map { $0.data! }
}

extension Device {
  /// A list of devices used for training.
  public static var trainingDevices: [Device] {
    let allDevices = Device.allDevices

    let tpuDevices = allDevices.filter { $0.kind == .TPU }
    // On CPU, run on the last device to allow device 1 testing. (MNIST would die
    // if it was run on a second device).
    let cpuDevices = [allDevices.filter { $0.kind == .CPU }.last!]
    return tpuDevices.count > 0 ? tpuDevices : cpuDevices
  }

  /// A list of devices used for cross replica sums when training on trainingDevices.
  public static var crossReplicaSumDevices: [Device] {
    let allDevices = Device.allDevices
    // Match `.trainingDevices` logic but include remote devices.
    let tpuDevices = allDevices.filter { $0.kind == .TPU || $0.kind == .REMOTE_TPU }
    let cpuDevices = [allDevices.filter { $0.kind == .CPU }.last!]
    return (tpuDevices.filter { $0.kind == .TPU }).count > 0 ? tpuDevices : cpuDevices
  }
}
