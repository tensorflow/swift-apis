import TensorFlow
import XCTest
import x10_device
import x10_tensor
import x10_xla_tensor_wrapper

typealias TFTensor_ = TensorFlow.Tensor
typealias X10Tensor_ = x10_tensor.Tensor
typealias TFTensor = TensorFlow.Tensor<Float>
typealias X10Tensor = x10_tensor.Tensor<Float>
typealias TensorRange = x10_tensor.TensorRange
typealias Conv2D = x10_tensor.Conv2D

private func TF<T>(_ x: x10_tensor.Tensor<T>) -> TensorFlow.Tensor<T> {
  return TensorFlow.Tensor<T>(shape: TensorFlow.TensorShape(x.shape.dimensions), scalars: x.scalars)
}

/// Returns true iff the absolute difference between all elements is at most `absTolerance`.
private func allClose(
  actual: TFTensor, expected: TFTensor, relTolerance: Float = 1e-5, absTolerance: Float = 1e-7
) -> Bool {
  return (abs(actual - expected) .<= absTolerance + relTolerance * abs(expected)).all()
}

private func TF(_ range: TensorRange) -> TensorFlow.TensorRange {
  switch range {
  case .ellipsis:
    return .ellipsis
  case .newAxis:
    return .newAxis
  case .squeezeAxis:
    return .squeezeAxis
  case let .index(i):
    return .index(i)
  case let .range(r, s):
    return .range(r, stride: s)
  case let .closedRange(r, s):
    return .closedRange(r, stride: s)
  case let .partialRangeFrom(r, s):
    return .partialRangeFrom(r, stride: s)
  case let .partialRangeUpTo(r, s):
    return .partialRangeUpTo(r, stride: s)
  case let .partialRangeThrough(r, s):
    return .partialRangeThrough(r, stride: s)
  }
}

private func assertEqualUnaryOperationGradients(
  _ xlaOp: @differentiable (X10Tensor) -> X10Tensor,
  _ tensorFlowOp: @differentiable (TFTensor) -> TFTensor,
  _ x: X10Tensor,
  _ outGrad: X10Tensor,
  relTolerance: Float = 1e-5,
  absTolerance: Float = 1e-7,
  file: StaticString = #file, line: UInt = #line
) {
  var (actual, actualPullback) = valueWithPullback(at: x, in: xlaOp)
  let useReducedPrecision = x.isReducedPrecision
  if useReducedPrecision {
    XCTAssert(outGrad.isReducedPrecision)
    XCTAssert(actual.isReducedPrecision)
    actual = actual.toFullPrecision
  }
  XCTAssert(!actual.isReducedPrecision)
  let (expected, expectedPullback) = valueWithPullback(at: TF(x), in: tensorFlowOp)
  XCTAssert(
    allClose(
      actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: absTolerance
    ), file: file,
    line: line)
  var actualOutGrad = actualPullback(outGrad)
  XCTAssertEqual(actualOutGrad.isReducedPrecision, useReducedPrecision)
  if actualOutGrad.isReducedPrecision {
    actualOutGrad = actualOutGrad.toFullPrecision
  }
  XCTAssert(
    allClose(
      actual: TF(actualOutGrad), expected: expectedPullback(TF(outGrad)),
      relTolerance: relTolerance, absTolerance: absTolerance),
    file: file, line: line)
}

private func assertEqualBinaryOperationGradients(
  _ xlaOp: @differentiable (X10Tensor, X10Tensor) -> X10Tensor,
  _ tensorFlowOp: @differentiable (TFTensor, TFTensor) -> TFTensor,
  _ x: X10Tensor,
  _ y: X10Tensor,
  _ outGrad: X10Tensor,
  relTolerance: Float = 1e-5,
  absTolerance: Float = 1e-7,
  file: StaticString = #file, line: UInt = #line
) {
  var (actual, actualPullback) = valueWithPullback(at: x, y, in: xlaOp)
  let useReducedPrecision = x.isReducedPrecision
  if useReducedPrecision {
    XCTAssert(y.isReducedPrecision)
    XCTAssert(outGrad.isReducedPrecision)
    XCTAssert(actual.isReducedPrecision)
    actual = actual.toFullPrecision
  }
  XCTAssert(!actual.isReducedPrecision)
  let (expected, expectedPullback) = valueWithPullback(at: TF(x), TF(y), in: tensorFlowOp)
  XCTAssert(
    allClose(
      actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: absTolerance
    ), file: file,
    line: line)
  var (actualGradX, actualGradY) = actualPullback(outGrad)
  XCTAssertEqual(actualGradX.isReducedPrecision, useReducedPrecision)
  XCTAssertEqual(actualGradY.isReducedPrecision, useReducedPrecision)
  if useReducedPrecision {
    actualGradX = actualGradX.toFullPrecision
    actualGradY = actualGradY.toFullPrecision
  }
  let (expectedGradX, expectedGradY) = expectedPullback(TF(outGrad))
  XCTAssert(
    allClose(
      actual: TF(actualGradX), expected: expectedGradX, relTolerance: relTolerance,
      absTolerance: absTolerance),
    file: file, line: line)
  XCTAssert(
    allClose(
      actual: TF(actualGradY), expected: expectedGradY, relTolerance: relTolerance,
      absTolerance: absTolerance),
    file: file, line: line)
}

protocol IntOrBool {
  init(_ v: Int)
}

extension Bool: IntOrBool {
  init(_ v: Int) {
    self = v != 0
  }
}

extension Int32: IntOrBool {}

extension X10Tensor {
  private static var randSeed = 0

  static func rand(_ dims: [Int]) -> X10Tensor {
    randSeed = randSeed + 1
    return _Raw.rand(dims, randSeed)
  }

  static func randint<T: IntOrBool>(_ low: Int, _ high: Int, _ shape: [Int]) -> X10Tensor_<T> {
    let numel = shape.reduce(1, *)
    return X10Tensor_<T>(
      shape: TensorShape(
        shape), scalars: (0..<numel).map { _ in T(Int.random(in: low..<high)) })
  }
}

final class TensorTests: XCTestCase {
  func testAbs() throws {
    let dims = [3, 2]
    let off = X10Tensor(
      shape: TensorShape(dims), scalars: [Float](repeating: 0.5, count: dims.reduce(1, *)))
    var x = X10Tensor.rand(dims) - off
    let expected = abs(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = abs(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testAcos() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = acos(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = acos(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 3e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
    }
  }

  func testAcosh() throws {
    var x = X10Tensor.rand([3, 2]) + 1
    let expected = acosh(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = acosh(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 8e-2 : 1e-4
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
    }
  }

  func testAdd() throws {
    var x = X10Tensor(shape: [2], scalars: [1, 2])
    var y = X10Tensor(shape: [2], scalars: [7, 19])
    let expected = TF(x) + TF(y)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      if useReducedPrecision {
        y = y.toReducedPrecision
      }
      var actual = x + y
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssertEqual(TF(actual), expected)
    }
  }

  func testAddInterop() throws {
    let x = X10Tensor(shape: [2], scalars: [1, 2], on: Device.defaultTFEager)
    let y = X10Tensor(shape: [2], scalars: [7, 19], on: Device.defaultTFEager)
    XCTAssertEqual((x + y).scalars, [8, 21])
  }

  func testAll() throws {
    let x: X10Tensor_<Bool> = X10Tensor.randint(0, 2, [2, 3, 4])
    for axis in -x.rank..<x.rank {
      let actual = TF(x.all(alongAxes: axis))
      let expected = TF(x).all(alongAxes: axis)
      XCTAssertEqual(actual, expected)
    }
    for axis in -x.rank..<x.rank {
      let actual = TF(x.all(squeezingAxes: axis))
      let expected = TF(x).all(squeezingAxes: axis)
      XCTAssertEqual(actual, expected)
    }
    let dims = [2, 3, 4]
    let allTrue = X10Tensor_<Bool>(
      shape: TensorShape(dims), scalars: [Bool](repeating: true, count: dims.reduce(1, *)))
    for axis in -allTrue.rank..<allTrue.rank {
      let actual = TF(allTrue.all(alongAxes: axis))
      let expected = TF(allTrue).all(alongAxes: axis)
      XCTAssertEqual(actual, expected)
    }
    for axis in -allTrue.rank..<allTrue.rank {
      let actual = TF(allTrue.all(squeezingAxes: axis))
      let expected = TF(allTrue).all(squeezingAxes: axis)
      XCTAssertEqual(actual, expected)
    }
  }

  func testAny() throws {
    let x: X10Tensor_<Bool> = X10Tensor.randint(0, 2, [2, 3, 4])
    for axis in -x.rank..<x.rank {
      let actual = TF(x.any(alongAxes: axis))
      let expected = TF(x).any(alongAxes: axis)
      XCTAssertEqual(actual, expected)
    }
    for axis in -x.rank..<x.rank {
      let actual = TF(x.any(squeezingAxes: axis))
      let expected = TF(x).any(squeezingAxes: axis)
      XCTAssertEqual(actual, expected)
    }
    let dims = [2, 3, 4]
    let allFalse = X10Tensor_<Bool>(
      shape: TensorShape(dims), scalars: [Bool](repeating: true, count: dims.reduce(1, *)))
    for axis in -allFalse.rank..<allFalse.rank {
      let actual = TF(allFalse.any(alongAxes: axis))
      let expected = TF(allFalse).any(alongAxes: axis)
      XCTAssertEqual(actual, expected)
    }
    for axis in -allFalse.rank..<allFalse.rank {
      let actual = TF(allFalse.any(squeezingAxes: axis))
      let expected = TF(allFalse).any(squeezingAxes: axis)
      XCTAssertEqual(actual, expected)
    }
  }

  func testApproximateEqual() throws {
    for useReducedPrecision in [false, true] {
      for tolerance in [0.2, 0.05] {
        var x = X10Tensor.rand([3, 2])
        var noise = X10Tensor(onesLike: x) * 0.1
        if useReducedPrecision {
          x = x.toReducedPrecision
        }
        if useReducedPrecision {
          noise = noise.toReducedPrecision
        }
        let y = x + noise
        let actual = x.elementsAlmostEqual(y, tolerance: Float(tolerance))
        let expected = TF(x).elementsAlmostEqual(TF(y), tolerance: Float(tolerance))
        XCTAssertEqual(TF(actual), expected)
      }
    }
  }

  func testArgmax() throws {
    var x = X10Tensor(shape: [3, 2], scalars: [1, 5, 43, 24, 64, 32])
    let tfX = TF(x)
    let expected = tfX.argmax(squeezingAxis: 0)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      let actual = TF(x.argmax(squeezingAxis: 0))
      XCTAssertEqual(actual, expected)
    }
  }

  func testArgmin() throws {
    for useReducedPrecision in [false, true] {
      var x = X10Tensor(shape: [3, 2], scalars: [1, 5, 43, 24, 64, 32])
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      let tfX = TF(x)
      for axis in [0, 1] {
        let actual = TF(x.argmin(squeezingAxis: axis))
        let expected = tfX.argmin(squeezingAxis: axis)
        XCTAssertEqual(actual, expected)
      }
    }
  }

  func testAsin() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = asin(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = asin(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
    }
  }

  func testAsinh() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = asinh(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = asinh(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-4
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-4))
    }
  }

  func testAtan2() throws {
    let y = X10Tensor([3, 5])
    let x = X10Tensor([2, 3])
    let actual = _Raw.atan2(y, x)
    let expected = _Raw.atan2(TF(y), TF(x))
    XCTAssert(allClose(actual: TF(actual), expected: expected))
  }

  func testAtan() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = atan(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = atan(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
    }
  }

  func testAtanh() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = atanh(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = atanh(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
    }
  }

  func testAvgPool() throws {
    for useReducedPrecision in [false, true] {
      for stride in 1..<3 {
        for padSame in [false, true] {
          var x = X10Tensor.rand([4, 28, 28, 1])
          let expected = avgPool2D(
            TF(x), filterSize: (1, 2, 2, 1), strides: (1, stride, stride, 1),
            padding: padSame ? Padding.same : Padding.valid)
          if useReducedPrecision {
            x = x.toReducedPrecision
          }
          var actual = avgPool2D(
            x, filterSize: (1, 2, 2, 1), strides: (1, stride, stride, 1),
            padding: padSame ? Padding.same : Padding.valid)
          if useReducedPrecision {
            XCTAssert(actual.isReducedPrecision)
            actual = actual.toFullPrecision
          }
          XCTAssert(!actual.isReducedPrecision)
          let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
          XCTAssert(
            allClose(
              actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-7
            ))
        }
      }
    }
  }

  func testAvgPoolGrad() throws {
    for useReducedPrecision in [false, true] {
      for stride in 1..<3 {
        for padSame in [false, true] {
          var x = X10Tensor.rand([4, 28, 28, 1])
          let outShape = avgPool2D(
            TF(x), filterSize: (1, 2, 2, 1), strides: (1, stride, stride, 1),
            padding: padSame ? Padding.same : Padding.valid
          ).shape
          var outGrad = X10Tensor.rand(outShape.dimensions)
          if useReducedPrecision {
            x = x.toReducedPrecision
            outGrad = outGrad.toReducedPrecision
          }
          let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
          assertEqualUnaryOperationGradients(
            { (_ x: X10Tensor) -> X10Tensor in
              avgPool2D(
                x, filterSize: (1, 2, 2, 1), strides: (1, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid)
            },
            { (_ x: TFTensor) -> TFTensor in
              avgPool2D(
                x, filterSize: (1, 2, 2, 1), strides: (1, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid)
            }, x, outGrad, relTolerance: relTolerance, absTolerance: 1e-6)
        }
      }
    }
  }

  func testAvgPool3DGrad() throws {
    for useReducedPrecision in [false, true] {
      for stride in 1..<3 {
        for padSame in [false, true] {
          var x = X10Tensor.rand([4, 28, 28, 28, 1])
          let outShape = avgPool3D(
            TF(x), filterSize: (1, 2, 2, 2, 1), strides: (1, stride, stride, stride, 1),
            padding: padSame ? Padding.same : Padding.valid
          ).shape
          var outGrad = X10Tensor.rand(outShape.dimensions)
          if useReducedPrecision {
            x = x.toReducedPrecision
            outGrad = outGrad.toReducedPrecision
          }
          let relTolerance: Float = useReducedPrecision ? 2e-2 : 1e-5
          assertEqualUnaryOperationGradients(
            { (_ x: X10Tensor) -> X10Tensor in
              avgPool3D(
                x, filterSize: (1, 2, 2, 2, 1), strides: (1, stride, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid)
            },
            { (_ x: TFTensor) -> TFTensor in
              avgPool3D(
                x, filterSize: (1, 2, 2, 2, 1), strides: (1, stride, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid)
            }, x, outGrad, relTolerance: relTolerance, absTolerance: 1e-6)
        }
      }
    }
  }

  func testBatchNorm() throws {
    let featureCount = 3
    let tfModel = TensorFlow.BatchNorm<Float>(featureCount: featureCount)
    for trainingPhase in [false, true] {
      x10_tensor.Context.local.learningPhase =
        trainingPhase ? .training : .inference
      TensorFlow.Context.local.learningPhase = trainingPhase ? .training : .inference
      var model = x10_tensor.BatchNorm<Float>(featureCount: featureCount)
      let shape = [2, 3, 4, featureCount]
      var x = X10Tensor(
        shape: TensorShape(shape),
        scalars: Array(stride(from: 0.0, to: Float(shape.reduce(1, *)), by: 1)))
      let expected = tfModel(TF(x))
      for useReducedPrecision in [false, true] {
        if useReducedPrecision {
          model = model.toReducedPrecision
          x = x.toReducedPrecision
        }
        var actual = model(x)
        if useReducedPrecision {
          XCTAssert(actual.isReducedPrecision)
          actual = actual.toFullPrecision
        }
        XCTAssert(!actual.isReducedPrecision)
        let relTolerance: Float = useReducedPrecision ? 7e-1 : 1e-3
        XCTAssert(
          allClose(actual: TF(actual), expected: expected, relTolerance: relTolerance))
      }
    }
  }

  func testBatchNormGrad() throws {
    let featureCount = 3
    let tfModel = TensorFlow.BatchNorm<Float>(featureCount: featureCount)
    let shape = [2, 3, 2, featureCount]
    for trainingPhase in [false, true] {
      x10_tensor.Context.local.learningPhase =
        trainingPhase ? .training : .inference
      TensorFlow.Context.local.learningPhase = trainingPhase ? .training : .inference
      var model = x10_tensor.BatchNorm<Float>(featureCount: featureCount)
      var x = X10Tensor(
        shape: TensorShape(shape),
        scalars: Array(stride(from: 0.0, to: Float(shape.reduce(1, *)), by: 1)))
      for useReducedPrecision in [false, true] {
        let 𝛁tfModel = gradient(
          at: tfModel,
          in: { tfModel -> TFTensor in
            tfModel(TF(x)).sum()
          })
        if useReducedPrecision {
          x = x.toReducedPrecision
          model = model.toReducedPrecision
        }
        let 𝛁model = gradient(
          at: model,
          in: { model -> X10Tensor in
            model(x).sum()
          })
        XCTAssertEqual(𝛁model.offset.isReducedPrecision, useReducedPrecision)
        XCTAssertEqual(𝛁model.scale.isReducedPrecision, useReducedPrecision)
        XCTAssert(allClose(actual: TF(𝛁model.offset), expected: 𝛁tfModel.offset))
        let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
        XCTAssert(
          allClose(actual: TF(𝛁model.scale), expected: 𝛁tfModel.scale, relTolerance: relTolerance))
      }
    }
  }

  func testBF16Conv2D() {
    let inChannels = 1
    let batchSize = 1
    let dims = [batchSize, 7, 7, inChannels]
    let input = X10Tensor.rand(dims)
    let inputBF16 = input.toReducedPrecision
    let inputF32 = inputBF16.toFullPrecision
    // Materialize the converted tensors so that the graph inputs are the type we want to test
    // (BF16 and F32).
    LazyTensorBarrier(on: input.device)
    let outChannels = 1
    let model = Conv2D<Float>(
      filterShape: (5, 5, inChannels, outChannels), padding: .same, activation: relu,
      filterInitializer: { shape in
        X10Tensor(
          shape: shape, scalars: [Float](repeating: 0.5, count: shape.dimensions.reduce(1, *)))
      })
    // Materialize the model weights to reflect the steady state by leaving random initialization
    // out of the interesting computation.
    LazyTensorBarrier(on: input.device)
    let mixedPrecisionModel = model.toReducedPrecision
    let outputBF16 = mixedPrecisionModel(inputBF16)
    XCTAssert(outputBF16.isReducedPrecision)
    // Materialize the BF16 output to properly test this output type.
    LazyTensorBarrier(on: input.device)
    let output = model(inputF32)
    let outputViaBF16 = outputBF16.toFullPrecision
    XCTAssert(
      allClose(
        actual: TF(outputViaBF16), expected: TF(output), relTolerance: 1e-2)
    )
  }

  func testBF16Construct() {
    let device = x10_device.Device.default
    let scalars: [Float] = [1, 2, 3, 4, 5, 6]
    var actual = X10Tensor(
      shape: [3, 2], scalars: scalars, toReducedPrecision: true, directlyOn: device)
    XCTAssert(actual.isReducedPrecision)
    actual = actual.toFullPrecision
    XCTAssert(!actual.isReducedPrecision)
    let expected = TFTensor(shape: [3, 2], scalars: scalars)
    XCTAssertEqual(TF(actual), expected)
  }

  func testBF16GradientPropagation() {
    let inChannels = 1
    let batchSize = 1
    let dims = [batchSize, 7, 7, inChannels]
    let input = X10Tensor.rand(dims)
    let inputBF16 = input.toReducedPrecision
    let inputF32 = inputBF16.toFullPrecision
    // Materialize the converted tensors so that the graph inputs are the type we want to test
    // (BF16 and F32).
    LazyTensorBarrier(on: input.device)
    let outChannels = 1
    let model = Conv2D<Float>(
      filterShape: (5, 5, inChannels, outChannels), padding: .same, activation: relu,
      filterInitializer: { shape in
        X10Tensor(
          shape: shape, scalars: [Float](repeating: 0.5, count: shape.dimensions.reduce(1, *)))
      })
    // Materialize the model weights to reflect the steady state by leaving random initialization
    // out of the interesting computation.
    LazyTensorBarrier(on: input.device)
    let mixedPrecisionModel = model.toReducedPrecision
    let 𝛁model = x10_tensor.gradient(
      at: mixedPrecisionModel,
      in: { mixedPrecisionModel -> X10Tensor in
        let ŷ = mixedPrecisionModel(inputBF16)
        let loss = ŷ.sum()
        return loss
      })
    let 𝛁modelViaBF16 = x10_tensor.gradient(
      at: model,
      in: { model -> X10Tensor in
        let ŷ = model(inputF32)
        let loss = ŷ.sum()
        return loss
      })
    LazyTensorBarrier(on: input.device)
    XCTAssert(allClose(actual: TF(𝛁modelViaBF16.bias), expected: TF(𝛁model.bias)))
    XCTAssert(
      allClose(actual: TF(𝛁modelViaBF16.filter), expected: TF(𝛁model.filter), relTolerance: 1e-2))
  }

  func testBF16Loopback() {
    let dims = [3, 2]
    let input = X10Tensor(
      shape: TensorShape(dims), scalars: [Float](repeating: 123456.7, count: dims.reduce(1, *)))
    let inputBF16 = input.toReducedPrecision
    XCTAssert(inputBF16.isReducedPrecision)
    // Materialize the input tensor so that the graph input is of BF16 type.
    LazyTensorBarrier(on: input.device)
    let inputReadBack = inputBF16.toFullPrecision
    XCTAssert(
      allClose(actual: TF(inputReadBack), expected: TF(input), relTolerance: 1e-3))
  }

  func testBF16SparseSoftmaxCrossEntropyWithLogits() throws {
    let labels = X10Tensor_<Int32>(shape: [2], scalars: [3, 4])
    let logits = X10Tensor.rand([2, 5])
    let logitsBF16 = logits.toReducedPrecision
    let logitsF32 = logitsBF16.toFullPrecision
    LazyTensorBarrier(on: logits.device)
    let outputBF16 = _Raw.sparseSoftmaxCrossEntropyWithLogits(features: logitsBF16, labels: labels)
    XCTAssert(outputBF16.loss.isReducedPrecision)
    XCTAssert(outputBF16.backprop.isReducedPrecision)
    let output = _Raw.sparseSoftmaxCrossEntropyWithLogits(features: logitsF32, labels: labels)
    let outputLossViaBF16 = outputBF16.loss.toFullPrecision
    let outputBackpropViaBF16 = outputBF16.backprop.toFullPrecision
    XCTAssert(
      allClose(
        actual: TF(outputLossViaBF16), expected: TF(output.loss), relTolerance: 1e-2))
    XCTAssert(
      allClose(
        actual: TF(outputBackpropViaBF16), expected: TF(output.backprop), relTolerance: 2e-2))
  }

  func testBF16Sum() {
    let dims = [3, 2]
    let input = X10Tensor.rand(dims)
    let inputBF16 = input.toReducedPrecision
    let inputF32 = inputBF16.toFullPrecision
    LazyTensorBarrier(on: input.device)
    let outputBF16 = inputBF16.sum()
    XCTAssert(outputBF16.isReducedPrecision)
    let output = inputF32.sum()
    let outputViaBF16 = outputBF16.toFullPrecision
    XCTAssert(allClose(actual: TF(outputViaBF16), expected: TF(output), relTolerance: 1e-2))
  }

  func testBroadcastDims() throws {
    for useReducedPrecision in [false, true] {
      for i in -3..<4 {
        var x = X10Tensor(shape: [2, 2, 2], scalars: [1, 5, 43, 24, 64, 32, 3, 4])
        let expected = TF(x).expandingShape(at: i)
        if useReducedPrecision {
          x = x.toReducedPrecision
        }
        var actual = x.expandingShape(at: i)
        if useReducedPrecision {
          XCTAssert(actual.isReducedPrecision)
          actual = actual.toFullPrecision
        }
        XCTAssert(!actual.isReducedPrecision)
        XCTAssertEqual(TF(actual), expected)
      }
    }
  }

  func testBroadcastTo() throws {
    var x = X10Tensor(shape: [2, 1, 3], scalars: [1, 5, 43, 24, 64, 32])
    let expected = TF(x).broadcasted(to: [9, 2, 8, 3])
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = x.broadcasted(to: [9, 2, 8, 3])
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssertEqual(TF(actual), expected)
    }
  }

  func testBroadcastGradientArgs() throws {
    func testBroadcastGradArgs(
      _ a: [Int], _ b: [Int],
      file: StaticString = #file, line: UInt = #line
    ) {
      let at = X10Tensor_<Int64>(a.map(Int64.init))
      let bt = X10Tensor_<Int64>(b.map(Int64.init))
      let actual = _Raw.broadcastGradientArgs(s0: at, s1: bt)
      let expected = _Raw.broadcastGradientArgs(s0: TF(at), s1: TF(bt))
      XCTAssertEqual(TF(actual.r0), expected.r0, file: file, line: line)
      XCTAssertEqual(TF(actual.r1), expected.r1, file: file, line: line)
    }
    testBroadcastGradArgs([8, 2, 1], [1, 2, 1])
    testBroadcastGradArgs([], [1, 2, 1])
    testBroadcastGradArgs([5], [23, 9, 1])
  }

  func testCast() throws {
    var x = X10Tensor(shape: [2], scalars: [1, 2])
    let expected = TFTensor_<Double>(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      let actual = TF(X10Tensor_<Double>(x))
      XCTAssertEqual(actual, expected)
    }
  }

  func testCeil() throws {
    var x = X10Tensor.rand([3, 2]) * 8
    let expected = ceil(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = ceil(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssert(
        allClose(actual: TF(actual), expected: expected, absTolerance: 1e-7))
    }
  }

  func testClipByValue() throws {
    let dims = [30, 20]
    let low: Float = 0.2
    let high: Float = 0.7
    for useReducedPrecision in [false, true] {
      for scalarClipValues in [false, true] {
        var x = X10Tensor.rand(dims)
        var clipValueMin =
          scalarClipValues
          ? X10Tensor(low) : X10Tensor(repeating: low, shape: TensorShape(dims))
        var clipValueMax =
          scalarClipValues
          ? X10Tensor(high) : X10Tensor(repeating: high, shape: TensorShape(dims))
        let expected = TF(x).clipped(min: TF(clipValueMin), max: TF(clipValueMax))
        if useReducedPrecision {
          x = x.toReducedPrecision
          clipValueMin = clipValueMin.toReducedPrecision
          clipValueMax = clipValueMax.toReducedPrecision
        }
        var actual = x.clipped(min: clipValueMin, max: clipValueMax)
        if useReducedPrecision {
          XCTAssert(actual.isReducedPrecision)
          actual = actual.toFullPrecision
        }
        XCTAssert(!actual.isReducedPrecision)
        let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
        XCTAssert(
          allClose(
            actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-7))
      }
    }
  }

  func testConcat() throws {
    var xs = [[3, 1, 2], [3, 8, 2], [3, 4, 2]].map { X10Tensor.rand($0) }
    let expected = TFTensor(concatenating: xs.map(TF), alongAxis: 1)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        xs = xs.toReducedPrecision
      }
      var actual = X10Tensor(concatenating: xs, alongAxis: 1)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(allClose(actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testConv2D() throws {
    let inChannels = 4
    let outChannels = 8
    let kernelSize = 5
    let inputSize = 14
    let batch = 2
    for useReducedPrecision in [false, true] {
      for stride in 1..<4 {
        for dilation in 1..<3 {
          for padSame in [false, true] {
            var input = X10Tensor.rand([batch, inputSize, inputSize, inChannels])
            var filter = X10Tensor.rand([kernelSize, kernelSize, inChannels, outChannels])
            let expected = conv2D(
              TF(input), filter: TF(filter), strides: (1, stride, stride, 1),
              padding: padSame ? Padding.same : Padding.valid, dilations: (1, dilation, dilation, 1)
            )
            if useReducedPrecision {
              input = input.toReducedPrecision
              filter = filter.toReducedPrecision
            }
            var actual =
              conv2D(
                input, filter: filter, strides: (1, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid,
                dilations: (1, dilation, dilation, 1)
              )
            if useReducedPrecision {
              XCTAssert(actual.isReducedPrecision)
              actual = actual.toFullPrecision
            }
            XCTAssert(!actual.isReducedPrecision)
            let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
            XCTAssert(
              allClose(
                actual: TF(actual), expected: expected, relTolerance: relTolerance,
                absTolerance: 1e-4))
          }
        }
      }
    }
  }

  func testConv2DGrad() throws {
    let inChannels = 4
    let outChannels = 8
    let kernelSize = 5
    let inputSize = 14
    let batch = 2
    // Dilated convolution gradients aren't supported by classic (no XLA) TF.
    let dilation = 1
    for useReducedPrecision in [false, true] {
      for stride in 1..<4 {
        for padSame in [false, true] {
          var input = X10Tensor.rand([batch, inputSize, inputSize, inChannels])
          var filter = X10Tensor.rand([kernelSize, kernelSize, inChannels, outChannels])
          let outShape = conv2D(
            TF(input), filter: TF(filter), strides: (1, stride, stride, 1),
            padding: padSame ? Padding.same : Padding.valid, dilations: (1, dilation, dilation, 1)
          )
          .shape
          var outGrad = X10Tensor.rand(outShape.dimensions)
          if useReducedPrecision {
            input = input.toReducedPrecision
            filter = filter.toReducedPrecision
            outGrad = outGrad.toReducedPrecision
          }
          let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
          assertEqualBinaryOperationGradients(
            {
              (_ input: X10Tensor, _ filter: X10Tensor) -> X10Tensor in
              conv2D(
                input, filter: filter, strides: (1, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid,
                dilations: (1, dilation, dilation, 1)
              )
            },
            { (_ input: TFTensor, _ filter: TFTensor) -> TFTensor in
              conv2D(
                input, filter: filter, strides: (1, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid,
                dilations: (1, dilation, dilation, 1)
              )
            }, input, filter, outGrad, relTolerance: relTolerance, absTolerance: 1e-4)
        }
      }
    }
  }

  func testConv3DGrad() throws {
    let inChannels = 4
    let outChannels = 8
    let kernelSize = 5
    let inputSize = 14
    let batch = 2
    for useReducedPrecision in [false, true] {
      for stride in 1..<4 {
        for padSame in [false, true] {
          var input = X10Tensor.rand([batch, inputSize, inputSize, inputSize, inChannels])
          var filter = X10Tensor.rand([kernelSize, kernelSize, kernelSize, inChannels, outChannels])
          let outShape = conv3D(
            TF(input), filter: TF(filter), strides: (1, stride, stride, stride, 1),
            padding: padSame ? Padding.same : Padding.valid
          )
          .shape
          var outGrad = X10Tensor.rand(outShape.dimensions)
          if useReducedPrecision {
            input = input.toReducedPrecision
            filter = filter.toReducedPrecision
            outGrad = outGrad.toReducedPrecision
          }
          let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
          assertEqualBinaryOperationGradients(
            {
              (_ input: X10Tensor, _ filter: X10Tensor) -> X10Tensor in
              conv3D(
                input, filter: filter, strides: (1, stride, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid
              )
            },
            { (_ input: TFTensor, _ filter: TFTensor) -> TFTensor in
              conv3D(
                input, filter: filter, strides: (1, stride, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid
              )
            }, input, filter, outGrad, relTolerance: relTolerance)
        }
      }
    }
  }

  func testCos() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = cos(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = cos(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
    }
  }

  func testCosh() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = cosh(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = cosh(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
    }
  }

  func testCumprod() throws {
    for useReducedPrecision in [false, true] {
      for (exclusive, reverse) in [(false, false), (true, false), (false, true), (true, true)] {
        var x = X10Tensor.rand([2, 4, 2])
        if useReducedPrecision {
          x = x.toReducedPrecision
        }
        var actual = x.cumulativeProduct(alongAxis: 1, exclusive: exclusive, reverse: reverse)
        if useReducedPrecision {
          XCTAssert(actual.isReducedPrecision)
          actual = actual.toFullPrecision
        }
        XCTAssert(!actual.isReducedPrecision)
        let expected = TF(x).cumulativeProduct(alongAxis: 1, exclusive: exclusive, reverse: reverse)
        let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
        XCTAssert(
          allClose(
            actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
      }
    }
  }

  func testCumsum() throws {
    for useReducedPrecision in [false, true] {
      for (exclusive, reverse) in [(false, false), (true, false), (false, true), (true, true)] {
        var x = X10Tensor.rand([2, 4, 2])
        if useReducedPrecision {
          x = x.toReducedPrecision
        }
        var actual = x.cumulativeSum(alongAxis: 1, exclusive: exclusive, reverse: reverse)
        if useReducedPrecision {
          XCTAssert(actual.isReducedPrecision)
          actual = actual.toFullPrecision
        }
        XCTAssert(!actual.isReducedPrecision)
        let expected = TF(x).cumulativeSum(alongAxis: 1, exclusive: exclusive, reverse: reverse)
        let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
        XCTAssert(
          allClose(
            actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
      }
    }
  }

  func testDepthwiseConv2DGrad() throws {
    let inChannels = 4
    let channelMultiplier = 2
    let kernelSize = 5
    let inputSize = 14
    let batch = 2
    for useReducedPrecision in [false, true] {
      for stride in 1..<4 {
        for padSame in [false, true] {
          var input = X10Tensor.rand([batch, inputSize, inputSize, inChannels])
          var filter = X10Tensor.rand([kernelSize, kernelSize, inChannels, channelMultiplier])
          let outShape = depthwiseConv2D(
            TF(input), filter: TF(filter), strides: (1, stride, stride, 1),
            padding: padSame ? Padding.same : Padding.valid
          )
          .shape
          var outGrad = X10Tensor.rand(outShape.dimensions)
          if useReducedPrecision {
            input = input.toReducedPrecision
            filter = filter.toReducedPrecision
            outGrad = outGrad.toReducedPrecision
          }
          let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
          assertEqualBinaryOperationGradients(
            {
              (_ input: X10Tensor, _ filter: X10Tensor) -> X10Tensor in
              depthwiseConv2D(
                input, filter: filter, strides: (1, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid
              )
            },
            { (_ input: TFTensor, _ filter: TFTensor) -> TFTensor in
              depthwiseConv2D(
                input, filter: filter, strides: (1, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid
              )
            }, input, filter, outGrad, relTolerance: relTolerance, absTolerance: 1e-4)
        }
      }
    }
  }

  func testDiv() throws {
    for useReducedPrecision in [false, true] {
      var x = X10Tensor(shape: [2], scalars: [1, 2])
      var y = X10Tensor(shape: [2], scalars: [7, 19])
      let expected = TF(x) / TF(y)
      if useReducedPrecision {
        x = x.toReducedPrecision
        y = y.toReducedPrecision
      }
      var actual = x / y
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(allClose(actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testDiagonalPart() throws {
    // TODO(b/146675105): Implement `_Raw.matrixDiagPart`, used by `Tensor.diagonalPart`.
    #if false
      do {
        let x = X10Tensor.rand([5, 5])
        let actual = TF(x.diagonalPart())
        let expected = TF(x).diagonalPart()
        XCTAssertEqual(actual, expected)
      }
      do {
        let x = X10Tensor.rand([5, 3, 5, 3])
        let actual = TF(x.diagonalPart())
        let expected = TF(x).diagonalPart()
        XCTAssertEqual(actual, expected)
      }
    #endif
  }

  func testElu() throws {
    var x = X10Tensor(shape: [6], scalars: [-1.0, -0.5, 0.5, 3.0, 4.0, 7.0])
    var outGrad = X10Tensor.rand(x.shape.dimensions)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        outGrad = outGrad.toReducedPrecision
      }
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      assertEqualUnaryOperationGradients(
        { elu($0) }, { elu($0) }, x, outGrad, relTolerance: relTolerance, absTolerance: 1e-5)
    }
  }

  func testEqual() throws {
    var x = X10Tensor(shape: [4], scalars: [1, 22, 3, 5])
    var y = X10Tensor(shape: [4], scalars: [7, 19, 3, 5])
    let expected = TF(x) .== TF(y)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        y = y.toReducedPrecision
      }
      let actual = TF(x .== y)
      XCTAssertEqual(actual, expected)
    }
  }

  func testExp() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = exp(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = exp(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
    }
  }

  func testExpm1() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = expm1(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = expm1(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
    }
  }

  func testFill() throws {
    let actual = TF(X10Tensor(repeating: 1.0, shape: [3, 4]))
    let expected = TFTensor(repeating: 1.0, shape: [3, 4])
    XCTAssertEqual(actual, expected)
  }

  func testFloor() throws {
    var x = X10Tensor.rand([3, 2]) * 10
    let expected = floor(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = floor(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssert(
        allClose(actual: TF(actual), expected: expected, absTolerance: 1e-7))
    }
  }

  func testGather() throws {
    let size = 4
    var params = X10Tensor.rand([size, size])
    let indices: X10Tensor_<Int32> = X10Tensor.randint(0, size, [5, 2, 3])
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        params = params.toReducedPrecision
      }
      var actual = _Raw.gather(params: params, indices: indices)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let expected = _Raw.gather(params: TF(params), indices: TF(indices))
      XCTAssertEqual(TF(actual), expected)
    }
  }

  func testGatherV2() throws {
    let size = 4
    let indices: X10Tensor_<Int32> = X10Tensor.randint(0, size, [5, 2, 3])
    for useReducedPrecision in [false, true] {
      var params = X10Tensor.rand([size, size])
      if useReducedPrecision {
        params = params.toReducedPrecision
      }
      for axis in -params.rank..<params.rank {
        let axisDim = X10Tensor_<Int32>(shape: [], scalars: [Int32(axis)])
        var actual = _Raw.gatherV2(params: params, indices: indices, axis: axisDim)
        if useReducedPrecision {
          XCTAssert(actual.isReducedPrecision)
          actual = actual.toFullPrecision
        }
        XCTAssert(!actual.isReducedPrecision)
        let expected = _Raw.gatherV2(params: TF(params), indices: TF(indices), axis: TF(axisDim))
        XCTAssertEqual(TF(actual), expected)
      }
    }
  }

  func testGreater() throws {
    let originalDims = [3, 2, 4]
    for useReducedPrecision in [false, true] {
      for broadcastDim in 0...originalDims.count {
        var dims = originalDims
        if broadcastDim < originalDims.count {
          dims[broadcastDim] = 1
        }
        var x = X10Tensor.rand(originalDims)
        var y = X10Tensor.rand(dims)
        if useReducedPrecision {
          x = x.toReducedPrecision
          y = y.toReducedPrecision
        }
        let actualXY = TF(x .> y)
        let expectedXY = TF(x) .> TF(y)
        XCTAssertEqual(actualXY, expectedXY)
        let actualYX = TF(y .> x)
        let expectedYX = TF(y) .> TF(x)
        XCTAssertEqual(actualYX, expectedYX)
      }
    }
  }

  func testGreaterEqual() throws {
    let originalDims = [3, 2, 4]
    for useReducedPrecision in [false, true] {
      for broadcastDim in 0...originalDims.count {
        var dims = originalDims
        if broadcastDim < originalDims.count {
          dims[broadcastDim] = 1
        }
        var x = X10Tensor.rand(originalDims)
        var y = X10Tensor.rand(dims)
        if useReducedPrecision {
          x = x.toReducedPrecision
          y = y.toReducedPrecision
        }
        let actualXY = TF(x .>= y)
        let expectedXY = TF(x) .>= TF(y)
        XCTAssertEqual(actualXY, expectedXY)
        let actualYX = TF(y .>= x)
        let expectedYX = TF(y) .>= TF(x)
        XCTAssertEqual(actualYX, expectedYX)
        let onesLhs = X10Tensor(onesLike: x)
        let onesRhs = X10Tensor(onesLike: y)
        let actualOnes = TF(onesLhs .>= onesRhs)
        let expectedOnes = TF(onesLhs) .>= TF(onesRhs)
        XCTAssertEqual(actualOnes, expectedOnes)
      }
    }
  }

  func testIndexAdvanced() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let slice2DExpected = TF(tensor3D)[1..<3, 0, 3...]
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
      }
      var slice2DActual = tensor3D[1..<3, 0, 3...]
      if useReducedPrecision {
        XCTAssert(slice2DActual.isReducedPrecision)
        slice2DActual = slice2DActual.toFullPrecision
      }
      XCTAssert(!slice2DActual.isReducedPrecision)
      XCTAssertEqual(TF(slice2DActual), slice2DExpected)
    }
  }

  func testIndexAdvancedGrad() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    for useReducedPrecision in [false, true] {
      let slice2DShape = tensor3D[1..<3, 0, 3...].shape
      var outGrad = X10Tensor.rand(slice2DShape.dimensions)
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
        outGrad = outGrad.toReducedPrecision
      }
      assertEqualUnaryOperationGradients(
        { $0[1..<3, 0, 3...] }, { $0[1..<3, 0, 3...] }, tensor3D,
        outGrad)
    }
  }

  func testIndexElement() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let element2DExpected = TF(tensor3D)[2]
    let element1DExpected = TF(tensor3D)[1][3]
    let element0DExpected = TF(tensor3D)[2][0][3]
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
      }
      var element2DActual = tensor3D[2]
      var element1DActual = tensor3D[1][3]
      var element0DActual = tensor3D[2][0][3]
      if useReducedPrecision {
        XCTAssert(element2DActual.isReducedPrecision)
        XCTAssert(element1DActual.isReducedPrecision)
        XCTAssert(element0DActual.isReducedPrecision)
        element2DActual = element2DActual.toFullPrecision
        element1DActual = element1DActual.toFullPrecision
        element0DActual = element0DActual.toFullPrecision
      }
      XCTAssert(!element2DActual.isReducedPrecision)
      XCTAssert(!element1DActual.isReducedPrecision)
      XCTAssert(!element0DActual.isReducedPrecision)
      XCTAssertEqual(TF(element2DActual), element2DExpected)
      XCTAssertEqual(TF(element1DActual), element1DExpected)
      XCTAssertEqual(TF(element0DActual), element0DExpected)
    }
  }

  func testIndexElementAssignment() throws {
    for useReducedPrecision in [false, true] {
      var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
      var tfTensor3D = TF(tensor3D)
      tensor3D[2] = X10Tensor(
        shape: [4, 5], scalars: Array(stride(from: 20.0, to: 40, by: 1)))
      tfTensor3D[2] = TFTensor(shape: [4, 5], scalars: Array(stride(from: 20.0, to: 40, by: 1)))
      let element2DExpected = tfTensor3D[2]
      let element1DExpected = tfTensor3D[1][3]
      let element0DExpected = tfTensor3D[2][0][3]
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
      }
      var element2DActual = tensor3D[2]
      var element1DActual = tensor3D[1][3]
      var element0DActual = tensor3D[2][0][3]
      if useReducedPrecision {
        XCTAssert(element2DActual.isReducedPrecision)
        XCTAssert(element1DActual.isReducedPrecision)
        XCTAssert(element0DActual.isReducedPrecision)
        element2DActual = element2DActual.toFullPrecision
        element1DActual = element1DActual.toFullPrecision
        element0DActual = element0DActual.toFullPrecision
      }
      XCTAssert(!element2DActual.isReducedPrecision)
      XCTAssert(!element1DActual.isReducedPrecision)
      XCTAssert(!element0DActual.isReducedPrecision)
      XCTAssertEqual(TF(element2DActual), element2DExpected)
      XCTAssertEqual(TF(element1DActual), element1DExpected)
      XCTAssertEqual(TF(element0DActual), element0DExpected)
    }
  }

  func testIndexElementGrad() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let element2DShape = tensor3D[2].shape
    let element1DShape = tensor3D[1][3].shape
    let element0DShape = tensor3D[2][0][3].shape
    var outGrad2D = X10Tensor.rand(element2DShape.dimensions)
    var outGrad1D = X10Tensor.rand(element1DShape.dimensions)
    var outGrad0D = X10Tensor.rand(element0DShape.dimensions)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
        outGrad2D = outGrad2D.toReducedPrecision
        outGrad1D = outGrad1D.toReducedPrecision
        outGrad0D = outGrad0D.toReducedPrecision
      }
      assertEqualUnaryOperationGradients(
        { $0[2] }, { $0[2] }, tensor3D, outGrad2D)
      assertEqualUnaryOperationGradients(
        { $0[1][3] }, { $0[1][3] }, tensor3D, outGrad1D)
      assertEqualUnaryOperationGradients(
        { $0[2][0][3] }, { $0[2][0][3] }, tensor3D, outGrad0D)
    }
  }

  func testIndexEllipsis() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let slice3DExpected = TF(tensor3D)[2..., TF(TensorRange.ellipsis)]
    let slice2DExpected = TF(tensor3D)[1][0..<2]
    let slice1DExpected = TF(tensor3D)[0][0][3..<5]
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
      }
      var slice3DActual = tensor3D[2..., TensorRange.ellipsis]
      var slice2DActual = tensor3D[1][0..<2]
      var slice1DActual = tensor3D[0][0][3..<5]
      if useReducedPrecision {
        XCTAssert(slice3DActual.isReducedPrecision)
        XCTAssert(slice2DActual.isReducedPrecision)
        XCTAssert(slice1DActual.isReducedPrecision)
        slice3DActual = slice3DActual.toFullPrecision
        slice2DActual = slice2DActual.toFullPrecision
        slice1DActual = slice1DActual.toFullPrecision
      }
      XCTAssert(!slice3DActual.isReducedPrecision)
      XCTAssert(!slice2DActual.isReducedPrecision)
      XCTAssert(!slice1DActual.isReducedPrecision)
      XCTAssertEqual(TF(slice3DActual), slice3DExpected)
      XCTAssertEqual(TF(slice2DActual), slice2DExpected)
      XCTAssertEqual(TF(slice1DActual), slice1DExpected)
    }
  }

  func testIndexEllipsisGrad() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let slice3DShape = tensor3D[2..., TensorRange.ellipsis].shape
    let slice2DShape = tensor3D[1][0..<2].shape
    let slice1DShape = tensor3D[0][0][3..<5].shape
    var outGrad3D = X10Tensor.rand(slice3DShape.dimensions)
    var outGrad2D = X10Tensor.rand(slice2DShape.dimensions)
    var outGrad1D = X10Tensor.rand(slice1DShape.dimensions)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
        outGrad3D = outGrad3D.toReducedPrecision
        outGrad2D = outGrad2D.toReducedPrecision
        outGrad1D = outGrad1D.toReducedPrecision
      }
      assertEqualUnaryOperationGradients(
        { $0[2..., TensorRange.ellipsis] }, { $0[2..., TF(TensorRange.ellipsis)] }, tensor3D,
        outGrad3D)
      assertEqualUnaryOperationGradients(
        { $0[1][0..<2] }, { $0[1][0..<2] }, tensor3D,
        outGrad2D)
      assertEqualUnaryOperationGradients(
        { $0[0][0][3..<5] }, { $0[0][0][3..<5] }, tensor3D, outGrad1D)
    }
  }

  func testIndexNestedElement() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let element1DExpected = TF(tensor3D)[1, 3]
    let element0DExpected = TF(tensor3D)[2, 0, 3]
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
      }
      var element1DActual = tensor3D[1, 3]
      var element0DActual = tensor3D[2, 0, 3]
      if useReducedPrecision {
        XCTAssert(element1DActual.isReducedPrecision)
        XCTAssert(element0DActual.isReducedPrecision)
        element1DActual = element1DActual.toFullPrecision
        element0DActual = element0DActual.toFullPrecision
      }
      XCTAssert(!element1DActual.isReducedPrecision)
      XCTAssert(!element0DActual.isReducedPrecision)
      XCTAssertEqual(TF(element1DActual), element1DExpected)
      XCTAssertEqual(TF(element0DActual), element0DExpected)
    }
  }

  func testIndexNestedElementGrad() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let element1DShape = tensor3D[1, 3].shape
    let element0DShape = tensor3D[2, 0, 3].shape
    var outGrad1D = X10Tensor.rand(element1DShape.dimensions)
    var outGrad0D = X10Tensor.rand(element0DShape.dimensions)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
        outGrad1D = outGrad1D.toReducedPrecision
        outGrad0D = outGrad0D.toReducedPrecision
      }
      assertEqualUnaryOperationGradients(
        { $0[1, 3] }, { $0[1, 3] }, tensor3D, outGrad1D)
      assertEqualUnaryOperationGradients(
        { $0[2, 0, 3] }, { $0[2, 0, 3] }, tensor3D, outGrad0D)
    }
  }

  func testIndexNewAxis() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let newAxis = TensorRange.newAxis
    let ellipsis = TensorRange.ellipsis
    let slice3DExpected = TF(tensor3D)[2..., TF(newAxis), TF(ellipsis)]
    let slice2DExpected = TF(tensor3D)[1, TF(newAxis)][0..<1, 0..<2]
    let slice1DExpected = TF(tensor3D)[0][TF(newAxis), 0][0..<1, 3..<5, TF(newAxis)]
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
      }
      var slice3DActual = tensor3D[2..., newAxis, ellipsis]
      var slice2DActual = tensor3D[1, newAxis][0..<1, 0..<2]
      var slice1DActual = tensor3D[0][newAxis, 0][0..<1, 3..<5, newAxis]
      if useReducedPrecision {
        XCTAssert(slice3DActual.isReducedPrecision)
        XCTAssert(slice2DActual.isReducedPrecision)
        XCTAssert(slice1DActual.isReducedPrecision)
        slice3DActual = slice3DActual.toFullPrecision
        slice2DActual = slice2DActual.toFullPrecision
        slice1DActual = slice1DActual.toFullPrecision
      }
      XCTAssert(!slice3DActual.isReducedPrecision)
      XCTAssert(!slice2DActual.isReducedPrecision)
      XCTAssert(!slice1DActual.isReducedPrecision)
      XCTAssertEqual(TF(slice3DActual), slice3DExpected)
      XCTAssertEqual(TF(slice2DActual), slice2DExpected)
      XCTAssertEqual(TF(slice1DActual), slice1DExpected)
    }
  }

  func testIndexNewAxisGrad() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let newAxis = TensorRange.newAxis
    let ellipsis = TensorRange.ellipsis
    let slice3DShape = tensor3D[2..., newAxis, ellipsis].shape
    let slice2DShape = tensor3D[1, newAxis][0..<1, 0..<2].shape
    let slice1DShape = TF(tensor3D[0][newAxis, 0][0..<1, 3..<5, newAxis]).shape
    var outGrad3D = X10Tensor.rand(slice3DShape.dimensions)
    var outGrad2D = X10Tensor.rand(slice2DShape.dimensions)
    var outGrad1D = X10Tensor.rand(slice1DShape.dimensions)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
        outGrad3D = outGrad3D.toReducedPrecision
        outGrad2D = outGrad2D.toReducedPrecision
        outGrad1D = outGrad1D.toReducedPrecision
      }
      assertEqualUnaryOperationGradients(
        { $0[2..., newAxis, ellipsis] }, { $0[2..., TF(newAxis), TF(ellipsis)] }, tensor3D,
        outGrad3D)
      assertEqualUnaryOperationGradients(
        { $0[1, newAxis][0..<1, 0..<2] }, { $0[1, TF(newAxis)][0..<1, 0..<2] }, tensor3D,
        outGrad2D)
      assertEqualUnaryOperationGradients(
        { $0[0][newAxis, 0][0..<1, 3..<5, newAxis] },
        { $0[0][TF(newAxis), 0][0..<1, 3..<5, TF(newAxis)] }, tensor3D,
        outGrad1D)
    }
  }

  func testIndexSlice() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let slice3DExpected = TF(tensor3D)[2...]
    let slice2DExpected = TF(tensor3D)[1][0..<2]
    let slice1DExpected = TF(tensor3D)[0][0][3..<5]
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
      }
      var slice3DActual = tensor3D[2...]
      var slice2DActual = tensor3D[1][0..<2]
      var slice1DActual = tensor3D[0][0][3..<5]
      if useReducedPrecision {
        XCTAssert(slice3DActual.isReducedPrecision)
        XCTAssert(slice2DActual.isReducedPrecision)
        XCTAssert(slice1DActual.isReducedPrecision)
        slice3DActual = slice3DActual.toFullPrecision
        slice2DActual = slice2DActual.toFullPrecision
        slice1DActual = slice1DActual.toFullPrecision
      }
      XCTAssert(!slice3DActual.isReducedPrecision)
      XCTAssert(!slice2DActual.isReducedPrecision)
      XCTAssert(!slice1DActual.isReducedPrecision)
      XCTAssertEqual(TF(slice3DActual), slice3DExpected)
      XCTAssertEqual(TF(slice2DActual), slice2DExpected)
      XCTAssertEqual(TF(slice1DActual), slice1DExpected)
    }
  }

  func testIndexSliceAssignment() throws {
    for useReducedPrecision in [false, true] {
      var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
      var tfTensor3D = TF(tensor3D)
      tensor3D[2, 0..<5, 0..<6] = X10Tensor(
        shape: [4, 5], scalars: Array(stride(from: 20.0, to: 40, by: 1)))
      tfTensor3D[2, 0..<5, 0..<6] = TFTensor(
        shape: [4, 5], scalars: Array(stride(from: 20.0, to: 40, by: 1)))
      let slice3DExpected = tfTensor3D[2...]
      let slice2DExpected = tfTensor3D[1][0..<2]
      let slice1DExpected = tfTensor3D[0][0][3..<5]
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
      }
      var slice3DActual = tensor3D[2...]
      var slice2DActual = tensor3D[1][0..<2]
      var slice1DActual = tensor3D[0][0][3..<5]
      if useReducedPrecision {
        XCTAssert(slice3DActual.isReducedPrecision)
        XCTAssert(slice2DActual.isReducedPrecision)
        XCTAssert(slice1DActual.isReducedPrecision)
        slice3DActual = slice3DActual.toFullPrecision
        slice2DActual = slice2DActual.toFullPrecision
        slice1DActual = slice1DActual.toFullPrecision
      }
      XCTAssert(!slice3DActual.isReducedPrecision)
      XCTAssert(!slice2DActual.isReducedPrecision)
      XCTAssert(!slice1DActual.isReducedPrecision)
      XCTAssertEqual(TF(slice3DActual), slice3DExpected)
      XCTAssertEqual(TF(slice2DActual), slice2DExpected)
      XCTAssertEqual(TF(slice1DActual), slice1DExpected)
    }
  }

  func testIndexSliceGrad() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let slice3DShape = tensor3D[2...].shape
    let slice2DShape = tensor3D[1][0..<2].shape
    let slice1DShape = tensor3D[0][0][3..<5].shape
    var outGrad3D = X10Tensor.rand(slice3DShape.dimensions)
    var outGrad2D = X10Tensor.rand(slice2DShape.dimensions)
    var outGrad1D = X10Tensor.rand(slice1DShape.dimensions)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
        outGrad3D = outGrad3D.toReducedPrecision
        outGrad2D = outGrad2D.toReducedPrecision
        outGrad1D = outGrad1D.toReducedPrecision
      }
      assertEqualUnaryOperationGradients(
        { $0[2...] }, { $0[2...] }, tensor3D,
        outGrad3D)
      assertEqualUnaryOperationGradients(
        { $0[1][0..<2] }, { $0[1][0..<2] }, tensor3D,
        outGrad2D)
      assertEqualUnaryOperationGradients(
        { $0[0][0][3..<5] }, { $0[0][0][3..<5] }, tensor3D,
        outGrad1D)
    }
  }

  func testIndexSqueezeAxis() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let newAxis = TensorRange.newAxis
    let ellipsis = TensorRange.ellipsis
    let squeezeAxis = TensorRange.squeezeAxis
    let slice3DExpected = TF(tensor3D)[2..., TF(newAxis), TF(ellipsis)][
      TF(squeezeAxis), TF(squeezeAxis)]
    let slice2DExpected = TF(tensor3D)[1, TF(newAxis)][TF(squeezeAxis), 0..<2]
    let slice1DExpected = TF(tensor3D)[0..<1, 0, 3..<5, TF(newAxis)][
      TF(squeezeAxis), TF(ellipsis), TF(squeezeAxis)]
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
      }
      var slice3DActual = tensor3D[2..., newAxis, ellipsis][squeezeAxis, squeezeAxis]
      var slice2DActual = tensor3D[1, newAxis][squeezeAxis, 0..<2]
      var slice1DActual = tensor3D[0..<1, 0, 3..<5, newAxis][squeezeAxis, ellipsis, squeezeAxis]
      if useReducedPrecision {
        XCTAssert(slice3DActual.isReducedPrecision)
        XCTAssert(slice2DActual.isReducedPrecision)
        XCTAssert(slice1DActual.isReducedPrecision)
        slice3DActual = slice3DActual.toFullPrecision
        slice2DActual = slice2DActual.toFullPrecision
        slice1DActual = slice1DActual.toFullPrecision
      }
      XCTAssert(!slice3DActual.isReducedPrecision)
      XCTAssert(!slice2DActual.isReducedPrecision)
      XCTAssert(!slice1DActual.isReducedPrecision)
      XCTAssertEqual(TF(slice3DActual), slice3DExpected)
      XCTAssertEqual(TF(slice2DActual), slice2DExpected)
      XCTAssertEqual(TF(slice1DActual), slice1DExpected)
    }
  }

  func testIndexSqueezeAxisGrad() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let newAxis = TensorRange.newAxis
    let ellipsis = TensorRange.ellipsis
    let squeezeAxis = TensorRange.squeezeAxis
    let slice3DShape = tensor3D[2..., newAxis, ellipsis][squeezeAxis, squeezeAxis].shape
    let slice2DShape = tensor3D[1, newAxis][squeezeAxis, 0..<2].shape
    let slice1DShape = tensor3D[0..<1, 0, 3..<5, newAxis][squeezeAxis, ellipsis, squeezeAxis].shape
    var outGrad3D = X10Tensor.rand(slice3DShape.dimensions)
    var outGrad2D = X10Tensor.rand(slice2DShape.dimensions)
    var outGrad1D = X10Tensor.rand(slice1DShape.dimensions)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
        outGrad3D = outGrad3D.toReducedPrecision
        outGrad2D = outGrad2D.toReducedPrecision
        outGrad1D = outGrad1D.toReducedPrecision
      }
      assertEqualUnaryOperationGradients(
        { $0[2..., newAxis, ellipsis][squeezeAxis, squeezeAxis] },
        { $0[2..., TF(newAxis), TF(ellipsis)][TF(squeezeAxis), TF(squeezeAxis)] },
        tensor3D, outGrad3D)
      assertEqualUnaryOperationGradients(
        { $0[1, newAxis][squeezeAxis, 0..<2] },
        { $0[1, TF(newAxis)][TF(squeezeAxis), 0..<2] },
        tensor3D, outGrad2D)
      assertEqualUnaryOperationGradients(
        { $0[0..<1, 0, 3..<5, newAxis][squeezeAxis, ellipsis, squeezeAxis] },
        { $0[0..<1, 0, 3..<5, TF(newAxis)][TF(squeezeAxis), TF(ellipsis), TF(squeezeAxis)] },
        tensor3D, outGrad1D)
    }
  }

  func testIndexStridedSlice() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let slice3DExpected = TF(tensor3D)[2...]
    let r1 = TensorRange.range(0..<3, stride: 2)
    let slice2DExpected = TF(tensor3D)[1][TF(r1)]
    let r2 = TensorRange.range(1..<5, stride: 2)
    let slice1DExpected = TF(tensor3D)[0][0][TF(r2)]
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
      }
      var slice3DActual = tensor3D[2...]
      var slice2DActual = tensor3D[1][r1]
      var slice1DActual = tensor3D[0][0][r2]
      if useReducedPrecision {
        XCTAssert(slice3DActual.isReducedPrecision)
        XCTAssert(slice2DActual.isReducedPrecision)
        XCTAssert(slice1DActual.isReducedPrecision)
        slice3DActual = slice3DActual.toFullPrecision
        slice2DActual = slice2DActual.toFullPrecision
        slice1DActual = slice1DActual.toFullPrecision
      }
      XCTAssert(!slice3DActual.isReducedPrecision)
      XCTAssert(!slice2DActual.isReducedPrecision)
      XCTAssert(!slice1DActual.isReducedPrecision)
      XCTAssertEqual(TF(slice3DActual), slice3DExpected)
      XCTAssertEqual(TF(slice2DActual), slice2DExpected)
      XCTAssertEqual(TF(slice1DActual), slice1DExpected)
    }
  }

  func testIndexStridedSliceGrad() throws {
    var tensor3D = X10Tensor(shape: [3, 4, 5], scalars: Array(stride(from: 0.0, to: 60, by: 1)))
    let slice3DShape = tensor3D[2...].shape
    let r1 = TensorRange.range(0..<3, stride: 2)
    let slice2DShape = tensor3D[1][r1].shape
    let r2 = TensorRange.range(1..<5, stride: 2)
    let slice1DShape = TF(tensor3D[0][0][r2]).shape
    var outGrad3D = X10Tensor.rand(slice3DShape.dimensions)
    var outGrad2D = X10Tensor.rand(slice2DShape.dimensions)
    var outGrad1D = X10Tensor.rand(slice1DShape.dimensions)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        tensor3D = tensor3D.toReducedPrecision
        outGrad3D = outGrad3D.toReducedPrecision
        outGrad2D = outGrad2D.toReducedPrecision
        outGrad1D = outGrad1D.toReducedPrecision
      }
      assertEqualUnaryOperationGradients(
        { $0[2...] }, { $0[2...] }, tensor3D, outGrad3D)
      assertEqualUnaryOperationGradients(
        { $0[1][r1] }, { $0[1][TF(r1)] }, tensor3D, outGrad2D)
      assertEqualUnaryOperationGradients(
        { $0[0][0][r2] }, { $0[0][0][TF(r2)] }, tensor3D, outGrad1D)
    }
  }

  func testInvertPermutation() throws {
    let scalars: [Int32] = [3, 4, 0, 2, 1]
    let input = X10Tensor_<Int32>(shape: [scalars.count], scalars: scalars)
    let actual = TF(_Raw.invertPermutation(input))
    let expected = _Raw.invertPermutation(TF(input))
    XCTAssertEqual(actual, expected)
  }

  func testIsFinite() throws {
    var x = X10Tensor(shape: [4], scalars: [Float.nan, Float.infinity, 0.5, 3.0])
    let expected = TF(x).isFinite
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      let actual = TF(x.isFinite)
      XCTAssertEqual(actual, expected)
    }
  }

  func testIsInfinite() throws {
    var x = X10Tensor(shape: [4], scalars: [Float.nan, Float.infinity, 0.5, 3.0])
    let expected = TF(x).isInfinite
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      let actual = TF(x.isInfinite)
      XCTAssertEqual(actual, expected)
    }
  }

  func testIsNaN() throws {
    var x = X10Tensor(shape: [4], scalars: [Float.nan, Float.infinity, 0.5, 3.0])
    let expected = TF(x).isNaN
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      let actual = TF(x.isNaN)
      XCTAssertEqual(actual, expected)
    }
  }

  func testLeakyRelu() throws {
    var x = X10Tensor(shape: [4], scalars: [-0.5, -0.25, 0.5, 3.0])
    let expected = leakyRelu(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = leakyRelu(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-3 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testLeakyReluGrad() throws {
    func leakyReluX10(_ arg: X10Tensor) -> X10Tensor {
      return leakyRelu(arg)
    }
    func leakyReluTF(_ arg: TFTensor) -> TFTensor {
      return leakyRelu(arg)
    }
    var x = X10Tensor(shape: [4], scalars: [-0.5, -0.25, 0.5, 3.0])
    var outGrad = X10Tensor(shape: [4], scalars: [1.5, 1.0, 2.5, 2.0])
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        outGrad = outGrad.toReducedPrecision
      }
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      assertEqualUnaryOperationGradients(
        leakyReluX10, leakyReluTF, x, outGrad, relTolerance: relTolerance)
    }
  }

  func testLess() throws {
    let originalDims = [3, 2, 4]
    for useReducedPrecision in [false, true] {
      for broadcastDim in 0...originalDims.count {
        var dims = originalDims
        if broadcastDim < originalDims.count {
          dims[broadcastDim] = 1
        }
        var x = X10Tensor.rand(originalDims)
        var y = X10Tensor.rand(dims)
        if useReducedPrecision {
          x = x.toReducedPrecision
          y = y.toReducedPrecision
        }
        let actualXY = TF(x .< y)
        let expectedXY = TF(x) .< TF(y)
        XCTAssertEqual(actualXY, expectedXY)
        let actualYX = TF(y .< x)
        let expectedYX = TF(y) .< TF(x)
        XCTAssertEqual(actualYX, expectedYX)
      }
    }
  }

  func testLessEqual() throws {
    let originalDims = [3, 2, 4]
    for useReducedPrecision in [false, true] {
      for broadcastDim in 0...originalDims.count {
        var dims = originalDims
        if broadcastDim < originalDims.count {
          dims[broadcastDim] = 1
        }
        var x = X10Tensor.rand(originalDims)
        var y = X10Tensor.rand(dims)
        if useReducedPrecision {
          x = x.toReducedPrecision
          y = y.toReducedPrecision
        }
        let actualXY = TF(x .<= y)
        let expectedXY = TF(x) .<= TF(y)
        XCTAssertEqual(actualXY, expectedXY)
        let actualYX = TF(y .<= x)
        let expectedYX = TF(y) .<= TF(x)
        XCTAssertEqual(actualYX, expectedYX)
        let onesLhs = X10Tensor(onesLike: x)
        let onesRhs = X10Tensor(onesLike: y)
        let actualOnes = TF(onesLhs .<= onesRhs)
        let expectedOnes = TF(onesLhs) .<= TF(onesRhs)
        XCTAssertEqual(actualOnes, expectedOnes)
      }
    }
  }

  func testLinSpace() throws {
    let device = x10_device.Device.default
    func testRanges(start: Float, stop: Float, num: Int32, useReducedPrecision: Bool) {
      var startTensor = X10Tensor(start)
      var stopTensor = X10Tensor(stop)
      if useReducedPrecision {
        startTensor = startTensor.toReducedPrecision
        stopTensor = stopTensor.toReducedPrecision
      }
      var x10 = _Raw.linSpace(
        start: startTensor, stop: stopTensor, num: X10Tensor_<Int32>(num),
        device: device)
      if useReducedPrecision {
        XCTAssert(x10.isReducedPrecision)
        x10 = x10.toFullPrecision
      }
      XCTAssert(!x10.isReducedPrecision)
      let tf = _Raw.linSpace(
        start: TFTensor(start), stop: TFTensor(stop), num: TFTensor_<Int32>(num))
      XCTAssert(allClose(actual: TF(x10), expected: tf, absTolerance: 1e-5))
    }
    for useReducedPrecision in [false, true] {
      testRanges(start: 0.0, stop: 5.0, num: 6, useReducedPrecision: useReducedPrecision)
      testRanges(start: 6.0, stop: 0.0, num: 3, useReducedPrecision: useReducedPrecision)
      testRanges(start: 0.0, stop: 0.0, num: 1, useReducedPrecision: useReducedPrecision)
      testRanges(start: 2.0, stop: 0.0, num: 1, useReducedPrecision: useReducedPrecision)
      testRanges(start: 20.0, stop: 0.0, num: 1, useReducedPrecision: useReducedPrecision)
    }
  }

  func testLog() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = log(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = log(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 5e-2 : 1e-4
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-4))
    }
  }

  func testLog1p() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = log1p(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = log1p(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-4
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-4))
    }
  }

  func testLogicalAnd() throws {
    let x: X10Tensor_<Bool> = X10Tensor.randint(0, 2, [2, 3, 4])
    let y: X10Tensor_<Bool> = X10Tensor.randint(0, 2, [2, 3, 4])
    let actual = TF(x.elementsLogicalOr(y))
    let expected = TF(x).elementsLogicalOr(TF(y))
    XCTAssertEqual(actual, expected)
  }

  func testLogicalNot() throws {
    let x: X10Tensor_<Bool> = X10Tensor.randint(0, 2, [2, 3, 4])
    let actual = TF(x.elementsLogicalNot())
    let expected = TF(x).elementsLogicalNot()
    XCTAssertEqual(actual, expected)
  }

  func testLogicalOr() throws {
    let x: X10Tensor_<Bool> = X10Tensor.randint(0, 2, [2, 3, 4])
    let y: X10Tensor_<Bool> = X10Tensor.randint(0, 2, [2, 3, 4])
    let actual = TF(x.elementsLogicalAnd(y))
    let expected = TF(x).elementsLogicalAnd(TF(y))
    XCTAssertEqual(actual, expected)
  }

  func testLogSoftmax() throws {
    var x = X10Tensor(
      shape: [2, 5], scalars: [0.98, 0.65, 0.832, 0.324, 0.3676, 0.777, 0.244, 0.950, 0.544, 0.445])
    let expected = logSoftmax(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = logSoftmax(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 2.0e-7))
    }
  }

  func testMatMul() throws {
    for useReducedPrecision in [false, true] {
      for (xShape, yShape, transposeX, transposeY) in [
        ([2, 2], [2, 2], false, false),
        ([2, 2], [2, 2], true, true),
        ([2, 2, 3, 8], [2, 9, 3], true, true),
        ([2, 2, 2, 2], [2, 2], true, true),
      ] {
        var x = X10Tensor.rand(xShape)
        var y = X10Tensor.rand(yShape)
        let expected = matmul(TF(x), transposed: transposeX, TF(y), transposed: transposeY)
        if useReducedPrecision {
          x = x.toReducedPrecision
          y = y.toReducedPrecision
        }
        var actual = matmul(x, transposed: transposeX, y, transposed: transposeY)
        if useReducedPrecision {
          XCTAssert(actual.isReducedPrecision)
          actual = actual.toFullPrecision
        }
        XCTAssert(!actual.isReducedPrecision)
        let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
        XCTAssert(
          allClose(
            actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-6))
      }
    }
  }

  func testMax() throws {
    for useReducedPrecision in [false, true] {
      for (shape, dims) in [([2, 3, 4], [1, 2]), ([3, 3, 2], [-2])] {
        let xFull = X10Tensor.rand(shape)
        do {
          var x = xFull
          if useReducedPrecision {
            x = x.toReducedPrecision
          }
          var actual = x.max(squeezingAxes: dims)
          if useReducedPrecision {
            XCTAssert(actual.isReducedPrecision)
            actual = actual.toFullPrecision
          }
          XCTAssert(!actual.isReducedPrecision)
          let expected = TF(x).max(squeezingAxes: dims)
          let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
          XCTAssert(
            allClose(
              actual: TF(actual), expected: expected, relTolerance: relTolerance))
        }
        do {
          var x = xFull
          if useReducedPrecision {
            x = x.toReducedPrecision
          }
          var actual = x.max(alongAxes: dims)
          if useReducedPrecision {
            XCTAssert(actual.isReducedPrecision)
            actual = actual.toFullPrecision
          }
          XCTAssert(!actual.isReducedPrecision)
          let expected = TF(x).max(alongAxes: dims)
          let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
          XCTAssert(
            allClose(
              actual: TF(actual), expected: expected, relTolerance: relTolerance))
        }
      }
    }
  }

  func testMaximum() throws {
    var x = X10Tensor.rand([4, 5])
    var y = X10Tensor.rand([4, 5])
    let expected = max(TF(x), TF(y))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        y = y.toReducedPrecision
      }
      var actual = max(x, y)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testMaxPool() throws {
    for useReducedPrecision in [false, true] {
      for stride in 1..<3 {
        for padSame in [false, true] {
          var x = X10Tensor.rand([4, 28, 28, 1])
          let expected = maxPool2D(
            TF(x), filterSize: (1, 2, 2, 1), strides: (1, stride, stride, 1),
            padding: padSame ? Padding.same : Padding.valid)
          if useReducedPrecision {
            x = x.toReducedPrecision
          }
          var actual = maxPool2D(
            x, filterSize: (1, 2, 2, 1), strides: (1, stride, stride, 1),
            padding: padSame ? Padding.same : Padding.valid)
          if useReducedPrecision {
            XCTAssert(actual.isReducedPrecision)
            actual = actual.toFullPrecision
          }
          XCTAssert(!actual.isReducedPrecision)
          let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
          XCTAssert(
            allClose(
              actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-7
            ))
        }
      }
    }
  }

  func testMaxPoolGrad() throws {
    for useReducedPrecision in [false, true] {
      for stride in 1..<3 {
        for padSame in [false, true] {
          var x = X10Tensor.rand([4, 28, 28, 1])
          let outShape = maxPool2D(
            TF(x), filterSize: (1, 2, 2, 1), strides: (1, stride, stride, 1),
            padding: padSame ? Padding.same : Padding.valid
          ).shape
          var outGrad = X10Tensor.rand(outShape.dimensions)
          if useReducedPrecision {
            x = x.toReducedPrecision
            outGrad = outGrad.toReducedPrecision
          }
          let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
          assertEqualUnaryOperationGradients(
            { (_ x: X10Tensor) -> X10Tensor in
              maxPool2D(
                x, filterSize: (1, 2, 2, 1), strides: (1, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid)
            },
            { (_ x: TFTensor) -> TFTensor in
              maxPool2D(
                x, filterSize: (1, 2, 2, 1), strides: (1, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid)
            }, x, outGrad, relTolerance: relTolerance, absTolerance: 1e-6)
        }
      }
    }
  }

  func testMaxPool3DGrad() throws {
    // TODO(asuhan): Figure out what's going on at higher sizes with bf16.
    let dims = [1, 6, 6, 6, 1]
    let elementCount = dims.reduce(1, *)
    let input = X10Tensor(
      shape: TensorShape(dims), scalars: Array(stride(from: 0.0, to: Float(elementCount), by: 1)))
    for useReducedPrecision in [false, true] {
      for stride in 1..<3 {
        for padSame in [false, true] {
          var x = input
          let outShape = maxPool3D(
            TF(x), filterSize: (1, 2, 2, 2, 1), strides: (1, stride, stride, stride, 1),
            padding: padSame ? Padding.same : Padding.valid
          ).shape
          var outGrad = X10Tensor(repeating: 1.0, shape: TensorShape(outShape.dimensions))
          if useReducedPrecision {
            x = x.toReducedPrecision
            outGrad = outGrad.toReducedPrecision
          }
          assertEqualUnaryOperationGradients(
            { (_ x: X10Tensor) -> X10Tensor in
              maxPool3D(
                x, filterSize: (1, 2, 2, 2, 1), strides: (1, stride, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid)
            },
            { (_ x: TFTensor) -> TFTensor in
              maxPool3D(
                x, filterSize: (1, 2, 2, 2, 1), strides: (1, stride, stride, stride, 1),
                padding: padSame ? Padding.same : Padding.valid)
            }, x, outGrad)
        }
      }
    }
  }

  func testMean() throws {
    var x = X10Tensor(shape: [3, 2], scalars: [1, 5, 43, 24, 64, 32])
    let expected = TF(x).mean(squeezingAxes: [0])
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = x.mean(squeezingAxes: [0])
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(allClose(actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testMeanBool() throws {
    let x: X10Tensor_<Bool> = X10Tensor_(shape: [4], scalars: [true, false, true, true])
    let expected = TFTensor(TF(x)).mean()
    let actual = TF(X10Tensor(x).mean())
    XCTAssertEqual(actual, expected)
  }

  func testMin() throws {
    for useReducedPrecision in [false, true] {
      for (shape, dims) in [([2, 3, 4], [1, 2]), ([3, 3, 2], [-2])] {
        let xFull = X10Tensor.rand(shape)
        do {
          var x = xFull
          if useReducedPrecision {
            x = x.toReducedPrecision
          }
          var actual = x.min(squeezingAxes: dims)
          if useReducedPrecision {
            XCTAssert(actual.isReducedPrecision)
            actual = actual.toFullPrecision
          }
          XCTAssert(!actual.isReducedPrecision)
          let expected = TF(x).min(squeezingAxes: dims)
          let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
          XCTAssert(
            allClose(
              actual: TF(actual), expected: expected, relTolerance: relTolerance))
        }
        do {
          var x = xFull
          if useReducedPrecision {
            x = x.toReducedPrecision
          }
          var actual = x.min(alongAxes: dims)
          if useReducedPrecision {
            XCTAssert(actual.isReducedPrecision)
            actual = actual.toFullPrecision
          }
          XCTAssert(!actual.isReducedPrecision)
          let expected = TF(x).min(alongAxes: dims)
          let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
          XCTAssert(
            allClose(
              actual: TF(actual), expected: expected, relTolerance: relTolerance))
        }
      }
    }
  }

  func testMinimum() throws {
    var x = X10Tensor.rand([4, 5])
    var y = X10Tensor.rand([4, 5])
    for useReducedPrecision in [false, true] {
      let expected = min(TF(x), TF(y))
      if useReducedPrecision {
        x = x.toReducedPrecision
        y = y.toReducedPrecision
      }
      var actual = min(x, y)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testMirrorPad() throws {
    let paddings = [(1, 2), (3, 4), (5, 6)]
    for useReducedPrecision in [false, true] {
      for reflect in [true, false] {
        var x = X10Tensor.rand([5, 7, 13])
        let expected = TF(x).padded(forSizes: paddings, mode: reflect ? .reflect : .symmetric)
        if useReducedPrecision {
          x = x.toReducedPrecision
        }
        var actual = x.padded(forSizes: paddings, mode: reflect ? .reflect : .symmetric)
        if useReducedPrecision {
          XCTAssert(actual.isReducedPrecision)
          actual = actual.toFullPrecision
        }
        XCTAssert(!actual.isReducedPrecision)
        let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
        XCTAssert(
          allClose(
            actual: TF(actual), expected: expected, relTolerance: relTolerance))
      }
    }
  }

  func testMirrorPadGrad() throws {
    let paddings = [(1, 2), (3, 4), (5, 6)]
    for useReducedPrecision in [false, true] {
      for reflect in [true, false] {
        var x = X10Tensor.rand([5, 7, 13])
        let outShape = TF(x).padded(forSizes: paddings, mode: reflect ? .reflect : .symmetric).shape
        var outGrad = X10Tensor.rand(outShape.dimensions)
        if useReducedPrecision {
          x = x.toReducedPrecision
          outGrad = outGrad.toReducedPrecision
        }
        let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
        assertEqualUnaryOperationGradients(
          {
            (_ x: X10Tensor) -> X10Tensor in
            x.padded(forSizes: paddings, mode: reflect ? .reflect : .symmetric)
          },
          {
            (_ x: TFTensor) -> TFTensor in
            x.padded(forSizes: paddings, mode: reflect ? .reflect : .symmetric)
          }, x, outGrad, relTolerance: relTolerance)
      }
    }
  }

  func testMod() throws {
    for useReducedPrecision in [false, true] {
      var x = X10Tensor((-30..<30).map { (v: Int64) -> Float in Float.init(v) })
      var y = X10Tensor(8)
      if useReducedPrecision {
        x = x.toReducedPrecision
        y = y.toReducedPrecision
      }
      var actual = x % y
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let expected = TF(x) % TF(y)
      XCTAssertEqual(TF(actual), expected)
    }
  }

  func testMul() throws {
    var x = X10Tensor(shape: [2], scalars: [1, 3])
    var y = X10Tensor(shape: [2], scalars: [7, 19])
    for useReducedPrecision in [false, true] {
      let expected = TF(x) * TF(y)
      if useReducedPrecision {
        x = x.toReducedPrecision
        y = y.toReducedPrecision
      }
      var actual = x * y
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssertEqual(TF(actual), expected)
    }
  }

  func testNotEqual() throws {
    var x = X10Tensor(shape: [4], scalars: [1, 22, 3, 5])
    var y = X10Tensor(shape: [4], scalars: [7, 19, 3, 5])
    for useReducedPrecision in [false, true] {
      let expected = TF(x) .!= TF(y)
      if useReducedPrecision {
        x = x.toReducedPrecision
        y = y.toReducedPrecision
      }
      let actual = TF(x .!= y)
      XCTAssertEqual(actual, expected)
    }
  }

  func testOneHot() throws {
    let labels = X10Tensor_<Int32>(shape: [2], scalars: [3, 4])
    let actual = TF(X10Tensor(oneHotAtIndices: labels, depth: 8))
    let expected = TFTensor(oneHotAtIndices: TF(labels), depth: 8)
    XCTAssertEqual(actual, expected)
  }

  func testOnesLike() throws {
    var x = X10Tensor.rand([3, 2, 8])
    let expected = TFTensor(onesLike: TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = X10Tensor(onesLike: x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssertEqual(TF(actual), expected)
    }
  }

  func testPack() throws {
    for useReducedPrecision in [false, true] {
      for dim in 0..<3 {
        var xs = [[3, 2, 2], [3, 2, 2], [3, 2, 2]].map { X10Tensor.rand($0) }
        let expected = TFTensor(stacking: xs.map(TF), alongAxis: dim)
        if useReducedPrecision {
          xs = xs.toReducedPrecision
        }
        var actual = X10Tensor(stacking: xs, alongAxis: dim)
        if useReducedPrecision {
          XCTAssert(actual.isReducedPrecision)
          actual = actual.toFullPrecision
        }
        XCTAssert(!actual.isReducedPrecision)
        let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
        XCTAssert(
          allClose(
            actual: TF(actual), expected: expected, relTolerance: relTolerance))
      }
    }
  }

  func testPadV1() throws {
    let paddings = X10Tensor_<Int32>(shape: [3, 2], scalars: [1, 2, 3, 4, 5, 6])
    var x = X10Tensor.rand([5, 7, 3])
    let expected = _Raw.pad(TF(x), paddings: TF(paddings))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = _Raw.pad(x, paddings: paddings)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testPadWithConstant() throws {
    let paddings = [(1, 2), (3, 4), (5, 6)]
    var x = X10Tensor.rand([5, 7, 3])
    let padValue = Float(3.0)
    let expected = TF(x).padded(forSizes: paddings, mode: .constant(padValue))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = x.padded(forSizes: paddings, mode: .constant(padValue))
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testPow() throws {
    let dims = [3, 2]
    var x = X10Tensor.rand(dims)
    var e = X10Tensor(
      shape: TensorShape(dims), scalars: [Float](repeating: 0.25, count: dims.reduce(1, *)))
    let expected = pow(TF(x), TF(e))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        e = e.toReducedPrecision
      }
      var actual = pow(x, e)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5)
      )
    }
  }

  func testProd() throws {
    var x = X10Tensor(shape: [3, 2], scalars: [1, 5, 43, 24, 64, 32])
    let expected = TF(x).product(squeezingAxes: [0])
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = x.product(squeezingAxes: [0])
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssertEqual(TF(actual), expected)
    }
  }

  // TODO(asuhan): Figure out whether we could fix accuracy issues with bf16.
  func testQR() throws {
    let dims = [4, 7]
    for m in dims {
      for n in dims {
        for fullMatrices in [true, false] {
          let x = X10Tensor.rand([m, n])
          let actual = x.qrDecomposition(fullMatrices: fullMatrices)
          let expected = TF(x).qrDecomposition(fullMatrices: fullMatrices)
          XCTAssert(
            allClose(actual: TF(actual.q), expected: expected.q, relTolerance: 2e-2))
          XCTAssert(
            allClose(actual: TF(actual.r), expected: expected.r, relTolerance: 1e-3))
        }
      }
    }
  }

  func testRange() throws {
    let start = Int32(3)
    let limit = Int32(18)
    let delta = Int32(3)
    for reverse in [false, true] {
      let rangeFrom = reverse ? limit : start
      let to = reverse ? start : limit
      let stride = reverse ? -delta : delta
      let actual = TF(X10Tensor_<Int32>(rangeFrom: rangeFrom, to: to, stride: stride))
      let expected = TFTensor_<Int32>(rangeFrom: rangeFrom, to: to, stride: stride)
      XCTAssertEqual(actual, expected)
    }
  }

  func testRank() throws {
    let x = X10Tensor.rand([3, 2])
    let actual = TF(x.rankTensor)
    let expected = TF(x).rankTensor
    XCTAssertEqual(actual, expected)
  }

  func testRelu() throws {
    var x = X10Tensor(shape: [2], scalars: [-0.5, 0.5])
    let expected = relu(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = relu(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssertEqual(TF(actual), expected)
    }
  }

  func testReluGrad() throws {
    func reluX10(_ arg: X10Tensor) -> X10Tensor {
      return relu(arg)
    }
    func reluTF(_ arg: TFTensor) -> TFTensor {
      return relu(arg)
    }
    var x = X10Tensor(shape: [2], scalars: [-0.5, 0.5])
    var outGrad = X10Tensor(shape: [2], scalars: [1.5, 2.5])
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        outGrad = outGrad.toReducedPrecision
      }
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      assertEqualUnaryOperationGradients(reluX10, reluTF, x, outGrad, relTolerance: relTolerance)
    }
  }

  func testRelu6() throws {
    var x = X10Tensor(shape: [5], scalars: [-0.5, 0.5, 4, 7, 8])
    let expected = relu6(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = relu6(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssertEqual(TF(actual), expected)
    }
  }

  func testRelu6Grad() throws {
    func relu6X10(_ arg: X10Tensor) -> X10Tensor {
      return relu6(arg)
    }
    func relu6TF(_ arg: TFTensor) -> TFTensor {
      return relu6(arg)
    }
    var x = X10Tensor(shape: [5], scalars: [-0.5, 0.5, 4, 7, 8])
    var outGrad = X10Tensor(shape: [5], scalars: [1.5, 2.5, 2, 5, 3])
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        outGrad = outGrad.toReducedPrecision
      }
      assertEqualUnaryOperationGradients(relu6X10, relu6TF, x, outGrad)
    }
  }

  func testReshape() throws {
    for useReducedPrecision in [false, true] {
      do {
        var x = X10Tensor(shape: [4, 2], scalars: [1, 5, 43, 23, 24, 64, 32, 32])
        let expected = TF(x).reshaped(to: [1, 8])
        if useReducedPrecision {
          x = x.toReducedPrecision
        }
        var actual = x.reshaped(to: [1, 8])
        if useReducedPrecision {
          XCTAssert(actual.isReducedPrecision)
          actual = actual.toFullPrecision
        }
        XCTAssert(!actual.isReducedPrecision)
        XCTAssertEqual(TF(actual), expected)
      }
      do {
        var x = X10Tensor(shape: [1, 1, 1], scalars: [3])
        let expected = TF(x).reshaped(to: [])
        if useReducedPrecision {
          x = x.toReducedPrecision
        }
        var actual = x.reshaped(to: [])
        if useReducedPrecision {
          XCTAssert(actual.isReducedPrecision)
          actual = actual.toFullPrecision
        }
        XCTAssert(!actual.isReducedPrecision)
        XCTAssertEqual(TF(actual), expected)
      }
    }
  }

  func testRound() throws {
    var x = X10Tensor([-3.5, -3.4, -3.6, -0.5, 0.5, -0.45, 0.45, 2.4, 2.6])
    let expected = round(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = round(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssertEqual(TF(actual), expected)
    }
  }

  func testRsqrt() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = rsqrt(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = rsqrt(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
    }
  }

  func testRsqrtGrad() throws {
    var x = X10Tensor.rand([3, 2])
    var outGrad = X10Tensor.rand(x.shape.dimensions)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        outGrad = outGrad.toReducedPrecision
      }
      let relTolerance: Float = useReducedPrecision ? 2e-2 : 1e-5
      assertEqualUnaryOperationGradients(
        { rsqrt($0) }, { rsqrt($0) }, x, outGrad, relTolerance: relTolerance, absTolerance: 1e-5)
    }
  }

  func testSelect() throws {
    let dims = [4, 2, 3]
    for useReducedPrecision in [false, true] {
      for broadcast in [false, true] {
        var t = X10Tensor.rand(dims)
        var e = X10Tensor.rand(dims)
        let condition: X10Tensor_<Bool> = X10Tensor.randint(0, 2, broadcast ? dims : [dims[0]])
        let expected = TF(t).replacing(with: TF(e), where: TF(condition))
        if useReducedPrecision {
          t = t.toReducedPrecision
          e = e.toReducedPrecision
        }
        var actual = t.replacing(with: e, where: condition)
        if useReducedPrecision {
          XCTAssert(actual.isReducedPrecision)
          actual = actual.toFullPrecision
        }
        XCTAssert(!actual.isReducedPrecision)
        let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
        XCTAssert(
          allClose(
            actual: TF(actual), expected: expected, relTolerance: relTolerance))
      }
    }
  }

  func testSelu() throws {
    var x = X10Tensor(shape: [6], scalars: [-1.0, -0.5, 0.5, 3.0, 4.0, 7.0])
    var outGrad = X10Tensor.rand(x.shape.dimensions)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        outGrad = outGrad.toReducedPrecision
      }
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      assertEqualUnaryOperationGradients(
        { selu($0) }, { selu($0) }, x, outGrad, relTolerance: relTolerance, absTolerance: 1e-5)
    }
  }

  func testSigmoid() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = sigmoid(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = sigmoid(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
    }
  }

  func testSigmoidGrad() throws {
    var x = X10Tensor.rand([3, 2])
    var outGrad = X10Tensor.rand(x.shape.dimensions)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        outGrad = outGrad.toReducedPrecision
      }
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      assertEqualUnaryOperationGradients(
        { sigmoid($0) }, { sigmoid($0) }, x, outGrad, relTolerance: relTolerance, absTolerance: 1e-5
      )
    }
  }

  func testSign() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = sign(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      let actual = TF(sign(x))
      XCTAssertEqual(actual, expected)
    }
  }

  func testSin() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = sin(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = sin(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5)
      )
    }
  }

  func testSinh() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = sinh(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = sinh(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
    }
  }

  func testSize() throws {
    let x = X10Tensor.rand([3, 2])
    let actual = TF(x.scalarCountTensor)
    let expected = TF(x).scalarCountTensor
    XCTAssertEqual(actual, expected)
  }

  func testSlice() throws {
    var x = X10Tensor.rand([5, 8])
    let expected = TF(x).slice(lowerBounds: [1, 2], upperBounds: [3, 6])
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = x.slice(lowerBounds: [1, 2], upperBounds: [3, 6])
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testSliceToEnd() throws {
    let lowerBounds: [Int32] = [1, 0]
    let sizes: [Int32] = [3, -1]
    var x = X10Tensor.rand([5, 8])
    let expected = TF(x).slice(
      lowerBounds: TFTensor_<Int32>(shape: [lowerBounds.count], scalars: lowerBounds),
      sizes: TFTensor_<Int32>(shape: [sizes.count], scalars: sizes))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = x.slice(
        lowerBounds: X10Tensor_<Int32>(shape: [lowerBounds.count], scalars: lowerBounds),
        sizes: X10Tensor_<Int32>(shape: [sizes.count], scalars: sizes))
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testSoftmax() throws {
    var x = X10Tensor(shape: [2, 2], scalars: [1, 2, 5, 3])
    let expected = softmax(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = softmax(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(allClose(actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testSoftmaxCrossEntropyWithLogits() throws {
    var features = X10Tensor.rand([3, 4])
    var labels = X10Tensor.rand([3, 4])
    var outGrad = X10Tensor(1.0)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        features = features.toReducedPrecision
        labels = labels.toReducedPrecision
        outGrad = outGrad.toReducedPrecision
      }
      let relTolerance: Float = useReducedPrecision ? 3e-2 : 1e-5
      assertEqualUnaryOperationGradients(
        { softmaxCrossEntropy(logits: $0, probabilities: labels) },
        { softmaxCrossEntropy(logits: $0, probabilities: TF(labels)) },
        features, outGrad, relTolerance: relTolerance, absTolerance: 1e-4)
    }
  }

  func testSoftplus() throws {
    func softplusX10(_ arg: X10Tensor) -> X10Tensor {
      return softplus(arg)
    }
    func softplusTF(_ arg: TFTensor) -> TFTensor {
      return softplus(arg)
    }
    var x = X10Tensor(shape: [6], scalars: [-1.0, -0.5, 0.5, 3.0, 4.0, 7.0])
    var outGrad = X10Tensor.rand(x.shape.dimensions)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        outGrad = outGrad.toReducedPrecision
      }
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-4
      assertEqualUnaryOperationGradients(
        softplusX10, softplusTF, x, outGrad, relTolerance: relTolerance, absTolerance: 1e-5)
    }
  }

  func testSoftsign() throws {
    func softsignX10(_ arg: X10Tensor) -> X10Tensor {
      return softsign(arg)
    }
    func softsignTF(_ arg: TFTensor) -> TFTensor {
      return softsign(arg)
    }
    var x = X10Tensor(shape: [6], scalars: [-1.0, -0.5, 0.5, 3.0, 4.0, 7.0])
    var outGrad = X10Tensor.rand(x.shape.dimensions)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        outGrad = outGrad.toReducedPrecision
      }
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-4
      assertEqualUnaryOperationGradients(
        softsignX10, softsignTF, x, outGrad, relTolerance: relTolerance, absTolerance: 1e-5)
    }
  }

  func testSparseSoftmaxCrossEntropyWithLogits() throws {
    let labels = X10Tensor_<Int32>(shape: [2], scalars: [3, 4])
    var logits = X10Tensor(
      shape: [2, 5], scalars: [0.98, 0.65, 0.832, 0.324, 0.3676, 0.777, 0.244, 0.950, 0.544, 0.445])
    let expected = _Raw.sparseSoftmaxCrossEntropyWithLogits(
      features: TF(logits), labels: TF(labels))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        logits = logits.toReducedPrecision
      }
      var (actualLoss, actualBackprop) = _Raw.sparseSoftmaxCrossEntropyWithLogits(
        features: logits, labels: labels)
      if useReducedPrecision {
        XCTAssert(actualLoss.isReducedPrecision)
        XCTAssert(actualBackprop.isReducedPrecision)
        actualLoss = actualLoss.toFullPrecision
        actualBackprop = actualBackprop.toFullPrecision
      }
      XCTAssert(!actualLoss.isReducedPrecision)
      XCTAssert(!actualBackprop.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-4
      XCTAssert(
        allClose(
          actual: TF(actualLoss), expected: expected.loss, relTolerance: relTolerance,
          absTolerance: 2e-7),
        file: #file,
        line: #line)
      XCTAssert(
        allClose(
          actual: TF(actualBackprop), expected: expected.backprop, relTolerance: relTolerance,
          absTolerance: 1e-7))
    }
  }

  func testSplit() throws {
    let valueDims = [9, 9, 9]
    let numSplit = Int64(3)
    for useReducedPrecision in [false, true] {
      for splitDim in -valueDims.count..<valueDims.count {
        var value = X10Tensor.rand(valueDims)
        let expectedList = _Raw.split(
          splitDim: TFTensor_<Int32>(Int32(splitDim)), value: TF(value), numSplit: numSplit)
        if useReducedPrecision {
          value = value.toReducedPrecision
        }
        var actualList = _Raw.split(
          splitDim: X10Tensor_<Int32>(Int32(splitDim)), value: value, numSplit: numSplit)
        if useReducedPrecision {
          for actual in actualList {
            XCTAssert(actual.isReducedPrecision)
          }
          actualList = actualList.toFullPrecision
        }
        for actual in actualList {
          XCTAssert(!actual.isReducedPrecision)
        }
        let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
        for (actual, expected) in zip(actualList, expectedList) {
          XCTAssert(
            allClose(
              actual: TF(actual), expected: expected, relTolerance: relTolerance)
          )
        }
      }
    }
  }

  func testSplitV() throws {
    let originalDims: [Int32] = [3, 2, 4]
    for useReducedPrecision in [false, true] {
      for inferDim in 0...originalDims.count {
        var dims = originalDims
        if inferDim < originalDims.count {
          dims[inferDim] = -1
        }
        let sizeSplits = X10Tensor_<Int32>(shape: [dims.count], scalars: dims)
        let valueDims = [9, 9, 9]
        for splitDim in -valueDims.count..<valueDims.count {
          var value = X10Tensor.rand(valueDims)
          let expectedList =
            _Raw.splitV(
              value: TF(value), sizeSplits: TF(sizeSplits),
              splitDim: TFTensor_<Int32>(Int32(splitDim)),
              numSplit: Int64(sizeSplits.shape.dimensions.first!))
          if useReducedPrecision {
            value = value.toReducedPrecision
          }
          var actualList = _Raw.splitV(
            value: value, sizeSplits: sizeSplits, splitDim: X10Tensor_<Int32>(Int32(splitDim)),
            numSplit: Int64(sizeSplits.shape.dimensions.first!))
          if useReducedPrecision {
            for actual in actualList {
              XCTAssert(actual.isReducedPrecision)
            }
            actualList = actualList.toFullPrecision
          }
          for actual in actualList {
            XCTAssert(!actual.isReducedPrecision)
          }
          let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
          for (actual, expected) in zip(actualList, expectedList) {
            XCTAssert(
              allClose(
                actual: TF(actual), expected: expected, relTolerance: relTolerance)
            )
          }
        }
      }
    }
  }

  func testSqrt() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = sqrt(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = sqrt(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5)
      )
    }
  }

  func testSquare() throws {
    var x = X10Tensor.rand([3, 2])
    let expected = TF(x).squared()
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = x.squared()
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance, absTolerance: 1e-5))
    }
  }

  func testSquaredDifference() throws {
    var x = X10Tensor.rand([3, 2])
    var y = X10Tensor.rand([3, 2])
    let expected = squaredDifference(TF(x), TF(y))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        y = y.toReducedPrecision
      }
      var actual = squaredDifference(x, y)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      let absTolerance: Float = useReducedPrecision ? 1e-3 : 1e-5
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected, relTolerance: relTolerance,
          absTolerance: absTolerance))
    }
  }

  func testSqueeze() throws {
    for useReducedPrecision in [false, true] {
      for (dims, onesDims) in [([1, 3, 4, 1], [0, 3]), ([2, 1, 2, 3], [1]), ([3, 1, 1, 2], [2])] {
        var x = X10Tensor.rand(dims)
        let expected = TF(x).squeezingShape(at: onesDims)
        if useReducedPrecision {
          x = x.toReducedPrecision
        }
        var actual = x.squeezingShape(at: onesDims)
        if useReducedPrecision {
          XCTAssert(actual.isReducedPrecision)
          actual = actual.toFullPrecision
        }
        XCTAssert(!actual.isReducedPrecision)
        let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
        XCTAssert(allClose(actual: TF(actual), expected: expected, relTolerance: relTolerance))
      }
    }
  }

  func testStatelessTruncatedNormal() throws {
    let seed = (graph: Int32(604_591_423), op: Int32(42_628_358))
    let t = TF(X10Tensor(randomTruncatedNormal: [2, 2], seed: seed))
    for scalar in t.scalars {
      XCTAssertLessThanOrEqual(scalar, 2.0)
      XCTAssertGreaterThanOrEqual(scalar, -2.0)
    }
  }

  func testStatelessUniformNormal() throws {
    let seed = (graph: Int32(604_591_423), op: Int32(42_628_358))
    let t = TF(X10Tensor(randomNormal: [2, 2], seed: seed))
    for scalar in t.scalars {
      XCTAssertLessThanOrEqual(scalar, 6.0)
      XCTAssertGreaterThanOrEqual(scalar, -6.0)
    }
  }

  func testStatelessUniformRandom() throws {
    let seed = (graph: Int32(604_591_423), op: Int32(42_628_358))
    let actual = TF(X10Tensor(randomUniform: [2, 2], seed: seed))
    for scalar in actual.scalars {
      XCTAssertLessThanOrEqual(scalar, 1.0)
      XCTAssertGreaterThanOrEqual(scalar, 0.0)
    }
  }

  func testStatelessUniformRandomInt() throws {
    let seed = (graph: Int32(604_591_423), op: Int32(42_628_358))
    let lowerBound = X10Tensor_<Int32>(0)
    let upperBound = X10Tensor_<Int32>(100)
    let t = X10Tensor_<Int32>(
      randomUniform: [2, 2], lowerBound: lowerBound, upperBound: upperBound, seed: seed)
    for scalar in t.scalars {
      XCTAssertLessThan(scalar, 100)
      XCTAssertGreaterThanOrEqual(scalar, 0)
    }
  }

  func testSub() throws {
    var x = X10Tensor(shape: [2], scalars: [1, 5])
    var y = X10Tensor(shape: [2], scalars: [7, 19])
    let expected = TF(x) - TF(y)
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        y = y.toReducedPrecision
      }
      var actual = x - y
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssert(
        allClose(
          actual: TF(actual), expected: expected))
    }
  }

  func testSum() throws {
    var x = X10Tensor(shape: [4, 2], scalars: [1, 5, 43, 23, 24, 64, 32, 32])
    let expected = TF(x).sum(squeezingAxes: [0])
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = x.sum(squeezingAxes: [0])
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssertEqual(TF(actual), expected)
    }
  }

  func testTan() throws {
    var x = X10Tensor(shape: [2, 2], scalars: [1, 2, 5, 3])
    let expected = tan(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = tan(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(allClose(actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testTanh() throws {
    var x = X10Tensor(shape: [2, 2], scalars: [1, 2, 5, 3])
    let expected = tanh(TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = tanh(x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-4
      XCTAssert(allClose(actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testTile() throws {
    var x = X10Tensor.rand([5, 2, 3])
    let multiples: [Int32] = [6, 15, 10]
    let expected = TF(x).tiled(
      multiples: TFTensor_<Int32>(shape: [multiples.count], scalars: multiples))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = x.tiled(
        multiples: X10Tensor_<Int32>(shape: [multiples.count], scalars: multiples))
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(allClose(actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testTranspose() throws {
    var x = X10Tensor(shape: [2, 4], scalars: [0, 1, 2, 3, 4, 5, 6, 7])
    let expected = TF(x).transposed(permutation: [0, 1])
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = x.transposed(permutation: [0, 1])
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssertEqual(TF(actual), expected)
    }
  }

  func testUnpack() throws {
    for useReducedPrecision in [false, true] {
      for dim in -2..<3 {
        var xs = [[3, 2, 2], [3, 2, 2], [3, 2, 2]].map { X10Tensor.rand($0) }
        let expected = xs
        if useReducedPrecision {
          xs = xs.toReducedPrecision
        }
        let result = X10Tensor(stacking: xs, alongAxis: dim).unstacked(alongAxis: dim)
        XCTAssertEqual(result.count, xs.count)
        for (x, unpacked) in zip(expected, result) {
          var actual = unpacked
          if useReducedPrecision {
            XCTAssert(actual.isReducedPrecision)
            actual = actual.toFullPrecision
          }
          XCTAssert(!actual.isReducedPrecision)
          let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
          XCTAssert(allClose(actual: TF(actual), expected: TF(x), relTolerance: relTolerance))
        }
      }
    }
  }

  func testUnsortedSegmentSum() throws {
    var data = X10Tensor.rand([3, 4])
    let segmentIds = X10Tensor_<Int32>(shape: [data.shape.dimensions[0]], scalars: [0, 1, 0])
    let numSegments = X10Tensor_<Int32>(shape: [], scalars: [2])
    let expected = _Raw.unsortedSegmentSum(
      data: TF(data), segmentIds: TF(segmentIds), numSegments: TF(numSegments))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        data = data.toReducedPrecision
      }
      var actual = _Raw.unsortedSegmentSum(
        data: data, segmentIds: segmentIds, numSegments: numSegments)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      let relTolerance: Float = useReducedPrecision ? 1e-2 : 1e-5
      XCTAssert(allClose(actual: TF(actual), expected: expected, relTolerance: relTolerance))
    }
  }

  func testXdivY() throws {
    var x = X10Tensor(shape: [2, 3], scalars: [0, 1, 0, 0, 4, 8])
    var y = X10Tensor(shape: [3], scalars: [0, 1, 2])
    let expected = _Raw.xdivy(TF(x), TF(y))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
        y = y.toReducedPrecision
      }
      var actual = _Raw.xdivy(x, y)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssert(
        allClose(actual: TF(actual), expected: expected, absTolerance: 1e-5))
    }
  }

  func testZerosLike() throws {
    var x = X10Tensor.rand([3, 2, 8])
    let expected = TFTensor(zerosLike: TF(x))
    for useReducedPrecision in [false, true] {
      if useReducedPrecision {
        x = x.toReducedPrecision
      }
      var actual = X10Tensor(zerosLike: x)
      if useReducedPrecision {
        XCTAssert(actual.isReducedPrecision)
        actual = actual.toFullPrecision
      }
      XCTAssert(!actual.isReducedPrecision)
      XCTAssertEqual(TF(actual), expected)
    }
  }
}

extension TensorTests {
  static var allTests = [
    ("testAbs", testAbs),
    ("testAcos", testAcos),
    ("testAcosh", testAcosh),
    ("testAdd", testAdd),
    ("testAddInterop", testAddInterop),
    ("testAll", testAll),
    ("testAny", testAny),
    ("testApproximateEqual", testApproximateEqual),
    ("testArgmax", testArgmax),
    ("testArgmin", testArgmin),
    ("testAsin", testAsin),
    ("testAsinh", testAsinh),
    ("testAtan2", testAtan2),
    ("testAtan", testAtan),
    ("testAtanh", testAtanh),
    ("testAvgPool", testAvgPool),
    ("testAvgPoolGrad", testAvgPoolGrad),
    ("testAvgPool3DGrad", testAvgPool3DGrad),
    ("testBatchNorm", testBatchNorm),
    ("testBatchNormGrad", testBatchNormGrad),
    ("testBF16Construct", testBF16Construct),
    ("testBF16Conv2D", testBF16Conv2D),
    ("testBF16GradientPropagation", testBF16GradientPropagation),
    ("testBF16Loopback", testBF16Loopback),
    ("testBF16SparseSoftmaxCrossEntropyWithLogits", testBF16SparseSoftmaxCrossEntropyWithLogits),
    ("testBF16Sum", testBF16Sum),
    ("testBroadcastDims", testBroadcastDims),
    ("testBroadcastTo", testBroadcastTo),
    ("testBroadcastGradientArgs", testBroadcastGradientArgs),
    ("testCast", testCast),
    ("testCeil", testCeil),
    ("testConcat", testConcat),
    ("testClipByValue", testClipByValue),
    ("testConv2D", testConv2D),
    ("testConv2DGrad", testConv2DGrad),
    ("testConv3DGrad", testConv3DGrad),
    ("testCos", testCos),
    ("testCosh", testCosh),
    ("testCumprod", testCumprod),
    ("testCumsum", testCumsum),
    ("testDepthwiseConv2DGrad", testDepthwiseConv2DGrad),
    ("testDiv", testDiv),
    ("testDiagonalPart", testDiagonalPart),
    ("testElu", testElu),
    ("testEqual", testEqual),
    ("testExp", testExp),
    ("testExpm1", testExpm1),
    ("testFill", testFill),
    ("testFloor", testFloor),
    ("testGather", testGather),
    ("testGatherV2", testGatherV2),
    ("testGreater", testGreater),
    ("testGreaterEqual", testGreaterEqual),
    ("testIndexAdvanced", testIndexAdvanced),
    ("testIndexAdvancedGrad", testIndexAdvancedGrad),
    ("testIndexElement", testIndexElement),
    ("testIndexElementAssignment", testIndexElementAssignment),
    ("testIndexElementGrad", testIndexElementGrad),
    ("testIndexEllipsis", testIndexEllipsis),
    ("testIndexEllipsisGrad", testIndexEllipsisGrad),
    ("testIndexNestedElement", testIndexNestedElement),
    ("testIndexNestedElementGrad", testIndexNestedElementGrad),
    ("testIndexNewAxis", testIndexNewAxis),
    ("testIndexNewAxisGrad", testIndexNewAxisGrad),
    ("testIndexSlice", testIndexSlice),
    ("testIndexSliceAssignment", testIndexSliceAssignment),
    ("testIndexSliceGrad", testIndexSliceGrad),
    ("testIndexSqueezeAxis", testIndexSqueezeAxis),
    ("testIndexSqueezeAxisGrad", testIndexSqueezeAxisGrad),
    ("testIndexStridedSlice", testIndexStridedSlice),
    ("testIndexStridedSliceGrad", testIndexStridedSliceGrad),
    ("testInvertPermutation", testInvertPermutation),
    ("testIsFinite", testIsFinite),
    ("testIsInfinite", testIsInfinite),
    ("testIsNaN", testIsNaN),
    ("testLeakyRelu", testLeakyRelu),
    ("testLeakyReluGrad", testLeakyReluGrad),
    ("testLess", testLess),
    ("testLessEqual", testLessEqual),
    ("testLinSpace", testLinSpace),
    ("testLog", testLog),
    ("testLog1p", testLog1p),
    ("testLogicalAnd", testLogicalAnd),
    ("testLogicalNot", testLogicalNot),
    ("testLogicalOr", testLogicalOr),
    ("testLogSoftmax", testLogSoftmax),
    ("testMatMul", testMatMul),
    ("testMax", testMax),
    ("testMaximum", testMaximum),
    ("testMaxPool", testMaxPool),
    ("testMaxPoolGrad", testMaxPoolGrad),
    ("testMaxPool3DGrad", testMaxPool3DGrad),
    ("testMean", testMean),
    ("testMeanBool", testMeanBool),
    ("testMin", testMin),
    ("testMinimum", testMinimum),
    ("testMirrorPad", testMirrorPad),
    ("testMirrorPadGrad", testMirrorPadGrad),
    ("testMod", testMod),
    ("testMul", testMul),
    ("testNotEqual", testNotEqual),
    ("testOneHot", testOneHot),
    ("testOnesLike", testOnesLike),
    ("testPack", testPack),
    ("testPadV1", testPadV1),
    ("testPadWithConstant", testPadWithConstant),
    ("testPow", testPow),
    ("testProd", testProd),
    ("testQR", testQR),
    ("testRange", testRange),
    ("testRank", testRank),
    ("testRelu", testRelu),
    ("testReluGrad", testReluGrad),
    ("testRelu6", testRelu6),
    ("testRelu6Grad", testRelu6Grad),
    ("testReshape", testReshape),
    ("testRound", testRound),
    ("testRsqrt", testRsqrt),
    ("testRsqrtGrad", testRsqrtGrad),
    ("testSelect", testSelect),
    ("testSelu", testSelu),
    ("testSigmoid", testSigmoid),
    ("testSigmoidGrad", testSigmoidGrad),
    ("testSign", testSign),
    ("testSin", testSin),
    ("testSinh", testSinh),
    ("testSize", testSize),
    ("testSlice", testSlice),
    ("testSliceToEnd", testSliceToEnd),
    ("testSoftmaxCrossEntropyWithLogits", testSoftmaxCrossEntropyWithLogits),
    ("testSoftmax", testSoftmax),
    ("testSoftplus", testSoftplus),
    ("testSoftsign", testSoftsign),
    ("testSparseSoftmaxCrossEntropyWithLogits", testSparseSoftmaxCrossEntropyWithLogits),
    ("testSplit", testSplit),
    ("testSplitV", testSplitV),
    ("testSqrt", testSqrt),
    ("testSquare", testSquare),
    ("testSquaredDifference", testSquaredDifference),
    ("testSqueeze", testSqueeze),
    ("testStatelessTruncatedNormal", testStatelessTruncatedNormal),
    ("testStatelessUniformNormal", testStatelessUniformNormal),
    ("testStatelessUniformRandom", testStatelessUniformRandom),
    ("testStatelessUniformRandomInt", testStatelessUniformRandomInt),
    ("testSub", testSub),
    ("testSum", testSum),
    ("testTan", testTan),
    ("testTanh", testTanh),
    ("testTile", testTile),
    ("testTranspose", testTranspose),
    ("testUnpack", testUnpack),
    ("testUnsortedSegmentSum", testUnsortedSegmentSum),
    ("testXdivY", testXdivY),
    ("testZerosLike", testZerosLike),
  ]
}

// Run with:
// export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
// export XRT_WORKERS="localservice:0;grpc://localhost:40934"

// TODO(b/130689556): Remove this environment setting once the bug is fixed.
setenv("XLA_FLAGS", "--xla_cpu_fast_math_honor_nans=true --xla_cpu_fast_math_honor_infs=true", 1)
SetMatMulPrecision(true)
XCTMain([
  testCase(TensorTests.allTests),
])
