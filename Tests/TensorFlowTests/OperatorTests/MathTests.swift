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

final class MathOperatorTests: XCTestCase {
  func testElementaryFunction(
    name: String,
    _ tensorOperator: (Tensor<Float>) -> Tensor<Float>,
    _ scalarOperator: (Float) -> Float,
    accuracy: Float = 1e-4,
    file: StaticString = #file, line: UInt = #line
  ) {
    let x = Tensor<Float>(randomNormal: [20], seed: (0, 0))
    let actual = tensorOperator(x).scalars
    let expected = x.scalars.map(scalarOperator)
    assertEqual(actual, expected, accuracy: accuracy, name, file: file, line: line)
  }

  func testElementaryFunctions() {
    testElementaryFunction(name: "sqrt", sqrt, Float.sqrt)
    testElementaryFunction(name: "cos", cos, Float.cos)
    testElementaryFunction(name: "sin", sin, Float.sin)
    testElementaryFunction(name: "tan", tan, Float.tan)
    testElementaryFunction(name: "cosh", cosh, Float.cosh)
    testElementaryFunction(name: "sinh", sinh, Float.sinh)
    testElementaryFunction(name: "tanh", tanh, Float.tanh)
    testElementaryFunction(name: "acos", acos, Float.acos)
    testElementaryFunction(name: "asin", asin, Float.asin)
    testElementaryFunction(name: "atan", atan, Float.atan)
    testElementaryFunction(name: "acosh", acosh, Float.acosh)
    testElementaryFunction(name: "asinh", asinh, Float.asinh)
    testElementaryFunction(name: "atanh", atanh, Float.atanh)
    testElementaryFunction(name: "exp", exp, Float.exp)
    testElementaryFunction(name: "exp2", exp2, Float.exp2)
    testElementaryFunction(name: "exp10", exp10, Float.exp10)
    testElementaryFunction(name: "expm1", expm1, Float.expm1)
    testElementaryFunction(name: "log", log, Float.log)
    testElementaryFunction(name: "log2", log2, Float.log2)
    testElementaryFunction(name: "log10", log10, Float.log10)
    testElementaryFunction(name: "log1p", log1p, Float.log1p)
    testElementaryFunction(
      name: "pow",
      { x in pow(x, x) }, { x in Float.pow(x, x) })
    testElementaryFunction(
      name: "pow",
      { x in pow(x, 3) }, { x in Float.pow(x, 3) })
    testElementaryFunction(
      name: "root",
      { x in root(x, 3) }, { x in Float.root(x, 3) })
  }

  func testAbs() {
    let x = Tensor<Float>([-1.0])
    let y = abs(x)
    let expectedY = Tensor<Float>([1.0])
    XCTAssertEqual(y, expectedY)
  }

  func testSquaredDifference() {
    let x = Tensor<Float>([-5.8])
    let y = Tensor<Float>([5.7])
    let z = squaredDifference(x, y)
    let approxZ = Tensor<Float>([132.25])
    XCTAssertEqual(z, approxZ)
  }

  func testZeros() {
    let x = Tensor<Float>(zeros: [1])
    let x1 = Tensor<Float>([0.0])
    XCTAssertEqual(x, x1)
  }

  func testLogSoftmax() {
    let x = Tensor<Float>([
      [32.0, 34.0, 35.0],
      [36.0, 37.0, 38.0],
    ])
    let y = logSoftmax(x)
    let y1 = Tensor<Float>([
      [-3.3490124, -1.3490123, -0.34901226],
      [-2.407606, -1.407606, -0.40760598],
    ])
    assertEqual(y, y1, accuracy: 0.0001)
  }

  func testMax() {
    let x = Tensor<Float>([58.0])
    let y = Tensor<Float>([57.0])
    let z = max(x, y)
    let expectedZ = Tensor<Float>([58.0])
    XCTAssertEqual(z, expectedZ)
  }

  func testMin() {
    let x = Tensor<Float>([58.0])
    let y = Tensor<Float>([57.0])
    let z = min(x, y)
    let expectedZ = Tensor<Float>([57.0])
    XCTAssertEqual(z, expectedZ)
  }

  func testRound() {
    let x = Tensor<Float>([58.76])
    let y = round(x)
    let expectedY = Tensor<Float>([59.0])
    XCTAssertEqual(y, expectedY)
  }

  func testSoftmax() {
    let x = Tensor<Float>([
      [-32.0, -34.0, -35.0],
      [-36.0, -37.0, -38.0],
    ])
    let y = softmax(x)
    let expectedY = Tensor<Float>([
      [0.8437947, 0.1141952, 0.042010065],
      [0.66524094, 0.24472848, 0.09003057],
    ])
    assertEqual(y, expectedY, accuracy: 0.0001)
  }

  func testSigmoid() {
    let x = Tensor<Float>([59.0])
    let y = sigmoid(x)
    let expectedY = Tensor<Float>([1.0])
    XCTAssertEqual(y, expectedY)
  }

  func testIdentity() {
    let x = Tensor<Float>([-5.8, -5.9])
    let y = identity(x)
    let expectedY = Tensor<Float>([-5.8, -5.9])
    XCTAssertEqual(y, expectedY)
  }

  func testClipping() {
    let x = Tensor<Float>([
      [0.45031791, 0.41123222, 0.53928467, 0.47167023, 0.15483777],
      [0.49975705, 0.71807549, 0.30396056, 0.26904690, 0.01404393],
      [0.16950939, 0.41085612, 0.79503016, 0.11977817, 0.99728241],
      [0.62510073, 0.17344792, 0.15406050, 0.40758517, 0.93683817],
      [0.15653343, 0.50502756, 0.99365925, 0.84617581, 0.17422509],
    ])
    let clippedX = x.clipped(min: 0.2, max: 0.5)
    let expectedClippedX = Tensor<Float>([
      [0.45031791, 0.41123222, 0.50000000, 0.47167023, 0.20000000],
      [0.49975705, 0.50000000, 0.30396056, 0.26904690, 0.20000000],
      [0.20000000, 0.41085612, 0.50000000, 0.20000000, 0.50000000],
      [0.50000000, 0.20000000, 0.20000000, 0.40758517, 0.50000000],
      [0.20000000, 0.50000000, 0.50000000, 0.50000000, 0.20000000],
    ])
    assertEqual(clippedX, expectedClippedX, accuracy: 0.0001)
  }

  func testRsqrt() {
    let x = Tensor<Double>([1, 0.25, 1.0 / 9.0, 0.0625, 0.04])
    let target = Tensor<Double>([1, 2, 3, 4, 5]).sum()
    let gradTarget = Tensor<Double>([-0.5, -4.0, -13.5, -32.0, -62.5])
    let (value, grad) = valueWithGradient(at: x) { rsqrt($0).sum() }
    XCTAssertEqual(value, target)
    XCTAssertEqual(grad, gradTarget)
  }

  func testLog1p() {
    let x = Tensor<Float>([1, 2, 3, 4, 5])
    let y = log1p(x)
    let expectedY = Tensor<Float>([0.69315, 1.09861, 1.38629, 1.60944, 1.79176])
    assertEqual(y, expectedY, accuracy: 0.0001)
  }

  func testLog1mexp() {
    let x = Tensor<Float>([-1, -2, -3, -4, -5])
    let y = log1mexp(x)
    let expectedY = Tensor<Float>([-0.45868, -0.14541, -0.05107, -0.01849, -0.00676])
    assertEqual(y, expectedY, accuracy: 0.0001)
  }

  func testExpm1() {
    let x = Tensor<Float>([1, 2, 3, 4, 5])
    let y = expm1(x)
    let expectedY = Tensor<Float>([1.71828, 6.38906, 19.08554, 53.59815, 147.41316])
    assertEqual(y, expectedY, accuracy: 0.0001)
  }

  func testSign() {
    let x = Tensor<Float>([[1, 2, -3, 4, 5], [1, 2, 3, 4, -5]])
    let y = sign(x)
    XCTAssertEqual(y, Tensor<Float>([[1, 1, -1, 1, 1], [1, 1, 1, 1, -1]]))
  }

  func testLogSigmoid() {
    let x = Tensor<Float>([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    let y = logSigmoid(x)
    assertEqual(y, log(sigmoid(x)), accuracy: 0.0001)
  }

  func testSoftplus() {
    let x = Tensor<Float>([1.0, 2.0, 3.0])
    let y = softplus(x)
    let expectedY = Tensor<Float>([1.3132616, 2.126928, 3.0485873])
    XCTAssertEqual(y, expectedY)
  }

  func testSoftsign() {
    let x = Tensor<Float>([1.0, 4.0, 3.0])
    let y = softsign(x)
    let expectedY = Tensor<Float>([0.5, 0.8, 0.75])
    XCTAssertEqual(y, expectedY)
  }

  func testElu() {
    let x = Tensor<Float>([-1.0, 2.0, 3.0])
    let y = elu(x)
    let expectedY = Tensor<Float>([-0.63212055, 2, 3])
    XCTAssertEqual(y, expectedY)
  }

  func testGelu() {
    let x = Tensor<Float>([2.0, 1.0, 7.0])
    let y = gelu(x)
    let expectedY = Tensor<Float>([1.95459769, 0.84119199, 7.0])
    XCTAssertEqual(y, expectedY)
  }

  func testRelu() {
    let x = Tensor<Float>([[-1.0, 2.0, 3.0]])
    let y = relu(x)
    let expectedY = Tensor<Float>([[0.0, 2.0, 3.0]])
    XCTAssertEqual(y, expectedY)
  }

  func testRelu6() {
    let x = Tensor<Float>([1.0, -2.0, 3.0, 4.0, 10.0])
    let y = relu6(x)
    let expectedY = Tensor<Float>([1.0, 0, 3.0, 4.0, 6.0])
    XCTAssertEqual(y, expectedY)
  }

  func testLeakyRelu() {
    let x = Tensor<Float>([[-1.0, 2.0, 3.0]])
    let y = leakyRelu(x, alpha: 0.4)
    let expectedY = Tensor<Float>([[-0.4, 2.0, 3.0]])
    XCTAssertEqual(y, expectedY)
  }

  func testSelu() {
    let x = Tensor<Float>([[-1.0, 2.0, 3.0]])
    let y = selu(x)
    let expectedY = Tensor<Float>([-1.111331, 2.101402, 3.152103])
    assertEqual(y, expectedY, accuracy: 1e-5)
  }

  func testSwish() {
    let x = Tensor<Float>([[-1.0, 2.0, 3.0]])
    let y = swish(x)
    let expectedY = Tensor<Float>([-0.26894143, 1.761594, 2.8577223])
    assertEqual(y, expectedY, accuracy: 1e-5)
  }

  func testHardSigmoid() {
    let x = Tensor<Float>([-4, -2, 0, 2, 4])
    let y = hardSigmoid(x)
    let expectedY = Tensor<Float>([0.0, 0.16666667, 0.5, 0.8333333, 1.0])
    assertEqual(y, expectedY, accuracy: 1e-5)
  }

  func testHardSwish() {
    let x = Tensor<Float>([-4, -2, 0, 2, 4])
    let y = hardSwish(x)
    let expectedY = Tensor<Float>([0.0, -0.33333334, 0.0, 1.6666666, 4.0])
    assertEqual(y, expectedY, accuracy: 1e-5)
  }

  func testIsFinite() {
    let x = Tensor<Float>([1, 2, 3, 4, -Float.infinity])
    let y = x.isFinite
    XCTAssertEqual(y, Tensor([true, true, true, true, false]))
  }

  func testIsInfinite() {
    let x = Tensor<Float>([1, 2, 3, 4, log(0.0)])
    let y = x.isInfinite
    XCTAssertEqual(y, Tensor([false, false, false, false, true]))
  }

  func testIsNaN() {
    let x = Tensor<Float>([1, 2, 3, 4, log(-5.0)])
    let y = x.isNaN
    XCTAssertEqual(y, Tensor([false, false, false, false, true]))
  }

  func testCosineSimilarity() {
    let x = Tensor<Float>([1, 2, 3, 4, 5, 6, 7, 8])
    let y = Tensor<Float>([0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    let z = cosineSimilarity(x, y)
    let output: Float = 1.0
    XCTAssertEqual(z, Tensor(output))
  }

  func testCosineDistance() {
    let x = Tensor<Float>([7.0])
    let y = Tensor<Float>([8.0])
    let z = cosineDistance(x, y)
    let output: Float = 0.0
    XCTAssertEqual(z, Tensor(output))
  }

  func testArgmax() {
    // 2 x 3
    let x = Tensor<Float>([[0, 1, 2], [3, 4, 5]])
    let argmax0 = x.argmax(squeezingAxis: 0)
    let argmax1 = x.argmax(squeezingAxis: 1)
    let scalarsArgmax = x.argmax()
    XCTAssertEqual(argmax0.array, ShapedArray(shape: [3], scalars: [1, 1, 1]))
    XCTAssertEqual(argmax1.array, ShapedArray(shape: [2], scalars: [2, 2]))
    XCTAssertEqual(scalarsArgmax.array, ShapedArray(shape: [], scalars: [5]))
  }

  func testReduction() {
    // 2 x 5
    let x = Tensor<Float>([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    XCTAssertEqual(x.sum(), Tensor(30))
    XCTAssertEqual(
      x.sum(squeezingAxes: 0),
      Tensor(shape: [5], scalars: [2, 4, 6, 8, 10]))
    XCTAssertEqual(
      x.sum(alongAxes: 0),
      Tensor(shape: [1, 5], scalars: [2, 4, 6, 8, 10]))

    XCTAssertEqual(x.product(), Tensor(14400))
    XCTAssertEqual(
      x.product(squeezingAxes: 0),
      Tensor(shape: [5], scalars: [1, 4, 9, 16, 25]))
    XCTAssertEqual(
      x.product(alongAxes: 0),
      Tensor(shape: [1, 5], scalars: [1, 4, 9, 16, 25]))

    XCTAssertEqual(x.mean(), Tensor(3))
    XCTAssertEqual(
      x.mean(squeezingAxes: 0),
      Tensor(shape: [5], scalars: [1, 2, 3, 4, 5]))
    XCTAssertEqual(
      x.mean(alongAxes: 0),
      Tensor(shape: [1, 5], scalars: [1, 2, 3, 4, 5]))
    XCTAssertEqual(
      x.mean(squeezingAxes: 1),
      Tensor(shape: [2], scalars: [3, 3]))
    XCTAssertEqual(
      x.mean(alongAxes: 1),
      Tensor(shape: [2, 1], scalars: [3, 3]))

    XCTAssertEqual(x.variance(), Tensor(2))
    XCTAssertEqual(
      x.variance(squeezingAxes: 0),
      Tensor(shape: [5], scalars: [0, 0, 0, 0, 0]))
    XCTAssertEqual(
      x.variance(alongAxes: 0),
      Tensor(shape: [1, 5], scalars: [0, 0, 0, 0, 0]))
    XCTAssertEqual(
      x.variance(squeezingAxes: 1),
      Tensor(shape: [2], scalars: [2, 2]))
    XCTAssertEqual(
      x.variance(alongAxes: 1),
      Tensor(shape: [2, 1], scalars: [2, 2]))
  }

  func testCumulativeSum() {
    // 2 x 3
    let x = Tensor<Float>([[0, 1, 2], [3, 4, 5]])
    let cumsum0 = x.cumulativeSum(alongAxis: 0)
    let cumsum1 = x.cumulativeSum(alongAxis: 1)
    let exclusiveCumsum0 = x.cumulativeSum(alongAxis: 0, exclusive: true)
    let exclusiveCumsum1 = x.cumulativeSum(alongAxis: 1, exclusive: true)
    let reverseCumsum0 = x.cumulativeSum(alongAxis: 0, reverse: true)
    let reverseCumsum1 = x.cumulativeSum(alongAxis: 1, reverse: true)
    let reverseExclusiveCumsum0 = x.cumulativeSum(alongAxis: 0, exclusive: true, reverse: true)
    let reverseExclusiveCumsum1 = x.cumulativeSum(alongAxis: 1, exclusive: true, reverse: true)
    XCTAssertEqual(cumsum0, Tensor<Float>([[0, 1, 2], [3, 5, 7]]))
    XCTAssertEqual(cumsum1, Tensor<Float>([[0, 1, 3], [3, 7, 12]]))
    XCTAssertEqual(exclusiveCumsum0, Tensor<Float>([[0, 0, 0], [0, 1, 2]]))
    XCTAssertEqual(exclusiveCumsum1, Tensor<Float>([[0, 0, 1], [0, 3, 7]]))
    XCTAssertEqual(reverseCumsum0, Tensor<Float>([[3, 5, 7], [3, 4, 5]]))
    XCTAssertEqual(reverseCumsum1, Tensor<Float>([[3, 3, 2], [12, 9, 5]]))
    XCTAssertEqual(reverseExclusiveCumsum0, Tensor<Float>([[3, 4, 5], [0, 0, 0]]))
    XCTAssertEqual(reverseExclusiveCumsum1, Tensor<Float>([[3, 2, 0], [9, 5, 0]]))
  }

  func testCumulativeProduct() {
    // 2 x 3
    let x = Tensor<Float>([[0, 1, 2], [3, 4, 5]])
    let cumprod0 = x.cumulativeProduct(alongAxis: 0)
    let cumprod1 = x.cumulativeProduct(alongAxis: 1)
    let exclusiveCumprod0 = x.cumulativeProduct(alongAxis: 0, exclusive: true)
    let exclusiveCumprod1 = x.cumulativeProduct(alongAxis: 1, exclusive: true)
    let reverseCumprod0 = x.cumulativeProduct(alongAxis: 0, reverse: true)
    let reverseCumprod1 = x.cumulativeProduct(alongAxis: 1, reverse: true)
    let reverseExclusiveCumprod0 = x.cumulativeProduct(
      alongAxis: 0,
      exclusive: true,
      reverse: true)
    let reverseExclusiveCumprod1 = x.cumulativeProduct(
      alongAxis: 1,
      exclusive: true,
      reverse: true)
    XCTAssertEqual(cumprod0, Tensor<Float>([[0, 1, 2], [0, 4, 10]]))
    XCTAssertEqual(cumprod1, Tensor<Float>([[0, 0, 0], [3, 12, 60]]))
    XCTAssertEqual(exclusiveCumprod0, Tensor<Float>([[1, 1, 1], [0, 1, 2]]))
    XCTAssertEqual(exclusiveCumprod1, Tensor<Float>([[1, 0, 0], [1, 3, 12]]))
    XCTAssertEqual(reverseCumprod0, Tensor<Float>([[0, 4, 10], [3, 4, 5]]))
    XCTAssertEqual(reverseCumprod1, Tensor<Float>([[0, 2, 2], [60, 20, 5]]))
    XCTAssertEqual(reverseExclusiveCumprod0, Tensor<Float>([[3, 4, 5], [1, 1, 1]]))
    XCTAssertEqual(reverseExclusiveCumprod1, Tensor<Float>([[2, 2, 1], [20, 5, 1]]))
  }

  func testStandardDeviation() {
    XCTAssertEqual(Tensor<Float>([1]).standardDeviation(), Tensor(0))
    XCTAssertEqual(Tensor<Float>([0, 1]).standardDeviation(alongAxes: 0), Tensor([0.5]))
    XCTAssertEqual(Tensor<Float>([0, 1]).standardDeviation(), Tensor(0.5))
    XCTAssertEqual(
      Tensor<Float>(rangeFrom: 0, to: 10, stride: 1).standardDeviation().scalarized(),
      2.87228132,
      accuracy: 0.001)
    let matrix = Tensor<Float>(rangeFrom: 0, to: 10, stride: 1).reshaped(to: [2, 5])
    XCTAssertEqual(matrix.standardDeviation().scalarized(), 2.87228132, accuracy: 0.001)
    let values = matrix.standardDeviation(alongAxes: 1).array.scalars
    XCTAssertEqual(Double(values[0]), 1.4142, accuracy: 0.0001)
    XCTAssertEqual(Double(values[1]), 1.4142, accuracy: 0.0001)
  }

  func testLogSumExp() {
    let x = Tensor<Float>([
      [0.45031791, 0.41123222, 0.53928467, 0.47167023, 0.15483777],
      [0.49975705, 0.71807549, 0.30396056, 0.2690469, 0.01404393],
      [0.16950939, 0.41085612, 0.79503016, 0.11977817, 0.99728241],
      [0.62510073, 0.17344792, 0.1540605, 0.40758517, 0.93683817],
      [0.15653343, 0.50502756, 0.99365925, 0.84617581, 0.17422509],
    ])
    let y0 = x.logSumExp()
    let y1 = x.logSumExp(squeezingAxes: 1)
    let y2 = x.logSumExp(alongAxes: 1)
    let expectedY0 = Tensor<Float>(3.713885997817954)
    let expectedY1 = Tensor<Float>(
      [2.02318908, 1.99835067, 2.16853826, 2.1137799, 2.20261244])
    let expectedY2 = Tensor<Float>(
      [[2.02318908], [1.99835067], [2.16853826], [2.1137799], [2.20261244]])
    assertEqual(y0, expectedY0, accuracy: 0.0001)
    assertEqual(y1, expectedY1, accuracy: 0.0001)
    assertEqual(y2, expectedY2, accuracy: 0.0001)

    let xSmall = Tensor<Float>([
      -301.9475, -265.2244, -275.77475, -235.28029, -277.2509, -396.6921, -400.01385,
    ])
    let ySmall = xSmall.logSumExp()
    let expectedYSmall = Tensor<Float>(-235.28029)
    assertEqual(ySmall, expectedYSmall, accuracy: 0.0001)
  }

  func testMoments() {
    let x = Tensor<Float>([
      [0.45031791, 0.41123222, 0.53928467, 0.47167023, 0.15483777],
      [0.49975705, 0.71807549, 0.30396056, 0.2690469, 0.01404393],
      [0.16950939, 0.41085612, 0.79503016, 0.11977817, 0.99728241],
      [0.62510073, 0.17344792, 0.1540605, 0.40758517, 0.93683817],
      [0.15653343, 0.50502756, 0.99365925, 0.84617581, 0.17422509],
    ])
    let moments = x.moments()
    let moments0 = x.moments(alongAxes: 0)
    let moments1 = x.moments(alongAxes: 1)
    let expectedMoments = Moments(
      mean: Tensor<Float>(0.4518935),
      variance: Tensor<Float>(0.0829807))
    let expectedMoments0 = Moments(
      mean: Tensor<Float>([0.3802437, 0.44372786, 0.55719903, 0.42285126, 0.45544547]),
      variance: Tensor<Float>([0.03472081, 0.03084241, 0.0948065, 0.05946582, 0.17792228]))
    let expectedMoments1 = Moments(
      mean: Tensor<Float>([0.40546856, 0.36097679, 0.49849125, 0.4594065, 0.53512423]),
      variance: Tensor<Float>([0.01742998, 0.05576876, 0.1192121, 0.0866179, 0.11629849]))
    assertEqual(moments.mean, expectedMoments.mean, accuracy: 0.0001)
    assertEqual(moments.variance, expectedMoments.variance, accuracy: 0.0001)
    assertEqual(moments0.mean, expectedMoments0.mean, accuracy: 0.0001)
    assertEqual(moments0.variance, expectedMoments0.variance, accuracy: 0.0001)
    assertEqual(moments1.mean, expectedMoments1.mean, accuracy: 0.0001)
    assertEqual(moments1.variance, expectedMoments1.variance, accuracy: 0.0001)
  }

  func testCeilAndFloor() {
    let x = Tensor<Float>([-1.3, -0.4, 0.5, 1.6])
    let xFloor = floor(x)
    let xCeil = ceil(x)
    XCTAssertEqual(xFloor.array, ShapedArray(shape: [4], scalars: [-2, -1, 0, 1]))
    XCTAssertEqual(xCeil.array, ShapedArray(shape: [4], scalars: [-1, 0, 1, 2]))
  }

  func testSimpleMath() {
    let x = Tensor<Float>([1.2, 1.2])
    let y = tanh(x)
    let array = y.array
    XCTAssertEqual([2], array.shape)
    XCTAssertEqual(Double(array.scalars[0]), 0.833655, accuracy: 0.0001)
    XCTAssertEqual(Double(array.scalars[1]), 0.833655, accuracy: 0.0001)
  }

  func test3Adds() {
    let a = Tensor<Float>([1])
    let b = Tensor<Float>([2])
    let c = Tensor<Float>([3])

    let o = a + b + c
    XCTAssertEqual(o.scalars, [6])
  }

  func testMultiOpMath() {
    let x = Tensor<Float>([1.2, 1.2])
    let y = Tensor<Float>([2.4, 2.4])
    let t1 = x + y
    let t2 = t1 * t1
    let t3 = sqrt(t2)

    let array1 = t1.array
    let array2 = t2.array
    let array3 = t3.array
    XCTAssertEqual(array1.shape, [2])
    XCTAssertEqual(array2.shape, [2])
    XCTAssertEqual(array3.shape, [2])
    XCTAssertEqual(Double(array1.scalars[0]), 3.6, accuracy: 0.0001)
    XCTAssertEqual(Double(array1.scalars[1]), 3.6, accuracy: 0.0001)
    XCTAssertEqual(Double(array2.scalars[0]), 12.96, accuracy: 0.0001)
    XCTAssertEqual(Double(array2.scalars[1]), 12.96, accuracy: 0.0001)
    XCTAssertEqual(Double(array3.scalars[0]), 3.6, accuracy: 0.0001)
    XCTAssertEqual(Double(array3.scalars[1]), 3.6, accuracy: 0.0001)
  }

  func testXWPlusB() {
    // Shape: 1 x 4
    let x = Tensor<Float>([[1.0, 2.0, 2.0, 1.0]])
    // Shape: 4 x 2
    let w = Tensor<Float>([[1.0, 0.0], [3.0, 0.0], [2.0, 3.0], [1.0, 0.0]])
    // Shape: 2
    let b = Tensor<Float>([0.5, 0.5])
    // Shape: 1 x 2 (broadcasted)
    let result = matmul(x, w) + b
    XCTAssertEqual(result.shape, [1, 2])
    XCTAssertEqual(result.scalars, [12.5, 6.5])
  }

  func testXORInference() {
    func xor(_ x: Float, _ y: Float) -> Float {
      let x = Tensor<Float>([x, y]).reshaped(to: [1, 2])

      // FIXME: If params are declared outside of `xor`, it would crash.
      // 2 x 4
      let w1 = Tensor<Float>(
        [
          [-1.83586664, -0.20809225, 0.47667537, 1.90780607],
          [-1.83523219, -0.51167348, 0.15490439, 1.91018065],
        ])
      // 1 x 4
      let b1 = Tensor<Float>([[2.54353216, 0.25132703, -0.16503136, -0.85754058]])
      // 4 x 1
      let w2 = Tensor<Float>([[3.04350065], [0.35590511], [-0.3252157], [3.49349223]])
      // 1 x 1
      let b2 = Tensor<Float>([[-0.74635993]])

      let o1 = tanh(matmul(x, w1) + b1)
      let y = tanh(matmul(o1, w2) + b2)
      return y.array.scalars[0]  // TODO: use better scalar getter
    }

    XCTAssertEqual(xor(0.0, 0.0), 0.0, accuracy: 0.1)
    XCTAssertEqual(xor(0.0, 1.0), 1.0, accuracy: 0.1)
    XCTAssertEqual(xor(1.0, 0.0), 1.0, accuracy: 0.1)
    XCTAssertEqual(xor(1.0, 1.0), 0.0, accuracy: 0.1)
  }

  func testMLPClassifierStruct() {
    struct MLPClassifier {
      // 2 x 4
      var w1 = Tensor<Float>([
        [1.0, 0.8, 0.4, 0.4],
        [0.4, 0.3, 0.2, 0.1],
      ])
      // 4 x 1
      var w2 = Tensor<Float>([[0.4], [0.4], [0.3], [0.9]])
      var b1 = Tensor<Float>(zeros: [1, 4])
      var b2 = Tensor<Float>(zeros: [1, 1])

      func prediction(for x: Tensor<Float>) -> Tensor<Float> {
        let o1 = tanh(matmul(x, w1) + b1)
        return tanh(matmul(o1, w2) + b2)
      }
    }

    let input = Tensor<Float>([[1, 0.5]])
    let classifier = MLPClassifier()
    let prediction = classifier.prediction(for: input)
    XCTAssertEqual(Double(prediction.scalars[0]), 0.816997, accuracy: 0.0001)
  }

  func testBroadcastedAddGradient() {
    func foo(_ x: Tensor<Float>, _ y: Tensor<Float>) -> Tensor<Float> {
      return (x + y).sum()
    }
    let x = Tensor<Float>(ones: [1, 2, 1, 4])
    let y = Tensor<Float>(ones: [4, 1, 3, 1])
    let (dx, dy) = gradient(at: x, y, in: foo)
    XCTAssertEqual(x.shape, dx.shape)
    XCTAssertEqual(y.shape, dy.shape)
  }

  static var allTests = [
    ("testElementaryFunctions", testElementaryFunctions),
    ("testAbs", testAbs),
    ("testSquaredDifference", testSquaredDifference),
    ("testZeros", testZeros),
    ("testLogSoftmax", testLogSoftmax),
    ("testMax", testMax),
    ("testMin", testMin),
    ("testRound", testRound),
    ("testSoftmax", testSoftmax),
    ("testSigmoid", testSigmoid),
    ("testIdentity", testIdentity),
    ("testClipping", testClipping),
    ("testRsqrt", testRsqrt),
    ("testLog1p", testLog1p),
    ("testLog1mexp", testLog1mexp),
    ("testExpm1", testExpm1),
    ("testSign", testSign),
    ("testLogSigmoid", testLogSigmoid),
    ("testSoftplus", testSoftplus),
    ("testSoftsign", testSoftsign),
    ("testElu", testElu),
    ("testGelu", testGelu),
    ("testRelu", testRelu),
    ("testRelu6", testRelu6),
    ("testLeakyRelu", testLeakyRelu),
    ("testSelu", testSelu),
    ("testSwish", testSwish),
    ("testHardSigmoid", testHardSigmoid),
    ("testHardSwish", testHardSwish),
    ("testIsFinite", testIsFinite),
    ("testIsInfinite", testIsInfinite),
    ("testIsNaN", testIsNaN),
    ("testCosineSimilarity", testCosineSimilarity),
    ("testCosineDistance", testCosineDistance),
    ("testArgmax", testArgmax),
    ("testReduction", testReduction),
    ("testCumulativeSum", testCumulativeSum),
    ("testCumulativeProduct", testCumulativeProduct),
    ("testStandardDeviation", testStandardDeviation),
    ("testLogSumExp", testLogSumExp),
    ("testMoments", testMoments),
    ("testCeilAndFloor", testCeilAndFloor),
    ("testSimpleMath", testSimpleMath),
    ("test3Adds", test3Adds),
    ("testMultiOpMath", testMultiOpMath),
    ("testXWPlusB", testXWPlusB),
    ("testXORInference", testXORInference),
    ("testMLPClassifierStruct", testMLPClassifierStruct),
    ("testBroadcastedAddGradient", testBroadcastedAddGradient),
  ]
}
