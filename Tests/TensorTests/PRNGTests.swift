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

@testable import Tensor

final class PRNGTests: XCTestCase {
  func testARC4() {
    do {
      let _ = ARC4RandomNumberGenerator(seed: [0])
      let _ = ARC4RandomNumberGenerator(seed: [1, 2, 3, 4, 5, 6, 7])
      let _ = ARC4RandomNumberGenerator(seed: Array(repeating: 255, count: 256))
      var rng = ARC4RandomNumberGenerator(seed: [1, 2, 3, 4, 5, 6, 7, 8])
      XCTAssertEqual(rng.next(), 0x97ab_8a1b_f0af_b961)
    }
    do {
      var rng = ARC4RandomNumberGenerator(
        seed: [
          0x1a, 0xda, 0x31, 0xd5, 0xcf, 0x68, 0x82, 0x21,
          0xc1, 0x09, 0x16, 0x39, 0x08, 0xeb, 0xe5, 0x1d,
          0xeb, 0xb4, 0x62, 0x27, 0xc6, 0xcc, 0x8b, 0x37,
          0x64, 0x19, 0x10, 0x83, 0x32, 0x22, 0x77, 0x2a,
        ])
      for _ in 0..<512 {
        _ = rng.next()
      }
      XCTAssertEqual(rng.next(), 0x370b_1c1f_e655_916d)
    }
    do {
      // Copy should not break original.
      var rng1 = ARC4RandomNumberGenerator(
        seed: [
          0x1a, 0xda, 0x31, 0xd5, 0xcf, 0x68, 0x82, 0x21,
          0xc1, 0x09, 0x16, 0x39, 0x08, 0xeb, 0xe5, 0x1d,
          0xeb, 0xb4, 0x62, 0x27, 0xc6, 0xcc, 0x8b, 0x37,
          0x64, 0x19, 0x10, 0x83, 0x32, 0x22, 0x77, 0x2a,
        ])
      for _ in 0..<256 {
        _ = rng1.next()
      }
      var rng2 = rng1
      for _ in 0..<1000 {
        _ = rng1.next()
      }
      for _ in 0..<256 {
        _ = rng2.next()
      }
      XCTAssertEqual(rng2.next(), 0x370b_1c1f_e655_916d)
    }
    // Performance test.
    do {
      var arc4 = ARC4RandomNumberGenerator(seed: 971_626_482_267_121)
      measure {
        for _ in 0..<1000 {
          _ = arc4.next()
        }
      }
    }
  }

  func testUniformDistribution() {
    do {
      // Uniform distribution is in range.
      var rng = ARC4RandomNumberGenerator(seed: UInt64(42))
      let dist = UniformFloatingPointDistribution<Double>(lowerBound: 10, upperBound: 42)
      for _ in 0..<1000 {
        let r = dist.next(using: &rng)
        XCTAssertGreaterThan(r, 10)
        XCTAssertLessThan(r, 42)
      }
    }
    do {
      var rng = ARC4RandomNumberGenerator(seed: UInt64(42))
      let dist = UniformFloatingPointDistribution<Double>(lowerBound: 10, upperBound: 50)
      let count = 100000
      var mean: Double = 0
      for _ in 0..<count {
        mean += dist.next(using: &rng)
      }
      mean /= Double(count)
      XCTAssertEqual(mean, 30, accuracy: 0.25)
    }
    do {
      var rng = ARC4RandomNumberGenerator(seed: UInt64(42))
      let dist = UniformFloatingPointDistribution<Double>(lowerBound: 10, upperBound: 50)
      let count = 100000
      var mean: Double = 0
      var meanSquare: Double = 0
      for _ in 0..<count {
        let r = dist.next(using: &rng)
        mean += r
        meanSquare += r * r
      }
      mean /= Double(count)
      meanSquare /= Double(count)
      let stdDev = (meanSquare - mean * mean).squareRoot()
      XCTAssertEqual(stdDev, (50 - 10) / 12.squareRoot(), accuracy: 0.25)
    }
  }

  func testNormalDistribution() {
    do {
      var rng = ARC4RandomNumberGenerator(seed: UInt64(42))
      let dist = NormalDistribution<Double>(mean: 10, standardDeviation: 50)
      let count = 100000
      var mean: Double = 0
      for _ in 0..<count {
        mean += dist.next(using: &rng)
      }
      mean /= Double(count)
      XCTAssertEqual(mean, 10, accuracy: 0.25)
    }
    do {
      var rng = ARC4RandomNumberGenerator(seed: UInt64(42))
      let dist = NormalDistribution<Double>(mean: 10, standardDeviation: 50)
      let count = 100000
      var mean: Double = 0
      var meanSquare: Double = 0
      for _ in 0..<count {
        let r = dist.next(using: &rng)
        mean += r
        meanSquare += r * r
      }
      mean /= Double(count)
      meanSquare /= Double(count)
      let stdDev = (meanSquare - mean * mean).squareRoot()
      XCTAssertEqual(stdDev, 50, accuracy: 0.25)
    }
  }

  func testUniformIntegerDistribution() {
    do {
      var rng = ARC4RandomNumberGenerator(seed: UInt64(42))
      let dist = UniformIntegerDistribution<UInt16>()
      let count = 100000
      var mean: Double = 0
      for _ in 0..<count {
        mean += Double(dist.next(using: &rng))
      }
      mean /= Double(count)
      XCTAssertEqual(mean, pow(2.0, 15.0), accuracy: 1000)
    }
    do {
      var rng = ARC4RandomNumberGenerator(seed: UInt64(42))
      let dist = UniformIntegerDistribution<UInt16>()
      let count = 100000
      var mean: Double = 0
      var meanSquare: Double = 0
      for _ in 0..<count {
        let r = dist.next(using: &rng)
        mean += Double(r)
        meanSquare += Double(r) * Double(r)
      }
      mean /= Double(count)
      meanSquare /= Double(count)
      let stdDev = (meanSquare - mean * mean).squareRoot()
      XCTAssertEqual(stdDev, pow(2.0, 16.0) / 12.squareRoot(), accuracy: 1000)
    }
  }

  func testThreefry() {
    // Check the PRNG output given different seeds against reference values.
    // This guards against accidental changes to the generator which should
    // result in changes to its output.
    var generator = ThreefryRandomNumberGenerator(uint64Seed: 123_159_812)
    XCTAssertEqual(generator.next(), 9_411_952_874_433_594_703)
    XCTAssertEqual(generator.next(), 6_992_100_569_504_761_807)
    XCTAssertEqual(generator.next(), 6_249_442_510_280_393_663)
    XCTAssertEqual(generator.next(), 13_096_801_615_464_354_606)
    XCTAssertEqual(generator.next(), 671_569_830_597_624_217)
    XCTAssertEqual(generator.next(), 9_499_516_613_746_162_591)
    XCTAssertEqual(generator.next(), 14_104_268_727_839_198_528)
    XCTAssertEqual(generator.next(), 2_729_105_059_420_396_781)

    generator = ThreefryRandomNumberGenerator(uint64Seed: 58_172_950_819_076)
    XCTAssertEqual(generator.next(), 8_181_320_043_134_006_362)
    XCTAssertEqual(generator.next(), 14_375_459_274_817_572_790)
    XCTAssertEqual(generator.next(), 1_051_151_592_956_420_496)
    XCTAssertEqual(generator.next(), 12_482_694_246_229_339_388)
    XCTAssertEqual(generator.next(), 2_543_901_658_316_819_773)
    XCTAssertEqual(generator.next(), 54_584_659_268_457_468)
    XCTAssertEqual(generator.next(), 4_068_621_515_934_625_604)
    XCTAssertEqual(generator.next(), 10_604_176_710_283_101_491)

    // Performance test.
    do {
      var philox = PhiloxRandomNumberGenerator(uint64Seed: 971_626_482_267_121)
      measure {
        for _ in 0..<1000 {
          _ = philox.next()
        }
      }
    }
  }

  func testPhilox() {
    // Check the PRNG output given different seeds against reference values.
    // This guards against accidental changes to the generator which should
    // result in changes to its output.
    var generator = PhiloxRandomNumberGenerator(uint64Seed: 971_626_482_267_121)
    XCTAssertEqual(generator.next(), 13_938_684_859_108_683_724)
    XCTAssertEqual(generator.next(), 14_733_436_676_625_682_935)
    XCTAssertEqual(generator.next(), 6_775_200_690_501_958_369)
    XCTAssertEqual(generator.next(), 4_888_384_230_122_468_581)
    XCTAssertEqual(generator.next(), 9_929_469_809_262_837_771)
    XCTAssertEqual(generator.next(), 4_887_275_522_116_356_711)
    XCTAssertEqual(generator.next(), 10_098_896_320_274_145_012)
    XCTAssertEqual(generator.next(), 8_966_522_427_706_988_112)

    generator = PhiloxRandomNumberGenerator(uint64Seed: 708_165_273_787)
    XCTAssertEqual(generator.next(), 17_296_679_597_944_579_603)
    XCTAssertEqual(generator.next(), 16_698_752_516_857_890_287)
    XCTAssertEqual(generator.next(), 8_389_709_598_422_976_467)
    XCTAssertEqual(generator.next(), 11_475_723_713_423_213_818)
    XCTAssertEqual(generator.next(), 11_475_016_682_221_315_199)
    XCTAssertEqual(generator.next(), 15_780_739_321_597_004_611)
    XCTAssertEqual(generator.next(), 1_610_199_061_186_607_604)
    XCTAssertEqual(generator.next(), 5_793_355_800_212_150_215)

    // Performance test.
    do {
      var threefry = ThreefryRandomNumberGenerator(uint64Seed: 971_626_482_267_121)
      measure {
        for _ in 0..<1000 {
          _ = threefry.next()
        }
      }
    }
  }

  static var allTests = [
    ("testARC4", testARC4),
    ("testUniformDistribution", testUniformDistribution),
    ("testNormalDistribution", testNormalDistribution),
    ("testUniformIntegerDistribution", testUniformIntegerDistribution),
    ("testThreefry", testThreefry),
    ("testPhilox", testPhilox),
  ]
}
