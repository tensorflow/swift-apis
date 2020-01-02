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

import XCTest
@testable import Tensor

final class PRNGTests: XCTestCase {
    func testARC4() {
        do {
            let _ = ARC4RandomNumberGenerator(seed: [0])
            let _ = ARC4RandomNumberGenerator(seed: [1, 2, 3, 4, 5, 6, 7])
            let _ = ARC4RandomNumberGenerator(seed: Array(repeating: 255, count: 256))
            var rng = ARC4RandomNumberGenerator(seed: [1, 2, 3, 4, 5, 6, 7, 8])
            XCTAssertEqual(rng.next(), 0x97ab8a1bf0afb961)
        }
        do {
            var rng = ARC4RandomNumberGenerator(
                seed: [0x1a, 0xda, 0x31, 0xd5, 0xcf, 0x68, 0x82, 0x21,
                       0xc1, 0x09, 0x16, 0x39, 0x08, 0xeb, 0xe5, 0x1d,
                       0xeb, 0xb4, 0x62, 0x27, 0xc6, 0xcc, 0x8b, 0x37,
                       0x64, 0x19, 0x10, 0x83, 0x32, 0x22, 0x77, 0x2a])
            for _ in 0..<512 {
                _ = rng.next()
            }
            XCTAssertEqual(rng.next(), 0x370b1c1fe655916d)
        }
        do {
            // Copy should not break original.
            var rng1 = ARC4RandomNumberGenerator(
                seed: [0x1a, 0xda, 0x31, 0xd5, 0xcf, 0x68, 0x82, 0x21,
                       0xc1, 0x09, 0x16, 0x39, 0x08, 0xeb, 0xe5, 0x1d,
                       0xeb, 0xb4, 0x62, 0x27, 0xc6, 0xcc, 0x8b, 0x37,
                       0x64, 0x19, 0x10, 0x83, 0x32, 0x22, 0x77, 0x2a])
            for _ in 0 ..< 256 {
                _ = rng1.next()
            }
            var rng2 = rng1
            for _ in 0 ..< 1000 {
                _ = rng1.next()
            }
            for _ in 0 ..< 256 {
                _ = rng2.next()
            }
            XCTAssertEqual(rng2.next(), 0x370b1c1fe655916d)
        }
        // Performance test.
        do {
            var arc4 = ARC4RandomNumberGenerator(seed: 971626482267121)
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
            for _ in 0 ..< 1000 {
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
            for _ in 0 ..< count {
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
            for _ in 0 ..< count {
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
            for _ in 0 ..< count {
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
            for _ in 0 ..< count {
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
            for _ in 0 ..< count {
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
            for _ in 0 ..< count {
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
        var generator = ThreefryRandomNumberGenerator(uint64Seed: 123159812)
        XCTAssertEqual(generator.next(), 9411952874433594703)
        XCTAssertEqual(generator.next(), 6992100569504761807)
        XCTAssertEqual(generator.next(), 6249442510280393663)
        XCTAssertEqual(generator.next(), 13096801615464354606)
        XCTAssertEqual(generator.next(), 671569830597624217)
        XCTAssertEqual(generator.next(), 9499516613746162591)
        XCTAssertEqual(generator.next(), 14104268727839198528)
        XCTAssertEqual(generator.next(), 2729105059420396781)

        generator = ThreefryRandomNumberGenerator(uint64Seed: 58172950819076)
        XCTAssertEqual(generator.next(), 8181320043134006362)
        XCTAssertEqual(generator.next(), 14375459274817572790)
        XCTAssertEqual(generator.next(), 1051151592956420496)
        XCTAssertEqual(generator.next(), 12482694246229339388)
        XCTAssertEqual(generator.next(), 2543901658316819773)
        XCTAssertEqual(generator.next(), 54584659268457468)
        XCTAssertEqual(generator.next(), 4068621515934625604)
        XCTAssertEqual(generator.next(), 10604176710283101491)

        // Performance test.
        do {
            var philox = PhiloxRandomNumberGenerator(uint64Seed: 971626482267121)
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
        var generator = PhiloxRandomNumberGenerator(uint64Seed: 971626482267121)
        XCTAssertEqual(generator.next(), 13938684859108683724)
        XCTAssertEqual(generator.next(), 14733436676625682935)
        XCTAssertEqual(generator.next(), 6775200690501958369)
        XCTAssertEqual(generator.next(), 4888384230122468581)
        XCTAssertEqual(generator.next(), 9929469809262837771)
        XCTAssertEqual(generator.next(), 4887275522116356711)
        XCTAssertEqual(generator.next(), 10098896320274145012)
        XCTAssertEqual(generator.next(), 8966522427706988112)

        generator = PhiloxRandomNumberGenerator(uint64Seed: 708165273787)
        XCTAssertEqual(generator.next(), 17296679597944579603)
        XCTAssertEqual(generator.next(), 16698752516857890287)
        XCTAssertEqual(generator.next(), 8389709598422976467)
        XCTAssertEqual(generator.next(), 11475723713423213818)
        XCTAssertEqual(generator.next(), 11475016682221315199)
        XCTAssertEqual(generator.next(), 15780739321597004611)
        XCTAssertEqual(generator.next(), 1610199061186607604)
        XCTAssertEqual(generator.next(), 5793355800212150215)

        // Performance test.
        do {
            var threefry = ThreefryRandomNumberGenerator(uint64Seed: 971626482267121)
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
