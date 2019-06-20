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

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Darwin
#else
import Glibc
#endif

//===------------------------------------------------------------------------------------------===//
// Random Number Generators
//===------------------------------------------------------------------------------------------===//

/// A type that provides seedable deterministic pseudo-random data.
///
/// A SeedableRandomNumberGenerator can be used anywhere where a
/// RandomNumberGenerator would be used. It is useful when the pseudo-random
/// data needs to be reproducible across runs.
///
/// Conforming to the SeedableRandomNumberGenerator Protocol
/// ========================================================
///
/// To make a custom type conform to the `SeedableRandomNumberGenerator`
/// protocol, implement the `init(seed: [UInt8])` initializer, as well as the
/// requirements for `RandomNumberGenerator`. The values returned by `next()`
/// must form a deterministic sequence that depends only on the seed provided
/// upon initialization.
public protocol SeedableRandomNumberGenerator: RandomNumberGenerator {
    init(seed: [UInt8])
    init<T: BinaryInteger>(seed: T)
}

public extension SeedableRandomNumberGenerator {
    init<T: BinaryInteger>(seed: T) {
        var newSeed: [UInt8] = []
        for i in 0..<seed.bitWidth / UInt8.bitWidth {
            newSeed.append(UInt8(truncatingIfNeeded: seed >> (UInt8.bitWidth * i)))
        }
        self.init(seed: newSeed)
    }
}

/// An implementation of `SeedableRandomNumberGenerator` using ARC4.
///
/// ARC4 is a stream cipher that generates a pseudo-random stream of bytes. This
/// PRNG uses the seed as its key.
///
/// ARC4 is described in Schneier, B., "Applied Cryptography: Protocols,
/// Algorithms, and Source Code in C", 2nd Edition, 1996.
///
/// An individual generator is not thread-safe, but distinct generators do not
/// share state. The random data generated is of high-quality, but is not
/// suitable for cryptographic applications.
@frozen
public struct ARC4RandomNumberGenerator: SeedableRandomNumberGenerator {
    public static var global = ARC4RandomNumberGenerator(seed: UInt32(time(nil)))
    var state: [UInt8] = Array(0...255)
    var iPos: UInt8 = 0
    var jPos: UInt8 = 0

    /// Initialize ARC4RandomNumberGenerator using an array of UInt8. The array
    /// must have length between 1 and 256 inclusive.
    public init(seed: [UInt8]) {
        precondition(seed.count > 0, "Length of seed must be positive")
        precondition(seed.count <= 256, "Length of seed must be at most 256")
        var j: UInt8 = 0
        for i: UInt8 in 0...255 {
            j &+= S(i) &+ seed[Int(i) % seed.count]
            swapAt(i, j)
        }
    }

    // Produce the next random UInt64 from the stream, and advance the internal
    // state.
    public mutating func next() -> UInt64 {
        var result: UInt64 = 0
        for _ in 0..<UInt64.bitWidth / UInt8.bitWidth {
            result <<= UInt8.bitWidth
            result += UInt64(nextByte())
        }
        return result
    }

    // Helper to access the state.
    private func S(_ index: UInt8) -> UInt8 {
        return state[Int(index)]
    }

    // Helper to swap elements of the state.
    private mutating func swapAt(_ i: UInt8, _ j: UInt8) {
        state.swapAt(Int(i), Int(j))
    }

    // Generates the next byte in the keystream.
    private mutating func nextByte() -> UInt8 {
        iPos &+= 1
        jPos &+= S(iPos)
        swapAt(iPos, jPos)
        return S(S(iPos) &+ S(jPos))
    }
}

private typealias UInt32x2 = (UInt32, UInt32)
private typealias UInt32x4 = (UInt32, UInt32, UInt32, UInt32)

/// An implementation of `SeedableRandomNumberGenerator` using Threefry.
/// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
/// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
///
/// This struct implements a 20-round Threefry2x32 PRNG. It must be seeded with
/// a 64-bit value.
///
/// An individual generator is not thread-safe, but distinct generators do not
/// share state. The random data generated is of high-quality, but is not
/// suitable for cryptographic applications.
public struct ThreefryRandomNumberGenerator: SeedableRandomNumberGenerator {
    public static var global = ThreefryRandomNumberGenerator(
        uint64Seed: UInt64(time(nil))
    )

    private let rot: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
      = (13, 15, 26, 6, 17, 29, 16, 24)

    private func rotl32(value: UInt32, n: UInt32) -> UInt32 {
        return (value << (n & 31)) | (value >> ((32 - n) & 31))
    }

    private var ctr: UInt64 = 0
    private let key: UInt32x2

    private func random(forCtr ctr: UInt32x2, key: UInt32x2) -> UInt32x2 {
        let skeinKsParity32: UInt32 = 0x1BD11BDA

        let ks0 = key.0
        let ks1 = key.1
        let ks2 = skeinKsParity32 ^ key.0 ^ key.1
        var X0 = ctr.0
        var X1 = ctr.1

        // 20 rounds
        // Key injection (r = 0)
        X0 &+= ks0
        X1 &+= ks1
        // R1
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.0)
        X1 ^= X0
        // R2
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.1)
        X1 ^= X0
        // R3
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.2)
        X1 ^= X0
        // R4
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.3)
        X1 ^= X0
        // Key injection (r = 1)
        X0 &+= ks1
        X1 &+= (ks2 + 1)
        // R5
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.4)
        X1 ^= X0
        // R6
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.5)
        X1 ^= X0
        // R7
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.6)
        X1 ^= X0
        // R8
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.7)
        X1 ^= X0
        // Key injection (r = 2)
        X0 &+= ks2
        X1 &+= (ks0 + 2)
        // R9
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.0)
        X1 ^= X0
        // R10
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.1)
        X1 ^= X0
        // R11
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.2)
        X1 ^= X0
        // R12
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.3)
        X1 ^= X0
        // Key injection (r = 3)
        X0 &+= ks0
        X1 &+= (ks1 + 3)
        // R13
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.4)
        X1 ^= X0
        // R14
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.5)
        X1 ^= X0
        // R15
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.6)
        X1 ^= X0
        // R16
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.7)
        X1 ^= X0
        // Key injection (r = 4)
        X0 &+= ks1
        X1 &+= (ks2 + 4)
        // R17
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.0)
        X1 ^= X0
        // R18
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.1)
        X1 ^= X0
        // R19
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.2)
        X1 ^= X0
        // R20
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.3)
        X1 ^= X0
        // Key injection (r = 5)
        X0 &+= ks2
        X1 &+= (ks0 + 5)

        return (X0, X1)
    }

    internal init(uint64Seed seed: UInt64) {
        key = seed.vector2
    }

    public init(seed: [UInt8]) {
        precondition(seed.count > 0, "Length of seed must be positive")
        precondition(seed.count <= 8, "Length of seed must be at most 8")
        var combinedSeed: UInt64 = 0
        for (i, byte) in seed.enumerated() {
            combinedSeed += UInt64(byte) << UInt64(8 * i)
        }
        self.init(uint64Seed: combinedSeed)
    }

    public mutating func next() -> UInt64 {
        defer { ctr += 1 }
        return UInt64(vector: random(forCtr: ctr.vector2, key: key))
    }
}

/// An implementation of `SeedableRandomNumberGenerator` using Philox.
/// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
/// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
///
/// This struct implements a 10-round Philox4x32 PRNG. It must be seeded with
/// a 64-bit value.
///
/// An individual generator is not thread-safe, but distinct generators do not
/// share state. The random data generated is of high-quality, but is not
/// suitable for cryptographic applications.
public struct PhiloxRandomNumberGenerator: SeedableRandomNumberGenerator {
    public static var global = PhiloxRandomNumberGenerator(
        uint64Seed: UInt64(time(nil))
    )

    private var ctr: UInt64 = 0
    private let key: UInt32x2

    // Since we generate two 64-bit values at a time, we only need to run the
    // generator every other invocation.
    private var useNextValue = false
    private var nextValue: UInt64 = 0

    private func bump(key: UInt32x2) -> UInt32x2 {
        let bumpConstantHi: UInt32 = 0x9E3779B9
        let bumpConstantLo: UInt32 = 0xBB67AE85
        return (key.0 &+ bumpConstantHi, key.1 &+ bumpConstantLo)
    }

    private func round(ctr: UInt32x4, key: UInt32x2) -> UInt32x4 {
        let roundConstant0: UInt64 = 0xD2511F53
        let roundConstant1: UInt64 = 0xCD9E8D57

        let product0: UInt64 = roundConstant0 &* UInt64(ctr.0)
        let hi0 = UInt32(truncatingIfNeeded: product0 >> 32)
        let lo0 = UInt32(truncatingIfNeeded: (product0 & 0x0000_0000_FFFF_FFFF))

        let product1: UInt64 = roundConstant1 &* UInt64(ctr.2)
        let hi1 = UInt32(truncatingIfNeeded: product1 >> 32)
        let lo1 = UInt32(truncatingIfNeeded: (product1 & 0x0000_0000_FFFF_FFFF))

        return (hi1 ^ ctr.1 ^ key.0, lo1, hi0 ^ ctr.3 ^ key.1, lo0)
    }

    private func random(forCtr initialCtr: UInt32x4, key initialKey: UInt32x2) -> UInt32x4 {
        var ctr = initialCtr
        var key = initialKey
        // 10 rounds
        // R1
        ctr = round(ctr: ctr, key: key)
        // R2
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R3
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R4
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R5
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R6
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R7
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R8
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R9
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R10
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)

        return ctr
    }

    internal init(uint64Seed seed: UInt64) {
        key = seed.vector2
    }

    public init(seed: [UInt8]) {
        precondition(seed.count > 0, "Length of seed must be positive")
        precondition(seed.count <= 8, "Length of seed must be at most 8")
        var combinedSeed: UInt64 = 0
        for (i, byte) in seed.enumerated() {
            combinedSeed += UInt64(byte) << UInt64(8 * i)
        }
        self.init(uint64Seed: combinedSeed)
    }

    public mutating func next() -> UInt64 {
        if useNextValue {
            useNextValue = false
            return nextValue
        }
        let (this, next) = makeUInt64Pair(random(forCtr: ctr.vector4, key: key))
        useNextValue = true
        nextValue = next
        ctr += 1
        return this
    }
}

/// Private helpers.
fileprivate extension UInt64 {
    var vector2: UInt32x2 {
        let msb = UInt32(truncatingIfNeeded: self >> 32)
        let lsb = UInt32(truncatingIfNeeded: self & 0x0000_0000_FFFF_FFFF)
        return (msb, lsb)
    }

    var vector4: UInt32x4 {
        let msb = UInt32(truncatingIfNeeded: self >> 32)
        let lsb = UInt32(truncatingIfNeeded: self & 0x0000_0000_FFFF_FFFF)
        return (0, 0, msb, lsb)
    }

    init(vector: UInt32x2) {
        self = (UInt64(vector.0) << 32) + UInt64(vector.1)
    }
}

private func makeUInt64Pair(_ vector: UInt32x4) -> (UInt64, UInt64) {
    let a = (UInt64(vector.0) << 32) + UInt64(vector.1)
    let b = (UInt64(vector.2) << 32) + UInt64(vector.3)
    return (a, b)
}

//===------------------------------------------------------------------------------------------===//
// Distributions
//===------------------------------------------------------------------------------------------===//

public protocol RandomDistribution {
    associatedtype Sample
    func next<G: RandomNumberGenerator>(using generator: inout G) -> Sample
}

@frozen
public struct UniformIntegerDistribution<T: FixedWidthInteger>: RandomDistribution {
    public let lowerBound: T
    public let upperBound: T

    public init(lowerBound: T = T.self.min, upperBound: T = T.self.max) {
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }

    public func next<G: RandomNumberGenerator>(using rng: inout G) -> T {
        return T.random(in: lowerBound...upperBound, using: &rng)
    }
}

@frozen
public struct UniformFloatingPointDistribution<T: BinaryFloatingPoint>: RandomDistribution
  where T.RawSignificand: FixedWidthInteger {
    public let lowerBound: T
    public let upperBound: T

    public init(lowerBound: T = 0, upperBound: T = 1) {
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }

    public func next<G: RandomNumberGenerator>(using rng: inout G) -> T {
        return T.random(in: lowerBound..<upperBound, using: &rng)
    }
}

@frozen
public struct NormalDistribution<T: BinaryFloatingPoint>: RandomDistribution
    where T.RawSignificand: FixedWidthInteger {
    public let mean: T
    public let standardDeviation: T
    private let uniformDist = UniformFloatingPointDistribution<T>()

    public init(mean: T = 0, standardDeviation: T = 1) {
        self.mean = mean
        self.standardDeviation = standardDeviation
    }

    public func next<G: RandomNumberGenerator>(using rng: inout G) -> T {
        // FIXME: Box-Muller can generate two values for only a little more than the
        // cost of one.
        let u1 = uniformDist.next(using: &rng)
        let u2 = uniformDist.next(using: &rng)
        let r = (-2 * T(log(Double(u1)))).squareRoot()
        let theta: Double = 2 * Double.pi * Double(u2)
        let normal01 = r * T(cos(theta))
        return mean + standardDeviation * normal01
    }
}

@frozen
public struct BetaDistribution: RandomDistribution {
    public let alpha: Float
    public let beta: Float
    private let uniformDistribution = UniformFloatingPointDistribution<Float>()

    public init(alpha: Float = 0, beta: Float = 1) {
        self.alpha = alpha
        self.beta = beta
    }

    public func next<G: RandomNumberGenerator>(using rng: inout G) -> Float {
        // Generate a sample using Cheng's sampling algorithm from:
        // R. C. H. Cheng, "Generating beta variates with nonintegral shape
        // parameters.". Communications of the ACM, 21, 317-322, 1978.
        let a = min(alpha, beta)
        let b = max(alpha, beta)
        if a > 1 {
            return BetaDistribution.chengsAlgorithmBB(alpha, a, b, using: &rng)
        } else {
            return BetaDistribution.chengsAlgorithmBC(alpha, b, a, using: &rng)
        }
    }

    /// Returns one sample from a Beta(alpha, beta) distribution using Cheng's BB
    /// algorithm, when both alpha and beta are greater than 1.
    ///
    /// - Parameters:
    ///   - alpha: First Beta distribution shape parameter.
    ///   - a: `min(alpha, beta)`.
    ///   - b: `max(alpha, beta)`.
    ///   - rng: Random number generator.
    ///
    /// - Returns: Sample obtained using Cheng's BB algorithm.
    private static func chengsAlgorithmBB<G: RandomNumberGenerator>(
        _ alpha0: Float,
        _ a: Float,
        _ b: Float,
        using rng: inout G
    ) -> Float {
        let alpha = a + b
        let beta  = sqrt((alpha - 2) / (2 * a * b - alpha))
        let gamma = a + 1 / beta

        var r: Float = 0.0
        var w: Float = 0.0
        var t: Float = 0.0

        repeat {
            let u1 = Float.random(in: 0.0...1.0, using: &rng)
            let u2 = Float.random(in: 0.0...1.0, using: &rng)
            let v = beta * (log(u1) - log1p(-u1))
            r = gamma * v - 1.3862944
            let z = u1 * u1 * u2
            w = a * exp(v)

            let s = a + r - w
            if s + 2.609438 >= 5 * z {
                break
            }

            t = log(z)
            if s >= t {
                break
            }
        } while r + alpha * (log(alpha) - log(b + w)) < t

        w = min(w, Float.greatestFiniteMagnitude)
        return a == alpha0 ? w / (b + w) : b / (b + w)
    }

    /// Returns one sample from a Beta(alpha, beta) distribution using Cheng's BC
    /// algorithm, when at least one of alpha and beta is less than 1.
    ///
    /// - Parameters:
    ///     - alpha: First Beta distribution shape parameter.
    ///     - a: `max(alpha, beta)`.
    ///     - b: `min(alpha, beta)`.
    ///     - rng: Random number generator.
    ///
    /// - Returns: Sample obtained using Cheng's BB algorithm.
    private static func chengsAlgorithmBC<G: RandomNumberGenerator>(
        _ alpha0: Float,
        _ a: Float,
        _ b: Float,
        using rng: inout G
    ) -> Float {
        let alpha = a + b
        let beta  = 1 / b
        let delta = 1 + a - b
        let k1    = delta * (0.0138889 + 0.0416667 * b) / (a * beta - 0.777778)
        let k2    = 0.25 + (0.5 + 0.25 / delta) * b

        var w: Float = 0.0

        while true {
            let u1 = Float.random(in: 0.0...1.0, using: &rng)
            let u2 = Float.random(in: 0.0...1.0, using: &rng)
            let y = u1 * u2
            let z = u1 * y

            if u1 < 0.5 {
                if 0.25 * u2 + z - y >= k1 {
                    continue
                }
            } else {
                if z <= 0.25 {
                    let v = beta * (log(u1) - log1p(-u1))
                    w = a * exp(v)
                    break
                }
                if z >= k2 {
                    continue
                }
            }

            let v = beta * (log(u1) - log1p(-u1))
            w = a * exp(v)
            if alpha * (log(alpha) - log(b + 1) + v) - 1.3862944 >= log(z) {
                break
            }
        }

        w = min(w, Float.greatestFiniteMagnitude)
        return a == alpha0 ? w / (b + w): b / (b + w)
    }
}
