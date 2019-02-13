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

#if !COMPILING_TENSORFLOW_MODULE
import TensorFlow
#endif

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
    private let rot:
        (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
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
        return UInt64(fromVector: random(forCtr: ctr.vector2, key: key))
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

    init(fromVector vector: UInt32x2) {
        self = (UInt64(vector.0) << 32) + UInt64(vector.1)
    }
}

private func makeUInt64Pair(_ vector: UInt32x4) -> (UInt64, UInt64) {
    let a = (UInt64(vector.0) << 32) + UInt64(vector.1)
    let b = (UInt64(vector.2) << 32) + UInt64(vector.3)
    return (a, b)
}
