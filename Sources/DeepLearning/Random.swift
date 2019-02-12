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

/// An implementation of `SeedableRandomNumberGenerator` using Threefry.
/// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
/// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
///
/// This struct implements a 20-rotation Threefry32x2 PRNG. It must be seeded
/// with a 64-bit value.
///
/// An individual generator is not thread-safe, but distinct generators do not
/// share state. The random data generated is of high-quality, but is not
/// suitable for cryptographic applications.
public struct ThreefryRandomNumberGenerator: SeedableRandomNumberGenerator {
    private typealias ThreefryArray = (UInt32, UInt32)

    private static let rot:
        (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
        = (13, 15, 26, 6, 17, 29, 16, 24)

    private static func rotl32(value: UInt32, n: UInt32) -> UInt32 {
        return (value << (n & 31)) | (value >> ((32 - n) & 31))
    }

    private var ctr: UInt64 = 0
    private let key: ThreefryArray

    private static func split(_ value: UInt64) -> ThreefryArray {
        let msb = UInt32(truncatingIfNeeded: (value & 0xFFFF_FFFF_0000_0000) >> 32)
        let lsb = UInt32(truncatingIfNeeded: value & 0x0000_0000_FFFF_FFFF)
        return (msb, lsb)
    }

    private static func combine(_ value: ThreefryArray) -> UInt64 {
        return (UInt64(value.0) << 32) + UInt64(value.1)
    }

    private static func calculateRandom(ctr: ThreefryArray, key: ThreefryArray) -> ThreefryArray {
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
        key = ThreefryRandomNumberGenerator.split(seed)
    }

    public init(seed: [UInt8]) {
        precondition(seed.count > 0, "Length of seed must be positive")
        precondition(seed.count <= 8, "Length of seed must be at most 8")
        var combinedSeed: UInt64 = 0
        for i in 0..<seed.count {
            combinedSeed += UInt64(seed[i]) << UInt64(8 * i)
        }
        self.init(uint64Seed: combinedSeed)
    }

    public mutating func next() -> UInt64 {
        defer { ctr += 1 }
        return ThreefryRandomNumberGenerator.combine(
            ThreefryRandomNumberGenerator.calculateRandom(
                ctr: ThreefryRandomNumberGenerator.split(ctr),
                key: key
            )
        )
    }
}
